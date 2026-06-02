"""Single-car simulator shared by manual, scripted, PPO, eval, and replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1rl.config import DISCRETE_ACTIONS, SimConfig, action_to_controls
from f1rl.geometry import (
    nearest_intersection_distance,
    project_point_to_polyline,
    sample_polyline_at,
    segment_intersects_any,
    wrap_radians,
)
from f1rl.physics import CarState, apply_physics, initial_car_state
from f1rl.telemetry import REWARD_COMPONENT_KEYS, StepTelemetry
from f1rl.track_model import TrackSpec, load_track_spec


@dataclass(slots=True)
class SimStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    telemetry: StepTelemetry


class MonzaSim:
    def __init__(self, config: SimConfig | None = None, track: TrackSpec | None = None) -> None:
        self.config = config or SimConfig()
        self.track = track or load_track_spec(self.config.track_path)
        self.boundary_segments = self.track.boundary_segments
        self.sensor_angles = self._sensor_angles()
        self.last_action_id = 0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steer = 0.0
        self.no_progress_steps = 0
        self.terminated = False
        self.truncated = False
        self.termination_reason = "active"
        self.completed_lap = False
        self.valid_lap = False
        self.finish_crossed = False
        self.next_checkpoint_index = 1
        self.checkpoints_passed = 0
        self.missed_checkpoint_count = 0
        self.segment_target_progress_m: float | None = None
        self.segment_complete = False
        self.curriculum_stage: str | None = None
        self.state = initial_car_state(self.track.start_pose)
        self.episode_start_progress_m = 0.0
        self._last_raw_progress_px = 0.0
        self._ray_cache_key: tuple[float, float, float] | None = None
        self._ray_cache_distances_m: np.ndarray | None = None
        self.last_telemetry: StepTelemetry | None = None
        self._reset_projection()

    @property
    def observation_dim(self) -> int:
        return 7 + len(self.sensor_angles) + len(self.config.lookahead_m)

    @property
    def action_dim(self) -> int:
        return len(DISCRETE_ACTIONS)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        options = options or {}
        rng = np.random.default_rng(seed)
        self.state = self._initial_state_from_options(options, rng)
        self.last_action_id = 0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steer = 0.0
        self.no_progress_steps = 0
        self.terminated = False
        self.truncated = False
        self.termination_reason = "active"
        self.completed_lap = False
        self.valid_lap = False
        self.finish_crossed = False
        self.segment_complete = False
        self.curriculum_stage = options.get("curriculum_stage")
        self.segment_target_progress_m = None
        self.last_telemetry = None
        self._reset_projection()
        self.episode_start_progress_m = self.state.monotonic_progress_m
        segment_length_m = options.get("segment_length_m")
        if segment_length_m is not None:
            self.segment_target_progress_m = self.episode_start_progress_m + max(float(segment_length_m), 1.0)
        self._reset_lap_validity()
        obs = self.observation()
        return obs, self.info(0.0, {key: 0.0 for key in REWARD_COMPONENT_KEYS}, False, False)

    def _initial_state_from_options(self, options: dict, rng: np.random.Generator) -> CarState:
        if "start_checkpoint" in options:
            checkpoint_idx = int(options["start_checkpoint"]) % len(self.track.checkpoints)
            start_progress_m = float(self.track.checkpoint_s[checkpoint_idx] * self.track.meters_per_pixel)
        elif "start_progress_m" in options:
            start_progress_m = float(options["start_progress_m"])
        else:
            state = initial_car_state(self.track.start_pose)
            if "start_speed_kph" in options:
                state.speed_mps = max(float(options["start_speed_kph"]) / 3.6, 0.0)
            return state

        point, heading = self.centerline_pose_at(start_progress_m)
        position_noise_m = float(options.get("position_noise_m", 0.0) or 0.0)
        if position_noise_m > 0.0:
            noise_px = rng.normal(0.0, position_noise_m / max(self.track.meters_per_pixel, 1e-6), size=2)
            point = point + noise_px.astype(np.float32)
        heading_noise_deg = float(options.get("heading_noise_deg", 0.0) or 0.0)
        if heading_noise_deg > 0.0:
            heading += float(np.deg2rad(rng.normal(0.0, heading_noise_deg)))
        speed_kph = float(options.get("start_speed_kph", 0.0) or 0.0)
        speed_noise_kph = float(options.get("speed_noise_kph", 0.0) or 0.0)
        if speed_noise_kph > 0.0:
            speed_kph += float(rng.normal(0.0, speed_noise_kph))
        return CarState(
            x=float(point[0]),
            y=float(point[1]),
            heading_rad=wrap_radians(heading),
            speed_mps=max(speed_kph / 3.6, 0.0),
        )

    def centerline_pose_at(self, progress_m: float) -> tuple[np.ndarray, float]:
        progress_px = (progress_m / self.track.meters_per_pixel) % self.track.length_px
        point = sample_polyline_at(
            self.track.centerline,
            self.track.centerline_s,
            np.asarray([progress_px], dtype=np.float32),
        )[0]
        idx = np.searchsorted(self.track.centerline_s, progress_px, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.track.centerline_s) - 2))
        start = self.track.centerline[idx]
        end = self.track.centerline[idx + 1]
        heading = float(np.arctan2(-(end[1] - start[1]), end[0] - start[0]))
        return point, heading

    def _reset_projection(self) -> None:
        raw_px, _, _, _ = project_point_to_polyline(
            self.state.position(), self.track.centerline, self.track.centerline_s
        )
        self._last_raw_progress_px = raw_px
        raw_m = raw_px * self.track.meters_per_pixel
        self.state.raw_progress_m = raw_m
        self.state.monotonic_progress_m = raw_m
        self._ray_cache_key = None
        self._ray_cache_distances_m = None

    def _reset_lap_validity(self) -> None:
        checkpoint_spacing = self.track.length_m / max(len(self.track.checkpoints), 1)
        start_index = int(self.state.monotonic_progress_m // checkpoint_spacing)
        self.checkpoints_passed = max(start_index, 0)
        self.next_checkpoint_index = min(self.checkpoints_passed + 1, len(self.track.checkpoints))
        self.missed_checkpoint_count = 0
        self.valid_lap = self.state.monotonic_progress_m <= checkpoint_spacing * 0.5
        self.finish_crossed = False

    def _update_checkpoint_validity(
        self, previous_progress_m: float, progress_delta_m: float, lateral_error_m: float
    ) -> None:
        checkpoint_count = len(self.track.checkpoints)
        checkpoint_spacing = self.track.length_m / max(checkpoint_count, 1)
        if progress_delta_m > checkpoint_spacing * 1.75:
            skipped = int(progress_delta_m // checkpoint_spacing)
            self.missed_checkpoint_count += max(skipped - 1, 1)
            self.valid_lap = False
        while self.next_checkpoint_index < checkpoint_count:
            threshold = checkpoint_spacing * self.next_checkpoint_index
            if self.state.monotonic_progress_m + 1e-6 < threshold:
                break
            if previous_progress_m <= threshold <= self.state.monotonic_progress_m + checkpoint_spacing * 0.75:
                if abs(lateral_error_m) > self.config.checkpoint_lateral_limit_m:
                    self.missed_checkpoint_count += 1
                    self.valid_lap = False
                self.checkpoints_passed = self.next_checkpoint_index
                self.next_checkpoint_index += 1
            else:
                self.missed_checkpoint_count += 1
                self.valid_lap = False
                self.checkpoints_passed = self.next_checkpoint_index
                self.next_checkpoint_index += 1
        self.state.checkpoint_index = int((self.state.monotonic_progress_m // checkpoint_spacing) % checkpoint_count)

    def _crossed_finish(self, movement: np.ndarray, old_progress_m: float) -> bool:
        target = (self.state.lap_index + 1) * self.track.length_m
        checkpoint_spacing = self.track.length_m / max(len(self.track.checkpoints), 1)
        near_finish = old_progress_m >= target - checkpoint_spacing * 2.0
        if not near_finish and self.state.monotonic_progress_m < target:
            return False
        physical_crossing = segment_intersects_any(movement, self.track.finish_line.reshape(1, 4))
        virtual_crossing = old_progress_m < target <= self.state.monotonic_progress_m
        return bool(near_finish and (physical_crossing or virtual_crossing))

    def _sensor_angles(self) -> np.ndarray:
        count = max(3, int(self.config.sensors.count))
        if count % 2 == 0:
            count += 1
        base = np.linspace(-1.0, 1.0, count, dtype=np.float32)
        biased = np.sign(base) * np.power(np.abs(base), self.config.sensors.forward_bias)
        return np.deg2rad(biased * (self.config.sensors.spread_deg / 2.0)).astype(np.float32)

    def build_sensor_rays(self) -> np.ndarray:
        max_px = self.config.sensors.range_m / self.track.meters_per_pixel
        rays = np.zeros((len(self.sensor_angles), 4), dtype=np.float32)
        rays[:, 0] = self.state.x
        rays[:, 1] = self.state.y
        for idx, rel in enumerate(self.sensor_angles):
            heading = self.state.heading_rad + float(rel)
            rays[idx, 2] = self.state.x + np.cos(heading) * max_px
            rays[idx, 3] = self.state.y - np.sin(heading) * max_px
        return rays

    def ray_distances_m(self) -> np.ndarray:
        cache_key = (float(self.state.x), float(self.state.y), float(self.state.heading_rad))
        if self._ray_cache_key == cache_key and self._ray_cache_distances_m is not None:
            return self._ray_cache_distances_m.copy()
        max_px = self.config.sensors.range_m / self.track.meters_per_pixel
        distances = [
            nearest_intersection_distance(ray, self.boundary_segments, max_px) * self.track.meters_per_pixel
            for ray in self.build_sensor_rays()
        ]
        values = np.asarray(distances, dtype=np.float32)
        self._ray_cache_key = cache_key
        self._ray_cache_distances_m = values
        return values.copy()

    def _track_errors(self) -> tuple[float, float, float]:
        raw_px, lateral_px, tangent, _ = project_point_to_polyline(
            self.state.position(),
            self.track.centerline,
            self.track.centerline_s,
            previous_progress=self._last_raw_progress_px,
            window=self.config.local_projection_window_m / self.track.meters_per_pixel,
        )
        heading_error = wrap_radians(tangent - self.state.heading_rad)
        return raw_px, lateral_px * self.track.meters_per_pixel, heading_error

    def _lookahead_heading_errors(self) -> list[float]:
        values: list[float] = []
        current_px = self.state.raw_progress_m / self.track.meters_per_pixel
        for lookahead_m in self.config.lookahead_m:
            target_px = (current_px + lookahead_m / self.track.meters_per_pixel) % self.track.length_px
            idx = np.searchsorted(self.track.centerline_s, target_px, side="right") - 1
            idx = int(np.clip(idx, 0, len(self.track.centerline_s) - 2))
            start = self.track.centerline[idx]
            end = self.track.centerline[idx + 1]
            tangent = float(np.arctan2(-(end[1] - start[1]), end[0] - start[0]))
            values.append(wrap_radians(tangent - self.state.heading_rad) / np.pi)
        return values

    def observation(self) -> np.ndarray:
        _, lateral_error_m, heading_error = self._track_errors()
        ray_obs = np.clip(self.ray_distances_m() / max(self.config.sensors.range_m, 1e-6), 0.0, 1.0) * 2.0 - 1.0
        progress_ratio = (self.state.monotonic_progress_m % self.track.length_m) / self.track.length_m
        obs = np.asarray(
            [
                self.state.speed_mps / self.config.car.max_speed_mps,
                np.clip(self.state.yaw_rate_rps / 2.0, -1.0, 1.0),
                np.clip(heading_error / np.pi, -1.0, 1.0),
                np.clip(lateral_error_m / 30.0, 0.0, 1.0) * 2.0 - 1.0,
                progress_ratio * 2.0 - 1.0,
                (self.last_action_id / max(self.action_dim - 1, 1)) * 2.0 - 1.0,
                self.last_steer,
                *ray_obs,
                *self._lookahead_heading_errors(),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action_id: int) -> SimStep:
        throttle, brake, steer = action_to_controls(action_id)
        return self.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=action_id)

    def step_controls(
        self,
        *,
        throttle: float,
        brake: float,
        steer: float,
        action_id: int = -1,
        collect_observation: bool = True,
        collect_rays: bool = True,
    ) -> SimStep:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended; call reset() first")
        previous_steer = self.last_steer
        previous_throttle = self.last_throttle
        previous_brake = self.last_brake
        old_progress_m = self.state.monotonic_progress_m
        old_raw_px = self._last_raw_progress_px
        old_speed_mps = self.state.speed_mps
        self.state, movement = apply_physics(
            self.state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=self.config.car,
            meters_per_pixel=self.track.meters_per_pixel,
        )
        self.last_action_id = int(action_id)
        self.last_throttle = float(throttle)
        self.last_brake = float(brake)
        self.last_steer = float(steer)

        raw_px, lateral_error_m, heading_error = self._track_errors()
        raw_delta_px = raw_px - old_raw_px
        if raw_delta_px < -0.5 * self.track.length_px:
            raw_delta_px += self.track.length_px
        elif raw_delta_px > 0.5 * self.track.length_px:
            raw_delta_px -= self.track.length_px
        progress_delta_m = max(0.0, raw_delta_px * self.track.meters_per_pixel)
        if progress_delta_m > self.config.local_projection_window_m:
            progress_delta_m = 0.0
        self._last_raw_progress_px = raw_px
        self.state.raw_progress_m = raw_px * self.track.meters_per_pixel
        self.state.monotonic_progress_m = old_progress_m + progress_delta_m

        self._update_checkpoint_validity(old_progress_m, progress_delta_m, lateral_error_m)

        collided = segment_intersects_any(movement, self.boundary_segments)
        off_track = not self.track.point_is_drivable(self.state.x, self.state.y)
        if progress_delta_m <= 1e-4:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        components["progress"] = progress_delta_m * self.config.reward.progress_scale
        lateral_excess_m = max(0.0, abs(lateral_error_m) - self.config.reward.lateral_deadzone_m)
        components["lateral"] = -self.config.reward.lateral_penalty_scale * lateral_excess_m
        ray_distances_m = self.ray_distances_m()
        min_ray_m = float(np.min(ray_distances_m)) if len(ray_distances_m) else self.config.reward.track_limit_safe_ray_m
        track_limit_excess_m = max(0.0, self.config.reward.track_limit_safe_ray_m - min_ray_m)
        speed_factor = max(1.0, (self.state.speed_mps * 3.6) / 100.0)
        components["track_limit"] = -self.config.reward.track_limit_penalty_scale * track_limit_excess_m * speed_factor

        if self.segment_target_progress_m is not None and self.state.monotonic_progress_m >= self.segment_target_progress_m:
            self.segment_complete = True
            components["finish"] = self.config.reward.finish_bonus
            self.truncated = True
            self.termination_reason = "segment_complete"

        finish_crossed_this_step = self._crossed_finish(movement, old_progress_m)
        if finish_crossed_this_step:
            self.finish_crossed = True
        lap_valid_now = (
            self.valid_lap
            and self.missed_checkpoint_count == 0
            and self.checkpoints_passed >= len(self.track.checkpoints) - 1
        )
        if (
            self.segment_target_progress_m is None
            and finish_crossed_this_step
            and lap_valid_now
            and self.state.monotonic_progress_m >= (self.state.lap_index + 1) * self.track.length_m
        ):
            self.state.lap_index += 1
            self.completed_lap = True
            components["finish"] = self.config.reward.finish_bonus
            self.truncated = True
            self.termination_reason = "lap_complete"
        if collided:
            components["collision"] = self.config.reward.collision_penalty
            self.terminated = True
            self.termination_reason = "collision"
        if off_track:
            components["off_track"] = self.config.reward.off_track_penalty
            self.terminated = True
            self.termination_reason = "off_track"
        if self.no_progress_steps >= self.config.no_progress_limit_steps:
            components["no_progress"] = self.config.reward.no_progress_penalty
            self.terminated = True
            self.termination_reason = "no_progress"
        if self.state.elapsed_steps >= self.config.max_steps:
            self.truncated = True
            self.termination_reason = "max_steps"
        components["smoothness"] = -self.config.reward.smoothness_penalty * abs(steer - previous_steer)
        reward = float(sum(components.values()))
        if self.terminated:
            self.state.alive = False

        obs = self.observation() if collect_observation else np.empty(0, dtype=np.float32)
        acceleration_mps2 = (self.state.speed_mps - old_speed_mps) / max(self.config.car.dt, 1e-9)
        lateral_g = (self.state.speed_mps * self.state.yaw_rate_rps) / 9.81
        curvature = self.state.yaw_rate_rps / max(self.state.speed_mps, 1e-6)
        telemetry = StepTelemetry(
            step_index=self.state.elapsed_steps,
            sim_time_s=self.state.elapsed_steps * self.config.car.dt,
            x=float(self.state.x),
            y=float(self.state.y),
            heading_deg=float(np.rad2deg(self.state.heading_rad)),
            speed_mps=float(self.state.speed_mps),
            speed_kph=float(self.state.speed_mps * 3.6),
            yaw_rate_rps=float(self.state.yaw_rate_rps),
            acceleration_mps2=float(acceleration_mps2),
            longitudinal_g=float(acceleration_mps2 / 9.81),
            lateral_g=float(lateral_g),
            curvature_rad_per_m=float(curvature),
            throttle=float(throttle),
            brake=float(brake),
            steering=float(steer),
            throttle_delta=float(throttle - previous_throttle),
            brake_delta=float(brake - previous_brake),
            steering_delta=float(steer - previous_steer),
            action_id=int(action_id),
            raw_progress_m=float(self.state.raw_progress_m),
            monotonic_progress_m=float(self.state.monotonic_progress_m),
            progress_delta_m=float(self.state.monotonic_progress_m - old_progress_m),
            lateral_error_m=float(lateral_error_m),
            racing_line_deviation_m=float(lateral_error_m),
            heading_error_deg=float(np.rad2deg(heading_error)),
            reference_progress_m=None,
            reference_speed_kph=None,
            ghost_gap_m=None,
            checkpoint_index=int(self.state.checkpoint_index),
            next_checkpoint_index=int(self.next_checkpoint_index),
            checkpoints_passed=int(self.checkpoints_passed),
            missed_checkpoint_count=int(self.missed_checkpoint_count),
            lap_index=int(self.state.lap_index),
            valid_lap=bool(lap_valid_now),
            finish_crossed=bool(self.finish_crossed),
            segment_complete=bool(self.segment_complete),
            curriculum_stage=self.curriculum_stage,
            segment_target_progress_m=self.segment_target_progress_m,
            ray_distances_m=[float(v) for v in ray_distances_m] if collect_rays else [],
            collided=bool(collided),
            off_track=bool(off_track),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            termination_reason=self.termination_reason,
            reward_total=reward,
            reward_components=components,
        )
        self.last_telemetry = telemetry
        return SimStep(
            observation=obs,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info(reward, components, collided, off_track),
            telemetry=telemetry,
        )

    def info(self, reward: float, components: dict[str, float], collided: bool, off_track: bool) -> dict:
        return {
            "x": float(self.state.x),
            "y": float(self.state.y),
            "speed_kph": float(self.state.speed_mps * 3.6),
            "checkpoint_index": int(self.state.checkpoint_index),
            "next_checkpoint_index": int(self.next_checkpoint_index),
            "checkpoints_passed": int(self.checkpoints_passed),
            "missed_checkpoint_count": int(self.missed_checkpoint_count),
            "lap_index": int(self.state.lap_index),
            "valid_lap": bool(self.valid_lap and self.missed_checkpoint_count == 0),
            "finish_crossed": bool(self.finish_crossed),
            "segment_complete": bool(self.segment_complete),
            "curriculum_stage": self.curriculum_stage,
            "segment_target_progress_m": self.segment_target_progress_m,
            "raw_progress_m": float(self.state.raw_progress_m),
            "monotonic_progress_m": float(self.state.monotonic_progress_m),
            "reward": float(reward),
            "reward_components": dict(components),
            "collided": bool(collided),
            "off_track": bool(off_track),
            "termination_reason": self.termination_reason,
        }
