"""Single-car simulator shared by manual, scripted, PPO, eval, and replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1rl.config import DISCRETE_ACTIONS, SimConfig, action_to_controls
from f1rl.geometry import (
    nearest_intersection_distance,
    project_point_to_polyline,
    segment_intersects_any,
    wrap_radians,
)
from f1rl.physics import apply_physics, initial_car_state
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
        self.last_steer = 0.0
        self.no_progress_steps = 0
        self.terminated = False
        self.truncated = False
        self.termination_reason = "active"
        self.completed_lap = False
        self.state = initial_car_state(self.track.start_pose)
        self._last_raw_progress_px = 0.0
        self._ray_cache_key: tuple[float, float, float] | None = None
        self._ray_cache_distances_m: np.ndarray | None = None
        self._reset_projection()

    @property
    def observation_dim(self) -> int:
        return 7 + len(self.sensor_angles) + len(self.config.lookahead_m)

    @property
    def action_dim(self) -> int:
        return len(DISCRETE_ACTIONS)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        del seed
        self.state = initial_car_state(self.track.start_pose)
        self.last_action_id = 0
        self.last_steer = 0.0
        self.no_progress_steps = 0
        self.terminated = False
        self.truncated = False
        self.termination_reason = "active"
        self.completed_lap = False
        self._reset_projection()
        obs = self.observation()
        return obs, self.info(0.0, {key: 0.0 for key in REWARD_COMPONENT_KEYS}, False, False)

    def _reset_projection(self) -> None:
        raw_px, _, _, _ = project_point_to_polyline(
            self.state.position(), self.track.centerline, self.track.centerline_s
        )
        self._last_raw_progress_px = raw_px
        raw_m = raw_px * self.track.meters_per_pixel
        self.state.raw_progress_m = raw_m
        self.state.monotonic_progress_m = raw_m

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

    def step_controls(self, *, throttle: float, brake: float, steer: float, action_id: int = -1) -> SimStep:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended; call reset() first")
        previous_steer = self.last_steer
        old_progress_m = self.state.monotonic_progress_m
        old_raw_px = self._last_raw_progress_px
        self.state, movement = apply_physics(
            self.state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=self.config.car,
            meters_per_pixel=self.track.meters_per_pixel,
        )
        self.last_action_id = int(action_id)
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

        checkpoint_spacing = self.track.length_m / max(len(self.track.checkpoints), 1)
        self.state.checkpoint_index = int((self.state.monotonic_progress_m // checkpoint_spacing) % len(self.track.checkpoints))
        if self.state.monotonic_progress_m >= (self.state.lap_index + 1) * self.track.length_m:
            self.state.lap_index += 1
            self.completed_lap = True

        collided = segment_intersects_any(movement, self.boundary_segments)
        off_track = not self.track.point_is_drivable(self.state.x, self.state.y)
        if progress_delta_m <= 1e-4:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        components["progress"] = progress_delta_m * self.config.reward.progress_scale
        if self.completed_lap:
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

        obs = self.observation()
        telemetry = StepTelemetry(
            step_index=self.state.elapsed_steps,
            sim_time_s=self.state.elapsed_steps * self.config.car.dt,
            x=float(self.state.x),
            y=float(self.state.y),
            heading_deg=float(np.rad2deg(self.state.heading_rad)),
            speed_mps=float(self.state.speed_mps),
            speed_kph=float(self.state.speed_mps * 3.6),
            yaw_rate_rps=float(self.state.yaw_rate_rps),
            throttle=float(throttle),
            brake=float(brake),
            steering=float(steer),
            action_id=int(action_id),
            raw_progress_m=float(self.state.raw_progress_m),
            monotonic_progress_m=float(self.state.monotonic_progress_m),
            progress_delta_m=float(self.state.monotonic_progress_m - old_progress_m),
            lateral_error_m=float(lateral_error_m),
            heading_error_deg=float(np.rad2deg(heading_error)),
            checkpoint_index=int(self.state.checkpoint_index),
            lap_index=int(self.state.lap_index),
            ray_distances_m=[float(v) for v in self.ray_distances_m()],
            collided=bool(collided),
            off_track=bool(off_track),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            termination_reason=self.termination_reason,
            reward_total=reward,
            reward_components=components,
        )
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
            "lap_index": int(self.state.lap_index),
            "raw_progress_m": float(self.state.raw_progress_m),
            "monotonic_progress_m": float(self.state.monotonic_progress_m),
            "reward": float(reward),
            "reward_components": dict(components),
            "collided": bool(collided),
            "off_track": bool(off_track),
            "termination_reason": self.termination_reason,
        }
