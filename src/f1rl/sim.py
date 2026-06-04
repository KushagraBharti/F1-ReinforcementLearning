"""Single-car simulator shared by manual, scripted, PPO, eval, and replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1rl.config import (
    SimConfig,
    action_to_controls,
    actions_for_action_set,
    multidiscrete_action_nvec,
    multidiscrete_action_to_controls,
)
from f1rl.geometry import (
    nearest_intersection_distance,
    project_point_to_polyline,
    sample_polyline_at,
    segment_intersects_any,
    wrap_radians,
)
from f1rl.physics import CarState, apply_physics, initial_car_state
from f1rl.state_snapshot import snapshot_from_mapping, snapshot_to_car_state
from f1rl.telemetry import REWARD_COMPONENT_KEYS, StepTelemetry
from f1rl.track_model import TrackSpec, load_track_spec
from f1rl.track_sections import distance_to_next_braking_gate, section_for_progress


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
        self.actions = actions_for_action_set(self.config.action_set)
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
        self.segment_target_min_speed_kph: float | None = None
        self.segment_target_max_speed_kph: float | None = None
        self.segment_target_max_lateral_error_m: float | None = None
        self.segment_target_max_heading_error_deg: float | None = None
        self.segment_target_max_abs_yaw_rate_rps: float | None = None
        self.segment_target_max_abs_steering: float | None = None
        self.segment_fail_on_speed_gate_miss = False
        self.segment_require_release = False
        self.segment_release_observed = False
        self.segment_release_max_brake = 0.1
        self.segment_release_max_throttle = 0.1
        self.segment_release_min_speed_kph = 135.0
        self.segment_release_max_speed_kph = 210.0
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
        brake_profiles = {"brake", "guidance", "racing", "racing_release", "racing_v2"}
        guidance_profiles = {"guidance", "racing", "racing_release", "racing_v2"}
        racing_profiles = {"racing", "racing_release", "racing_v2"}
        brake_features = 3 if self.config.observation_profile in brake_profiles else 0
        guidance_features = 2 if self.config.observation_profile in guidance_profiles else 0
        racing_features = 0
        if self.config.observation_profile in racing_profiles:
            racing_features = 1 + 2 + len(self.config.lookahead_m) + 1
        if self.config.observation_profile == "racing_release":
            racing_features += 4
        if self.config.observation_profile == "racing_v2":
            racing_features += 4
        return (
            7
            + len(self.sensor_angles)
            + len(self.config.lookahead_m)
            + brake_features
            + guidance_features
            + racing_features
        )

    @property
    def action_dim(self) -> int:
        if self.config.action_mode == "continuous":
            return 2
        if self.config.action_mode == "multidiscrete":
            drive_count, steer_count = multidiscrete_action_nvec()
            return drive_count * steer_count
        return len(self.actions)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        options = options or {}
        rng = np.random.default_rng(seed)
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
        self.segment_complete = False
        self.curriculum_stage = options.get("curriculum_stage")
        self.segment_target_progress_m = None
        self.segment_target_min_speed_kph = None
        self.segment_target_max_speed_kph = None
        self.segment_target_max_lateral_error_m = None
        self.segment_target_max_heading_error_deg = None
        self.segment_target_max_abs_yaw_rate_rps = None
        self.segment_target_max_abs_steering = None
        self.segment_fail_on_speed_gate_miss = bool(options.get("segment_fail_on_speed_gate_miss", False))
        self.segment_require_release = bool(options.get("segment_require_release", False))
        self.segment_release_observed = False
        self.segment_release_max_brake = float(options.get("segment_release_max_brake", 0.1))
        self.segment_release_max_throttle = float(options.get("segment_release_max_throttle", 0.1))
        self.segment_release_min_speed_kph = float(
            options.get("segment_release_min_speed_kph", self.config.reward.scaffold_release_min_speed_kph)
        )
        self.segment_release_max_speed_kph = float(
            options.get("segment_release_max_speed_kph", self.config.reward.scaffold_release_max_speed_kph)
        )
        self.last_telemetry = None
        if "state_snapshot" in options:
            self._restore_snapshot(options["state_snapshot"], rng=rng, options=options)
        else:
            self.state = self._initial_state_from_options(options, rng)
            self.last_action_id = 0
            self.last_throttle = 0.0
            self.last_brake = 0.0
            self.last_steer = 0.0
            self._reset_projection()
            self._reset_lap_validity()
        self.episode_start_progress_m = self.state.monotonic_progress_m
        segment_length_m = options.get("segment_length_m")
        if segment_length_m is not None:
            self.segment_target_progress_m = self.episode_start_progress_m + max(float(segment_length_m), 1.0)
            target_min_speed_kph = options.get("segment_target_min_speed_kph")
            if target_min_speed_kph is not None:
                self.segment_target_min_speed_kph = float(target_min_speed_kph)
            target_max_speed_kph = options.get("segment_target_max_speed_kph")
            if target_max_speed_kph is not None:
                self.segment_target_max_speed_kph = float(target_max_speed_kph)
            target_max_lateral_error_m = options.get("segment_target_max_lateral_error_m")
            if target_max_lateral_error_m is not None:
                self.segment_target_max_lateral_error_m = float(target_max_lateral_error_m)
            target_max_heading_error_deg = options.get("segment_target_max_heading_error_deg")
            if target_max_heading_error_deg is not None:
                self.segment_target_max_heading_error_deg = float(target_max_heading_error_deg)
            target_max_abs_yaw_rate_rps = options.get("segment_target_max_abs_yaw_rate_rps")
            if target_max_abs_yaw_rate_rps is not None:
                self.segment_target_max_abs_yaw_rate_rps = float(target_max_abs_yaw_rate_rps)
            target_max_abs_steering = options.get("segment_target_max_abs_steering")
            if target_max_abs_steering is not None:
                self.segment_target_max_abs_steering = float(target_max_abs_steering)
        if bool(options.get("collect_observation", True)):
            obs = self.observation()
        else:
            obs = np.zeros(self.observation_dim, dtype=np.float32)
        return obs, self.info(0.0, {key: 0.0 for key in REWARD_COMPONENT_KEYS}, False, False)

    def _restore_snapshot(self, value: object, *, rng: np.random.Generator, options: dict) -> None:
        snapshot = snapshot_from_mapping(value)  # type: ignore[arg-type]
        self.state = snapshot_to_car_state(snapshot)
        position_noise_m = float(options.get("position_noise_m", 0.0) or 0.0)
        if position_noise_m > 0.0:
            noise_px = rng.normal(0.0, position_noise_m / max(self.track.meters_per_pixel, 1e-6), size=2)
            self.state.x += float(noise_px[0])
            self.state.y += float(noise_px[1])
        heading_noise_deg = float(options.get("heading_noise_deg", 0.0) or 0.0)
        if heading_noise_deg > 0.0:
            self.state.heading_rad = wrap_radians(
                self.state.heading_rad + float(np.deg2rad(rng.normal(0.0, heading_noise_deg)))
            )
        speed_noise_kph = float(options.get("speed_noise_kph", 0.0) or 0.0)
        if speed_noise_kph > 0.0:
            self.state.speed_mps = max(
                0.0,
                self.state.speed_mps + float(rng.normal(0.0, speed_noise_kph)) / 3.6,
            )
        self.last_action_id = snapshot.last_action_id
        self.last_throttle = float(np.clip(snapshot.last_throttle, 0.0, 1.0))
        self.last_brake = float(np.clip(snapshot.last_brake, 0.0, 1.0))
        self.last_steer = float(np.clip(snapshot.last_steer, -1.0, 1.0))
        self._last_raw_progress_px = snapshot.raw_progress_m / self.track.meters_per_pixel
        self._ray_cache_key = None
        self._ray_cache_distances_m = None
        if position_noise_m > 0.0:
            self._reset_projection()
        else:
            self.state.raw_progress_m = snapshot.raw_progress_m
            self.state.monotonic_progress_m = snapshot.monotonic_progress_m
        self.next_checkpoint_index = int(snapshot.next_checkpoint_index)
        self.checkpoints_passed = int(snapshot.checkpoints_passed)
        self.missed_checkpoint_count = int(snapshot.missed_checkpoint_count)
        self.valid_lap = bool(snapshot.valid_lap)
        self.finish_crossed = False
        self.completed_lap = False
        self.segment_complete = False
        self.segment_target_min_speed_kph = None
        self.segment_target_max_speed_kph = None
        self.segment_target_max_lateral_error_m = None
        self.segment_target_max_heading_error_deg = None
        self.segment_target_max_abs_yaw_rate_rps = None
        self.segment_target_max_abs_steering = None
        if self.curriculum_stage is None:
            self.curriculum_stage = snapshot.curriculum_stage

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

    def _track_errors(self) -> tuple[float, float, float, float]:
        raw_px, lateral_px, tangent, projection = project_point_to_polyline(
            self.state.position(),
            self.track.centerline,
            self.track.centerline_s,
            previous_progress=self._last_raw_progress_px,
            window=self.config.local_projection_window_m / self.track.meters_per_pixel,
        )
        heading_error = wrap_radians(tangent - self.state.heading_rad)
        signed_lateral_px = self._signed_lateral_px(projection, tangent, lateral_px)
        return raw_px, lateral_px * self.track.meters_per_pixel, heading_error, signed_lateral_px * self.track.meters_per_pixel

    def _signed_lateral_px(self, projection: np.ndarray, tangent_heading_rad: float, lateral_px: float) -> float:
        if lateral_px <= 1e-6:
            return 0.0
        offset = self.state.position() - projection
        tangent = np.asarray([np.cos(tangent_heading_rad), -np.sin(tangent_heading_rad)], dtype=np.float32)
        cross = float(tangent[0] * offset[1] - tangent[1] * offset[0])
        if abs(cross) <= 1e-9:
            return 0.0
        return float(np.sign(cross) * lateral_px)

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

    def _target_speed_kph(self, lookahead_heading_errors: list[float]) -> float:
        lookahead_heading_deg = max(abs(value) for value in lookahead_heading_errors) * 180.0
        return max(
            self.config.reward.speed_target_min_kph,
            self.config.reward.speed_target_max_kph
            - self.config.reward.speed_target_heading_scale * lookahead_heading_deg,
        )

    def _lookahead_target_speed_features(self, lookahead_heading_errors: list[float]) -> list[float]:
        max_speed_kph = max(self.config.car.max_speed_mps * 3.6, 1e-6)
        target_speeds = [
            max(
                self.config.reward.speed_target_min_kph,
                self.config.reward.speed_target_max_kph
                - self.config.reward.speed_target_heading_scale * abs(error) * 180.0,
            )
            for error in lookahead_heading_errors
        ]
        return [float(np.clip(speed / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0) for speed in target_speeds]

    def _brake_observation_features(self, target_speed_kph: float) -> list[float]:
        speed_kph = self.state.speed_mps * 3.6
        max_speed_kph = max(self.config.car.max_speed_mps * 3.6, 1e-6)
        target_norm = np.clip(target_speed_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0
        speed_error_norm = np.clip((speed_kph - target_speed_kph) / 200.0, -1.0, 1.0)
        brake_demand_norm = np.clip(
            (speed_kph - target_speed_kph - self.config.reward.speed_target_deadzone_kph) / 200.0,
            0.0,
            1.0,
        )
        return [float(target_norm), float(speed_error_norm), float(brake_demand_norm * 2.0 - 1.0)]

    def _target_steer(self) -> float:
        speed_kph = self.state.speed_mps * 3.6
        lookahead_m = float(np.clip(35.0 + speed_kph * 0.28, 45.0, 145.0))
        target_px = (
            self.state.raw_progress_m / self.track.meters_per_pixel
            + lookahead_m / self.track.meters_per_pixel
        )
        target = sample_polyline_at(
            self.track.centerline,
            self.track.centerline_s,
            np.asarray([target_px], dtype=np.float32),
        )[0]
        dx = float(target[0] - self.state.x)
        dy = float(target[1] - self.state.y)
        desired = float(np.arctan2(-dy, dx))
        heading_error = wrap_radians(desired - self.state.heading_rad)
        max_steer_rad = max(float(np.deg2rad(self.config.car.max_steer_deg)), 1e-6)
        return float(np.clip(heading_error / max_steer_rad, -1.0, 1.0))

    def _guidance_observation_features(self, target_steer: float) -> list[float]:
        steer_error = float(np.clip(target_steer - self.last_steer, -1.0, 1.0))
        return [float(target_steer), steer_error]

    def _racing_observation_features(self, signed_lateral_error_m: float, lookahead_errors: list[float]) -> list[float]:
        distance_to_gate_m = distance_to_next_braking_gate(self.state.monotonic_progress_m)
        gate_distance_norm = np.clip(distance_to_gate_m / 1000.0, 0.0, 1.0) * 2.0 - 1.0
        return [
            float(np.clip(signed_lateral_error_m / 30.0, -1.0, 1.0)),
            float(np.clip(self.last_throttle, 0.0, 1.0) * 2.0 - 1.0),
            float(np.clip(self.last_brake, 0.0, 1.0) * 2.0 - 1.0),
            *self._lookahead_target_speed_features(lookahead_errors),
            float(gate_distance_norm),
        ]

    def _section_brake_observation_features(self) -> list[float]:
        section = section_for_progress(self.state.monotonic_progress_m)
        lap_progress_m = self.state.monotonic_progress_m % self.track.length_m
        speed_kph = self.state.speed_mps * 3.6
        max_speed_kph = max(self.config.car.max_speed_mps * 3.6, 1e-6)
        target_norm = np.clip(section.target_speed_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0
        surplus_norm = np.clip(
            (speed_kph - section.target_speed_kph - self.config.reward.speed_target_deadzone_kph) / 220.0,
            0.0,
            1.0,
        )
        in_brake_zone = (
            section.brake_start_m is not None
            and section.turn_in_m is not None
            and section.brake_start_m <= lap_progress_m <= section.turn_in_m
        )
        brake_zone_flag = 1.0 if in_brake_zone else -1.0
        phase = -1.0
        if section.brake_start_m is not None and section.turn_in_m is not None:
            span_m = max(section.turn_in_m - section.brake_start_m, 1e-6)
            phase = np.clip((lap_progress_m - section.brake_start_m) / span_m, 0.0, 1.0) * 2.0 - 1.0
        return [
            float(target_norm),
            float(surplus_norm * 2.0 - 1.0),
            float(brake_zone_flag),
            float(phase),
        ]

    def _release_observation_features(self) -> list[float]:
        speed_kph = self.state.speed_mps * 3.6
        max_speed_kph = max(self.config.car.max_speed_mps * 3.6, 1e-6)
        release_min_kph = min(
            self.config.reward.scaffold_release_min_speed_kph,
            self.config.reward.scaffold_release_max_speed_kph,
        )
        release_max_kph = max(
            self.config.reward.scaffold_release_min_speed_kph,
            self.config.reward.scaffold_release_max_speed_kph,
        )
        threshold_norm = np.clip(release_max_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0
        surplus_norm = np.clip((speed_kph - release_max_kph) / 200.0, -1.0, 1.0)
        deficit_norm = np.clip((release_min_kph - speed_kph) / 120.0, -1.0, 1.0)
        in_band = 1.0 if release_min_kph <= speed_kph <= release_max_kph else -1.0
        return [
            float(threshold_norm),
            float(surplus_norm),
            float(deficit_norm),
            float(in_band),
        ]

    def search_features(
        self,
        *,
        segment_start_progress_m: float | None = None,
        segment_target_progress_m: float | None = None,
    ) -> dict[str, float]:
        """Normalized features for non-PPO search controllers."""
        _, lateral_error_m, heading_error, signed_lateral_error_m = self._track_errors()
        lookahead_errors = self._lookahead_heading_errors()
        target_speed_kph = self._target_speed_kph(lookahead_errors)
        speed_kph = self.state.speed_mps * 3.6
        max_speed_kph = max(self.config.car.max_speed_mps * 3.6, 1e-6)
        speed_error_norm = float(np.clip((speed_kph - target_speed_kph) / 220.0, -1.0, 1.0))
        brake_demand = float(
            np.clip(
                (speed_kph - target_speed_kph - self.config.reward.speed_target_deadzone_kph) / 220.0,
                0.0,
                1.0,
            )
        )
        if segment_start_progress_m is not None and segment_target_progress_m is not None:
            segment_span_m = max(float(segment_target_progress_m) - float(segment_start_progress_m), 1e-6)
            segment_progress_ratio = float(
                np.clip((self.state.monotonic_progress_m - float(segment_start_progress_m)) / segment_span_m, 0.0, 1.0)
            )
        else:
            segment_progress_ratio = float((self.state.monotonic_progress_m % self.track.length_m) / self.track.length_m)
        curvature = self.state.yaw_rate_rps / max(self.state.speed_mps, 1e-6)
        features = {
            "bias": 1.0,
            "speed_norm": float(np.clip(speed_kph / max_speed_kph, 0.0, 1.0)),
            "target_speed_norm": float(np.clip(target_speed_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0),
            "speed_error_norm": speed_error_norm,
            "brake_demand": brake_demand,
            "signed_lateral_error_norm": float(np.clip(signed_lateral_error_m / 30.0, -1.0, 1.0)),
            "heading_error_norm": float(np.clip(heading_error / np.pi, -1.0, 1.0)),
            "yaw_rate_norm": float(np.clip(self.state.yaw_rate_rps / 2.0, -1.0, 1.0)),
            "curvature_norm": float(np.clip(curvature / 0.08, -1.0, 1.0)),
            "target_steer": float(self._target_steer()),
            "last_throttle": float(np.clip(self.last_throttle, 0.0, 1.0)),
            "last_brake": float(np.clip(self.last_brake, 0.0, 1.0)),
            "last_steer": float(np.clip(self.last_steer, -1.0, 1.0)),
            "segment_progress_ratio": segment_progress_ratio * 2.0 - 1.0,
            "lateral_error_m": float(lateral_error_m),
            "signed_lateral_error_m": float(signed_lateral_error_m),
            "heading_error_deg": float(np.rad2deg(heading_error)),
            "target_speed_kph": float(target_speed_kph),
            "speed_kph": float(speed_kph),
        }
        for index in range(4):
            features[f"lookahead_{index}"] = float(lookahead_errors[index]) if index < len(lookahead_errors) else 0.0
        return features

    def _scaffold_reward_components(
        self,
        *,
        progress_m: float,
        speed_kph: float,
        throttle: float,
        brake: float,
        lateral_error_m: float,
        heading_error_deg: float,
        min_ray_m: float,
        collided: bool,
        off_track: bool,
    ) -> dict[str, float]:
        reward = self.config.reward
        scale = float(reward.scaffold_scale)
        if scale <= 0.0:
            return {}
        section = section_for_progress(progress_m)
        if section.brake_start_m is None or section.turn_in_m is None:
            return {}
        lap_progress_m = progress_m % self.track.length_m
        target_speed_kph = section.target_speed_kph
        components: dict[str, float] = {}
        in_brake_zone = section.brake_start_m <= lap_progress_m <= section.turn_in_m
        release_min_kph = min(reward.scaffold_release_min_speed_kph, reward.scaffold_release_max_speed_kph)
        release_max_kph = max(reward.scaffold_release_min_speed_kph, reward.scaffold_release_max_speed_kph)
        brake_reward_target_kph = target_speed_kph
        if reward.scaffold_release_reward_scale > 0.0 or reward.scaffold_overbrake_penalty_scale > 0.0:
            brake_reward_target_kph = max(brake_reward_target_kph, release_max_kph)
        speed_surplus_kph = max(0.0, speed_kph - brake_reward_target_kph)
        speed_surplus_ratio = speed_surplus_kph / 100.0
        if in_brake_zone and speed_surplus_kph > self.config.reward.speed_target_deadzone_kph:
            components["scaffold_brake"] = scale * reward.scaffold_brake_reward_scale * speed_surplus_ratio * brake
            components["scaffold_no_throttle"] = (
                -scale * reward.scaffold_no_throttle_penalty_scale * speed_surplus_ratio * throttle
            )
        if in_brake_zone:
            release_width_kph = max(release_max_kph - release_min_kph, 1.0)
            release_center_kph = 0.5 * (release_min_kph + release_max_kph)
            release_score = max(0.0, 1.0 - abs(speed_kph - release_center_kph) / (0.5 * release_width_kph))
            release_control = max(0.0, 1.0 - brake) * max(0.0, 1.0 - throttle)
            components["scaffold_release"] = (
                scale * reward.scaffold_release_reward_scale * release_score * release_control
            )
            if speed_kph < release_min_kph:
                overbrake_ratio = (release_min_kph - speed_kph) / 100.0
                components["scaffold_overbrake"] = (
                    -scale * reward.scaffold_overbrake_penalty_scale * overbrake_ratio * brake
                )
            if reward.scaffold_brake_curve_penalty_scale > 0.0:
                span_m = max(section.turn_in_m - section.brake_start_m, 1.0)
                phase = float(np.clip((lap_progress_m - section.brake_start_m) / span_m, 0.0, 1.0))
                curve_start_kph = max(reward.scaffold_brake_curve_start_speed_kph, target_speed_kph)
                curve_target_kph = curve_start_kph + (target_speed_kph - curve_start_kph) * phase
                curve_error_kph = max(
                    0.0,
                    abs(speed_kph - curve_target_kph) - reward.scaffold_brake_curve_deadzone_kph,
                )
                components["scaffold_brake_curve"] = (
                    -scale * reward.scaffold_brake_curve_penalty_scale * curve_error_kph / 100.0
                )
        turn_in_distance_m = abs(lap_progress_m - section.turn_in_m)
        if turn_in_distance_m <= 35.0:
            speed_error_kph = max(0.0, abs(speed_kph - target_speed_kph) - 12.0)
            components["scaffold_turn_in_speed"] = (
                -scale * reward.scaffold_turn_in_speed_penalty_scale * speed_error_kph / 100.0
            )
        if section.exit_m is not None and section.turn_in_m <= lap_progress_m <= section.exit_m:
            clean_ratio = 0.0 if collided or off_track else max(0.0, min(min_ray_m / 20.0, 1.0))
            components["scaffold_apex_clean"] = scale * reward.scaffold_apex_clean_reward_scale * clean_ratio
            corridor_excess_m = max(
                0.0,
                abs(lateral_error_m) - reward.scaffold_corridor_center_deadzone_m,
            )
            components["scaffold_corridor_center"] = (
                -scale * reward.scaffold_corridor_center_penalty_scale * (corridor_excess_m / 10.0) ** 2
            )
        if section.exit_m is not None and abs(lap_progress_m - section.exit_m) <= 60.0:
            alignment_ratio = max(0.0, 1.0 - abs(heading_error_deg) / 35.0)
            useful_exit_speed_kph = max(target_speed_kph + 35.0, 120.0)
            speed_ratio = max(0.0, 1.0 - abs(speed_kph - useful_exit_speed_kph) / 90.0)
            components["scaffold_exit_alignment"] = (
                scale * reward.scaffold_exit_alignment_reward_scale * alignment_ratio
            )
            components["scaffold_exit_speed"] = scale * reward.scaffold_exit_speed_reward_scale * speed_ratio
        if (
            self.segment_target_progress_m is not None
            and reward.scaffold_segment_speed_penalty_scale > 0.0
            and (
                self.segment_target_min_speed_kph is not None
                or self.segment_target_max_speed_kph is not None
            )
        ):
            remaining_m = self.segment_target_progress_m - progress_m
            if -5.0 <= remaining_m <= 120.0:
                low_error_kph = (
                    max(0.0, self.segment_target_min_speed_kph - speed_kph)
                    if self.segment_target_min_speed_kph is not None
                    else 0.0
                )
                high_error_kph = (
                    max(0.0, speed_kph - self.segment_target_max_speed_kph)
                    if self.segment_target_max_speed_kph is not None
                    else 0.0
                )
                components["scaffold_segment_speed"] = (
                    -scale
                    * reward.scaffold_segment_speed_penalty_scale
                    * ((low_error_kph + high_error_kph) / 100.0) ** 2
                )
        return components

    def _assist_reward_components(
        self,
        *,
        progress_m: float,
        progress_reward: float,
        lateral_error_m: float,
        speed_kph: float,
        throttle: float,
        brake: float,
        steer: float,
    ) -> tuple[dict[str, float], str | None]:
        assist = self.config.assist
        if not assist.enabled:
            return {}, None
        section = section_for_progress(progress_m)
        if section.brake_start_m is None or section.turn_in_m is None:
            return {}, None
        lap_progress_m = progress_m % self.track.length_m
        target_speed_kph = section.target_speed_kph
        speed_surplus_kph = max(0.0, speed_kph - target_speed_kph)
        speed_surplus_ratio = speed_surplus_kph / 100.0
        components: dict[str, float] = {}
        termination_reason: str | None = None
        in_brake_zone = section.brake_start_m <= lap_progress_m <= section.turn_in_m
        if in_brake_zone and speed_surplus_kph > self.config.reward.speed_target_deadzone_kph:
            if abs(assist.brake_zone_progress_multiplier - 1.0) > 1e-12:
                components["assist_brake_zone_progress_suppression"] = (
                    progress_reward * (assist.brake_zone_progress_multiplier - 1.0)
                )
            components["assist_throttle_brake_demand"] = (
                -assist.throttle_brake_demand_penalty_scale * speed_surplus_ratio * throttle
            )
            if assist.throttle_brake_demand_terminate and throttle >= assist.throttle_brake_demand_min_throttle:
                termination_reason = "assist_throttle_brake_demand"
            if speed_kph >= assist.no_brake_min_speed_kph and brake < assist.no_brake_min_brake:
                components["assist_no_brake_gate"] = assist.no_brake_penalty
                if assist.no_brake_terminate and termination_reason is None:
                    termination_reason = "assist_no_brake_gate"
        if in_brake_zone and speed_kph < assist.overbrake_max_speed_kph and brake >= assist.overbrake_min_brake:
            components["assist_overbrake_gate"] = assist.overbrake_penalty * brake
            if assist.overbrake_terminate and termination_reason is None:
                termination_reason = "assist_overbrake_gate"
        near_turn_in = abs(lap_progress_m - section.turn_in_m) <= 35.0
        if near_turn_in and speed_kph > target_speed_kph + assist.overspeed_turn_in_margin_kph:
            components["assist_overspeed_gate"] = assist.overspeed_turn_in_penalty
            if assist.overspeed_turn_in_terminate:
                termination_reason = "assist_overspeed_gate"
        if (
            assist.steering_gate_end_m > assist.steering_gate_start_m
            and assist.steering_gate_start_m <= lap_progress_m <= assist.steering_gate_end_m
            and speed_kph >= assist.steering_gate_min_speed_kph
        ):
            min_abs_steer = max(0.0, assist.steering_gate_min_abs_steer)
            if assist.steering_gate_required_sign < -0.5:
                steering_gate_violation = steer > -min_abs_steer
            elif assist.steering_gate_required_sign > 0.5:
                steering_gate_violation = steer < min_abs_steer
            else:
                steering_gate_violation = abs(steer) < min_abs_steer
            if steering_gate_violation:
                components["assist_steering_gate"] = assist.steering_gate_penalty
                if assist.steering_gate_terminate and termination_reason is None:
                    termination_reason = "assist_steering_gate"
        if (
            assist.forbidden_steering_gate_end_m > assist.forbidden_steering_gate_start_m
            and assist.forbidden_steering_gate_start_m <= lap_progress_m <= assist.forbidden_steering_gate_end_m
            and speed_kph >= assist.forbidden_steering_gate_min_speed_kph
        ):
            min_abs_steer = max(0.0, assist.forbidden_steering_gate_min_abs_steer)
            if assist.forbidden_steering_gate_sign < -0.5:
                forbidden_steering_violation = steer < -min_abs_steer
            elif assist.forbidden_steering_gate_sign > 0.5:
                forbidden_steering_violation = steer > min_abs_steer
            else:
                forbidden_steering_violation = abs(steer) > min_abs_steer
            if forbidden_steering_violation:
                components["assist_forbidden_steering_gate"] = assist.forbidden_steering_gate_penalty
                if assist.forbidden_steering_gate_terminate and termination_reason is None:
                    termination_reason = "assist_forbidden_steering_gate"
        if assist.virtual_corridor_m > 0.0 and abs(lateral_error_m) > assist.virtual_corridor_m:
            components["assist_virtual_corridor"] = assist.virtual_corridor_penalty
            if assist.virtual_corridor_terminate and termination_reason is None:
                termination_reason = "assist_virtual_corridor"
        return components, termination_reason

    def observation(self) -> np.ndarray:
        _, lateral_error_m, heading_error, signed_lateral_error_m = self._track_errors()
        ray_obs = np.clip(self.ray_distances_m() / max(self.config.sensors.range_m, 1e-6), 0.0, 1.0) * 2.0 - 1.0
        progress_ratio = (self.state.monotonic_progress_m % self.track.length_m) / self.track.length_m
        lookahead_errors = self._lookahead_heading_errors()
        brake_features: list[float] = []
        guidance_features: list[float] = []
        racing_features: list[float] = []
        if self.config.observation_profile in {"brake", "guidance", "racing", "racing_release", "racing_v2"}:
            brake_features = self._brake_observation_features(self._target_speed_kph(lookahead_errors))
        if self.config.observation_profile in {"guidance", "racing", "racing_release", "racing_v2"}:
            guidance_features = self._guidance_observation_features(self._target_steer())
        if self.config.observation_profile in {"racing", "racing_release", "racing_v2"}:
            racing_features = self._racing_observation_features(signed_lateral_error_m, lookahead_errors)
        if self.config.observation_profile == "racing_release":
            racing_features = [*racing_features, *self._release_observation_features()]
        if self.config.observation_profile == "racing_v2":
            racing_features = [*racing_features, *self._section_brake_observation_features()]
        obs = np.asarray(
            [
                self.state.speed_mps / self.config.car.max_speed_mps,
                np.clip(self.state.yaw_rate_rps / 2.0, -1.0, 1.0),
                np.clip(heading_error / np.pi, -1.0, 1.0),
                np.clip(abs(lateral_error_m) / 30.0, 0.0, 1.0) * 2.0 - 1.0,
                progress_ratio * 2.0 - 1.0,
                self._last_action_observation(),
                self.last_steer,
                *ray_obs,
                *lookahead_errors,
                *brake_features,
                *guidance_features,
                *racing_features,
            ],
            dtype=np.float32,
        )
        return obs

    def _last_action_observation(self) -> float:
        if self.config.action_mode == "continuous":
            return float(np.clip(self.last_throttle - self.last_brake, -1.0, 1.0))
        return float((self.last_action_id / max(self.action_dim - 1, 1)) * 2.0 - 1.0)

    def step(self, action_id: int) -> SimStep:
        resolved_action_id = int(action_id) % self.action_dim
        throttle, brake, steer = action_to_controls(resolved_action_id, action_set=self.config.action_set)
        return self.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=resolved_action_id)

    def step_continuous(self, action: np.ndarray | list[float] | tuple[float, ...]) -> SimStep:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape[0] < 2:
            raise ValueError("Continuous actions must contain drive and steer values.")
        drive = float(np.clip(action_array[0], -1.0, 1.0))
        steer = float(np.clip(action_array[1], -1.0, 1.0))
        if self.config.continuous_action_scheme == "exclusive_throttle_bias":
            if drive >= 0.0:
                throttle = 0.25 + 0.75 * drive
                brake = 0.0
            else:
                throttle = 0.0
                brake = -drive
        elif self.config.continuous_action_scheme == "throttle_bias":
            throttle = 0.5 + 0.5 * drive
            brake = max(-drive, 0.0)
        else:
            throttle = max(drive, 0.0)
            brake = max(-drive, 0.0)
        return self.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=-1)

    def step_multidiscrete(self, action: np.ndarray | list[int] | tuple[int, ...]) -> SimStep:
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        throttle, brake, steer, action_id = multidiscrete_action_to_controls(action_array)
        return self.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=action_id)

    def _action_name(self, action_id: int) -> str:
        if action_id == -101:
            return "reference_control"
        if action_id == -100:
            return "reference_ghost"
        if action_id == -10:
            return "scripted_controls"
        if action_id < 0:
            return "continuous"
        if self.config.action_mode == "discrete":
            return self.actions[int(action_id) % len(self.actions)][0]
        if self.config.action_mode == "multidiscrete":
            _, steer_count = multidiscrete_action_nvec()
            drive_index = int(action_id) // steer_count
            steer_index = int(action_id) % steer_count
            return f"multidiscrete_drive_{drive_index}_steer_{steer_index}"
        return f"action_{action_id}"

    def step_controls(
        self,
        *,
        throttle: float,
        brake: float,
        steer: float,
        action_id: int = -1,
        collect_observation: bool = True,
        collect_rays: bool = True,
        compute_reward: bool = True,
    ) -> SimStep:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended; call reset() first")
        if (
            self.config.launch_guard_progress_m > 0.0
            and self.state.monotonic_progress_m <= self.config.launch_guard_progress_m
            and self.state.speed_mps * 3.6 < self.config.launch_guard_min_speed_kph
            and throttle <= 1e-6
            and brake > 1e-6
        ):
            throttle = max(float(self.config.launch_guard_throttle), 0.0)
            brake = 0.0
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

        raw_px, lateral_error_m, heading_error, _ = self._track_errors()
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

        heading_error_deg = abs(float(np.rad2deg(heading_error)))
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        if compute_reward:
            components["progress"] = progress_delta_m * self.config.reward.progress_scale
            lateral_excess_m = max(0.0, abs(lateral_error_m) - self.config.reward.lateral_deadzone_m)
            components["lateral"] = -self.config.reward.lateral_penalty_scale * lateral_excess_m
            ray_distances_m = self.ray_distances_m()
            min_ray_m = (
                float(np.min(ray_distances_m))
                if len(ray_distances_m)
                else self.config.reward.track_limit_safe_ray_m
            )
            track_limit_excess_m = max(0.0, self.config.reward.track_limit_safe_ray_m - min_ray_m)
            speed_factor = max(1.0, (self.state.speed_mps * 3.6) / 100.0)
            components["track_limit"] = -self.config.reward.track_limit_penalty_scale * track_limit_excess_m * speed_factor
            heading_excess_deg = max(0.0, heading_error_deg - self.config.reward.heading_deadzone_deg)
            components["heading"] = -self.config.reward.heading_penalty_scale * heading_excess_deg * speed_factor
            target_speed_kph = self._target_speed_kph(self._lookahead_heading_errors())
            speed_target_excess_kph = max(
                0.0,
                self.state.speed_mps * 3.6 - target_speed_kph - self.config.reward.speed_target_deadzone_kph,
            )
            components["speed_target"] = -self.config.reward.speed_target_penalty_scale * speed_target_excess_kph
            overspeed_ratio = speed_target_excess_kph / 100.0
            components["overspeed_action"] = (
                -self.config.reward.overspeed_throttle_penalty_scale * overspeed_ratio * throttle
                + self.config.reward.overspeed_brake_reward_scale * overspeed_ratio * brake
            )
            target_steer = self._target_steer()
            steering_target_error = max(
                0.0,
                abs(float(steer) - target_steer) - self.config.reward.steering_target_deadzone,
            )
            components["steering_target"] = (
                -self.config.reward.steering_target_penalty_scale * steering_target_error * speed_factor
            )
            components.update(
                self._scaffold_reward_components(
                    progress_m=self.state.monotonic_progress_m,
                    speed_kph=self.state.speed_mps * 3.6,
                    throttle=throttle,
                    brake=brake,
                    lateral_error_m=lateral_error_m,
                    heading_error_deg=heading_error_deg,
                    min_ray_m=min_ray_m,
                    collided=collided,
                    off_track=off_track,
                )
            )
            assist_components, assist_termination_reason = self._assist_reward_components(
                progress_m=self.state.monotonic_progress_m,
                progress_reward=components["progress"],
                lateral_error_m=lateral_error_m,
                speed_kph=self.state.speed_mps * 3.6,
                throttle=throttle,
                brake=brake,
                steer=steer,
            )
            components.update(assist_components)
            if assist_termination_reason is not None:
                self.terminated = True
                self.termination_reason = assist_termination_reason
        else:
            ray_distances_m = self.ray_distances_m() if collect_rays else np.empty(0, dtype=np.float32)

        speed_kph_after = self.state.speed_mps * 3.6
        if self.segment_require_release and not self.segment_release_observed:
            release_min_kph = min(self.segment_release_min_speed_kph, self.segment_release_max_speed_kph)
            release_max_kph = max(self.segment_release_min_speed_kph, self.segment_release_max_speed_kph)
            if (
                release_min_kph <= speed_kph_after <= release_max_kph
                and brake <= self.segment_release_max_brake
                and throttle <= self.segment_release_max_throttle
            ):
                self.segment_release_observed = True

        if not self.terminated and self.segment_target_progress_m is not None:
            target_crossed = (
                old_progress_m < self.segment_target_progress_m <= self.state.monotonic_progress_m
            )
            target_reached = self.state.monotonic_progress_m >= self.segment_target_progress_m
            speed_gate_ok = (
                (
                    self.segment_target_min_speed_kph is None
                    or speed_kph_after >= self.segment_target_min_speed_kph
                )
                and (
                    self.segment_target_max_speed_kph is None
                    or speed_kph_after <= self.segment_target_max_speed_kph
                )
            )
            lateral_gate_ok = (
                self.segment_target_max_lateral_error_m is None
                or abs(lateral_error_m) <= self.segment_target_max_lateral_error_m
            )
            heading_gate_ok = (
                self.segment_target_max_heading_error_deg is None
                or heading_error_deg <= self.segment_target_max_heading_error_deg
            )
            yaw_rate_gate_ok = (
                self.segment_target_max_abs_yaw_rate_rps is None
                or abs(self.state.yaw_rate_rps) <= self.segment_target_max_abs_yaw_rate_rps
            )
            steering_gate_ok = (
                self.segment_target_max_abs_steering is None
                or abs(self.state.steering) <= self.segment_target_max_abs_steering
            )
            release_gate_ok = not self.segment_require_release or self.segment_release_observed
            if (
                target_reached
                and speed_gate_ok
                and lateral_gate_ok
                and heading_gate_ok
                and yaw_rate_gate_ok
                and steering_gate_ok
                and release_gate_ok
            ):
                self.segment_complete = True
                components["finish"] = self.config.reward.finish_bonus
                self.truncated = True
                self.termination_reason = "segment_complete"
            elif target_crossed and speed_gate_ok and self.segment_require_release and not self.segment_release_observed:
                self.truncated = True
                self.termination_reason = "segment_release_gate_failed"
            elif target_crossed and self.segment_fail_on_speed_gate_miss:
                if self.segment_target_min_speed_kph is not None and speed_kph_after < self.segment_target_min_speed_kph:
                    self.truncated = True
                    self.termination_reason = "segment_min_speed_gate_failed"
                elif self.segment_target_max_speed_kph is not None and speed_kph_after > self.segment_target_max_speed_kph:
                    self.truncated = True
                    self.termination_reason = "segment_speed_gate_failed"
                elif self.segment_target_max_lateral_error_m is not None and not lateral_gate_ok:
                    self.truncated = True
                    self.termination_reason = "segment_lateral_gate_failed"
                elif self.segment_target_max_heading_error_deg is not None and not heading_gate_ok:
                    self.truncated = True
                    self.termination_reason = "segment_heading_gate_failed"
                elif self.segment_target_max_abs_yaw_rate_rps is not None and not yaw_rate_gate_ok:
                    self.truncated = True
                    self.termination_reason = "segment_yaw_rate_gate_failed"
                elif self.segment_target_max_abs_steering is not None and not steering_gate_ok:
                    self.truncated = True
                    self.termination_reason = "segment_steering_gate_failed"

        finish_crossed_this_step = self._crossed_finish(movement, old_progress_m)
        if finish_crossed_this_step:
            self.finish_crossed = True
        lap_valid_now = (
            self.valid_lap
            and self.missed_checkpoint_count == 0
            and self.checkpoints_passed >= len(self.track.checkpoints) - 1
        )
        if (
            not self.terminated
            and
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
            action_name=self._action_name(int(action_id)),
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
            "segment_target_min_speed_kph": self.segment_target_min_speed_kph,
            "segment_target_max_speed_kph": self.segment_target_max_speed_kph,
            "segment_target_max_lateral_error_m": self.segment_target_max_lateral_error_m,
            "segment_target_max_heading_error_deg": self.segment_target_max_heading_error_deg,
            "segment_target_max_abs_yaw_rate_rps": self.segment_target_max_abs_yaw_rate_rps,
            "segment_target_max_abs_steering": self.segment_target_max_abs_steering,
            "segment_fail_on_speed_gate_miss": bool(self.segment_fail_on_speed_gate_miss),
            "segment_require_release": bool(self.segment_require_release),
            "segment_release_observed": bool(self.segment_release_observed),
            "raw_progress_m": float(self.state.raw_progress_m),
            "monotonic_progress_m": float(self.state.monotonic_progress_m),
            "reward": float(reward),
            "reward_components": dict(components),
            "collided": bool(collided),
            "off_track": bool(off_track),
            "termination_reason": self.termination_reason,
        }
