"""Torch-native batched simulator and telemetry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from f1rl.config import EnvConfig
from f1rl.track import TrackDefinition, build_track


TERMINATION_REASONS = (
    "active",
    "collision",
    "no_progress",
    "reverse_speed",
    "reverse_progress",
    "heading_away",
    "wall_hugging",
    "checkpoint_deadline",
    "max_steps",
    "lap_complete",
)
TERMINATION_TO_CODE = {reason: idx for idx, reason in enumerate(TERMINATION_REASONS)}


@dataclass(slots=True, frozen=True)
class TelemetryThresholds:
    moving_speed_threshold_kph: float = 3.0
    moving_progress_threshold: float = 0.01


@dataclass(slots=True)
class TrackTensorBundle:
    definition: TrackDefinition
    goals: torch.Tensor
    goal_midpoints: torch.Tensor
    collision_segments: torch.Tensor
    centerline_segments: torch.Tensor
    meters_per_world_unit: float
    average_track_width: float


@dataclass(slots=True)
class BatchStepResult:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    telemetry: dict[str, torch.Tensor]
    checkpoint_events: list[dict[str, float | int]]


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def _segment_intersection_mask(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized segment intersection between [...,4] tensors."""
    x1, y1, x2, y2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x3, y3, x4, y4 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    denom_ok = torch.abs(denom) >= 1e-9
    safe_denom = torch.where(denom_ok, denom, torch.ones_like(denom))
    s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / safe_denom
    t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / safe_denom
    valid = denom_ok & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    px = x1 + s * (x2 - x1)
    py = y1 + s * (y2 - y1)
    return valid, px, py


def _nearest_intersection_distance(rays: torch.Tensor, segments: torch.Tensor, max_distance: float) -> torch.Tensor:
    rays_expanded = rays.unsqueeze(-2)
    seg_expanded = segments.view(*([1] * (rays.ndim - 1)), segments.shape[0], 4)
    valid, px, py = _segment_intersection_mask(rays_expanded, seg_expanded)
    dx = px - rays_expanded[..., 0]
    dy = py - rays_expanded[..., 1]
    distances = torch.sqrt(dx * dx + dy * dy)
    inf = torch.full_like(distances, max_distance)
    masked = torch.where(valid, distances, inf)
    return masked.min(dim=-1).values


def _point_to_segments_distance(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    start = segments[:, 0:2]
    end = segments[:, 2:4]
    line = end - start
    norm = torch.sum(line * line, dim=-1).clamp_min(1e-9)
    diff = points.unsqueeze(-2) - start.unsqueeze(0)
    t = (diff * line.unsqueeze(0)).sum(dim=-1) / norm.unsqueeze(0)
    t = t.clamp(0.0, 1.0)
    projection = start.unsqueeze(0) + t.unsqueeze(-1) * line.unsqueeze(0)
    distances = torch.linalg.norm(points.unsqueeze(-2) - projection, dim=-1)
    return distances.min(dim=-1).values


def _build_sensor_angles(config: EnvConfig) -> torch.Tensor:
    count = int(max(3, config.dynamics.sensor_count))
    if count % 2 == 0:
        count += 1
    base = torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
    biased = torch.sign(base) * torch.pow(torch.abs(base), config.dynamics.sensor_forward_bias)
    return biased * (config.dynamics.sensor_spread_deg / 2.0)


class TorchSimBatch:
    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        *,
        num_cars: int,
        device: str = "auto",
        track_definition: TrackDefinition | None = None,
        telemetry_thresholds: TelemetryThresholds | None = None,
    ) -> None:
        self.config = EnvConfig.from_dict(env_config)
        self.device = resolve_device(device)
        self.cpu_device = torch.device("cpu")
        self.num_cars = max(1, int(num_cars))
        self.telemetry_thresholds = telemetry_thresholds or TelemetryThresholds()
        definition = track_definition or build_track(self.config.track)
        self.track = self._upload_track(definition)
        self.sensor_angles = _build_sensor_angles(self.config).to(self.device)
        self.sensor_count = int(self.sensor_angles.shape[0])
        self.observation_dim = self.sensor_count + 6
        self.action_dim = 2
        self._goal_deadlines = list(self.config.checkpoint_deadlines)
        self._build_state()

    def _upload_track(self, definition: TrackDefinition) -> TrackTensorBundle:
        centerline_segments = torch.from_numpy(
            np.column_stack((definition.centerline[:-1], definition.centerline[1:])).astype(np.float32)
        )
        return TrackTensorBundle(
            definition=definition,
            goals=torch.from_numpy(definition.goals.astype(np.float32)).to(self.device),
            goal_midpoints=torch.from_numpy(
                np.column_stack(
                    (
                        (definition.goals[:, 0] + definition.goals[:, 2]) * 0.5,
                        (definition.goals[:, 1] + definition.goals[:, 3]) * 0.5,
                    )
                ).astype(np.float32)
            ).to(self.device),
            collision_segments=torch.from_numpy(definition.collision_segments.astype(np.float32)).to(self.device),
            centerline_segments=centerline_segments.to(self.device),
            meters_per_world_unit=float(definition.meters_per_world_unit),
            average_track_width=float(definition.average_track_width),
        )

    def _build_state(self) -> None:
        shape = (self.num_cars,)
        self.positions = torch.zeros((self.num_cars, 2), dtype=torch.float32, device=self.device)
        self.headings_deg = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.speeds = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.steering_deg = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.next_goal_idx = torch.ones(shape, dtype=torch.long, device=self.device)
        self.lap_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.step_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.total_reward = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.alive = torch.ones(shape, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.reason_codes = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.last_actions = torch.zeros((self.num_cars, 2), dtype=torch.float32, device=self.device)
        self.no_progress_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.reverse_speed_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.reverse_progress_events = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.heading_away_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.wall_hugging_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.positive_progress_total = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.distance_travelled = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.last_checkpoint_step = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.min_wall_distance_ratio = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.current_checkpoint_index = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.latest_heading_error = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.latest_goal_distance = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.latest_lateral_offset = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def reset(self, seed: int = 0) -> torch.Tensor:
        self._build_state()
        rng = np.random.default_rng(seed)
        start_pos = np.asarray(self.config.start_pos, dtype=np.float32)
        start_heading = self._default_spawn_heading()
        positions = np.repeat(start_pos[None, :], self.num_cars, axis=0)
        headings = np.full((self.num_cars,), start_heading, dtype=np.float32)
        if self.config.spawn_position_jitter_px > 0.0:
            jitter = rng.uniform(
                low=-self.config.spawn_position_jitter_px,
                high=self.config.spawn_position_jitter_px,
                size=(self.num_cars, 2),
            ).astype(np.float32)
            positions += jitter
        if self.config.spawn_heading_jitter_deg > 0.0:
            headings += rng.uniform(
                low=-self.config.spawn_heading_jitter_deg,
                high=self.config.spawn_heading_jitter_deg,
                size=(self.num_cars,),
            ).astype(np.float32)
        self.positions.copy_(torch.from_numpy(positions).to(self.device))
        self.headings_deg.copy_(torch.from_numpy(headings).to(self.device))
        self.next_goal_idx.fill_(1)
        self.current_checkpoint_index.zero_()
        return self.compute_observations()

    def _default_spawn_heading(self) -> float:
        midpoint = self.track.definition.goal_midpoint(1)
        dx = float(midpoint[0] - float(self.config.start_pos[0]))
        dy = float(midpoint[1] - float(self.config.start_pos[1]))
        return float(math.degrees(math.atan2(-dy, dx)))

    def _wrap_degrees(self, angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + 180.0, 360.0) - 180.0

    def _track_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_midpoints = self.track.goal_midpoints[self.next_goal_idx]
        delta = goal_midpoints - self.positions
        desired_heading = torch.rad2deg(torch.atan2(-delta[:, 1], delta[:, 0]))
        heading_error = self._wrap_degrees(desired_heading - self.headings_deg)
        heading_error_norm = (heading_error / 90.0).clamp(-1.0, 1.0)
        goal_distance = torch.linalg.norm(delta, dim=-1)
        goal_distance_norm = (goal_distance / self.config.dynamics.sensor_range).clamp(0.0, 1.0) * 2.0 - 1.0
        centerline_offset = _point_to_segments_distance(self.positions, self.track.centerline_segments)
        centerline_scale = max(self.track.average_track_width * 0.5, 1e-6)
        lateral_offset_norm = (centerline_offset / centerline_scale).clamp(0.0, 1.0) * 2.0 - 1.0
        self.latest_heading_error = heading_error_norm
        self.latest_goal_distance = goal_distance_norm
        self.latest_lateral_offset = lateral_offset_norm
        return heading_error_norm, goal_distance_norm, lateral_offset_norm

    def build_sensor_rays(self) -> torch.Tensor:
        angles = self.headings_deg.unsqueeze(-1) + self.sensor_angles.unsqueeze(0)
        heading = torch.deg2rad(angles)
        x = self.positions[:, 0].unsqueeze(-1)
        y = self.positions[:, 1].unsqueeze(-1)
        rays = torch.empty((self.num_cars, self.sensor_count, 4), dtype=torch.float32, device=self.device)
        rays[..., 0] = x
        rays[..., 1] = y
        rays[..., 2] = x + torch.cos(heading) * self.config.dynamics.sensor_range
        rays[..., 3] = y - torch.sin(heading) * self.config.dynamics.sensor_range
        return rays

    def sensor_distances(self) -> torch.Tensor:
        rays = self.build_sensor_rays()
        distances = _nearest_intersection_distance(
            rays,
            self.track.collision_segments,
            self.config.dynamics.sensor_range,
        )
        return distances

    def compute_observations(self) -> torch.Tensor:
        distances = self.sensor_distances()
        heading_error_norm, goal_distance_norm, lateral_offset_norm = self._track_state()
        distance_obs = (distances / self.config.dynamics.sensor_range).clamp(0.0, 1.0) * 2.0 - 1.0
        speed_min = -self.config.dynamics.max_reverse_speed
        speed_max = self.config.dynamics.max_speed
        speed_obs = ((self.speeds - speed_min) / max(speed_max - speed_min, 1e-6)).clamp(0.0, 1.0) * 2.0 - 1.0
        extras = torch.stack(
            (
                speed_obs,
                heading_error_norm.clamp(-1.0, 1.0),
                goal_distance_norm.clamp(-1.0, 1.0),
                lateral_offset_norm.clamp(-1.0, 1.0),
                self.last_actions[:, 0].clamp(-1.0, 1.0),
                self.last_actions[:, 1].clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
        return torch.cat((distance_obs, extras), dim=-1).to(dtype=torch.float32)

    def _goal_transition(self, movement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_goals = self.track.goals[self.next_goal_idx]
        prev_indices = torch.remainder(self.next_goal_idx - 2, self.track.goals.shape[0])
        prev_goals = self.track.goals[prev_indices]
        crossed_next = _segment_intersection_mask(movement, next_goals)[0]
        crossed_prev = _segment_intersection_mask(movement, prev_goals)[0]
        progress_delta = crossed_next.to(torch.long) - crossed_prev.to(torch.long)

        next_goal_idx = self.next_goal_idx.clone()
        lap_finished = torch.zeros_like(crossed_next)
        next_goal_idx = torch.where(crossed_next, torch.remainder(next_goal_idx + 1, self.track.goals.shape[0]), next_goal_idx)
        lap_finished = crossed_next & (next_goal_idx == 1)
        self.lap_count = self.lap_count + lap_finished.to(torch.long)
        next_goal_idx = torch.where(crossed_prev, torch.remainder(next_goal_idx - 1, self.track.goals.shape[0]), next_goal_idx)
        self.next_goal_idx = next_goal_idx
        self.current_checkpoint_index = torch.remainder(self.next_goal_idx - 1, self.track.goals.shape[0])
        return progress_delta, lap_finished

    def _movement_collision(self, movement: torch.Tensor) -> torch.Tensor:
        expanded = movement.unsqueeze(-2)
        segs = self.track.collision_segments.unsqueeze(0)
        valid = _segment_intersection_mask(expanded, segs)[0]
        return valid.any(dim=-1)

    def _set_termination(self, mask: torch.Tensor, reason: str) -> None:
        code = TERMINATION_TO_CODE[reason]
        new_mask = mask & ~(self.terminated | self.truncated)
        self.reason_codes = torch.where(new_mask, torch.full_like(self.reason_codes, code), self.reason_codes)
        if reason in {"max_steps", "lap_complete"}:
            self.truncated = self.truncated | new_mask
        else:
            self.terminated = self.terminated | new_mask

    def step(self, actions: torch.Tensor | np.ndarray) -> BatchStepResult:
        active = ~(self.terminated | self.truncated)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-1.0, 1.0)
        actions = torch.where(active.unsqueeze(-1), actions, self.last_actions)
        previous_actions = self.last_actions.clone()
        self.last_actions = actions

        steering_input = actions[:, 0]
        throttle_input = actions[:, 1]
        speed = self.speeds.clone()
        dyn = self.config.dynamics

        forward_mask = throttle_input > 0.0
        braking_mask = throttle_input < 0.0
        speed = torch.where(forward_mask, speed + throttle_input * dyn.engine_acceleration_rate * dyn.dt, speed)
        speed = torch.where(
            braking_mask & (speed > 0.0),
            torch.maximum(speed + throttle_input * dyn.brake_deceleration_rate * dyn.dt, torch.zeros_like(speed)),
            speed,
        )
        speed = torch.where(
            braking_mask & ~(speed > 0.0),
            speed + throttle_input * dyn.reverse_acceleration_rate * dyn.dt,
            speed,
        )
        no_throttle = ~forward_mask & ~braking_mask
        speed = torch.where(
            no_throttle & (speed > 0.0),
            torch.maximum(speed - dyn.coast_deceleration_rate * dyn.dt, torch.zeros_like(speed)),
            speed,
        )
        speed = torch.where(
            no_throttle & (speed < 0.0),
            torch.minimum(speed + dyn.coast_deceleration_rate * dyn.dt, torch.zeros_like(speed)),
            speed,
        )
        speed = speed * max(0.0, 1.0 - dyn.friction)
        speed = torch.where(
            torch.abs(speed) > 1e-6,
            speed - torch.sign(speed) * dyn.drag_coefficient * speed * speed * dyn.dt,
            speed,
        )
        speed = speed.clamp(-dyn.max_reverse_speed, dyn.max_speed)

        target_steering = steering_input * dyn.max_steer_angle_deg
        self.steering_deg = self.steering_deg + (target_steering - self.steering_deg) * dyn.steering_response
        speed_ratio = torch.minimum(torch.abs(speed) / max(dyn.max_speed, 1e-6), torch.ones_like(speed))
        effective_steer = self.steering_deg * (1.0 - dyn.high_speed_steer_reduction * speed_ratio * speed_ratio)
        yaw_rate = (speed / max(dyn.wheelbase, 1e-6)) * torch.tan(torch.deg2rad(effective_steer)) * dyn.lateral_grip
        headings = self._wrap_degrees(self.headings_deg + torch.rad2deg(yaw_rate) * dyn.dt)
        heading_rad = torch.deg2rad(headings)
        travel_distance = speed * dyn.dt * (1.0 / max(self.track.meters_per_world_unit, 1e-6))
        delta = torch.stack((torch.cos(heading_rad) * travel_distance, -torch.sin(heading_rad) * travel_distance), dim=-1)

        new_positions = self.positions + delta
        movement = torch.cat((self.positions, new_positions), dim=-1)
        self.distance_travelled = self.distance_travelled + torch.linalg.norm(delta, dim=-1)
        self.positions = torch.where(active.unsqueeze(-1), new_positions, self.positions)
        self.headings_deg = torch.where(active, headings, self.headings_deg)
        self.speeds = torch.where(active, speed, self.speeds)
        self.step_count = self.step_count + active.to(torch.long)

        progress_delta, lap_finished = self._goal_transition(movement)
        collided = self._movement_collision(movement) & active
        heading_error_norm, _, lateral_offset_norm = self._track_state()

        reward_cfg = self.config.reward
        reward = torch.full((self.num_cars,), reward_cfg.step_penalty, dtype=torch.float32, device=self.device)
        reward_components: dict[str, torch.Tensor] = {
            "step_penalty": torch.full_like(reward, reward_cfg.step_penalty),
            "progress": torch.zeros_like(reward),
            "collision": torch.zeros_like(reward),
            "speed": torch.zeros_like(reward),
            "alignment": torch.zeros_like(reward),
            "centerline": torch.zeros_like(reward),
            "stall": torch.zeros_like(reward),
            "steering_change": torch.zeros_like(reward),
            "lap_bonus": torch.zeros_like(reward),
            "reverse": torch.zeros_like(reward),
            "negative_speed": torch.zeros_like(reward),
        }

        forward_speed_ratio = torch.clamp(self.speeds, min=0.0) / max(dyn.max_speed, 1e-6)
        reward_components["speed"] = reward_cfg.forward_speed_weight * forward_speed_ratio * torch.clamp(
            1.0 - torch.abs(heading_error_norm), min=0.0
        )
        reverse_speed_ratio = torch.minimum(torch.abs(self.speeds) / max(dyn.max_reverse_speed, 1e-6), torch.ones_like(reward))
        reward_components["negative_speed"] = torch.where(
            self.speeds < 0.0,
            -reward_cfg.negative_speed_penalty_weight * reverse_speed_ratio,
            torch.zeros_like(reward),
        )
        reward_components["alignment"] = reward_cfg.alignment_weight * torch.clamp(
            1.0 - torch.abs(heading_error_norm),
            min=0.0,
        )
        reward_components["centerline"] = -reward_cfg.centerline_penalty_weight * torch.clamp(lateral_offset_norm, min=0.0)
        reward_components["steering_change"] = -reward_cfg.steering_change_penalty_weight * torch.abs(
            actions[:, 0] - previous_actions[:, 0]
        )
        reward_components["progress"] = torch.where(
            progress_delta > 0,
            torch.full_like(reward, reward_cfg.progress_reward) * progress_delta.to(torch.float32),
            torch.zeros_like(reward),
        )
        reward_components["reverse"] = torch.where(
            progress_delta < 0,
            torch.full_like(reward, reward_cfg.reverse_penalty) * torch.abs(progress_delta).to(torch.float32),
            torch.zeros_like(reward),
        )
        reward_components["lap_bonus"] = torch.where(
            lap_finished,
            torch.full_like(reward, reward_cfg.lap_bonus),
            torch.zeros_like(reward),
        )

        slow_mask = torch.abs(self.speeds) < reward_cfg.stall_speed_threshold
        reward_components["stall"] = torch.where(
            slow_mask,
            torch.full_like(reward, reward_cfg.idle_penalty),
            torch.zeros_like(reward),
        )
        no_progress_mask = ~(progress_delta > 0) & slow_mask
        self.no_progress_steps = torch.where(no_progress_mask, self.no_progress_steps + 1, torch.zeros_like(self.no_progress_steps))
        reward_components["stall"] = torch.where(
            no_progress_mask,
            reward_components["stall"] + reward_cfg.stall_penalty,
            reward_components["stall"],
        )
        self.reverse_speed_steps = torch.where(
            self.speeds <= -self.config.reverse_speed_threshold,
            self.reverse_speed_steps + 1,
            torch.zeros_like(self.reverse_speed_steps),
        )
        self.reverse_progress_events = torch.where(
            progress_delta < 0,
            self.reverse_progress_events + 1,
            torch.where((progress_delta > 0) | (self.speeds >= 0.0), torch.zeros_like(self.reverse_progress_events), self.reverse_progress_events),
        )
        heading_away_mask = (
            self.step_count > self.config.spawn_grace_steps
        ) & (torch.abs(heading_error_norm) >= self.config.heading_away_threshold)
        self.heading_away_steps = torch.where(heading_away_mask, self.heading_away_steps + 1, torch.zeros_like(self.heading_away_steps))
        self.positive_progress_total = self.positive_progress_total + torch.clamp(progress_delta, min=0)
        self.last_checkpoint_step = torch.where(progress_delta > 0, self.step_count, self.last_checkpoint_step)

        observations = self.compute_observations()
        wall_distance_ratio = torch.clamp((observations[:, : self.sensor_count] + 1.0) * 0.5, 0.0, 1.0).min(dim=-1).values
        self.min_wall_distance_ratio = torch.minimum(self.min_wall_distance_ratio, wall_distance_ratio)
        wall_hugging_mask = (
            self.step_count > self.config.spawn_grace_steps
        ) & (wall_distance_ratio <= self.config.wall_hugging_distance_ratio_threshold)
        self.wall_hugging_steps = torch.where(wall_hugging_mask, self.wall_hugging_steps + 1, torch.zeros_like(self.wall_hugging_steps))

        reward_components["collision"] = torch.where(
            collided,
            torch.full_like(reward, reward_cfg.collision_penalty),
            torch.zeros_like(reward),
        )
        reward = sum(reward_components.values())
        self.total_reward = self.total_reward + reward * active.to(torch.float32)
        self.alive = self.alive & ~collided

        self._set_termination(collided & active, "collision")
        if self.config.terminate_on_no_progress:
            self._set_termination(self.no_progress_steps >= self.config.no_progress_limit_steps, "no_progress")
        self._set_termination(self.reverse_speed_steps >= self.config.reverse_speed_limit_steps, "reverse_speed")
        self._set_termination(self.reverse_progress_events >= self.config.reverse_progress_limit, "reverse_progress")
        if self.config.heading_away_limit_steps > 0:
            self._set_termination(self.heading_away_steps >= self.config.heading_away_limit_steps, "heading_away")
        if self.config.wall_hugging_limit_steps > 0:
            self._set_termination(self.wall_hugging_steps >= self.config.wall_hugging_limit_steps, "wall_hugging")
        if self._goal_deadlines:
            checkpoint_missed = torch.zeros_like(active)
            for checkpoint_target, deadline_step in self._goal_deadlines:
                checkpoint_missed = checkpoint_missed | (
                    (self.step_count >= int(deadline_step)) & (self.positive_progress_total < int(checkpoint_target))
                )
            self._set_termination(checkpoint_missed, "checkpoint_deadline")
        self._set_termination(self.step_count >= self.config.max_steps, "max_steps")
        self._set_termination(self.lap_count >= self.config.max_laps, "lap_complete")

        moving_mask = (
            torch.abs(self.speeds * 3.6) >= self.telemetry_thresholds.moving_speed_threshold_kph
        ) | (progress_delta.to(torch.float32) >= self.telemetry_thresholds.moving_progress_threshold)

        checkpoint_events: list[dict[str, float | int]] = []
        progress_cpu = progress_delta.detach().cpu().tolist()
        positions_cpu = self.positions.detach().cpu().tolist()
        laps_cpu = self.lap_count.detach().cpu().tolist()
        checkpoints_cpu = self.current_checkpoint_index.detach().cpu().tolist()
        for idx, delta_value in enumerate(progress_cpu):
            if int(delta_value) == 0:
                continue
            checkpoint_events.append(
                {
                    "car_index": idx,
                    "progress_delta": int(delta_value),
                    "checkpoint_index": int(checkpoints_cpu[idx]),
                    "lap_count": int(laps_cpu[idx]),
                    "step": int(self.step_count[idx].item()),
                    "x": float(positions_cpu[idx][0]),
                    "y": float(positions_cpu[idx][1]),
                }
            )

        telemetry = {
            "x": self.positions[:, 0].clone(),
            "y": self.positions[:, 1].clone(),
            "heading_deg": self.headings_deg.clone(),
            "speed": self.speeds.clone(),
            "speed_kph": self.speeds * 3.6,
            "distance_travelled": self.distance_travelled.clone(),
            "lap_count": self.lap_count.clone(),
            "next_goal_idx": self.next_goal_idx.clone(),
            "current_checkpoint_index": self.current_checkpoint_index.clone(),
            "progress_delta": progress_delta.clone(),
            "positive_progress_total": self.positive_progress_total.clone(),
            "time_since_last_checkpoint": (self.step_count - self.last_checkpoint_step).clone(),
            "throttle": actions[:, 1].clone(),
            "steering": actions[:, 0].clone(),
            "brake": torch.clamp(-actions[:, 1], min=0.0),
            "min_wall_distance_ratio": self.min_wall_distance_ratio.clone(),
            "heading_error_norm": self.latest_heading_error.clone(),
            "goal_distance_norm": self.latest_goal_distance.clone(),
            "lateral_offset_norm": self.latest_lateral_offset.clone(),
            "no_progress_steps": self.no_progress_steps.clone(),
            "reverse_speed_steps": self.reverse_speed_steps.clone(),
            "reverse_progress_events": self.reverse_progress_events.clone(),
            "heading_away_steps": self.heading_away_steps.clone(),
            "wall_hugging_steps": self.wall_hugging_steps.clone(),
            "alive": (~(self.terminated | self.truncated)).clone(),
            "moving": moving_mask.clone(),
            "terminated": self.terminated.clone(),
            "truncated": self.truncated.clone(),
            "termination_code": self.reason_codes.clone(),
        }
        return BatchStepResult(
            observations=observations,
            rewards=reward,
            terminated=self.terminated.clone(),
            truncated=self.truncated.clone(),
            telemetry=telemetry,
            checkpoint_events=checkpoint_events,
        )

    def telemetry_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        telemetry = {
            key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
            for key, value in self.snapshot().items()
        }
        for idx in range(self.num_cars):
            rows.append(
                {
                    "car_index": idx,
                    "x": float(telemetry["x"][idx]),
                    "y": float(telemetry["y"][idx]),
                    "speed": float(telemetry["speed"][idx]),
                    "speed_kph": float(telemetry["speed_kph"][idx]),
                    "distance_travelled": float(telemetry["distance_travelled"][idx]),
                    "lap_count": int(telemetry["lap_count"][idx]),
                    "current_checkpoint_index": int(telemetry["current_checkpoint_index"][idx]),
                    "time_since_last_checkpoint": int(telemetry["time_since_last_checkpoint"][idx]),
                    "termination_reason": TERMINATION_REASONS[int(telemetry["termination_code"][idx])],
                }
            )
        return rows

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "x": self.positions[:, 0].clone(),
            "y": self.positions[:, 1].clone(),
            "heading_deg": self.headings_deg.clone(),
            "speed": self.speeds.clone(),
            "speed_kph": self.speeds * 3.6,
            "distance_travelled": self.distance_travelled.clone(),
            "lap_count": self.lap_count.clone(),
            "current_checkpoint_index": self.current_checkpoint_index.clone(),
            "positive_progress_total": self.positive_progress_total.clone(),
            "time_since_last_checkpoint": (self.step_count - self.last_checkpoint_step).clone(),
            "termination_code": self.reason_codes.clone(),
            "terminated": self.terminated.clone(),
            "truncated": self.truncated.clone(),
        }

    def runtime_metadata(self, *, policy_device: str, renderer_backend: str) -> dict[str, Any]:
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else None
        return {
            "sim_device": str(self.device),
            "policy_device": policy_device,
            "renderer_backend": renderer_backend,
            "cuda_available": torch.cuda.is_available(),
            "selected_device_name": device_name,
            "telemetry_thresholds": {
                "moving_speed_threshold_kph": self.telemetry_thresholds.moving_speed_threshold_kph,
                "moving_progress_threshold": self.telemetry_thresholds.moving_progress_threshold,
            },
            "track_length_meters": self.config.track.track_length_meters,
        }
