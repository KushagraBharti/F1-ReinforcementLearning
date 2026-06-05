# pyright: reportPrivateImportUsage=false
"""Batched GPU observations matching ``MonzaSim.observation``."""

from __future__ import annotations

import math

import torch

from f1rl.config import LEARNED_POLICY_V1_FEATURES, SimConfig
from f1rl.gpu_features import target_steer_batch
from f1rl.gpu_track import (
    distance_to_next_braking_gate_batch,
    lookahead_heading_errors_batch,
    ray_distances_batch,
    sensor_angles_tensor,
    track_errors_batch,
)
from f1rl.gpu_types import GpuCarBatch, GpuTrackTensors
from f1rl.track_sections import MONZA_SECTIONS


def _section_tensors(*, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "start": torch.tensor([section.start_m for section in MONZA_SECTIONS], device=device, dtype=dtype),
        "end": torch.tensor([section.end_m for section in MONZA_SECTIONS], device=device, dtype=dtype),
        "target": torch.tensor([section.target_speed_kph for section in MONZA_SECTIONS], device=device, dtype=dtype),
        "brake": torch.tensor(
            [section.brake_start_m if section.brake_start_m is not None else -1.0 for section in MONZA_SECTIONS],
            device=device,
            dtype=dtype,
        ),
        "turn": torch.tensor(
            [section.turn_in_m if section.turn_in_m is not None else -1.0 for section in MONZA_SECTIONS],
            device=device,
            dtype=dtype,
        ),
    }


def _section_values(
    progress_m: torch.Tensor,
    config: SimConfig,
    track: GpuTrackTensors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sections = _section_tensors(device=progress_m.device, dtype=progress_m.dtype)
    lap = torch.remainder(progress_m, track.length_m)
    mask = (sections["start"][None, :] <= lap[:, None]) & (lap[:, None] < sections["end"][None, :])
    index = torch.argmax(mask.to(torch.int64), dim=1)
    target = sections["target"][index]
    brake = sections["brake"][index]
    turn = sections["turn"][index]
    has_brake_zone = (brake >= 0.0) & (turn >= 0.0)
    in_brake_zone = has_brake_zone & (brake <= lap) & (lap <= turn)
    brake_zone_flag = torch.where(in_brake_zone, torch.ones_like(lap), torch.full_like(lap, -1.0))
    span = torch.clamp(turn - brake, min=1e-6)
    phase = torch.clamp((lap - brake) / span, min=0.0, max=1.0) * 2.0 - 1.0
    phase = torch.where(has_brake_zone, phase, torch.full_like(phase, -1.0))
    max_speed_kph = max(float(config.car.max_speed_mps * 3.6), 1e-6)
    target_norm = torch.clamp(target / max_speed_kph, min=0.0, max=1.0) * 2.0 - 1.0
    return target_norm, brake_zone_flag, phase, target


def observation_batch(
    state: GpuCarBatch,
    track: GpuTrackTensors,
    config: SimConfig,
    *,
    sensor_angles: torch.Tensor | None = None,
    ray_chunk_size: int = 256,
) -> torch.Tensor:
    """Return observation matrix shaped ``[N, observation_dim]``."""

    if sensor_angles is None:
        sensor_angles = sensor_angles_tensor(
            count=config.sensors.count,
            spread_deg=config.sensors.spread_deg,
            forward_bias=config.sensors.forward_bias,
            device=state.device,
            dtype=state.dtype,
        )
    errors = track_errors_batch(
        state.position_xy(),
        state.heading_rad,
        track,
        previous_progress_px=state.last_raw_progress_px,
        window_px=float(config.local_projection_window_m) / torch.clamp(track.meters_per_pixel, min=1e-6),
    )
    rays_m = ray_distances_batch(
        state.x,
        state.y,
        state.heading_rad,
        track,
        sensor_angles=sensor_angles,
        range_m=config.sensors.range_m,
        chunk_size=ray_chunk_size,
    )
    ray_obs = torch.clamp(rays_m / max(float(config.sensors.range_m), 1e-6), min=0.0, max=1.0) * 2.0 - 1.0
    progress_ratio = torch.remainder(state.monotonic_progress_m, track.length_m) / torch.clamp(track.length_m, min=1e-6)
    lookahead_errors = lookahead_heading_errors_batch(
        raw_progress_m=state.raw_progress_m,
        heading_rad=state.heading_rad,
        lookahead_m=tuple(float(value) for value in config.lookahead_m),
        track=track,
    )
    lookahead_abs_max = (
        torch.max(torch.abs(lookahead_errors), dim=1).values
        if lookahead_errors.shape[1] > 0
        else torch.zeros_like(state.speed_mps)
    )
    target_speed_kph = torch.maximum(
        torch.full_like(state.speed_mps, config.reward.speed_target_min_kph),
        torch.full_like(state.speed_mps, config.reward.speed_target_max_kph)
        - config.reward.speed_target_heading_scale * lookahead_abs_max * 180.0,
    )
    speed_kph = state.speed_mps * 3.6
    max_speed_kph = max(float(config.car.max_speed_mps * 3.6), 1e-6)
    target_steer = target_steer_batch(state, track, config)
    obs_parts = [
        torch.clamp(state.speed_mps / max(float(config.car.max_speed_mps), 1e-6), min=-1.0, max=1.0)[:, None],
        torch.clamp(state.yaw_rate_rps / 2.0, min=-1.0, max=1.0)[:, None],
        torch.clamp(errors["heading_error_rad"] / math.pi, min=-1.0, max=1.0)[:, None],
        (torch.clamp(torch.abs(errors["lateral_error_m"]) / 30.0, min=0.0, max=1.0) * 2.0 - 1.0)[:, None],
        (progress_ratio * 2.0 - 1.0)[:, None],
        torch.clamp(state.last_throttle - state.last_brake, min=-1.0, max=1.0)[:, None],
        torch.clamp(state.last_steer, min=-1.0, max=1.0)[:, None],
        ray_obs,
        lookahead_errors,
    ]
    if config.observation_profile in {"brake", "guidance", "racing", "racing_release", "racing_v2", "learned_policy_v1"}:
        target_norm = torch.clamp(target_speed_kph / max_speed_kph, min=0.0, max=1.0) * 2.0 - 1.0
        speed_error_norm = torch.clamp((speed_kph - target_speed_kph) / 200.0, min=-1.0, max=1.0)
        brake_demand_norm = torch.clamp(
            (speed_kph - target_speed_kph - config.reward.speed_target_deadzone_kph) / 200.0,
            min=0.0,
            max=1.0,
        )
        obs_parts.append(torch.stack((target_norm, speed_error_norm, brake_demand_norm * 2.0 - 1.0), dim=1))
    if config.observation_profile in {"guidance", "racing", "racing_release", "racing_v2", "learned_policy_v1"}:
        steer_error = torch.clamp(target_steer - state.last_steer, min=-1.0, max=1.0)
        obs_parts.append(torch.stack((target_steer, steer_error), dim=1))
    if config.observation_profile in {"racing", "racing_release", "racing_v2", "learned_policy_v1"}:
        if lookahead_errors.shape[1] > 0:
            lookahead_target_speeds = torch.clamp(
                torch.full_like(lookahead_errors, config.reward.speed_target_max_kph)
                - config.reward.speed_target_heading_scale * torch.abs(lookahead_errors) * 180.0,
                min=config.reward.speed_target_min_kph,
                max=max_speed_kph,
            )
        else:
            lookahead_target_speeds = torch.empty((state.size, 0), device=state.device, dtype=state.dtype)
        gate_distance_m = distance_to_next_braking_gate_batch(
            state.monotonic_progress_m,
            gates_m=torch.tensor(
                [section.brake_start_m for section in MONZA_SECTIONS if section.brake_start_m is not None],
                device=state.device,
                dtype=state.dtype,
            ),
            length_m=track.length_m,
        )
        racing_parts = [
            torch.clamp(errors["signed_lateral_error_m"] / 30.0, min=-1.0, max=1.0)[:, None],
            (torch.clamp(state.last_throttle, min=0.0, max=1.0) * 2.0 - 1.0)[:, None],
            (torch.clamp(state.last_brake, min=0.0, max=1.0) * 2.0 - 1.0)[:, None],
            torch.clamp(lookahead_target_speeds / max_speed_kph, min=0.0, max=1.0) * 2.0 - 1.0,
            (torch.clamp(gate_distance_m / 1000.0, min=0.0, max=1.0) * 2.0 - 1.0)[:, None],
        ]
        racing = torch.cat(racing_parts, dim=1)
        if config.observation_profile == "racing_release":
            release_min = min(config.reward.scaffold_release_min_speed_kph, config.reward.scaffold_release_max_speed_kph)
            release_max = max(config.reward.scaffold_release_min_speed_kph, config.reward.scaffold_release_max_speed_kph)
            threshold_norm = torch.full_like(state.speed_mps, release_max / max_speed_kph)
            threshold_norm = torch.clamp(threshold_norm, min=0.0, max=1.0) * 2.0 - 1.0
            surplus_norm = torch.clamp((speed_kph - release_max) / 200.0, min=-1.0, max=1.0)
            deficit_norm = torch.clamp((release_min - speed_kph) / 120.0, min=-1.0, max=1.0)
            in_band = torch.where(
                (speed_kph >= release_min) & (speed_kph <= release_max),
                torch.ones_like(speed_kph),
                torch.full_like(speed_kph, -1.0),
            )
            racing = torch.cat((racing, torch.stack((threshold_norm, surplus_norm, deficit_norm, in_band), dim=1)), dim=1)
        if config.observation_profile in {"racing_v2", "learned_policy_v1"}:
            target_norm, brake_zone_flag, phase, section_target = _section_values(
                state.monotonic_progress_m,
                config,
                track,
            )
            surplus_norm = torch.clamp(
                (speed_kph - section_target - config.reward.speed_target_deadzone_kph) / 220.0,
                min=0.0,
                max=1.0,
            )
            racing = torch.cat(
                (racing, torch.stack((target_norm, surplus_norm * 2.0 - 1.0, brake_zone_flag, phase), dim=1)),
                dim=1,
            )
        obs_parts.append(racing)
    if config.observation_profile == "learned_policy_v1":
        if lookahead_errors.shape[1] > 0:
            lookahead_target_speeds = torch.clamp(
                torch.full_like(lookahead_errors, config.reward.speed_target_max_kph)
                - config.reward.speed_target_heading_scale * torch.abs(lookahead_errors) * 180.0,
                min=config.reward.speed_target_min_kph,
                max=max_speed_kph,
            )
            near_target_speed_kph = lookahead_target_speeds[:, 0]
            min_future_target_speed_kph = torch.minimum(target_speed_kph, torch.min(lookahead_target_speeds, dim=1).values)
        else:
            near_target_speed_kph = target_speed_kph
            min_future_target_speed_kph = target_speed_kph
        gate_distance_m = distance_to_next_braking_gate_batch(
            state.monotonic_progress_m,
            gates_m=torch.tensor(
                [section.brake_start_m for section in MONZA_SECTIONS if section.brake_start_m is not None],
                device=state.device,
                dtype=state.dtype,
            ),
            length_m=track.length_m,
        )
        target_speed_drop_kph = torch.clamp(near_target_speed_kph - min_future_target_speed_kph, min=0.0)
        curvature = state.yaw_rate_rps / torch.clamp(state.speed_mps, min=1e-6)
        learned_by_name = {
            "bias": torch.ones_like(state.speed_mps),
            "speed_norm": torch.clamp(speed_kph / max_speed_kph, min=0.0, max=1.0),
            "target_speed_norm": torch.clamp(target_speed_kph / max_speed_kph, min=0.0, max=1.0) * 2.0 - 1.0,
            "speed_error_norm": torch.clamp((speed_kph - target_speed_kph) / 220.0, min=-1.0, max=1.0),
            "brake_demand": torch.clamp(
                (speed_kph - target_speed_kph - config.reward.speed_target_deadzone_kph) / 220.0,
                min=0.0,
                max=1.0,
            ),
            "future_brake_demand": torch.clamp(
                (speed_kph - min_future_target_speed_kph - config.reward.speed_target_deadzone_kph) / 220.0,
                min=0.0,
                max=1.0,
            ),
            "target_speed_drop_norm": torch.clamp(target_speed_drop_kph / 180.0, min=0.0, max=1.0),
            "brake_gate_proximity": 1.0 - torch.clamp(gate_distance_m / 900.0, min=0.0, max=1.0),
            "brake_gate_distance_norm": torch.clamp(gate_distance_m / 1000.0, min=0.0, max=1.0) * 2.0 - 1.0,
            "lookahead_abs_max": torch.clamp(lookahead_abs_max / math.pi, min=0.0, max=1.0),
            "signed_lateral_error_norm": torch.clamp(errors["signed_lateral_error_m"] / 30.0, min=-1.0, max=1.0),
            "heading_error_norm": torch.clamp(errors["heading_error_rad"] / math.pi, min=-1.0, max=1.0),
            "yaw_rate_norm": torch.clamp(state.yaw_rate_rps / 2.0, min=-1.0, max=1.0),
            "curvature_norm": torch.clamp(curvature / 0.08, min=-1.0, max=1.0),
            "target_steer": target_steer,
            "last_throttle": torch.clamp(state.last_throttle, min=0.0, max=1.0),
            "last_brake": torch.clamp(state.last_brake, min=0.0, max=1.0),
            "last_steer": torch.clamp(state.last_steer, min=-1.0, max=1.0),
            "segment_progress_ratio": progress_ratio * 2.0 - 1.0,
        }
        for index in range(4):
            if index < lookahead_errors.shape[1]:
                learned_by_name[f"lookahead_{index}"] = lookahead_errors[:, index]
            else:
                learned_by_name[f"lookahead_{index}"] = torch.zeros_like(state.speed_mps)
        obs_parts.append(
            torch.stack(
                [torch.clamp(learned_by_name[name], min=-1.0, max=1.0) for name in LEARNED_POLICY_V1_FEATURES],
                dim=1,
            )
        )
    return torch.cat(obs_parts, dim=1).to(dtype=torch.float32)
