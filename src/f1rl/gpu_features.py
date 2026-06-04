# pyright: reportPrivateImportUsage=false
"""Batched controller feature extraction for GPU evolution search."""

from __future__ import annotations

import math

import torch

from f1rl.config import SimConfig
from f1rl.gpu_track import (
    distance_to_next_braking_gate_batch,
    lookahead_heading_errors_batch,
    sample_centerline_at_batch,
    track_errors_batch,
    wrap_radians_batch,
)
from f1rl.gpu_types import GpuCarBatch, GpuTrackTensors
from f1rl.track_sections import MONZA_SECTIONS


def controller_controls_batch(
    controller_weights: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched equivalent of evolution_search._controller_controls."""

    logits = torch.bmm(controller_weights, features.unsqueeze(2)).squeeze(2)
    throttle_raw = torch.sigmoid(torch.clamp(logits[:, 0], min=-40.0, max=40.0))
    brake_raw = torch.sigmoid(torch.clamp(logits[:, 1], min=-40.0, max=40.0))
    brake_dominates = brake_raw > throttle_raw
    throttle = torch.where(brake_dominates, throttle_raw * (1.0 - brake_raw), throttle_raw)
    brake = torch.where(brake_dominates, brake_raw, brake_raw * (1.0 - throttle_raw))
    steer = torch.tanh(logits[:, 2])
    return (
        torch.clamp(throttle, 0.0, 1.0),
        torch.clamp(brake, 0.0, 1.0),
        torch.clamp(steer, -1.0, 1.0),
    )


def braking_gate_tensor(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gates = [section.brake_start_m for section in MONZA_SECTIONS if section.brake_start_m is not None]
    if not gates:
        return torch.empty((0,), device=device, dtype=dtype)
    return torch.tensor([float(gate) for gate in gates], device=device, dtype=dtype)


def target_steer_batch(state: GpuCarBatch, track: GpuTrackTensors, config: SimConfig) -> torch.Tensor:
    speed_kph = state.speed_mps * 3.6
    lookahead_m = torch.clamp(35.0 + speed_kph * 0.28, min=45.0, max=145.0)
    target_px = state.raw_progress_m / torch.clamp(track.meters_per_pixel, min=1e-6) + lookahead_m / torch.clamp(
        track.meters_per_pixel,
        min=1e-6,
    )
    target, _ = sample_centerline_at_batch(track, target_px)
    dx = target[:, 0] - state.x
    dy = target[:, 1] - state.y
    desired = torch.atan2(-dy, dx)
    heading_error = wrap_radians_batch(desired - state.heading_rad)
    max_steer_rad = max(float(math.radians(config.car.max_steer_deg)), 1e-6)
    return torch.clamp(heading_error / max_steer_rad, min=-1.0, max=1.0)


def search_features_batch(
    state: GpuCarBatch,
    track: GpuTrackTensors,
    config: SimConfig,
    *,
    feature_names: tuple[str, ...],
    segment_start_progress_m: torch.Tensor,
    segment_target_progress_m: float,
    braking_gates_m: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the controller feature matrix and diagnostic tensors."""

    errors = track_errors_batch(
        state.position_xy(),
        state.heading_rad,
        track,
        previous_progress_px=state.last_raw_progress_px,
        window_px=float(config.local_projection_window_m) / torch.clamp(track.meters_per_pixel, min=1e-6),
    )
    lookahead_errors = lookahead_heading_errors_batch(
        raw_progress_m=state.raw_progress_m,
        heading_rad=state.heading_rad,
        lookahead_m=tuple(float(value) for value in config.lookahead_m),
        track=track,
    )
    if lookahead_errors.shape[1] == 0:
        lookahead_abs_max = torch.zeros_like(state.speed_mps)
    else:
        lookahead_abs_max = torch.max(torch.abs(lookahead_errors), dim=1).values
    target_speed_kph = torch.maximum(
        torch.full_like(state.speed_mps, config.reward.speed_target_min_kph),
        torch.full_like(state.speed_mps, config.reward.speed_target_max_kph)
        - config.reward.speed_target_heading_scale * lookahead_abs_max * 180.0,
    )
    max_speed_kph = max(float(config.car.max_speed_mps * 3.6), 1e-6)
    if lookahead_errors.shape[1] == 0:
        lookahead_target_speeds = target_speed_kph[:, None]
    else:
        lookahead_target_speeds = torch.clamp(
            torch.full_like(lookahead_errors, config.reward.speed_target_max_kph)
            - config.reward.speed_target_heading_scale * torch.abs(lookahead_errors) * 180.0,
            min=config.reward.speed_target_min_kph,
            max=max_speed_kph,
        )
    near_target_speed_kph = (
        lookahead_target_speeds[:, 0] if lookahead_target_speeds.shape[1] > 0 else target_speed_kph
    )
    min_future_target_speed_kph = torch.minimum(
        target_speed_kph,
        torch.min(lookahead_target_speeds, dim=1).values,
    )
    target_speed_drop_kph = torch.clamp(near_target_speed_kph - min_future_target_speed_kph, min=0.0)
    speed_kph = state.speed_mps * 3.6
    future_brake_demand = torch.clamp(
        (speed_kph - min_future_target_speed_kph - config.reward.speed_target_deadzone_kph) / 220.0,
        min=0.0,
        max=1.0,
    )
    gates_tensor = braking_gates_m
    if gates_tensor is None:
        gates_tensor = braking_gate_tensor(device=state.device, dtype=state.dtype)
    if gates_tensor.numel() > 0:
        braking_gate_distance_m = distance_to_next_braking_gate_batch(
            state.monotonic_progress_m,
            gates_m=gates_tensor,
            length_m=track.length_m,
        )
    else:
        braking_gate_distance_m = torch.ones_like(state.speed_mps) * track.length_m
    brake_gate_proximity = 1.0 - torch.clamp(braking_gate_distance_m / 900.0, min=0.0, max=1.0)
    speed_error_norm = torch.clamp((speed_kph - target_speed_kph) / 220.0, min=-1.0, max=1.0)
    brake_demand = torch.clamp(
        (speed_kph - target_speed_kph - config.reward.speed_target_deadzone_kph) / 220.0,
        min=0.0,
        max=1.0,
    )
    segment_span_m = torch.clamp(float(segment_target_progress_m) - segment_start_progress_m, min=1e-6)
    segment_progress_ratio = torch.clamp(
        (state.monotonic_progress_m - segment_start_progress_m) / segment_span_m,
        min=0.0,
        max=1.0,
    )
    curvature = state.yaw_rate_rps / torch.clamp(state.speed_mps, min=1e-6)
    target_steer = target_steer_batch(state, track, config)
    diagnostics: dict[str, torch.Tensor] = {
        "raw_progress_px": errors["raw_progress_px"],
        "lateral_error_m": errors["lateral_error_m"],
        "signed_lateral_error_m": errors["signed_lateral_error_m"],
        "heading_error_rad": errors["heading_error_rad"],
        "heading_error_deg": errors["heading_error_rad"] * (180.0 / math.pi),
        "target_speed_kph": target_speed_kph,
        "near_target_speed_kph": near_target_speed_kph,
        "min_future_target_speed_kph": min_future_target_speed_kph,
        "target_speed_drop_kph": target_speed_drop_kph,
        "braking_gate_distance_m": braking_gate_distance_m,
        "speed_kph": speed_kph,
        "future_brake_demand": future_brake_demand,
        "target_speed_drop_norm": torch.clamp(target_speed_drop_kph / 180.0, min=0.0, max=1.0),
        "brake_demand": brake_demand,
        "brake_gate_proximity": brake_gate_proximity,
        "brake_gate_distance_norm": torch.clamp(braking_gate_distance_m / 1000.0, min=0.0, max=1.0) * 2.0 - 1.0,
        "lookahead_abs_max": torch.clamp(lookahead_abs_max / math.pi, min=0.0, max=1.0),
        "target_steer": target_steer,
    }
    feature_values: dict[str, torch.Tensor] = {
        "bias": torch.ones_like(state.speed_mps),
        "speed_norm": torch.clamp(speed_kph / max_speed_kph, min=0.0, max=1.0),
        "target_speed_norm": torch.clamp(target_speed_kph / max_speed_kph, min=0.0, max=1.0) * 2.0 - 1.0,
        "speed_error_norm": speed_error_norm,
        "brake_demand": brake_demand,
        "future_brake_demand": future_brake_demand,
        "target_speed_drop_norm": diagnostics["target_speed_drop_norm"],
        "brake_gate_proximity": brake_gate_proximity,
        "brake_gate_distance_norm": diagnostics["brake_gate_distance_norm"],
        "lookahead_abs_max": diagnostics["lookahead_abs_max"],
        "signed_lateral_error_norm": torch.clamp(errors["signed_lateral_error_m"] / 30.0, min=-1.0, max=1.0),
        "heading_error_norm": torch.clamp(errors["heading_error_rad"] / math.pi, min=-1.0, max=1.0),
        "yaw_rate_norm": torch.clamp(state.yaw_rate_rps / 2.0, min=-1.0, max=1.0),
        "curvature_norm": torch.clamp(curvature / 0.08, min=-1.0, max=1.0),
        "target_steer": target_steer,
        "last_throttle": torch.clamp(state.last_throttle, min=0.0, max=1.0),
        "last_brake": torch.clamp(state.last_brake, min=0.0, max=1.0),
        "last_steer": torch.clamp(state.last_steer, min=-1.0, max=1.0),
        "segment_progress_ratio": segment_progress_ratio * 2.0 - 1.0,
    }
    for index in range(4):
        if index < lookahead_errors.shape[1]:
            feature_values[f"lookahead_{index}"] = lookahead_errors[:, index]
        else:
            feature_values[f"lookahead_{index}"] = torch.zeros_like(state.speed_mps)
    features = torch.stack(
        [feature_values.get(name, torch.zeros_like(state.speed_mps)) for name in feature_names],
        dim=1,
    )
    return features, diagnostics
