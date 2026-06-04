# pyright: reportPrivateImportUsage=false
"""Batched PyTorch vehicle dynamics for GPU evolution search."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from f1rl.gpu_types import GpuCarBatch, GpuCarParams, GpuMovementBatch


def grip_limit_g_batch(params: GpuCarParams, speed_mps: torch.Tensor) -> torch.Tensor:
    aero_grip = params.aero_grip_per_mps2 * speed_mps * speed_mps
    return torch.clamp(params.grip_g + aero_grip, min=params.grip_g, max=params.max_grip_g)


def apply_physics_batch(
    state: GpuCarBatch,
    *,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    params: GpuCarParams,
    meters_per_pixel: torch.Tensor,
) -> tuple[GpuCarBatch, GpuMovementBatch]:
    """Apply the same bicycle dynamics as ``f1rl.physics.apply_physics`` to a batch."""

    throttle = torch.clamp(throttle.to(device=state.device, dtype=state.dtype), 0.0, 1.0)
    brake = torch.clamp(brake.to(device=state.device, dtype=state.dtype), 0.0, 1.0)
    steer = torch.clamp(steer.to(device=state.device, dtype=state.dtype), -1.0, 1.0)
    dt = params.dt

    speed = torch.clamp(state.speed_mps, min=0.0)
    target_steering = steer * math.radians(params.max_steer_deg)
    steering_delta = target_steering - state.steering
    max_delta = params.steer_response * dt
    steering = state.steering + torch.clamp(steering_delta, min=-max_delta, max=max_delta)
    effective_steering = steering / (1.0 + params.steering_speed_sensitivity * speed * speed)

    max_total_accel = grip_limit_g_batch(params, speed) * 9.81
    steering_active = (torch.abs(effective_steering) > 1e-6) & (speed > 1e-6)
    requested_yaw = speed / max(params.wheelbase_m, 1e-6) * torch.tan(effective_steering)
    requested_yaw = torch.where(steering_active, requested_yaw, torch.zeros_like(requested_yaw))
    requested_lateral = torch.abs(speed * requested_yaw)
    lateral_accel = torch.minimum(requested_lateral, max_total_accel)
    longitudinal_capacity = torch.sqrt(
        torch.clamp(max_total_accel * max_total_accel - lateral_accel * lateral_accel, min=0.0)
    )
    drive_limit = torch.minimum(
        torch.full_like(longitudinal_capacity, min(params.engine_accel_mps2, params.max_drive_g * 9.81)),
        longitudinal_capacity,
    )
    brake_limit = torch.minimum(
        torch.full_like(longitudinal_capacity, min(params.brake_accel_mps2, params.max_brake_g * 9.81)),
        longitudinal_capacity,
    )

    longitudinal_accel = throttle * drive_limit - brake * brake_limit
    coasting = (throttle <= 1e-6) & (brake <= 1e-6)
    longitudinal_accel = longitudinal_accel - torch.where(
        coasting,
        torch.full_like(longitudinal_accel, params.rolling_resistance_mps2),
        torch.zeros_like(longitudinal_accel),
    )
    longitudinal_accel = longitudinal_accel - params.drag_coefficient * speed * speed
    speed = torch.clamp(speed + longitudinal_accel * dt, min=0.0, max=params.max_speed_mps)

    moving_steering = (torch.abs(effective_steering) > 1e-6) & (speed > 1e-6)
    yaw_rate = speed / max(params.wheelbase_m, 1e-6) * torch.tan(effective_steering)
    yaw_rate = torch.where(moving_steering, yaw_rate, torch.zeros_like(yaw_rate))
    lateral_accel = torch.abs(speed * yaw_rate)
    max_lateral = grip_limit_g_batch(params, speed) * 9.81
    yaw_rate = torch.where(
        lateral_accel > max_lateral,
        yaw_rate * max_lateral / torch.clamp(lateral_accel, min=1e-6),
        yaw_rate,
    )

    heading = torch.remainder(state.heading_rad + yaw_rate * dt + math.pi, 2.0 * math.pi) - math.pi
    distance_px = speed * dt / torch.clamp(meters_per_pixel.to(device=state.device, dtype=state.dtype), min=1e-6)
    dx = torch.cos(heading) * distance_px
    dy = -torch.sin(heading) * distance_px
    x_new = state.x + dx
    y_new = state.y + dy
    movement = GpuMovementBatch(x0=state.x, y0=state.y, x1=x_new, y1=y_new)
    return (
        replace(
            state,
            x=x_new,
            y=y_new,
            heading_rad=heading,
            speed_mps=speed,
            yaw_rate_rps=yaw_rate,
            steering=steering,
            elapsed_steps=state.elapsed_steps + 1,
        ),
        movement,
    )
