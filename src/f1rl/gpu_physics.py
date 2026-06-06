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


def _tire_lateral_force_v2_batch(
    slip_angle_rad: torch.Tensor,
    load_n: torch.Tensor,
    *,
    stiffness_n_per_rad: float,
    peak_mu: float | torch.Tensor,
    shape_c: float,
    slip_angle_peak_rad: float,
    post_peak_falloff: float,
    load_sensitivity: float,
    reference_load_n: float,
    surface_mu: float,
) -> torch.Tensor:
    load_n = torch.clamp(load_n, min=1.0)
    load_ratio = load_n / max(reference_load_n, 1.0)
    mu = peak_mu * surface_mu * (1.0 - load_sensitivity * torch.clamp(load_ratio - 1.0, min=0.0))
    peak_force_n = torch.clamp(mu * load_n, min=1.0)
    b = stiffness_n_per_rad / torch.clamp(shape_c * peak_force_n, min=1e-6)
    force = peak_force_n * torch.sin(shape_c * torch.atan(b * slip_angle_rad))
    excess = torch.clamp((torch.abs(slip_angle_rad) - slip_angle_peak_rad) / max(slip_angle_peak_rad, 1e-6), min=0.0, max=1.0)
    return force * (1.0 - post_peak_falloff * excess)


def _mechanical_grip_scale_v2_batch(speed_mps: torch.Tensor, params: GpuCarParams) -> torch.Tensor:
    transition = max(params.v2_mechanical_grip_transition_mps, 1e-6)
    t = torch.clamp(speed_mps / transition, min=0.0, max=1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)
    return params.v2_mechanical_grip_low_speed_scale + (
        params.v2_mechanical_grip_high_speed_scale - params.v2_mechanical_grip_low_speed_scale
    ) * smooth_t


def _weight_transfer_v2_batch(
    params: GpuCarParams,
    *,
    longitudinal_accel_mps2: torch.Tensor,
    lateral_accel_mps2: torch.Tensor,
    speed_mps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_front = params.mass * 9.81 * params.v2_front_weight_distribution
    static_rear = params.mass * 9.81 - static_front
    aero_total = params.v2_aero_downforce_n_per_mps2 * speed_mps * speed_mps
    aero_front = aero_total * params.v2_aero_balance_front
    aero_rear = aero_total - aero_front
    long_transfer = params.mass * longitudinal_accel_mps2 * params.v2_cg_height_m / max(params.wheelbase_m, 1e-6)
    lateral_unload = (
        torch.abs(params.mass * lateral_accel_mps2 * params.v2_cg_height_m / max(params.v2_track_width_m, 1e-6))
        * 0.12
    )
    front_load = torch.clamp(
        static_front + aero_front - long_transfer - lateral_unload * params.v2_front_weight_distribution,
        min=1.0,
    )
    rear_load = torch.clamp(
        static_rear + aero_rear + long_transfer - lateral_unload * (1.0 - params.v2_front_weight_distribution),
        min=1.0,
    )
    return front_load, rear_load


def _gear_rpm_v2_batch(speed_mps: torch.Tensor, params: GpuCarParams) -> tuple[torch.Tensor, torch.Tensor]:
    ratios = torch.tensor(params.v2_gear_ratios, device=speed_mps.device, dtype=speed_mps.dtype)
    wheel_rps = torch.clamp(speed_mps, min=0.0) / max(2.0 * math.pi * params.v2_wheel_radius_m, 1e-6)
    rpm_by_gear = torch.clamp(
        wheel_rps[:, None] * ratios[None, :] * params.v2_final_drive_ratio * 60.0,
        min=params.v2_idle_rpm,
        max=params.v2_max_rpm,
    )
    eligible = rpm_by_gear <= params.v2_shift_up_rpm
    first_eligible = torch.argmax(eligible.to(dtype=torch.int64), dim=1)
    all_over = ~torch.any(eligible, dim=1)
    gear_index = torch.where(all_over, torch.full_like(first_eligible, ratios.numel() - 1), first_eligible)
    rpm = rpm_by_gear[torch.arange(speed_mps.shape[0], device=speed_mps.device), gear_index]
    return gear_index + 1, rpm


def _torque_factor_v2_batch(rpm: torch.Tensor, params: GpuCarParams) -> torch.Tensor:
    low_span = max(params.v2_torque_peak_rpm - params.v2_idle_rpm, 1.0)
    low_ratio = torch.clamp((rpm - params.v2_idle_rpm) / low_span, min=0.0, max=1.0)
    low = params.v2_torque_low_rpm_factor + (1.0 - params.v2_torque_low_rpm_factor) * low_ratio
    high_span = max(params.v2_max_rpm - params.v2_torque_peak_rpm, 1.0)
    high_ratio = torch.clamp((rpm - params.v2_torque_peak_rpm) / high_span, min=0.0, max=1.0)
    high = 1.0 - (1.0 - params.v2_torque_high_rpm_factor) * high_ratio
    return torch.where(rpm <= params.v2_torque_peak_rpm, low, high)


def _apply_physics_v2_batch(
    state: GpuCarBatch,
    *,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    params: GpuCarParams,
    meters_per_pixel: torch.Tensor,
) -> tuple[GpuCarBatch, GpuMovementBatch]:
    throttle = torch.clamp(throttle.to(device=state.device, dtype=state.dtype), 0.0, 1.0)
    brake = torch.clamp(brake.to(device=state.device, dtype=state.dtype), 0.0, 1.0)
    steer = torch.clamp(steer.to(device=state.device, dtype=state.dtype), -1.0, 1.0)
    dt = params.dt
    speed = torch.clamp(state.speed_mps, min=0.0)

    target_steering = steer * math.radians(params.v2_max_steer_deg)
    steering_delta = target_steering - state.steering
    max_delta = params.v2_steer_response * dt
    steering = state.steering + torch.clamp(steering_delta, min=-max_delta, max=max_delta)
    wheel_lock = (brake >= params.v2_brake_lock_threshold) & (speed >= params.v2_brake_lock_min_speed_mps)
    effective_steering = steering / (1.0 + params.v2_steering_speed_sensitivity * speed * speed)
    effective_steering = torch.where(
        wheel_lock,
        effective_steering * torch.clamp(1.0 - params.v2_brake_lock_steer_loss * brake, min=0.0),
        effective_steering,
    )

    previous_lateral_accel = torch.abs(speed * state.yaw_rate_rps)
    front_load, rear_load = _weight_transfer_v2_batch(
        params,
        longitudinal_accel_mps2=torch.zeros_like(speed),
        lateral_accel_mps2=previous_lateral_accel,
        speed_mps=speed,
    )
    reference_front = params.mass * 9.81 * params.v2_front_weight_distribution
    reference_rear = params.mass * 9.81 - reference_front
    grip_scale = _mechanical_grip_scale_v2_batch(speed, params)
    front_slip = torch.atan2(
        state.yaw_rate_rps * params.v2_front_axle_distance_m,
        torch.clamp(torch.abs(speed), min=1e-3),
    ) - effective_steering
    rear_slip = -torch.atan2(
        state.yaw_rate_rps * params.v2_rear_axle_distance_m,
        torch.clamp(torch.abs(speed), min=1e-3),
    ) - params.v2_rear_slip_steer_coupling * effective_steering
    slip_peak = math.radians(params.v2_slip_angle_peak_deg)
    front_force = _tire_lateral_force_v2_batch(
        front_slip,
        front_load,
        stiffness_n_per_rad=params.v2_front_cornering_stiffness_n_per_rad,
        peak_mu=params.v2_front_peak_mu * grip_scale,
        shape_c=params.v2_tire_shape_c,
        slip_angle_peak_rad=slip_peak,
        post_peak_falloff=params.v2_post_peak_falloff,
        load_sensitivity=params.v2_load_sensitivity,
        reference_load_n=reference_front,
        surface_mu=params.v2_surface_mu,
    )
    rear_force = _tire_lateral_force_v2_batch(
        rear_slip,
        rear_load,
        stiffness_n_per_rad=params.v2_rear_cornering_stiffness_n_per_rad,
        peak_mu=params.v2_rear_peak_mu * grip_scale,
        shape_c=params.v2_tire_shape_c,
        slip_angle_peak_rad=slip_peak,
        post_peak_falloff=params.v2_post_peak_falloff,
        load_sensitivity=params.v2_load_sensitivity,
        reference_load_n=reference_rear,
        surface_mu=params.v2_surface_mu,
    )
    lateral_capacity = (torch.abs(front_force) + torch.abs(rear_force)) / max(params.mass, 1e-6)
    requested_yaw = speed / max(params.wheelbase_m, 1e-6) * torch.tan(effective_steering)
    steering_active = (torch.abs(effective_steering) > 1e-6) & (speed > 1e-6)
    requested_lateral = torch.abs(speed * requested_yaw)
    lateral_accel = torch.minimum(requested_lateral, lateral_capacity)
    yaw_rate = torch.sign(requested_yaw) * lateral_accel / torch.clamp(speed, min=1e-6)
    yaw_rate = torch.where(steering_active, yaw_rate, torch.zeros_like(yaw_rate))
    lateral_accel = torch.where(steering_active, lateral_accel, torch.zeros_like(lateral_accel))

    total_peak_force = (
        params.v2_front_peak_mu * grip_scale * front_load * params.v2_surface_mu
        + params.v2_rear_peak_mu * grip_scale * rear_load * params.v2_surface_mu
    )
    total_accel_limit = total_peak_force / max(params.mass, 1e-6)
    longitudinal_capacity = torch.sqrt(torch.clamp(total_accel_limit * total_accel_limit - lateral_accel * lateral_accel, min=0.0))
    _, rpm = _gear_rpm_v2_batch(speed, params)
    torque_factor = _torque_factor_v2_batch(rpm, params)
    power_force = params.v2_engine_power_w * params.v2_drivetrain_efficiency * torque_factor / torch.clamp(
        speed,
        min=params.v2_power_min_speed_mps,
    )
    drive_limit = torch.minimum(
        torch.minimum(
            torch.full_like(speed, params.v2_max_drive_g * 9.81),
            power_force / max(params.mass, 1e-6),
        ),
        longitudinal_capacity,
    )
    brake_limit = torch.minimum(torch.full_like(speed, params.v2_max_brake_g * 9.81), longitudinal_capacity)
    brake_limit = torch.where(wheel_lock, brake_limit * 0.82, brake_limit)
    longitudinal_accel = throttle * drive_limit - brake * brake_limit
    coasting = (throttle <= 1e-6) & (brake <= 1e-6)
    longitudinal_accel = longitudinal_accel - torch.where(
        coasting,
        torch.full_like(longitudinal_accel, params.v2_rolling_resistance_mps2),
        torch.zeros_like(longitudinal_accel),
    )
    longitudinal_accel = longitudinal_accel - params.v2_drag_coefficient * speed * speed
    tire_saturation = torch.clamp(torch.maximum(torch.abs(front_slip), torch.abs(rear_slip)) / max(slip_peak, 1e-6), min=0.0, max=3.0)
    longitudinal_accel = longitudinal_accel - params.v2_tire_scrub_drag * torch.clamp(tire_saturation - 1.0, min=0.0)
    speed = torch.clamp(speed + longitudinal_accel * dt, min=0.0, max=params.v2_max_speed_mps)

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

    if params.physics_model == "v2":
        return _apply_physics_v2_batch(
            state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=params,
            meters_per_pixel=meters_per_pixel,
        )

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
