# ruff: noqa: UP018
# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""Warp fused kernels for the future high-throughput GPU rollout backend."""

from __future__ import annotations

import math
from dataclasses import fields
from functools import lru_cache
from typing import Any

import torch

from f1rl.config import MONZA_LENGTH_METERS, SimConfig
from f1rl.gpu_fast_warp import require_warp
from f1rl.gpu_features import braking_gate_tensor
from f1rl.gpu_scoring import TERMINATION_REASON_TO_ID, GpuScoreAccumulator, GpuScoringDiagnostics
from f1rl.gpu_types import GpuCarBatch, GpuCarParams, GpuMovementBatch, GpuTrackTensors

_WARP_SCORE_PROFILE_IDS = {
    "max_progress": 0,
    "farthest_distance": 1,
    "frontier_fast": 2,
    "early_pace": 3,
    "clean_exit": 4,
    "fast_valid_lap": 5,
    "time_attack": 6,
    "lap_pace": 7,
    "fast_frontier": 8,
    "frontier": 9,
    "frontier_recovery": 10,
    "frontier_novelty": 11,
    "risk_seeking": 12,
    "clean_distance": 13,
    "brake_zone": 14,
    "apex": 15,
    "exit_speed": 16,
    "full_lap_validity": 17,
}

_WARP_CONTROLLER_FEATURE_IDS = {
    "bias": 0,
    "speed_norm": 1,
    "target_speed_norm": 2,
    "speed_error_norm": 3,
    "brake_demand": 4,
    "future_brake_demand": 5,
    "target_speed_drop_norm": 6,
    "brake_gate_proximity": 7,
    "brake_gate_distance_norm": 8,
    "lookahead_abs_max": 9,
    "signed_lateral_error_norm": 10,
    "heading_error_norm": 11,
    "yaw_rate_norm": 12,
    "curvature_norm": 13,
    "target_steer": 14,
    "last_throttle": 15,
    "last_brake": 16,
    "last_steer": 17,
    "segment_progress_ratio": 18,
    "lookahead_0": 19,
    "lookahead_1": 20,
    "lookahead_2": 21,
    "lookahead_3": 22,
}
_CONTROLLER_DOMINANCE_EPSILON = 1.0e-3


def controller_feature_ids_tensor(
    feature_names: tuple[str, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return fused-controller feature ids; unknown names intentionally map to zero features."""

    return torch.tensor(
        [_WARP_CONTROLLER_FEATURE_IDS.get(name, -1) for name in feature_names],
        device=device,
        dtype=torch.int64,
    )


def _warp_device_for_torch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError("Warp fused physics currently requires a CUDA torch device.")
    if device.index is None:
        return "cuda"
    return f"cuda:{device.index}"


@lru_cache(maxsize=1)
def _physics_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _apply_physics_kernel(
        x_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steering_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        elapsed_steps_in: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        throttle_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        x_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        yaw_rate_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steering_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        elapsed_steps_out: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        movement_x0: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_y0: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_x1: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_y1: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        wheelbase_m: float,
        max_steer_rad: float,
        steer_response: float,
        engine_accel_mps2: float,
        brake_accel_mps2: float,
        drag_coefficient: float,
        rolling_resistance_mps2: float,
        grip_g: float,
        aero_grip_per_mps2: float,
        max_grip_g: float,
        max_drive_g: float,
        max_brake_g: float,
        steering_speed_sensitivity: float,
        max_speed_mps: float,
        dt: float,
        meters_per_pixel: float,
    ) -> None:
        tid = wp.tid()
        throttle = wp.clamp(throttle_in[tid], 0.0, 1.0)
        brake = wp.clamp(brake_in[tid], 0.0, 1.0)
        steer = wp.clamp(steer_in[tid], -1.0, 1.0)
        wheelbase = wp.float32(wheelbase_m)
        max_steer = wp.float32(max_steer_rad)
        steer_rate = wp.float32(steer_response)
        engine_accel = wp.float32(engine_accel_mps2)
        brake_accel = wp.float32(brake_accel_mps2)
        drag = wp.float32(drag_coefficient)
        rolling = wp.float32(rolling_resistance_mps2)
        base_grip = wp.float32(grip_g)
        aero_grip = wp.float32(aero_grip_per_mps2)
        max_grip = wp.float32(max_grip_g)
        max_drive = wp.float32(max_drive_g)
        max_brake = wp.float32(max_brake_g)
        steer_speed_sensitivity = wp.float32(steering_speed_sensitivity)
        max_speed = wp.float32(max_speed_mps)
        step_dt = wp.float32(dt)
        meters_px = wp.float32(meters_per_pixel)
        gravity = wp.float32(9.81)

        speed = wp.max(speed_in[tid], 0.0)
        target_steering = steer * max_steer
        steering_delta = target_steering - steering_in[tid]
        max_delta = steer_rate * step_dt
        steering = steering_in[tid] + wp.clamp(steering_delta, -max_delta, max_delta)
        effective_steering = steering / (1.0 + steer_speed_sensitivity * speed * speed)

        grip = wp.clamp(base_grip + aero_grip * speed * speed, base_grip, max_grip)
        max_total_accel = grip * gravity

        requested_lateral = wp.float32(0.0)
        if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
            requested_yaw = speed / wp.max(wheelbase, 1.0e-6) * wp.tan(effective_steering)
            requested_lateral = wp.abs(speed * requested_yaw)
        lateral_accel = wp.min(requested_lateral, max_total_accel)
        longitudinal_capacity = wp.sqrt(wp.max(max_total_accel * max_total_accel - lateral_accel * lateral_accel, 0.0))
        if longitudinal_capacity < wp.float32(1.0e-2):
            longitudinal_capacity = wp.float32(0.0)

        drive_static_limit = wp.min(engine_accel, max_drive * gravity)
        brake_static_limit = wp.min(brake_accel, max_brake * gravity)
        drive_limit = wp.min(drive_static_limit, longitudinal_capacity)
        brake_limit = wp.min(brake_static_limit, longitudinal_capacity)

        longitudinal_accel = throttle * drive_limit - brake * brake_limit
        if throttle <= 1.0e-6 and brake <= 1.0e-6:
            longitudinal_accel = longitudinal_accel - rolling
        longitudinal_accel = longitudinal_accel - drag * speed * speed
        speed = wp.clamp(speed + longitudinal_accel * step_dt, 0.0, max_speed)

        yaw_rate = wp.float32(0.0)
        if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
            yaw_rate = speed / wp.max(wheelbase, 1.0e-6) * wp.tan(effective_steering)
            lateral_accel = wp.abs(speed * yaw_rate)
            grip = wp.clamp(base_grip + aero_grip * speed * speed, base_grip, max_grip)
            max_lateral = grip * gravity
            if lateral_accel > max_lateral:
                yaw_rate = yaw_rate * max_lateral / wp.max(lateral_accel, 1.0e-6)

        pi = 3.141592653589793
        two_pi = 6.283185307179586
        heading_raw = heading_in[tid] + yaw_rate * step_dt + pi
        heading = heading_raw - two_pi * wp.floor(heading_raw / two_pi) - pi
        distance_px = speed * step_dt / wp.max(meters_px, 1.0e-6)
        dx = wp.cos(heading) * distance_px
        dy = -wp.sin(heading) * distance_px
        x_new = x_in[tid] + dx
        y_new = y_in[tid] + dy

        x_out[tid] = x_new
        y_out[tid] = y_new
        heading_out[tid] = heading
        speed_out[tid] = speed
        yaw_rate_out[tid] = yaw_rate
        steering_out[tid] = steering
        elapsed_steps_out[tid] = elapsed_steps_in[tid] + wp.int64(1)
        movement_x0[tid] = x_in[tid]
        movement_y0[tid] = y_in[tid]
        movement_x1[tid] = x_new
        movement_y1[tid] = y_new

    return _apply_physics_kernel


@lru_cache(maxsize=1)
def _physics_v2_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _apply_physics_v2_kernel(
        x_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        yaw_rate_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steering_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        elapsed_steps_in: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        throttle_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        x_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        yaw_rate_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steering_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        elapsed_steps_out: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        movement_x0: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_y0: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_x1: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        movement_y1: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        mass: float,
        wheelbase_m: float,
        max_steer_rad: float,
        steer_response: float,
        steering_speed_sensitivity: float,
        dt: float,
        meters_per_pixel: float,
        front_weight_distribution: float,
        cg_height_m: float,
        track_width_m: float,
        front_axle_distance_m: float,
        rear_axle_distance_m: float,
        front_cornering_stiffness_n_per_rad: float,
        rear_cornering_stiffness_n_per_rad: float,
        front_peak_mu: float,
        rear_peak_mu: float,
        mechanical_grip_low_speed_scale: float,
        mechanical_grip_high_speed_scale: float,
        mechanical_grip_transition_mps: float,
        tire_shape_c: float,
        slip_angle_peak_rad: float,
        rear_slip_steer_coupling: float,
        post_peak_falloff: float,
        load_sensitivity: float,
        aero_downforce_n_per_mps2: float,
        aero_balance_front: float,
        engine_power_w: float,
        drivetrain_efficiency: float,
        power_min_speed_mps: float,
        max_drive_g: float,
        max_brake_g: float,
        brake_lock_threshold: float,
        brake_lock_min_speed_mps: float,
        brake_lock_steer_loss: float,
        drag_coefficient: float,
        rolling_resistance_mps2: float,
        max_speed_mps: float,
        gear_ratio_1: float,
        gear_ratio_2: float,
        gear_ratio_3: float,
        gear_ratio_4: float,
        gear_ratio_5: float,
        gear_ratio_6: float,
        gear_ratio_7: float,
        gear_ratio_8: float,
        final_drive_ratio: float,
        wheel_radius_m: float,
        idle_rpm: float,
        shift_up_rpm: float,
        max_rpm: float,
        torque_peak_rpm: float,
        torque_low_rpm_factor: float,
        torque_high_rpm_factor: float,
        tire_scrub_drag: float,
        surface_mu: float,
    ) -> None:
        tid = wp.tid()
        throttle = wp.clamp(throttle_in[tid], 0.0, 1.0)
        brake = wp.clamp(brake_in[tid], 0.0, 1.0)
        steer = wp.clamp(steer_in[tid], -1.0, 1.0)
        vehicle_mass = wp.float32(mass)
        wheelbase = wp.float32(wheelbase_m)
        max_steer = wp.float32(max_steer_rad)
        steer_rate = wp.float32(steer_response)
        steer_speed_sensitivity = wp.float32(steering_speed_sensitivity)
        step_dt = wp.float32(dt)
        meters_px = wp.float32(meters_per_pixel)
        gravity = wp.float32(9.81)

        speed = wp.max(speed_in[tid], 0.0)
        target_steering = steer * max_steer
        steering_delta = target_steering - steering_in[tid]
        max_delta = steer_rate * step_dt
        steering = steering_in[tid] + wp.clamp(steering_delta, -max_delta, max_delta)

        wheel_lock = brake >= brake_lock_threshold and speed >= brake_lock_min_speed_mps
        effective_steering = steering / (1.0 + steer_speed_sensitivity * speed * speed)
        if wheel_lock:
            effective_steering = effective_steering * wp.clamp(1.0 - brake_lock_steer_loss * brake, 0.0, 1.0)

        previous_lateral_accel = wp.abs(speed * yaw_rate_in[tid])
        static_front = vehicle_mass * gravity * front_weight_distribution
        static_rear = vehicle_mass * gravity - static_front
        aero_total = aero_downforce_n_per_mps2 * speed * speed
        aero_front = aero_total * aero_balance_front
        aero_rear = aero_total - aero_front
        lateral_unload = wp.abs(vehicle_mass * previous_lateral_accel * cg_height_m / wp.max(track_width_m, 1.0e-6)) * 0.12
        front_load = wp.max(static_front + aero_front - lateral_unload * front_weight_distribution, 1.0)
        rear_load = wp.max(static_rear + aero_rear - lateral_unload * (1.0 - front_weight_distribution), 1.0)
        reference_front = vehicle_mass * gravity * front_weight_distribution
        reference_rear = vehicle_mass * gravity - reference_front

        front_slip = wp.atan2(yaw_rate_in[tid] * front_axle_distance_m, wp.max(wp.abs(speed), 1.0e-3)) - effective_steering
        rear_slip = (
            -wp.atan2(yaw_rate_in[tid] * rear_axle_distance_m, wp.max(wp.abs(speed), 1.0e-3))
            - rear_slip_steer_coupling * effective_steering
        )
        grip_t = wp.clamp(speed / wp.max(mechanical_grip_transition_mps, 1.0e-6), 0.0, 1.0)
        grip_smooth_t = grip_t * grip_t * (3.0 - 2.0 * grip_t)
        mechanical_grip_scale = mechanical_grip_low_speed_scale + (
            mechanical_grip_high_speed_scale - mechanical_grip_low_speed_scale
        ) * grip_smooth_t

        front_load_ratio = front_load / wp.max(reference_front, 1.0)
        front_mu = front_peak_mu * mechanical_grip_scale * surface_mu * (
            1.0 - load_sensitivity * wp.max(front_load_ratio - 1.0, 0.0)
        )
        front_peak_force = wp.max(front_mu * front_load, 1.0)
        front_b = front_cornering_stiffness_n_per_rad / wp.max(tire_shape_c * front_peak_force, 1.0e-6)
        front_force = front_peak_force * wp.sin(tire_shape_c * wp.atan(front_b * front_slip))
        front_excess = wp.clamp((wp.abs(front_slip) - slip_angle_peak_rad) / wp.max(slip_angle_peak_rad, 1.0e-6), 0.0, 1.0)
        front_force = front_force * (1.0 - post_peak_falloff * front_excess)

        rear_load_ratio = rear_load / wp.max(reference_rear, 1.0)
        rear_mu = rear_peak_mu * mechanical_grip_scale * surface_mu * (
            1.0 - load_sensitivity * wp.max(rear_load_ratio - 1.0, 0.0)
        )
        rear_peak_force = wp.max(rear_mu * rear_load, 1.0)
        rear_b = rear_cornering_stiffness_n_per_rad / wp.max(tire_shape_c * rear_peak_force, 1.0e-6)
        rear_force = rear_peak_force * wp.sin(tire_shape_c * wp.atan(rear_b * rear_slip))
        rear_excess = wp.clamp((wp.abs(rear_slip) - slip_angle_peak_rad) / wp.max(slip_angle_peak_rad, 1.0e-6), 0.0, 1.0)
        rear_force = rear_force * (1.0 - post_peak_falloff * rear_excess)

        lateral_capacity = (wp.abs(front_force) + wp.abs(rear_force)) / wp.max(vehicle_mass, 1.0e-6)
        requested_yaw = speed / wp.max(wheelbase, 1.0e-6) * wp.tan(effective_steering)
        requested_lateral = wp.abs(speed * requested_yaw)
        lateral_accel = wp.min(requested_lateral, lateral_capacity)
        yaw_rate = wp.float32(0.0)
        if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
            yaw_rate = wp.sign(requested_yaw) * lateral_accel / wp.max(speed, 1.0e-6)
        else:
            lateral_accel = wp.float32(0.0)

        total_peak_force = (
            front_peak_mu * mechanical_grip_scale * front_load * surface_mu
            + rear_peak_mu * mechanical_grip_scale * rear_load * surface_mu
        )
        total_accel_limit = total_peak_force / wp.max(vehicle_mass, 1.0e-6)
        longitudinal_capacity = wp.sqrt(wp.max(total_accel_limit * total_accel_limit - lateral_accel * lateral_accel, 0.0))

        wheel_rps = wp.max(speed, 0.0) / wp.max(2.0 * 3.141592653589793 * wheel_radius_m, 1.0e-6)
        rpm = wp.clamp(wheel_rps * gear_ratio_1 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
        if rpm > shift_up_rpm:
            rpm = wp.clamp(wheel_rps * gear_ratio_2 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
            if rpm > shift_up_rpm:
                rpm = wp.clamp(wheel_rps * gear_ratio_3 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
                if rpm > shift_up_rpm:
                    rpm = wp.clamp(wheel_rps * gear_ratio_4 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
                    if rpm > shift_up_rpm:
                        rpm = wp.clamp(wheel_rps * gear_ratio_5 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
                        if rpm > shift_up_rpm:
                            rpm = wp.clamp(wheel_rps * gear_ratio_6 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
                            if rpm > shift_up_rpm:
                                rpm = wp.clamp(wheel_rps * gear_ratio_7 * final_drive_ratio * 60.0, idle_rpm, max_rpm)
                                if rpm > shift_up_rpm:
                                    rpm = wp.clamp(wheel_rps * gear_ratio_8 * final_drive_ratio * 60.0, idle_rpm, max_rpm)

        torque_factor = wp.float32(1.0)
        if rpm <= torque_peak_rpm:
            low_span = wp.max(torque_peak_rpm - idle_rpm, 1.0)
            low_ratio = wp.clamp((rpm - idle_rpm) / low_span, 0.0, 1.0)
            torque_factor = torque_low_rpm_factor + (1.0 - torque_low_rpm_factor) * low_ratio
        else:
            high_span = wp.max(max_rpm - torque_peak_rpm, 1.0)
            high_ratio = wp.clamp((rpm - torque_peak_rpm) / high_span, 0.0, 1.0)
            torque_factor = 1.0 - (1.0 - torque_high_rpm_factor) * high_ratio

        power_force = engine_power_w * drivetrain_efficiency * torque_factor / wp.max(speed, power_min_speed_mps)
        drive_limit = wp.min(wp.min(max_drive_g * gravity, power_force / wp.max(vehicle_mass, 1.0e-6)), longitudinal_capacity)
        brake_limit = wp.min(max_brake_g * gravity, longitudinal_capacity)
        if wheel_lock:
            brake_limit = brake_limit * 0.82

        longitudinal_accel = throttle * drive_limit - brake * brake_limit
        if throttle <= 1.0e-6 and brake <= 1.0e-6:
            longitudinal_accel = longitudinal_accel - rolling_resistance_mps2
        longitudinal_accel = longitudinal_accel - drag_coefficient * speed * speed
        tire_saturation = wp.clamp(wp.max(wp.abs(front_slip), wp.abs(rear_slip)) / wp.max(slip_angle_peak_rad, 1.0e-6), 0.0, 3.0)
        longitudinal_accel = longitudinal_accel - tire_scrub_drag * wp.max(tire_saturation - 1.0, 0.0)
        speed = wp.clamp(speed + longitudinal_accel * step_dt, 0.0, max_speed_mps)

        pi = 3.141592653589793
        two_pi = 6.283185307179586
        heading_raw = heading_in[tid] + yaw_rate * step_dt + pi
        heading = heading_raw - two_pi * wp.floor(heading_raw / two_pi) - pi
        distance_px = speed * step_dt / wp.max(meters_px, 1.0e-6)
        dx = wp.cos(heading) * distance_px
        dy = -wp.sin(heading) * distance_px
        x_new = x_in[tid] + dx
        y_new = y_in[tid] + dy

        x_out[tid] = x_new
        y_out[tid] = y_new
        heading_out[tid] = heading
        speed_out[tid] = speed
        yaw_rate_out[tid] = yaw_rate
        steering_out[tid] = steering
        elapsed_steps_out[tid] = elapsed_steps_in[tid] + wp.int64(1)
        movement_x0[tid] = x_in[tid]
        movement_y0[tid] = y_in[tid]
        movement_x1[tid] = x_new
        movement_y1[tid] = y_new

    return _apply_physics_v2_kernel


@lru_cache(maxsize=1)
def _local_projection_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _project_local_kernel(
        points_xy: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        previous_progress_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        previous_segment_idx: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        centerline_xy: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_vec: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_len: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_len2: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_cumdist_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_mid_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        local_projection_indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
        raw_progress_px_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        lateral_error_m_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        signed_lateral_error_m_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_error_rad_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        segment_idx_out: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        projection_x_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        projection_y_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        local_count: int,
        centerline_segment_count: int,
        length_px: float,
        meters_per_pixel: float,
        window_px: float,
    ) -> None:
        row = wp.tid()
        pi = 3.141592653589793
        two_pi = 6.283185307179586
        px = points_xy[row, 0]
        py = points_xy[row, 1]
        prev = previous_progress_px[row]
        prev = prev - length_px * wp.floor(prev / length_px)
        base_idx = previous_segment_idx[row]
        if base_idx < wp.int64(0):
            base_idx = wp.int64(0)
        max_segment_idx = wp.int64(centerline_segment_count - 1)
        if base_idx > max_segment_idx:
            base_idx = max_segment_idx

        has_local = bool(False)
        for candidate_offset in range(local_count):
            segment_idx = local_projection_indices[base_idx, candidate_offset]
            if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                continue
            len2 = centerline_len2[segment_idx]
            if len2 <= 1.0e-9:
                continue
            mid = centerline_mid_px[segment_idx]
            wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
            if wrapped_delta <= window_px:
                has_local = True

        best_score = float(3.4028234663852886e38)
        best_segment_idx = wp.int64(0)
        best_raw_px = float(0.0)
        best_lateral_px = float(0.0)
        best_projection_x = float(0.0)
        best_projection_y = float(0.0)
        best_tangent = float(0.0)

        for candidate_offset in range(local_count):
            segment_idx = local_projection_indices[base_idx, candidate_offset]
            if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                continue
            len2 = centerline_len2[segment_idx]
            if len2 <= 1.0e-9:
                continue
            mid = centerline_mid_px[segment_idx]
            wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
            if has_local and wrapped_delta > window_px:
                continue

            sx = centerline_xy[segment_idx, 0]
            sy = centerline_xy[segment_idx, 1]
            vx = centerline_vec[segment_idx, 0]
            vy = centerline_vec[segment_idx, 1]
            rel_x = px - sx
            rel_y = py - sy
            t = wp.clamp((rel_x * vx + rel_y * vy) / len2, 0.0, 1.0)
            projection_x = sx + t * vx
            projection_y = sy + t * vy
            delta_x = px - projection_x
            delta_y = py - projection_y
            lateral_px = wp.sqrt(delta_x * delta_x + delta_y * delta_y)
            raw_progress_px = centerline_cumdist_px[segment_idx] + centerline_len[segment_idx] * t
            signed_delta = (raw_progress_px - prev + length_px * 0.5) - length_px * wp.floor((raw_progress_px - prev + length_px * 0.5) / length_px) - length_px * 0.5
            score = lateral_px + wp.max(-signed_delta, 0.0) * 0.05
            if score < best_score:
                best_score = score
                best_segment_idx = segment_idx
                best_raw_px = raw_progress_px - length_px * wp.floor(raw_progress_px / length_px)
                best_lateral_px = lateral_px
                best_projection_x = projection_x
                best_projection_y = projection_y
                best_tangent = wp.atan2(-vy, vx)

        offset_x = px - best_projection_x
        offset_y = py - best_projection_y
        tangent_x = wp.cos(best_tangent)
        tangent_y = -wp.sin(best_tangent)
        cross = tangent_x * offset_y - tangent_y * offset_x
        signed_lateral_px = float(0.0)
        if best_lateral_px > 1.0e-6 and wp.abs(cross) > 1.0e-9:
            if cross > 0.0:
                signed_lateral_px = best_lateral_px
            else:
                signed_lateral_px = -best_lateral_px
        heading_error = best_tangent - heading_rad[row] + pi
        heading_error = heading_error - two_pi * wp.floor(heading_error / two_pi) - pi

        raw_progress_px_out[row] = best_raw_px
        lateral_error_m_out[row] = best_lateral_px * meters_per_pixel
        signed_lateral_error_m_out[row] = signed_lateral_px * meters_per_pixel
        heading_error_rad_out[row] = heading_error
        segment_idx_out[row] = best_segment_idx
        projection_x_out[row] = best_projection_x
        projection_y_out[row] = best_projection_y

    return _project_local_kernel


@lru_cache(maxsize=1)
def _drivable_mask_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _point_is_drivable_kernel(
        x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        drivable_mask: wp.array2d(dtype=wp.bool),  # type: ignore[valid-type]
        output: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        width: int,
        height: int,
    ) -> None:
        tid = wp.tid()
        xi = wp.int32(wp.round(x[tid]))
        yi = wp.int32(wp.round(y[tid]))
        if xi >= 0 and yi >= 0 and xi < width and yi < height:
            output[tid] = drivable_mask[yi, xi]
        else:
            output[tid] = False

    return _point_is_drivable_kernel


@lru_cache(maxsize=1)
def _grid_collision_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _segments_intersect_grid_kernel(
        movements: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        boundary_segments: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        boundary_grid_indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
        boundary_grid_counts: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        output: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        grid_width: int,
        grid_height: int,
        max_segments_per_cell: int,
        query_span: int,
        cell_size_px: float,
    ) -> None:
        tid = wp.tid()
        x1 = movements[tid, 0]
        y1 = movements[tid, 1]
        x2 = movements[tid, 2]
        y2 = movements[tid, 3]
        min_x = wp.min(x1, x2)
        min_y = wp.min(y1, y2)
        max_x = wp.max(x1, x2)
        max_y = wp.max(y1, y2)
        cell_size = wp.max(cell_size_px, 1.0)
        min_cell_x = wp.int32(wp.floor(min_x / cell_size))
        min_cell_y = wp.int32(wp.floor(min_y / cell_size))
        max_cell_x = wp.int32(wp.floor(max_x / cell_size))
        max_cell_y = wp.int32(wp.floor(max_y / cell_size))
        if min_cell_x < 0:
            min_cell_x = 0
        if min_cell_y < 0:
            min_cell_y = 0
        if max_cell_x < 0:
            max_cell_x = 0
        if max_cell_y < 0:
            max_cell_y = 0
        if min_cell_x >= grid_width:
            min_cell_x = grid_width - 1
        if max_cell_x >= grid_width:
            max_cell_x = grid_width - 1
        if min_cell_y >= grid_height:
            min_cell_y = grid_height - 1
        if max_cell_y >= grid_height:
            max_cell_y = grid_height - 1

        dx12 = x2 - x1
        dy12 = y2 - y1
        hit = bool(False)
        span = query_span
        if span < 1:
            span = 1
        for offset_y in range(span):
            cell_y = min_cell_y + offset_y
            valid_y = cell_y <= max_cell_y
            if cell_y < 0:
                cell_y = 0
            if cell_y >= grid_height:
                cell_y = grid_height - 1
            for offset_x in range(span):
                cell_x = min_cell_x + offset_x
                valid_cell = valid_y and cell_x <= max_cell_x
                if cell_x < 0:
                    cell_x = 0
                if cell_x >= grid_width:
                    cell_x = grid_width - 1
                cell_id = cell_y * grid_width + cell_x
                count = boundary_grid_counts[cell_id]
                for candidate_offset in range(max_segments_per_cell):
                    if not valid_cell or wp.int64(candidate_offset) >= count:
                        continue
                    segment_id = boundary_grid_indices[cell_id, candidate_offset]
                    if segment_id < wp.int64(0):
                        continue
                    x3 = boundary_segments[segment_id, 0]
                    y3 = boundary_segments[segment_id, 1]
                    x4 = boundary_segments[segment_id, 2]
                    y4 = boundary_segments[segment_id, 3]
                    dx34 = x4 - x3
                    dy34 = y4 - y3
                    denom = dy34 * dx12 - dx34 * dy12
                    if wp.abs(denom) >= 1.0e-9:
                        s = (dx34 * (y1 - y3) - dy34 * (x1 - x3)) / denom
                        t = (dx12 * (y1 - y3) - dy12 * (x1 - x3)) / denom
                        if s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0:
                            hit = True
        output[tid] = hit

    return _segments_intersect_grid_kernel


@lru_cache(maxsize=1)
def _controller_controls_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _controller_kernel(
        controller_weights: wp.array3d(dtype=wp.float32),  # type: ignore[valid-type]
        features: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        throttle_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        feature_count: int,
    ) -> None:
        row = wp.tid()
        throttle_logit = float(0.0)
        brake_logit = float(0.0)
        steer_logit = float(0.0)
        for feature_index in range(feature_count):
            value = features[row, feature_index]
            throttle_logit = throttle_logit + controller_weights[row, 0, feature_index] * value
            brake_logit = brake_logit + controller_weights[row, 1, feature_index] * value
            steer_logit = steer_logit + controller_weights[row, 2, feature_index] * value

        throttle_clamped = wp.clamp(throttle_logit, -40.0, 40.0)
        brake_clamped = wp.clamp(brake_logit, -40.0, 40.0)
        throttle_raw = 1.0 / (1.0 + wp.exp(-throttle_clamped))
        brake_raw = 1.0 / (1.0 + wp.exp(-brake_clamped))
        throttle = throttle_raw
        brake = brake_raw * (1.0 - throttle_raw)
        if brake_raw + wp.float32(1.0e-3) >= throttle_raw:
            throttle = throttle_raw * (1.0 - brake_raw)
            brake = brake_raw
        throttle_out[row] = wp.clamp(throttle, 0.0, 1.0)
        brake_out[row] = wp.clamp(brake, 0.0, 1.0)
        steer_out[row] = wp.clamp(wp.tanh(steer_logit), -1.0, 1.0)

    return _controller_kernel


@lru_cache(maxsize=1)
def _phase_controls_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _phase_kernel(
        action_controls: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        phase_action_ids: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
        phase_thresholds: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        progress_delta_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        throttle_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        action_id_out: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        phase_count: int,
        use_progress: bool,
    ) -> None:
        row = wp.tid()
        metric = wp.float32(elapsed_steps[row])
        if use_progress:
            metric = progress_delta_m[row]

        phase_index = int(0)
        for index in range(phase_count):
            if metric >= phase_thresholds[row, index]:
                phase_index = phase_index + 1
        if phase_index >= phase_count:
            phase_index = phase_count - 1

        action_id = phase_action_ids[row, phase_index]
        action_id_out[row] = action_id
        throttle_out[row] = action_controls[action_id, 0]
        brake_out[row] = action_controls[action_id, 1]
        steer_out[row] = action_controls[action_id, 2]

    return _phase_kernel


@lru_cache(maxsize=1)
def _open_step_bookkeeping_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _bookkeeping_kernel(
        active: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        old_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        old_raw_progress_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        old_lap_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        raw_progress_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        segment_idx: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_error_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        collided_input: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        drivable_input: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        physical_finish_input: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        segment_target_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        last_segment_idx: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_monotonic_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_raw_progress_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_checkpoint_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_next_checkpoint_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_checkpoints_passed: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_lap_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_no_progress_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_alive: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_terminated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_truncated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_segment_release_observed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_last_throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_steer: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        progress_delta_m_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        collided_out: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        off_track_out: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        telemetry_valid_lap_out: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        length_px: float,
        meters_per_pixel: float,
        local_projection_window_m: float,
        checkpoint_spacing_m: float,
        checkpoint_lateral_limit_m: float,
        track_length_m: float,
        checkpoint_count: int,
        no_progress_limit_steps: int,
        max_steps: int,
        terminate_at_target_progress: bool,
        has_target_min_speed_kph: bool,
        target_min_speed_kph: float,
        has_target_max_speed_kph: bool,
        target_max_speed_kph: float,
        has_target_max_lateral_error_m: bool,
        target_max_lateral_error_m: float,
        has_target_max_heading_error_deg: bool,
        target_max_heading_error_deg: float,
        has_target_max_abs_yaw_rate_rps: bool,
        target_max_abs_yaw_rate_rps: float,
        has_target_max_abs_steering: bool,
        target_max_abs_steering: float,
        segment_fail_on_speed_gate_miss: bool,
        segment_require_release: bool,
        segment_release_min_speed_kph: float,
        segment_release_max_speed_kph: float,
        segment_release_max_brake: float,
        segment_release_max_throttle: float,
        segment_complete_id: int,
        segment_release_gate_failed_id: int,
        segment_min_speed_gate_failed_id: int,
        segment_speed_gate_failed_id: int,
        segment_lateral_gate_failed_id: int,
        segment_heading_gate_failed_id: int,
        segment_yaw_rate_gate_failed_id: int,
        segment_steering_gate_failed_id: int,
        lap_complete_id: int,
        collision_id: int,
        off_track_id: int,
        no_progress_id: int,
        max_steps_id: int,
    ) -> None:
        row = wp.tid()
        is_active = active[row]
        raw_delta_px = raw_progress_px[row] - old_raw_progress_px[row]
        if raw_delta_px < -0.5 * length_px:
            raw_delta_px = raw_delta_px + length_px
        if raw_delta_px > 0.5 * length_px:
            raw_delta_px = raw_delta_px - length_px
        progress_delta_m = wp.max(raw_delta_px * meters_per_pixel, 0.0)
        if progress_delta_m > local_projection_window_m:
            progress_delta_m = 0.0
        if is_active:
            state_last_throttle[row] = throttle[row]
            state_last_brake[row] = brake[row]
            state_last_steer[row] = steer[row]
            last_segment_idx[row] = segment_idx[row]

            state_last_raw_progress_px[row] = raw_progress_px[row]
            state_raw_progress_m[row] = raw_progress_px[row] * meters_per_pixel
            state_monotonic_progress_m[row] = old_progress_m[row] + progress_delta_m

            skipped = progress_delta_m > checkpoint_spacing_m * 1.75
            if skipped:
                skipped_count = wp.int64(wp.floor(progress_delta_m / checkpoint_spacing_m)) - wp.int64(1)
                if skipped_count < wp.int64(1):
                    skipped_count = wp.int64(1)
                state_missed_checkpoint_count[row] = state_missed_checkpoint_count[row] + skipped_count
                state_valid_lap[row] = False

            next_idx = state_next_checkpoint_index[row]
            threshold = checkpoint_spacing_m * wp.float32(next_idx)
            due = (
                next_idx < wp.int64(checkpoint_count)
                and state_monotonic_progress_m[row] + 1.0e-6 >= threshold
            )
            expected = (
                old_progress_m[row] <= threshold
                and threshold <= state_monotonic_progress_m[row] + checkpoint_spacing_m * 0.75
            )
            lateral_ok = wp.abs(lateral_error_m[row]) <= checkpoint_lateral_limit_m
            if due:
                if not (expected and lateral_ok):
                    state_missed_checkpoint_count[row] = state_missed_checkpoint_count[row] + wp.int64(1)
                    state_valid_lap[row] = False
                state_checkpoints_passed[row] = next_idx
                state_next_checkpoint_index[row] = next_idx + wp.int64(1)

            if progress_delta_m <= 1.0e-4:
                state_no_progress_steps[row] = state_no_progress_steps[row] + wp.int64(1)
            else:
                state_no_progress_steps[row] = wp.int64(0)

            speed_kph_after = state_speed_mps[row] * 3.6
            heading_error_abs_deg = wp.abs(heading_error_rad[row]) * (180.0 / 3.141592653589793)
            if segment_require_release:
                release_min = segment_release_min_speed_kph
                release_max = segment_release_max_speed_kph
                if release_min > release_max:
                    swap = release_min
                    release_min = release_max
                    release_max = swap
                release_observed = (
                    speed_kph_after >= release_min
                    and speed_kph_after <= release_max
                    and brake[row] <= segment_release_max_brake
                    and throttle[row] <= segment_release_max_throttle
                )
                if release_observed:
                    state_segment_release_observed[row] = True

            if terminate_at_target_progress:
                not_already_terminated = is_active and not state_terminated[row]
                target_progress_m = segment_target_progress_m[row]
                target_crossed = old_progress_m[row] < target_progress_m and target_progress_m <= state_monotonic_progress_m[row]
                target_reached = state_monotonic_progress_m[row] >= target_progress_m
                speed_gate_ok = True
                if has_target_min_speed_kph:
                    speed_gate_ok = speed_gate_ok and speed_kph_after >= target_min_speed_kph
                if has_target_max_speed_kph:
                    speed_gate_ok = speed_gate_ok and speed_kph_after <= target_max_speed_kph
                lateral_gate_ok = True
                if has_target_max_lateral_error_m:
                    lateral_gate_ok = wp.abs(lateral_error_m[row]) <= target_max_lateral_error_m
                heading_gate_ok = True
                if has_target_max_heading_error_deg:
                    heading_gate_ok = heading_error_abs_deg <= target_max_heading_error_deg
                yaw_rate_gate_ok = True
                if has_target_max_abs_yaw_rate_rps:
                    yaw_rate_gate_ok = wp.abs(state_yaw_rate_rps[row]) <= target_max_abs_yaw_rate_rps
                steering_gate_ok = True
                if has_target_max_abs_steering:
                    steering_gate_ok = wp.abs(state_steering[row]) <= target_max_abs_steering
                release_gate_ok = True
                if segment_require_release:
                    release_gate_ok = state_segment_release_observed[row]
                complete = (
                    not_already_terminated
                    and target_reached
                    and speed_gate_ok
                    and lateral_gate_ok
                    and heading_gate_ok
                    and yaw_rate_gate_ok
                    and steering_gate_ok
                    and release_gate_ok
                )
                if complete:
                    state_segment_complete[row] = True
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_complete_id)
                release_failed = (
                    not_already_terminated
                    and target_crossed
                    and speed_gate_ok
                    and segment_require_release
                    and not state_segment_release_observed[row]
                    and not complete
                )
                if release_failed:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_release_gate_failed_id)
                gate_miss = (
                    not_already_terminated
                    and target_crossed
                    and segment_fail_on_speed_gate_miss
                    and not complete
                    and not release_failed
                )
                if gate_miss and has_target_min_speed_kph and speed_kph_after < target_min_speed_kph:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_min_speed_gate_failed_id)
                if gate_miss and has_target_max_speed_kph and speed_kph_after > target_max_speed_kph:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_speed_gate_failed_id)
                if gate_miss and has_target_max_lateral_error_m and not lateral_gate_ok:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_lateral_gate_failed_id)
                if gate_miss and has_target_max_heading_error_deg and not heading_gate_ok:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_heading_gate_failed_id)
                if gate_miss and has_target_max_abs_yaw_rate_rps and not yaw_rate_gate_ok:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_yaw_rate_gate_failed_id)
                if gate_miss and has_target_max_abs_steering and not steering_gate_ok:
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(segment_steering_gate_failed_id)

        checkpoint_index = wp.int64(wp.floor(state_monotonic_progress_m[row] / checkpoint_spacing_m))
        if checkpoint_count > 0:
            checkpoint_index = checkpoint_index % wp.int64(checkpoint_count)
        state_checkpoint_index[row] = checkpoint_index

        target_lap_progress_m = (wp.float32(old_lap_index[row]) + 1.0) * track_length_m
        near_finish = old_progress_m[row] >= target_lap_progress_m - checkpoint_spacing_m * 2.0
        virtual_finish = old_progress_m[row] < target_lap_progress_m and target_lap_progress_m <= state_monotonic_progress_m[row]
        crossed_finish = is_active and near_finish and (physical_finish_input[row] or virtual_finish)
        if crossed_finish:
            state_finish_crossed[row] = True

        last_checkpoint_index = checkpoint_count - 1
        if last_checkpoint_index < 0:
            last_checkpoint_index = 0
        telemetry_valid_lap = (
            state_valid_lap[row]
            and state_missed_checkpoint_count[row] == wp.int64(0)
            and state_checkpoints_passed[row] >= wp.int64(last_checkpoint_index)
        )
        lap_complete = (
            is_active
            and not terminate_at_target_progress
            and crossed_finish
            and telemetry_valid_lap
            and state_monotonic_progress_m[row] >= target_lap_progress_m
            and not state_terminated[row]
        )
        if lap_complete:
            state_lap_index[row] = state_lap_index[row] + wp.int64(1)
            state_completed_lap[row] = True
            state_truncated[row] = True
            state_termination_reason_id[row] = wp.int64(lap_complete_id)

        collided = is_active and collided_input[row]
        off_track = is_active and (not drivable_input[row])
        if collided:
            state_terminated[row] = True
            state_termination_reason_id[row] = wp.int64(collision_id)
        if off_track:
            state_terminated[row] = True
            state_termination_reason_id[row] = wp.int64(off_track_id)
        no_progress_terminated = is_active and state_no_progress_steps[row] >= wp.int64(no_progress_limit_steps)
        if no_progress_terminated:
            state_terminated[row] = True
            state_termination_reason_id[row] = wp.int64(no_progress_id)
        max_steps_reached = is_active and state_elapsed_steps[row] >= wp.int64(max_steps)
        if max_steps_reached:
            state_truncated[row] = True
            state_termination_reason_id[row] = wp.int64(max_steps_id)
        if state_terminated[row]:
            state_alive[row] = False

        progress_delta_m_out[row] = progress_delta_m
        collided_out[row] = collided
        off_track_out[row] = off_track
        telemetry_valid_lap_out[row] = telemetry_valid_lap

    return _bookkeeping_kernel


@lru_cache(maxsize=1)
def _commit_physics_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _commit_kernel(
        active: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        next_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        next_elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
    ) -> None:
        row = wp.tid()
        if active[row]:
            state_x[row] = next_x[row]
            state_y[row] = next_y[row]
            state_heading_rad[row] = next_heading_rad[row]
            state_speed_mps[row] = next_speed_mps[row]
            state_yaw_rate_rps[row] = next_yaw_rate_rps[row]
            state_steering[row] = next_steering[row]
            state_elapsed_steps[row] = next_elapsed_steps[row]

    return _commit_kernel


@lru_cache(maxsize=1)
def _segments_intersect_any_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _segments_any_kernel(
        movements: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        segments: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        output: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        segment_count: int,
    ) -> None:
        row = wp.tid()
        x1 = movements[row, 0]
        y1 = movements[row, 1]
        x2 = movements[row, 2]
        y2 = movements[row, 3]
        dx12 = x2 - x1
        dy12 = y2 - y1
        hit = bool(False)
        for segment_index in range(segment_count):
            x3 = segments[segment_index, 0]
            y3 = segments[segment_index, 1]
            x4 = segments[segment_index, 2]
            y4 = segments[segment_index, 3]
            dx34 = x4 - x3
            dy34 = y4 - y3
            denom = dy34 * dx12 - dx34 * dy12
            if wp.abs(denom) >= 1.0e-9:
                s = (dx34 * (y1 - y3) - dy34 * (x1 - x3)) / denom
                t = (dx12 * (y1 - y3) - dy12 * (x1 - x3)) / denom
                if s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0:
                    hit = True
        output[row] = hit

    return _segments_any_kernel


@lru_cache(maxsize=1)
def _score_accumulator_update_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _update_acc_kernel(
        active: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        collided_input: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        off_track_input: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        steer: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_monotonic_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_terminated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_truncated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        diag_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_heading_error_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_brake_gate_proximity: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_target_speed_drop_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_telemetry_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        diag_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_near_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_min_future_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_target_speed_drop_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_braking_gate_distance_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_brake_gate_distance_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        diag_lookahead_abs_max: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_start_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_start_speed_for_score_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_heading_error_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_heading_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_heading_error_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_curvature_rad_per_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_step_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_sim_time_s: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_collided: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_off_track: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_terminated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_truncated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_near_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_min_future_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_target_speed_drop_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_target_speed_drop_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_gate_proximity: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_braking_gate_distance_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_gate_distance_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_lookahead_abs_max: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_time_to_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_time_to_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_sum_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_count_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_sum_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_count_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_max_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_row_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_brake_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_max_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        dt: float,
    ) -> None:
        row = wp.tid()
        if not active[row]:
            return

        speed_kph = state_speed_mps[row] * 3.6
        lateral_error_m = diag_lateral_error_m[row]
        heading_error_deg = diag_heading_error_deg[row]
        progress_m = state_monotonic_progress_m[row]
        sim_time_s = wp.float32(state_elapsed_steps[row]) * dt
        progress_delta_m = wp.max(progress_m - acc_start_progress_m[row], 0.0)

        if progress_m >= acc_best_progress_m[row]:
            acc_best_progress_m[row] = progress_m
            acc_best_speed_kph[row] = speed_kph
            acc_best_lateral_error_m[row] = wp.abs(lateral_error_m)
            acc_best_heading_error_deg[row] = wp.abs(heading_error_deg)

        if progress_delta_m <= 300.0:
            acc_speed_sum_first_300_m[row] = acc_speed_sum_first_300_m[row] + speed_kph
            acc_speed_count_first_300_m[row] = acc_speed_count_first_300_m[row] + 1.0
            acc_brake_sum_first_300_m[row] = acc_brake_sum_first_300_m[row] + brake[row]
            acc_brake_count_first_300_m[row] = acc_brake_count_first_300_m[row] + 1.0
        if progress_delta_m <= 450.0:
            acc_speed_sum_first_450_m[row] = acc_speed_sum_first_450_m[row] + speed_kph
            acc_speed_count_first_450_m[row] = acc_speed_count_first_450_m[row] + 1.0
            acc_brake_sum_first_450_m[row] = acc_brake_sum_first_450_m[row] + brake[row]
            acc_brake_count_first_450_m[row] = acc_brake_count_first_450_m[row] + 1.0
        if acc_time_to_300_m[row] < 0.0 and progress_delta_m >= 300.0:
            acc_time_to_300_m[row] = sim_time_s
        if acc_time_to_450_m[row] < 0.0 and progress_delta_m >= 450.0:
            acc_time_to_450_m[row] = sim_time_s

        demand = (
            (diag_future_brake_demand[row] >= 0.22)
            or (diag_brake_gate_proximity[row] >= 0.70)
            or (diag_target_speed_drop_norm[row] >= 0.18)
        ) and speed_kph >= 135.0
        if demand:
            acc_demand_brake_sum[row] = acc_demand_brake_sum[row] + brake[row]
            acc_demand_throttle_sum[row] = acc_demand_throttle_sum[row] + throttle[row]
            acc_demand_count[row] = acc_demand_count[row] + 1.0
            if diag_future_brake_demand[row] > acc_demand_max_future_brake_demand[row]:
                acc_demand_max_future_brake_demand[row] = diag_future_brake_demand[row]
        if brake[row] > acc_max_brake[row]:
            acc_max_brake[row] = brake[row]
        acc_throttle_sum[row] = acc_throttle_sum[row] + throttle[row]
        acc_row_count[row] = acc_row_count[row] + 1.0
        if acc_row_count[row] <= 1.0:
            acc_start_speed_for_score_kph[row] = speed_kph

        acc_final_x[row] = state_x[row]
        acc_final_y[row] = state_y[row]
        acc_final_heading_deg[row] = state_heading_rad[row] * (180.0 / 3.141592653589793)
        acc_final_speed_mps[row] = state_speed_mps[row]
        acc_final_speed_kph[row] = speed_kph
        acc_final_raw_progress_m[row] = state_raw_progress_m[row]
        acc_final_progress_m[row] = progress_m
        acc_final_lateral_error_m[row] = lateral_error_m
        acc_final_heading_error_deg[row] = heading_error_deg
        acc_final_yaw_rate_rps[row] = state_yaw_rate_rps[row]
        acc_final_curvature_rad_per_m[row] = state_yaw_rate_rps[row] / wp.max(state_speed_mps[row], 1.0e-6)
        acc_final_steering[row] = steer[row]
        acc_final_throttle[row] = throttle[row]
        acc_final_brake[row] = brake[row]
        acc_final_step_index[row] = state_elapsed_steps[row]
        acc_final_sim_time_s[row] = sim_time_s
        acc_final_missed_checkpoint_count[row] = state_missed_checkpoint_count[row]
        acc_final_valid_lap[row] = diag_telemetry_valid_lap[row]
        acc_final_finish_crossed[row] = state_finish_crossed[row]
        acc_final_completed_lap[row] = state_completed_lap[row]
        acc_final_segment_complete[row] = state_segment_complete[row]
        acc_final_collided[row] = collided_input[row]
        acc_final_off_track[row] = off_track_input[row]
        acc_final_terminated[row] = state_terminated[row]
        acc_final_truncated[row] = state_truncated[row]
        acc_final_termination_reason_id[row] = state_termination_reason_id[row]
        acc_final_target_speed_kph[row] = diag_target_speed_kph[row]
        acc_final_near_target_speed_kph[row] = diag_near_target_speed_kph[row]
        acc_final_min_future_target_speed_kph[row] = diag_min_future_target_speed_kph[row]
        acc_final_target_speed_drop_kph[row] = diag_target_speed_drop_kph[row]
        acc_final_target_speed_drop_norm[row] = diag_target_speed_drop_norm[row]
        acc_final_brake_demand[row] = diag_brake_demand[row]
        acc_final_future_brake_demand[row] = diag_future_brake_demand[row]
        acc_final_brake_gate_proximity[row] = diag_brake_gate_proximity[row]
        acc_final_braking_gate_distance_m[row] = diag_braking_gate_distance_m[row]
        acc_final_brake_gate_distance_norm[row] = diag_brake_gate_distance_norm[row]
        acc_final_lookahead_abs_max[row] = diag_lookahead_abs_max[row]

    return _update_acc_kernel


@lru_cache(maxsize=1)
def _score_profile_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _score_kernel(
        start_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        best_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        best_speed_kph_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        best_lateral_error_m_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        best_heading_error_deg_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        start_speed_for_score_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_speed_kph_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_lateral_error_m_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_heading_error_deg_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_yaw_rate_rps_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_steering_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        final_collided: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_off_track: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        final_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        final_sim_time_s: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        row_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_sum_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_count_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        final_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        time_to_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        demand_brake_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        demand_throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        demand_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        demand_max_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        max_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        output: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        profile_id: int,
        target_progress_m: float,
        terminate_at_target_progress: bool,
        monza_length_m: float,
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        no_progress_id: int,
        max_steps_id: int,
        collision_id: int,
        off_track_id: int,
    ) -> None:
        row = wp.tid()
        start_progress = start_progress_m[row]
        best_progress = best_progress_m[row]
        raw_progress = best_progress - start_progress
        progress_to_target = raw_progress
        if terminate_at_target_progress:
            progress_to_target = wp.min(best_progress, target_progress_m) - start_progress
        remaining_m = wp.max(target_progress_m - best_progress, 0.0)
        final_speed_kph = final_speed_kph_in[row]
        final_lateral_error_m = wp.abs(final_lateral_error_m_in[row])
        final_heading_error_deg = wp.abs(final_heading_error_deg_in[row])
        final_yaw_rate_rps = wp.abs(final_yaw_rate_rps_in[row])
        final_steering = wp.abs(final_steering_in[row])
        missed_checkpoints = wp.float32(final_missed_checkpoint_count[row])
        clean = (not final_collided[row]) and (not final_off_track[row])
        milestone_complete = best_progress >= target_progress_m
        segment_complete = final_segment_complete[row] or milestone_complete
        valid_finish = final_completed_lap[row] or (final_valid_lap[row] and final_finish_crossed[row])
        target_span_m = wp.max(target_progress_m - start_progress, 1.0)
        progress_ratio = wp.clamp(wp.min(raw_progress, target_span_m) / target_span_m, 0.0, 1.0)
        frontier_m = wp.max(progress_to_target - target_span_m * 0.70, 0.0)
        near_target_m = wp.max(progress_to_target - target_span_m * 0.88, 0.0)
        beyond_target_m = wp.max(best_progress - target_progress_m, 0.0)
        elapsed_s = final_sim_time_s[row]
        if elapsed_s <= 0.0:
            elapsed_s = row_count[row] / 60.0
        pace_kph = raw_progress / wp.max(elapsed_s, 1.0e-6) * 3.6
        best_speed_kph = best_speed_kph_in[row]
        best_lateral_error_m = wp.abs(best_lateral_error_m_in[row])
        best_heading_error_deg = wp.abs(best_heading_error_deg_in[row])
        avg_speed_first_300_m = final_speed_kph
        if speed_count_first_300_m[row] > 0.0:
            avg_speed_first_300_m = speed_sum_first_300_m[row] / wp.max(speed_count_first_300_m[row], 1.0)
        avg_speed_first_450_m = final_speed_kph
        if speed_count_first_450_m[row] > 0.0:
            avg_speed_first_450_m = speed_sum_first_450_m[row] / wp.max(speed_count_first_450_m[row], 1.0)
        avg_brake_first_450_m = final_brake[row]
        if brake_count_first_450_m[row] > 0.0:
            avg_brake_first_450_m = brake_sum_first_450_m[row] / wp.max(brake_count_first_450_m[row], 1.0)
        early_slow_penalty = wp.max(125.0 - avg_speed_first_300_m, 0.0) * 34.0
        if raw_progress >= 450.0:
            early_slow_penalty = early_slow_penalty + wp.max(145.0 - avg_speed_first_450_m, 0.0) * 18.0
        early_brake_penalty = wp.max(avg_brake_first_450_m - 0.32, 0.0) * 1800.0
        late_progress_factor = wp.clamp((best_progress - 3000.0) / 2200.0, 0.0, 1.0)
        frontier_quality_factor = wp.clamp((best_progress - 1500.0) / 3500.0, 0.0, 1.0)
        lap_progress_ratio = wp.clamp(best_progress / monza_length_m, 0.0, 1.0)
        focus_start_m = wp.min(frontier_focus_start_m, frontier_focus_end_m)
        focus_end_m = wp.max(frontier_focus_start_m, frontier_focus_end_m)
        focus_span_m = wp.max(1.0, focus_end_m - focus_start_m)
        focus_progress_m = wp.clamp(best_progress - focus_start_m, 0.0, focus_span_m)
        focus_progress_ratio = focus_progress_m / focus_span_m
        reached_focus = best_progress >= focus_start_m
        cleared_focus = best_progress >= focus_end_m
        avg_brake_demand_zone = 0.0
        avg_throttle_demand_zone = 0.0
        if demand_count[row] > 0.0:
            avg_brake_demand_zone = demand_brake_sum[row] / wp.max(demand_count[row], 1.0)
            avg_throttle_demand_zone = demand_throttle_sum[row] / wp.max(demand_count[row], 1.0)
        max_future_brake_demand = demand_max_future_brake_demand[row]
        setup_penalty = frontier_quality_factor * (
            wp.max(final_lateral_error_m - 12.0, 0.0) * 620.0
            + wp.max(final_heading_error_deg - 22.0, 0.0) * 720.0
            + wp.max(final_yaw_rate_rps - 0.85, 0.0) * 2200.0
        )
        late_setup_penalty = late_progress_factor * (
            wp.max(final_lateral_error_m - 9.0, 0.0) * 900.0
            + wp.max(final_heading_error_deg - 16.0, 0.0) * 1050.0
            + wp.max(115.0 - final_speed_kph, 0.0) * 120.0
        )
        brake_demand_penalty = frontier_quality_factor * max_future_brake_demand * (
            wp.max(0.32 - avg_brake_demand_zone, 0.0) * 18000.0
            + avg_throttle_demand_zone * 7500.0
        )
        viability_penalty = setup_penalty + late_setup_penalty + brake_demand_penalty
        focus_lateral_penalty = wp.max(best_lateral_error_m - 9.0, 0.0) * 420.0
        focus_heading_penalty = wp.max(best_heading_error_deg - 18.0, 0.0) * 520.0
        focus_stop_penalty = 0.0
        if reached_focus:
            focus_stop_penalty = wp.max(90.0 - final_speed_kph, 0.0) * 170.0
        valid_elapsed_s = 3.4028234663852886e38
        if valid_finish and elapsed_s > 0.0:
            valid_elapsed_s = elapsed_s
        valid_lap_speed_bonus = 0.0
        if valid_finish:
            valid_lap_speed_bonus = (
                8500000.0
                + wp.max(220.0 - valid_elapsed_s, 0.0) * 58000.0
                + wp.max(170.0 - valid_elapsed_s, 0.0) * 72000.0
                + wp.max(130.0 - valid_elapsed_s, 0.0) * 120000.0
                + wp.max(100.0 - valid_elapsed_s, 0.0) * 180000.0
                + wp.max(85.0 - valid_elapsed_s, 0.0) * 260000.0
                + wp.min(pace_kph, 380.0) * 16000.0
                - wp.max(valid_elapsed_s - 100.0, 0.0) * 34000.0
                - wp.max(valid_elapsed_s - 130.0, 0.0) * 80000.0
                - wp.max(valid_elapsed_s - 170.0, 0.0) * 140000.0
            )

        score = progress_to_target * 34.0 - remaining_m * 36.0
        score = score + frontier_m * 65.0 + near_target_m * 120.0 + beyond_target_m * 160.0
        if segment_complete:
            score = score + 160000.0
        if valid_finish:
            if profile_id == 17:
                score = score + 300000.0
            else:
                score = score + 240000.0
        reason_id = final_termination_reason_id[row]
        reason_collisionish = reason_id == wp.int64(collision_id) or reason_id == wp.int64(off_track_id)
        collision_penalty = 24000.0
        if profile_id == 9:
            collision_penalty = 4500.0
        elif profile_id == 12:
            collision_penalty = 7000.0
        if reason_collisionish:
            score = score - collision_penalty
        if reason_id >= wp.int64(2) and reason_id <= wp.int64(8):
            score = score - 7500.0
        if reason_id == wp.int64(no_progress_id):
            score = score - 3500.0
        if reason_id == wp.int64(max_steps_id) and final_speed_kph < 20.0:
            score = score - 6000.0
        if clean:
            score = score + 3500.0 + progress_ratio * 3000.0

        time_to_450_valid = time_to_450_m[row] >= 0.0
        if profile_id == 5:
            score = score + raw_progress * 20.0 + best_progress * 16.0
            score = score + frontier_m * 70.0 + near_target_m * 160.0 + beyond_target_m * 250.0
            score = score + wp.min(pace_kph, 380.0) * 4600.0
            score = score + wp.min(final_speed_kph, 360.0) * 120.0
            if valid_finish:
                score = score + valid_lap_speed_bonus
                score = score - wp.max(valid_elapsed_s - 150.0, 0.0) * 95000.0
                score = score - wp.max(valid_elapsed_s - 180.0, 0.0) * 140000.0
            else:
                score = score - remaining_m * 70.0
                if reason_id == wp.int64(max_steps_id):
                    score = score - (80000.0 + wp.max(elapsed_s - 120.0, 0.0) * 800.0)
            if time_to_450_valid:
                score = score + wp.max(16.5 - time_to_450_m[row], 0.0) * 1200.0
            score = score - early_slow_penalty * 2.40
            score = score - early_brake_penalty * 1.60
            if valid_finish:
                score = score - viability_penalty * 0.35
            else:
                score = score - viability_penalty * 0.70
            score = score - missed_checkpoints * 12000.0
        elif profile_id == 6:
            score = score + raw_progress * 12.0 + best_progress * 10.0
            score = score + frontier_m * 55.0 + near_target_m * 130.0 + beyond_target_m * 190.0
            score = score + wp.min(pace_kph, 400.0) * 8800.0 * wp.max(0.20, progress_ratio)
            if valid_finish:
                score = score + 6500000.0
                score = score + wp.max(180.0 - valid_elapsed_s, 0.0) * 95000.0
                score = score + wp.max(120.0 - valid_elapsed_s, 0.0) * 160000.0
                score = score + wp.max(90.0 - valid_elapsed_s, 0.0) * 280000.0
                score = score - wp.max(valid_elapsed_s - 120.0, 0.0) * 80000.0
                score = score - wp.max(valid_elapsed_s - 150.0, 0.0) * 150000.0
            else:
                score = score - remaining_m * 90.0
                if reason_id == wp.int64(max_steps_id) or reason_id == wp.int64(no_progress_id):
                    score = score - 90000.0
            score = score - early_slow_penalty * 2.10
            score = score - viability_penalty * 0.45
            score = score - missed_checkpoints * 14000.0
        elif profile_id == 7:
            score = score + raw_progress * 24.0 + best_progress * 12.0
            score = score + frontier_m * 60.0 + near_target_m * 135.0 + beyond_target_m * 210.0
            score = score + wp.min(pace_kph, 390.0) * 3800.0 * wp.max(0.30, frontier_quality_factor)
            score = score + wp.min(avg_speed_first_450_m, 340.0) * 150.0
            if valid_finish:
                score = score + 2800000.0
                score = score + wp.max(190.0 - valid_elapsed_s, 0.0) * 50000.0
                score = score - wp.max(valid_elapsed_s - 150.0, 0.0) * 55000.0
            if reason_id == wp.int64(max_steps_id):
                score = score - wp.max(elapsed_s - 150.0, 0.0) * 1200.0
            score = score - early_slow_penalty * 1.80
            score = score - viability_penalty * 0.50
            score = score - missed_checkpoints * 8000.0
        elif profile_id == 8:
            progress_pace_factor = 0.18 + lap_progress_ratio * 1.95
            score = score + raw_progress * 18.0 + best_progress * 120.0
            score = score + frontier_m * 70.0 + near_target_m * 130.0 + beyond_target_m * 170.0
            score = score + wp.min(pace_kph, 360.0) * 13500.0 * progress_pace_factor
            score = score + wp.min(final_speed_kph, 360.0) * 1200.0 * wp.max(0.15, lap_progress_ratio)
            score = score + wp.min(avg_speed_first_450_m, 340.0) * 160.0
            if best_progress >= 3000.0:
                score = score + 260000.0 + wp.min(pace_kph, 330.0) * 1200.0
            if best_progress >= 4000.0:
                score = score + 360000.0 + wp.min(pace_kph, 330.0) * 1700.0
            if best_progress >= 5000.0:
                score = score + 520000.0 + wp.min(pace_kph, 330.0) * 2400.0
            if valid_finish:
                score = score - (650000.0 + wp.max(valid_elapsed_s - 150.0, 0.0) * 90000.0)
            else:
                score = score - remaining_m * 28.0
            if best_progress >= 2400.0:
                score = score - wp.max(175.0 - pace_kph, 0.0) * 14000.0
            if (reason_id == wp.int64(max_steps_id) or reason_id == wp.int64(no_progress_id)) and not valid_finish:
                score = score - (65000.0 + wp.max(elapsed_s - 130.0, 0.0) * 900.0)
            score = score - early_slow_penalty * 1.45
            score = score - viability_penalty * 0.38
            score = score - missed_checkpoints * 10000.0
        elif profile_id == 9:
            score = score + best_progress * 12.0 + wp.min(final_speed_kph, 360.0) * 18.0
            score = score + frontier_m * 95.0 + near_target_m * 210.0 + beyond_target_m * 320.0
            score = score + wp.min(pace_kph, 300.0) * 16.0
            score = score - viability_penalty * 0.42
            score = score - final_lateral_error_m * 72.0
            score = score - final_heading_error_deg * 26.0
            score = score - final_yaw_rate_rps * 330.0
            score = score - final_steering * 300.0
            score = score - missed_checkpoints * 2500.0
        elif profile_id == 10:
            score = score + raw_progress * 18.0 + best_progress * 10.0
            score = score + focus_progress_m * 1250.0 + focus_progress_ratio * 28000.0
            score = score + wp.min(best_speed_kph, 300.0) * 35.0
            score = score + wp.min(final_speed_kph, 260.0) * 115.0
            score = score - focus_lateral_penalty - focus_heading_penalty - focus_stop_penalty
            score = score - missed_checkpoints * 3000.0
            if cleared_focus:
                score = score + 55000.0
            if reached_focus and reason_id == wp.int64(no_progress_id):
                score = score - 55000.0
            if reason_id == wp.int64(no_progress_id):
                score = score - 18000.0
            if reason_collisionish and reached_focus:
                score = score - 10000.0
        elif profile_id == 11:
            speed_bucket = wp.min(wp.floor(wp.max(final_speed_kph, 0.0) / 55.0), 5.0)
            lateral_bucket = wp.min(wp.floor(best_lateral_error_m / 4.0), 5.0)
            heading_bucket = wp.min(wp.floor(best_heading_error_deg / 10.0), 5.0)
            novelty_hint = speed_bucket * 1100.0 + (5.0 - lateral_bucket) * 750.0 + (5.0 - heading_bucket) * 650.0
            score = score + raw_progress * 16.0 + focus_progress_m * 900.0 + beyond_target_m * 140.0
            score = score + novelty_hint
            score = score + wp.min(pace_kph, 280.0) * 18.0
            score = score + wp.min(final_speed_kph, 320.0) * 70.0
            score = score - focus_lateral_penalty * 0.80 - focus_heading_penalty * 0.80 - focus_stop_penalty * 1.20
            if cleared_focus:
                score = score + 36000.0
            if reached_focus and reason_id == wp.int64(no_progress_id):
                score = score - 65000.0
            if reason_id == wp.int64(no_progress_id):
                score = score - 20000.0
        elif profile_id == 12:
            score = score + best_progress * 11.0 + wp.min(final_speed_kph, 380.0) * 24.0
            score = score + wp.min(pace_kph, 320.0) * 22.0
            score = score + frontier_m * 70.0 + near_target_m * 150.0 + beyond_target_m * 220.0
            score = score - viability_penalty * 0.18
            score = score - final_lateral_error_m * 52.0
            score = score - final_heading_error_deg * 18.0
        elif profile_id == 13:
            score = score + raw_progress * 28.0 + best_progress * 15.0
            score = score + frontier_m * 58.0 + near_target_m * 115.0 + beyond_target_m * 190.0
            score = score + wp.min(pace_kph, 280.0) * 18.0
            score = score - viability_penalty * 0.82
            score = score - final_lateral_error_m * 210.0
            score = score - final_heading_error_deg * 80.0
            score = score - final_yaw_rate_rps * 560.0
            score = score - final_steering * 520.0
            score = score - missed_checkpoints * 4500.0
            if clean:
                score = score + 7000.0
        elif profile_id == 14:
            avg_throttle = throttle_sum[row] / wp.max(row_count[row], 1.0)
            speed_drop = wp.max(start_speed_for_score_kph[row] - final_speed_kph, 0.0)
            score = score + max_brake[row] * 18000.0 + speed_drop * 120.0
            score = score - avg_throttle * 4000.0
            score = score - final_lateral_error_m * 110.0
        elif profile_id == 15:
            score = score + wp.min(final_speed_kph, 240.0) * 3.0
            score = score - final_lateral_error_m * 340.0
            score = score - final_heading_error_deg * 130.0
            score = score - final_yaw_rate_rps * 850.0
            score = score - missed_checkpoints * 5000.0
        elif profile_id == 16:
            score = score + wp.min(final_speed_kph, 340.0) * 18.0
            score = score - final_lateral_error_m * 150.0
            score = score - final_heading_error_deg * 45.0
            score = score - final_steering * 500.0
        elif profile_id == 17:
            score = score + best_progress * 5.0
            if final_valid_lap[row]:
                score = score + 10000.0
            else:
                score = score - 10000.0
            score = score - missed_checkpoints * 10000.0
            score = score - final_lateral_error_m * 120.0
            score = score - final_heading_error_deg * 40.0
        elif profile_id == 1:
            score = score + raw_progress * 31.0 + best_progress * 22.0
            score = score + frontier_m * 75.0 + near_target_m * 160.0 + beyond_target_m * 280.0
            score = score + wp.min(pace_kph, 300.0) * 20.0
            score = score - viability_penalty * 0.22
            score = score - missed_checkpoints * 1600.0
            if reason_collisionish:
                score = score - 1500.0
        elif profile_id == 2:
            score = score + best_progress * 18.0 + wp.min(final_speed_kph, 380.0) * 20.0
            score = score + wp.min(pace_kph, 310.0) * 86.0
            score = score + frontier_m * 105.0 + near_target_m * 235.0 + beyond_target_m * 340.0
            if valid_finish:
                score = score + valid_lap_speed_bonus * 0.18
            if time_to_450_valid:
                score = score + wp.max(18.0 - time_to_450_m[row], 0.0) * 260.0
            score = score + wp.min(avg_speed_first_450_m, 320.0) * 28.0
            score = score - early_slow_penalty
            score = score - early_brake_penalty
            score = score - viability_penalty * 0.58
            score = score - final_lateral_error_m * 70.0
            score = score - final_heading_error_deg * 24.0
            score = score - final_yaw_rate_rps * 320.0
            score = score - final_steering * 260.0
            score = score - missed_checkpoints * 2500.0
        elif profile_id == 3:
            score = score + raw_progress * 20.0 + wp.min(pace_kph, 320.0) * 125.0
            score = score + wp.min(avg_speed_first_300_m, 300.0) * 56.0
            score = score + wp.min(avg_speed_first_450_m, 320.0) * 38.0
            if time_to_450_valid:
                score = score + wp.max(20.0 - time_to_450_m[row], 0.0) * 420.0
            score = score + frontier_m * 38.0 + beyond_target_m * 90.0
            score = score - early_slow_penalty * 1.55
            score = score - early_brake_penalty * 1.35
            score = score - final_lateral_error_m * 45.0
            score = score - final_heading_error_deg * 15.0
            score = score - missed_checkpoints * 1800.0
        elif profile_id == 4:
            score = score + wp.min(final_speed_kph, 320.0) * 4.0
            score = score - final_lateral_error_m * 260.0
            score = score - final_heading_error_deg * 95.0
            score = score - final_yaw_rate_rps * 600.0
            score = score - final_steering * 800.0
            score = score - missed_checkpoints * 5000.0
        else:
            score = score + wp.min(final_speed_kph, 320.0) * 5.0
            score = score - final_lateral_error_m * 120.0
            score = score - final_heading_error_deg * 35.0
            score = score - final_yaw_rate_rps * 250.0
            score = score - final_steering * 500.0
            score = score - missed_checkpoints * 2500.0
        output[row] = score

    return _score_kernel


def apply_physics_warp_batch(
    state: GpuCarBatch,
    *,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    params: GpuCarParams,
    meters_per_pixel: torch.Tensor,
) -> tuple[GpuCarBatch, GpuMovementBatch]:
    """Apply one fused Warp physics step matching ``apply_physics_batch`` for CUDA float32 batches."""

    if params.physics_model not in {"v1", "v2"}:
        raise ValueError(f"Unsupported Warp fused physics_model={params.physics_model!r}.")
    if params.physics_model == "v2" and len(params.v2_gear_ratios) != 8:
        raise ValueError("Warp fused V2 physics expects exactly 8 gear ratios.")
    if state.dtype != torch.float32:
        raise ValueError("Warp fused physics currently supports torch.float32 state tensors only.")
    if state.device.type != "cuda":
        raise ValueError("Warp fused physics currently supports CUDA state tensors only.")
    throttle = throttle.to(device=state.device, dtype=state.dtype)
    brake = brake.to(device=state.device, dtype=state.dtype)
    steer = steer.to(device=state.device, dtype=state.dtype)
    if throttle.shape != state.x.shape or brake.shape != state.x.shape or steer.shape != state.x.shape:
        raise ValueError("Warp fused physics controls must match the state batch shape.")
    if meters_per_pixel.numel() != 1:
        raise ValueError("Warp fused physics expects scalar meters_per_pixel.")

    x_out = torch.empty_like(state.x)
    y_out = torch.empty_like(state.y)
    heading_out = torch.empty_like(state.heading_rad)
    speed_out = torch.empty_like(state.speed_mps)
    yaw_rate_out = torch.empty_like(state.yaw_rate_rps)
    steering_out = torch.empty_like(state.steering)
    elapsed_steps_out = torch.empty_like(state.elapsed_steps)
    movement_x0 = torch.empty_like(state.x)
    movement_y0 = torch.empty_like(state.y)
    movement_x1 = torch.empty_like(state.x)
    movement_y1 = torch.empty_like(state.y)

    wp = require_warp()
    if params.physics_model == "v2":
        gear_ratios = tuple(float(value) for value in params.v2_gear_ratios)
        wp.launch(
            _physics_v2_kernel(),
            dim=int(state.x.numel()),
            inputs=[
                wp.from_torch(state.x),
                wp.from_torch(state.y),
                wp.from_torch(state.heading_rad),
                wp.from_torch(state.speed_mps),
                wp.from_torch(state.yaw_rate_rps),
                wp.from_torch(state.steering),
                wp.from_torch(state.elapsed_steps),
                wp.from_torch(throttle),
                wp.from_torch(brake),
                wp.from_torch(steer),
                wp.from_torch(x_out),
                wp.from_torch(y_out),
                wp.from_torch(heading_out),
                wp.from_torch(speed_out),
                wp.from_torch(yaw_rate_out),
                wp.from_torch(steering_out),
                wp.from_torch(elapsed_steps_out),
                wp.from_torch(movement_x0),
                wp.from_torch(movement_y0),
                wp.from_torch(movement_x1),
                wp.from_torch(movement_y1),
                float(params.mass),
                float(params.wheelbase_m),
                float(params.v2_max_steer_deg) * 3.141592653589793 / 180.0,
                float(params.v2_steer_response),
                float(params.v2_steering_speed_sensitivity),
                float(params.dt),
                float(meters_per_pixel.detach().cpu().item()),
                float(params.v2_front_weight_distribution),
                float(params.v2_cg_height_m),
                float(params.v2_track_width_m),
                float(params.v2_front_axle_distance_m),
                float(params.v2_rear_axle_distance_m),
                float(params.v2_front_cornering_stiffness_n_per_rad),
                float(params.v2_rear_cornering_stiffness_n_per_rad),
                float(params.v2_front_peak_mu),
                float(params.v2_rear_peak_mu),
                float(params.v2_mechanical_grip_low_speed_scale),
                float(params.v2_mechanical_grip_high_speed_scale),
                float(params.v2_mechanical_grip_transition_mps),
                float(params.v2_tire_shape_c),
                math.radians(float(params.v2_slip_angle_peak_deg)),
                float(params.v2_rear_slip_steer_coupling),
                float(params.v2_post_peak_falloff),
                float(params.v2_load_sensitivity),
                float(params.v2_aero_downforce_n_per_mps2),
                float(params.v2_aero_balance_front),
                float(params.v2_engine_power_w),
                float(params.v2_drivetrain_efficiency),
                float(params.v2_power_min_speed_mps),
                float(params.v2_max_drive_g),
                float(params.v2_max_brake_g),
                float(params.v2_brake_lock_threshold),
                float(params.v2_brake_lock_min_speed_mps),
                float(params.v2_brake_lock_steer_loss),
                float(params.v2_drag_coefficient),
                float(params.v2_rolling_resistance_mps2),
                float(params.v2_max_speed_mps),
                gear_ratios[0],
                gear_ratios[1],
                gear_ratios[2],
                gear_ratios[3],
                gear_ratios[4],
                gear_ratios[5],
                gear_ratios[6],
                gear_ratios[7],
                float(params.v2_final_drive_ratio),
                float(params.v2_wheel_radius_m),
                float(params.v2_idle_rpm),
                float(params.v2_shift_up_rpm),
                float(params.v2_max_rpm),
                float(params.v2_torque_peak_rpm),
                float(params.v2_torque_low_rpm_factor),
                float(params.v2_torque_high_rpm_factor),
                float(params.v2_tire_scrub_drag),
                float(params.v2_surface_mu),
            ],
            device=_warp_device_for_torch(state.device),
        )
    else:
        wp.launch(
            _physics_kernel(),
            dim=int(state.x.numel()),
            inputs=[
                wp.from_torch(state.x),
                wp.from_torch(state.y),
                wp.from_torch(state.heading_rad),
                wp.from_torch(state.speed_mps),
                wp.from_torch(state.steering),
                wp.from_torch(state.elapsed_steps),
                wp.from_torch(throttle),
                wp.from_torch(brake),
                wp.from_torch(steer),
                wp.from_torch(x_out),
                wp.from_torch(y_out),
                wp.from_torch(heading_out),
                wp.from_torch(speed_out),
                wp.from_torch(yaw_rate_out),
                wp.from_torch(steering_out),
                wp.from_torch(elapsed_steps_out),
                wp.from_torch(movement_x0),
                wp.from_torch(movement_y0),
                wp.from_torch(movement_x1),
                wp.from_torch(movement_y1),
                float(params.wheelbase_m),
                float(params.max_steer_deg) * 3.141592653589793 / 180.0,
                float(params.steer_response),
                float(params.engine_accel_mps2),
                float(params.brake_accel_mps2),
                float(params.drag_coefficient),
                float(params.rolling_resistance_mps2),
                float(params.grip_g),
                float(params.aero_grip_per_mps2),
                float(params.max_grip_g),
                float(params.max_drive_g),
                float(params.max_brake_g),
                float(params.steering_speed_sensitivity),
                float(params.max_speed_mps),
                float(params.dt),
                float(meters_per_pixel.detach().cpu().item()),
            ],
            device=_warp_device_for_torch(state.device),
        )

    return (
        GpuCarBatch(
            x=x_out,
            y=y_out,
            heading_rad=heading_out,
            speed_mps=speed_out,
            yaw_rate_rps=yaw_rate_out,
            steering=steering_out,
            raw_progress_m=state.raw_progress_m,
            monotonic_progress_m=state.monotonic_progress_m,
            last_raw_progress_px=state.last_raw_progress_px,
            checkpoint_index=state.checkpoint_index,
            next_checkpoint_index=state.next_checkpoint_index,
            checkpoints_passed=state.checkpoints_passed,
            missed_checkpoint_count=state.missed_checkpoint_count,
            lap_index=state.lap_index,
            elapsed_steps=elapsed_steps_out,
            no_progress_steps=state.no_progress_steps,
            alive=state.alive,
            terminated=state.terminated,
            truncated=state.truncated,
            termination_reason_id=state.termination_reason_id,
            valid_lap=state.valid_lap,
            finish_crossed=state.finish_crossed,
            completed_lap=state.completed_lap,
            segment_complete=state.segment_complete,
            segment_release_observed=state.segment_release_observed,
            last_throttle=state.last_throttle,
            last_brake=state.last_brake,
            last_steer=state.last_steer,
        ),
        GpuMovementBatch(x0=movement_x0, y0=movement_y0, x1=movement_x1, y1=movement_y1),
    )


def track_errors_warp_local_batch(
    points_xy: torch.Tensor,
    heading_rad: torch.Tensor,
    track: GpuTrackTensors,
    *,
    previous_progress_px: torch.Tensor,
    previous_segment_idx: torch.Tensor,
    window_px: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Project points with the rollout local-window path using one Warp kernel."""

    if points_xy.device.type != "cuda" or heading_rad.device.type != "cuda":
        raise ValueError("Warp local projection currently requires CUDA tensors.")
    if points_xy.dtype != torch.float32 or heading_rad.dtype != torch.float32:
        raise ValueError("Warp local projection currently supports torch.float32 tensors only.")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape [batch, 2].")
    batch_size = int(points_xy.shape[0])
    if heading_rad.shape != (batch_size,):
        raise ValueError("heading_rad must have shape [batch].")
    if previous_progress_px.shape != (batch_size,) or previous_segment_idx.shape != (batch_size,):
        raise ValueError("previous progress and segment tensors must have shape [batch].")
    if track.device != points_xy.device or track.dtype != points_xy.dtype:
        raise ValueError("Warp local projection track tensors must match point tensor device and dtype.")
    if not points_xy.is_contiguous():
        points_xy = points_xy.contiguous()
    heading_rad = heading_rad.contiguous()
    previous_progress_px = previous_progress_px.to(device=points_xy.device, dtype=torch.float32).contiguous()
    previous_segment_idx = previous_segment_idx.to(device=points_xy.device, dtype=torch.int64).contiguous()
    if isinstance(window_px, torch.Tensor):
        if window_px.numel() != 1:
            raise ValueError("Warp local projection expects scalar window_px.")
        window_value = float(window_px.detach().cpu().item())
    else:
        window_value = float(window_px)

    raw_progress_px = torch.empty(batch_size, device=points_xy.device, dtype=torch.float32)
    lateral_error_m = torch.empty_like(raw_progress_px)
    signed_lateral_error_m = torch.empty_like(raw_progress_px)
    heading_error_rad = torch.empty_like(raw_progress_px)
    segment_idx = torch.empty(batch_size, device=points_xy.device, dtype=torch.int64)
    projection_x = torch.empty_like(raw_progress_px)
    projection_y = torch.empty_like(raw_progress_px)

    wp = require_warp()
    wp.launch(
        _local_projection_kernel(),
        dim=batch_size,
        inputs=[
            wp.from_torch(points_xy),
            wp.from_torch(heading_rad),
            wp.from_torch(previous_progress_px),
            wp.from_torch(previous_segment_idx),
            wp.from_torch(track.centerline_xy),
            wp.from_torch(track.centerline_segment_vec),
            wp.from_torch(track.centerline_segment_len),
            wp.from_torch(track.centerline_segment_len2),
            wp.from_torch(track.centerline_cumdist_px),
            wp.from_torch(track.centerline_segment_mid_px),
            wp.from_torch(track.local_projection_indices),
            wp.from_torch(raw_progress_px),
            wp.from_torch(lateral_error_m),
            wp.from_torch(signed_lateral_error_m),
            wp.from_torch(heading_error_rad),
            wp.from_torch(segment_idx),
            wp.from_torch(projection_x),
            wp.from_torch(projection_y),
            int(track.local_projection_indices.shape[1]),
            int(track.centerline_xy.shape[0]),
            float(track.length_px.detach().cpu().item()),
            float(track.meters_per_pixel.detach().cpu().item()),
            window_value,
        ],
        device=_warp_device_for_torch(points_xy.device),
    )
    return {
        "raw_progress_px": raw_progress_px,
        "lateral_error_m": lateral_error_m,
        "signed_lateral_error_m": signed_lateral_error_m,
        "heading_error_rad": heading_error_rad,
        "segment_idx": segment_idx,
        "projection_x": projection_x,
        "projection_y": projection_y,
    }


def point_is_drivable_warp_batch(x: torch.Tensor, y: torch.Tensor, track: GpuTrackTensors) -> torch.Tensor:
    """Return drivable-mask membership with one Warp kernel."""

    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("Warp drivable-mask lookup currently requires CUDA tensors.")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("Warp drivable-mask lookup currently supports torch.float32 tensors only.")
    if x.shape != y.shape:
        raise ValueError("Warp drivable-mask lookup x/y tensors must have the same shape.")
    if track.drivable_mask.device != x.device:
        raise ValueError("Warp drivable-mask lookup track mask must be on the same CUDA device.")
    x = x.contiguous()
    y = y.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.bool)
    wp = require_warp()
    height = int(track.drivable_mask.shape[0])
    width = int(track.drivable_mask.shape[1])
    wp.launch(
        _drivable_mask_kernel(),
        dim=int(x.numel()),
        inputs=[
            wp.from_torch(x),
            wp.from_torch(y),
            wp.from_torch(track.drivable_mask),
            wp.from_torch(output),
            width,
            height,
        ],
        device=_warp_device_for_torch(x.device),
    )
    return output


def segments_intersect_any_warp_grid_batch(
    movements: torch.Tensor,
    track: GpuTrackTensors,
    *,
    query_span: int = 3,
) -> torch.Tensor:
    """Exact movement/boundary intersection using the precomputed boundary-cell grid in one Warp kernel."""

    if movements.device.type != "cuda":
        raise ValueError("Warp grid collision currently requires CUDA tensors.")
    if movements.dtype != torch.float32:
        raise ValueError("Warp grid collision currently supports torch.float32 tensors only.")
    if movements.ndim != 2 or movements.shape[1] != 4:
        raise ValueError("movements must have shape [batch, 4].")
    if track.boundary_segments.device != movements.device or track.boundary_segments.dtype != movements.dtype:
        raise ValueError("Warp grid collision track boundary tensors must match movement tensor device and dtype.")
    if track.boundary_segments.numel() == 0:
        return torch.zeros(movements.shape[0], device=movements.device, dtype=torch.bool)
    if track.boundary_grid_indices.numel() == 0:
        raise ValueError("Warp grid collision requires precomputed boundary-grid indices.")
    movements = movements.contiguous()
    output = torch.empty(movements.shape[0], device=movements.device, dtype=torch.bool)
    wp = require_warp()
    wp.launch(
        _grid_collision_kernel(),
        dim=int(movements.shape[0]),
        inputs=[
            wp.from_torch(movements),
            wp.from_torch(track.boundary_segments),
            wp.from_torch(track.boundary_grid_indices),
            wp.from_torch(track.boundary_grid_counts),
            wp.from_torch(output),
            int(track.boundary_grid_width),
            int(track.boundary_grid_height),
            int(track.boundary_grid_indices.shape[1]),
            max(1, int(query_span)),
            float(track.boundary_grid_cell_size_px.detach().cpu().item()),
        ],
        device=_warp_device_for_torch(movements.device),
    )
    return output


def segments_intersect_any_warp_batch(movements: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """Exact all-segment intersection in one Warp kernel for small segment sets such as finish lines."""

    if movements.device.type != "cuda":
        raise ValueError("Warp segment intersection currently requires CUDA tensors.")
    if movements.dtype != torch.float32:
        raise ValueError("Warp segment intersection currently supports torch.float32 tensors only.")
    if movements.ndim != 2 or movements.shape[1] != 4:
        raise ValueError("movements must have shape [batch, 4].")
    if segments.device != movements.device or segments.dtype != movements.dtype:
        raise ValueError("Warp segment tensors must match movement tensor device and dtype.")
    if segments.ndim != 2 or segments.shape[1] != 4:
        raise ValueError("segments must have shape [segment_count, 4].")
    movements = movements.contiguous()
    segments = segments.contiguous()
    output = torch.empty(movements.shape[0], device=movements.device, dtype=torch.bool)
    wp = require_warp()
    wp.launch(
        _segments_intersect_any_kernel(),
        dim=int(movements.shape[0]),
        inputs=[
            wp.from_torch(movements),
            wp.from_torch(segments),
            wp.from_torch(output),
            int(segments.shape[0]),
        ],
        device=_warp_device_for_torch(movements.device),
    )
    return output


def commit_physics_warp_batch(state: GpuCarBatch, next_state: GpuCarBatch, active: torch.Tensor) -> None:
    """Copy physics-updated fields into ``state`` for active rows only."""

    if state.device.type != "cuda" or next_state.device.type != "cuda":
        raise ValueError("Warp physics commit requires CUDA state tensors.")
    if state.dtype != torch.float32 or next_state.dtype != torch.float32:
        raise ValueError("Warp physics commit currently supports torch.float32 tensors only.")
    if state.size != next_state.size:
        raise ValueError("state and next_state batch sizes must match.")
    if active.device != state.device or active.dtype != torch.bool or active.shape != state.x.shape:
        raise ValueError("active must be a CUDA bool tensor matching the state batch shape.")
    wp = require_warp()
    wp.launch(
        _commit_physics_kernel(),
        dim=state.size,
        inputs=[
            wp.from_torch(active.contiguous()),
            wp.from_torch(next_state.x.contiguous()),
            wp.from_torch(next_state.y.contiguous()),
            wp.from_torch(next_state.heading_rad.contiguous()),
            wp.from_torch(next_state.speed_mps.contiguous()),
            wp.from_torch(next_state.yaw_rate_rps.contiguous()),
            wp.from_torch(next_state.steering.contiguous()),
            wp.from_torch(next_state.elapsed_steps.contiguous()),
            wp.from_torch(state.x),
            wp.from_torch(state.y),
            wp.from_torch(state.heading_rad),
            wp.from_torch(state.speed_mps),
            wp.from_torch(state.yaw_rate_rps),
            wp.from_torch(state.steering),
            wp.from_torch(state.elapsed_steps),
        ],
        device=_warp_device_for_torch(state.device),
    )


def open_step_bookkeeping_warp_batch(
    state: GpuCarBatch,
    *,
    last_segment_idx: torch.Tensor,
    active: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    old_progress_m: torch.Tensor,
    old_raw_progress_px: torch.Tensor,
    old_lap_index: torch.Tensor,
    raw_progress_px: torch.Tensor,
    segment_idx: torch.Tensor,
    lateral_error_m: torch.Tensor,
    heading_error_rad: torch.Tensor,
    collided: torch.Tensor,
    drivable: torch.Tensor,
    physical_finish: torch.Tensor,
    segment_target_progress_m: torch.Tensor | None = None,
    gates: Any | None = None,
    track: GpuTrackTensors,
    local_projection_window_m: float,
    checkpoint_lateral_limit_m: float,
    no_progress_limit_steps: int,
    max_steps: int,
) -> dict[str, torch.Tensor]:
    """Open-distance Warp bookkeeping for one post-physics/post-projection rollout step.

    This mirrors the no-target-termination branch of ``GpuMonzaBatch._step`` after
    physics, projection, collision, and drivable-mask outputs are available.
    """

    if state.device.type != "cuda":
        raise ValueError("Warp step bookkeeping requires CUDA state tensors.")
    if state.dtype != torch.float32:
        raise ValueError("Warp step bookkeeping currently supports torch.float32 state tensors only.")
    if track.device != state.device or track.dtype != state.dtype:
        raise ValueError("Warp step bookkeeping track tensors must match state device and dtype.")
    batch_size = state.size
    float_inputs = {
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "old_progress_m": old_progress_m,
        "old_raw_progress_px": old_raw_progress_px,
        "raw_progress_px": raw_progress_px,
        "lateral_error_m": lateral_error_m,
        "heading_error_rad": heading_error_rad,
    }
    checked_float_inputs: dict[str, torch.Tensor] = {}
    for name, tensor in float_inputs.items():
        checked = _require_cuda_float32_tensor(name, tensor)
        if checked.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [batch].")
        checked_float_inputs[name] = checked
    bool_inputs = {
        "active": active,
        "collided": collided,
        "drivable": drivable,
        "physical_finish": physical_finish,
    }
    checked_bool_inputs: dict[str, torch.Tensor] = {}
    for name, tensor in bool_inputs.items():
        if tensor.device != state.device or tensor.dtype != torch.bool or tensor.shape != (batch_size,):
            raise ValueError(f"{name} must be a CUDA bool tensor with shape [batch].")
        checked_bool_inputs[name] = tensor.contiguous()
    int_inputs = {
        "old_lap_index": old_lap_index,
        "segment_idx": segment_idx,
    }
    checked_int_inputs: dict[str, torch.Tensor] = {}
    for name, tensor in int_inputs.items():
        if tensor.device != state.device or tensor.dtype != torch.int64 or tensor.shape != (batch_size,):
            raise ValueError(f"{name} must be a CUDA int64 tensor with shape [batch].")
        checked_int_inputs[name] = tensor.contiguous()
    if last_segment_idx.device != state.device or last_segment_idx.dtype != torch.int64 or last_segment_idx.shape != (batch_size,):
        raise ValueError("last_segment_idx must be a CUDA int64 tensor with shape [batch].")
    if not last_segment_idx.is_contiguous():
        raise ValueError("last_segment_idx must be contiguous for in-place Warp updates.")
    terminate_at_target_progress = bool(getattr(gates, "terminate_at_target_progress", False))
    if segment_target_progress_m is None:
        if terminate_at_target_progress:
            raise ValueError("segment_target_progress_m is required when terminate_at_target_progress is true.")
        target_progress_tensor = state.monotonic_progress_m
    else:
        target_progress_tensor = _require_cuda_float32_tensor("segment_target_progress_m", segment_target_progress_m)
        if target_progress_tensor.shape != (batch_size,):
            raise ValueError("segment_target_progress_m must have shape [batch].")
    target_min_speed_kph = getattr(gates, "target_min_speed_kph", None)
    target_max_speed_kph = getattr(gates, "target_max_speed_kph", None)
    target_max_lateral_error_m = getattr(gates, "target_max_lateral_error_m", None)
    target_max_heading_error_deg = getattr(gates, "target_max_heading_error_deg", None)
    target_max_abs_yaw_rate_rps = getattr(gates, "target_max_abs_yaw_rate_rps", None)
    target_max_abs_steering = getattr(gates, "target_max_abs_steering", None)
    release_min_speed_kph = float(getattr(gates, "segment_release_min_speed_kph", 0.0))
    release_max_speed_kph = float(getattr(gates, "segment_release_max_speed_kph", 0.0))

    progress_delta_m = torch.empty(batch_size, device=state.device, dtype=torch.float32)
    collided_out = torch.empty(batch_size, device=state.device, dtype=torch.bool)
    off_track_out = torch.empty_like(collided_out)
    telemetry_valid_lap = torch.empty_like(collided_out)
    wp = require_warp()
    wp.launch(
        _open_step_bookkeeping_kernel(),
        dim=batch_size,
        inputs=[
            wp.from_torch(checked_bool_inputs["active"]),
            wp.from_torch(checked_float_inputs["throttle"]),
            wp.from_torch(checked_float_inputs["brake"]),
            wp.from_torch(checked_float_inputs["steer"]),
            wp.from_torch(checked_float_inputs["old_progress_m"]),
            wp.from_torch(checked_float_inputs["old_raw_progress_px"]),
            wp.from_torch(checked_int_inputs["old_lap_index"]),
            wp.from_torch(checked_float_inputs["raw_progress_px"]),
            wp.from_torch(checked_int_inputs["segment_idx"]),
            wp.from_torch(checked_float_inputs["lateral_error_m"]),
            wp.from_torch(checked_float_inputs["heading_error_rad"]),
            wp.from_torch(checked_bool_inputs["collided"]),
            wp.from_torch(checked_bool_inputs["drivable"]),
            wp.from_torch(checked_bool_inputs["physical_finish"]),
            wp.from_torch(target_progress_tensor.contiguous()),
            wp.from_torch(last_segment_idx),
            wp.from_torch(state.speed_mps),
            wp.from_torch(state.yaw_rate_rps),
            wp.from_torch(state.steering),
            wp.from_torch(state.raw_progress_m),
            wp.from_torch(state.monotonic_progress_m),
            wp.from_torch(state.last_raw_progress_px),
            wp.from_torch(state.checkpoint_index),
            wp.from_torch(state.next_checkpoint_index),
            wp.from_torch(state.checkpoints_passed),
            wp.from_torch(state.missed_checkpoint_count),
            wp.from_torch(state.lap_index),
            wp.from_torch(state.elapsed_steps),
            wp.from_torch(state.no_progress_steps),
            wp.from_torch(state.alive),
            wp.from_torch(state.terminated),
            wp.from_torch(state.truncated),
            wp.from_torch(state.termination_reason_id),
            wp.from_torch(state.valid_lap),
            wp.from_torch(state.finish_crossed),
            wp.from_torch(state.completed_lap),
            wp.from_torch(state.segment_complete),
            wp.from_torch(state.segment_release_observed),
            wp.from_torch(state.last_throttle),
            wp.from_torch(state.last_brake),
            wp.from_torch(state.last_steer),
            wp.from_torch(progress_delta_m),
            wp.from_torch(collided_out),
            wp.from_torch(off_track_out),
            wp.from_torch(telemetry_valid_lap),
            float(track.length_px.detach().cpu().item()),
            float(track.meters_per_pixel.detach().cpu().item()),
            float(local_projection_window_m),
            float(track.checkpoint_spacing_m.detach().cpu().item()),
            float(checkpoint_lateral_limit_m),
            float(track.length_m.detach().cpu().item()),
            int(track.checkpoint_count),
            int(no_progress_limit_steps),
            int(max_steps),
            terminate_at_target_progress,
            target_min_speed_kph is not None,
            0.0 if target_min_speed_kph is None else float(target_min_speed_kph),
            target_max_speed_kph is not None,
            0.0 if target_max_speed_kph is None else float(target_max_speed_kph),
            target_max_lateral_error_m is not None,
            0.0 if target_max_lateral_error_m is None else float(target_max_lateral_error_m),
            target_max_heading_error_deg is not None,
            0.0 if target_max_heading_error_deg is None else float(target_max_heading_error_deg),
            target_max_abs_yaw_rate_rps is not None,
            0.0 if target_max_abs_yaw_rate_rps is None else float(target_max_abs_yaw_rate_rps),
            target_max_abs_steering is not None,
            0.0 if target_max_abs_steering is None else float(target_max_abs_steering),
            bool(getattr(gates, "segment_fail_on_speed_gate_miss", False)),
            bool(getattr(gates, "segment_require_release", False)),
            min(release_min_speed_kph, release_max_speed_kph),
            max(release_min_speed_kph, release_max_speed_kph),
            float(getattr(gates, "segment_release_max_brake", 0.0)),
            float(getattr(gates, "segment_release_max_throttle", 0.0)),
            int(TERMINATION_REASON_TO_ID["segment_complete"]),
            int(TERMINATION_REASON_TO_ID["segment_release_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_min_speed_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_speed_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_lateral_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_heading_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_yaw_rate_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["segment_steering_gate_failed"]),
            int(TERMINATION_REASON_TO_ID["lap_complete"]),
            int(TERMINATION_REASON_TO_ID["collision"]),
            int(TERMINATION_REASON_TO_ID["off_track"]),
            int(TERMINATION_REASON_TO_ID["no_progress"]),
            int(TERMINATION_REASON_TO_ID["max_steps"]),
        ],
        device=_warp_device_for_torch(state.device),
    )
    return {
        "progress_delta_m": progress_delta_m,
        "collided": collided_out,
        "off_track": off_track_out,
        "telemetry_valid_lap": telemetry_valid_lap,
    }


def open_step_warp_batch(
    state: GpuCarBatch,
    *,
    last_segment_idx: torch.Tensor,
    active: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    params: GpuCarParams,
    track: GpuTrackTensors,
    sim_config: SimConfig,
    segment_target_progress_m: torch.Tensor | None = None,
    gates: Any | None = None,
    collision_check: bool = True,
    collision_mode: str = "exact_grid",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one Warp-backed step and mutate ``state`` in place."""

    if state.device.type != "cuda" or state.dtype != torch.float32:
        raise ValueError("Warp open step currently requires CUDA float32 state tensors.")
    active = active.to(device=state.device, dtype=torch.bool).contiguous()
    throttle = _require_cuda_float32_tensor("throttle", throttle.to(device=state.device, dtype=torch.float32))
    brake = _require_cuda_float32_tensor("brake", brake.to(device=state.device, dtype=torch.float32))
    steer = _require_cuda_float32_tensor("steer", steer.to(device=state.device, dtype=torch.float32))
    if throttle.shape != state.x.shape or brake.shape != state.x.shape or steer.shape != state.x.shape:
        raise ValueError("controls must match the state batch shape.")

    if sim_config.launch_guard_progress_m > 0.0 and sim_config.launch_guard_min_speed_kph > 0.0:
        guard = (
            active
            & (state.monotonic_progress_m <= float(sim_config.launch_guard_progress_m))
            & (state.speed_mps * 3.6 < float(sim_config.launch_guard_min_speed_kph))
            & (throttle <= 1.0e-6)
            & (brake > 1.0e-6)
        )
        throttle = torch.where(guard, torch.full_like(throttle, float(sim_config.launch_guard_throttle)), throttle)
        brake = torch.where(guard, torch.zeros_like(brake), brake)

    old_progress_m = state.monotonic_progress_m.clone()
    old_raw_progress_px = state.last_raw_progress_px.clone()
    old_lap_index = state.lap_index.clone()
    next_state, movement = apply_physics_warp_batch(
        state,
        throttle=throttle,
        brake=brake,
        steer=steer,
        params=params,
        meters_per_pixel=track.meters_per_pixel,
    )
    commit_physics_warp_batch(state, next_state, active)
    window_px = float(sim_config.local_projection_window_m) / max(float(track.meters_per_pixel.detach().cpu().item()), 1.0e-6)
    errors = track_errors_warp_local_batch(
        state.position_xy(),
        state.heading_rad,
        track,
        previous_progress_px=old_raw_progress_px,
        previous_segment_idx=last_segment_idx,
        window_px=window_px,
    )
    movements = movement.as_segments()
    if collision_check:
        if collision_mode == "exact_grid":
            collided = segments_intersect_any_warp_grid_batch(movements, track)
        elif collision_mode == "exact_all_segments":
            collided = segments_intersect_any_warp_batch(movements, track.boundary_segments)
        else:
            raise ValueError(f"Unsupported Warp open-step collision mode {collision_mode!r}.")
    else:
        collided = torch.zeros(state.size, device=state.device, dtype=torch.bool)
    drivable = point_is_drivable_warp_batch(state.x, state.y, track)
    physical_finish = segments_intersect_any_warp_batch(movements, track.finish_line)
    bookkeeping = open_step_bookkeeping_warp_batch(
        state,
        last_segment_idx=last_segment_idx,
        active=active,
        throttle=throttle,
        brake=brake,
        steer=steer,
        old_progress_m=old_progress_m,
        old_raw_progress_px=old_raw_progress_px,
        old_lap_index=old_lap_index,
        raw_progress_px=errors["raw_progress_px"],
        segment_idx=errors["segment_idx"],
        lateral_error_m=errors["lateral_error_m"],
        heading_error_rad=errors["heading_error_rad"],
        collided=collided,
        drivable=drivable,
        physical_finish=physical_finish,
        segment_target_progress_m=segment_target_progress_m,
        gates=gates,
        track=track,
        local_projection_window_m=sim_config.local_projection_window_m,
        checkpoint_lateral_limit_m=sim_config.checkpoint_lateral_limit_m,
        no_progress_limit_steps=sim_config.no_progress_limit_steps,
        max_steps=sim_config.max_steps,
    )
    errors["progress_delta_m"] = bookkeeping["progress_delta_m"]
    errors["telemetry_valid_lap"] = bookkeeping["telemetry_valid_lap"]
    return errors, bookkeeping["collided"], bookkeeping["off_track"], bookkeeping["telemetry_valid_lap"]


def controller_controls_warp_batch(
    controller_weights: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp version of ``controller_controls_batch`` for CUDA float32 controller populations."""

    if controller_weights.device.type != "cuda" or features.device.type != "cuda":
        raise ValueError("Warp controller controls currently require CUDA tensors.")
    if controller_weights.dtype != torch.float32 or features.dtype != torch.float32:
        raise ValueError("Warp controller controls currently support torch.float32 tensors only.")
    if controller_weights.ndim != 3 or controller_weights.shape[1] != 3:
        raise ValueError("controller_weights must have shape [batch, 3, feature_count].")
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_count].")
    if controller_weights.shape[0] != features.shape[0] or controller_weights.shape[2] != features.shape[1]:
        raise ValueError("controller weight and feature batch dimensions do not match.")
    controller_weights = controller_weights.contiguous()
    features = features.contiguous()
    throttle = torch.empty(features.shape[0], device=features.device, dtype=torch.float32)
    brake = torch.empty_like(throttle)
    steer = torch.empty_like(throttle)
    wp = require_warp()
    wp.launch(
        _controller_controls_kernel(),
        dim=int(features.shape[0]),
        inputs=[
            wp.from_torch(controller_weights),
            wp.from_torch(features),
            wp.from_torch(throttle),
            wp.from_torch(brake),
            wp.from_torch(steer),
            int(features.shape[1]),
        ],
        device=_warp_device_for_torch(features.device),
    )
    return throttle, brake, steer


def phase_controls_warp_batch(
    *,
    action_controls: torch.Tensor,
    phase_action_ids: torch.Tensor,
    phase_thresholds: torch.Tensor,
    elapsed_steps: torch.Tensor,
    progress_delta_m: torch.Tensor,
    use_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp version of phase/progress-phase discrete control selection."""

    if action_controls.device.type != "cuda":
        raise ValueError("Warp phase controls currently require CUDA tensors.")
    if action_controls.dtype != torch.float32:
        raise ValueError("Warp phase controls currently support torch.float32 action controls only.")
    if action_controls.ndim != 2 or action_controls.shape[1] != 3:
        raise ValueError("action_controls must have shape [action_count, 3].")
    if phase_action_ids.device.type != "cuda" or phase_action_ids.dtype != torch.int64:
        raise ValueError("phase_action_ids must be a CUDA int64 tensor.")
    if phase_thresholds.device.type != "cuda":
        raise ValueError("phase_thresholds must be on CUDA.")
    if phase_thresholds.dtype != torch.float32:
        phase_thresholds = phase_thresholds.to(device=action_controls.device, dtype=torch.float32)
    if elapsed_steps.device.type != "cuda" or elapsed_steps.dtype != torch.int64:
        raise ValueError("elapsed_steps must be a CUDA int64 tensor.")
    if progress_delta_m.device.type != "cuda" or progress_delta_m.dtype != torch.float32:
        raise ValueError("progress_delta_m must be a CUDA float32 tensor.")
    if phase_action_ids.ndim != 2 or phase_thresholds.ndim != 2:
        raise ValueError("phase_action_ids and phase_thresholds must have shape [batch, phase_count].")
    if phase_action_ids.shape != phase_thresholds.shape:
        raise ValueError("phase_action_ids and phase_thresholds must have matching shapes.")
    if elapsed_steps.shape != progress_delta_m.shape or elapsed_steps.shape[0] != phase_action_ids.shape[0]:
        raise ValueError("phase control state tensors must match the phase batch size.")
    action_controls = action_controls.contiguous()
    phase_action_ids = phase_action_ids.contiguous()
    phase_thresholds = phase_thresholds.contiguous()
    elapsed_steps = elapsed_steps.contiguous()
    progress_delta_m = progress_delta_m.contiguous()
    throttle = torch.empty_like(progress_delta_m)
    brake = torch.empty_like(progress_delta_m)
    steer = torch.empty_like(progress_delta_m)
    action_id = torch.empty_like(elapsed_steps)
    wp = require_warp()
    wp.launch(
        _phase_controls_kernel(),
        dim=int(progress_delta_m.numel()),
        inputs=[
            wp.from_torch(action_controls),
            wp.from_torch(phase_action_ids),
            wp.from_torch(phase_thresholds),
            wp.from_torch(elapsed_steps),
            wp.from_torch(progress_delta_m),
            wp.from_torch(throttle),
            wp.from_torch(brake),
            wp.from_torch(steer),
            wp.from_torch(action_id),
            int(phase_action_ids.shape[1]),
            bool(use_progress),
        ],
        device=_warp_device_for_torch(action_controls.device),
    )
    return throttle, brake, steer, action_id


@lru_cache(maxsize=1)
def _search_feature_assembly_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _assemble_search_features_kernel(
        x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        monotonic_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        last_throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        last_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        last_steer: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        signed_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        heading_error_rad_in: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_xy: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_vec: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_len: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_cumdist_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        braking_gates_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        lookahead_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        segment_start_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        feature_ids: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        features_out: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        heading_error_deg_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        target_speed_kph_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        near_target_speed_kph_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        min_future_target_speed_kph_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        target_speed_drop_kph_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        braking_gate_distance_m_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        speed_kph_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        future_brake_demand_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        target_speed_drop_norm_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_demand_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_gate_proximity_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        brake_gate_distance_norm_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        lookahead_abs_max_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        target_steer_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_segment_count: int,
        braking_gate_count: int,
        lookahead_count: int,
        feature_count: int,
        length_px: float,
        length_m: float,
        meters_per_pixel: float,
        segment_target_progress_m: float,
        speed_target_min_kph: float,
        speed_target_max_kph: float,
        speed_target_heading_scale: float,
        speed_target_deadzone_kph: float,
        max_speed_kph: float,
        max_steer_rad: float,
    ) -> None:
        row = wp.tid()
        pi = 3.141592653589793
        two_pi = 6.283185307179586
        inv_mpp = 1.0 / wp.max(meters_per_pixel, 1.0e-6)
        speed_kph = speed_mps[row] * 3.6

        lookahead_abs_max = float(0.0)
        lookahead_0 = float(0.0)
        lookahead_1 = float(0.0)
        lookahead_2 = float(0.0)
        lookahead_3 = float(0.0)
        near_target_speed_kph = float(0.0)
        min_lookahead_target_speed_kph = float(3.4028234663852886e38)

        for lookahead_index in range(lookahead_count):
            target_px = raw_progress_m[row] * inv_mpp + lookahead_m[lookahead_index] * inv_mpp
            wrapped = target_px - length_px * wp.floor(target_px / length_px)
            sample_idx = wp.int64(0)
            for segment_index in range(centerline_segment_count):
                if centerline_cumdist_px[segment_index] <= wrapped:
                    sample_idx = wp.int64(segment_index)
            vx = centerline_vec[sample_idx, 0]
            vy = centerline_vec[sample_idx, 1]
            tangent = wp.atan2(-vy, vx)
            heading_error = tangent - heading_rad[row] + pi
            heading_error = heading_error - two_pi * wp.floor(heading_error / two_pi) - pi
            lookahead_error = heading_error / pi
            abs_error = wp.abs(lookahead_error)
            lookahead_abs_max = wp.max(lookahead_abs_max, abs_error)
            target_speed_for_lookahead = wp.clamp(
                speed_target_max_kph - speed_target_heading_scale * abs_error * 180.0,
                speed_target_min_kph,
                max_speed_kph,
            )
            if lookahead_index == 0:
                lookahead_0 = lookahead_error
                near_target_speed_kph = target_speed_for_lookahead
            elif lookahead_index == 1:
                lookahead_1 = lookahead_error
            elif lookahead_index == 2:
                lookahead_2 = lookahead_error
            elif lookahead_index == 3:
                lookahead_3 = lookahead_error
            min_lookahead_target_speed_kph = wp.min(min_lookahead_target_speed_kph, target_speed_for_lookahead)

        target_speed_kph = wp.max(
            speed_target_min_kph,
            speed_target_max_kph - speed_target_heading_scale * lookahead_abs_max * 180.0,
        )
        if lookahead_count == 0:
            near_target_speed_kph = target_speed_kph
            min_lookahead_target_speed_kph = target_speed_kph
        min_future_target_speed_kph = wp.min(target_speed_kph, min_lookahead_target_speed_kph)
        target_speed_drop_kph = wp.max(near_target_speed_kph - min_future_target_speed_kph, 0.0)
        future_brake_demand = wp.clamp(
            (speed_kph - min_future_target_speed_kph - speed_target_deadzone_kph) / 220.0,
            0.0,
            1.0,
        )

        lap_progress_m = monotonic_progress_m[row] - length_m * wp.floor(monotonic_progress_m[row] / length_m)
        braking_gate_distance_m = length_m
        for gate_index in range(braking_gate_count):
            gate = braking_gates_m[gate_index]
            gate_delta = gate - lap_progress_m
            distance = gate_delta
            if gate_delta < 0.0:
                distance = length_m - lap_progress_m + gate
            if wp.abs(gate_delta) <= 1.0e-3:
                distance = 0.0
            braking_gate_distance_m = wp.min(braking_gate_distance_m, distance)

        brake_gate_proximity = 1.0 - wp.clamp(braking_gate_distance_m / 900.0, 0.0, 1.0)
        speed_error_norm = wp.clamp((speed_kph - target_speed_kph) / 220.0, -1.0, 1.0)
        brake_demand = wp.clamp((speed_kph - target_speed_kph - speed_target_deadzone_kph) / 220.0, 0.0, 1.0)
        segment_span_m = wp.max(segment_target_progress_m - segment_start_progress_m[row], 1.0e-6)
        segment_progress_ratio = wp.clamp(
            (monotonic_progress_m[row] - segment_start_progress_m[row]) / segment_span_m,
            0.0,
            1.0,
        )
        curvature = yaw_rate_rps[row] / wp.max(speed_mps[row], 1.0e-6)

        target_lookahead_m = wp.clamp(35.0 + speed_kph * 0.28, 45.0, 145.0)
        target_px = raw_progress_m[row] * inv_mpp + target_lookahead_m * inv_mpp
        wrapped = target_px - length_px * wp.floor(target_px / length_px)
        sample_idx = wp.int64(0)
        for segment_index in range(centerline_segment_count):
            if centerline_cumdist_px[segment_index] <= wrapped:
                sample_idx = wp.int64(segment_index)
        sx = centerline_xy[sample_idx, 0]
        sy = centerline_xy[sample_idx, 1]
        vx = centerline_vec[sample_idx, 0]
        vy = centerline_vec[sample_idx, 1]
        seg_len = wp.max(centerline_len[sample_idx], 1.0e-6)
        t = (wrapped - centerline_cumdist_px[sample_idx]) / seg_len
        target_x = sx + vx * t
        target_y = sy + vy * t
        desired = wp.atan2(-(target_y - y[row]), target_x - x[row])
        target_heading_error = desired - heading_rad[row] + pi
        target_heading_error = target_heading_error - two_pi * wp.floor(target_heading_error / two_pi) - pi
        target_steer = wp.clamp(target_heading_error / wp.max(max_steer_rad, 1.0e-6), -1.0, 1.0)

        target_speed_drop_norm = wp.clamp(target_speed_drop_kph / 180.0, 0.0, 1.0)
        brake_gate_distance_norm = wp.clamp(braking_gate_distance_m / 1000.0, 0.0, 1.0) * 2.0 - 1.0
        lookahead_abs_max_diag = wp.clamp(lookahead_abs_max / pi, 0.0, 1.0)

        heading_error_deg_out[row] = heading_error_rad_in[row] * (180.0 / pi)
        target_speed_kph_out[row] = target_speed_kph
        near_target_speed_kph_out[row] = near_target_speed_kph
        min_future_target_speed_kph_out[row] = min_future_target_speed_kph
        target_speed_drop_kph_out[row] = target_speed_drop_kph
        braking_gate_distance_m_out[row] = braking_gate_distance_m
        speed_kph_out[row] = speed_kph
        future_brake_demand_out[row] = future_brake_demand
        target_speed_drop_norm_out[row] = target_speed_drop_norm
        brake_demand_out[row] = brake_demand
        brake_gate_proximity_out[row] = brake_gate_proximity
        brake_gate_distance_norm_out[row] = brake_gate_distance_norm
        lookahead_abs_max_out[row] = lookahead_abs_max_diag
        target_steer_out[row] = target_steer

        for feature_index in range(feature_count):
            feature_id = feature_ids[feature_index]
            value = float(0.0)
            if feature_id == wp.int64(0):
                value = 1.0
            elif feature_id == wp.int64(1):
                value = wp.clamp(speed_kph / max_speed_kph, 0.0, 1.0)
            elif feature_id == wp.int64(2):
                value = wp.clamp(target_speed_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0
            elif feature_id == wp.int64(3):
                value = speed_error_norm
            elif feature_id == wp.int64(4):
                value = brake_demand
            elif feature_id == wp.int64(5):
                value = future_brake_demand
            elif feature_id == wp.int64(6):
                value = target_speed_drop_norm
            elif feature_id == wp.int64(7):
                value = brake_gate_proximity
            elif feature_id == wp.int64(8):
                value = brake_gate_distance_norm
            elif feature_id == wp.int64(9):
                value = lookahead_abs_max_diag
            elif feature_id == wp.int64(10):
                value = wp.clamp(signed_lateral_error_m[row] / 30.0, -1.0, 1.0)
            elif feature_id == wp.int64(11):
                value = wp.clamp(heading_error_rad_in[row] / pi, -1.0, 1.0)
            elif feature_id == wp.int64(12):
                value = wp.clamp(yaw_rate_rps[row] / 2.0, -1.0, 1.0)
            elif feature_id == wp.int64(13):
                value = wp.clamp(curvature / 0.08, -1.0, 1.0)
            elif feature_id == wp.int64(14):
                value = target_steer
            elif feature_id == wp.int64(15):
                value = wp.clamp(last_throttle[row], 0.0, 1.0)
            elif feature_id == wp.int64(16):
                value = wp.clamp(last_brake[row], 0.0, 1.0)
            elif feature_id == wp.int64(17):
                value = wp.clamp(last_steer[row], -1.0, 1.0)
            elif feature_id == wp.int64(18):
                value = segment_progress_ratio * 2.0 - 1.0
            elif feature_id == wp.int64(19):
                value = lookahead_0
            elif feature_id == wp.int64(20):
                value = lookahead_1
            elif feature_id == wp.int64(21):
                value = lookahead_2
            elif feature_id == wp.int64(22):
                value = lookahead_3
            features_out[row, feature_index] = value

    return _assemble_search_features_kernel


def search_features_warp_batch(
    state: GpuCarBatch,
    track: GpuTrackTensors,
    config: SimConfig,
    *,
    feature_names: tuple[str, ...],
    segment_start_progress_m: torch.Tensor,
    segment_target_progress_m: float,
    braking_gates_m: torch.Tensor | None = None,
    lookahead_m: torch.Tensor | None = None,
    local_projection_window_px: torch.Tensor | None = None,
    previous_segment_idx: torch.Tensor | None = None,
    zero_feature: torch.Tensor | None = None,
    feature_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fused-backend controller features using Warp local projection."""

    if state.device.type != "cuda" or state.dtype != torch.float32:
        raise ValueError("Warp search features currently require CUDA float32 state tensors.")
    if track.device != state.device or track.dtype != state.dtype:
        raise ValueError("Warp search feature track tensors must match state device and dtype.")
    if previous_segment_idx is None:
        raise ValueError("Warp search features require cached previous segment indices.")
    if segment_start_progress_m.shape != state.speed_mps.shape:
        raise ValueError("segment_start_progress_m must have shape [batch].")
    window_px = (
        local_projection_window_px
        if local_projection_window_px is not None
        else float(config.local_projection_window_m) / torch.clamp(track.meters_per_pixel, min=1e-6)
    )
    errors = track_errors_warp_local_batch(
        state.position_xy(),
        state.heading_rad,
        track,
        previous_progress_px=state.last_raw_progress_px,
        previous_segment_idx=previous_segment_idx,
        window_px=window_px,
    )
    gates_tensor = braking_gates_m
    if gates_tensor is None:
        gates_tensor = braking_gate_tensor(device=state.device, dtype=state.dtype)
    gates_tensor = gates_tensor.to(device=state.device, dtype=torch.float32).contiguous()
    lookahead_tensor = (
        lookahead_m.to(device=state.device, dtype=torch.float32)
        if lookahead_m is not None
        else torch.tensor(tuple(float(value) for value in config.lookahead_m), device=state.device, dtype=torch.float32)
    ).contiguous()
    segment_start = segment_start_progress_m.to(device=state.device, dtype=torch.float32).contiguous()
    if feature_ids is None:
        feature_ids_tensor = controller_feature_ids_tensor(feature_names, device=state.device)
    else:
        feature_ids_tensor = feature_ids.to(device=state.device, dtype=torch.int64).contiguous()
    if int(feature_ids_tensor.numel()) != len(feature_names):
        raise ValueError("feature_ids must match feature_names length.")

    batch_size = int(state.size)
    feature_count = int(feature_ids_tensor.numel())
    features = torch.empty((batch_size, feature_count), device=state.device, dtype=torch.float32)
    heading_error_deg = torch.empty_like(state.speed_mps)
    target_speed_kph = torch.empty_like(state.speed_mps)
    near_target_speed_kph = torch.empty_like(state.speed_mps)
    min_future_target_speed_kph = torch.empty_like(state.speed_mps)
    target_speed_drop_kph = torch.empty_like(state.speed_mps)
    braking_gate_distance_m = torch.empty_like(state.speed_mps)
    speed_kph = torch.empty_like(state.speed_mps)
    future_brake_demand = torch.empty_like(state.speed_mps)
    target_speed_drop_norm = torch.empty_like(state.speed_mps)
    brake_demand = torch.empty_like(state.speed_mps)
    brake_gate_proximity = torch.empty_like(state.speed_mps)
    brake_gate_distance_norm = torch.empty_like(state.speed_mps)
    lookahead_abs_max = torch.empty_like(state.speed_mps)
    target_steer = torch.empty_like(state.speed_mps)

    wp = require_warp()
    wp.launch(
        _search_feature_assembly_kernel(),
        dim=batch_size,
        inputs=[
            wp.from_torch(state.x.contiguous()),
            wp.from_torch(state.y.contiguous()),
            wp.from_torch(state.heading_rad.contiguous()),
            wp.from_torch(state.speed_mps.contiguous()),
            wp.from_torch(state.yaw_rate_rps.contiguous()),
            wp.from_torch(state.raw_progress_m.contiguous()),
            wp.from_torch(state.monotonic_progress_m.contiguous()),
            wp.from_torch(state.last_throttle.contiguous()),
            wp.from_torch(state.last_brake.contiguous()),
            wp.from_torch(state.last_steer.contiguous()),
            wp.from_torch(errors["signed_lateral_error_m"]),
            wp.from_torch(errors["heading_error_rad"]),
            wp.from_torch(track.centerline_xy),
            wp.from_torch(track.centerline_segment_vec),
            wp.from_torch(track.centerline_segment_len),
            wp.from_torch(track.centerline_cumdist_px),
            wp.from_torch(gates_tensor),
            wp.from_torch(lookahead_tensor),
            wp.from_torch(segment_start),
            wp.from_torch(feature_ids_tensor),
            wp.from_torch(features),
            wp.from_torch(heading_error_deg),
            wp.from_torch(target_speed_kph),
            wp.from_torch(near_target_speed_kph),
            wp.from_torch(min_future_target_speed_kph),
            wp.from_torch(target_speed_drop_kph),
            wp.from_torch(braking_gate_distance_m),
            wp.from_torch(speed_kph),
            wp.from_torch(future_brake_demand),
            wp.from_torch(target_speed_drop_norm),
            wp.from_torch(brake_demand),
            wp.from_torch(brake_gate_proximity),
            wp.from_torch(brake_gate_distance_norm),
            wp.from_torch(lookahead_abs_max),
            wp.from_torch(target_steer),
            int(track.centerline_xy.shape[0]),
            int(gates_tensor.numel()),
            int(lookahead_tensor.numel()),
            feature_count,
            float(track.length_px.detach().cpu().item()),
            float(track.length_m.detach().cpu().item()),
            float(track.meters_per_pixel.detach().cpu().item()),
            float(segment_target_progress_m),
            float(config.reward.speed_target_min_kph),
            float(config.reward.speed_target_max_kph),
            float(config.reward.speed_target_heading_scale),
            float(config.reward.speed_target_deadzone_kph),
            max(float(config.car.max_speed_mps * 3.6), 1e-6),
            max(float(math.radians(config.car.max_steer_deg)), 1e-6),
        ],
        device=_warp_device_for_torch(state.device),
    )
    diagnostics: dict[str, torch.Tensor] = {
        "raw_progress_px": errors["raw_progress_px"],
        "lateral_error_m": errors["lateral_error_m"],
        "signed_lateral_error_m": errors["signed_lateral_error_m"],
        "heading_error_rad": errors["heading_error_rad"],
        "segment_idx": errors["segment_idx"],
        "heading_error_deg": heading_error_deg,
        "target_speed_kph": target_speed_kph,
        "near_target_speed_kph": near_target_speed_kph,
        "min_future_target_speed_kph": min_future_target_speed_kph,
        "target_speed_drop_kph": target_speed_drop_kph,
        "braking_gate_distance_m": braking_gate_distance_m,
        "speed_kph": speed_kph,
        "future_brake_demand": future_brake_demand,
        "target_speed_drop_norm": target_speed_drop_norm,
        "brake_demand": brake_demand,
        "brake_gate_proximity": brake_gate_proximity,
        "brake_gate_distance_norm": brake_gate_distance_norm,
        "lookahead_abs_max": lookahead_abs_max,
        "target_steer": target_steer,
    }
    return features, diagnostics


@lru_cache(maxsize=1)
def _persistent_controller_open_rollout_kernel() -> Any:
    wp = require_warp()

    @wp.kernel
    def _persistent_controller_open_kernel(
        controller_weights: wp.array3d(dtype=wp.float32),  # type: ignore[valid-type]
        feature_ids: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        lookahead_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        braking_gates_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_xy: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_vec: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_len: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_len2: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_cumdist_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        centerline_mid_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        local_projection_indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
        boundary_segments: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        boundary_grid_indices: wp.array2d(dtype=wp.int64),  # type: ignore[valid-type]
        boundary_grid_counts: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        finish_line: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
        drivable_mask: wp.array2d(dtype=wp.bool),  # type: ignore[valid-type]
        last_segment_idx: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        final_action_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        sim_steps_per_row: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_heading_rad: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_monotonic_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_raw_progress_px: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_checkpoint_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_next_checkpoint_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_checkpoints_passed: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_lap_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_elapsed_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_no_progress_steps: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_alive: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_terminated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_truncated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        state_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_segment_release_observed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        state_last_throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        state_last_steer: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_start_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_start_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_start_speed_for_score_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_best_heading_error_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_x: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_y: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_heading_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_speed_mps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_raw_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_progress_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_lateral_error_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_heading_error_deg: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_yaw_rate_rps: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_curvature_rad_per_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_steering: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_throttle: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_step_index: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_sim_time_s: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_missed_checkpoint_count: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_valid_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_finish_crossed: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_completed_lap: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_segment_complete: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_collided: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_off_track: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_terminated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_truncated: wp.array(dtype=wp.bool),  # type: ignore[valid-type]
        acc_final_termination_reason_id: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
        acc_final_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_near_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_min_future_target_speed_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_target_speed_drop_kph: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_target_speed_drop_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_gate_proximity: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_braking_gate_distance_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_brake_gate_distance_norm: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_final_lookahead_abs_max: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_time_to_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_time_to_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_sum_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_count_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_speed_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_sum_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_count_first_300_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_sum_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_brake_count_first_450_m: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_max_brake: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_row_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_brake_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_throttle_sum: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_count: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        acc_demand_max_future_brake_demand: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        max_steps: int,
        feature_count: int,
        lookahead_count: int,
        braking_gate_count: int,
        centerline_segment_count: int,
        local_count: int,
        boundary_grid_width: int,
        boundary_grid_height: int,
        max_segments_per_cell: int,
        grid_query_span: int,
        drivable_width: int,
        drivable_height: int,
        collision_check: bool,
        length_px: float,
        meters_per_pixel: float,
        track_length_m: float,
        local_projection_window_m: float,
        local_projection_window_px: float,
        boundary_grid_cell_size_px: float,
        checkpoint_spacing_m: float,
        checkpoint_lateral_limit_m: float,
        checkpoint_count: int,
        no_progress_limit_steps: int,
        target_progress_m: float,
        wheelbase_m: float,
        max_steer_rad: float,
        steer_response: float,
        engine_accel_mps2: float,
        brake_accel_mps2: float,
        drag_coefficient: float,
        rolling_resistance_mps2: float,
        grip_g: float,
        aero_grip_per_mps2: float,
        max_grip_g: float,
        max_drive_g: float,
        max_brake_g: float,
        steering_speed_sensitivity: float,
        max_speed_mps: float,
        dt: float,
        physics_model_v2: bool,
        mass: float,
        v2_front_weight_distribution: float,
        v2_cg_height_m: float,
        v2_track_width_m: float,
        v2_front_axle_distance_m: float,
        v2_rear_axle_distance_m: float,
        v2_front_cornering_stiffness_n_per_rad: float,
        v2_rear_cornering_stiffness_n_per_rad: float,
        v2_front_peak_mu: float,
        v2_rear_peak_mu: float,
        v2_mechanical_grip_low_speed_scale: float,
        v2_mechanical_grip_high_speed_scale: float,
        v2_mechanical_grip_transition_mps: float,
        v2_tire_shape_c: float,
        v2_slip_angle_peak_rad: float,
        v2_rear_slip_steer_coupling: float,
        v2_post_peak_falloff: float,
        v2_load_sensitivity: float,
        v2_aero_downforce_n_per_mps2: float,
        v2_aero_balance_front: float,
        v2_engine_power_w: float,
        v2_drivetrain_efficiency: float,
        v2_power_min_speed_mps: float,
        v2_max_drive_g: float,
        v2_max_brake_g: float,
        v2_brake_lock_threshold: float,
        v2_brake_lock_min_speed_mps: float,
        v2_brake_lock_steer_loss: float,
        v2_drag_coefficient: float,
        v2_rolling_resistance_mps2: float,
        v2_max_speed_mps: float,
        v2_gear_ratio_1: float,
        v2_gear_ratio_2: float,
        v2_gear_ratio_3: float,
        v2_gear_ratio_4: float,
        v2_gear_ratio_5: float,
        v2_gear_ratio_6: float,
        v2_gear_ratio_7: float,
        v2_gear_ratio_8: float,
        v2_final_drive_ratio: float,
        v2_wheel_radius_m: float,
        v2_idle_rpm: float,
        v2_shift_up_rpm: float,
        v2_max_rpm: float,
        v2_torque_peak_rpm: float,
        v2_torque_low_rpm_factor: float,
        v2_torque_high_rpm_factor: float,
        v2_tire_scrub_drag: float,
        v2_surface_mu: float,
        speed_target_min_kph: float,
        speed_target_max_kph: float,
        speed_target_heading_scale: float,
        speed_target_deadzone_kph: float,
        launch_guard_progress_m: float,
        launch_guard_min_speed_kph: float,
        launch_guard_throttle: float,
        lap_complete_id: int,
        collision_id: int,
        off_track_id: int,
        no_progress_id: int,
        max_steps_id: int,
    ) -> None:
        row = wp.tid()
        pi = 3.141592653589793
        two_pi = 6.283185307179586
        max_speed_kph = wp.max(max_speed_mps * 3.6, 1.0e-6)
        inv_mpp = 1.0 / wp.max(meters_per_pixel, 1.0e-6)
        row_steps = wp.int64(0)

        for _step_index in range(max_steps):
            is_active = state_alive[row] and (not state_terminated[row]) and (not state_truncated[row])
            if is_active:
                px = state_x[row]
                py = state_y[row]
                heading = state_heading_rad[row]
                prev = state_last_raw_progress_px[row]
                prev = prev - length_px * wp.floor(prev / length_px)
                base_idx = last_segment_idx[row]
                max_segment_idx = wp.int64(centerline_segment_count - 1)
                if base_idx < wp.int64(0):
                    base_idx = wp.int64(0)
                if base_idx > max_segment_idx:
                    base_idx = max_segment_idx

                has_local = bool(False)
                for candidate_offset in range(local_count):
                    segment_idx = local_projection_indices[base_idx, candidate_offset]
                    if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                        continue
                    len2 = centerline_len2[segment_idx]
                    if len2 <= 1.0e-9:
                        continue
                    mid = centerline_mid_px[segment_idx]
                    wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
                    if wrapped_delta <= local_projection_window_px:
                        has_local = True

                best_score = float(3.4028234663852886e38)
                control_segment_idx = wp.int64(0)
                control_lateral_px = float(0.0)
                control_projection_x = float(0.0)
                control_projection_y = float(0.0)
                control_tangent = float(0.0)

                for candidate_offset in range(local_count):
                    segment_idx = local_projection_indices[base_idx, candidate_offset]
                    if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                        continue
                    len2 = centerline_len2[segment_idx]
                    if len2 <= 1.0e-9:
                        continue
                    mid = centerline_mid_px[segment_idx]
                    wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
                    if has_local and wrapped_delta > local_projection_window_px:
                        continue
                    sx = centerline_xy[segment_idx, 0]
                    sy = centerline_xy[segment_idx, 1]
                    vx = centerline_vec[segment_idx, 0]
                    vy = centerline_vec[segment_idx, 1]
                    rel_x = px - sx
                    rel_y = py - sy
                    t = wp.clamp((rel_x * vx + rel_y * vy) / len2, 0.0, 1.0)
                    projection_x = sx + t * vx
                    projection_y = sy + t * vy
                    delta_x = px - projection_x
                    delta_y = py - projection_y
                    lateral_px = wp.sqrt(delta_x * delta_x + delta_y * delta_y)
                    raw_progress_px = centerline_cumdist_px[segment_idx] + centerline_len[segment_idx] * t
                    signed_delta = (raw_progress_px - prev + length_px * 0.5) - length_px * wp.floor((raw_progress_px - prev + length_px * 0.5) / length_px) - length_px * 0.5
                    score = lateral_px + wp.max(-signed_delta, 0.0) * 0.05
                    if score < best_score:
                        best_score = score
                        control_segment_idx = segment_idx
                        control_lateral_px = lateral_px
                        control_projection_x = projection_x
                        control_projection_y = projection_y
                        control_tangent = wp.atan2(-vy, vx)

                offset_x = px - control_projection_x
                offset_y = py - control_projection_y
                tangent_x = wp.cos(control_tangent)
                tangent_y = -wp.sin(control_tangent)
                cross = tangent_x * offset_y - tangent_y * offset_x
                control_signed_lateral_px = float(0.0)
                if control_lateral_px > 1.0e-6 and wp.abs(cross) > 1.0e-9:
                    if cross > 0.0:
                        control_signed_lateral_px = control_lateral_px
                    else:
                        control_signed_lateral_px = -control_lateral_px
                control_signed_lateral_m = control_signed_lateral_px * meters_per_pixel
                control_heading_error_rad = control_tangent - heading + pi
                control_heading_error_rad = control_heading_error_rad - two_pi * wp.floor(control_heading_error_rad / two_pi) - pi
                last_segment_idx[row] = control_segment_idx

                speed_kph_before = state_speed_mps[row] * 3.6
                lookahead_abs_max = float(0.0)
                lookahead_0 = float(0.0)
                lookahead_1 = float(0.0)
                lookahead_2 = float(0.0)
                lookahead_3 = float(0.0)
                near_target_speed_kph = float(0.0)
                min_lookahead_target_speed_kph = float(3.4028234663852886e38)
                for lookahead_index in range(lookahead_count):
                    target_px = state_raw_progress_m[row] * inv_mpp + lookahead_m[lookahead_index] * inv_mpp
                    wrapped = target_px - length_px * wp.floor(target_px / length_px)
                    sample_idx = wp.int64(0)
                    for segment_index in range(centerline_segment_count):
                        if centerline_cumdist_px[segment_index] <= wrapped:
                            sample_idx = wp.int64(segment_index)
                    vx = centerline_vec[sample_idx, 0]
                    vy = centerline_vec[sample_idx, 1]
                    tangent = wp.atan2(-vy, vx)
                    lookahead_heading_error = tangent - heading + pi
                    lookahead_heading_error = lookahead_heading_error - two_pi * wp.floor(lookahead_heading_error / two_pi) - pi
                    lookahead_error = lookahead_heading_error / pi
                    abs_error = wp.abs(lookahead_error)
                    lookahead_abs_max = wp.max(lookahead_abs_max, abs_error)
                    target_speed_for_lookahead = wp.clamp(
                        speed_target_max_kph - speed_target_heading_scale * abs_error * 180.0,
                        speed_target_min_kph,
                        max_speed_kph,
                    )
                    if lookahead_index == 0:
                        lookahead_0 = lookahead_error
                        near_target_speed_kph = target_speed_for_lookahead
                    elif lookahead_index == 1:
                        lookahead_1 = lookahead_error
                    elif lookahead_index == 2:
                        lookahead_2 = lookahead_error
                    elif lookahead_index == 3:
                        lookahead_3 = lookahead_error
                    min_lookahead_target_speed_kph = wp.min(min_lookahead_target_speed_kph, target_speed_for_lookahead)

                target_speed_kph = wp.max(
                    speed_target_min_kph,
                    speed_target_max_kph - speed_target_heading_scale * lookahead_abs_max * 180.0,
                )
                if lookahead_count == 0:
                    near_target_speed_kph = target_speed_kph
                    min_lookahead_target_speed_kph = target_speed_kph
                min_future_target_speed_kph = wp.min(target_speed_kph, min_lookahead_target_speed_kph)
                target_speed_drop_kph = wp.max(near_target_speed_kph - min_future_target_speed_kph, 0.0)
                future_brake_demand = wp.clamp(
                    (speed_kph_before - min_future_target_speed_kph - speed_target_deadzone_kph) / 220.0,
                    0.0,
                    1.0,
                )
                lap_progress_m = state_monotonic_progress_m[row] - track_length_m * wp.floor(state_monotonic_progress_m[row] / track_length_m)
                braking_gate_distance_m = track_length_m
                for gate_index in range(braking_gate_count):
                    gate = braking_gates_m[gate_index]
                    gate_delta = gate - lap_progress_m
                    distance = gate_delta
                    if gate_delta < 0.0:
                        distance = track_length_m - lap_progress_m + gate
                    if wp.abs(gate_delta) <= 1.0e-3:
                        distance = 0.0
                    braking_gate_distance_m = wp.min(braking_gate_distance_m, distance)
                brake_gate_proximity = 1.0 - wp.clamp(braking_gate_distance_m / 900.0, 0.0, 1.0)
                speed_error_norm = wp.clamp((speed_kph_before - target_speed_kph) / 220.0, -1.0, 1.0)
                brake_demand = wp.clamp((speed_kph_before - target_speed_kph - speed_target_deadzone_kph) / 220.0, 0.0, 1.0)
                segment_span_m = wp.max(target_progress_m - acc_start_progress_m[row], 1.0e-6)
                segment_progress_ratio = wp.clamp(
                    (state_monotonic_progress_m[row] - acc_start_progress_m[row]) / segment_span_m,
                    0.0,
                    1.0,
                )
                curvature = state_yaw_rate_rps[row] / wp.max(state_speed_mps[row], 1.0e-6)
                target_lookahead_m = wp.clamp(35.0 + speed_kph_before * 0.28, 45.0, 145.0)
                target_px = state_raw_progress_m[row] * inv_mpp + target_lookahead_m * inv_mpp
                wrapped = target_px - length_px * wp.floor(target_px / length_px)
                sample_idx = wp.int64(0)
                for segment_index in range(centerline_segment_count):
                    if centerline_cumdist_px[segment_index] <= wrapped:
                        sample_idx = wp.int64(segment_index)
                sx = centerline_xy[sample_idx, 0]
                sy = centerline_xy[sample_idx, 1]
                vx = centerline_vec[sample_idx, 0]
                vy = centerline_vec[sample_idx, 1]
                seg_len = wp.max(centerline_len[sample_idx], 1.0e-6)
                sample_t = (wrapped - centerline_cumdist_px[sample_idx]) / seg_len
                target_x = sx + vx * sample_t
                target_y = sy + vy * sample_t
                desired = wp.atan2(-(target_y - state_y[row]), target_x - state_x[row])
                target_heading_error = desired - heading + pi
                target_heading_error = target_heading_error - two_pi * wp.floor(target_heading_error / two_pi) - pi
                target_steer = wp.clamp(target_heading_error / wp.max(max_steer_rad, 1.0e-6), -1.0, 1.0)
                target_speed_drop_norm = wp.clamp(target_speed_drop_kph / 180.0, 0.0, 1.0)
                brake_gate_distance_norm = wp.clamp(braking_gate_distance_m / 1000.0, 0.0, 1.0) * 2.0 - 1.0
                lookahead_abs_max_diag = wp.clamp(lookahead_abs_max / pi, 0.0, 1.0)
                control_heading_error_deg = control_heading_error_rad * (180.0 / pi)

                throttle_logit = float(0.0)
                brake_logit = float(0.0)
                steer_logit = float(0.0)
                for feature_index in range(feature_count):
                    feature_id = feature_ids[feature_index]
                    feature_value = float(0.0)
                    if feature_id == wp.int64(0):
                        feature_value = 1.0
                    elif feature_id == wp.int64(1):
                        feature_value = wp.clamp(speed_kph_before / max_speed_kph, 0.0, 1.0)
                    elif feature_id == wp.int64(2):
                        feature_value = wp.clamp(target_speed_kph / max_speed_kph, 0.0, 1.0) * 2.0 - 1.0
                    elif feature_id == wp.int64(3):
                        feature_value = speed_error_norm
                    elif feature_id == wp.int64(4):
                        feature_value = brake_demand
                    elif feature_id == wp.int64(5):
                        feature_value = future_brake_demand
                    elif feature_id == wp.int64(6):
                        feature_value = target_speed_drop_norm
                    elif feature_id == wp.int64(7):
                        feature_value = brake_gate_proximity
                    elif feature_id == wp.int64(8):
                        feature_value = brake_gate_distance_norm
                    elif feature_id == wp.int64(9):
                        feature_value = lookahead_abs_max_diag
                    elif feature_id == wp.int64(10):
                        feature_value = wp.clamp(control_signed_lateral_m / 30.0, -1.0, 1.0)
                    elif feature_id == wp.int64(11):
                        feature_value = wp.clamp(control_heading_error_rad / pi, -1.0, 1.0)
                    elif feature_id == wp.int64(12):
                        feature_value = wp.clamp(state_yaw_rate_rps[row] / 2.0, -1.0, 1.0)
                    elif feature_id == wp.int64(13):
                        feature_value = wp.clamp(curvature / 0.08, -1.0, 1.0)
                    elif feature_id == wp.int64(14):
                        feature_value = target_steer
                    elif feature_id == wp.int64(15):
                        feature_value = wp.clamp(state_last_throttle[row], 0.0, 1.0)
                    elif feature_id == wp.int64(16):
                        feature_value = wp.clamp(state_last_brake[row], 0.0, 1.0)
                    elif feature_id == wp.int64(17):
                        feature_value = wp.clamp(state_last_steer[row], -1.0, 1.0)
                    elif feature_id == wp.int64(18):
                        feature_value = segment_progress_ratio * 2.0 - 1.0
                    elif feature_id == wp.int64(19):
                        feature_value = lookahead_0
                    elif feature_id == wp.int64(20):
                        feature_value = lookahead_1
                    elif feature_id == wp.int64(21):
                        feature_value = lookahead_2
                    elif feature_id == wp.int64(22):
                        feature_value = lookahead_3
                    throttle_logit = throttle_logit + controller_weights[row, 0, feature_index] * feature_value
                    brake_logit = brake_logit + controller_weights[row, 1, feature_index] * feature_value
                    steer_logit = steer_logit + controller_weights[row, 2, feature_index] * feature_value

                throttle_raw = 1.0 / (1.0 + wp.exp(-wp.clamp(throttle_logit, -40.0, 40.0)))
                brake_raw = 1.0 / (1.0 + wp.exp(-wp.clamp(brake_logit, -40.0, 40.0)))
                throttle = throttle_raw
                brake = brake_raw * (1.0 - throttle_raw)
                if brake_raw + wp.float32(1.0e-3) >= throttle_raw:
                    throttle = throttle_raw * (1.0 - brake_raw)
                    brake = brake_raw
                throttle = wp.clamp(throttle, 0.0, 1.0)
                brake = wp.clamp(brake, 0.0, 1.0)
                steer = wp.clamp(wp.tanh(steer_logit), -1.0, 1.0)
                if (
                    launch_guard_progress_m > 0.0
                    and launch_guard_min_speed_kph > 0.0
                    and state_monotonic_progress_m[row] <= launch_guard_progress_m
                    and speed_kph_before < launch_guard_min_speed_kph
                    and throttle <= 1.0e-6
                    and brake > 1.0e-6
                ):
                    throttle = launch_guard_throttle
                    brake = 0.0

                old_x = state_x[row]
                old_y = state_y[row]
                old_progress_m = state_monotonic_progress_m[row]
                old_raw_progress_px = state_last_raw_progress_px[row]
                old_lap_index = state_lap_index[row]

                gravity = wp.float32(9.81)
                speed = wp.max(state_speed_mps[row], 0.0)
                target_steering = steer * max_steer_rad
                steering_delta = target_steering - state_steering[row]
                max_delta = steer_response * dt
                steering = state_steering[row] + wp.clamp(steering_delta, -max_delta, max_delta)
                yaw_rate = wp.float32(0.0)
                if physics_model_v2:
                    vehicle_mass = wp.float32(mass)
                    effective_steering = steering / (1.0 + steering_speed_sensitivity * speed * speed)
                    wheel_lock = brake >= v2_brake_lock_threshold and speed >= v2_brake_lock_min_speed_mps
                    if wheel_lock:
                        effective_steering = effective_steering * wp.clamp(1.0 - v2_brake_lock_steer_loss * brake, 0.0, 1.0)

                    previous_lateral_accel = wp.abs(speed * state_yaw_rate_rps[row])
                    static_front = vehicle_mass * gravity * v2_front_weight_distribution
                    static_rear = vehicle_mass * gravity - static_front
                    aero_total = v2_aero_downforce_n_per_mps2 * speed * speed
                    aero_front = aero_total * v2_aero_balance_front
                    aero_rear = aero_total - aero_front
                    lateral_unload = (
                        wp.abs(vehicle_mass * previous_lateral_accel * v2_cg_height_m / wp.max(v2_track_width_m, 1.0e-6))
                        * 0.12
                    )
                    front_load = wp.max(static_front + aero_front - lateral_unload * v2_front_weight_distribution, 1.0)
                    rear_load = wp.max(static_rear + aero_rear - lateral_unload * (1.0 - v2_front_weight_distribution), 1.0)
                    reference_front = vehicle_mass * gravity * v2_front_weight_distribution
                    reference_rear = vehicle_mass * gravity - reference_front

                    front_slip = (
                        wp.atan2(state_yaw_rate_rps[row] * v2_front_axle_distance_m, wp.max(wp.abs(speed), 1.0e-3))
                        - effective_steering
                    )
                    rear_slip = -wp.atan2(
                        state_yaw_rate_rps[row] * v2_rear_axle_distance_m,
                        wp.max(wp.abs(speed), 1.0e-3),
                    ) - v2_rear_slip_steer_coupling * effective_steering
                    grip_t = wp.clamp(speed / wp.max(v2_mechanical_grip_transition_mps, 1.0e-6), 0.0, 1.0)
                    grip_smooth_t = grip_t * grip_t * (3.0 - 2.0 * grip_t)
                    mechanical_grip_scale = v2_mechanical_grip_low_speed_scale + (
                        v2_mechanical_grip_high_speed_scale - v2_mechanical_grip_low_speed_scale
                    ) * grip_smooth_t

                    front_load_ratio = front_load / wp.max(reference_front, 1.0)
                    front_mu = v2_front_peak_mu * mechanical_grip_scale * v2_surface_mu * (
                        1.0 - v2_load_sensitivity * wp.max(front_load_ratio - 1.0, 0.0)
                    )
                    front_peak_force = wp.max(front_mu * front_load, 1.0)
                    front_b = v2_front_cornering_stiffness_n_per_rad / wp.max(v2_tire_shape_c * front_peak_force, 1.0e-6)
                    front_force = front_peak_force * wp.sin(v2_tire_shape_c * wp.atan(front_b * front_slip))
                    front_excess = wp.clamp(
                        (wp.abs(front_slip) - v2_slip_angle_peak_rad) / wp.max(v2_slip_angle_peak_rad, 1.0e-6),
                        0.0,
                        1.0,
                    )
                    front_force = front_force * (1.0 - v2_post_peak_falloff * front_excess)

                    rear_load_ratio = rear_load / wp.max(reference_rear, 1.0)
                    rear_mu = v2_rear_peak_mu * mechanical_grip_scale * v2_surface_mu * (
                        1.0 - v2_load_sensitivity * wp.max(rear_load_ratio - 1.0, 0.0)
                    )
                    rear_peak_force = wp.max(rear_mu * rear_load, 1.0)
                    rear_b = v2_rear_cornering_stiffness_n_per_rad / wp.max(v2_tire_shape_c * rear_peak_force, 1.0e-6)
                    rear_force = rear_peak_force * wp.sin(v2_tire_shape_c * wp.atan(rear_b * rear_slip))
                    rear_excess = wp.clamp(
                        (wp.abs(rear_slip) - v2_slip_angle_peak_rad) / wp.max(v2_slip_angle_peak_rad, 1.0e-6),
                        0.0,
                        1.0,
                    )
                    rear_force = rear_force * (1.0 - v2_post_peak_falloff * rear_excess)

                    lateral_capacity = (wp.abs(front_force) + wp.abs(rear_force)) / wp.max(vehicle_mass, 1.0e-6)
                    requested_yaw = speed / wp.max(wheelbase_m, 1.0e-6) * wp.tan(effective_steering)
                    requested_lateral = wp.abs(speed * requested_yaw)
                    lateral_accel = wp.min(requested_lateral, lateral_capacity)
                    if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
                        yaw_rate = wp.sign(requested_yaw) * lateral_accel / wp.max(speed, 1.0e-6)
                    else:
                        lateral_accel = wp.float32(0.0)

                    total_peak_force = (
                        v2_front_peak_mu * mechanical_grip_scale * front_load * v2_surface_mu
                        + v2_rear_peak_mu * mechanical_grip_scale * rear_load * v2_surface_mu
                    )
                    total_accel_limit = total_peak_force / wp.max(vehicle_mass, 1.0e-6)
                    longitudinal_capacity = wp.sqrt(
                        wp.max(total_accel_limit * total_accel_limit - lateral_accel * lateral_accel, 0.0)
                    )

                    wheel_rps = wp.max(speed, 0.0) / wp.max(2.0 * pi * v2_wheel_radius_m, 1.0e-6)
                    rpm = wp.clamp(wheel_rps * v2_gear_ratio_1 * v2_final_drive_ratio * 60.0, v2_idle_rpm, v2_max_rpm)
                    if rpm > v2_shift_up_rpm:
                        rpm = wp.clamp(wheel_rps * v2_gear_ratio_2 * v2_final_drive_ratio * 60.0, v2_idle_rpm, v2_max_rpm)
                        if rpm > v2_shift_up_rpm:
                            rpm = wp.clamp(wheel_rps * v2_gear_ratio_3 * v2_final_drive_ratio * 60.0, v2_idle_rpm, v2_max_rpm)
                            if rpm > v2_shift_up_rpm:
                                rpm = wp.clamp(wheel_rps * v2_gear_ratio_4 * v2_final_drive_ratio * 60.0, v2_idle_rpm, v2_max_rpm)
                                if rpm > v2_shift_up_rpm:
                                    rpm = wp.clamp(
                                        wheel_rps * v2_gear_ratio_5 * v2_final_drive_ratio * 60.0,
                                        v2_idle_rpm,
                                        v2_max_rpm,
                                    )
                                    if rpm > v2_shift_up_rpm:
                                        rpm = wp.clamp(
                                            wheel_rps * v2_gear_ratio_6 * v2_final_drive_ratio * 60.0,
                                            v2_idle_rpm,
                                            v2_max_rpm,
                                        )
                                        if rpm > v2_shift_up_rpm:
                                            rpm = wp.clamp(
                                                wheel_rps * v2_gear_ratio_7 * v2_final_drive_ratio * 60.0,
                                                v2_idle_rpm,
                                                v2_max_rpm,
                                            )
                                            if rpm > v2_shift_up_rpm:
                                                rpm = wp.clamp(
                                                    wheel_rps * v2_gear_ratio_8 * v2_final_drive_ratio * 60.0,
                                                    v2_idle_rpm,
                                                    v2_max_rpm,
                                                )

                    torque_factor = wp.float32(1.0)
                    if rpm <= v2_torque_peak_rpm:
                        low_span = wp.max(v2_torque_peak_rpm - v2_idle_rpm, 1.0)
                        low_ratio = wp.clamp((rpm - v2_idle_rpm) / low_span, 0.0, 1.0)
                        torque_factor = v2_torque_low_rpm_factor + (1.0 - v2_torque_low_rpm_factor) * low_ratio
                    else:
                        high_span = wp.max(v2_max_rpm - v2_torque_peak_rpm, 1.0)
                        high_ratio = wp.clamp((rpm - v2_torque_peak_rpm) / high_span, 0.0, 1.0)
                        torque_factor = 1.0 - (1.0 - v2_torque_high_rpm_factor) * high_ratio

                    power_force = v2_engine_power_w * v2_drivetrain_efficiency * torque_factor / wp.max(
                        speed,
                        v2_power_min_speed_mps,
                    )
                    drive_limit = wp.min(
                        wp.min(v2_max_drive_g * gravity, power_force / wp.max(vehicle_mass, 1.0e-6)),
                        longitudinal_capacity,
                    )
                    brake_limit = wp.min(v2_max_brake_g * gravity, longitudinal_capacity)
                    if wheel_lock:
                        brake_limit = brake_limit * 0.82

                    longitudinal_accel = throttle * drive_limit - brake * brake_limit
                    if throttle <= 1.0e-6 and brake <= 1.0e-6:
                        longitudinal_accel = longitudinal_accel - v2_rolling_resistance_mps2
                    longitudinal_accel = longitudinal_accel - v2_drag_coefficient * speed * speed
                    tire_saturation = wp.clamp(
                        wp.max(wp.abs(front_slip), wp.abs(rear_slip)) / wp.max(v2_slip_angle_peak_rad, 1.0e-6),
                        0.0,
                        3.0,
                    )
                    longitudinal_accel = longitudinal_accel - v2_tire_scrub_drag * wp.max(tire_saturation - 1.0, 0.0)
                    speed = wp.clamp(speed + longitudinal_accel * dt, 0.0, v2_max_speed_mps)
                else:
                    effective_steering = steering / (1.0 + steering_speed_sensitivity * speed * speed)
                    grip = wp.clamp(grip_g + aero_grip_per_mps2 * speed * speed, grip_g, max_grip_g)
                    max_total_accel = grip * gravity
                    requested_lateral = wp.float32(0.0)
                    if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
                        requested_yaw = speed / wp.max(wheelbase_m, 1.0e-6) * wp.tan(effective_steering)
                        requested_lateral = wp.abs(speed * requested_yaw)
                    lateral_accel = wp.min(requested_lateral, max_total_accel)
                    longitudinal_capacity = wp.sqrt(wp.max(max_total_accel * max_total_accel - lateral_accel * lateral_accel, 0.0))
                    if longitudinal_capacity < wp.float32(1.0e-2):
                        longitudinal_capacity = wp.float32(0.0)
                    drive_static_limit = wp.min(engine_accel_mps2, max_drive_g * gravity)
                    brake_static_limit = wp.min(brake_accel_mps2, max_brake_g * gravity)
                    drive_limit = wp.min(drive_static_limit, longitudinal_capacity)
                    brake_limit = wp.min(brake_static_limit, longitudinal_capacity)
                    longitudinal_accel = throttle * drive_limit - brake * brake_limit
                    if throttle <= 1.0e-6 and brake <= 1.0e-6:
                        longitudinal_accel = longitudinal_accel - rolling_resistance_mps2
                    longitudinal_accel = longitudinal_accel - drag_coefficient * speed * speed
                    speed = wp.clamp(speed + longitudinal_accel * dt, 0.0, max_speed_mps)
                    if wp.abs(effective_steering) > 1.0e-6 and speed > 1.0e-6:
                        yaw_rate = speed / wp.max(wheelbase_m, 1.0e-6) * wp.tan(effective_steering)
                        lateral_accel_after = wp.abs(speed * yaw_rate)
                        grip_after = wp.clamp(grip_g + aero_grip_per_mps2 * speed * speed, grip_g, max_grip_g)
                        max_lateral_after = grip_after * gravity
                        if lateral_accel_after > max_lateral_after:
                            yaw_rate = yaw_rate * max_lateral_after / wp.max(lateral_accel_after, 1.0e-6)
                heading_raw = state_heading_rad[row] + yaw_rate * dt + pi
                heading = heading_raw - two_pi * wp.floor(heading_raw / two_pi) - pi
                distance_px = speed * dt * inv_mpp
                new_x = old_x + wp.cos(heading) * distance_px
                new_y = old_y - wp.sin(heading) * distance_px

                state_x[row] = new_x
                state_y[row] = new_y
                state_heading_rad[row] = heading
                state_speed_mps[row] = speed
                state_yaw_rate_rps[row] = yaw_rate
                state_steering[row] = steering
                state_elapsed_steps[row] = state_elapsed_steps[row] + wp.int64(1)
                row_steps = row_steps + wp.int64(1)
                final_action_id[row] = wp.int64(-2)
                state_last_throttle[row] = throttle
                state_last_brake[row] = brake
                state_last_steer[row] = steer

                px = new_x
                py = new_y
                prev = old_raw_progress_px
                prev = prev - length_px * wp.floor(prev / length_px)
                base_idx = last_segment_idx[row]
                if base_idx < wp.int64(0):
                    base_idx = wp.int64(0)
                if base_idx > max_segment_idx:
                    base_idx = max_segment_idx
                has_local = bool(False)
                for candidate_offset in range(local_count):
                    segment_idx = local_projection_indices[base_idx, candidate_offset]
                    if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                        continue
                    len2 = centerline_len2[segment_idx]
                    if len2 <= 1.0e-9:
                        continue
                    mid = centerline_mid_px[segment_idx]
                    wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
                    if wrapped_delta <= local_projection_window_px:
                        has_local = True

                best_score = float(3.4028234663852886e38)
                step_segment_idx = wp.int64(0)
                step_raw_px = float(0.0)
                step_lateral_px = float(0.0)
                step_tangent = float(0.0)
                for candidate_offset in range(local_count):
                    segment_idx = local_projection_indices[base_idx, candidate_offset]
                    if segment_idx < wp.int64(0) or segment_idx > max_segment_idx:
                        continue
                    len2 = centerline_len2[segment_idx]
                    if len2 <= 1.0e-9:
                        continue
                    mid = centerline_mid_px[segment_idx]
                    wrapped_delta = wp.abs((mid - prev + length_px * 0.5) - length_px * wp.floor((mid - prev + length_px * 0.5) / length_px) - length_px * 0.5)
                    if has_local and wrapped_delta > local_projection_window_px:
                        continue
                    sx = centerline_xy[segment_idx, 0]
                    sy = centerline_xy[segment_idx, 1]
                    vx = centerline_vec[segment_idx, 0]
                    vy = centerline_vec[segment_idx, 1]
                    rel_x = px - sx
                    rel_y = py - sy
                    t = wp.clamp((rel_x * vx + rel_y * vy) / len2, 0.0, 1.0)
                    projection_x = sx + t * vx
                    projection_y = sy + t * vy
                    delta_x = px - projection_x
                    delta_y = py - projection_y
                    lateral_px = wp.sqrt(delta_x * delta_x + delta_y * delta_y)
                    raw_progress_px = centerline_cumdist_px[segment_idx] + centerline_len[segment_idx] * t
                    signed_delta = (raw_progress_px - prev + length_px * 0.5) - length_px * wp.floor((raw_progress_px - prev + length_px * 0.5) / length_px) - length_px * 0.5
                    score = lateral_px + wp.max(-signed_delta, 0.0) * 0.05
                    if score < best_score:
                        best_score = score
                        step_segment_idx = segment_idx
                        step_raw_px = raw_progress_px - length_px * wp.floor(raw_progress_px / length_px)
                        step_lateral_px = lateral_px
                        step_tangent = wp.atan2(-vy, vx)
                step_heading_error_rad = step_tangent - heading + pi
                step_heading_error_rad = step_heading_error_rad - two_pi * wp.floor(step_heading_error_rad / two_pi) - pi
                step_lateral_error_m = step_lateral_px * meters_per_pixel
                last_segment_idx[row] = step_segment_idx

                raw_delta_px = step_raw_px - old_raw_progress_px
                if raw_delta_px < -0.5 * length_px:
                    raw_delta_px = raw_delta_px + length_px
                if raw_delta_px > 0.5 * length_px:
                    raw_delta_px = raw_delta_px - length_px
                progress_delta_m = wp.max(raw_delta_px * meters_per_pixel, 0.0)
                if progress_delta_m > local_projection_window_m:
                    progress_delta_m = 0.0
                state_last_raw_progress_px[row] = step_raw_px
                state_raw_progress_m[row] = step_raw_px * meters_per_pixel
                state_monotonic_progress_m[row] = old_progress_m + progress_delta_m

                skipped = progress_delta_m > checkpoint_spacing_m * 1.75
                if skipped:
                    skipped_count = wp.int64(wp.floor(progress_delta_m / checkpoint_spacing_m)) - wp.int64(1)
                    if skipped_count < wp.int64(1):
                        skipped_count = wp.int64(1)
                    state_missed_checkpoint_count[row] = state_missed_checkpoint_count[row] + skipped_count
                    state_valid_lap[row] = False
                next_idx = state_next_checkpoint_index[row]
                threshold = checkpoint_spacing_m * wp.float32(next_idx)
                due = next_idx < wp.int64(checkpoint_count) and state_monotonic_progress_m[row] + 1.0e-6 >= threshold
                expected = old_progress_m <= threshold and threshold <= state_monotonic_progress_m[row] + checkpoint_spacing_m * 0.75
                lateral_ok = wp.abs(step_lateral_error_m) <= checkpoint_lateral_limit_m
                if due:
                    if not (expected and lateral_ok):
                        state_missed_checkpoint_count[row] = state_missed_checkpoint_count[row] + wp.int64(1)
                        state_valid_lap[row] = False
                    state_checkpoints_passed[row] = next_idx
                    state_next_checkpoint_index[row] = next_idx + wp.int64(1)
                checkpoint_index = wp.int64(wp.floor(state_monotonic_progress_m[row] / checkpoint_spacing_m))
                if checkpoint_count > 0:
                    checkpoint_index = checkpoint_index % wp.int64(checkpoint_count)
                state_checkpoint_index[row] = checkpoint_index

                if progress_delta_m <= 1.0e-4:
                    state_no_progress_steps[row] = state_no_progress_steps[row] + wp.int64(1)
                else:
                    state_no_progress_steps[row] = wp.int64(0)

                collided = bool(False)
                if collision_check:
                    min_x = wp.min(old_x, new_x)
                    min_y = wp.min(old_y, new_y)
                    max_x = wp.max(old_x, new_x)
                    max_y = wp.max(old_y, new_y)
                    cell_size = wp.max(boundary_grid_cell_size_px, 1.0)
                    min_cell_x = wp.int32(wp.floor(min_x / cell_size))
                    min_cell_y = wp.int32(wp.floor(min_y / cell_size))
                    max_cell_x = wp.int32(wp.floor(max_x / cell_size))
                    max_cell_y = wp.int32(wp.floor(max_y / cell_size))
                    if min_cell_x < 0:
                        min_cell_x = 0
                    if min_cell_y < 0:
                        min_cell_y = 0
                    if max_cell_x < 0:
                        max_cell_x = 0
                    if max_cell_y < 0:
                        max_cell_y = 0
                    if min_cell_x >= boundary_grid_width:
                        min_cell_x = boundary_grid_width - 1
                    if max_cell_x >= boundary_grid_width:
                        max_cell_x = boundary_grid_width - 1
                    if min_cell_y >= boundary_grid_height:
                        min_cell_y = boundary_grid_height - 1
                    if max_cell_y >= boundary_grid_height:
                        max_cell_y = boundary_grid_height - 1
                    dx12 = new_x - old_x
                    dy12 = new_y - old_y
                    span = grid_query_span
                    if span < 1:
                        span = 1
                    for offset_y_grid in range(span):
                        cell_y = min_cell_y + offset_y_grid
                        valid_y = cell_y <= max_cell_y
                        if cell_y < 0:
                            cell_y = 0
                        if cell_y >= boundary_grid_height:
                            cell_y = boundary_grid_height - 1
                        for offset_x_grid in range(span):
                            cell_x = min_cell_x + offset_x_grid
                            valid_cell = valid_y and cell_x <= max_cell_x
                            if cell_x < 0:
                                cell_x = 0
                            if cell_x >= boundary_grid_width:
                                cell_x = boundary_grid_width - 1
                            cell_id = cell_y * boundary_grid_width + cell_x
                            count = boundary_grid_counts[cell_id]
                            for candidate_offset in range(max_segments_per_cell):
                                if not valid_cell or wp.int64(candidate_offset) >= count:
                                    continue
                                segment_id = boundary_grid_indices[cell_id, candidate_offset]
                                if segment_id < wp.int64(0):
                                    continue
                                x3 = boundary_segments[segment_id, 0]
                                y3 = boundary_segments[segment_id, 1]
                                x4 = boundary_segments[segment_id, 2]
                                y4 = boundary_segments[segment_id, 3]
                                dx34 = x4 - x3
                                dy34 = y4 - y3
                                denom = dy34 * dx12 - dx34 * dy12
                                if wp.abs(denom) >= 1.0e-9:
                                    s = (dx34 * (old_y - y3) - dy34 * (old_x - x3)) / denom
                                    t = (dx12 * (old_y - y3) - dy12 * (old_x - x3)) / denom
                                    if s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0:
                                        collided = True

                xi = wp.int32(wp.round(new_x))
                yi = wp.int32(wp.round(new_y))
                drivable = bool(False)
                if xi >= 0 and yi >= 0 and xi < drivable_width and yi < drivable_height:
                    drivable = drivable_mask[yi, xi]
                off_track = not drivable

                physical_finish = bool(False)
                x3 = finish_line[0, 0]
                y3 = finish_line[0, 1]
                x4 = finish_line[0, 2]
                y4 = finish_line[0, 3]
                dx12_finish = new_x - old_x
                dy12_finish = new_y - old_y
                dx34_finish = x4 - x3
                dy34_finish = y4 - y3
                denom_finish = dy34_finish * dx12_finish - dx34_finish * dy12_finish
                if wp.abs(denom_finish) >= 1.0e-9:
                    s_finish = (dx34_finish * (old_y - y3) - dy34_finish * (old_x - x3)) / denom_finish
                    t_finish = (dx12_finish * (old_y - y3) - dy12_finish * (old_x - x3)) / denom_finish
                    if s_finish >= 0.0 and s_finish <= 1.0 and t_finish >= 0.0 and t_finish <= 1.0:
                        physical_finish = True

                target_lap_progress_m = (wp.float32(old_lap_index) + 1.0) * track_length_m
                near_finish = old_progress_m >= target_lap_progress_m - checkpoint_spacing_m * 2.0
                virtual_finish = old_progress_m < target_lap_progress_m and target_lap_progress_m <= state_monotonic_progress_m[row]
                crossed_finish = near_finish and (physical_finish or virtual_finish)
                if crossed_finish:
                    state_finish_crossed[row] = True
                last_checkpoint_index = checkpoint_count - 1
                if last_checkpoint_index < 0:
                    last_checkpoint_index = 0
                telemetry_valid_lap = (
                    state_valid_lap[row]
                    and state_missed_checkpoint_count[row] == wp.int64(0)
                    and state_checkpoints_passed[row] >= wp.int64(last_checkpoint_index)
                )
                lap_complete = (
                    crossed_finish
                    and telemetry_valid_lap
                    and state_monotonic_progress_m[row] >= target_lap_progress_m
                    and not state_terminated[row]
                )
                if lap_complete:
                    state_lap_index[row] = state_lap_index[row] + wp.int64(1)
                    state_completed_lap[row] = True
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(lap_complete_id)
                if collided:
                    state_terminated[row] = True
                    state_termination_reason_id[row] = wp.int64(collision_id)
                if off_track:
                    state_terminated[row] = True
                    state_termination_reason_id[row] = wp.int64(off_track_id)
                if state_no_progress_steps[row] >= wp.int64(no_progress_limit_steps):
                    state_terminated[row] = True
                    state_termination_reason_id[row] = wp.int64(no_progress_id)
                if state_elapsed_steps[row] >= wp.int64(max_steps):
                    state_truncated[row] = True
                    state_termination_reason_id[row] = wp.int64(max_steps_id)
                if state_terminated[row]:
                    state_alive[row] = False

                speed_kph_after = state_speed_mps[row] * 3.6
                sim_time_s = wp.float32(state_elapsed_steps[row]) * dt
                progress_m = state_monotonic_progress_m[row]
                progress_from_start_m = wp.max(progress_m - acc_start_progress_m[row], 0.0)
                if progress_m >= acc_best_progress_m[row]:
                    acc_best_progress_m[row] = progress_m
                    acc_best_speed_kph[row] = speed_kph_after
                    acc_best_lateral_error_m[row] = wp.abs(step_lateral_error_m)
                    acc_best_heading_error_deg[row] = wp.abs(control_heading_error_deg)
                if progress_from_start_m <= 300.0:
                    acc_speed_sum_first_300_m[row] = acc_speed_sum_first_300_m[row] + speed_kph_after
                    acc_speed_count_first_300_m[row] = acc_speed_count_first_300_m[row] + 1.0
                    acc_brake_sum_first_300_m[row] = acc_brake_sum_first_300_m[row] + brake
                    acc_brake_count_first_300_m[row] = acc_brake_count_first_300_m[row] + 1.0
                if progress_from_start_m <= 450.0:
                    acc_speed_sum_first_450_m[row] = acc_speed_sum_first_450_m[row] + speed_kph_after
                    acc_speed_count_first_450_m[row] = acc_speed_count_first_450_m[row] + 1.0
                    acc_brake_sum_first_450_m[row] = acc_brake_sum_first_450_m[row] + brake
                    acc_brake_count_first_450_m[row] = acc_brake_count_first_450_m[row] + 1.0
                if acc_time_to_300_m[row] < 0.0 and progress_from_start_m >= 300.0:
                    acc_time_to_300_m[row] = sim_time_s
                if acc_time_to_450_m[row] < 0.0 and progress_from_start_m >= 450.0:
                    acc_time_to_450_m[row] = sim_time_s
                demand = (
                    (future_brake_demand >= 0.22)
                    or (brake_gate_proximity >= 0.70)
                    or (target_speed_drop_norm >= 0.18)
                ) and speed_kph_after >= 135.0
                if demand:
                    acc_demand_brake_sum[row] = acc_demand_brake_sum[row] + brake
                    acc_demand_throttle_sum[row] = acc_demand_throttle_sum[row] + throttle
                    acc_demand_count[row] = acc_demand_count[row] + 1.0
                    if future_brake_demand > acc_demand_max_future_brake_demand[row]:
                        acc_demand_max_future_brake_demand[row] = future_brake_demand
                if brake > acc_max_brake[row]:
                    acc_max_brake[row] = brake
                acc_throttle_sum[row] = acc_throttle_sum[row] + throttle
                acc_row_count[row] = acc_row_count[row] + 1.0
                if acc_row_count[row] <= 1.0:
                    acc_start_speed_for_score_kph[row] = speed_kph_after

                acc_final_x[row] = state_x[row]
                acc_final_y[row] = state_y[row]
                acc_final_heading_deg[row] = state_heading_rad[row] * (180.0 / pi)
                acc_final_speed_mps[row] = state_speed_mps[row]
                acc_final_speed_kph[row] = speed_kph_after
                acc_final_raw_progress_m[row] = state_raw_progress_m[row]
                acc_final_progress_m[row] = state_monotonic_progress_m[row]
                acc_final_lateral_error_m[row] = step_lateral_error_m
                acc_final_heading_error_deg[row] = control_heading_error_deg
                acc_final_yaw_rate_rps[row] = state_yaw_rate_rps[row]
                acc_final_curvature_rad_per_m[row] = state_yaw_rate_rps[row] / wp.max(state_speed_mps[row], 1.0e-6)
                acc_final_steering[row] = steer
                acc_final_throttle[row] = throttle
                acc_final_brake[row] = brake
                acc_final_step_index[row] = state_elapsed_steps[row]
                acc_final_sim_time_s[row] = sim_time_s
                acc_final_missed_checkpoint_count[row] = state_missed_checkpoint_count[row]
                acc_final_valid_lap[row] = telemetry_valid_lap
                acc_final_finish_crossed[row] = state_finish_crossed[row]
                acc_final_completed_lap[row] = state_completed_lap[row]
                acc_final_segment_complete[row] = state_segment_complete[row]
                acc_final_collided[row] = collided
                acc_final_off_track[row] = off_track
                acc_final_terminated[row] = state_terminated[row]
                acc_final_truncated[row] = state_truncated[row]
                acc_final_termination_reason_id[row] = state_termination_reason_id[row]
                acc_final_target_speed_kph[row] = target_speed_kph
                acc_final_near_target_speed_kph[row] = near_target_speed_kph
                acc_final_min_future_target_speed_kph[row] = min_future_target_speed_kph
                acc_final_target_speed_drop_kph[row] = target_speed_drop_kph
                acc_final_target_speed_drop_norm[row] = target_speed_drop_norm
                acc_final_brake_demand[row] = brake_demand
                acc_final_future_brake_demand[row] = future_brake_demand
                acc_final_brake_gate_proximity[row] = brake_gate_proximity
                acc_final_braking_gate_distance_m[row] = braking_gate_distance_m
                acc_final_brake_gate_distance_norm[row] = brake_gate_distance_norm
                acc_final_lookahead_abs_max[row] = lookahead_abs_max_diag

        sim_steps_per_row[row] = row_steps

    return _persistent_controller_open_kernel


def persistent_controller_open_rollout_warp_batch(
    state: GpuCarBatch,
    accumulator: GpuScoreAccumulator,
    *,
    last_segment_idx: torch.Tensor,
    controller_weights: torch.Tensor,
    feature_ids: torch.Tensor,
    braking_gates_m: torch.Tensor,
    lookahead_m: torch.Tensor,
    params: GpuCarParams,
    track: GpuTrackTensors,
    sim_config: SimConfig,
    target_progress_m: float,
    final_action_id: torch.Tensor,
    sim_steps_per_row: torch.Tensor,
    collision_check: bool = True,
) -> None:
    """Run a controller/no-target open-distance rollout in one persistent Warp kernel."""

    if state.device.type != "cuda" or state.dtype != torch.float32:
        raise ValueError("Persistent Warp rollout currently requires CUDA float32 state tensors.")
    if track.device != state.device or track.dtype != state.dtype:
        raise ValueError("Persistent Warp rollout track tensors must match state device and dtype.")
    if controller_weights.device != state.device or controller_weights.dtype != torch.float32:
        raise ValueError("Persistent Warp controller weights must be CUDA float32 tensors.")
    if controller_weights.ndim != 3 or controller_weights.shape[1] != 3 or controller_weights.shape[0] != state.size:
        raise ValueError("controller_weights must have shape [batch, 3, feature_count].")
    if feature_ids.device != state.device or feature_ids.dtype != torch.int64:
        raise ValueError("feature_ids must be a CUDA int64 tensor.")
    if int(feature_ids.numel()) != int(controller_weights.shape[2]):
        raise ValueError("feature_ids length must match controller feature count.")
    if params.physics_model == "v2" and len(params.v2_gear_ratios) != 8:
        raise ValueError("Persistent Warp V2 rollout expects exactly 8 gear ratios.")
    if last_segment_idx.shape != state.x.shape or final_action_id.shape != state.x.shape:
        raise ValueError("last_segment_idx and final_action_id must match state batch shape.")
    if sim_steps_per_row.shape != state.x.shape or sim_steps_per_row.dtype != torch.int64:
        raise ValueError("sim_steps_per_row must be an int64 tensor with shape [batch].")
    braking_gates = braking_gates_m.to(device=state.device, dtype=torch.float32).contiguous()
    lookahead = lookahead_m.to(device=state.device, dtype=torch.float32).contiguous()
    feature_ids = feature_ids.to(device=state.device, dtype=torch.int64).contiguous()
    controller_weights = controller_weights.contiguous()
    last_segment_idx = last_segment_idx.to(device=state.device, dtype=torch.int64).contiguous()
    final_action_id = final_action_id.to(device=state.device, dtype=torch.int64).contiguous()
    sim_steps_per_row = sim_steps_per_row.to(device=state.device, dtype=torch.int64).contiguous()
    sim_steps_per_row.zero_()
    wp = require_warp()
    accumulator_inputs = [wp.from_torch(getattr(accumulator, field.name)) for field in fields(GpuScoreAccumulator)]
    gear_ratios = tuple(float(value) for value in params.v2_gear_ratios)
    physics_v2_enabled = params.physics_model == "v2"
    max_steer_deg = params.v2_max_steer_deg if physics_v2_enabled else params.max_steer_deg
    steer_response = params.v2_steer_response if physics_v2_enabled else params.steer_response
    steering_speed_sensitivity = (
        params.v2_steering_speed_sensitivity if physics_v2_enabled else params.steering_speed_sensitivity
    )
    wp.launch(
        _persistent_controller_open_rollout_kernel(),
        dim=state.size,
        inputs=[
            wp.from_torch(controller_weights),
            wp.from_torch(feature_ids),
            wp.from_torch(lookahead),
            wp.from_torch(braking_gates),
            wp.from_torch(track.centerline_xy),
            wp.from_torch(track.centerline_segment_vec),
            wp.from_torch(track.centerline_segment_len),
            wp.from_torch(track.centerline_segment_len2),
            wp.from_torch(track.centerline_cumdist_px),
            wp.from_torch(track.centerline_segment_mid_px),
            wp.from_torch(track.local_projection_indices),
            wp.from_torch(track.boundary_segments),
            wp.from_torch(track.boundary_grid_indices),
            wp.from_torch(track.boundary_grid_counts),
            wp.from_torch(track.finish_line),
            wp.from_torch(track.drivable_mask),
            wp.from_torch(last_segment_idx),
            wp.from_torch(final_action_id),
            wp.from_torch(sim_steps_per_row),
            wp.from_torch(state.x),
            wp.from_torch(state.y),
            wp.from_torch(state.heading_rad),
            wp.from_torch(state.speed_mps),
            wp.from_torch(state.yaw_rate_rps),
            wp.from_torch(state.steering),
            wp.from_torch(state.raw_progress_m),
            wp.from_torch(state.monotonic_progress_m),
            wp.from_torch(state.last_raw_progress_px),
            wp.from_torch(state.checkpoint_index),
            wp.from_torch(state.next_checkpoint_index),
            wp.from_torch(state.checkpoints_passed),
            wp.from_torch(state.missed_checkpoint_count),
            wp.from_torch(state.lap_index),
            wp.from_torch(state.elapsed_steps),
            wp.from_torch(state.no_progress_steps),
            wp.from_torch(state.alive),
            wp.from_torch(state.terminated),
            wp.from_torch(state.truncated),
            wp.from_torch(state.termination_reason_id),
            wp.from_torch(state.valid_lap),
            wp.from_torch(state.finish_crossed),
            wp.from_torch(state.completed_lap),
            wp.from_torch(state.segment_complete),
            wp.from_torch(state.segment_release_observed),
            wp.from_torch(state.last_throttle),
            wp.from_torch(state.last_brake),
            wp.from_torch(state.last_steer),
            *accumulator_inputs,
            int(sim_config.max_steps),
            int(feature_ids.numel()),
            int(lookahead.numel()),
            int(braking_gates.numel()),
            int(track.centerline_xy.shape[0]),
            int(track.local_projection_indices.shape[1]),
            int(track.boundary_grid_width),
            int(track.boundary_grid_height),
            int(track.boundary_grid_indices.shape[1]),
            3,
            int(track.drivable_mask.shape[1]),
            int(track.drivable_mask.shape[0]),
            bool(collision_check),
            float(track.length_px.detach().cpu().item()),
            float(track.meters_per_pixel.detach().cpu().item()),
            float(track.length_m.detach().cpu().item()),
            float(sim_config.local_projection_window_m),
            float(sim_config.local_projection_window_m)
            / max(float(track.meters_per_pixel.detach().cpu().item()), 1.0e-6),
            float(track.boundary_grid_cell_size_px.detach().cpu().item()),
            float(track.checkpoint_spacing_m.detach().cpu().item()),
            float(sim_config.checkpoint_lateral_limit_m),
            int(track.checkpoint_count),
            int(sim_config.no_progress_limit_steps),
            float(target_progress_m),
            float(params.wheelbase_m),
            float(max_steer_deg) * math.pi / 180.0,
            float(steer_response),
            float(params.engine_accel_mps2),
            float(params.brake_accel_mps2),
            float(params.drag_coefficient),
            float(params.rolling_resistance_mps2),
            float(params.grip_g),
            float(params.aero_grip_per_mps2),
            float(params.max_grip_g),
            float(params.max_drive_g),
            float(params.max_brake_g),
            float(steering_speed_sensitivity),
            float(params.max_speed_mps),
            float(params.dt),
            physics_v2_enabled,
            float(params.mass),
            float(params.v2_front_weight_distribution),
            float(params.v2_cg_height_m),
            float(params.v2_track_width_m),
            float(params.v2_front_axle_distance_m),
            float(params.v2_rear_axle_distance_m),
            float(params.v2_front_cornering_stiffness_n_per_rad),
            float(params.v2_rear_cornering_stiffness_n_per_rad),
            float(params.v2_front_peak_mu),
            float(params.v2_rear_peak_mu),
            float(params.v2_mechanical_grip_low_speed_scale),
            float(params.v2_mechanical_grip_high_speed_scale),
            float(params.v2_mechanical_grip_transition_mps),
            float(params.v2_tire_shape_c),
            math.radians(float(params.v2_slip_angle_peak_deg)),
            float(params.v2_rear_slip_steer_coupling),
            float(params.v2_post_peak_falloff),
            float(params.v2_load_sensitivity),
            float(params.v2_aero_downforce_n_per_mps2),
            float(params.v2_aero_balance_front),
            float(params.v2_engine_power_w),
            float(params.v2_drivetrain_efficiency),
            float(params.v2_power_min_speed_mps),
            float(params.v2_max_drive_g),
            float(params.v2_max_brake_g),
            float(params.v2_brake_lock_threshold),
            float(params.v2_brake_lock_min_speed_mps),
            float(params.v2_brake_lock_steer_loss),
            float(params.v2_drag_coefficient),
            float(params.v2_rolling_resistance_mps2),
            float(params.v2_max_speed_mps),
            gear_ratios[0],
            gear_ratios[1],
            gear_ratios[2],
            gear_ratios[3],
            gear_ratios[4],
            gear_ratios[5],
            gear_ratios[6],
            gear_ratios[7],
            float(params.v2_final_drive_ratio),
            float(params.v2_wheel_radius_m),
            float(params.v2_idle_rpm),
            float(params.v2_shift_up_rpm),
            float(params.v2_max_rpm),
            float(params.v2_torque_peak_rpm),
            float(params.v2_torque_low_rpm_factor),
            float(params.v2_torque_high_rpm_factor),
            float(params.v2_tire_scrub_drag),
            float(params.v2_surface_mu),
            float(sim_config.reward.speed_target_min_kph),
            float(sim_config.reward.speed_target_max_kph),
            float(sim_config.reward.speed_target_heading_scale),
            float(sim_config.reward.speed_target_deadzone_kph),
            float(sim_config.launch_guard_progress_m),
            float(sim_config.launch_guard_min_speed_kph),
            float(sim_config.launch_guard_throttle),
            int(TERMINATION_REASON_TO_ID["lap_complete"]),
            int(TERMINATION_REASON_TO_ID["collision"]),
            int(TERMINATION_REASON_TO_ID["off_track"]),
            int(TERMINATION_REASON_TO_ID["no_progress"]),
            int(TERMINATION_REASON_TO_ID["max_steps"]),
        ],
        device=_warp_device_for_torch(state.device),
    )


def _require_cuda_float32_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA for Warp scoring.")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32 for Warp scoring.")
    return tensor.contiguous()


def _require_mutable_cuda_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA for Warp mutation.")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} for Warp mutation.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for Warp mutation.")
    return tensor


def update_score_accumulator_warp_batch(
    acc: GpuScoreAccumulator,
    *,
    state: GpuCarBatch,
    diagnostics: GpuScoringDiagnostics,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    active: torch.Tensor,
    collided: torch.Tensor,
    off_track: torch.Tensor,
    config: SimConfig,
    zero: torch.Tensor | None = None,
) -> None:
    """Warp in-place equivalent of ``update_score_accumulator_static`` for fused rollouts."""

    del zero
    if state.device.type != "cuda" or state.dtype != torch.float32:
        raise ValueError("Warp score accumulator update currently requires CUDA float32 state tensors.")
    shape = state.x.shape
    active = active.to(device=state.device, dtype=torch.bool).contiguous()
    collided = collided.to(device=state.device, dtype=torch.bool).contiguous()
    off_track = off_track.to(device=state.device, dtype=torch.bool).contiguous()
    throttle = _require_cuda_float32_tensor("throttle", throttle.to(device=state.device, dtype=torch.float32))
    brake = _require_cuda_float32_tensor("brake", brake.to(device=state.device, dtype=torch.float32))
    steer = _require_cuda_float32_tensor("steer", steer.to(device=state.device, dtype=torch.float32))
    if (
        active.shape != shape
        or collided.shape != shape
        or off_track.shape != shape
        or throttle.shape != shape
        or brake.shape != shape
        or steer.shape != shape
    ):
        raise ValueError("Warp score accumulator inputs must match the state batch shape.")

    def state_float(name: str) -> torch.Tensor:
        return _require_cuda_float32_tensor(f"state.{name}", getattr(state, name))

    def state_int(name: str) -> torch.Tensor:
        tensor = getattr(state, name)
        if tensor.device.type != "cuda" or tensor.dtype != torch.int64:
            raise ValueError(f"state.{name} must be CUDA int64 for Warp scoring.")
        return tensor.contiguous()

    def state_bool(name: str) -> torch.Tensor:
        tensor = getattr(state, name)
        if tensor.device.type != "cuda" or tensor.dtype != torch.bool:
            raise ValueError(f"state.{name} must be CUDA bool for Warp scoring.")
        return tensor.contiguous()

    def diag_float(name: str) -> torch.Tensor:
        return _require_cuda_float32_tensor(f"diagnostics.{name}", getattr(diagnostics, name))

    def diag_bool(name: str) -> torch.Tensor:
        tensor = getattr(diagnostics, name)
        if tensor.device.type != "cuda" or tensor.dtype != torch.bool:
            raise ValueError(f"diagnostics.{name} must be CUDA bool for Warp scoring.")
        return tensor.contiguous()

    def acc_float(name: str) -> torch.Tensor:
        return _require_mutable_cuda_tensor(f"acc.{name}", getattr(acc, name), torch.float32)

    def acc_int(name: str) -> torch.Tensor:
        return _require_mutable_cuda_tensor(f"acc.{name}", getattr(acc, name), torch.int64)

    def acc_bool(name: str) -> torch.Tensor:
        return _require_mutable_cuda_tensor(f"acc.{name}", getattr(acc, name), torch.bool)

    wp = require_warp()
    wp.launch(
        _score_accumulator_update_kernel(),
        dim=int(state.x.numel()),
        inputs=[
            wp.from_torch(active),
            wp.from_torch(collided),
            wp.from_torch(off_track),
            wp.from_torch(throttle),
            wp.from_torch(brake),
            wp.from_torch(steer),
            wp.from_torch(state_float("x")),
            wp.from_torch(state_float("y")),
            wp.from_torch(state_float("heading_rad")),
            wp.from_torch(state_float("speed_mps")),
            wp.from_torch(state_float("raw_progress_m")),
            wp.from_torch(state_float("monotonic_progress_m")),
            wp.from_torch(state_float("yaw_rate_rps")),
            wp.from_torch(state_int("elapsed_steps")),
            wp.from_torch(state_int("missed_checkpoint_count")),
            wp.from_torch(state_bool("finish_crossed")),
            wp.from_torch(state_bool("completed_lap")),
            wp.from_torch(state_bool("segment_complete")),
            wp.from_torch(state_bool("terminated")),
            wp.from_torch(state_bool("truncated")),
            wp.from_torch(state_int("termination_reason_id")),
            wp.from_torch(diag_float("lateral_error_m")),
            wp.from_torch(diag_float("heading_error_deg")),
            wp.from_torch(diag_float("future_brake_demand")),
            wp.from_torch(diag_float("brake_gate_proximity")),
            wp.from_torch(diag_float("target_speed_drop_norm")),
            wp.from_torch(diag_bool("telemetry_valid_lap")),
            wp.from_torch(diag_float("target_speed_kph")),
            wp.from_torch(diag_float("near_target_speed_kph")),
            wp.from_torch(diag_float("min_future_target_speed_kph")),
            wp.from_torch(diag_float("target_speed_drop_kph")),
            wp.from_torch(diag_float("brake_demand")),
            wp.from_torch(diag_float("braking_gate_distance_m")),
            wp.from_torch(diag_float("brake_gate_distance_norm")),
            wp.from_torch(diag_float("lookahead_abs_max")),
            wp.from_torch(acc_float("start_progress_m")),
            wp.from_torch(acc_float("start_speed_for_score_kph")),
            wp.from_torch(acc_float("best_progress_m")),
            wp.from_torch(acc_float("best_speed_kph")),
            wp.from_torch(acc_float("best_lateral_error_m")),
            wp.from_torch(acc_float("best_heading_error_deg")),
            wp.from_torch(acc_float("final_x")),
            wp.from_torch(acc_float("final_y")),
            wp.from_torch(acc_float("final_heading_deg")),
            wp.from_torch(acc_float("final_speed_mps")),
            wp.from_torch(acc_float("final_speed_kph")),
            wp.from_torch(acc_float("final_raw_progress_m")),
            wp.from_torch(acc_float("final_progress_m")),
            wp.from_torch(acc_float("final_lateral_error_m")),
            wp.from_torch(acc_float("final_heading_error_deg")),
            wp.from_torch(acc_float("final_yaw_rate_rps")),
            wp.from_torch(acc_float("final_curvature_rad_per_m")),
            wp.from_torch(acc_float("final_steering")),
            wp.from_torch(acc_float("final_throttle")),
            wp.from_torch(acc_float("final_brake")),
            wp.from_torch(acc_int("final_step_index")),
            wp.from_torch(acc_float("final_sim_time_s")),
            wp.from_torch(acc_int("final_missed_checkpoint_count")),
            wp.from_torch(acc_bool("final_valid_lap")),
            wp.from_torch(acc_bool("final_finish_crossed")),
            wp.from_torch(acc_bool("final_completed_lap")),
            wp.from_torch(acc_bool("final_segment_complete")),
            wp.from_torch(acc_bool("final_collided")),
            wp.from_torch(acc_bool("final_off_track")),
            wp.from_torch(acc_bool("final_terminated")),
            wp.from_torch(acc_bool("final_truncated")),
            wp.from_torch(acc_int("final_termination_reason_id")),
            wp.from_torch(acc_float("final_target_speed_kph")),
            wp.from_torch(acc_float("final_near_target_speed_kph")),
            wp.from_torch(acc_float("final_min_future_target_speed_kph")),
            wp.from_torch(acc_float("final_target_speed_drop_kph")),
            wp.from_torch(acc_float("final_target_speed_drop_norm")),
            wp.from_torch(acc_float("final_brake_demand")),
            wp.from_torch(acc_float("final_future_brake_demand")),
            wp.from_torch(acc_float("final_brake_gate_proximity")),
            wp.from_torch(acc_float("final_braking_gate_distance_m")),
            wp.from_torch(acc_float("final_brake_gate_distance_norm")),
            wp.from_torch(acc_float("final_lookahead_abs_max")),
            wp.from_torch(acc_float("time_to_300_m")),
            wp.from_torch(acc_float("time_to_450_m")),
            wp.from_torch(acc_float("speed_sum_first_300_m")),
            wp.from_torch(acc_float("speed_count_first_300_m")),
            wp.from_torch(acc_float("speed_sum_first_450_m")),
            wp.from_torch(acc_float("speed_count_first_450_m")),
            wp.from_torch(acc_float("brake_sum_first_300_m")),
            wp.from_torch(acc_float("brake_count_first_300_m")),
            wp.from_torch(acc_float("brake_sum_first_450_m")),
            wp.from_torch(acc_float("brake_count_first_450_m")),
            wp.from_torch(acc_float("max_brake")),
            wp.from_torch(acc_float("throttle_sum")),
            wp.from_torch(acc_float("row_count")),
            wp.from_torch(acc_float("demand_brake_sum")),
            wp.from_torch(acc_float("demand_throttle_sum")),
            wp.from_torch(acc_float("demand_count")),
            wp.from_torch(acc_float("demand_max_future_brake_demand")),
            float(config.car.dt),
        ],
        device=_warp_device_for_torch(state.device),
    )


def score_profiles_warp_batch(
    acc: GpuScoreAccumulator,
    *,
    profiles: tuple[str, ...],
    target_progress_m: float,
    terminate_at_target_progress: bool,
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
) -> dict[str, torch.Tensor]:
    """Warp scorer for current evolution-search profiles used by the fused backend path."""

    unsupported = [profile for profile in profiles if profile not in _WARP_SCORE_PROFILE_IDS]
    if unsupported:
        supported = ", ".join(sorted(_WARP_SCORE_PROFILE_IDS))
        raise ValueError(f"Warp core scoring does not support profiles {unsupported!r}; supported: {supported}")
    start_progress_m = _require_cuda_float32_tensor("start_progress_m", acc.start_progress_m)
    output_by_profile: dict[str, torch.Tensor] = {}
    tensors = [
        start_progress_m,
        _require_cuda_float32_tensor("best_progress_m", acc.best_progress_m),
        _require_cuda_float32_tensor("best_speed_kph", acc.best_speed_kph),
        _require_cuda_float32_tensor("best_lateral_error_m", acc.best_lateral_error_m),
        _require_cuda_float32_tensor("best_heading_error_deg", acc.best_heading_error_deg),
        _require_cuda_float32_tensor("start_speed_for_score_kph", acc.start_speed_for_score_kph),
        _require_cuda_float32_tensor("final_speed_kph", acc.final_speed_kph),
        _require_cuda_float32_tensor("final_lateral_error_m", acc.final_lateral_error_m),
        _require_cuda_float32_tensor("final_heading_error_deg", acc.final_heading_error_deg),
        _require_cuda_float32_tensor("final_yaw_rate_rps", acc.final_yaw_rate_rps),
        _require_cuda_float32_tensor("final_steering", acc.final_steering),
    ]
    int_tensors = [
        acc.final_missed_checkpoint_count.to(device=start_progress_m.device, dtype=torch.int64).contiguous(),
        acc.final_termination_reason_id.to(device=start_progress_m.device, dtype=torch.int64).contiguous(),
    ]
    bool_tensors = [
        acc.final_collided.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
        acc.final_off_track.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
        acc.final_segment_complete.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
        acc.final_completed_lap.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
        acc.final_valid_lap.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
        acc.final_finish_crossed.to(device=start_progress_m.device, dtype=torch.bool).contiguous(),
    ]
    more_float_tensors = [
        _require_cuda_float32_tensor("final_sim_time_s", acc.final_sim_time_s),
        _require_cuda_float32_tensor("row_count", acc.row_count),
        _require_cuda_float32_tensor("speed_sum_first_300_m", acc.speed_sum_first_300_m),
        _require_cuda_float32_tensor("speed_count_first_300_m", acc.speed_count_first_300_m),
        _require_cuda_float32_tensor("speed_sum_first_450_m", acc.speed_sum_first_450_m),
        _require_cuda_float32_tensor("speed_count_first_450_m", acc.speed_count_first_450_m),
        _require_cuda_float32_tensor("brake_sum_first_450_m", acc.brake_sum_first_450_m),
        _require_cuda_float32_tensor("brake_count_first_450_m", acc.brake_count_first_450_m),
        _require_cuda_float32_tensor("final_brake", acc.final_brake),
        _require_cuda_float32_tensor("time_to_450_m", acc.time_to_450_m),
        _require_cuda_float32_tensor("demand_brake_sum", acc.demand_brake_sum),
        _require_cuda_float32_tensor("demand_throttle_sum", acc.demand_throttle_sum),
        _require_cuda_float32_tensor("demand_count", acc.demand_count),
        _require_cuda_float32_tensor("demand_max_future_brake_demand", acc.demand_max_future_brake_demand),
        _require_cuda_float32_tensor("max_brake", acc.max_brake),
        _require_cuda_float32_tensor("throttle_sum", acc.throttle_sum),
    ]
    wp = require_warp()
    for profile in profiles:
        output = torch.empty_like(start_progress_m)
        wp.launch(
            _score_profile_kernel(),
            dim=int(start_progress_m.numel()),
            inputs=[
                *[wp.from_torch(tensor) for tensor in tensors],
                wp.from_torch(int_tensors[0]),
                wp.from_torch(bool_tensors[0]),
                wp.from_torch(bool_tensors[1]),
                wp.from_torch(bool_tensors[2]),
                wp.from_torch(bool_tensors[3]),
                wp.from_torch(bool_tensors[4]),
                wp.from_torch(bool_tensors[5]),
                wp.from_torch(int_tensors[1]),
                *[wp.from_torch(tensor) for tensor in more_float_tensors],
                wp.from_torch(output),
                int(_WARP_SCORE_PROFILE_IDS[profile]),
                float(target_progress_m),
                bool(terminate_at_target_progress),
                float(MONZA_LENGTH_METERS),
                float(frontier_focus_start_m),
                float(frontier_focus_end_m),
                int(TERMINATION_REASON_TO_ID["no_progress"]),
                int(TERMINATION_REASON_TO_ID["max_steps"]),
                int(TERMINATION_REASON_TO_ID["collision"]),
                int(TERMINATION_REASON_TO_ID["off_track"]),
            ],
            device=_warp_device_for_torch(start_progress_m.device),
        )
        output_by_profile[profile] = output
    return output_by_profile


score_profiles_warp_core_batch = score_profiles_warp_batch
