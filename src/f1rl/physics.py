"""Vehicle state and medium-simple top-down bicycle dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from f1rl.config import CarParams, PhysicsV2Params


@dataclass(slots=True)
class CarState:
    x: float
    y: float
    heading_rad: float
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    steering: float = 0.0
    lap_index: int = 0
    checkpoint_index: int = 0
    raw_progress_m: float = 0.0
    monotonic_progress_m: float = 0.0
    elapsed_steps: int = 0
    alive: bool = True
    gear: int = 1
    rpm: float = 0.0
    front_slip_angle_rad: float = 0.0
    rear_slip_angle_rad: float = 0.0
    front_load_n: float = 0.0
    rear_load_n: float = 0.0
    front_lateral_force_n: float = 0.0
    rear_lateral_force_n: float = 0.0
    tire_saturation: float = 0.0
    surface_mu: float = 1.0
    wheel_lock: bool = False

    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)


def initial_car_state(start_pose: np.ndarray) -> CarState:
    return CarState(x=float(start_pose[0]), y=float(start_pose[1]), heading_rad=float(start_pose[2]))


def grip_limit_g(params: CarParams, speed_mps: float) -> float:
    aero_grip = params.aero_grip_per_mps2 * speed_mps * speed_mps
    return float(np.clip(params.grip_g + aero_grip, params.grip_g, params.max_grip_g))


def tire_lateral_force_v2(
    slip_angle_rad: float,
    load_n: float,
    *,
    stiffness_n_per_rad: float,
    peak_mu: float,
    shape_c: float,
    slip_angle_peak_rad: float,
    post_peak_falloff: float,
    load_sensitivity: float,
    reference_load_n: float,
    surface_mu: float,
) -> float:
    """Pacejka-lite lateral force curve used by CPU and GPU V2 implementations."""

    load_n = max(float(load_n), 1.0)
    reference_load_n = max(float(reference_load_n), 1.0)
    load_ratio = load_n / reference_load_n
    mu = peak_mu * surface_mu * (1.0 - load_sensitivity * max(load_ratio - 1.0, 0.0))
    peak_force_n = max(mu * load_n, 1.0)
    b = stiffness_n_per_rad / max(shape_c * peak_force_n, 1e-6)
    alpha = float(slip_angle_rad)
    force = peak_force_n * math.sin(shape_c * math.atan(b * alpha))
    abs_alpha = abs(alpha)
    if abs_alpha > slip_angle_peak_rad:
        excess = min((abs_alpha - slip_angle_peak_rad) / max(slip_angle_peak_rad, 1e-6), 1.0)
        force *= 1.0 - post_peak_falloff * excess
    return float(force)


def mechanical_grip_scale_v2(speed_mps: float, v2: PhysicsV2Params) -> float:
    """Speed-dependent mechanical grip scale for V2 tire calibration."""

    transition = max(float(v2.mechanical_grip_transition_mps), 1e-6)
    t = float(np.clip(max(float(speed_mps), 0.0) / transition, 0.0, 1.0))
    smooth_t = t * t * (3.0 - 2.0 * t)
    return float(
        v2.mechanical_grip_low_speed_scale
        + (v2.mechanical_grip_high_speed_scale - v2.mechanical_grip_low_speed_scale) * smooth_t
    )


def weight_transfer_v2(
    *,
    params: CarParams,
    v2: PhysicsV2Params,
    longitudinal_accel_mps2: float,
    lateral_accel_mps2: float,
    speed_mps: float,
) -> tuple[float, float]:
    static_front = params.mass * 9.81 * v2.front_weight_distribution
    static_rear = params.mass * 9.81 - static_front
    aero_total = v2.aero_downforce_n_per_mps2 * speed_mps * speed_mps
    aero_front = aero_total * v2.aero_balance_front
    aero_rear = aero_total - aero_front
    long_transfer = params.mass * longitudinal_accel_mps2 * v2.cg_height_m / max(params.wheelbase_m, 1e-6)
    lateral_unload = abs(params.mass * lateral_accel_mps2 * v2.cg_height_m / max(v2.track_width_m, 1e-6)) * 0.12
    front_load = max(static_front + aero_front - long_transfer - lateral_unload * v2.front_weight_distribution, 1.0)
    rear_load = max(static_rear + aero_rear + long_transfer - lateral_unload * (1.0 - v2.front_weight_distribution), 1.0)
    return float(front_load), float(rear_load)


def gear_and_rpm_v2(speed_mps: float, previous_gear: int, v2: PhysicsV2Params) -> tuple[int, float]:
    wheel_rps = max(speed_mps, 0.0) / max(2.0 * math.pi * v2.wheel_radius_m, 1e-6)
    previous_gear = int(np.clip(previous_gear, 1, len(v2.gear_ratios)))

    def rpm_for(gear: int) -> float:
        ratio = v2.gear_ratios[gear - 1] * v2.final_drive_ratio
        return max(v2.idle_rpm, wheel_rps * ratio * 60.0)

    gear = previous_gear
    rpm = rpm_for(gear)
    while gear < len(v2.gear_ratios) and rpm > v2.shift_up_rpm:
        gear += 1
        rpm = rpm_for(gear)
    while gear > 1 and rpm < v2.shift_down_rpm:
        lower_rpm = rpm_for(gear - 1)
        if lower_rpm > v2.max_rpm:
            break
        gear -= 1
        rpm = lower_rpm
    return gear, float(min(rpm, v2.max_rpm))


def _torque_factor_v2(rpm: float, v2: PhysicsV2Params) -> float:
    if rpm <= v2.torque_peak_rpm:
        span = max(v2.torque_peak_rpm - v2.idle_rpm, 1.0)
        ratio = np.clip((rpm - v2.idle_rpm) / span, 0.0, 1.0)
        return float(v2.torque_low_rpm_factor + (1.0 - v2.torque_low_rpm_factor) * ratio)
    span = max(v2.max_rpm - v2.torque_peak_rpm, 1.0)
    ratio = np.clip((rpm - v2.torque_peak_rpm) / span, 0.0, 1.0)
    return float(1.0 - (1.0 - v2.torque_high_rpm_factor) * ratio)


def apply_physics_v1(
    state: CarState,
    *,
    throttle: float,
    brake: float,
    steer: float,
    params: CarParams,
    meters_per_pixel: float,
) -> tuple[CarState, np.ndarray]:
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    steer = float(np.clip(steer, -1.0, 1.0))
    dt = params.dt

    speed = max(0.0, float(state.speed_mps))
    target_steering = steer * np.deg2rad(params.max_steer_deg)
    steering_delta = target_steering - state.steering
    max_delta = params.steer_response * dt
    steering = state.steering + float(np.clip(steering_delta, -max_delta, max_delta))
    effective_steering = steering / (1.0 + params.steering_speed_sensitivity * speed * speed)

    max_total_accel = grip_limit_g(params, speed) * 9.81
    if abs(effective_steering) > 1e-6 and speed > 1e-6:
        requested_yaw = speed / max(params.wheelbase_m, 1e-6) * np.tan(effective_steering)
        requested_lateral = abs(speed * requested_yaw)
    else:
        requested_yaw = 0.0
        requested_lateral = 0.0
    lateral_accel = min(requested_lateral, max_total_accel)
    longitudinal_capacity = float(np.sqrt(max(max_total_accel * max_total_accel - lateral_accel * lateral_accel, 0.0)))
    drive_limit = min(params.engine_accel_mps2, params.max_drive_g * 9.81, longitudinal_capacity)
    brake_limit = min(params.brake_accel_mps2, params.max_brake_g * 9.81, longitudinal_capacity)

    longitudinal_accel = throttle * drive_limit - brake * brake_limit
    if throttle <= 1e-6 and brake <= 1e-6:
        longitudinal_accel -= params.rolling_resistance_mps2
    longitudinal_accel -= params.drag_coefficient * speed * speed
    speed = float(np.clip(speed + longitudinal_accel * dt, 0.0, params.max_speed_mps))

    if abs(effective_steering) > 1e-6 and speed > 1e-6:
        yaw_rate = speed / max(params.wheelbase_m, 1e-6) * np.tan(effective_steering)
        lateral_accel = abs(speed * yaw_rate)
        max_lateral = grip_limit_g(params, speed) * 9.81
        if lateral_accel > max_lateral:
            yaw_rate *= max_lateral / max(lateral_accel, 1e-6)
    else:
        yaw_rate = 0.0

    heading = float((state.heading_rad + yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi)
    distance_px = speed * dt / max(meters_per_pixel, 1e-6)
    dx = float(np.cos(heading) * distance_px)
    dy = float(-np.sin(heading) * distance_px)
    x_new = state.x + dx
    y_new = state.y + dy
    movement = np.asarray([state.x, state.y, x_new, y_new], dtype=np.float32)
    return (
        replace(
            state,
            x=x_new,
            y=y_new,
            heading_rad=heading,
            speed_mps=speed,
            yaw_rate_rps=float(yaw_rate),
            steering=steering,
            elapsed_steps=state.elapsed_steps + 1,
            gear=1,
            rpm=0.0,
            front_slip_angle_rad=0.0,
            rear_slip_angle_rad=0.0,
            front_load_n=0.0,
            rear_load_n=0.0,
            front_lateral_force_n=0.0,
            rear_lateral_force_n=0.0,
            tire_saturation=0.0,
            surface_mu=1.0,
            wheel_lock=False,
        ),
        movement,
    )


def apply_physics_v2(
    state: CarState,
    *,
    throttle: float,
    brake: float,
    steer: float,
    params: CarParams,
    v2: PhysicsV2Params,
    meters_per_pixel: float,
) -> tuple[CarState, np.ndarray]:
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    steer = float(np.clip(steer, -1.0, 1.0))
    dt = params.dt
    speed = max(0.0, float(state.speed_mps))

    target_steering = steer * math.radians(v2.max_steer_deg)
    steering_delta = target_steering - state.steering
    max_delta = v2.steer_response * dt
    steering = state.steering + float(np.clip(steering_delta, -max_delta, max_delta))
    wheel_lock = bool(brake >= v2.brake_lock_threshold and speed >= v2.brake_lock_min_speed_mps)
    effective_steering = steering / (1.0 + v2.steering_speed_sensitivity * speed * speed)
    if wheel_lock:
        effective_steering *= max(0.0, 1.0 - v2.brake_lock_steer_loss * brake)

    previous_lateral_accel = abs(speed * state.yaw_rate_rps)
    front_load, rear_load = weight_transfer_v2(
        params=params,
        v2=v2,
        longitudinal_accel_mps2=0.0,
        lateral_accel_mps2=previous_lateral_accel,
        speed_mps=speed,
    )
    reference_front = params.mass * 9.81 * v2.front_weight_distribution
    reference_rear = params.mass * 9.81 - reference_front
    grip_scale = mechanical_grip_scale_v2(speed, v2)
    front_slip = math.atan2(state.yaw_rate_rps * v2.front_axle_distance_m, max(abs(speed), 1e-3)) - effective_steering
    rear_slip = (
        -math.atan2(state.yaw_rate_rps * v2.rear_axle_distance_m, max(abs(speed), 1e-3))
        - v2.rear_slip_steer_coupling * effective_steering
    )
    slip_peak = math.radians(v2.slip_angle_peak_deg)
    front_force = tire_lateral_force_v2(
        front_slip,
        front_load,
        stiffness_n_per_rad=v2.front_cornering_stiffness_n_per_rad,
        peak_mu=v2.front_peak_mu * grip_scale,
        shape_c=v2.tire_shape_c,
        slip_angle_peak_rad=slip_peak,
        post_peak_falloff=v2.post_peak_falloff,
        load_sensitivity=v2.load_sensitivity,
        reference_load_n=reference_front,
        surface_mu=v2.surface_mu,
    )
    rear_force = tire_lateral_force_v2(
        rear_slip,
        rear_load,
        stiffness_n_per_rad=v2.rear_cornering_stiffness_n_per_rad,
        peak_mu=v2.rear_peak_mu * grip_scale,
        shape_c=v2.tire_shape_c,
        slip_angle_peak_rad=slip_peak,
        post_peak_falloff=v2.post_peak_falloff,
        load_sensitivity=v2.load_sensitivity,
        reference_load_n=reference_rear,
        surface_mu=v2.surface_mu,
    )
    lateral_capacity = (abs(front_force) + abs(rear_force)) / max(params.mass, 1e-6)
    if abs(effective_steering) > 1e-6 and speed > 1e-6:
        requested_yaw = speed / max(params.wheelbase_m, 1e-6) * math.tan(effective_steering)
        requested_lateral = abs(speed * requested_yaw)
        lateral_accel = min(requested_lateral, lateral_capacity)
        yaw_rate = math.copysign(lateral_accel / max(speed, 1e-6), requested_yaw)
    else:
        lateral_accel = 0.0
        yaw_rate = 0.0

    total_peak_force = (
        v2.front_peak_mu * grip_scale * front_load * v2.surface_mu
        + v2.rear_peak_mu * grip_scale * rear_load * v2.surface_mu
    )
    total_accel_limit = total_peak_force / max(params.mass, 1e-6)
    longitudinal_capacity = math.sqrt(max(total_accel_limit * total_accel_limit - lateral_accel * lateral_accel, 0.0))
    gear, rpm = gear_and_rpm_v2(speed, 1, v2)
    torque_factor = _torque_factor_v2(rpm, v2)
    power_force = v2.engine_power_w * v2.drivetrain_efficiency * torque_factor / max(speed, v2.power_min_speed_mps)
    drive_limit = min(v2.max_drive_g * 9.81, power_force / max(params.mass, 1e-6), longitudinal_capacity)
    brake_limit = min(v2.max_brake_g * 9.81, longitudinal_capacity)
    if wheel_lock:
        brake_limit *= 0.82

    longitudinal_accel = throttle * drive_limit - brake * brake_limit
    if throttle <= 1e-6 and brake <= 1e-6:
        longitudinal_accel -= v2.rolling_resistance_mps2
    longitudinal_accel -= v2.drag_coefficient * speed * speed
    tire_saturation = float(
        np.clip(
            max(abs(front_slip), abs(rear_slip)) / max(slip_peak, 1e-6),
            0.0,
            3.0,
        )
    )
    longitudinal_accel -= v2.tire_scrub_drag * max(tire_saturation - 1.0, 0.0)
    next_speed = float(np.clip(speed + longitudinal_accel * dt, 0.0, v2.max_speed_mps))
    gear, rpm = gear_and_rpm_v2(next_speed, 1, v2)
    front_load, rear_load = weight_transfer_v2(
        params=params,
        v2=v2,
        longitudinal_accel_mps2=longitudinal_accel,
        lateral_accel_mps2=lateral_accel,
        speed_mps=next_speed,
    )

    heading = float((state.heading_rad + yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi)
    distance_px = next_speed * dt / max(meters_per_pixel, 1e-6)
    dx = float(math.cos(heading) * distance_px)
    dy = float(-math.sin(heading) * distance_px)
    x_new = state.x + dx
    y_new = state.y + dy
    movement = np.asarray([state.x, state.y, x_new, y_new], dtype=np.float32)
    return (
        replace(
            state,
            x=x_new,
            y=y_new,
            heading_rad=heading,
            speed_mps=next_speed,
            yaw_rate_rps=float(yaw_rate),
            steering=steering,
            elapsed_steps=state.elapsed_steps + 1,
            gear=gear,
            rpm=rpm,
            front_slip_angle_rad=float(front_slip),
            rear_slip_angle_rad=float(rear_slip),
            front_load_n=front_load,
            rear_load_n=rear_load,
            front_lateral_force_n=float(front_force),
            rear_lateral_force_n=float(rear_force),
            tire_saturation=tire_saturation,
            surface_mu=v2.surface_mu,
            wheel_lock=wheel_lock,
        ),
        movement,
    )


def apply_physics(
    state: CarState,
    *,
    throttle: float,
    brake: float,
    steer: float,
    params: CarParams,
    meters_per_pixel: float,
    physics_model: str = "v1",
    physics_v2: PhysicsV2Params | None = None,
) -> tuple[CarState, np.ndarray]:
    if physics_model == "v1":
        return apply_physics_v1(
            state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=params,
            meters_per_pixel=meters_per_pixel,
        )
    if physics_model == "v2":
        return apply_physics_v2(
            state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=params,
            v2=physics_v2 or PhysicsV2Params(),
            meters_per_pixel=meters_per_pixel,
        )
    raise ValueError(f"Unknown physics_model {physics_model!r}; expected 'v1' or 'v2'.")
