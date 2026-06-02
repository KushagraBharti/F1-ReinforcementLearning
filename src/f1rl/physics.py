"""Vehicle state and medium-simple top-down bicycle dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from f1rl.config import CarParams


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

    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)


def initial_car_state(start_pose: np.ndarray) -> CarState:
    return CarState(x=float(start_pose[0]), y=float(start_pose[1]), heading_rad=float(start_pose[2]))


def grip_limit_g(params: CarParams, speed_mps: float) -> float:
    aero_grip = params.aero_grip_per_mps2 * speed_mps * speed_mps
    return float(np.clip(params.grip_g + aero_grip, params.grip_g, params.max_grip_g))


def apply_physics(
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
        ),
        movement,
    )
