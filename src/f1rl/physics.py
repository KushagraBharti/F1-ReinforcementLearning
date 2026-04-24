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
    speed += throttle * params.engine_accel_mps2 * dt
    speed -= brake * params.brake_accel_mps2 * dt
    if throttle <= 1e-6 and brake <= 1e-6:
        speed -= params.rolling_resistance_mps2 * dt
    speed -= params.drag_coefficient * speed * speed * dt
    speed = float(np.clip(speed, 0.0, params.max_speed_mps))

    target_steering = steer * np.deg2rad(params.max_steer_deg)
    steering_delta = target_steering - state.steering
    max_delta = params.steer_response * dt
    steering = state.steering + float(np.clip(steering_delta, -max_delta, max_delta))

    if abs(steering) > 1e-6 and speed > 1e-6:
        yaw_rate = speed / max(params.wheelbase_m, 1e-6) * np.tan(steering)
        lateral_accel = abs(speed * yaw_rate)
        max_lateral = params.grip_g * 9.81
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
