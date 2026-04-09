"""Vehicle dynamics and observation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from f1rl.config import DynamicsConfig
from f1rl.geometry import nearest_intersection_distance


@dataclass(slots=True)
class CarState:
    x: float
    y: float
    heading_deg: float
    speed: float
    steering_angle_deg: float
    next_goal_idx: int
    lap_count: int
    step_count: int
    alive: bool = True
    total_reward: float = 0.0
    last_action: tuple[float, float] = (0.0, 0.0)


def new_car_state(start_pos: tuple[float, float], heading_deg: float) -> CarState:
    return CarState(
        x=float(start_pos[0]),
        y=float(start_pos[1]),
        heading_deg=float(heading_deg),
        speed=0.0,
        steering_angle_deg=0.0,
        next_goal_idx=1,
        lap_count=0,
        step_count=0,
    )


def _move_toward_zero(value: float, delta: float) -> float:
    if value > 0.0:
        return max(0.0, value - delta)
    if value < 0.0:
        return min(0.0, value + delta)
    return value


def apply_dynamics(state: CarState, action: np.ndarray, config: DynamicsConfig) -> tuple[CarState, np.ndarray]:
    steering_input = float(np.clip(action[0], -1.0, 1.0))
    throttle_input = float(np.clip(action[1], -1.0, 1.0))

    speed = float(state.speed)
    if throttle_input > 0.0:
        speed += throttle_input * config.engine_acceleration_rate * config.dt
    elif throttle_input < 0.0:
        if speed > 0.0:
            speed = max(0.0, speed + throttle_input * config.brake_deceleration_rate * config.dt)
        else:
            speed += throttle_input * config.reverse_acceleration_rate * config.dt
    else:
        speed = _move_toward_zero(speed, config.coast_deceleration_rate * config.dt)

    speed *= max(0.0, 1.0 - config.friction)
    if abs(speed) > 1e-6:
        speed -= float(np.sign(speed)) * config.drag_coefficient * speed * speed * config.dt
    speed = float(np.clip(speed, -config.max_reverse_speed, config.max_speed))

    target_steering_angle = steering_input * config.max_steer_angle_deg
    steering_angle_deg = state.steering_angle_deg + (
        target_steering_angle - state.steering_angle_deg
    ) * config.steering_response

    speed_ratio = min(abs(speed) / max(config.max_speed, 1e-6), 1.0)
    effective_steer_deg = steering_angle_deg * (1.0 - config.high_speed_steer_reduction * speed_ratio * speed_ratio)
    heading_rad = np.deg2rad(state.heading_deg)
    yaw_rate = (speed / max(config.wheelbase, 1e-6)) * np.tan(np.deg2rad(effective_steer_deg)) * config.lateral_grip
    heading_deg = state.heading_deg + np.rad2deg(yaw_rate) * config.dt
    heading_deg = ((heading_deg + 180.0) % 360.0) - 180.0

    heading_rad = np.deg2rad(heading_deg)
    dx = float(np.cos(heading_rad) * speed * config.dt)
    dy = float(-np.sin(heading_rad) * speed * config.dt)
    x_new = state.x + dx
    y_new = state.y + dy

    movement = np.array([state.x, state.y, x_new, y_new], dtype=np.float32)
    new_state = replace(
        state,
        x=x_new,
        y=y_new,
        heading_deg=heading_deg,
        speed=speed,
        steering_angle_deg=steering_angle_deg,
        step_count=state.step_count + 1,
        last_action=(steering_input, throttle_input),
    )
    return new_state, movement


def sensor_angles(config: DynamicsConfig) -> np.ndarray:
    count = int(max(3, config.sensor_count))
    if count % 2 == 0:
        count += 1
    base = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    biased = np.sign(base) * np.power(np.abs(base), config.sensor_forward_bias)
    return biased * (config.sensor_spread_deg / 2.0)


def build_sensor_rays(state: CarState, config: DynamicsConfig) -> np.ndarray:
    angles = sensor_angles(config)
    rays = np.zeros((angles.shape[0], 4), dtype=np.float32)
    rays[:, 0] = state.x
    rays[:, 1] = state.y
    for idx, rel_angle in enumerate(angles):
        heading = np.deg2rad(state.heading_deg + float(rel_angle))
        rays[idx, 2] = state.x + np.cos(heading) * config.sensor_range
        rays[idx, 3] = state.y - np.sin(heading) * config.sensor_range
    return rays


def sensor_distances(state: CarState, config: DynamicsConfig, collision_segments: np.ndarray) -> np.ndarray:
    rays = build_sensor_rays(state, config)
    distances = np.full((rays.shape[0],), config.sensor_range, dtype=np.float32)
    for idx, ray in enumerate(rays):
        distances[idx] = nearest_intersection_distance(ray, collision_segments, config.sensor_range)
    return distances


def observation_from_state(
    *,
    distances: np.ndarray,
    speed: float,
    max_forward_speed: float,
    max_reverse_speed: float,
    sensor_range: float,
    heading_error_norm: float,
    goal_distance_norm: float,
    lateral_offset_norm: float,
    steering_norm: float,
    throttle_norm: float,
) -> np.ndarray:
    distance_obs = np.interp(distances, [0.0, sensor_range], [-1.0, 1.0]).astype(np.float32)
    speed_min = -max_reverse_speed
    speed_max = max_forward_speed
    speed_obs = np.interp(speed, [speed_min, speed_max], [-1.0, 1.0])
    extras = np.array(
        [
            speed_obs,
            np.clip(heading_error_norm, -1.0, 1.0),
            np.clip(goal_distance_norm, -1.0, 1.0),
            np.clip(lateral_offset_norm, -1.0, 1.0),
            np.clip(steering_norm, -1.0, 1.0),
            np.clip(throttle_norm, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate((distance_obs, extras)).astype(np.float32)
