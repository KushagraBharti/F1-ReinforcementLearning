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
        next_goal_idx=1,
        lap_count=0,
        step_count=0,
    )


def apply_dynamics(state: CarState, action: np.ndarray, config: DynamicsConfig) -> tuple[CarState, np.ndarray]:
    steering = float(np.clip(action[0], -1.0, 1.0))
    throttle = float(np.clip(action[1], -1.0, 1.0))

    speed = state.speed
    if throttle >= 0.0:
        speed += throttle * config.acceleration_rate * config.dt
    else:
        speed += throttle * config.braking_rate * config.dt
    speed *= max(0.0, 1.0 - config.friction)
    speed = float(np.clip(speed, 0.0, config.max_speed))

    steering_scale = 0.3 + 0.7 * (speed / max(config.max_speed, 1e-6))
    heading_deg = state.heading_deg + steering * config.steering_rate_deg * steering_scale * config.dt
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
        step_count=state.step_count + 1,
        last_action=(steering, throttle),
    )
    return new_state, movement


def sensor_angles(config: DynamicsConfig) -> np.ndarray:
    count = int(max(3, config.sensor_count))
    if count % 2 == 0:
        count += 1
    return np.linspace(
        -config.sensor_spread_deg / 2.0,
        config.sensor_spread_deg / 2.0,
        count,
        dtype=np.float32,
    )


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


def observation_from_distances(
    distances: np.ndarray,
    speed: float,
    max_speed: float,
    sensor_range: float,
) -> np.ndarray:
    distance_obs = np.interp(distances, [0.0, sensor_range], [-1.0, 1.0])
    speed_obs = np.interp(speed, [0.0, max_speed], [-1.0, 1.0])
    obs = np.concatenate((distance_obs.astype(np.float32), np.array([speed_obs], dtype=np.float32)))
    return obs.astype(np.float32)
