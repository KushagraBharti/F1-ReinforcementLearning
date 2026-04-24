"""Deterministic scripted controllers for manual and smoke workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ScriptedController:
    """Sensor-based deterministic controller."""

    sensor_count: int
    steering_gain: float = 2.2
    steering_d_gain: float = 0.9
    cruise_throttle: float = 0.18
    caution_throttle: float = 0.09
    brake_throttle: float = 0.0
    min_throttle: float = 0.0
    caution_front: float = 0.34
    brake_front: float = 0.18
    high_speed_ratio: float = 0.92
    stuck_speed_ratio: float = 0.04
    stuck_front_ratio: float = 0.18
    stuck_steps: int = 18
    recovery_steer: float = 0.85
    recovery_throttle: float = -0.45
    goals: np.ndarray | None = None
    lookahead_goals: int = 4
    heading_gain: float = 1.35
    heading_d_gain: float = 0.2
    _prev_error: float = 0.0
    _prev_heading_error: float = 0.0
    _stuck_counter: int = 0
    _recovery_direction: int = 1

    def reset(self) -> None:
        self._prev_error = 0.0
        self._prev_heading_error = 0.0
        self._stuck_counter = 0
        self._recovery_direction = 1

    def _normalize_observation(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        sensors = np.asarray(obs[: self.sensor_count], dtype=np.float32)
        sensors = np.clip((sensors + 1.0) * 0.5, 0.0, 1.0)
        speed_ratio = float(np.clip(abs(float(obs[self.sensor_count])), 0.0, 1.0))
        return sensors, speed_ratio

    @staticmethod
    def _wrap_degrees(angle: float) -> float:
        return float((angle + 180.0) % 360.0 - 180.0)

    def _goal_steering(self, info: dict[str, float] | None) -> float | None:
        if self.goals is None or info is None:
            return None
        required_keys = {"x", "y", "heading_deg", "next_goal_idx"}
        if not required_keys.issubset(info):
            return None

        goal_index = (int(info["next_goal_idx"]) + self.lookahead_goals) % self.goals.shape[0]
        goal_line = self.goals[goal_index]
        target_x = float((goal_line[0] + goal_line[2]) * 0.5)
        target_y = float((goal_line[1] + goal_line[3]) * 0.5)
        dx = target_x - float(info["x"])
        dy = target_y - float(info["y"])

        desired_heading = float(np.degrees(np.arctan2(-dy, dx)))
        heading_error = self._wrap_degrees(desired_heading - float(info["heading_deg"]))
        heading_delta = heading_error - self._prev_heading_error
        self._prev_heading_error = heading_error
        return float(
            np.clip(
                self.heading_gain * (heading_error / 45.0) + self.heading_d_gain * (heading_delta / 45.0),
                -1.0,
                1.0,
            )
        )

    def action(self, obs: np.ndarray, info: dict[str, float] | None = None) -> np.ndarray:
        sensors, speed_ratio = self._normalize_observation(obs)
        mid = sensors.shape[0] // 2
        front = float(sensors[mid])
        right_clear = float(np.mean(sensors[:mid])) if mid > 0 else front
        left_clear = float(np.mean(sensors[mid + 1 :])) if mid + 1 < sensors.shape[0] else front

        center_error = left_clear - right_clear
        derivative = center_error - self._prev_error
        self._prev_error = center_error
        sensor_steering = np.clip(
            -(self.steering_gain * center_error + self.steering_d_gain * derivative),
            -1.0,
            1.0,
        )
        goal_steering = self._goal_steering(info)
        if goal_steering is None:
            steering = sensor_steering
        else:
            steering = np.clip(0.3 * sensor_steering + 0.7 * goal_steering, -1.0, 1.0)

        if front < self.brake_front:
            throttle = self.brake_throttle
        elif front < self.caution_front:
            throttle = self.caution_throttle
        else:
            throttle = self.cruise_throttle

        throttle *= max(0.4, 1.0 - 0.45 * abs(float(steering)))
        if speed_ratio > self.high_speed_ratio and front < 0.7:
            throttle = min(throttle, self.caution_throttle)
        throttle = max(self.min_throttle, throttle)

        if speed_ratio < self.stuck_speed_ratio and front < self.stuck_front_ratio:
            self._stuck_counter += 1
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)

        if self._stuck_counter >= self.stuck_steps:
            steering = float(self._recovery_direction * self.recovery_steer)
            throttle = self.recovery_throttle
            self._recovery_direction *= -1
            self._stuck_counter = 0

        return np.array([float(steering), float(throttle)], dtype=np.float32)
