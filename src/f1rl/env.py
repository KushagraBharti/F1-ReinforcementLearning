"""Gymnasium environment for the F1-style driving simulator."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from f1rl.config import EnvConfig, to_dict
from f1rl.dynamics import (
    CarState,
    apply_dynamics,
    new_car_state,
    observation_from_distances,
    sensor_angles,
    sensor_distances,
)
from f1rl.geometry import line_intersection, segment_intersects_any
from f1rl.renderer import PygameRenderer
from f1rl.track import TrackDefinition, build_track


class F1RaceEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, env_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = EnvConfig.from_dict(env_config)
        self.track: TrackDefinition = build_track(self.config.track)
        self._default_spawn_heading_deg = self._compute_spawn_heading(self.config.start_pos)
        self.car: CarState = new_car_state(self.config.start_pos, self._resolve_start_heading(None))
        self.renderer: PygameRenderer | None = None
        self.terminated = False
        self.truncated = False
        self.last_step_reward = 0.0
        self.info: dict[str, Any] = {}
        self._render_mode = self.config.render_mode

        sensor_count = sensor_angles(self.config.dynamics).shape[0]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(sensor_count + 1,),
            dtype=np.float32,
        )

    def _compute_spawn_heading(self, start_pos: tuple[float, float]) -> float:
        # Align spawn heading with the direction from spawn to first target checkpoint midpoint.
        goal = self.track.get_goal(1)
        target_x = float((goal[0] + goal[2]) * 0.5)
        target_y = float((goal[1] + goal[3]) * 0.5)
        dx = target_x - float(start_pos[0])
        dy = target_y - float(start_pos[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.degrees(math.atan2(-dy, dx)))

    def _resolve_start_heading(self, option_heading: Any) -> float:
        if option_heading is not None:
            return float(option_heading)
        if self.config.start_heading_deg is not None:
            return float(self.config.start_heading_deg)
        return float(self._default_spawn_heading_deg)

    def _goal_transition(self, movement: np.ndarray) -> tuple[int, bool]:
        next_goal = self.track.get_goal(self.car.next_goal_idx)
        prev_goal = self.track.get_goal(self.car.next_goal_idx - 2)

        crossed_next = (
            line_intersection(
                movement[0],
                movement[1],
                movement[2],
                movement[3],
                next_goal[0],
                next_goal[1],
                next_goal[2],
                next_goal[3],
            )
            is not None
        )
        crossed_prev = (
            line_intersection(
                movement[0],
                movement[1],
                movement[2],
                movement[3],
                prev_goal[0],
                prev_goal[1],
                prev_goal[2],
                prev_goal[3],
            )
            is not None
        )

        lap_finished = False
        if crossed_next:
            self.car.next_goal_idx = (self.car.next_goal_idx + 1) % self.track.num_goals
            if self.car.next_goal_idx == 1:
                self.car.lap_count += 1
                lap_finished = True
            return 1, lap_finished

        if crossed_prev:
            self.car.next_goal_idx = (self.car.next_goal_idx - 1) % self.track.num_goals
            return -1, lap_finished

        return 0, lap_finished

    def _compute_observation(self) -> np.ndarray:
        distances = sensor_distances(self.car, self.config.dynamics, self.track.collision_segments)
        return observation_from_distances(
            distances=distances,
            speed=self.car.speed,
            max_speed=self.config.dynamics.max_speed,
            sensor_range=self.config.dynamics.sensor_range,
        )

    def _build_info(self, collided: bool, progress_delta: int) -> dict[str, Any]:
        return {
            "x": float(self.car.x),
            "y": float(self.car.y),
            "heading_deg": float(self.car.heading_deg),
            "speed": float(self.car.speed),
            "lap_count": int(self.car.lap_count),
            "next_goal_idx": int(self.car.next_goal_idx),
            "collided": collided,
            "progress_delta": progress_delta,
            "step_count": int(self.car.step_count),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "start_pos" in options:
            start_pos = tuple(options["start_pos"])
        else:
            start_pos = self.config.start_pos

        option_heading = options.get("start_heading_deg") if options else None
        start_heading = self._resolve_start_heading(option_heading)
        self.car = new_car_state(start_pos=start_pos, heading_deg=start_heading)
        self.terminated = False
        self.truncated = False
        self.last_step_reward = 0.0
        obs = self._compute_observation()
        self.info = self._build_info(collided=False, progress_delta=0)
        return obs, self.info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended. Call reset() before stepping again.")

        clipped_action = np.asarray(action, dtype=np.float32)
        self.car, movement = apply_dynamics(self.car, clipped_action, self.config.dynamics)
        progress_delta, lap_finished = self._goal_transition(movement)
        collided = segment_intersects_any(movement, self.track.collision_segments)

        reward_cfg = self.config.reward
        reward = reward_cfg.step_penalty + reward_cfg.speed_weight * (
            self.car.speed / max(self.config.dynamics.max_speed, 1e-6)
        )
        if self.car.speed < reward_cfg.idle_speed_threshold:
            reward += reward_cfg.idle_penalty

        if progress_delta > 0:
            reward += reward_cfg.progress_reward
        elif progress_delta < 0:
            reward += reward_cfg.reverse_penalty

        if lap_finished:
            reward += reward_cfg.lap_bonus

        if collided:
            reward += reward_cfg.collision_penalty
            self.car.alive = False
            if self.config.terminate_on_collision:
                self.terminated = True

        if self.car.step_count >= self.config.max_steps:
            self.truncated = True
        if self.car.lap_count >= self.config.max_laps:
            self.truncated = True

        self.car.total_reward += reward
        self.last_step_reward = reward
        obs = self._compute_observation()
        self.info = self._build_info(collided=collided, progress_delta=progress_delta)
        return obs, float(reward), self.terminated, self.truncated, self.info

    def render(self) -> np.ndarray | None:
        if self._render_mode is None and not self.config.render.enabled:
            return None
        if self.renderer is None:
            self.renderer = PygameRenderer(config=self.config, track=self.track)
        return self.renderer.render(
            car_state=self.car,
            total_reward=self.car.total_reward,
            lap_reward=self.last_step_reward,
            next_goal_idx=self.car.next_goal_idx,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_config_dict(self) -> dict[str, Any]:
        return to_dict(self.config)
