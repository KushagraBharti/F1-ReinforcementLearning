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
    observation_from_state,
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
        self.last_reward_components: dict[str, float] = {}
        self.info: dict[str, Any] = {}
        self._render_mode = self.config.render_mode
        self._no_progress_steps = 0
        self._reverse_speed_steps = 0
        self._reverse_progress_events = 0
        self._last_action = np.zeros((2,), dtype=np.float32)
        self.sensor_count = sensor_angles(self.config.dynamics).shape[0]
        self.extra_obs_size = 6

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.sensor_count + self.extra_obs_size,),
            dtype=np.float32,
        )

    def _compute_spawn_heading(self, start_pos: tuple[float, float]) -> float:
        goal_midpoint = self.track.goal_midpoint(1)
        dx = float(goal_midpoint[0] - float(start_pos[0]))
        dy = float(goal_midpoint[1] - float(start_pos[1]))
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.degrees(math.atan2(-dy, dx)))

    def _resolve_start_heading(self, option_heading: Any) -> float:
        if option_heading is not None:
            return float(option_heading)
        if self.config.start_heading_deg is not None:
            return float(self.config.start_heading_deg)
        return float(self._default_spawn_heading_deg)

    @staticmethod
    def _wrap_degrees(angle: float) -> float:
        return float((angle + 180.0) % 360.0 - 180.0)

    def _track_state(self) -> tuple[float, float, float]:
        goal_midpoint = self.track.goal_midpoint(self.car.next_goal_idx)
        dx = float(goal_midpoint[0] - self.car.x)
        dy = float(goal_midpoint[1] - self.car.y)
        desired_heading = float(math.degrees(math.atan2(-dy, dx)))
        heading_error = self._wrap_degrees(desired_heading - self.car.heading_deg)
        heading_error_norm = float(np.clip(heading_error / 90.0, -1.0, 1.0))

        goal_distance = float(np.hypot(dx, dy))
        goal_distance_norm = float(np.clip(goal_distance / self.config.dynamics.sensor_range, 0.0, 1.0) * 2.0 - 1.0)

        centerline_offset = self.track.centerline_offset(self.car.x, self.car.y)
        centerline_scale = max(self.track.average_track_width * 0.5, 1e-6)
        lateral_offset_norm = float(np.clip(centerline_offset / centerline_scale, 0.0, 1.0) * 2.0 - 1.0)
        return heading_error_norm, goal_distance_norm, lateral_offset_norm

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
        heading_error_norm, goal_distance_norm, lateral_offset_norm = self._track_state()
        return observation_from_state(
            distances=distances,
            speed=self.car.speed,
            max_forward_speed=self.config.dynamics.max_speed,
            max_reverse_speed=self.config.dynamics.max_reverse_speed,
            sensor_range=self.config.dynamics.sensor_range,
            heading_error_norm=heading_error_norm,
            goal_distance_norm=goal_distance_norm,
            lateral_offset_norm=lateral_offset_norm,
            steering_norm=self.car.last_action[0],
            throttle_norm=self.car.last_action[1],
        )

    def _build_info(self, collided: bool, progress_delta: int, terminated_reason: str | None = None) -> dict[str, Any]:
        heading_error_norm, goal_distance_norm, lateral_offset_norm = self._track_state()
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
            "heading_error_norm": heading_error_norm,
            "goal_distance_norm": goal_distance_norm,
            "lateral_offset_norm": lateral_offset_norm,
            "no_progress_steps": int(self._no_progress_steps),
            "reverse_speed_steps": int(self._reverse_speed_steps),
            "reverse_progress_events": int(self._reverse_progress_events),
            "reward_components": dict(self.last_reward_components),
            "terminated_reason": terminated_reason,
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
        self.last_reward_components = {}
        self._no_progress_steps = 0
        self._reverse_speed_steps = 0
        self._reverse_progress_events = 0
        self._last_action = np.zeros((2,), dtype=np.float32)
        obs = self._compute_observation()
        self.info = self._build_info(collided=False, progress_delta=0)
        return obs, self.info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended. Call reset() before stepping again.")

        clipped_action = np.asarray(action, dtype=np.float32)
        previous_action = self._last_action.copy()
        self.car, movement = apply_dynamics(self.car, clipped_action, self.config.dynamics)
        self._last_action = clipped_action
        progress_delta, lap_finished = self._goal_transition(movement)
        collided = segment_intersects_any(movement, self.track.collision_segments)
        heading_error_norm, _, lateral_offset_norm = self._track_state()

        reward_cfg = self.config.reward
        reward_components = {
            "step_penalty": reward_cfg.step_penalty,
            "progress": 0.0,
            "collision": 0.0,
            "speed": 0.0,
            "alignment": 0.0,
            "centerline": 0.0,
            "stall": 0.0,
            "steering_change": 0.0,
            "lap_bonus": 0.0,
            "reverse": 0.0,
            "negative_speed": 0.0,
        }

        forward_speed_ratio = max(self.car.speed, 0.0) / max(self.config.dynamics.max_speed, 1e-6)
        reward_components["speed"] = reward_cfg.forward_speed_weight * forward_speed_ratio * max(
            0.0, 1.0 - abs(heading_error_norm)
        )
        if self.car.speed < 0.0:
            reverse_speed_ratio = min(abs(self.car.speed) / max(self.config.dynamics.max_reverse_speed, 1e-6), 1.0)
            reward_components["negative_speed"] = -reward_cfg.negative_speed_penalty_weight * reverse_speed_ratio
        reward_components["alignment"] = reward_cfg.alignment_weight * max(0.0, 1.0 - abs(heading_error_norm))
        reward_components["centerline"] = -reward_cfg.centerline_penalty_weight * max(0.0, lateral_offset_norm)
        reward_components["steering_change"] = -reward_cfg.steering_change_penalty_weight * abs(
            float(clipped_action[0] - previous_action[0])
        )

        if progress_delta > 0:
            reward_components["progress"] = reward_cfg.progress_reward * progress_delta
        elif progress_delta < 0:
            reward_components["reverse"] = reward_cfg.reverse_penalty * abs(progress_delta)

        if lap_finished:
            reward_components["lap_bonus"] = reward_cfg.lap_bonus

        if abs(self.car.speed) < reward_cfg.stall_speed_threshold:
            reward_components["stall"] += reward_cfg.idle_penalty

        if progress_delta > 0 or abs(self.car.speed) >= reward_cfg.stall_speed_threshold:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1
            reward_components["stall"] += reward_cfg.stall_penalty

        if self.car.speed <= -self.config.reverse_speed_threshold:
            self._reverse_speed_steps += 1
        else:
            self._reverse_speed_steps = 0

        if progress_delta < 0:
            self._reverse_progress_events += 1
        elif progress_delta > 0 or self.car.speed >= 0.0:
            self._reverse_progress_events = 0

        terminated_reason = None
        if collided:
            reward_components["collision"] = reward_cfg.collision_penalty
            self.car.alive = False
            terminated_reason = "collision"
            if self.config.terminate_on_collision:
                self.terminated = True

        if self.config.terminate_on_no_progress and self._no_progress_steps >= self.config.no_progress_limit_steps:
            self.terminated = True
            terminated_reason = terminated_reason or "no_progress"

        if self._reverse_speed_steps >= self.config.reverse_speed_limit_steps:
            self.terminated = True
            terminated_reason = terminated_reason or "reverse_speed"

        if self._reverse_progress_events >= self.config.reverse_progress_limit:
            self.terminated = True
            terminated_reason = terminated_reason or "reverse_progress"

        if self.car.step_count >= self.config.max_steps:
            self.truncated = True
            terminated_reason = terminated_reason or "max_steps"
        if self.car.lap_count >= self.config.max_laps:
            self.truncated = True
            terminated_reason = terminated_reason or "lap_complete"

        reward = float(sum(reward_components.values()))
        self.last_reward_components = reward_components
        self.car.total_reward += reward
        self.last_step_reward = reward
        obs = self._compute_observation()
        self.info = self._build_info(
            collided=collided,
            progress_delta=progress_delta,
            terminated_reason=terminated_reason,
        )
        return obs, reward, self.terminated, self.truncated, self.info

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
            extra_hud={
                "heading_err": f"{self.info.get('heading_error_norm', 0.0):.2f}",
                "offset": f"{self.info.get('lateral_offset_norm', 0.0):.2f}",
                "reason": self.info.get("terminated_reason") or "-",
            },
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_config_dict(self) -> dict[str, Any]:
        return to_dict(self.config)
