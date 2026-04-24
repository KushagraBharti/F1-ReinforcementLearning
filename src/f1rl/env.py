"""Gymnasium wrapper over the shared Monza simulator."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from f1rl.config import DISCRETE_ACTIONS, SimConfig
from f1rl.render import PygameRenderer
from f1rl.sim import MonzaSim
from f1rl.telemetry import StepTelemetry


class MonzaEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: SimConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or SimConfig()
        self.render_mode = render_mode
        self.sim = MonzaSim(self.config)
        obs, _ = self.sim.reset()
        self.last_telemetry: StepTelemetry | None = None
        self.action_space = gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=obs.shape, dtype=np.float32)
        self.renderer: PygameRenderer | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        obs, info = self.sim.reset(seed=seed)
        self.last_telemetry = None
        return obs.astype(np.float32), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.sim.step(int(action))
        self.last_telemetry = result.telemetry
        return (
            result.observation.astype(np.float32),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            result.info,
        )

    def render(self) -> np.ndarray | None:
        if self.renderer is None:
            self.renderer = PygameRenderer(self.sim.track, self.config)
        return self.renderer.render(self.sim, human=self.render_mode == "human")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
