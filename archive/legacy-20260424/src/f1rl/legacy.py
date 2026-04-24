"""Legacy API compatibility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from f1rl.env import F1RaceEnv


class Race(F1RaceEnv):
    """Compatibility wrapper for legacy code expecting old Gym signatures."""

    def __init__(self, env_config: dict[str, Any] | None = None) -> None:
        legacy_config = env_config or {}
        if legacy_config.get("gui", True):
            legacy_config.setdefault("render_mode", "human")
            legacy_config.setdefault("headless", False)
            render_cfg = legacy_config.setdefault("render", {})
            render_cfg.setdefault("enabled", True)
        super().__init__(legacy_config)

    def reset(self):  # type: ignore[override]
        obs, _ = super().reset()
        return obs

    def step(self, action=np.array([0.0, 0.0], dtype=np.float32)):  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(np.asarray(action, dtype=np.float32))
        done = bool(terminated or truncated)
        return obs, reward, done, info
