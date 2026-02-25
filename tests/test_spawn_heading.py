from __future__ import annotations

import math

from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv


def test_default_spawn_heading_aligns_with_first_goal() -> None:
    cfg = EnvConfig()
    cfg.render.enabled = False
    cfg.render_mode = None
    cfg.start_heading_deg = None

    env = F1RaceEnv(to_dict(cfg))
    try:
        _, info = env.reset(seed=7)
        goal = env.track.get_goal(1)
        target_x = float((goal[0] + goal[2]) * 0.5)
        target_y = float((goal[1] + goal[3]) * 0.5)
        expected = math.degrees(math.atan2(-(target_y - info["y"]), (target_x - info["x"])))
        assert abs(float(info["heading_deg"]) - expected) < 1e-3
    finally:
        env.close()


def test_explicit_spawn_heading_override_is_respected() -> None:
    cfg = EnvConfig()
    cfg.render.enabled = False
    cfg.render_mode = None
    cfg.start_heading_deg = 45.0

    env = F1RaceEnv(to_dict(cfg))
    try:
        _, info = env.reset(seed=7)
        assert float(info["heading_deg"]) == 45.0
    finally:
        env.close()
