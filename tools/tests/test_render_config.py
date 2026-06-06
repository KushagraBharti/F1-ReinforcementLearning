from __future__ import annotations

from f1rl.config import RenderConfig, SimConfig


def test_render_fps_matches_physics_step() -> None:
    assert RenderConfig().fps == round(1.0 / SimConfig().car.dt)
