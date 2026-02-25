import numpy as np

from f1rl.config import DynamicsConfig
from f1rl.dynamics import apply_dynamics, new_car_state


def test_speed_increases_with_throttle() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    new_state, _ = apply_dynamics(state, np.array([0.0, 1.0], dtype=np.float32), cfg)
    assert new_state.speed > state.speed


def test_speed_clamped_non_negative() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    state.speed = 0.01
    new_state, _ = apply_dynamics(state, np.array([0.0, -1.0], dtype=np.float32), cfg)
    assert new_state.speed >= 0.0
    assert new_state.speed <= cfg.max_speed
