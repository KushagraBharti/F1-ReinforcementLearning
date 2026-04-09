import numpy as np

from f1rl.config import DynamicsConfig
from f1rl.dynamics import apply_dynamics, new_car_state


def test_speed_increases_with_forward_throttle() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    new_state, _ = apply_dynamics(state, np.array([0.0, 1.0], dtype=np.float32), cfg)
    assert new_state.speed > state.speed


def test_reverse_input_produces_non_positive_speed() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    new_state, _ = apply_dynamics(state, np.array([0.0, -1.0], dtype=np.float32), cfg)
    assert new_state.speed <= 0.0
    assert new_state.speed >= -cfg.max_reverse_speed


def test_reverse_input_brakes_forward_motion_before_reversing() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    state.speed = 1.0
    new_state, _ = apply_dynamics(state, np.array([0.0, -1.0], dtype=np.float32), cfg)
    assert new_state.speed >= 0.0


def test_steering_input_changes_heading_and_steering_angle() -> None:
    cfg = DynamicsConfig()
    state = new_car_state((0.0, 0.0), heading_deg=0.0)
    state.speed = 4.0
    new_state, _ = apply_dynamics(state, np.array([1.0, 0.0], dtype=np.float32), cfg)
    assert new_state.heading_deg != state.heading_deg
    assert new_state.steering_angle_deg != state.steering_angle_deg
