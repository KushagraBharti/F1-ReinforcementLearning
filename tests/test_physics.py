import numpy as np

from f1rl.config import CarParams
from f1rl.physics import CarState, apply_physics


def test_acceleration_and_braking() -> None:
    params = CarParams()
    state = CarState(x=0.0, y=0.0, heading_rad=0.0)
    faster, _ = apply_physics(
        state,
        throttle=1.0,
        brake=0.0,
        steer=0.0,
        params=params,
        meters_per_pixel=1.0,
    )
    slower, _ = apply_physics(
        faster,
        throttle=0.0,
        brake=1.0,
        steer=0.0,
        params=params,
        meters_per_pixel=1.0,
    )
    assert faster.speed_mps > state.speed_mps
    assert slower.speed_mps < faster.speed_mps


def test_steering_changes_heading() -> None:
    params = CarParams()
    state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=35.0)
    turned, _ = apply_physics(
        state,
        throttle=0.0,
        brake=0.0,
        steer=1.0,
        params=params,
        meters_per_pixel=1.0,
    )
    assert not np.isclose(turned.heading_rad, state.heading_rad)
    assert abs(turned.yaw_rate_rps) > 0.0
