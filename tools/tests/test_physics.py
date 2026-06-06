import numpy as np

from f1rl.config import CarParams, PhysicsV2Params, SimConfig
from f1rl.physics import (
    CarState,
    apply_physics,
    gear_and_rpm_v2,
    tire_lateral_force_v2,
    weight_transfer_v2,
)


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


def test_sim_config_defaults_to_v1_and_v2_is_explicit() -> None:
    assert SimConfig().physics_model == "v1"
    v2_config = SimConfig(physics_model="v2")
    assert v2_config.physics_model == "v2"
    assert v2_config.physics_calibration_id == v2_config.physics_v2.calibration_id


def test_v2_tire_curve_peaks_and_falls_off() -> None:
    v2 = PhysicsV2Params()
    load = 4_000.0
    reference = 4_000.0
    zero = tire_lateral_force_v2(
        0.0,
        load,
        stiffness_n_per_rad=v2.front_cornering_stiffness_n_per_rad,
        peak_mu=v2.front_peak_mu,
        shape_c=v2.tire_shape_c,
        slip_angle_peak_rad=np.deg2rad(v2.slip_angle_peak_deg),
        post_peak_falloff=v2.post_peak_falloff,
        load_sensitivity=v2.load_sensitivity,
        reference_load_n=reference,
        surface_mu=v2.surface_mu,
    )
    near_peak = tire_lateral_force_v2(
        float(np.deg2rad(v2.slip_angle_peak_deg)),
        load,
        stiffness_n_per_rad=v2.front_cornering_stiffness_n_per_rad,
        peak_mu=v2.front_peak_mu,
        shape_c=v2.tire_shape_c,
        slip_angle_peak_rad=np.deg2rad(v2.slip_angle_peak_deg),
        post_peak_falloff=v2.post_peak_falloff,
        load_sensitivity=v2.load_sensitivity,
        reference_load_n=reference,
        surface_mu=v2.surface_mu,
    )
    past_peak = tire_lateral_force_v2(
        float(np.deg2rad(v2.slip_angle_peak_deg * 3.0)),
        load,
        stiffness_n_per_rad=v2.front_cornering_stiffness_n_per_rad,
        peak_mu=v2.front_peak_mu,
        shape_c=v2.tire_shape_c,
        slip_angle_peak_rad=np.deg2rad(v2.slip_angle_peak_deg),
        post_peak_falloff=v2.post_peak_falloff,
        load_sensitivity=v2.load_sensitivity,
        reference_load_n=reference,
        surface_mu=v2.surface_mu,
    )
    assert abs(zero) < 1e-6
    assert near_peak > 0.0
    assert past_peak < near_peak


def test_v2_weight_transfer_braking_moves_load_forward() -> None:
    params = CarParams()
    v2 = PhysicsV2Params()
    steady_front, steady_rear = weight_transfer_v2(
        params=params,
        v2=v2,
        longitudinal_accel_mps2=0.0,
        lateral_accel_mps2=0.0,
        speed_mps=50.0,
    )
    braking_front, braking_rear = weight_transfer_v2(
        params=params,
        v2=v2,
        longitudinal_accel_mps2=-20.0,
        lateral_accel_mps2=0.0,
        speed_mps=50.0,
    )
    assert braking_front > steady_front
    assert braking_rear < steady_rear
    assert np.isclose(braking_front + braking_rear, steady_front + steady_rear, rtol=0.04)


def test_v2_gear_and_rpm_are_speed_plausible() -> None:
    v2 = PhysicsV2Params()
    low_gear, low_rpm = gear_and_rpm_v2(25.0, 1, v2)
    high_gear, high_rpm = gear_and_rpm_v2(90.0, 1, v2)
    assert 1 <= low_gear <= high_gear <= len(v2.gear_ratios)
    assert v2.idle_rpm <= low_rpm <= v2.max_rpm
    assert v2.idle_rpm <= high_rpm <= v2.max_rpm


def test_v2_physics_populates_diagnostics() -> None:
    params = CarParams()
    v2 = PhysicsV2Params()
    state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=45.0)
    next_state, _ = apply_physics(
        state,
        throttle=0.7,
        brake=0.0,
        steer=0.6,
        params=params,
        meters_per_pixel=1.0,
        physics_model="v2",
        physics_v2=v2,
    )
    assert next_state.speed_mps >= 0.0
    assert next_state.gear >= 1
    assert next_state.rpm >= v2.idle_rpm
    assert next_state.front_load_n > 0.0
    assert next_state.rear_load_n > 0.0
    assert next_state.tire_saturation >= 0.0
