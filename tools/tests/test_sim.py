import numpy as np

from f1rl.config import (
    BRAKE_COAST_DISCRETE_ACTIONS,
    BRAKE_RELEASE_DISCRETE_ACTIONS,
    BRAKE_STRAIGHT_DISCRETE_ACTIONS,
    DELAYED_LEFT_DISCRETE_ACTIONS,
    DELAYED_TURN_DISCRETE_ACTIONS,
    DISCRETE_ACTIONS,
    EXIT_TINY_DISCRETE_ACTIONS,
    EXPANDED_DISCRETE_ACTIONS,
    LEARNED_POLICY_V1_FEATURES,
    LEGACY_DISCRETE_ACTIONS,
    RACING_DISCRETE_ACTIONS,
    RELEASE_BRAKE_DISCRETE_ACTIONS,
    STABILIZE_RIGHT_DISCRETE_ACTIONS,
    STRAIGHT_DISCRETE_ACTIONS,
    TURNIN_MICRO_DISCRETE_ACTIONS,
    TURNIN_POWER_DISCRETE_ACTIONS,
    AssistConfig,
    RewardConfig,
    SimConfig,
    action_to_controls,
    multidiscrete_action_nvec,
)
from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS


def _racing_signed_lateral_index(sim: MonzaSim) -> int:
    return 7 + len(sim.sensor_angles) + len(sim.config.lookahead_m) + 3 + 2


def _racing_v2_section_feature_index(sim: MonzaSim) -> int:
    racing_core_features = 1 + 2 + len(sim.config.lookahead_m) + 1
    return _racing_signed_lateral_index(sim) + racing_core_features


def test_discrete_action_table_preserves_original_ids_and_adds_soft_controls() -> None:
    original_actions = (
        ("coast", 0.0, 0.0, 0.0),
        ("throttle", 1.0, 0.0, 0.0),
        ("brake", 0.0, 1.0, 0.0),
        ("left", 0.0, 0.0, -1.0),
        ("right", 0.0, 0.0, 1.0),
        ("throttle_left", 1.0, 0.0, -1.0),
        ("throttle_right", 1.0, 0.0, 1.0),
        ("brake_left", 0.0, 1.0, -1.0),
        ("brake_right", 0.0, 1.0, 1.0),
    )
    assert LEGACY_DISCRETE_ACTIONS == original_actions
    assert DISCRETE_ACTIONS == LEGACY_DISCRETE_ACTIONS
    assert EXPANDED_DISCRETE_ACTIONS[: len(original_actions)] == original_actions
    assert ("half_throttle_soft_left", 0.5, 0.0, -0.45) in EXPANDED_DISCRETE_ACTIONS
    assert ("soft_brake_right", 0.0, 0.35, 0.45) in EXPANDED_DISCRETE_ACTIONS
    assert ("coast", 0.0, 0.0, 0.0) not in RACING_DISCRETE_ACTIONS
    assert ("left", 0.0, 0.0, -1.0) not in RACING_DISCRETE_ACTIONS
    assert ("brake_left", 0.0, 1.0, -1.0) not in BRAKE_STRAIGHT_DISCRETE_ACTIONS
    assert STRAIGHT_DISCRETE_ACTIONS == (
        ("coast", 0.0, 0.0, 0.0),
        ("throttle", 1.0, 0.0, 0.0),
        ("brake", 0.0, 1.0, 0.0),
    )
    assert BRAKE_COAST_DISCRETE_ACTIONS == (
        ("brake", 0.0, 1.0, 0.0),
        ("coast", 0.0, 0.0, 0.0),
    )
    assert BRAKE_RELEASE_DISCRETE_ACTIONS == (
        ("brake", 0.0, 1.0, 0.0),
        ("trail_brake", 0.0, 0.35, 0.0),
        ("coast", 0.0, 0.0, 0.0),
    )
    assert RELEASE_BRAKE_DISCRETE_ACTIONS == (
        ("coast", 0.0, 0.0, 0.0),
        ("trail_brake", 0.0, 0.35, 0.0),
        ("brake", 0.0, 1.0, 0.0),
    )
    assert all(action[3] == 0.0 for action in STRAIGHT_DISCRETE_ACTIONS)
    assert action_to_controls(2, action_set="brake_straight") == (0.0, 1.0, 0.0)
    assert action_to_controls(2, action_set="straight") == (0.0, 1.0, 0.0)
    assert action_to_controls(0, action_set="brake_coast") == (0.0, 1.0, 0.0)
    assert action_to_controls(1, action_set="brake_coast") == (0.0, 0.0, 0.0)
    assert action_to_controls(1, action_set="brake_release") == (0.0, 0.35, 0.0)
    assert action_to_controls(0, action_set="release_brake") == (0.0, 0.0, 0.0)
    assert action_to_controls(0, action_set="turnin_power") == (0.22, 0.0, 0.0)
    assert action_to_controls(4, action_set="turnin_power") == (0.5, 0.0, -0.45)
    assert action_to_controls(0, action_set="turnin_micro") == (0.22, 0.0, 0.0)
    assert action_to_controls(4, action_set="turnin_micro") == (0.5, 0.0, -0.15)
    assert action_to_controls(1, action_set="delayed_turn") == (0.22, 0.0, -0.12)
    assert action_to_controls(7, action_set="delayed_turn") == (0.0, 0.25, 0.0)
    assert action_to_controls(1, action_set="delayed_left") == (0.22, 0.0, -0.12)
    assert all(action[3] <= 0.0 for action in DELAYED_LEFT_DISCRETE_ACTIONS)
    assert action_to_controls(0, action_set="stabilize_right") == (0.22, 0.0, 0.0)
    assert action_to_controls(2, action_set="stabilize_right") == (0.22, 0.0, 0.12)
    assert all(action[3] >= 0.0 for action in STABILIZE_RIGHT_DISCRETE_ACTIONS)
    assert action_to_controls(0, action_set="exit_tiny") == (0.22, 0.0, -0.06)
    assert action_to_controls(8, action_set="exit_tiny") == (0.0, 0.25, 0.0)
    assert all(action[3] != 0.0 or action[2] > 0.0 for action in EXIT_TINY_DISCRETE_ACTIONS)
    assert action_to_controls(14, action_set="expanded") == (0.5, 0.0, -0.45)


def test_sim_action_set_controls_action_dimension() -> None:
    assert MonzaSim(SimConfig(action_set="legacy")).action_dim == 9
    assert MonzaSim(SimConfig(action_set="brake_coast")).action_dim == 2
    assert MonzaSim(SimConfig(action_set="brake_release")).action_dim == 3
    assert MonzaSim(SimConfig(action_set="release_brake")).action_dim == 3
    assert MonzaSim(SimConfig(action_set="brake_straight")).action_dim == 5
    assert MonzaSim(SimConfig(action_set="delayed_left")).action_dim == len(DELAYED_LEFT_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="delayed_turn")).action_dim == len(DELAYED_TURN_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="stabilize_right")).action_dim == len(STABILIZE_RIGHT_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="straight")).action_dim == 3
    assert MonzaSim(SimConfig(action_set="expanded")).action_dim == 21
    assert MonzaSim(SimConfig(action_set="exit_tiny")).action_dim == len(EXIT_TINY_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="racing")).action_dim == len(RACING_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="turnin_micro")).action_dim == len(TURNIN_MICRO_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_set="turnin_power")).action_dim == len(TURNIN_POWER_DISCRETE_ACTIONS)
    assert MonzaSim(SimConfig(action_mode="continuous")).action_dim == 2
    assert MonzaSim(SimConfig(action_mode="multidiscrete")).action_dim == 25
    assert multidiscrete_action_nvec() == (5, 5)


def test_sim_observation_and_reward_schema() -> None:
    sim = MonzaSim()
    obs, info = sim.reset(seed=1)
    assert obs.shape == (sim.observation_dim,)
    assert set(info["reward_components"]) == set(REWARD_COMPONENT_KEYS)
    assert info["valid_lap"] is True
    assert info["finish_crossed"] is False
    assert info["next_checkpoint_index"] >= 1
    result = sim.step(1)
    assert result.observation.shape == (sim.observation_dim,)
    assert set(result.telemetry.reward_components) == set(REWARD_COMPONENT_KEYS)
    assert len(result.telemetry.ray_distances_m) == sim.config.sensors.count
    assert result.telemetry.next_checkpoint_index >= 1
    assert result.telemetry.missed_checkpoint_count == 0


def test_racing_observation_profile_values_are_bounded() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing"))
    obs, _ = sim.reset(seed=1)
    assert obs.shape == (sim.observation_dim,)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    for action_id in (1, 5, 8, 2):
        result = sim.step(action_id)
        assert result.observation.shape == (sim.observation_dim,)
        assert np.all(result.observation >= -1.0)
        assert np.all(result.observation <= 1.0)
        if result.terminated or result.truncated:
            break


def test_racing_v2_observation_profile_values_are_bounded() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing_v2"))
    obs, _ = sim.reset(seed=1)
    assert obs.shape == (sim.observation_dim,)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    for action_id in (1, 5, 8, 2):
        result = sim.step(action_id)
        assert result.observation.shape == (sim.observation_dim,)
        assert np.all(result.observation >= -1.0)
        assert np.all(result.observation <= 1.0)
        if result.terminated or result.truncated:
            break


def test_learned_policy_v1_observation_profile_values_are_bounded() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="learned_policy_v1"))
    obs, _ = sim.reset(seed=1)
    racing_v2_dim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing_v2")).observation_dim

    assert obs.shape == (sim.observation_dim,)
    assert sim.observation_dim == racing_v2_dim + len(LEARNED_POLICY_V1_FEATURES)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    for action_id in (1, 5, 8, 2):
        result = sim.step(action_id)
        assert result.observation.shape == (sim.observation_dim,)
        assert np.all(result.observation >= -1.0)
        assert np.all(result.observation <= 1.0)
        if result.terminated or result.truncated:
            break


def test_racing_release_observation_profile_values_are_bounded() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing_release"))
    obs, _ = sim.reset(seed=1)
    assert obs.shape == (sim.observation_dim,)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    assert sim.observation_dim == MonzaSim(SimConfig(max_steps=20, observation_profile="racing")).observation_dim + 4


def test_racing_release_observation_marks_release_band() -> None:
    config = SimConfig(
        max_steps=20,
        observation_profile="racing_release",
        reward=RewardConfig(
            scaffold_release_min_speed_kph=135.0,
            scaffold_release_max_speed_kph=185.0,
        ),
    )
    fast = MonzaSim(config)
    fast.reset(seed=1, options={"start_progress_m": 530.0, "start_speed_kph": 260.0})
    fast_threshold, fast_surplus, fast_deficit, fast_in_band = fast._release_observation_features()

    release = MonzaSim(config)
    release.reset(seed=1, options={"start_progress_m": 600.0, "start_speed_kph": 170.0})
    release_threshold, release_surplus, release_deficit, release_in_band = release._release_observation_features()

    assert fast_threshold == release_threshold
    assert fast_surplus > 0.0
    assert fast_deficit < 0.0
    assert fast_in_band < 0.0
    assert release_surplus < 0.0
    assert release_deficit < 0.0
    assert release_in_band > 0.0


def test_racing_v2_observation_profile_marks_rettifilo_brake_zone() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing_v2"))
    obs, _ = sim.reset(seed=1, options={"start_progress_m": 600.0, "start_speed_kph": 260.0})
    section_index = _racing_v2_section_feature_index(sim)
    section_target_speed = float(obs[section_index])
    section_speed_surplus = float(obs[section_index + 1])
    brake_zone_flag = float(obs[section_index + 2])
    brake_zone_phase = float(obs[section_index + 3])

    assert section_target_speed < 0.0
    assert section_speed_surplus > 0.0
    assert brake_zone_flag == 1.0
    assert np.isclose(brake_zone_phase, -0.2)


def test_racing_observation_profile_signed_lateral_flips_on_opposite_sides() -> None:
    sim = MonzaSim(SimConfig(max_steps=20, observation_profile="racing"))
    sim.reset(seed=1)
    progress_m = 600.0
    point, heading = sim.centerline_pose_at(progress_m)
    direction = np.asarray([np.cos(heading), -np.sin(heading)], dtype=np.float32)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    offset_px = normal * (10.0 / sim.track.meters_per_pixel)
    signed_lateral_index = _racing_signed_lateral_index(sim)

    sim.state.x = float(point[0] + offset_px[0])
    sim.state.y = float(point[1] + offset_px[1])
    sim.state.heading_rad = heading
    sim.state.raw_progress_m = progress_m
    sim.state.monotonic_progress_m = progress_m
    sim._last_raw_progress_px = progress_m / sim.track.meters_per_pixel
    positive_side = float(sim.observation()[signed_lateral_index])

    sim.state.x = float(point[0] - offset_px[0])
    sim.state.y = float(point[1] - offset_px[1])
    sim.state.heading_rad = heading
    sim.state.raw_progress_m = progress_m
    sim.state.monotonic_progress_m = progress_m
    sim._last_raw_progress_px = progress_m / sim.track.meters_per_pixel
    negative_side = float(sim.observation()[signed_lateral_index])

    assert positive_side > 0.1
    assert negative_side < -0.1


def test_scaffold_reward_components_are_zero_by_default() -> None:
    sim = MonzaSim(SimConfig(max_steps=20))
    sim.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    result = sim.step_controls(throttle=1.0, brake=0.0, steer=0.0)
    components = result.telemetry.reward_components
    assert components["scaffold_brake"] == 0.0
    assert components["scaffold_no_throttle"] == 0.0
    assert components["scaffold_turn_in_speed"] == 0.0
    assert components["scaffold_apex_clean"] == 0.0
    assert components["scaffold_exit_alignment"] == 0.0
    assert components["scaffold_exit_speed"] == 0.0
    assert components["scaffold_release"] == 0.0
    assert components["scaffold_overbrake"] == 0.0
    assert components["scaffold_brake_curve"] == 0.0
    assert components["scaffold_corridor_center"] == 0.0
    assert components["scaffold_segment_speed"] == 0.0


def test_scaffold_rewards_credit_braking_and_penalize_throttle_in_brake_zone() -> None:
    config = SimConfig(
        max_steps=20,
        reward=RewardConfig(
            scaffold_brake_reward_scale=1.0,
            scaffold_no_throttle_penalty_scale=1.0,
        ),
    )
    braking = MonzaSim(config)
    braking.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    braking_result = braking.step_controls(throttle=0.0, brake=1.0, steer=0.0)

    throttle = MonzaSim(config)
    throttle.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    throttle_result = throttle.step_controls(throttle=1.0, brake=0.0, steer=0.0)

    assert braking_result.telemetry.reward_components["scaffold_brake"] > 0.0
    assert braking_result.telemetry.reward_components["scaffold_no_throttle"] == 0.0
    assert throttle_result.telemetry.reward_components["scaffold_brake"] == 0.0
    assert throttle_result.telemetry.reward_components["scaffold_no_throttle"] < 0.0


def test_release_scaffold_stops_brake_credit_inside_release_window() -> None:
    config = SimConfig(
        max_steps=20,
        reward=RewardConfig(
            scaffold_brake_reward_scale=1.0,
            scaffold_release_reward_scale=1.0,
            scaffold_release_min_speed_kph=135.0,
            scaffold_release_max_speed_kph=210.0,
        ),
    )
    too_fast = MonzaSim(config)
    too_fast.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    too_fast_result = too_fast.step_controls(throttle=0.0, brake=1.0, steer=0.0)

    release_speed = MonzaSim(config)
    release_speed.reset(seed=1, options={"start_progress_m": 600.0, "start_speed_kph": 170.0})
    release_speed_result = release_speed.step_controls(throttle=0.0, brake=1.0, steer=0.0)

    assert too_fast_result.telemetry.reward_components["scaffold_brake"] > 0.0
    assert release_speed_result.telemetry.reward_components["scaffold_brake"] == 0.0


def test_scaffold_rewards_release_and_penalize_overbraking_in_brake_zone() -> None:
    config = SimConfig(
        max_steps=20,
        reward=RewardConfig(
            scaffold_release_reward_scale=1.0,
            scaffold_overbrake_penalty_scale=1.0,
            scaffold_release_min_speed_kph=135.0,
            scaffold_release_max_speed_kph=210.0,
        ),
    )
    release = MonzaSim(config)
    release.reset(seed=1, options={"start_progress_m": 600.0, "start_speed_kph": 170.0})
    release_result = release.step_controls(throttle=0.0, brake=0.0, steer=0.0)

    overbrake = MonzaSim(config)
    overbrake.reset(seed=1, options={"start_progress_m": 620.0, "start_speed_kph": 90.0})
    overbrake_result = overbrake.step_controls(throttle=0.0, brake=1.0, steer=0.0)

    assert release_result.telemetry.reward_components["scaffold_release"] > 0.0
    assert release_result.telemetry.reward_components["scaffold_overbrake"] == 0.0
    assert overbrake_result.telemetry.reward_components["scaffold_release"] == 0.0
    assert overbrake_result.telemetry.reward_components["scaffold_overbrake"] < 0.0


def test_brake_curve_scaffold_penalizes_speed_profile_deviation() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            reward=RewardConfig(
                scaffold_brake_curve_penalty_scale=100.0,
                scaffold_brake_curve_start_speed_kph=335.0,
                scaffold_brake_curve_deadzone_kph=5.0,
            ),
        )
    )

    near_curve = sim._scaffold_reward_components(
        progress_m=650.0,
        speed_kph=192.0,
        throttle=0.0,
        brake=0.0,
        lateral_error_m=0.5,
        heading_error_deg=0.0,
        min_ray_m=50.0,
        collided=False,
        off_track=False,
    )
    too_slow = sim._scaffold_reward_components(
        progress_m=650.0,
        speed_kph=135.0,
        throttle=0.0,
        brake=0.0,
        lateral_error_m=0.5,
        heading_error_deg=0.0,
        min_ray_m=50.0,
        collided=False,
        off_track=False,
    )
    too_fast = sim._scaffold_reward_components(
        progress_m=650.0,
        speed_kph=260.0,
        throttle=0.0,
        brake=0.0,
        lateral_error_m=0.5,
        heading_error_deg=0.0,
        min_ray_m=50.0,
        collided=False,
        off_track=False,
    )

    assert near_curve["scaffold_brake_curve"] == 0.0
    assert too_slow["scaffold_brake_curve"] < 0.0
    assert too_fast["scaffold_brake_curve"] < 0.0


def test_segment_speed_scaffold_penalizes_target_band_miss_near_segment_target() -> None:
    config = SimConfig(
        max_steps=20,
        reward=RewardConfig(scaffold_segment_speed_penalty_scale=100.0),
    )
    too_slow = MonzaSim(config)
    too_slow.reset(
        seed=1,
        options={
            "start_progress_m": 900.0,
            "start_speed_kph": 80.0,
            "segment_length_m": 30.0,
            "segment_target_min_speed_kph": 105.0,
            "segment_target_max_speed_kph": 165.0,
        },
    )
    too_slow_result = too_slow.step_controls(throttle=0.0, brake=0.0, steer=0.0)

    in_band = MonzaSim(config)
    in_band.reset(
        seed=1,
        options={
            "start_progress_m": 900.0,
            "start_speed_kph": 130.0,
            "segment_length_m": 30.0,
            "segment_target_min_speed_kph": 105.0,
            "segment_target_max_speed_kph": 165.0,
        },
    )
    in_band_result = in_band.step_controls(throttle=0.0, brake=0.0, steer=0.0)

    assert too_slow_result.telemetry.reward_components["scaffold_segment_speed"] < 0.0
    assert in_band_result.telemetry.reward_components["scaffold_segment_speed"] == 0.0


def test_corridor_center_scaffold_penalizes_exit_lateral_drift() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            reward=RewardConfig(
                scaffold_corridor_center_penalty_scale=100.0,
                scaffold_corridor_center_deadzone_m=2.0,
            ),
        )
    )

    centered = sim._scaffold_reward_components(
        progress_m=800.0,
        speed_kph=120.0,
        throttle=0.0,
        brake=0.0,
        lateral_error_m=1.0,
        heading_error_deg=0.0,
        min_ray_m=50.0,
        collided=False,
        off_track=False,
    )
    drifting = sim._scaffold_reward_components(
        progress_m=800.0,
        speed_kph=120.0,
        throttle=0.0,
        brake=0.0,
        lateral_error_m=12.0,
        heading_error_deg=0.0,
        min_ray_m=50.0,
        collided=False,
        off_track=False,
    )

    assert centered["scaffold_corridor_center"] == 0.0
    assert drifting["scaffold_corridor_center"] < 0.0


def test_assist_components_are_zero_by_default() -> None:
    sim = MonzaSim(SimConfig(max_steps=20))
    sim.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    result = sim.step_controls(throttle=1.0, brake=0.0, steer=0.0)
    components = result.telemetry.reward_components
    assert components["assist_overspeed_gate"] == 0.0
    assert components["assist_throttle_brake_demand"] == 0.0
    assert components["assist_no_brake_gate"] == 0.0
    assert components["assist_overbrake_gate"] == 0.0
    assert components["assist_steering_gate"] == 0.0
    assert components["assist_forbidden_steering_gate"] == 0.0
    assert components["assist_virtual_corridor"] == 0.0
    assert components["assist_brake_zone_progress_suppression"] == 0.0


def test_assist_penalizes_throttle_and_no_brake_in_brake_zone() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                throttle_brake_demand_penalty_scale=1.0,
                no_brake_penalty=-5.0,
                no_brake_min_brake=0.1,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 520.0, "start_speed_kph": 260.0})
    result = sim.step_controls(throttle=1.0, brake=0.0, steer=0.0)
    components = result.telemetry.reward_components
    assert components["assist_throttle_brake_demand"] < 0.0
    assert components["assist_no_brake_gate"] == -5.0


def test_assist_can_terminate_throttle_in_brake_demand() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                throttle_brake_demand_penalty_scale=10.0,
                throttle_brake_demand_terminate=True,
                throttle_brake_demand_min_throttle=0.5,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 530.0, "start_speed_kph": 300.0})
    result = sim.step_controls(throttle=1.0, brake=0.0, steer=0.0)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_throttle_brake_demand"
    assert result.telemetry.reward_components["assist_throttle_brake_demand"] < 0.0


def test_assist_no_brake_gate_can_require_brake_above_configured_speed() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                no_brake_penalty=-44.0,
                no_brake_min_brake=0.5,
                no_brake_min_speed_kph=185.0,
                no_brake_terminate=True,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 530.0, "start_speed_kph": 260.0})
    result = sim.step_controls(throttle=0.0, brake=0.0, steer=0.0)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_no_brake_gate"
    assert result.telemetry.reward_components["assist_no_brake_gate"] == -44.0


def test_assist_no_brake_gate_allows_release_below_configured_speed() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                no_brake_penalty=-44.0,
                no_brake_min_brake=0.5,
                no_brake_min_speed_kph=185.0,
                no_brake_terminate=True,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 600.0, "start_speed_kph": 170.0})
    result = sim.step_controls(throttle=0.0, brake=0.0, steer=0.0)
    assert result.terminated is False
    assert result.telemetry.reward_components["assist_no_brake_gate"] == 0.0


def test_assist_steering_gate_can_require_signed_steer_in_window() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                steering_gate_penalty=-33.0,
                steering_gate_terminate=True,
                steering_gate_start_m=850.0,
                steering_gate_end_m=930.0,
                steering_gate_min_abs_steer=0.05,
                steering_gate_required_sign=-1.0,
                steering_gate_min_speed_kph=80.0,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 880.0, "start_speed_kph": 160.0})
    result = sim.step_controls(throttle=0.22, brake=0.0, steer=0.0)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_steering_gate"
    assert result.telemetry.reward_components["assist_steering_gate"] == -33.0

    sim.reset(seed=1, options={"start_progress_m": 880.0, "start_speed_kph": 160.0})
    allowed = sim.step_controls(throttle=0.22, brake=0.0, steer=-0.06)
    assert allowed.terminated is False
    assert allowed.telemetry.reward_components["assist_steering_gate"] == 0.0


def test_assist_forbidden_steering_gate_can_block_signed_steer_in_window() -> None:
    config = SimConfig(
        max_steps=20,
        assist=AssistConfig(
            enabled=True,
            forbidden_steering_gate_penalty=-44.0,
            forbidden_steering_gate_terminate=True,
            forbidden_steering_gate_start_m=900.0,
            forbidden_steering_gate_end_m=930.0,
            forbidden_steering_gate_min_abs_steer=0.05,
            forbidden_steering_gate_sign=-1.0,
            forbidden_steering_gate_min_speed_kph=80.0,
        ),
    )
    sim = MonzaSim(config)
    sim.reset(seed=1, options={"start_progress_m": 910.0, "start_speed_kph": 150.0})
    result = sim.step_controls(throttle=0.2, brake=0.0, steer=-0.12)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_forbidden_steering_gate"
    assert result.telemetry.reward_components["assist_forbidden_steering_gate"] == -44.0

    allowed = MonzaSim(config)
    allowed.reset(seed=1, options={"start_progress_m": 910.0, "start_speed_kph": 150.0})
    allowed_result = allowed.step_controls(throttle=0.2, brake=0.0, steer=0.0)
    assert allowed_result.telemetry.reward_components["assist_forbidden_steering_gate"] == 0.0


def test_assist_can_suppress_brake_zone_progress_reward() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                brake_zone_progress_multiplier=0.0,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 530.0, "start_speed_kph": 300.0})
    result = sim.step_controls(throttle=0.0, brake=1.0, steer=0.0)
    components = result.telemetry.reward_components
    assert components["progress"] > 0.0
    assert np.isclose(components["assist_brake_zone_progress_suppression"], -components["progress"])


def test_assist_overbrake_gate_can_terminate() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                overbrake_penalty=-33.0,
                overbrake_terminate=True,
                overbrake_max_speed_kph=135.0,
                overbrake_min_brake=0.5,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 620.0, "start_speed_kph": 90.0})
    result = sim.step_controls(throttle=0.0, brake=1.0, steer=0.0)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_overbrake_gate"
    assert result.telemetry.reward_components["assist_overbrake_gate"] == -33.0


def test_assist_overspeed_turn_in_gate_can_terminate() -> None:
    sim = MonzaSim(
        SimConfig(
            max_steps=20,
            assist=AssistConfig(
                enabled=True,
                overspeed_turn_in_terminate=True,
                overspeed_turn_in_margin_kph=10.0,
                overspeed_turn_in_penalty=-12.0,
            ),
        )
    )
    sim.reset(seed=1, options={"start_progress_m": 718.0, "start_speed_kph": 220.0})
    result = sim.step_controls(throttle=1.0, brake=0.0, steer=0.0)
    assert result.terminated is True
    assert result.telemetry.termination_reason == "assist_overspeed_gate"
    assert result.telemetry.reward_components["assist_overspeed_gate"] == -12.0


def test_continuous_action_maps_drive_and_steer_controls() -> None:
    sim = MonzaSim(SimConfig(action_mode="continuous", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_continuous([0.75, -0.25])
    assert result.telemetry.throttle == 0.75
    assert result.telemetry.brake == 0.0
    assert result.telemetry.steering == -0.25
    assert result.telemetry.action_id == -1

    sim = MonzaSim(SimConfig(action_mode="continuous", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_continuous([-0.5, 0.25])
    assert result.telemetry.throttle == 0.0
    assert result.telemetry.brake == 0.5
    assert result.telemetry.steering == 0.25


def test_throttle_bias_continuous_scheme_keeps_zero_drive_moving() -> None:
    sim = MonzaSim(SimConfig(action_mode="continuous", continuous_action_scheme="throttle_bias", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_continuous([0.0, 0.25])
    assert result.telemetry.throttle == 0.5
    assert result.telemetry.brake == 0.0
    assert result.telemetry.steering == 0.25

    sim = MonzaSim(SimConfig(action_mode="continuous", continuous_action_scheme="throttle_bias", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_continuous([-0.5, -0.25])
    assert result.telemetry.throttle == 0.25
    assert result.telemetry.brake == 0.5
    assert result.telemetry.steering == -0.25


def test_exclusive_throttle_bias_continuous_scheme_does_not_overlap_throttle_and_brake() -> None:
    sim = MonzaSim(
        SimConfig(action_mode="continuous", continuous_action_scheme="exclusive_throttle_bias", max_steps=20)
    )
    sim.reset(seed=1)
    result = sim.step_continuous([0.0, 0.25])
    assert result.telemetry.throttle == 0.25
    assert result.telemetry.brake == 0.0
    assert result.telemetry.steering == 0.25

    sim = MonzaSim(
        SimConfig(action_mode="continuous", continuous_action_scheme="exclusive_throttle_bias", max_steps=20)
    )
    sim.reset(seed=1)
    result = sim.step_continuous([-0.5, -0.25])
    assert result.telemetry.throttle == 0.0
    assert result.telemetry.brake == 0.5
    assert result.telemetry.steering == -0.25

    sim = MonzaSim(
        SimConfig(action_mode="continuous", continuous_action_scheme="exclusive_throttle_bias", max_steps=20)
    )
    sim.reset(seed=1)
    result = sim.step_continuous([0.5, 0.0])
    assert result.telemetry.throttle == 0.625
    assert result.telemetry.brake == 0.0


def test_multidiscrete_action_factorizes_drive_and_steering() -> None:
    sim = MonzaSim(SimConfig(action_mode="multidiscrete", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_multidiscrete([4, 1])
    assert result.telemetry.throttle == 1.0
    assert result.telemetry.brake == 0.0
    assert result.telemetry.steering == -0.45
    assert result.telemetry.action_id == 21
    assert result.telemetry.action_name == "multidiscrete_drive_4_steer_1"

    sim = MonzaSim(SimConfig(action_mode="multidiscrete", max_steps=20))
    sim.reset(seed=1)
    result = sim.step_multidiscrete([1, 3])
    assert result.telemetry.throttle == 0.0
    assert result.telemetry.brake == 0.35
    assert result.telemetry.steering == 0.45


def test_discrete_action_name_uses_configured_action_set() -> None:
    sim = MonzaSim(SimConfig(action_set="racing", max_steps=20))
    sim.reset(seed=1)
    result = sim.step(2)
    assert result.telemetry.action_id == 2
    assert result.telemetry.action_name == "throttle"
    assert result.telemetry.throttle == 1.0
    assert result.telemetry.brake == 0.0


def test_reset_can_skip_observation_collection_for_fast_state_probes() -> None:
    sim = MonzaSim(SimConfig(observation_profile="racing_release"))

    obs, info = sim.reset(seed=1, options={"collect_observation": False})

    assert obs.shape == (sim.observation_dim,)
    assert obs.sum() == 0.0
    assert info["termination_reason"] == "active"


def test_launch_guard_remaps_brake_only_start_when_enabled() -> None:
    disabled = MonzaSim(SimConfig(max_steps=20, action_mode="multidiscrete"))
    disabled.reset(seed=1)
    disabled_step = disabled.step_multidiscrete([1, 2])
    assert disabled_step.telemetry.throttle == 0.0
    assert disabled_step.telemetry.brake == 0.35

    enabled = MonzaSim(
        SimConfig(
            max_steps=20,
            action_mode="multidiscrete",
            launch_guard_progress_m=80.0,
            launch_guard_min_speed_kph=25.0,
            launch_guard_throttle=0.22,
        )
    )
    enabled.reset(seed=1)
    enabled_step = enabled.step_multidiscrete([1, 2])
    assert enabled_step.telemetry.throttle == 0.22
    assert enabled_step.telemetry.brake == 0.0


def test_progress_is_monotonic() -> None:
    sim = MonzaSim()
    sim.reset(seed=1)
    previous = sim.state.monotonic_progress_m
    for _ in range(5):
        result = sim.step(1)
        assert result.telemetry.monotonic_progress_m >= previous
        previous = result.telemetry.monotonic_progress_m
        if result.terminated or result.truncated:
            break


def test_checkpoint_skip_invalidates_lap() -> None:
    sim = MonzaSim()
    sim.reset(seed=1)
    sim.state.monotonic_progress_m = sim.track.length_m
    sim._update_checkpoint_validity(0.0, sim.track.length_m, 0.0)
    assert sim.valid_lap is False
    assert sim.missed_checkpoint_count > 0


def test_checkpoint_crossing_far_from_centerline_invalidates_lap() -> None:
    sim = MonzaSim()
    sim.reset(seed=1)
    spacing = sim.track.length_m / len(sim.track.checkpoints)
    sim.state.monotonic_progress_m = spacing * 1.1
    sim._update_checkpoint_validity(0.0, spacing * 1.1, sim.config.checkpoint_lateral_limit_m + 1.0)
    assert sim.valid_lap is False
    assert sim.missed_checkpoint_count > 0
