from f1rl.config import DISCRETE_ACTIONS, action_to_controls
from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS


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
    assert DISCRETE_ACTIONS[: len(original_actions)] == original_actions
    assert ("half_throttle_soft_left", 0.5, 0.0, -0.45) in DISCRETE_ACTIONS
    assert ("soft_brake_right", 0.0, 0.35, 0.45) in DISCRETE_ACTIONS
    assert action_to_controls(14) == (0.5, 0.0, -0.45)


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
