from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS


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
