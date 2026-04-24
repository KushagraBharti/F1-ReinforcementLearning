from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS


def test_sim_observation_and_reward_schema() -> None:
    sim = MonzaSim()
    obs, info = sim.reset(seed=1)
    assert obs.shape == (sim.observation_dim,)
    assert set(info["reward_components"]) == set(REWARD_COMPONENT_KEYS)
    result = sim.step(1)
    assert result.observation.shape == (sim.observation_dim,)
    assert set(result.telemetry.reward_components) == set(REWARD_COMPONENT_KEYS)
    assert len(result.telemetry.ray_distances_m) == sim.config.sensors.count


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
