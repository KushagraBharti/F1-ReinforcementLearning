from f1rl.constants import DEFAULT_START_POS
from f1rl.env import F1RaceEnv


def test_scripted_warm_start_resets_episode_counters_after_bootstrap() -> None:
    env = F1RaceEnv({"scripted_warm_start_steps": 5})
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert env.car.step_count == 0
        assert env.car.total_reward == 0.0
        assert env.terminated is False
        assert env.truncated is False
        assert env.car.next_goal_idx >= 1
        assert (float(env.car.x), float(env.car.y)) != DEFAULT_START_POS
        assert info["step_count"] == 0
        assert "speed_kph" in info
        assert env.track.meters_per_world_unit > 0.0
    finally:
        env.close()
