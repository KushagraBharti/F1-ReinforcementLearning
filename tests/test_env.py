from gymnasium.utils.env_checker import check_env

from f1rl.config import SimConfig
from f1rl.env import MonzaEnv


def test_env_checker_passes() -> None:
    env = MonzaEnv(SimConfig(max_steps=20))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_env_step_contract() -> None:
    env = MonzaEnv(SimConfig(max_steps=20))
    try:
        obs, info = env.reset(seed=123)
        next_obs, reward, terminated, truncated, step_info = env.step(1)
        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in info
        assert "reward_components" in step_info
    finally:
        env.close()
