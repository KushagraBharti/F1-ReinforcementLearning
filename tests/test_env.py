from gymnasium.utils.env_checker import check_env

from f1rl.config import SimConfig
from f1rl.env import MonzaEnv


def test_env_checker_passes() -> None:
    env = MonzaEnv(SimConfig(max_steps=20))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_env_checker_passes_for_racing_observation_profile() -> None:
    env = MonzaEnv(SimConfig(max_steps=20, observation_profile="racing"))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_env_checker_passes_for_racing_v2_observation_profile() -> None:
    env = MonzaEnv(SimConfig(max_steps=20, observation_profile="racing_v2"))
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


def test_env_action_space_follows_sim_config() -> None:
    legacy = MonzaEnv(SimConfig(max_steps=20, action_set="legacy"))
    expanded = MonzaEnv(SimConfig(max_steps=20, action_set="expanded"))
    racing = MonzaEnv(SimConfig(max_steps=20, action_set="racing"))
    continuous = MonzaEnv(SimConfig(max_steps=20, action_mode="continuous"))
    multidiscrete = MonzaEnv(SimConfig(max_steps=20, action_mode="multidiscrete"))
    try:
        assert legacy.action_space.n == 9
        assert expanded.action_space.n == 21
        assert racing.action_space.n == 20
        assert continuous.action_space.shape == (2,)
        assert tuple(multidiscrete.action_space.nvec) == (5, 5)
    finally:
        legacy.close()
        expanded.close()
        racing.close()
        continuous.close()
        multidiscrete.close()


def test_env_brake_observation_profile_adds_target_speed_features() -> None:
    base = MonzaEnv(SimConfig(max_steps=20, observation_profile="base"))
    brake = MonzaEnv(SimConfig(max_steps=20, observation_profile="brake"))
    guidance = MonzaEnv(SimConfig(max_steps=20, observation_profile="guidance"))
    racing = MonzaEnv(SimConfig(max_steps=20, observation_profile="racing"))
    racing_v2 = MonzaEnv(SimConfig(max_steps=20, observation_profile="racing_v2"))
    try:
        base_obs, _ = base.reset(seed=1)
        brake_obs, _ = brake.reset(seed=1)
        guidance_obs, _ = guidance.reset(seed=1)
        racing_obs, _ = racing.reset(seed=1)
        racing_v2_obs, _ = racing_v2.reset(seed=1)
        assert brake_obs.shape[0] == base_obs.shape[0] + 3
        assert guidance_obs.shape[0] == base_obs.shape[0] + 5
        assert racing_obs.shape[0] == base_obs.shape[0] + 13
        assert racing_v2_obs.shape[0] == base_obs.shape[0] + 17
        assert brake.observation_space.contains(brake_obs)
        assert guidance.observation_space.contains(guidance_obs)
        assert racing.observation_space.contains(racing_obs)
        assert racing_v2.observation_space.contains(racing_v2_obs)
    finally:
        base.close()
        brake.close()
        guidance.close()
        racing.close()
        racing_v2.close()


def test_env_continuous_step_contract() -> None:
    env = MonzaEnv(SimConfig(max_steps=20, action_mode="continuous"))
    try:
        obs, _ = env.reset(seed=123)
        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in step_info
    finally:
        env.close()


def test_env_multidiscrete_step_contract() -> None:
    env = MonzaEnv(SimConfig(max_steps=20, action_mode="multidiscrete"))
    try:
        obs, _ = env.reset(seed=123)
        next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in step_info
    finally:
        env.close()


def test_env_can_update_reward_scaffold_scale() -> None:
    env = MonzaEnv(SimConfig(max_steps=20))
    try:
        env.set_reward_scaffold_scale(0.25)
        assert env.config.reward.scaffold_scale == 0.25
        assert env.sim.config.reward.scaffold_scale == 0.25
        env.set_reward_scaffold_scale(2.0)
        assert env.config.reward.scaffold_scale == 1.0
    finally:
        env.close()
