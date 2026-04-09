from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv


def test_progress_reward_is_positive() -> None:
    cfg = EnvConfig()
    cfg.render.enabled = False
    cfg.render_mode = None
    env = F1RaceEnv(to_dict(cfg))
    try:
        _, reward, _, _, info = env.step(env.action_space.sample() * 0.0)
        assert "reward_components" in info
        assert "alignment" in info["reward_components"]
        assert "centerline" in info["reward_components"]
        assert "stall" in info["reward_components"]
        assert info["reward_components"]["stall"] <= 0.0
        assert isinstance(reward, float)
    finally:
        env.close()


def test_no_progress_termination_triggers() -> None:
    cfg = EnvConfig()
    cfg.render.enabled = False
    cfg.render_mode = None
    cfg.no_progress_limit_steps = 3
    env = F1RaceEnv(to_dict(cfg))
    try:
        env.reset(seed=7)
        terminated = False
        for _ in range(4):
            _, _, terminated, _, info = env.step(env.action_space.sample() * 0.0)
            if terminated:
                break
        assert terminated is True
        assert info["terminated_reason"] == "no_progress"
    finally:
        env.close()


def test_reverse_speed_termination_triggers() -> None:
    cfg = EnvConfig()
    cfg.render.enabled = False
    cfg.render_mode = None
    cfg.reverse_speed_limit_steps = 1
    cfg.reverse_speed_threshold = 0.01
    env = F1RaceEnv(to_dict(cfg))
    try:
        env.reset(seed=7)
        env.car.speed = -0.5
        _, _, terminated, _, info = env.step(env.action_space.sample() * 0.0)
        assert terminated is True
        assert info["terminated_reason"] == "reverse_speed"
    finally:
        env.close()
