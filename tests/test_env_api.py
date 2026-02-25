from gymnasium.utils.env_checker import check_env

from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv


def test_env_checker() -> None:
    cfg = EnvConfig()
    cfg.max_steps = 64
    cfg.render.enabled = False
    cfg.render_mode = None
    env = F1RaceEnv(to_dict(cfg))
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()
