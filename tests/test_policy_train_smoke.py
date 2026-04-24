import importlib.util

import pytest

from f1rl.train import run_training


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_smoke_checkpoint() -> None:
    checkpoint = run_training(
        timesteps=64,
        seed=5,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
    )
    assert checkpoint.exists()
