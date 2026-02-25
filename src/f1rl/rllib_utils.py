"""RLlib integration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from f1rl.env import F1RaceEnv


def build_ppo_config(
    *,
    env_config: dict[str, Any],
    seed: int,
    smoke: bool,
    num_env_runners: int,
):
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig().framework("torch").environment(env=F1RaceEnv, env_config=env_config).debugging(seed=seed)

    # Prefer modern env-runners API and fall back for older versions.
    if hasattr(config, "env_runners"):
        maybe_config = config.env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=64 if smoke else 200,
        )
        if maybe_config is not None:
            config = maybe_config
    elif hasattr(config, "rollouts"):
        maybe_config = config.rollouts(
            num_rollout_workers=num_env_runners,
            num_envs_per_worker=1,
            rollout_fragment_length=64 if smoke else 200,
        )
        if maybe_config is not None:
            config = maybe_config

    train_batch_size = 256 if smoke else 4000
    maybe_config = config.training(
        lr=5e-4 if smoke else 3e-4,
        train_batch_size=train_batch_size,
        gamma=0.99,
    )
    if maybe_config is not None:
        config = maybe_config

    if hasattr(config, "resources"):
        maybe_config = config.resources(num_gpus=0)
        if maybe_config is not None:
            config = maybe_config
    return config


def extract_action(action_output):
    if isinstance(action_output, tuple):
        return action_output[0]
    return action_output


def compute_inference_action(algo: Any, obs: np.ndarray, explore: bool = False) -> np.ndarray:
    """Compute an action using RLModule APIs with fallback to legacy helper."""
    try:
        import torch
        from ray.rllib.core.columns import Columns

        module = algo.get_module()
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        output = module.forward_inference({Columns.OBS: torch.from_numpy(obs_batch)})

        if Columns.ACTIONS in output:
            action_tensor = output[Columns.ACTIONS]
        else:
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(output[Columns.ACTION_DIST_INPUTS])
            if not explore and hasattr(dist, "to_deterministic"):
                dist = dist.to_deterministic()
            action_tensor = dist.sample()

        action = action_tensor.detach().cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32)
    except Exception:
        action_output = algo.compute_single_action(obs, explore=explore)
        return np.asarray(extract_action(action_output), dtype=np.float32)
