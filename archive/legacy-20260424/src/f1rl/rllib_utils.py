"""RLlib integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from gymnasium.envs.registration import VectorizeMode

from f1rl.ray_compat import prepare_ray_import

prepare_ray_import()
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from f1rl.env import F1RaceEnv
from f1rl.hardware import detect_use_gpu


@dataclass(slots=True, frozen=True)
class TrainingProfile:
    name: str
    rollout_fragment_length: int
    train_batch_size_per_learner: int
    minibatch_size: int
    lr: float
    gamma: float
    lambda_: float
    num_epochs: int
    entropy_coeff: float
    clip_param: float
    grad_clip: float
    model_hidden_layers: tuple[int, ...]
    evaluation_interval: int
    evaluation_duration: int
    num_env_runners: int
    num_envs_per_env_runner: int
    vector_mode: str


TRAINING_PROFILES: dict[str, TrainingProfile] = {
    "smoke": TrainingProfile(
        name="smoke",
        rollout_fragment_length=64,
        train_batch_size_per_learner=512,
        minibatch_size=128,
        lr=5e-4,
        gamma=0.99,
        lambda_=0.95,
        num_epochs=4,
        entropy_coeff=0.002,
        clip_param=0.2,
        grad_clip=0.5,
        model_hidden_layers=(128, 128),
        evaluation_interval=1,
        evaluation_duration=1,
        num_env_runners=1,
        num_envs_per_env_runner=2,
        vector_mode="sync",
    ),
    "benchmark": TrainingProfile(
        name="benchmark",
        rollout_fragment_length=96,
        train_batch_size_per_learner=576,
        minibatch_size=192,
        lr=3e-4,
        gamma=0.995,
        lambda_=0.97,
        num_epochs=6,
        entropy_coeff=0.0015,
        clip_param=0.18,
        grad_clip=0.5,
        model_hidden_layers=(256, 256),
        evaluation_interval=1,
        evaluation_duration=1,
        num_env_runners=2,
        num_envs_per_env_runner=3,
        vector_mode="sync",
    ),
    "performance": TrainingProfile(
        name="performance",
        rollout_fragment_length=200,
        train_batch_size_per_learner=4096,
        minibatch_size=512,
        lr=2.5e-4,
        gamma=0.997,
        lambda_=0.98,
        num_epochs=8,
        entropy_coeff=0.001,
        clip_param=0.16,
        grad_clip=0.5,
        model_hidden_layers=(256, 256, 128),
        evaluation_interval=2,
        evaluation_duration=3,
        num_env_runners=4,
        num_envs_per_env_runner=4,
        vector_mode="sync",
    ),
}


def get_training_profile(mode: str) -> TrainingProfile:
    try:
        return TRAINING_PROFILES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported training mode: {mode}") from exc


def profile_model_config(profile: TrainingProfile) -> dict[str, Any]:
    model_config = DefaultModelConfig(
        fcnet_hiddens=list(profile.model_hidden_layers),
        fcnet_activation="relu",
        vf_share_layers=True,
    )
    return {
        "fcnet_hiddens": list(model_config.fcnet_hiddens),
        "fcnet_activation": model_config.fcnet_activation,
        "vf_share_layers": model_config.vf_share_layers,
    }


def resolve_parallelism(
    *,
    profile: TrainingProfile,
    logical_cars: int | None,
    num_env_runners: int | None,
    num_envs_per_env_runner: int | None,
    vector_mode: str | None,
) -> tuple[int, int, VectorizeMode]:
    if logical_cars is not None and logical_cars > 0 and num_env_runners is None and num_envs_per_env_runner is None:
        default_runners = 2 if logical_cars <= 32 else 4
        default_runners = min(default_runners, logical_cars)
        envs_per_runner = int(ceil(logical_cars / max(default_runners, 1)))
        resolved_vector_mode = VectorizeMode.ASYNC if vector_mode is None else {
            "sync": VectorizeMode.SYNC,
            "async": VectorizeMode.ASYNC,
        }[vector_mode]
        return default_runners, envs_per_runner, resolved_vector_mode

    resolved_vector_mode = {
        "sync": VectorizeMode.SYNC,
        "async": VectorizeMode.ASYNC,
        None: VectorizeMode[profile.vector_mode.upper()],
    }[vector_mode]
    return (
        profile.num_env_runners if num_env_runners is None else num_env_runners,
        profile.num_envs_per_env_runner if num_envs_per_env_runner is None else num_envs_per_env_runner,
        resolved_vector_mode,
    )


def build_ppo_config(
    *,
    env_config: dict[str, Any],
    seed: int,
    mode: str,
    logical_cars: int | None = None,
    num_env_runners: int | None = None,
    num_envs_per_env_runner: int | None = None,
    vector_mode: str | None = None,
    device: str = "auto",
    ppo_overrides: dict[str, Any] | None = None,
    rl_module_load_state_path: str | None = None,
):
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module import RLModuleSpec

    profile = get_training_profile(mode)
    use_gpu = detect_use_gpu(device)
    resolved_num_env_runners, resolved_num_envs_per_runner, resolved_vector_mode = resolve_parallelism(
        profile=profile,
        logical_cars=logical_cars,
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        vector_mode=vector_mode,
    )
    model_config = profile_model_config(profile)
    train_batch_size_per_learner = int(
        (ppo_overrides or {}).get("train_batch_size_per_learner", profile.train_batch_size_per_learner)
    )
    requested_rollout_fragment_length = (ppo_overrides or {}).get("rollout_fragment_length")
    parallel_envs = max(1, resolved_num_env_runners * resolved_num_envs_per_runner)
    if requested_rollout_fragment_length is not None:
        rollout_fragment_length = int(requested_rollout_fragment_length)
    elif logical_cars is not None:
        rollout_fragment_length = max(1, int(round(train_batch_size_per_learner / parallel_envs)))
    else:
        rollout_fragment_length = profile.rollout_fragment_length

    config = (
        PPOConfig()
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment(env=F1RaceEnv, env_config=env_config)
        .debugging(seed=seed)
    )

    config = config.env_runners(
        num_env_runners=resolved_num_env_runners,
        create_local_env_runner=False,
        num_envs_per_env_runner=resolved_num_envs_per_runner,
        gym_env_vectorize_mode=resolved_vector_mode,
        num_gpus_per_env_runner=0,
        sample_timeout_s=300.0,
        max_requests_in_flight_per_env_runner=1,
        rollout_fragment_length=rollout_fragment_length,
    )

    config = config.fault_tolerance(
        restart_failed_env_runners=False,
        restart_failed_sub_environments=False,
    )

    config = config.learners(
        num_learners=0,
        num_cpus_per_learner=1,
        num_gpus_per_learner=1 if use_gpu else 0,
        local_gpu_idx=0 if use_gpu else None,
    )

    config = config.training(
        train_batch_size_per_learner=train_batch_size_per_learner,
        minibatch_size=profile.minibatch_size,
        lr=profile.lr,
        gamma=profile.gamma,
        lambda_=profile.lambda_,
        entropy_coeff=profile.entropy_coeff,
        clip_param=profile.clip_param,
        grad_clip=profile.grad_clip,
        num_epochs=profile.num_epochs,
    )

    if ppo_overrides:
        allowed_overrides = {
            key: value
            for key, value in ppo_overrides.items()
            if key
            in {
                "train_batch_size_per_learner",
                "minibatch_size",
                "lr",
                "gamma",
                "lambda_",
                "entropy_coeff",
                "clip_param",
                "grad_clip",
                "num_epochs",
            }
        }
        if allowed_overrides:
            config = config.training(**allowed_overrides)

    min_sample_timesteps = train_batch_size_per_learner
    config = config.reporting(
        min_sample_timesteps_per_iteration=min_sample_timesteps,
    )

    config = config.evaluation(
        evaluation_interval=profile.evaluation_interval,
        evaluation_duration=profile.evaluation_duration,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=0,
        evaluation_parallel_to_training=False,
        evaluation_config=AlgorithmConfig.overrides(
            explore=False,
            num_envs_per_env_runner=1,
            gym_env_vectorize_mode=VectorizeMode.SYNC,
        ),
    )

    if rl_module_load_state_path:
        env = F1RaceEnv(env_config)
        try:
            default_spec = config.get_default_rl_module_spec()
            rl_module_spec = RLModuleSpec(
                module_class=default_spec.module_class,
                observation_space=env.observation_space,
                action_space=env.action_space,
                inference_only=False,
                model_config=model_config,
                catalog_class=default_spec.catalog_class,
                load_state_path=rl_module_load_state_path,
            )
        finally:
            env.close()
        config = config.rl_module(model_config=model_config, rl_module_spec=rl_module_spec)
    else:
        config = config.rl_module(model_config=model_config)
    return config


def extract_action(action_output: Any) -> Any:
    if isinstance(action_output, tuple):
        return action_output[0]
    return action_output


def compute_module_action(module: Any, obs: np.ndarray, explore: bool = False) -> np.ndarray:
    import torch
    from ray.rllib.core.columns import Columns

    obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
    with torch.no_grad():
        output = module.forward_inference({Columns.OBS: torch.from_numpy(obs_batch)})

        if Columns.ACTIONS in output:
            action_tensor = output[Columns.ACTIONS]
        else:
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(output[Columns.ACTION_DIST_INPUTS])
            if not explore and hasattr(dist, "to_deterministic"):
                dist = dist.to_deterministic()
            action_tensor = dist.sample()
    return np.asarray(action_tensor.detach().cpu().numpy()[0], dtype=np.float32)


def compute_inference_action(algo: Any, obs: np.ndarray, explore: bool = False) -> np.ndarray:
    """Compute an action using RLModule APIs with fallback to legacy helper."""
    try:
        module = algo.get_module()
        return compute_module_action(module, obs, explore=explore)
    except Exception:
        action_output = algo.compute_single_action(obs, explore=explore)
        return np.asarray(extract_action(action_output), dtype=np.float32)
