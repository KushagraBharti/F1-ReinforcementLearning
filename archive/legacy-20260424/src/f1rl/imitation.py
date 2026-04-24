"""Behavior cloning from scripted controller trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.artifacts import new_run_paths
from f1rl.config import EnvConfig, to_dict
from f1rl.controllers import ScriptedController
from f1rl.env import F1RaceEnv
from f1rl.ray_compat import prepare_ray_import
from f1rl.rllib_utils import profile_model_config, get_training_profile


@dataclass(slots=True)
class EpisodeImitationSummary:
    seed: int
    steps: int
    lap_count: int
    positive_progress_total: int
    total_reward: float
    terminated_reason: str | None


class BehaviorCloningPolicy:
    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch

        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        with torch.no_grad():
            actions = self.model(obs_tensor)
        return np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)


class RLModuleBehaviorCloningPolicy:
    def __init__(self, module: Any, device: Any) -> None:
        self.module = module
        self.device = device

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.compute_actions(np.asarray(obs, dtype=np.float32)[None, :])[0]

    def compute_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        import torch

        from ray.rllib.core.columns import Columns

        obs_tensor = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        with torch.no_grad():
            output = self.module.forward_inference({Columns.OBS: obs_tensor})
            action_dist_inputs = output[Columns.ACTION_DIST_INPUTS]
            action_dim = int(action_dist_inputs.shape[-1] // 2)
            actions = torch.tanh(action_dist_inputs[..., :action_dim])
        return np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior-cloned policy from scripted rollouts.")
    parser.add_argument("--episodes", type=int, default=12, help="Number of scripted episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per scripted episode.")
    parser.add_argument("--epochs", type=int, default=12, help="Behavior cloning epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer width.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--ppo-mode", choices=["smoke", "benchmark", "performance"], default="smoke")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[101, 202, 303])
    parser.add_argument("--run-tag", default="scripted")
    return parser.parse_args(argv)


def _resolve_device(device: str) -> Any:
    import torch

    if device == "cpu":
        return torch.device("cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU device for imitation training, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_env_config(max_steps: int) -> dict[str, Any]:
    config = EnvConfig()
    config.max_steps = max_steps
    config.headless = True
    config.render.enabled = False
    return to_dict(config)


def collect_scripted_dataset(
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[EpisodeImitationSummary], dict[str, Any]]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    summaries: list[EpisodeImitationSummary] = []
    env_config = _build_env_config(max_steps)

    for episode_idx in range(max(1, episodes)):
        env = F1RaceEnv(env_config)
        try:
            controller = ScriptedController(sensor_count=env.sensor_count, goals=env.track.goals.copy())
            obs, info = env.reset(seed=seed + episode_idx)
            controller.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = controller.action(obs, info)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            summaries.append(
                EpisodeImitationSummary(
                    seed=seed + episode_idx,
                    steps=int(info["step_count"]),
                    lap_count=int(info["lap_count"]),
                    positive_progress_total=int(info["positive_progress_total"]),
                    total_reward=total_reward,
                    terminated_reason=info.get("terminated_reason"),
                )
            )
        finally:
            env.close()

    dataset_meta = {
        "episodes": len(summaries),
        "samples": len(observations),
        "avg_progress_total": float(np.mean([item.positive_progress_total for item in summaries])),
        "best_progress_total": max((item.positive_progress_total for item in summaries), default=0),
    }
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        summaries,
        dataset_meta,
    )


def _build_model(obs_dim: int, action_dim: int, hidden_size: int) -> Any:
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_dim),
        nn.Tanh(),
    )


def _build_ppo_rlmodule(*, env_config: dict[str, Any], model_config: dict[str, Any], inference_only: bool = False) -> Any:
    prepare_ray_import()
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module import RLModuleSpec

    env = F1RaceEnv(env_config)
    try:
        base_config = (
            PPOConfig()
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
            .environment(env=F1RaceEnv, env_config=env_config)
        )
        default_spec = base_config.get_default_rl_module_spec()
        module_spec = RLModuleSpec(
            module_class=default_spec.module_class,
            observation_space=env.observation_space,
            action_space=env.action_space,
            inference_only=inference_only,
            model_config=model_config,
            catalog_class=default_spec.catalog_class,
        )
        return module_spec.build()
    finally:
        env.close()


def train_behavior_cloning(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: Any,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = _build_model(observations.shape[1], actions.shape[1], hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    num_samples = obs_tensor.shape[0]
    batch_size = max(1, min(batch_size, num_samples))
    losses: list[float] = []

    started = time.perf_counter()
    for _ in range(max(1, epochs)):
        permutation = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            pred = model(obs_tensor[batch_idx])
            loss = loss_fn(pred, action_tensor[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    metrics = {
        "final_loss": float(losses[-1]) if losses else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "train_time_s": float(time.perf_counter() - started),
    }
    return model, metrics


def train_ppo_module_behavior_cloning(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    env_config: dict[str, Any],
    model_config: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: Any,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    import torch
    from torch import nn

    from ray.rllib.core.columns import Columns

    torch.manual_seed(seed)
    module = _build_ppo_rlmodule(env_config=env_config, model_config=model_config, inference_only=False).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    num_samples = obs_tensor.shape[0]
    batch_size = max(1, min(batch_size, num_samples))
    action_dim = int(action_tensor.shape[-1])
    losses: list[float] = []

    started = time.perf_counter()
    for _ in range(max(1, epochs)):
        permutation = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            output = module.forward_train({Columns.OBS: obs_tensor[batch_idx]})
            action_dist_inputs = output[Columns.ACTION_DIST_INPUTS]
            pred_actions = torch.tanh(action_dist_inputs[..., :action_dim])
            loss = loss_fn(pred_actions, action_tensor[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    metrics = {
        "final_loss": float(losses[-1]) if losses else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "train_time_s": float(time.perf_counter() - started),
    }
    return module, metrics


def evaluate_policy(
    *,
    policy: BehaviorCloningPolicy,
    max_steps: int,
    seeds: list[int],
) -> tuple[list[EpisodeImitationSummary], dict[str, float]]:
    summaries: list[EpisodeImitationSummary] = []
    env_config = _build_env_config(max_steps)
    for seed in seeds:
        env = F1RaceEnv(env_config)
        try:
            obs, info = env.reset(seed=seed)
            total_reward = 0.0
            done = False
            while not done:
                action = policy.act(np.asarray(obs, dtype=np.float32)[None, :])[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            summaries.append(
                EpisodeImitationSummary(
                    seed=int(seed),
                    steps=int(info["step_count"]),
                    lap_count=int(info["lap_count"]),
                    positive_progress_total=int(info["positive_progress_total"]),
                    total_reward=total_reward,
                    terminated_reason=info.get("terminated_reason"),
                )
            )
        finally:
            env.close()

    aggregate = {
        "episodes": len(summaries),
        "avg_progress_total": float(np.mean([item.positive_progress_total for item in summaries])),
        "best_progress_total": max((item.positive_progress_total for item in summaries), default=0),
        "avg_steps": float(np.mean([item.steps for item in summaries])),
        "avg_reward": float(np.mean([item.total_reward for item in summaries])),
        "completion_rate": float(np.mean([item.lap_count >= 1 for item in summaries])),
    }
    return summaries, aggregate


def run_imitation(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    run_paths = new_run_paths(prefix=f"bc-{args.run_tag}")
    env_config = _build_env_config(args.max_steps)
    model_config = profile_model_config(get_training_profile(args.ppo_mode))
    observations, actions, dataset_summaries, dataset_meta = collect_scripted_dataset(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    np.savez_compressed(run_paths.metrics / "scripted_dataset.npz", observations=observations, actions=actions)
    (run_paths.metrics / "dataset_summary.json").write_text(
        json.dumps(
            {
                "aggregate": dataset_meta,
                "episodes": [asdict(item) for item in dataset_summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device = _resolve_device(args.device)
    model, train_metrics = train_behavior_cloning(
        observations=observations,
        actions=actions,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )

    import torch

    model_path = run_paths.checkpoints / "bc_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": int(observations.shape[1]),
            "action_dim": int(actions.shape[1]),
            "hidden_size": int(args.hidden_size),
        },
        model_path,
    )

    policy = BehaviorCloningPolicy(model=model.eval(), device=device)
    eval_summaries, eval_aggregate = evaluate_policy(
        policy=policy,
        max_steps=args.max_steps,
        seeds=list(args.eval_seeds),
    )

    ppo_module, ppo_train_metrics = train_ppo_module_behavior_cloning(
        observations=observations,
        actions=actions,
        env_config=env_config,
        model_config=model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )
    ppo_module_path = run_paths.checkpoints / "ppo_pretrain_module"
    ppo_module.save_to_path(str(ppo_module_path))
    if hasattr(ppo_module, "eval"):
        ppo_module.eval()
    ppo_policy = RLModuleBehaviorCloningPolicy(module=ppo_module, device=device)
    ppo_eval_summaries, ppo_eval_aggregate = evaluate_policy(
        policy=ppo_policy,
        max_steps=args.max_steps,
        seeds=list(args.eval_seeds),
    )

    summary = {
        "device": str(device),
        "ppo_mode": args.ppo_mode,
        "dataset": {
            "aggregate": dataset_meta,
            "episodes": [asdict(item) for item in dataset_summaries],
        },
        "train": train_metrics,
        "eval": {
            "aggregate": eval_aggregate,
            "episodes": [asdict(item) for item in eval_summaries],
        },
        "ppo_pretrain": {
            "train": ppo_train_metrics,
            "eval": {
                "aggregate": ppo_eval_aggregate,
                "episodes": [asdict(item) for item in ppo_eval_summaries],
            },
        },
        "model_path": str(model_path),
        "ppo_pretrain_module_path": str(ppo_module_path),
    }
    summary_path = run_paths.root / "bc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path, summary = run_imitation(args)
    print(
        f"bc_complete episodes={summary['dataset']['aggregate']['episodes']} "
        f"samples={summary['dataset']['aggregate']['samples']} "
        f"avg_progress={summary['eval']['aggregate']['avg_progress_total']:.3f} "
        f"summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
