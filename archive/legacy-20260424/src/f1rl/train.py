"""Torch-native PPO training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.hardware import detect_use_gpu, get_torch_runtime_info, require_gpu_available
from f1rl.logging_utils import configure_logging
from f1rl.stages import build_stage_env_overrides, logical_car_target
from f1rl.torch_agent import (
    ActorCriticPolicy,
    PPOHyperParams,
    load_torch_policy,
    resolve_torch_device,
    save_torch_checkpoint,
)
from f1rl.torch_ppo import collect_rollout, ppo_update
from f1rl.torch_runtime import TorchSimBatch


TRAINING_PROFILES = {
    "smoke": {"iterations": 2, "num_cars": 32, "horizon": 64},
    "benchmark": {"iterations": 8, "num_cars": 64, "horizon": 128},
    "performance": {"iterations": 60, "num_cars": 128, "horizon": 256},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policy on the torch-native F1 simulator.")
    parser.add_argument("--mode", choices=sorted(TRAINING_PROFILES), default="smoke")
    parser.add_argument("--swarm-stage", choices=["auto", "competence", "lap", "stability", "performance"], default="auto")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-cars", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--env-config-json", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--num-env-runners", type=int, default=None)
    parser.add_argument("--num-envs-per-env-runner", type=int, default=None)
    parser.add_argument("--vector-mode", choices=["sync", "async"], default=None)
    return parser.parse_args(argv)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def training_env_config(mode: str, *, stage: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = EnvConfig()
    config.max_steps = {
        "smoke": 300,
        "benchmark": 900,
        "performance": 1800,
    }[mode]
    config.max_laps = 1
    config.render.enabled = False
    config.render_mode = None
    config.headless = True
    env_config = to_dict(config)
    if stage != "auto":
        env_config = _deep_merge(env_config, build_stage_env_overrides(stage))
    return _deep_merge(env_config, overrides or {})


def run_training(args: argparse.Namespace) -> int:
    if args.require_gpu:
        require_gpu_available(args.device)

    profile = TRAINING_PROFILES[args.mode]
    resolved_stage = "competence" if args.swarm_stage == "auto" else args.swarm_stage
    env_overrides = json.loads(args.env_config_json) if args.env_config_json else {}
    env_config = training_env_config(args.mode, stage=resolved_stage, overrides=env_overrides)
    sim_device = args.sim_device or ("gpu" if detect_use_gpu(args.device) else "cpu")
    policy_device = args.policy_device or sim_device
    resolved_policy_device = resolve_torch_device(policy_device)
    resolved_num_cars = args.num_cars or logical_car_target(resolved_stage) or profile["num_cars"]
    ppo = PPOHyperParams(
        horizon=args.horizon or profile["horizon"],
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
    )
    iterations = args.iterations if args.iterations is not None else profile["iterations"]
    runtime = get_torch_runtime_info()
    run_prefix = f"train-{args.mode}" if not args.run_tag else f"train-{args.mode}-{args.run_tag}"
    run_paths = new_run_paths(prefix=run_prefix)
    logger = configure_logging(run_paths.logs / "train.log", logger_name="f1rl.train")

    simulator = TorchSimBatch(env_config, num_cars=resolved_num_cars, device=sim_device)
    observations = simulator.reset(seed=args.seed)

    if args.resume:
        loaded = load_torch_policy(resolve_checkpoint(args.resume), device=str(resolved_policy_device))
        policy = loaded.model
    else:
        policy = ActorCriticPolicy(simulator.observation_dim, simulator.action_dim).to(resolved_policy_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate)

    csv_path = run_paths.metrics / "metrics.csv"
    jsonl_path = run_paths.metrics / "metrics.jsonl"
    metadata_path = run_paths.root / "run_metadata.json"
    metadata = {
        "mode": args.mode,
        "iterations": iterations,
        "seed": args.seed,
        "swarm_stage": resolved_stage,
        "num_cars": resolved_num_cars,
        "ppo": asdict(ppo),
        "sim_device": str(simulator.device),
        "policy_device": str(resolved_policy_device),
        "torch": {
            "version": runtime.torch_version,
            "cuda_available": runtime.cuda_available,
            "cuda_version": runtime.cuda_version,
            "device_name": runtime.device_name,
        },
        "env_config": env_config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_checkpoint_path: Path | None = None
    try:
        for iteration in range(1, iterations + 1):
            rollout_started = time.perf_counter()
            rollout = collect_rollout(simulator, policy, ppo, deterministic=False)
            update_metrics = ppo_update(policy, optimizer, rollout, ppo)
            elapsed = time.perf_counter() - rollout_started
            snapshots = simulator.snapshot()
            row = {
                "training_iteration": iteration,
                "timesteps_total": int(iteration * ppo.horizon * resolved_num_cars),
                "episode_reward_mean": float(rollout.rewards.mean().item()),
                "episode_len_mean": float(simulator.step_count.float().mean().item()),
                "avg_speed_kph": float(snapshots["speed_kph"].mean().item()),
                "avg_distance_travelled": float(snapshots["distance_travelled"].mean().item()),
                "avg_checkpoints_reached": float(snapshots["positive_progress_total"].float().mean().item()),
                "alive_cars": int((~(snapshots["terminated"] | snapshots["truncated"])).sum().item()),
                "samples_per_second": float((ppo.horizon * resolved_num_cars) / max(elapsed, 1e-6)),
                "iteration_wall_time_s": float(elapsed),
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
            }
            metric_rows.append(row)
            logger.info("iteration=%s metrics=%s", iteration, row)
            with jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")

            if iteration % max(args.checkpoint_every, 1) == 0:
                checkpoint_dir = run_paths.checkpoints / f"checkpoint_{iteration:06d}"
                save_torch_checkpoint(
                    checkpoint_dir,
                    model=policy,
                    optimizer=optimizer,
                    env_config=env_config,
                    training_metadata={
                        "training_iteration": iteration,
                        "elapsed_s": time.perf_counter() - started,
                        "sim_device": str(simulator.device),
                        "policy_device": str(resolved_policy_device),
                    },
                )
                last_checkpoint_path = checkpoint_dir

        if metric_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(metric_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metric_rows)
    finally:
        del observations

    if last_checkpoint_path is None:
        last_checkpoint_path = run_paths.checkpoints
    print(
        f"train_complete backend=torch_native mode={args.mode} iterations={iterations} "
        f"sim_device={simulator.device} policy_device={resolved_policy_device} checkpoint={last_checkpoint_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_training(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
