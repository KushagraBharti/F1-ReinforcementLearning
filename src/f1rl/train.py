"""RLlib training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any

import ray

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.logging_utils import configure_logging
from f1rl.rllib_utils import build_ppo_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policy for F1 race environment.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of training iterations.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-env-runners", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save interval in iterations.")
    return parser.parse_args(argv)


def training_env_config(mode: str) -> dict[str, Any]:
    config = EnvConfig()
    config.max_steps = 300 if mode == "smoke" else 1800
    config.max_laps = 1
    config.render_mode = None
    config.headless = True
    config.render.enabled = False
    config.render.export_frames = False
    return to_dict(config)


def _metric_value(result: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    node: Any = result
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if isinstance(node, (int, float)):
        return float(node)
    return default


def _extract_training_metrics(result: dict[str, Any]) -> dict[str, float]:
    episode_reward = result.get("episode_reward_mean")
    if episode_reward is None:
        episode_reward = _metric_value(result, "env_runners", "episode_return_mean", default=0.0)
    episode_len = result.get("episode_len_mean")
    if episode_len is None:
        episode_len = _metric_value(result, "env_runners", "episode_len_mean", default=0.0)
    timesteps_total = result.get("timesteps_total")
    if timesteps_total is None:
        timesteps_total = _metric_value(result, "num_env_steps_sampled_lifetime", default=0.0)
    return {
        "training_iteration": float(result.get("training_iteration", 0)),
        "episode_reward_mean": float(episode_reward),
        "episode_len_mean": float(episode_len),
        "timesteps_total": float(timesteps_total),
        "time_total_s": float(result.get("time_total_s", 0.0)),
    }


def _checkpoint_path(checkpoint_obj: Any) -> str:
    if isinstance(checkpoint_obj, str):
        return checkpoint_obj
    if hasattr(checkpoint_obj, "path"):
        return str(checkpoint_obj.path)
    if hasattr(checkpoint_obj, "checkpoint") and hasattr(checkpoint_obj.checkpoint, "path"):
        return str(checkpoint_obj.checkpoint.path)
    return str(checkpoint_obj)


def run_training(args: argparse.Namespace) -> int:
    smoke = args.mode == "smoke"
    iterations = args.iterations if args.iterations is not None else (2 if smoke else 60)
    env_config = training_env_config(args.mode)
    run_paths = new_run_paths(prefix=f"train-{args.mode}")
    logger = configure_logging(run_paths.logs / "train.log", logger_name="f1rl.train")

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    csv_path = run_paths.metrics / "metrics.csv"
    jsonl_path = run_paths.metrics / "metrics.jsonl"
    metadata_path = run_paths.root / "run_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "mode": args.mode,
                "iterations": iterations,
                "seed": args.seed,
                "num_env_runners": args.num_env_runners,
                "env_config": env_config,
            },
            metadata_file,
            indent=2,
        )

    if args.resume:
        from ray.rllib.algorithms.algorithm import Algorithm

        checkpoint = resolve_checkpoint(args.resume)
        logger.info("Restoring algorithm from checkpoint", extra={"checkpoint": str(checkpoint)})
        algo = Algorithm.from_checkpoint(str(checkpoint))
    else:
        config = build_ppo_config(
            env_config=env_config,
            seed=args.seed,
            smoke=smoke,
            num_env_runners=args.num_env_runners,
        )
        if hasattr(config, "build_algo"):
            algo = config.build_algo()
        else:
            algo = config.build()

    fieldnames = ["training_iteration", "episode_reward_mean", "episode_len_mean", "timesteps_total", "time_total_s"]
    latest_checkpoint_path = ""
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
            "w", encoding="utf-8"
        ) as jsonl_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(1, iterations + 1):
                result = algo.train()
                metrics = _extract_training_metrics(result)
                writer.writerow(metrics)
                jsonl_file.write(json.dumps(metrics) + "\n")
                jsonl_file.flush()

                logger.info(
                    "train_iteration",
                    extra={
                        "iteration": idx,
                        "episode_reward_mean": metrics["episode_reward_mean"],
                        "episode_len_mean": metrics["episode_len_mean"],
                        "timesteps_total": metrics["timesteps_total"],
                    },
                )

                if idx % max(args.checkpoint_every, 1) == 0 or idx == iterations:
                    ckpt = algo.save(checkpoint_dir=str(run_paths.checkpoints))
                    latest_checkpoint_path = _checkpoint_path(ckpt)
                    logger.info(
                        "checkpoint_saved",
                        extra={"iteration": idx, "checkpoint_path": latest_checkpoint_path},
                    )
    finally:
        algo.stop()
        ray.shutdown()

    print(
        f"training_complete mode={args.mode} iterations={iterations} run_dir={run_paths.root} "
        f"checkpoint={latest_checkpoint_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
