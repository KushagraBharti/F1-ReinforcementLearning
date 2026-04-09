"""RLlib training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import ray

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.hardware import detect_use_gpu, get_torch_runtime_info, require_gpu_available
from f1rl.inference import export_inference_artifact
from f1rl.logging_utils import configure_logging
from f1rl.rllib_utils import (
    build_ppo_config,
    get_training_profile,
    profile_model_config,
    resolve_parallelism,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policy for F1 race environment.")
    parser.add_argument("--mode", choices=["smoke", "benchmark", "performance"], default="smoke")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of training iterations.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-env-runners", type=int, default=None)
    parser.add_argument("--num-envs-per-env-runner", type=int, default=None)
    parser.add_argument("--vector-mode", choices=["sync", "async"], default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save interval in iterations.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA-enabled Torch is unavailable.")
    parser.add_argument("--env-config-json", default=None, help="JSON object of env config overrides.")
    parser.add_argument("--run-tag", default=None, help="Optional suffix for run prefix.")
    return parser.parse_args(argv)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def training_env_config(mode: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = EnvConfig()
    config.max_steps = {
        "smoke": 300,
        "benchmark": 900,
        "performance": 1800,
    }[mode]
    config.max_laps = 1
    config.render_mode = None
    config.headless = True
    config.render.enabled = False
    config.render.export_frames = False
    env_config = to_dict(config)
    return _deep_merge(env_config, overrides or {})


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

    evaluation_return = _metric_value(result, "evaluation", "env_runners", "episode_return_mean", default=0.0)
    evaluation_len = _metric_value(result, "evaluation", "env_runners", "episode_len_mean", default=0.0)
    return {
        "training_iteration": float(result.get("training_iteration", 0)),
        "episode_reward_mean": float(episode_reward),
        "episode_len_mean": float(episode_len),
        "timesteps_total": float(timesteps_total),
        "time_total_s": float(result.get("time_total_s", 0.0)),
        "evaluation_episode_reward_mean": float(evaluation_return),
        "evaluation_episode_len_mean": float(evaluation_len),
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
    if args.require_gpu:
        require_gpu_available(args.device)
    profile = get_training_profile(args.mode)
    runtime = get_torch_runtime_info()
    use_gpu = detect_use_gpu(args.device)
    resolved_num_env_runners, resolved_num_envs_per_runner, resolved_vector_mode = resolve_parallelism(
        profile=profile,
        num_env_runners=args.num_env_runners,
        num_envs_per_env_runner=args.num_envs_per_env_runner,
        vector_mode=args.vector_mode,
    )
    model_config = profile_model_config(profile)
    env_overrides = json.loads(args.env_config_json) if args.env_config_json else {}
    iterations = args.iterations if args.iterations is not None else {
        "smoke": 2,
        "benchmark": 6,
        "performance": 60,
    }[args.mode]
    env_config = training_env_config(args.mode, overrides=env_overrides)
    run_prefix = f"train-{args.mode}" if not args.run_tag else f"train-{args.mode}-{args.run_tag}"
    run_paths = new_run_paths(prefix=run_prefix)
    logger = configure_logging(run_paths.logs / "train.log", logger_name="f1rl.train")

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    csv_path = run_paths.metrics / "metrics.csv"
    jsonl_path = run_paths.metrics / "metrics.jsonl"
    metadata_path = run_paths.root / "run_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "mode": args.mode,
                "profile": profile.name,
                "iterations": iterations,
                "seed": args.seed,
                "num_env_runners": resolved_num_env_runners,
                "num_envs_per_env_runner": resolved_num_envs_per_runner,
                "vector_mode": resolved_vector_mode.value,
                "device": args.device,
                "use_gpu": use_gpu,
                "torch": {
                    "version": runtime.torch_version,
                    "cuda_available": runtime.cuda_available,
                    "cuda_version": runtime.cuda_version,
                    "device_name": runtime.device_name,
                },
                "model_config": model_config,
                "env_config": env_config,
                "env_overrides": env_overrides,
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
            mode=args.mode,
            num_env_runners=resolved_num_env_runners,
            num_envs_per_env_runner=resolved_num_envs_per_runner,
            vector_mode=resolved_vector_mode.value,
            device=args.device,
        )
        if hasattr(config, "build_algo"):
            algo = config.build_algo()
        else:
            algo = config.build()

    fieldnames = [
        "training_iteration",
        "episode_reward_mean",
        "episode_len_mean",
        "timesteps_total",
        "time_total_s",
        "evaluation_episode_reward_mean",
        "evaluation_episode_len_mean",
    ]
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
                        "evaluation_episode_reward_mean": metrics["evaluation_episode_reward_mean"],
                        "evaluation_episode_len_mean": metrics["evaluation_episode_len_mean"],
                        "timesteps_total": metrics["timesteps_total"],
                        "use_gpu": use_gpu,
                    },
                )

                if idx % max(args.checkpoint_every, 1) == 0 or idx == iterations:
                    ckpt = algo.save(checkpoint_dir=str(run_paths.checkpoints))
                    latest_checkpoint_path = _checkpoint_path(ckpt)
                    export_inference_artifact(
                        algo=algo,
                        checkpoint_path=Path(latest_checkpoint_path),
                        artifact_dir=run_paths.inference,
                        env_config=env_config,
                        model_config=model_config,
                    )
                    logger.info(
                        "checkpoint_saved",
                        extra={"iteration": idx, "checkpoint_path": latest_checkpoint_path},
                    )
    finally:
        algo.stop()
        ray.shutdown()

    print(
        f"training_complete mode={args.mode} iterations={iterations} run_dir={run_paths.root} "
        f"checkpoint={latest_checkpoint_path} use_gpu={use_gpu}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
