"""Evaluate or render a trained RLlib checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from f1rl.artifacts import new_run_paths, owning_run_dir, resolve_checkpoint
from f1rl.env import F1RaceEnv
from f1rl.inference import (
    deep_merge_dicts,
    default_eval_env_config,
    load_inference_policy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved RLlib checkpoint.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render a human-visible window.")
    parser.add_argument("--export-frames", action="store_true", help="Save frames under artifacts.")
    parser.add_argument(
        "--legacy-algorithm-restore",
        action="store_true",
        help="Use full RLlib Algorithm checkpoint restore instead of the lightweight inference artifact.",
    )
    return parser.parse_args(argv)


def eval_env_config(args: argparse.Namespace) -> dict:
    return default_eval_env_config(
        steps=max(300, args.steps),
        render=args.render,
        headless=args.headless,
        export_frames=args.export_frames,
        frame_prefix="eval",
    )


def _training_env_config_for_checkpoint(checkpoint_path: Path) -> dict:
    metadata_path = owning_run_dir(checkpoint_path) / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = metadata.get("env_config", {})
    return env_config if isinstance(env_config, dict) else {}


def run_eval(args: argparse.Namespace) -> int:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    backend = "legacy_algorithm" if args.legacy_algorithm_restore else "artifact"
    algo = None
    if args.legacy_algorithm_restore:
        from ray.rllib.algorithms.algorithm import Algorithm

        from f1rl.rllib_utils import compute_inference_action

        algo = Algorithm.from_checkpoint(str(checkpoint_path))

        def policy_fn(obs):
            return compute_inference_action(algo, obs, explore=False)

    else:
        policy = load_inference_policy(checkpoint_path)

        def policy_fn(obs):
            return policy.compute_action(obs, explore=False)

    env = F1RaceEnv(deep_merge_dicts(_training_env_config_for_checkpoint(checkpoint_path), eval_env_config(args)))
    run_paths = new_run_paths(prefix="eval")

    total_reward = 0.0
    total_steps = 0
    episode_count = 0
    try:
        obs, _ = env.reset(seed=args.seed)
        while total_steps < args.steps and episode_count < args.episodes:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            total_steps += 1

            if env.config.render.enabled:
                env.render()

            if terminated or truncated:
                episode_count += 1
                obs, _ = env.reset(seed=args.seed + episode_count)

        summary = {
            "checkpoint": str(checkpoint_path),
            "backend": backend,
            "steps": total_steps,
            "episodes": episode_count,
            "total_reward": total_reward,
            "avg_reward_per_step": total_reward / max(total_steps, 1),
        }
        summary_path = run_paths.root / "eval_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(
            f"eval_complete checkpoint={checkpoint_path} steps={total_steps} "
            f"episodes={episode_count} reward={total_reward:.3f} backend={backend} summary={summary_path}"
        )
        return 0
    finally:
        env.close()
        if algo is not None:
            algo.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
