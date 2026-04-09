"""Headless or rendered rollout entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from f1rl.artifacts import owning_run_dir, resolve_checkpoint
from f1rl.env import F1RaceEnv
from f1rl.inference import deep_merge_dicts, default_eval_env_config, load_inference_policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out random or checkpoint policy.")
    parser.add_argument("--steps", type=int, default=300, help="Total rollout steps.")
    parser.add_argument("--episodes", type=int, default=3, help="Maximum episodes before stopping.")
    parser.add_argument("--seed", type=int, default=11, help="Seed for resets.")
    parser.add_argument("--headless", action="store_true", help="Disable display window.")
    parser.add_argument("--policy", choices=["random", "checkpoint"], default="random")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--render", action="store_true", help="Render environment while rolling out.")
    parser.add_argument("--export-frames", action="store_true", help="Save rendered frames under artifacts.")
    parser.add_argument(
        "--legacy-algorithm-restore",
        action="store_true",
        help="Use full RLlib Algorithm checkpoint restore instead of the lightweight inference artifact.",
    )
    return parser.parse_args(argv)


def build_rollout_env_config(args: argparse.Namespace) -> dict:
    return default_eval_env_config(
        steps=max(200, args.steps),
        render=args.render,
        headless=args.headless,
        export_frames=args.export_frames,
        frame_prefix="rollout",
    )


def _training_env_config_for_checkpoint(checkpoint_path: Path) -> dict:
    metadata_path = owning_run_dir(checkpoint_path) / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = metadata.get("env_config", {})
    return env_config if isinstance(env_config, dict) else {}


def _checkpoint_policy(checkpoint_path: str, *, legacy_algorithm_restore: bool):
    checkpoint = resolve_checkpoint(checkpoint_path)
    if legacy_algorithm_restore:
        from ray.rllib.algorithms.algorithm import Algorithm

        from f1rl.rllib_utils import compute_inference_action

        algo = Algorithm.from_checkpoint(str(checkpoint))

        def policy_fn(obs: np.ndarray) -> np.ndarray:
            return compute_inference_action(algo, obs, explore=False)

        return algo, checkpoint, policy_fn, "legacy_algorithm"

    policy = load_inference_policy(checkpoint)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return policy.compute_action(obs, explore=False)

    return None, checkpoint, policy_fn, "artifact"


def run_rollout(args: argparse.Namespace) -> int:
    algo = None
    checkpoint = None
    backend = "random"
    if args.policy == "checkpoint":
        algo, checkpoint, policy_fn, backend = _checkpoint_policy(
            args.checkpoint,
            legacy_algorithm_restore=args.legacy_algorithm_restore,
        )
    else:
        rng = np.random.default_rng(args.seed)

        def policy_fn(obs: np.ndarray) -> np.ndarray:
            _ = obs
            return rng.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)

    env_config = build_rollout_env_config(args)
    if checkpoint is not None:
        env_config = deep_merge_dicts(_training_env_config_for_checkpoint(checkpoint), env_config)
    env = F1RaceEnv(env_config)
    episode_count = 0
    total_reward = 0.0
    total_steps = 0
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

        print(
            f"rollout_complete policy={args.policy} backend={backend} steps={total_steps} "
            f"episodes={episode_count} reward={total_reward:.3f}"
        )
        return 0
    finally:
        env.close()
        if algo is not None:
            algo.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_rollout(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
