"""Evaluate or render a trained RLlib checkpoint."""

from __future__ import annotations

import argparse
import json
import sys

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv
from f1rl.rllib_utils import compute_inference_action


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved RLlib checkpoint.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render a human-visible window.")
    parser.add_argument("--export-frames", action="store_true", help="Save frames under artifacts.")
    return parser.parse_args(argv)


def eval_env_config(args: argparse.Namespace) -> dict:
    config = EnvConfig()
    config.max_steps = max(300, args.steps)
    config.render_mode = "human" if args.render and not args.headless else "rgb_array" if args.export_frames else None
    config.render.enabled = args.render or args.export_frames
    config.headless = args.headless
    config.render.export_frames = args.export_frames
    config.render.frame_prefix = "eval"
    return to_dict(config)


def run_eval(args: argparse.Namespace) -> int:
    from ray.rllib.algorithms.algorithm import Algorithm

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    env = F1RaceEnv(eval_env_config(args))
    run_paths = new_run_paths(prefix="eval")

    total_reward = 0.0
    total_steps = 0
    episode_count = 0
    try:
        obs, _ = env.reset(seed=args.seed)
        while total_steps < args.steps and episode_count < args.episodes:
            action = compute_inference_action(algo, obs, explore=False)
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
            f"episodes={episode_count} reward={total_reward:.3f} summary={summary_path}"
        )
        return 0
    finally:
        env.close()
        algo.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
