"""Headless or rendered rollout entrypoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from f1rl.artifacts import resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv
from f1rl.rllib_utils import compute_inference_action


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
    return parser.parse_args(argv)


def build_rollout_env_config(args: argparse.Namespace) -> dict:
    config = EnvConfig()
    config.max_steps = max(200, args.steps)
    config.render_mode = "human" if args.render and not args.headless else "rgb_array" if args.export_frames else None
    config.render.enabled = args.render or args.export_frames
    config.headless = args.headless
    config.render.export_frames = args.export_frames
    config.render.frame_prefix = "rollout"
    config.render.draw_hud = args.render or args.export_frames
    return to_dict(config)


def _checkpoint_policy(checkpoint_path: str):
    from ray.rllib.algorithms.algorithm import Algorithm

    checkpoint = resolve_checkpoint(checkpoint_path)
    algo = Algorithm.from_checkpoint(str(checkpoint))

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return compute_inference_action(algo, obs, explore=False)

    return algo, policy_fn


def run_rollout(args: argparse.Namespace) -> int:
    algo = None
    if args.policy == "checkpoint":
        algo, policy_fn = _checkpoint_policy(args.checkpoint)
    else:
        rng = np.random.default_rng(args.seed)

        def policy_fn(obs: np.ndarray) -> np.ndarray:
            _ = obs
            return rng.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)

    env = F1RaceEnv(build_rollout_env_config(args))
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
            f"rollout_complete policy={args.policy} steps={total_steps} "
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
