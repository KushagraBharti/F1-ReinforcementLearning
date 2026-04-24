"""Headless or rendered rollout entrypoint for the torch-native runtime."""

from __future__ import annotations

import argparse
import sys

import torch

from f1rl.inference import load_inference_policy, training_env_config_for_checkpoint
from f1rl.torch_runtime import TERMINATION_REASONS, TorchSimBatch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out random or checkpoint policy.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--policy", choices=["random", "checkpoint"], default="random")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser.parse_args(argv)


def run_rollout(args: argparse.Namespace) -> int:
    policy = None
    backend = "random"
    env_config = {}
    if args.policy == "checkpoint":
        policy = load_inference_policy(args.checkpoint, device=args.device)
        env_config = training_env_config_for_checkpoint(args.checkpoint)
        backend = "torch_native"
    simulator = TorchSimBatch(env_config, num_cars=1, device=args.device)
    observations = simulator.reset(seed=args.seed)
    rng = torch.Generator(device=simulator.device)
    rng.manual_seed(args.seed)
    total_reward = 0.0
    episode_count = 0
    while int(simulator.step_count.max().item()) < args.steps and episode_count < args.episodes:
        if policy is None:
            action = torch.rand((1, simulator.action_dim), device=simulator.device, generator=rng) * 2.0 - 1.0
        else:
            action = policy.compute_actions(observations, deterministic=True)
        step = simulator.step(action)
        observations = step.observations
        total_reward += float(step.rewards.sum().item())
        if bool((step.terminated | step.truncated).any().item()):
            episode_count += 1
            if episode_count < args.episodes:
                observations = simulator.reset(seed=args.seed + episode_count)

    print(
        f"rollout_complete policy={args.policy} backend={backend} steps={int(simulator.step_count.max().item())} "
        f"episodes={episode_count} reward={total_reward:.3f} reason={TERMINATION_REASONS[int(simulator.reason_codes[0].item())]}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_rollout(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
