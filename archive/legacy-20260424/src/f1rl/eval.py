"""Evaluate a torch-native checkpoint."""

from __future__ import annotations

import argparse
import json
import sys

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.inference import load_inference_policy, training_env_config_for_checkpoint
from f1rl.torch_runtime import TERMINATION_REASONS, TorchSimBatch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved torch-native checkpoint.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser.parse_args(argv)


def run_eval(args: argparse.Namespace) -> int:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    policy = load_inference_policy(checkpoint_path, device=args.device)
    env_config = training_env_config_for_checkpoint(checkpoint_path)
    simulator = TorchSimBatch(env_config, num_cars=1, device=args.device)
    observations = simulator.reset(seed=args.seed)
    run_paths = new_run_paths(prefix="eval")

    total_reward = 0.0
    episode_count = 0
    checkpoint_events: list[dict[str, float | int]] = []
    while int(simulator.step_count.max().item()) < args.steps and episode_count < args.episodes:
        action = policy.compute_actions(observations, deterministic=True)
        step = simulator.step(action)
        observations = step.observations
        total_reward += float(step.rewards.sum().item())
        checkpoint_events.extend(step.checkpoint_events)
        if bool((step.terminated | step.truncated).any().item()):
            episode_count += 1
            if episode_count < args.episodes:
                observations = simulator.reset(seed=args.seed + episode_count)

    snapshot = simulator.snapshot()
    summary = {
        "checkpoint": str(checkpoint_path),
        "backend": "torch_native",
        "steps": int(simulator.step_count.max().item()),
        "episodes": episode_count,
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / max(int(simulator.step_count.max().item()), 1),
        "avg_speed_kph": float(snapshot["speed_kph"].mean().item()),
        "distance_travelled": float(snapshot["distance_travelled"].max().item()),
        "current_checkpoint_index": int(snapshot["current_checkpoint_index"].max().item()),
        "termination_reason": TERMINATION_REASONS[int(snapshot["termination_code"][0].item())],
        "checkpoint_events": checkpoint_events,
        "runtime": simulator.runtime_metadata(policy_device=policy.device, renderer_backend="headless"),
    }
    summary_path = run_paths.root / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"eval_complete checkpoint={checkpoint_path} steps={summary['steps']} "
        f"episodes={episode_count} reward={total_reward:.3f} backend=torch_native summary={summary_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_eval(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
