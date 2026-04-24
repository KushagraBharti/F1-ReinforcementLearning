"""Checkpoint evaluation and telemetry export."""

from __future__ import annotations

import argparse
import sys

from f1rl.config import ARTIFACTS_DIR, SimConfig
from f1rl.env import MonzaEnv
from f1rl.policy_io import load_sb3_ppo
from f1rl.telemetry import TelemetryWriter


def run_eval(*, checkpoint: str, steps: int, seed: int, device: str) -> int:
    env = MonzaEnv(SimConfig(max_steps=steps))
    model = load_sb3_ppo(checkpoint, env=env, device=device)
    obs, _ = env.reset(seed=seed)
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="eval", seed=seed)
    try:
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if env.last_telemetry is not None:
                writer.write_step(env.last_telemetry)
            if terminated or truncated:
                break
    finally:
        env.close()
    summary = writer.close_episode(
        termination_reason=env.sim.termination_reason,
        completed_lap=env.sim.completed_lap,
    )
    print(f"eval_complete run={writer.root} reason={summary.termination_reason}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint.")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_eval(checkpoint=args.checkpoint, steps=args.steps, seed=args.seed, device=args.device)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
