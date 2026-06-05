"""Replay policy swarm telemetry grouped by policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from f1rl.replay import run_replay_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay policy checkpoint swarm telemetry.")
    parser.add_argument("path")
    parser.add_argument("--by-checkpoint", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-limit", type=int)
    parser.add_argument("--sort-by", default="name")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_replay_paths(
        [Path(args.path)],
        headless=args.headless,
        speed=args.speed,
        limit=args.limit,
        sort_by=args.sort_by,
        by_generation=args.by_checkpoint,
        generation_limit=args.checkpoint_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
