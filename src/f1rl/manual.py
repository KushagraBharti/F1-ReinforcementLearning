"""Manual Pygame driving mode."""

from __future__ import annotations

import argparse
import sys

from f1rl.config import ARTIFACTS_DIR, RenderConfig, SimConfig
from f1rl.render import PygameRenderer
from f1rl.sim import MonzaSim
from f1rl.telemetry import TelemetryWriter


def run_manual(*, max_steps: int, seed: int, headless: bool) -> int:
    sim = MonzaSim(SimConfig(max_steps=max_steps))
    sim.reset(seed=seed)
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="manual" if not headless else "manual-headless", seed=seed)
    renderer = None if headless else PygameRenderer(sim.track, sim.config, render_config=RenderConfig())
    try:
        for _ in range(max_steps):
            if renderer is None:
                action = 1
            else:
                if not renderer.poll():
                    break
                if renderer.reset_pressed():
                    sim.reset(seed=seed)
                action = renderer.keyboard_action()
            result = sim.step(action)
            writer.write_step(result.telemetry)
            if renderer is not None:
                renderer.render(sim, human=True)
            if result.terminated or result.truncated:
                break
    finally:
        if renderer is not None:
            renderer.close()
    summary = writer.close_episode(termination_reason=sim.termination_reason, completed_lap=sim.completed_lap)
    print(f"manual_complete run={writer.root} reason={summary.termination_reason}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Monza driving mode.")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_manual(max_steps=args.max_steps, seed=args.seed, headless=args.headless)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
