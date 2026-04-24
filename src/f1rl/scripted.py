"""Deterministic geometric baseline driver."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from f1rl.config import ARTIFACTS_DIR, SimConfig
from f1rl.geometry import sample_polyline_at, wrap_radians
from f1rl.sim import MonzaSim
from f1rl.telemetry import TelemetryWriter


class ScriptedController:
    def __init__(self, lookahead_m: float = 95.0) -> None:
        self.lookahead_m = lookahead_m

    def action(self, sim: MonzaSim) -> int:
        target_px = (
            sim.state.monotonic_progress_m / sim.track.meters_per_pixel
            + self.lookahead_m / sim.track.meters_per_pixel
        )
        target = sample_polyline_at(sim.track.centerline, sim.track.centerline_s, np.asarray([target_px], dtype=np.float32))[0]
        dx = float(target[0] - sim.state.x)
        dy = float(target[1] - sim.state.y)
        desired = float(np.arctan2(-dy, dx))
        error = wrap_radians(desired - sim.state.heading_rad)
        abs_error = abs(error)
        steer_left = error > 0.08
        steer_right = error < -0.08
        brake = abs_error > 0.7 or (abs_error > 0.42 and sim.state.speed_mps > 38.0)
        throttle = not brake and abs_error < 0.75
        if throttle and steer_left:
            return 5
        if throttle and steer_right:
            return 6
        if brake and steer_left:
            return 7
        if brake and steer_right:
            return 8
        if throttle:
            return 1
        if brake:
            return 2
        if steer_left:
            return 3
        if steer_right:
            return 4
        return 0


def run_scripted(*, steps: int, seed: int, telemetry: bool = True) -> int:
    sim = MonzaSim(SimConfig(max_steps=steps))
    sim.reset(seed=seed)
    controller = ScriptedController()
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="scripted", seed=seed) if telemetry else None
    try:
        for _ in range(steps):
            result = sim.step(controller.action(sim))
            if writer:
                writer.write_step(result.telemetry)
            if result.terminated or result.truncated:
                break
        if writer:
            summary = writer.close_episode(
                termination_reason=sim.termination_reason,
                completed_lap=sim.completed_lap,
            )
            print(f"scripted_complete run={writer.root} reason={summary.termination_reason}")
    finally:
        pass
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic scripted Monza driver.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-telemetry", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_scripted(steps=args.steps, seed=args.seed, telemetry=not args.no_telemetry)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
