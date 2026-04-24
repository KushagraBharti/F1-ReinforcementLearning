"""Replay telemetry JSONL files."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import SimConfig
from f1rl.render import PygameRenderer
from f1rl.sim import MonzaSim
from f1rl.telemetry import load_steps


def _apply_row(sim: MonzaSim, row: dict[str, Any]) -> None:
    sim.state.x = float(row["x"])
    sim.state.y = float(row["y"])
    sim.state.heading_rad = float(np.deg2rad(row["heading_deg"]))
    sim.state.speed_mps = float(row["speed_mps"])
    sim.state.yaw_rate_rps = float(row["yaw_rate_rps"])
    sim.state.checkpoint_index = int(row["checkpoint_index"])
    sim.state.lap_index = int(row["lap_index"])
    sim.state.raw_progress_m = float(row["raw_progress_m"])
    sim.state.monotonic_progress_m = float(row["monotonic_progress_m"])
    sim.state.alive = not bool(row.get("terminated", False))
    sim.termination_reason = str(row["termination_reason"])


def _sleep_until_replay_time(renderer: PygameRenderer, *, start_wall_s: float, target_replay_s: float) -> bool:
    while True:
        if not renderer.poll():
            return False
        remaining = target_replay_s - (time.perf_counter() - start_wall_s)
        if remaining <= 0.0:
            return True
        time.sleep(min(remaining, 0.01))


def run_replay(path: Path, *, headless: bool, speed: float = 1.0, realtime: bool = True) -> int:
    steps = load_steps(path)
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if headless:
        duration = float(steps[-1]["sim_time_s"]) - float(steps[0]["sim_time_s"]) if steps else 0.0
        print(f"replay_loaded steps={len(steps)} duration={duration:.3f}s path={path}")
        return 0
    sim = MonzaSim(SimConfig())
    renderer = PygameRenderer(sim.track, sim.config)
    first_time_s = float(steps[0]["sim_time_s"]) if steps else 0.0
    start_wall_s = time.perf_counter()
    try:
        for row in steps:
            if realtime:
                target_s = (float(row["sim_time_s"]) - first_time_s) / speed
                if not _sleep_until_replay_time(renderer, start_wall_s=start_wall_s, target_replay_s=target_s):
                    break
            elif not renderer.poll():
                break
            _apply_row(sim, row)
            replay_time_s = float(row["sim_time_s"]) - first_time_s
            renderer.render(
                sim,
                human=True,
                extra_lines=[
                    f"replay {replay_time_s:6.2f}s",
                    f"speed x{speed:.2f}",
                ],
            )
    finally:
        renderer.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved telemetry JSONL file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier; default is real time.")
    parser.add_argument("--no-timing", action="store_true", help="Advance one telemetry row per render frame.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_replay(args.path, headless=args.headless, speed=args.speed, realtime=not args.no_timing)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
