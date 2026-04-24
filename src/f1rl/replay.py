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


def _lerp_angle_deg(start: float, end: float, alpha: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return start + delta * alpha


def _interpolate_row(steps: list[dict[str, Any]], times: np.ndarray, replay_time_s: float) -> dict[str, Any]:
    if not steps:
        raise ValueError("cannot interpolate empty replay")
    first_time = float(times[0])
    target_time = first_time + replay_time_s
    if target_time <= first_time:
        return dict(steps[0])
    if target_time >= float(times[-1]):
        return dict(steps[-1])

    idx = int(np.searchsorted(times, target_time, side="right") - 1)
    idx = int(np.clip(idx, 0, len(steps) - 2))
    current = steps[idx]
    nxt = steps[idx + 1]
    span = max(float(times[idx + 1] - times[idx]), 1e-9)
    alpha = float(np.clip((target_time - float(times[idx])) / span, 0.0, 1.0))
    row = dict(current)

    numeric_keys = (
        "sim_time_s",
        "x",
        "y",
        "speed_mps",
        "speed_kph",
        "yaw_rate_rps",
        "throttle",
        "brake",
        "steering",
        "raw_progress_m",
        "monotonic_progress_m",
        "progress_delta_m",
        "lateral_error_m",
        "heading_error_deg",
        "reward_total",
    )
    for key in numeric_keys:
        if key in current and key in nxt:
            row[key] = float(current[key]) + (float(nxt[key]) - float(current[key])) * alpha
    row["heading_deg"] = _lerp_angle_deg(float(current["heading_deg"]), float(nxt["heading_deg"]), alpha)
    row["checkpoint_index"] = int(current["checkpoint_index"])
    row["lap_index"] = int(current["lap_index"])
    row["terminated"] = False
    row["truncated"] = False
    row["termination_reason"] = "active"
    return row


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
    times = np.asarray([float(row["sim_time_s"]) for row in steps], dtype=np.float64)
    first_time_s = float(times[0]) if len(times) else 0.0
    duration_s = float(times[-1] - times[0]) if len(times) else 0.0
    start_wall_s = time.perf_counter()
    try:
        if realtime:
            while True:
                if not renderer.poll():
                    break
                replay_time_s = min((time.perf_counter() - start_wall_s) * speed, duration_s)
                row = _interpolate_row(steps, times, replay_time_s)
                _apply_row(sim, row)
                renderer.render(
                    sim,
                    human=True,
                    extra_lines=[
                        f"replay {replay_time_s:6.2f}s",
                        f"speed x{speed:.2f}",
                    ],
                )
                if replay_time_s >= duration_s:
                    break
        else:
            for row in steps:
                if not renderer.poll():
                    break
                _apply_row(sim, row)
                replay_time_s = float(row["sim_time_s"]) - first_time_s
                renderer.render(
                    sim,
                    human=True,
                    extra_lines=[
                        f"replay {replay_time_s:6.2f}s",
                        "untimed",
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
