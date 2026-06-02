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
    def __init__(self, lookahead_m: float = 32.0) -> None:
        self.lookahead_m = lookahead_m

    def controls(self, sim: MonzaSim) -> tuple[float, float, float]:
        speed_kph = sim.state.speed_mps * 3.6
        _, lateral_error_m, _ = sim._track_errors()
        dynamic_lookahead_m = float(np.clip(self.lookahead_m + speed_kph * 0.08, 28.0, 70.0))
        target_px = (
            sim.state.monotonic_progress_m / sim.track.meters_per_pixel
            + dynamic_lookahead_m / sim.track.meters_per_pixel
        )
        target = sample_polyline_at(sim.track.centerline, sim.track.centerline_s, np.asarray([target_px], dtype=np.float32))[0]
        dx = float(target[0] - sim.state.x)
        dy = float(target[1] - sim.state.y)
        desired = float(np.arctan2(-dy, dx))
        error = wrap_radians(desired - sim.state.heading_rad)
        abs_error = abs(error)
        steer = float(np.clip(error / np.deg2rad(22.0), -1.0, 1.0))
        target_speed_kph = float(np.clip(105.0 - abs_error * 120.0 - abs(lateral_error_m) * 6.0, 40.0, 105.0))
        if abs_error > 0.9:
            target_speed_kph = min(target_speed_kph, 55.0)
        if abs(lateral_error_m) > 7.0:
            target_speed_kph = min(target_speed_kph, 45.0)
        speed_error = target_speed_kph - speed_kph
        if speed_error > 8.0 and abs_error < 1.0:
            throttle = float(np.clip(speed_error / 55.0, 0.12, 0.55))
            brake = 0.0
        elif speed_error < -3.0:
            throttle = 0.0
            brake = float(np.clip(-speed_error / 60.0, 0.1, 0.9))
        else:
            throttle = 0.12 if abs_error < 0.35 else 0.0
            brake = 0.0
        return throttle, brake, steer

    def action(self, sim: MonzaSim) -> int:
        throttle, brake, steer = self.controls(sim)
        steer_left = steer < -0.35
        steer_right = steer > 0.35
        throttle_on = throttle > 0.1
        brake_on = brake > 0.1
        if throttle_on and steer_left:
            return 5
        if throttle_on and steer_right:
            return 6
        if brake_on and steer_left:
            return 7
        if brake_on and steer_right:
            return 8
        if throttle_on:
            return 1
        if brake_on:
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
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="scripted", seed=seed, lap_length_m=sim.track.length_m) if telemetry else None
    try:
        for _ in range(steps):
            throttle, brake, steer = controller.controls(sim)
            result = sim.step_controls(
                throttle=throttle,
                brake=brake,
                steer=steer,
                action_id=-10,
                collect_observation=False,
                collect_rays=telemetry,
            )
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
        else:
            print(
                "scripted_complete "
                f"reason={sim.termination_reason} completed={sim.completed_lap} "
                f"progress={sim.state.monotonic_progress_m:.1f}m "
                f"time={sim.state.elapsed_steps * sim.config.car.dt:.1f}s"
            )
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
