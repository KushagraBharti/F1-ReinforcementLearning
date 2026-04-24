"""Manual Pygame driving mode."""

from __future__ import annotations

import argparse
import sys
import time

from f1rl.config import ARTIFACTS_DIR, RenderConfig, SimConfig
from f1rl.reference_agent import load_reference_profile, reference_pose_at
from f1rl.render import PygameRenderer, RenderGhost
from f1rl.sim import MonzaSim
from f1rl.telemetry import TelemetryWriter

MAX_CATCHUP_STEPS_PER_FRAME = 8


def run_manual(
    *,
    max_steps: int,
    seed: int,
    headless: bool,
    ghost_reference: bool = False,
    flying_start: bool = False,
) -> int:
    sim = MonzaSim(SimConfig(max_steps=max_steps))
    sim.reset(seed=seed)
    reference_profile = load_reference_profile() if ghost_reference or flying_start else None
    if flying_start and reference_profile is not None:
        sim.state.speed_mps = reference_profile.speed_at(0.0) / 3.6
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="manual" if not headless else "manual-headless", seed=seed)
    renderer = None if headless else PygameRenderer(sim.track, sim.config, render_config=RenderConfig())
    result = None
    try:
        if renderer is None:
            for _ in range(max_steps):
                action = 1
                result = sim.step(action)
                writer.write_step(result.telemetry)
                if result.terminated or result.truncated:
                    break
        else:
            steps_run = 0
            next_step_s = time.perf_counter()
            while steps_run < max_steps:
                if not renderer.poll():
                    break
                if renderer.reset_pressed():
                    sim.reset(seed=seed)
                    if flying_start and reference_profile is not None:
                        sim.state.speed_mps = reference_profile.speed_at(0.0) / 3.6
                    next_step_s = time.perf_counter()
                action = renderer.keyboard_action()

                now_s = time.perf_counter()
                steps_this_frame = 0
                while now_s >= next_step_s and steps_run < max_steps:
                    result = sim.step(action)
                    writer.write_step(result.telemetry)
                    steps_run += 1
                    next_step_s += sim.config.car.dt
                    steps_this_frame += 1
                    if result.terminated or result.truncated:
                        break
                    if steps_this_frame >= MAX_CATCHUP_STEPS_PER_FRAME:
                        # Avoid an unbounded catch-up spiral if rendering or disk IO stalls.
                        next_step_s = time.perf_counter()
                        break

                extra_lines: list[str] = []
                ghosts: list[RenderGhost] = []
                if reference_profile is not None and ghost_reference:
                    elapsed_s = sim.state.elapsed_steps * sim.config.car.dt
                    reference = reference_pose_at(sim, reference_profile, elapsed_s)
                    gap_m = reference.progress_m - sim.state.monotonic_progress_m
                    ghosts.append(
                        RenderGhost(
                            x=reference.x,
                            y=reference.y,
                            heading_rad=reference.heading_rad,
                            speed_kph=reference.speed_kph,
                            label="REF",
                        )
                    )
                    extra_lines.extend(
                        [
                            f"time {elapsed_s:6.2f}s",
                            f"ref {reference.speed_kph:6.1f} kph",
                            f"gap {gap_m:7.1f} m",
                        ]
                    )
                renderer.render(sim, human=True, extra_lines=extra_lines, ghosts=ghosts)
                if result is not None and (result.terminated or result.truncated):
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
    parser.add_argument("--ghost-reference", action="store_true", help="Overlay the Fast-F1 VER Monza reference lap.")
    parser.add_argument("--flying-start", action="store_true", help="Start manual mode at reference-lap entry speed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_manual(
        max_steps=args.max_steps,
        seed=args.seed,
        headless=args.headless,
        ghost_reference=args.ghost_reference,
        flying_start=args.flying_start,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
