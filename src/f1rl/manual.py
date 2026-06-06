"""Manual Pygame driving mode."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from f1rl.calibration import reference_trace_features
from f1rl.config import ARTIFACTS_DIR, RenderConfig, SimConfig, action_to_controls
from f1rl.reference_agent import ReferenceProfile, load_reference_profile, reference_pose_at
from f1rl.render import PygameRenderer, RenderGhost
from f1rl.sim import MonzaSim
from f1rl.telemetry import StepTelemetry, TelemetryWriter

MAX_CATCHUP_STEPS_PER_FRAME = 8


@dataclass(slots=True)
class ManualStart:
    reset_options: dict[str, float]
    reference_time_offset_s: float = 0.0
    reference_start_distance_m: float = 0.0


def _sim_progress_from_reference_distance(sim: MonzaSim, profile: ReferenceProfile, distance_m: float) -> float:
    return float(distance_m / max(profile.distance_max_m, 1e-9) * sim.track.length_m)


def _reference_distance_from_sim_progress(sim: MonzaSim, profile: ReferenceProfile, progress_m: float) -> float:
    return float(progress_m / max(sim.track.length_m, 1e-9) * profile.distance_max_m)


def _sustained_section_start_distance(section_name: str, lead_in_m: float) -> float:
    normalized = section_name.strip().lower().replace("-", "_")
    targets = reference_trace_features()["sustained_corner_targets"]
    for target in targets:
        name = str(target["name"]).lower()
        index = int(target["index"]) + 1
        aliases = {
            name,
            f"sustained_{index:02d}",
            f"sustained_corner_{index:02d}",
            f"section_{index:02d}",
            str(index),
        }
        if normalized in aliases:
            return max(float(target["start_distance_m"]) - max(float(lead_in_m), 0.0), 0.0)
    valid = ", ".join(str(target["name"]) for target in targets)
    raise ValueError(f"unknown sustained corner section '{section_name}'. Valid sections: {valid}")


def _resolve_manual_start(
    *,
    sim: MonzaSim,
    reference_profile: ReferenceProfile | None,
    flying_start: bool,
    start_progress_m: float | None,
    start_speed_kph: float | None,
    start_section: str | None,
    start_section_lead_in_m: float,
) -> ManualStart:
    options: dict[str, float] = {}
    reference_start_distance_m = 0.0
    if start_section is not None:
        if reference_profile is None:
            raise ValueError("--start-section requires the FastF1 reference profile.")
        reference_start_distance_m = _sustained_section_start_distance(start_section, start_section_lead_in_m)
        options["start_progress_m"] = _sim_progress_from_reference_distance(
            sim,
            reference_profile,
            reference_start_distance_m,
        )
    elif start_progress_m is not None:
        options["start_progress_m"] = float(start_progress_m)
        if reference_profile is not None:
            reference_start_distance_m = _reference_distance_from_sim_progress(
                sim,
                reference_profile,
                float(start_progress_m),
            )

    if start_speed_kph is not None:
        options["start_speed_kph"] = float(start_speed_kph)
    elif flying_start and reference_profile is not None:
        options["start_speed_kph"] = reference_profile.speed_at(reference_start_distance_m)
    elif start_section is not None and reference_profile is not None:
        options["start_speed_kph"] = reference_profile.speed_at(reference_start_distance_m)

    reference_time_offset_s = (
        reference_profile.time_at(reference_start_distance_m) if reference_profile is not None else 0.0
    )
    return ManualStart(
        reset_options=options,
        reference_time_offset_s=reference_time_offset_s,
        reference_start_distance_m=reference_start_distance_m,
    )


def _attach_reference_telemetry(
    sim: MonzaSim,
    telemetry: StepTelemetry,
    reference_profile: ReferenceProfile | None,
    *,
    reference_time_offset_s: float = 0.0,
) -> None:
    if reference_profile is None:
        return
    elapsed_s = reference_time_offset_s + sim.state.elapsed_steps * sim.config.car.dt
    reference = reference_pose_at(sim, reference_profile, elapsed_s)
    telemetry.reference_progress_m = reference.progress_m
    telemetry.reference_speed_kph = reference.speed_kph
    telemetry.ghost_gap_m = reference.progress_m - sim.state.monotonic_progress_m


def run_manual(
    *,
    max_steps: int,
    seed: int,
    headless: bool,
    physics_model: str = "v1",
    ghost_reference: bool = False,
    flying_start: bool = False,
    start_progress_m: float | None = None,
    start_speed_kph: float | None = None,
    start_section: str | None = None,
    start_section_lead_in_m: float = 120.0,
) -> int:
    sim = MonzaSim(SimConfig(max_steps=max_steps, physics_model=physics_model))
    reference_profile = load_reference_profile() if ghost_reference or flying_start or start_section is not None else None
    manual_start = _resolve_manual_start(
        sim=sim,
        reference_profile=reference_profile,
        flying_start=flying_start,
        start_progress_m=start_progress_m,
        start_speed_kph=start_speed_kph,
        start_section=start_section,
        start_section_lead_in_m=start_section_lead_in_m,
    )
    sim.reset(seed=seed, options=manual_start.reset_options)
    writer = TelemetryWriter(
        ARTIFACTS_DIR,
        mode="manual" if not headless else "manual-headless",
        seed=seed,
        lap_length_m=sim.track.length_m,
    )
    renderer = None if headless else PygameRenderer(sim.track, sim.config, render_config=RenderConfig())
    result = None
    try:
        if renderer is None:
            for _ in range(max_steps):
                action = 1
                result = sim.step(action)
                if ghost_reference:
                    _attach_reference_telemetry(
                        sim,
                        result.telemetry,
                        reference_profile,
                        reference_time_offset_s=manual_start.reference_time_offset_s,
                    )
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
                    sim.reset(seed=seed, options=manual_start.reset_options)
                    next_step_s = time.perf_counter()
                action = renderer.keyboard_action()

                now_s = time.perf_counter()
                steps_this_frame = 0
                while now_s >= next_step_s and steps_run < max_steps:
                    throttle, brake, steer = action_to_controls(action)
                    result = sim.step_controls(
                        throttle=throttle,
                        brake=brake,
                        steer=steer,
                        action_id=action,
                        collect_observation=False,
                        collect_rays=True,
                    )
                    if ghost_reference:
                        _attach_reference_telemetry(
                            sim,
                            result.telemetry,
                            reference_profile,
                            reference_time_offset_s=manual_start.reference_time_offset_s,
                        )
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
                    elapsed_s = manual_start.reference_time_offset_s + sim.state.elapsed_steps * sim.config.car.dt
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
    parser.add_argument("--physics-model", choices=("v1", "v2"), default="v1")
    parser.add_argument("--ghost-reference", action="store_true", help="Overlay the Fast-F1 VER Monza reference lap.")
    parser.add_argument("--flying-start", action="store_true", help="Start manual mode at reference-lap entry speed.")
    parser.add_argument("--start-progress-m", type=float, help="Start at a specific simulator progress in meters.")
    parser.add_argument("--start-speed-kph", type=float, help="Start at a specific speed in kph.")
    parser.add_argument(
        "--start-section",
        help="Start before a named FastF1 sustained-corner target, e.g. sustained_corner_03 or section_03.",
    )
    parser.add_argument(
        "--start-section-lead-in-m",
        type=float,
        default=120.0,
        help="Lead-in distance before --start-section, in reference meters.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_manual(
        max_steps=args.max_steps,
        seed=args.seed,
        headless=args.headless,
        physics_model=args.physics_model,
        ghost_reference=args.ghost_reference,
        flying_start=args.flying_start,
        start_progress_m=args.start_progress_m,
        start_speed_kph=args.start_speed_kph,
        start_section=args.start_section,
        start_section_lead_in_m=args.start_section_lead_in_m,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
