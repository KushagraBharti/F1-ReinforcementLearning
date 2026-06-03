"""Strict fixed-action schedule probes for short curriculum blockers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

from f1rl.config import ARTIFACTS_DIR, SimConfig, actions_for_action_set
from f1rl.policy_io import resolve_ppo_eval_config
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import StateSnapshot, snapshot_from_sim, snapshot_to_dict


class Phase(NamedTuple):
    action: str
    steps: int


class Schedule(NamedTuple):
    name: str
    phases: tuple[Phase, ...]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _available_action_ids(action_set: str) -> dict[str, int]:
    return {name: index for index, (name, _, _, _) in enumerate(actions_for_action_set(action_set))}


def _phase_name(action: str, steps: int) -> str:
    return f"{action}x{steps}"


def default_schedules(action_set: str, *, max_steps: int) -> list[Schedule]:
    """Generate compact steering/throttle probes without assuming turn direction."""
    action_ids = _available_action_ids(action_set)
    action_specs = {
        name: (throttle, brake, steer)
        for name, throttle, brake, steer in actions_for_action_set(action_set)
    }

    def has(action: str) -> bool:
        return action in action_ids

    schedules: list[Schedule] = []

    def add(phases: list[tuple[str, int]]) -> None:
        kept = [Phase(action, int(steps)) for action, steps in phases if has(action) and steps > 0]
        if not kept:
            return
        name = "__".join(_phase_name(phase.action, phase.steps) for phase in kept)
        schedules.append(Schedule(name=name, phases=tuple(kept)))

    constants = list(action_ids)
    constants.extend(
        [
        "coast",
        "half_throttle",
        "throttle",
        "soft_left",
        "soft_right",
        "left",
        "right",
        "half_throttle_soft_left",
        "half_throttle_soft_right",
        "throttle_soft_left",
        "throttle_soft_right",
        "half_throttle_left",
        "half_throttle_right",
        "throttle_left",
            "throttle_right",
        ]
    )
    for action in constants:
        add([(action, max_steps)])

    generic_left = [name for name, (_, _, steer) in action_specs.items() if steer < -1e-6]
    generic_right = [name for name, (_, _, steer) in action_specs.items() if steer > 1e-6]
    generic_straight = [name for name, (_, _, steer) in action_specs.items() if abs(steer) <= 1e-6]
    generic_power = [
        name
        for name in generic_straight
        if action_specs[name][0] > 0.0 and action_specs[name][1] <= 0.0
    ]
    generic_neutral = [
        name
        for name in ("coast", "maintenance", "half_throttle", "soft_brake")
        if has(name)
    ] or generic_straight[:1]

    delay_steps = (60, 120, 180, 240, 300, 360, 420)
    turn_steps = (6, 12, 24, 48, 72, 96, 120)
    settle_steps = (12, 24, 48, 72)
    for side in ("left", "right"):
        steer_only = [f"soft_{side}", side]
        powered_soft = [f"half_throttle_soft_{side}", f"throttle_soft_{side}"]
        powered_hard = [f"half_throttle_{side}", f"throttle_{side}"]
        powered_any = [*powered_soft, *powered_hard]
        straight_power = ["half_throttle", "throttle"]
        for delay_duration in delay_steps:
            for turn_duration in turn_steps:
                for first_steer in steer_only:
                    for second_action in [*powered_soft, *straight_power]:
                        add(
                            [
                                ("coast", delay_duration),
                                (first_steer, turn_duration),
                                (second_action, max_steps),
                            ]
                        )
                for first_steer in powered_soft:
                    for second_action in [*powered_any, *straight_power]:
                        add(
                            [
                                ("coast", delay_duration),
                                (first_steer, turn_duration),
                                (second_action, max_steps),
                            ]
                        )
            for first_action in [*steer_only, *powered_soft]:
                add([("coast", delay_duration), (first_action, max_steps)])
        for turn_duration in turn_steps:
            for first_action in [*steer_only, *powered_soft]:
                for second_action in [*powered_any, *straight_power]:
                    add([(first_action, turn_duration), (second_action, max_steps)])
            for settle_duration in settle_steps:
                for first_action in steer_only:
                    for second_action in powered_soft:
                        for third_action in [*powered_any, *straight_power]:
                            add(
                                [
                                    (first_action, turn_duration),
                                    (second_action, settle_duration),
                                    (third_action, max_steps),
                                ]
                            )
                for first_action in powered_soft:
                    for second_action in straight_power:
                        for third_action in powered_any:
                            add(
                                [
                                    (first_action, turn_duration),
                                    (second_action, settle_duration),
                                    (third_action, max_steps),
                                ]
                            )

    for side_actions in (generic_left, generic_right):
        if not side_actions:
            continue
        for first_action in side_actions:
            for turn_duration in turn_steps:
                for second_action in [*side_actions, *generic_power, *generic_straight]:
                    add([(first_action, turn_duration), (second_action, max_steps)])
                for settle_duration in settle_steps:
                    for second_action in generic_straight:
                        for third_action in side_actions:
                            add(
                                [
                                    (first_action, turn_duration),
                                    (second_action, settle_duration),
                                    (third_action, max_steps),
                                ]
                            )
        for delay_action in generic_neutral:
            for delay_duration in delay_steps:
                for first_action in side_actions:
                    add([(delay_action, delay_duration), (first_action, max_steps)])
                    for turn_duration in turn_steps:
                        for second_action in [*side_actions, *generic_power, *generic_straight]:
                            add(
                                [
                                    (delay_action, delay_duration),
                                    (first_action, turn_duration),
                                    (second_action, max_steps),
                                ]
                            )

    unique: dict[str, Schedule] = {}
    for schedule in schedules:
        unique.setdefault(schedule.name, schedule)
    return list(unique.values())


def rotation_bridge_schedules(action_set: str, *, max_steps: int) -> list[Schedule]:
    """Generate a small preset for the Rettifilo post-release rotation bridge."""
    action_ids = _available_action_ids(action_set)

    def has(action: str) -> bool:
        return action in action_ids

    schedules: list[Schedule] = []

    def add(phases: list[tuple[str, int]]) -> None:
        kept = [Phase(action, int(steps)) for action, steps in phases if has(action) and steps > 0]
        if not kept:
            return
        name = "__".join(_phase_name(phase.action, phase.steps) for phase in kept)
        schedules.append(Schedule(name=name, phases=tuple(kept)))

    neutral_actions = [action for action in ("maintenance", "soft_brake") if has(action)]
    left_actions = [
        action
        for action in (
            "maintenance_soft_left",
            "soft_brake_soft_left",
            "maintenance_micro_left",
            "soft_brake_micro_left",
            "maintenance_tiny_left",
            "soft_brake_tiny_left",
        )
        if has(action)
    ]
    right_actions = [
        action
        for action in (
            "maintenance_soft_right",
            "soft_brake_soft_right",
        )
        if has(action)
    ]
    settle_actions = neutral_actions or [action for action in action_ids if has(action)][:1]

    for action in [*neutral_actions, *left_actions, *right_actions]:
        add([(action, max_steps)])

    delay_steps = (0, 6, 12, 18)
    lead_steps = (6, 12)
    turn_steps = (12, 24, 36)

    for delay in delay_steps:
        for neutral in neutral_actions:
            for left in left_actions:
                add([(neutral, delay), (left, max_steps)])
                for turn in turn_steps:
                    for settle in settle_actions:
                        add([(neutral, delay), (left, turn), (settle, max_steps)])

    for right in right_actions:
        for lead in lead_steps:
            for left in left_actions:
                add([(right, lead), (left, max_steps)])
                for turn in turn_steps:
                    for settle in settle_actions:
                        add([(right, lead), (left, turn), (settle, max_steps)])

    unique: dict[str, Schedule] = {}
    for schedule in schedules:
        unique.setdefault(schedule.name, schedule)
    return list(unique.values())


def schedules_for_preset(action_set: str, *, max_steps: int, schedule_preset: str) -> list[Schedule]:
    if schedule_preset == "default":
        return default_schedules(action_set, max_steps=max_steps)
    if schedule_preset == "rotation_bridge":
        return rotation_bridge_schedules(action_set, max_steps=max_steps)
    raise ValueError(f"Unknown schedule preset {schedule_preset!r}.")


def _schedule_action(schedule: Schedule, step_index: int) -> str:
    elapsed = 0
    for phase in schedule.phases:
        elapsed += phase.steps
        if step_index < elapsed:
            return phase.action
    return schedule.phases[-1].action


def _score(rows: list[dict[str, Any]], *, start_progress_m: float, target_progress_m: float) -> float:
    if not rows:
        return float("-inf")
    final = rows[-1]
    best_progress_m = max(float(row["monotonic_progress_m"]) for row in rows)
    progress_to_target_m = min(best_progress_m, target_progress_m) - start_progress_m
    remaining_m = max(0.0, target_progress_m - best_progress_m)
    score = progress_to_target_m - remaining_m * 2.0
    if final.get("segment_complete"):
        score += 100_000.0
    reason = final.get("termination_reason")
    if reason in {"collision", "off_track", "assist_virtual_corridor"}:
        score -= 5_000.0
    elif reason == "segment_speed_gate_failed":
        score -= 2_000.0
    elif reason == "no_progress":
        score -= 1_000.0
    if not final.get("collided") and not final.get("off_track"):
        score += 500.0
    score += min(float(final.get("speed_kph", 0.0)), 260.0)
    score -= abs(float(final.get("lateral_error_m", 0.0))) * 50.0
    score -= abs(float(final.get("heading_error_deg", 0.0))) * 10.0
    score -= int(final.get("missed_checkpoint_count", 0)) * 1000.0
    return float(score)


def _run_schedule(
    *,
    sim: MonzaSim,
    snapshot: StateSnapshot,
    schedule: Schedule,
    action_ids: dict[str, int],
    target_progress_m: float,
    target_max_speed_kph: float | None,
    max_steps: int,
    seed: int,
    collect_full_telemetry: bool,
    segment_require_release: bool,
    segment_fail_on_speed_gate_miss: bool,
) -> tuple[list[dict[str, Any]], MonzaSim]:
    sim.reset(
        seed=seed,
        options={
            "state_snapshot": snapshot_to_dict(snapshot),
            "segment_length_m": max(float(target_progress_m) - snapshot.monotonic_progress_m, 1.0),
            "segment_target_max_speed_kph": target_max_speed_kph,
            "segment_fail_on_speed_gate_miss": segment_fail_on_speed_gate_miss,
            "segment_require_release": segment_require_release,
            "curriculum_stage": "action-search",
            "collect_observation": collect_full_telemetry,
        },
    )
    action_specs = {name: spec for name, *spec in actions_for_action_set(sim.config.action_set)}
    rows: list[dict[str, Any]] = []
    for step_index in range(max_steps):
        action_name = _schedule_action(schedule, step_index)
        throttle, brake, steer = action_specs[action_name]
        result = sim.step_controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            action_id=action_ids[action_name],
            collect_observation=collect_full_telemetry,
            collect_rays=collect_full_telemetry,
        )
        if collect_full_telemetry:
            rows.append(asdict(result.telemetry))
        else:
            telemetry = result.telemetry
            rows.append(
                {
                    "monotonic_progress_m": telemetry.monotonic_progress_m,
                    "speed_kph": telemetry.speed_kph,
                    "lateral_error_m": telemetry.lateral_error_m,
                    "heading_error_deg": telemetry.heading_error_deg,
                    "missed_checkpoint_count": telemetry.missed_checkpoint_count,
                    "segment_complete": telemetry.segment_complete,
                    "collided": telemetry.collided,
                    "off_track": telemetry.off_track,
                    "termination_reason": telemetry.termination_reason,
                }
            )
        if result.terminated or result.truncated:
            break
    return rows, sim


def run_action_search(
    *,
    state_library: Path,
    checkpoint: Path | str,
    output_dir: Path,
    target_progress_m: float,
    target_max_speed_kph: float | None,
    max_steps: int,
    seed: int,
    top_k: int,
    start_min_progress_m: float | None = None,
    start_max_progress_m: float | None = None,
    max_schedules: int | None = None,
    schedule_offset: int = 0,
    schedule_preset: str = "default",
    segment_require_release: bool = False,
    segment_fail_on_speed_gate_miss: bool = True,
    metadata_mode: str = "require",
    override_action_set: str | None = None,
) -> Path:
    snapshots = load_state_library(state_library)
    if start_min_progress_m is not None:
        snapshots = [
            snapshot for snapshot in snapshots if snapshot.monotonic_progress_m + 1e-9 >= start_min_progress_m
        ]
    if start_max_progress_m is not None:
        snapshots = [
            snapshot for snapshot in snapshots if snapshot.monotonic_progress_m - 1e-9 <= start_max_progress_m
        ]
    if not snapshots:
        raise ValueError("No state-library snapshots match requested action-search range.")

    ppo_config = resolve_ppo_eval_config(
        checkpoint,
        max_steps=max_steps,
        fallback_config=SimConfig(max_steps=max_steps),
        metadata_mode=metadata_mode,
    )
    sim_config = ppo_config.sim_config
    if override_action_set is not None:
        actions_for_action_set(override_action_set)
        sim_config.action_set = override_action_set
    action_ids = _available_action_ids(sim_config.action_set)
    schedules = schedules_for_preset(
        sim_config.action_set,
        max_steps=max_steps,
        schedule_preset=schedule_preset,
    )
    if schedule_offset > 0:
        schedules = schedules[int(schedule_offset) :]
    if max_schedules is not None:
        schedules = schedules[: max(int(max_schedules), 1)]
    if not schedules:
        raise ValueError(f"No default schedules are available for action set {sim_config.action_set!r}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_telemetry"
    probe_sim = MonzaSim(sim_config)
    attempts: list[dict[str, Any]] = []
    ranked: list[tuple[float, StateSnapshot, Schedule, dict[str, Any]]] = []
    attempt_index = 0
    for snapshot in snapshots:
        for schedule in schedules:
            rows, _ = _run_schedule(
                sim=probe_sim,
                snapshot=snapshot,
                schedule=schedule,
                action_ids=action_ids,
                target_progress_m=target_progress_m,
                target_max_speed_kph=target_max_speed_kph,
                max_steps=max_steps,
                seed=seed + attempt_index,
                collect_full_telemetry=False,
                segment_require_release=segment_require_release,
                segment_fail_on_speed_gate_miss=segment_fail_on_speed_gate_miss,
            )
            score = _score(
                rows,
                start_progress_m=snapshot.monotonic_progress_m,
                target_progress_m=target_progress_m,
            )
            final = rows[-1] if rows else {}
            best_progress_m = max([snapshot.monotonic_progress_m, *[float(row["monotonic_progress_m"]) for row in rows]])
            summary = {
                "attempt": attempt_index,
                "seed": seed + attempt_index,
                "score": score,
                "schedule": schedule.name,
                "schedule_phases": [phase._asdict() for phase in schedule.phases],
                "start_snapshot_id": snapshot.id,
                "start_progress_m": snapshot.monotonic_progress_m,
                "start_speed_kph": snapshot.speed_mps * 3.6,
                "best_progress_m": best_progress_m,
                "final_progress_m": final.get("monotonic_progress_m", snapshot.monotonic_progress_m),
                "remaining_m": max(0.0, target_progress_m - best_progress_m),
                "segment_complete": bool(final.get("segment_complete", False)),
                "termination_reason": final.get("termination_reason", "empty"),
                "collided": bool(final.get("collided", False)),
                "off_track": bool(final.get("off_track", False)),
                "final_speed_kph": final.get("speed_kph"),
                "final_lateral_error_m": final.get("lateral_error_m"),
                "final_heading_error_deg": final.get("heading_error_deg"),
                "steps": len(rows),
            }
            attempts.append(summary)
            ranked.append((score, snapshot, schedule, summary))
            attempt_index += 1

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_items = ranked[:top_k]
    elite_snapshots: list[StateSnapshot] = []
    replay_sim = MonzaSim(sim_config)
    for rank, (_, snapshot, schedule, summary) in enumerate(top_items):
        rows, sim = _run_schedule(
            sim=replay_sim,
            snapshot=snapshot,
            schedule=schedule,
            action_ids=action_ids,
            target_progress_m=target_progress_m,
            target_max_speed_kph=target_max_speed_kph,
            max_steps=max_steps,
            seed=int(summary["seed"]),
            collect_full_telemetry=True,
            segment_require_release=segment_require_release,
            segment_fail_on_speed_gate_miss=segment_fail_on_speed_gate_miss,
        )
        telemetry_path = selected_dir / f"action-rank-{rank:03d}-attempt-{summary['attempt']:05d}-steps.jsonl"
        _write_jsonl(telemetry_path, rows)
        summary["selected_telemetry"] = str(telemetry_path)
        elite_snapshots.append(snapshot_from_sim(sim, source="action_search", source_file=str(state_library)))

    _write_jsonl(output_dir / "attempts.jsonl", attempts)
    elite_library_path = output_dir / "elite_state_library.json"
    write_state_library(
        elite_library_path,
        elite_snapshots,
        source="action_search",
        metadata={
            "state_library": str(state_library),
            "checkpoint": str(checkpoint),
            "target_progress_m": target_progress_m,
            "target_max_speed_kph": target_max_speed_kph,
            "max_steps": max_steps,
            "seed": seed,
            "top_k": top_k,
            "start_min_progress_m": start_min_progress_m,
            "start_max_progress_m": start_max_progress_m,
            "schedule_count": len(schedules),
            "schedule_preset": schedule_preset,
            "schedule_offset": schedule_offset,
            "attempt_count": len(attempts),
            "segment_require_release": segment_require_release,
            "segment_fail_on_speed_gate_miss": segment_fail_on_speed_gate_miss,
            "metadata_mode": metadata_mode,
            "override_action_set": override_action_set,
        },
    )
    termination_reasons = Counter(str(attempt["termination_reason"]) for attempt in attempts)
    completion_count = sum(1 for attempt in attempts if attempt["segment_complete"])
    summary = {
        "run_id": output_dir.name,
        "state_library": str(state_library),
        "checkpoint_config": ppo_config.report(),
        "override_action_set": override_action_set,
        "elite_state_library": str(elite_library_path),
        "target_progress_m": target_progress_m,
        "target_max_speed_kph": target_max_speed_kph,
        "snapshot_count": len(snapshots),
        "schedule_count": len(schedules),
        "schedule_preset": schedule_preset,
        "attempt_count": len(attempts),
        "completion_count": completion_count,
        "completion_rate": completion_count / max(len(attempts), 1),
        "best_attempt": top_items[0][3] if top_items else None,
        "top_attempts": [item[3] for item in top_items],
        "termination_reasons": dict(termination_reasons),
    }
    (output_dir / "action_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"action-search-{timestamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe strict short sections with fixed action schedules.")
    parser.add_argument("--state-library", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-progress-m", type=float, required=True)
    parser.add_argument("--target-max-speed-kph", type=float)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--start-min-progress-m", type=float)
    parser.add_argument("--start-max-progress-m", type=float)
    parser.add_argument("--max-schedules", type=int)
    parser.add_argument("--schedule-offset", type=int, default=0)
    parser.add_argument("--schedule-preset", choices=["default", "rotation_bridge"], default="default")
    parser.add_argument("--segment-require-release", action="store_true")
    parser.add_argument(
        "--no-segment-fail-on-speed-gate-miss",
        action="store_true",
        help="Allow target crossing above speed gate to continue instead of truncating.",
    )
    parser.add_argument("--metadata-mode", choices=["auto", "require", "ignore"], default="require")
    parser.add_argument("--override-action-set")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    run_root = run_action_search(
        state_library=args.state_library,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        target_progress_m=args.target_progress_m,
        target_max_speed_kph=args.target_max_speed_kph,
        max_steps=args.max_steps,
        seed=args.seed,
        top_k=args.top_k,
        start_min_progress_m=args.start_min_progress_m,
        start_max_progress_m=args.start_max_progress_m,
        max_schedules=args.max_schedules,
        schedule_offset=args.schedule_offset,
        schedule_preset=args.schedule_preset,
        segment_require_release=bool(args.segment_require_release),
        segment_fail_on_speed_gate_miss=not bool(args.no_segment_fail_on_speed_gate_miss),
        metadata_mode=args.metadata_mode,
        override_action_set=args.override_action_set,
    )
    print(f"action_search_complete run={run_root} summary={run_root / 'action_search_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
