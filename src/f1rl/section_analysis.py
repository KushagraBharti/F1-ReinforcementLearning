"""Section-level telemetry analysis for Monza learning failures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from f1rl.config import LEGACY_DISCRETE_ACTIONS, MONZA_LENGTH_METERS
from f1rl.telemetry import REWARD_COMPONENT_KEYS, load_steps
from f1rl.track_sections import MONZA_SECTIONS, TrackSection, section_for_progress


def _progress(row: dict[str, Any]) -> float:
    return float(row.get("monotonic_progress_m", 0.0))


def _lap_progress(row: dict[str, Any]) -> float:
    return _progress(row) % MONZA_LENGTH_METERS


def _speed(row: dict[str, Any]) -> float:
    return float(row.get("speed_kph", 0.0))


def _min_ray(row: dict[str, Any]) -> float | None:
    rays = row.get("ray_distances_m", [])
    if not rays:
        return None
    return min(float(value) for value in rays)


def _action_label(row: dict[str, Any]) -> str:
    action_name = row.get("action_name")
    if action_name:
        return str(action_name)
    action_id = int(row.get("action_id", 0) or 0)
    if action_id == -10:
        return "scripted_controls"
    if action_id < 0:
        return "continuous"
    if action_id < len(LEGACY_DISCRETE_ACTIONS):
        return LEGACY_DISCRETE_ACTIONS[action_id][0]
    return f"action_{action_id}"


def _control_label(row: dict[str, Any]) -> str:
    throttle = float(row.get("throttle", 0.0))
    brake = float(row.get("brake", 0.0))
    steer = float(row.get("steering", 0.0))
    drive = "brake" if brake > 0.15 else "throttle" if throttle > 0.35 else "coast"
    if steer < -0.25:
        turn = "left"
    elif steer > 0.25:
        turn = "right"
    else:
        turn = "straight"
    return f"{drive}_{turn}"


def _reward_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    for row in rows:
        components = row.get("reward_components", {})
        for key in REWARD_COMPONENT_KEYS:
            totals[key] += float(components.get(key, 0.0))
    return totals


def _histogram(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_sections(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for section in MONZA_SECTIONS:
        rows = [row for row in steps if section.start_m <= _lap_progress(row) < section.end_m]
        if not rows:
            summaries.append(
                {
                    "section": section.name,
                    "start_m": section.start_m,
                    "end_m": section.end_m,
                    "rows": 0,
                    "target_speed_kph": section.target_speed_kph,
                }
            )
            continue
        speeds = [_speed(row) for row in rows]
        throttles = [float(row.get("throttle", 0.0)) for row in rows]
        brakes = [float(row.get("brake", 0.0)) for row in rows]
        lateral_errors = [abs(float(row.get("lateral_error_m", row.get("racing_line_deviation_m", 0.0)))) for row in rows]
        heading_errors = [abs(float(row.get("heading_error_deg", 0.0))) for row in rows]
        ray_values = [value for row in rows if (value := _min_ray(row)) is not None]
        brake_rows = [row for row in rows if float(row.get("brake", 0.0)) > 0.1]
        last_brake_index = max(
            (idx for idx, row in enumerate(rows) if float(row.get("brake", 0.0)) > 0.1),
            default=None,
        )
        throttle_reapply_progress_m = None
        if last_brake_index is not None:
            for row in rows[last_brake_index + 1 :]:
                if float(row.get("throttle", 0.0)) > 0.35:
                    throttle_reapply_progress_m = _progress(row)
                    break
        terminal_rows = [row for row in rows if row.get("termination_reason") not in {None, "active"}]
        summaries.append(
            {
                "section": section.name,
                "start_m": section.start_m,
                "end_m": section.end_m,
                "rows": len(rows),
                "target_speed_kph": section.target_speed_kph,
                "entry_progress_m": _progress(rows[0]),
                "exit_progress_m": _progress(rows[-1]),
                "entry_speed_kph": speeds[0],
                "exit_speed_kph": speeds[-1],
                "min_speed_kph": min(speeds),
                "max_speed_kph": max(speeds),
                "avg_speed_kph": _mean(speeds),
                "brake_start_progress_m": _progress(brake_rows[0]) if brake_rows else None,
                "throttle_reapplication_progress_m": throttle_reapply_progress_m,
                "avg_throttle": _mean(throttles),
                "avg_brake": _mean(brakes),
                "action_histogram": _histogram([_action_label(row) for row in rows]),
                "control_histogram": _histogram([_control_label(row) for row in rows]),
                "max_speed_surplus_kph": max(speed - section.target_speed_kph for speed in speeds),
                "min_ray_distance_m": min(ray_values) if ray_values else None,
                "avg_abs_lateral_error_m": _mean(lateral_errors),
                "max_abs_lateral_error_m": max(lateral_errors) if lateral_errors else None,
                "avg_abs_heading_error_deg": _mean(heading_errors),
                "max_abs_heading_error_deg": max(heading_errors) if heading_errors else None,
                "reward_totals": _reward_totals(rows),
                "termination_reason": terminal_rows[-1].get("termination_reason") if terminal_rows else None,
            }
        )
    return summaries


def _event_from_row(kind: str, reason: str, row: dict[str, Any], section: TrackSection) -> dict[str, Any]:
    min_ray = _min_ray(row)
    return {
        "kind": kind,
        "reason": reason,
        "section": section.name,
        "progress_m": _progress(row),
        "lap_progress_m": _lap_progress(row),
        "speed_kph": _speed(row),
        "target_speed_kph": section.target_speed_kph,
        "speed_surplus_kph": _speed(row) - section.target_speed_kph,
        "throttle": float(row.get("throttle", 0.0)),
        "brake": float(row.get("brake", 0.0)),
        "steering": float(row.get("steering", 0.0)),
        "lateral_error_m": float(row.get("lateral_error_m", row.get("racing_line_deviation_m", 0.0))),
        "heading_error_deg": float(row.get("heading_error_deg", 0.0)),
        "min_ray_distance_m": min_ray,
        "action": _action_label(row),
        "control": _control_label(row),
        "step_index": int(row.get("step_index", 0) or 0),
        "sim_time_s": float(row.get("sim_time_s", 0.0)),
    }


def detect_first_bad_event(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    braked_in_section: dict[str, bool] = {section.name: False for section in MONZA_SECTIONS}
    for row in steps:
        section = section_for_progress(_progress(row))
        lap_progress_m = _lap_progress(row)
        speed = _speed(row)
        throttle = float(row.get("throttle", 0.0))
        brake = float(row.get("brake", 0.0))
        heading_error = abs(float(row.get("heading_error_deg", 0.0)))
        lateral_error = abs(float(row.get("lateral_error_m", row.get("racing_line_deviation_m", 0.0))))
        min_ray = _min_ray(row)
        speed_surplus = speed - section.target_speed_kph
        in_brake_zone = (
            section.brake_start_m is not None
            and lap_progress_m >= section.brake_start_m
            and lap_progress_m <= (section.exit_m or section.end_m)
        )
        if brake > 0.1:
            braked_in_section[section.name] = True
        if in_brake_zone and speed_surplus > 45.0 and throttle > 0.45 and brake < 0.1:
            return _event_from_row(
                "throttle_during_brake_demand",
                "car is overspeed in a braking zone while still applying throttle",
                row,
                section,
            )
        if (
            section.turn_in_m is not None
            and lap_progress_m >= section.turn_in_m
            and speed_surplus > 35.0
            and not braked_in_section[section.name]
        ):
            return _event_from_row(
                "no_brake_before_turn_in",
                "car reached turn-in overspeed without a meaningful prior brake input",
                row,
                section,
            )
        if in_brake_zone and speed_surplus > 75.0:
            return _event_from_row(
                "overspeed_at_braking_zone",
                "car is far above the section target speed in the braking zone",
                row,
                section,
            )
        if min_ray is not None and min_ray < 1.0 and speed > 40.0:
            return _event_from_row("boundary_contact_risk", "minimum ray distance is below 1m at speed", row, section)
        if lateral_error > 22.0:
            return _event_from_row("excessive_lateral_error", "car is too far from the centerline/racing corridor", row, section)
        if heading_error > 55.0 and speed > 60.0:
            return _event_from_row("wrong_heading", "heading error is excessive at speed", row, section)
        reason = str(row.get("termination_reason", "active"))
        if reason in {"collision", "off_track", "no_progress"}:
            return _event_from_row(reason, f"episode terminated with {reason}", row, section)
    return None


def terminal_event(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not steps:
        return None
    final = steps[-1]
    reason = str(final.get("termination_reason", "active"))
    if reason == "active":
        return None
    return _event_from_row(reason, f"episode ended with {reason}", final, section_for_progress(_progress(final)))


def action_histogram_before_event(steps: list[dict[str, Any]], event: dict[str, Any] | None, *, window: int = 180) -> dict[str, int]:
    if not steps:
        return {}
    if event is None:
        selected = steps[-window:]
    else:
        event_step = int(event.get("step_index", steps[-1].get("step_index", 0)) or 0)
        before = [row for row in steps if int(row.get("step_index", 0) or 0) <= event_step]
        selected = before[-window:]
    return _histogram([_action_label(row) for row in selected])


def control_histogram_before_event(steps: list[dict[str, Any]], event: dict[str, Any] | None, *, window: int = 180) -> dict[str, int]:
    if not steps:
        return {}
    if event is None:
        selected = steps[-window:]
    else:
        event_step = int(event.get("step_index", steps[-1].get("step_index", 0)) or 0)
        before = [row for row in steps if int(row.get("step_index", 0) or 0) <= event_step]
        selected = before[-window:]
    return _histogram([_control_label(row) for row in selected])


def failure_report(steps: list[dict[str, Any]]) -> dict[str, Any]:
    first_bad = detect_first_bad_event(steps)
    terminal = terminal_event(steps)
    event = first_bad or terminal
    failed_section = event["section"] if event else None
    return {
        "failed_section": failed_section,
        "first_bad_event": first_bad,
        "terminal_event": terminal,
        "action_histogram_before_failure": action_histogram_before_event(steps, event),
        "control_histogram_before_failure": control_histogram_before_event(steps, event),
    }


def analyze_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "section_summaries": summarize_sections(steps),
        "failure_report": failure_report(steps),
    }


def analyze_telemetry_file(path: Path) -> dict[str, Any]:
    return analyze_steps(load_steps(path))
