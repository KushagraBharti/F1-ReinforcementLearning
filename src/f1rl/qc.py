"""Quality-control report generation for simulator, telemetry, and manual review."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from f1rl.calibration import calibration_report, reference_trace_features
from f1rl.config import ARTIFACTS_DIR, MONZA_LENGTH_METERS
from f1rl.scripted import ScriptedController
from f1rl.section_analysis import analyze_steps
from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS, load_steps


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: list[float], quantile: float) -> float | None:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return float(clean[lower] * (1.0 - fraction) + clean[upper] * fraction)


def _abs_optional_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(abs(float(value)))
    return values


def _reward_totals(steps: list[dict[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    for step in steps:
        components = step.get("reward_components", {})
        for key in REWARD_COMPONENT_KEYS:
            totals[key] += float(components.get(key, 0.0))
    return totals


def _reference_section_to_sim_progress(distance_m: float, reference_distance_m: float) -> float:
    return float(distance_m / max(reference_distance_m, 1e-9) * MONZA_LENGTH_METERS)


def sustained_corner_diagnostics(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps:
        return []
    trace = reference_trace_features()
    reference_distance_m = float(trace["distance_m"])
    diagnostics: list[dict[str, Any]] = []
    for target in trace["sustained_corner_targets"]:
        start_progress_m = _reference_section_to_sim_progress(float(target["start_distance_m"]), reference_distance_m)
        end_progress_m = _reference_section_to_sim_progress(float(target["end_distance_m"]), reference_distance_m)
        rows = [
            row
            for row in steps
            if start_progress_m <= float(row.get("monotonic_progress_m", 0.0)) % MONZA_LENGTH_METERS <= end_progress_m
        ]
        if not rows:
            diagnostics.append(
                {
                    "section": str(target["name"]),
                    "rows": 0,
                    "sim_start_progress_m": start_progress_m,
                    "sim_end_progress_m": end_progress_m,
                    "reference_start_distance_m": float(target["start_distance_m"]),
                    "reference_end_distance_m": float(target["end_distance_m"]),
                    "reference_mean_speed_kph": float(target["mean_speed_kph"]),
                    "reference_lateral_g_p90": float(target["lateral_g_p90"]),
                }
            )
            continue
        speeds = [float(row.get("speed_kph", 0.0)) for row in rows]
        ref_speeds = [float(row["reference_speed_kph"]) for row in rows if row.get("reference_speed_kph") is not None]
        speed_errors = [
            float(row.get("speed_kph", 0.0)) - float(row["reference_speed_kph"])
            for row in rows
            if row.get("reference_speed_kph") is not None
        ]
        lateral_errors = [
            abs(float(row.get("lateral_error_m", row.get("racing_line_deviation_m", 0.0)))) for row in rows
        ]
        heading_errors = _abs_optional_values(rows, "heading_error_deg")
        steering = _abs_optional_values(rows, "steering")
        lateral_g = _abs_optional_values(rows, "lateral_g")
        front_slip = _abs_optional_values(rows, "front_slip_angle_deg")
        rear_slip = _abs_optional_values(rows, "rear_slip_angle_deg")
        front_force = _abs_optional_values(rows, "front_lateral_force_n")
        rear_force = _abs_optional_values(rows, "rear_lateral_force_n")
        ghost_gaps = _abs_optional_values(rows, "ghost_gap_m")
        tire_saturation = _abs_optional_values(rows, "tire_saturation")
        throttles = [float(row.get("throttle", 0.0)) for row in rows]
        brakes = [float(row.get("brake", 0.0)) for row in rows]
        off_track_rate = _mean([1.0 if bool(row.get("off_track", False)) else 0.0 for row in rows])
        collided_count = sum(1 for row in rows if bool(row.get("collided", False)))
        steering_saturation_rate = _mean([1.0 if value >= 0.98 else 0.0 for value in steering])
        lateral_error_p95 = _quantile(lateral_errors, 0.95)
        steering_saturation_flag = steering_saturation_rate > 0.35
        lateral_error_flag = lateral_error_p95 is not None and lateral_error_p95 > 4.0
        off_track_flag = off_track_rate > 0.0 or collided_count > 0
        front_slip_p90 = _quantile(front_slip, 0.90)
        rear_slip_p90 = _quantile(rear_slip, 0.90)
        diagnostics.append(
            {
                "section": str(target["name"]),
                "rows": len(rows),
                "sim_start_progress_m": start_progress_m,
                "sim_end_progress_m": end_progress_m,
                "reference_start_distance_m": float(target["start_distance_m"]),
                "reference_end_distance_m": float(target["end_distance_m"]),
                "reference_mean_speed_kph": float(target["mean_speed_kph"]),
                "reference_min_speed_kph": float(target["min_speed_kph"]),
                "reference_max_speed_kph": float(target["max_speed_kph"]),
                "reference_lateral_g_p75": float(target["lateral_g_p75"]),
                "reference_lateral_g_p90": float(target["lateral_g_p90"]),
                "entry_progress_m": float(rows[0].get("monotonic_progress_m", 0.0)),
                "exit_progress_m": float(rows[-1].get("monotonic_progress_m", 0.0)),
                "mean_speed_kph": _mean(speeds),
                "min_speed_kph": min(speeds),
                "max_speed_kph": max(speeds),
                "mean_reference_speed_kph": _mean(ref_speeds) if ref_speeds else None,
                "mean_speed_error_kph": _mean(speed_errors) if speed_errors else None,
                "p95_abs_speed_error_kph": _quantile([abs(value) for value in speed_errors], 0.95),
                "p95_abs_ghost_gap_m": _quantile(ghost_gaps, 0.95),
                "p95_abs_lateral_error_m": lateral_error_p95,
                "max_abs_lateral_error_m": max(lateral_errors) if lateral_errors else None,
                "p95_abs_heading_error_deg": _quantile(heading_errors, 0.95),
                "mean_abs_steering": _mean(steering),
                "steering_saturation_rate": steering_saturation_rate,
                "lateral_g_p90": _quantile(lateral_g, 0.90),
                "lateral_g_max": max(lateral_g) if lateral_g else None,
                "front_slip_angle_p90_deg": front_slip_p90,
                "rear_slip_angle_p90_deg": rear_slip_p90,
                "front_minus_rear_slip_p90_deg": (
                    front_slip_p90 - rear_slip_p90
                    if front_slip_p90 is not None and rear_slip_p90 is not None
                    else None
                ),
                "front_lateral_force_p90_n": _quantile(front_force, 0.90),
                "rear_lateral_force_p90_n": _quantile(rear_force, 0.90),
                "tire_saturation_p90": _quantile(tire_saturation, 0.90),
                "mean_throttle": _mean(throttles),
                "mean_brake": _mean(brakes),
                "off_track_rate": off_track_rate,
                "collided_count": collided_count,
                "manual_review_flags": {
                    "steering_saturation": steering_saturation_flag,
                    "lateral_error": lateral_error_flag,
                    "off_track_or_collision": off_track_flag,
                },
                "manual_review_pass": not (steering_saturation_flag or lateral_error_flag or off_track_flag),
            }
        )
    return diagnostics


def latest_telemetry_path(root: Path = ARTIFACTS_DIR) -> Path | None:
    candidates = [path for path in root.glob("**/steps.jsonl") if path.is_file()]
    candidates.extend(path for path in root.glob("**/steps.jsonl.gz") if path.is_file())
    candidates.extend(path for path in root.glob("**/selected_telemetry/*-steps.jsonl") if path.is_file())
    candidates.extend(path for path in root.glob("**/selected_telemetry/*-steps.jsonl.gz") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def telemetry_paths_from_input(path: Path | None, *, max_files: int = 20) -> list[Path]:
    if path is None:
        latest = latest_telemetry_path()
        return [latest] if latest is not None else []
    if path.is_file():
        return [path]
    if path.is_dir():
        candidates = sorted(
            candidate for candidate in path.glob("**/*-steps.jsonl") if candidate.is_file()
        )
        candidates.extend(
            sorted(candidate for candidate in path.glob("**/*-steps.jsonl.gz") if candidate.is_file())
        )
        if not candidates and (path / "steps.jsonl").is_file():
            candidates = [path / "steps.jsonl"]
        if not candidates and (path / "steps.jsonl.gz").is_file():
            candidates = [path / "steps.jsonl.gz"]
        return candidates[:max_files]
    raise FileNotFoundError(f"Telemetry path does not exist: {path}")


def summarize_steps(path: Path) -> dict[str, Any]:
    steps = load_steps(path)
    if not steps:
        return {"path": str(path), "steps": 0}
    final = steps[-1]
    speeds = [float(step.get("speed_kph", 0.0)) for step in steps]
    progress = [float(step.get("monotonic_progress_m", 0.0)) for step in steps]
    lateral_errors = [abs(float(step.get("racing_line_deviation_m", step.get("lateral_error_m", 0.0)))) for step in steps]
    lateral_g = [abs(float(step.get("lateral_g", 0.0))) for step in steps]
    longitudinal_g = [float(step.get("longitudinal_g", 0.0)) for step in steps]
    ray_mins = [
        min(float(value) for value in step.get("ray_distances_m", []))
        for step in steps
        if step.get("ray_distances_m")
    ]
    summary = {
        "path": str(path),
        "steps": len(steps),
        "duration_s": float(final.get("sim_time_s", 0.0)),
        "termination_reason": final.get("termination_reason", "unknown"),
        "completed_lap": bool(final.get("termination_reason") == "lap_complete"),
        "valid_lap": bool(final.get("valid_lap", False)),
        "finish_crossed": bool(final.get("finish_crossed", False)),
        "segment_complete": bool(final.get("segment_complete", False)),
        "final_progress_m": float(final.get("monotonic_progress_m", 0.0)),
        "best_progress_m": max(progress),
        "checkpoints_passed": int(final.get("checkpoints_passed", final.get("checkpoint_index", 0))),
        "missed_checkpoint_count": int(final.get("missed_checkpoint_count", 0)),
        "avg_speed_kph": _mean(speeds),
        "max_speed_kph": max(speeds),
        "avg_abs_racing_line_deviation_m": _mean(lateral_errors),
        "max_abs_racing_line_deviation_m": max(lateral_errors),
        "avg_lateral_g": _mean(lateral_g),
        "max_lateral_g": max(lateral_g),
        "avg_longitudinal_g": _mean(longitudinal_g),
        "min_longitudinal_g": min(longitudinal_g),
        "min_ray_distance_m": min(ray_mins) if ray_mins else None,
        "avg_min_ray_distance_m": _mean(ray_mins) if ray_mins else None,
        "reward_total": sum(float(step.get("reward_total", 0.0)) for step in steps),
        "reward_totals": _reward_totals(steps),
        "sustained_corner_diagnostics": sustained_corner_diagnostics(steps),
    }
    summary.update(analyze_steps(steps))
    return summary


def track_qc() -> dict[str, Any]:
    sim = MonzaSim()
    obs, info = sim.reset(seed=7)
    return {
        "track_name": sim.track.name,
        "source_image_size": sim.track.source_image_size,
        "track_length_m": sim.track.length_m,
        "track_length_px": sim.track.length_px,
        "meters_per_pixel": sim.track.meters_per_pixel,
        "checkpoint_count": len(sim.track.checkpoints),
        "boundary_segment_count": int(sim.boundary_segments.shape[0]),
        "sensor_ray_count": len(sim.sensor_angles),
        "observation_dim": int(obs.shape[0]),
        "action_dim": sim.action_dim,
        "initial_valid_lap": bool(info["valid_lap"]),
        "checkpoint_lateral_limit_m": sim.config.checkpoint_lateral_limit_m,
    }


def lap_validity_qc() -> dict[str, Any]:
    sim = MonzaSim()
    sim.reset(seed=7)
    spacing = sim.track.length_m / max(len(sim.track.checkpoints), 1)

    skip_sim = MonzaSim()
    skip_sim.reset(seed=7)
    skip_sim.state.monotonic_progress_m = skip_sim.track.length_m
    skip_sim._update_checkpoint_validity(0.0, skip_sim.track.length_m, 0.0)

    lateral_sim = MonzaSim()
    lateral_sim.reset(seed=7)
    lateral_sim.state.monotonic_progress_m = spacing * 1.1
    lateral_sim._update_checkpoint_validity(
        0.0,
        spacing * 1.1,
        lateral_sim.config.checkpoint_lateral_limit_m + 1.0,
    )

    return {
        "checkpoint_spacing_m": spacing,
        "ordered_checkpoint_count": len(sim.track.checkpoints),
        "finish_requires_all_checkpoints": True,
        "finish_requires_physical_or_virtual_crossing": True,
        "huge_progress_jump_invalidates_lap": not skip_sim.valid_lap
        and skip_sim.missed_checkpoint_count > 0,
        "wide_checkpoint_crossing_invalidates_lap": not lateral_sim.valid_lap
        and lateral_sim.missed_checkpoint_count > 0,
        "skip_missed_checkpoint_count": skip_sim.missed_checkpoint_count,
        "wide_crossing_missed_checkpoint_count": lateral_sim.missed_checkpoint_count,
    }


def scripted_rollout_steps(*, steps: int, seed: int) -> list[dict[str, Any]]:
    sim = MonzaSim()
    sim.config.max_steps = steps
    sim.reset(seed=seed)
    controller = ScriptedController()
    rows: list[dict[str, Any]] = []
    for _ in range(steps):
        throttle, brake, steer = controller.controls(sim)
        result = sim.step_controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            action_id=-10,
            collect_observation=False,
            collect_rays=False,
        )
        rows.append(asdict(result.telemetry))
        if result.terminated or result.truncated:
            break
    return rows


def scripted_qc(*, steps: int, seed: int) -> dict[str, Any]:
    rows = scripted_rollout_steps(steps=steps, seed=seed)
    if not rows:
        return {"steps": 0, "termination_reason": "empty", "completed_lap": False, "valid_lap": False}
    final = rows[-1]
    analysis = analyze_steps(rows)
    return {
        "steps": len(rows),
        "elapsed_time_s": float(final.get("sim_time_s", 0.0)),
        "termination_reason": final.get("termination_reason", "unknown"),
        "completed_lap": final.get("termination_reason") == "lap_complete",
        "valid_lap": bool(final.get("valid_lap", False)),
        "finish_crossed": bool(final.get("finish_crossed", False)),
        "progress_m": float(final.get("monotonic_progress_m", 0.0)),
        "checkpoints_passed": int(final.get("checkpoints_passed", 0)),
        "missed_checkpoint_count": int(final.get("missed_checkpoint_count", 0)),
        "section_summaries": analysis["section_summaries"],
        "failure_report": analysis["failure_report"],
    }


def physics_qc() -> dict[str, Any]:
    report = calibration_report()
    targets = report["targets"]
    estimates = report["sim_estimates"]
    v2 = report["physics_models"]["v2"]
    v2_errors = v2["error_terms"]
    return {
        "reference_source": targets["source"],
        "reference_lap_time_s": targets["lap_time_s"],
        "reference_max_speed_kph": targets["max_speed_kph"],
        "reference_mean_speed_kph": targets["mean_speed_kph"],
        "reference_lateral_g_p95": targets["lateral_g_p95"],
        "reference_radius_p05_m": targets["radius_p05_m"],
        "sim_terminal_speed_kph": estimates["terminal_speed_kph"],
        "sim_speed_after_5s_full_throttle_kph": estimates["speed_after_5s_full_throttle_kph"],
        "sim_speed_after_8s_full_throttle_kph": estimates["speed_after_8s_full_throttle_kph"],
        "sim_braking_330_to_150_kph_m": estimates["braking_330_to_150_kph_m"],
        "sim_braking_330_to_100_kph_m": estimates["braking_330_to_100_kph_m"],
        "sim_steering_limited_radius_m": estimates["steering_limited_radius_m"],
        "sim_cornering_capacity": estimates["cornering_capacity"],
        "v2_physics_version": v2["physics_version"],
        "v2_physics_calibration_id": v2["physics_calibration_id"],
        "v2_max_speed_error_kph": v2_errors["max_speed_error_kph"],
        "v2_speed_trace_accel_p95_mae_mps2": v2_errors["speed_trace_accel_p95_mae_mps2"],
        "v2_mean_abs_braking_zone_distance_error_m": v2_errors["mean_abs_braking_zone_distance_error_m"],
        "v2_min_robust_corner_lateral_g_margin": v2_errors["min_robust_corner_lateral_g_margin"],
        "v2_sustained_corner_reference_control_pass_rate": v2_errors[
            "sustained_corner_reference_control_pass_rate"
        ],
        "v2_max_sustained_corner_reference_p95_abs_lateral_error_m": v2_errors[
            "max_sustained_corner_reference_p95_abs_lateral_error_m"
        ],
    }


def _values(steps: list[dict[str, Any]], key: str) -> list[float]:
    return [float(step.get(key, 0.0)) for step in steps]


def _min_ray_values(steps: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for step in steps:
        rays = step.get("ray_distances_m", [])
        values.append(min(float(value) for value in rays) if rays else 0.0)
    return values


def _svg_line(values: list[float], *, title: str, width: int = 760, height: int = 180) -> str:
    escaped_title = html.escape(title)
    if not values:
        return f"<h3>{escaped_title}</h3><p>No data.</p>"
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1e-9)
    points: list[str] = []
    count = max(len(values) - 1, 1)
    for idx, value in enumerate(values):
        x = idx / count * width
        y = height - ((value - vmin) / span * height)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f"<h3>{escaped_title}</h3>"
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
        "role='img' aria-label='line chart'>"
        "<rect x='0' y='0' width='100%' height='100%' fill='#101418'/>"
        f"<polyline fill='none' stroke='#32d296' stroke-width='2' points='{' '.join(points)}'/>"
        f"<text x='8' y='18' fill='#dbe5ea' font-size='13'>min {vmin:.2f} max {vmax:.2f}</text>"
        "</svg>"
    )


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                rendered = f"{value:.2f}"
            elif isinstance(value, dict):
                rendered = json.dumps(value, sort_keys=True)
            else:
                rendered = "" if value is None else str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def compact_failure_row(summary: dict[str, Any]) -> dict[str, Any]:
    failure = summary.get("failure_report") or {}
    first_bad = failure.get("first_bad_event") or {}
    terminal = failure.get("terminal_event") or {}
    return {
        "path": summary.get("path"),
        "steps": summary.get("steps"),
        "best_progress_m": summary.get("best_progress_m"),
        "termination_reason": summary.get("termination_reason"),
        "failed_section": failure.get("failed_section"),
        "first_bad_kind": first_bad.get("kind"),
        "first_bad_progress_m": first_bad.get("progress_m"),
        "first_bad_speed_kph": first_bad.get("speed_kph"),
        "first_bad_reason": first_bad.get("reason"),
        "terminal_kind": terminal.get("kind"),
        "terminal_progress_m": terminal.get("progress_m"),
        "terminal_speed_kph": terminal.get("speed_kph"),
        "actions_before_failure": failure.get("action_histogram_before_failure"),
        "controls_before_failure": failure.get("control_histogram_before_failure"),
    }


def _section_table_rows(section_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in section_summaries:
        if int(summary.get("rows", 0) or 0) == 0:
            continue
        rows.append(
            {
                "section": summary.get("section"),
                "rows": summary.get("rows"),
                "entry_speed_kph": summary.get("entry_speed_kph"),
                "min_speed_kph": summary.get("min_speed_kph"),
                "exit_speed_kph": summary.get("exit_speed_kph"),
                "max_speed_surplus_kph": summary.get("max_speed_surplus_kph"),
                "brake_start_progress_m": summary.get("brake_start_progress_m"),
                "avg_throttle": summary.get("avg_throttle"),
                "avg_brake": summary.get("avg_brake"),
                "min_ray_distance_m": summary.get("min_ray_distance_m"),
                "max_abs_lateral_error_m": summary.get("max_abs_lateral_error_m"),
                "termination_reason": summary.get("termination_reason"),
            }
        )
    return rows


def write_dashboard(root: Path, telemetry_path: Path | None, telemetry_summary: dict[str, Any] | None) -> Path | None:
    if telemetry_path is None:
        return None
    steps = load_steps(telemetry_path)
    if not steps:
        return None
    charts = [
        _svg_line(_values(steps, "speed_kph"), title="Speed kph"),
        _svg_line(_values(steps, "monotonic_progress_m"), title="Progress m"),
        _svg_line(_values(steps, "reward_total"), title="Reward per step"),
        _svg_line([abs(v) for v in _values(steps, "racing_line_deviation_m")], title="Abs racing-line deviation m"),
        _svg_line(_min_ray_values(steps), title="Minimum ray distance m"),
    ]
    section_rows = _section_table_rows(telemetry_summary.get("section_summaries", []) if telemetry_summary else [])
    failure_rows = [compact_failure_row(telemetry_summary)] if telemetry_summary else []
    summary_json = html.escape(json.dumps(telemetry_summary or {}, indent=2))
    source = html.escape(str(telemetry_path))
    body = "\n".join(charts)
    section_table = _html_table(
        section_rows,
        [
            ("section", "Section"),
            ("rows", "Rows"),
            ("entry_speed_kph", "Entry kph"),
            ("min_speed_kph", "Min kph"),
            ("exit_speed_kph", "Exit kph"),
            ("max_speed_surplus_kph", "Max surplus"),
            ("brake_start_progress_m", "Brake start m"),
            ("avg_throttle", "Avg throttle"),
            ("avg_brake", "Avg brake"),
            ("min_ray_distance_m", "Min ray m"),
            ("termination_reason", "Termination"),
        ],
    )
    failure_table = _html_table(
        failure_rows,
        [
            ("failed_section", "Failed section"),
            ("first_bad_kind", "First bad event"),
            ("first_bad_progress_m", "Bad progress m"),
            ("first_bad_speed_kph", "Bad speed kph"),
            ("first_bad_reason", "Reason"),
            ("terminal_kind", "Terminal"),
            ("terminal_progress_m", "Terminal progress m"),
        ],
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>F1RL Telemetry QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f7f8; color: #182026; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    pre {{ background: #101418; color: #dbe5ea; padding: 16px; overflow: auto; }}
    svg {{ display: block; max-width: 100%; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #d9e1e5; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eaf0f3; }}
  </style>
</head>
<body>
  <h1>F1RL Telemetry QC</h1>
  <p>Source: <code>{source}</code></p>
  <h2>Summary</h2>
  <pre>{summary_json}</pre>
  <h2>Failure Table</h2>
  {failure_table}
  <h2>Section Summary</h2>
  {section_table}
  <h2>Charts</h2>
  {body}
</body>
</html>
"""
    path = root / "telemetry_dashboard.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def write_manual_checklist(root: Path) -> Path:
    text = """# Manual QC Checklist

Run these checks before starting the next long RL goal.

## Physics V2 Manual Handoff
- `uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start`
- `uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120`
- `uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_02 --start-section-lead-in-m 120`
- `uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_03 --start-section-lead-in-m 120`
- Confirm the HUD reports `physics_model=v2` behavior through `physics_v2.0.10-fastf1-manual-balance-fix` and calibration id `monza_2022_2024_fastf1_multilap_v2_manual_balance_fix`.
- Confirm manual left/right steering now matches the visible car response: pressing left should visually turn the car left and pressing right should visually turn the car right.
- Confirm the HUD is right-aligned in free screen space and no longer covers the left-side driving line at the manual start.
- Confirm the FastF1 reference ghost appears immediately, the ghost speed is visible, and the ghost gap changes for driving reasons rather than timer drift.
- Check the run down to Rettifilo: top-end speed should feel close to the 2024 VER Monza reference, with the ghost reaching about `348 kph` before braking.
- Check the first major braking zone: heavy braking should be strong but not instant, and the car should still punish late turn-in or full-brake steering.
- Check medium/high-speed corners such as Lesmo/Ascari/Parabolica qualitatively against the ghost; do not require exact human lap time, but reject obvious arcade grip, impossible rotation, or uncatchable instability.
- For sustained-corner section runs, verify the car can hold the intended arc without obvious steering saturation or front-end washout.
- After exiting manual mode, inspect the latest `artifacts\\runs\\manual-*\\episode_summary.json` and confirm `physics_model=v2`, `physics_version=physics_v2.0.10-fastf1-manual-balance-fix`, calibration id `monza_2022_2024_fastf1_multilap_v2_manual_balance_fix`, and non-null ghost-gap fields when `--ghost-reference` was used.
- Run `uv run --no-sync python -m f1rl.qc --telemetry <manual-run-dir>` and inspect `sustained_corner_diagnostics` in `qc_report.json` for lateral error, steering saturation, slip angles, front/rear lateral force, lateral-g, and off-track/collision flags.
- Do not establish `scripted_threshold`, run scaled V2 ES/RL, export V2 datasets, or train V2 BC/SAC until this manual handoff is approved.

## Manual Driving
- `uv run f1-manual`
- Confirm the car points forward at spawn.
- Confirm full throttle visually matches the HUD speed.
- Confirm braking, steering, and coast feel plausible.
- Confirm the corrected left/right manual steering mapping remains intuitive with arrow keys and A/D keys.
- Confirm rays render from the car nose and rotate with the car.
- Confirm the HUD shows speed, progress, checkpoint/lap state, reward, and reason fields clearly enough for debugging.

## Ghost Overlay
- `uv run f1-manual --ghost-reference`
- Confirm the manual timer advances in real time.
- Confirm the reference ghost timer advances in the same time base as replay.
- Confirm the ghost is visibly faster for legitimate speed/line reasons, not renderer timing drift.

## Replay
- `uv run f1-replay artifacts\\reference-ghost-20260602-093941\\steps.jsonl`
- Confirm replay timing matches real seconds unless `--speed` or `--no-timing` is used.
- Confirm rays, car orientation, and track scale match manual mode.

## Lap Validity
- Drive off track intentionally and confirm immediate termination.
- Drive a clean partial lap and confirm checkpoint count increases in order.
- Confirm benchmark/QC JSON reports missed checkpoints and invalid laps when they occur.

## Physics Feel
- Compare manual top speed, braking distance, and corner speeds against the Fast-F1 reference ghost qualitatively.
- Do not expect exact F1 lap time from manual control; use the reference ghost as calibration and the scripted driver as a physically controlled sanity baseline.
"""
    path = root / "manual_qc_checklist.md"
    path.write_text(text, encoding="utf-8")
    return path


def manual_gate_readiness(
    *,
    physics_report: dict[str, Any],
    telemetry_summary: dict[str, Any] | None,
    manual_checklist: Path,
) -> dict[str, Any]:
    sustained_rows = []
    if telemetry_summary is not None:
        sustained_rows = [
            row
            for row in telemetry_summary.get("sustained_corner_diagnostics", [])
            if int(row.get("rows", 0) or 0) > 0
        ]
    observed_pass = None
    if sustained_rows:
        observed_pass = all(bool(row.get("manual_review_pass", False)) for row in sustained_rows)
    return {
        "gate": "physics_v2_manual_handoff",
        "status": "awaiting_user_manual_approval",
        "manual_approval_recorded": False,
        "scripted_threshold_s": None,
        "scripted_threshold_status": "unset",
        "physics_model": "v2",
        "physics_version": physics_report["v2_physics_version"],
        "physics_calibration_id": physics_report["v2_physics_calibration_id"],
        "calibration_summary": {
            "max_speed_error_kph": physics_report["v2_max_speed_error_kph"],
            "speed_trace_accel_p95_mae_mps2": physics_report["v2_speed_trace_accel_p95_mae_mps2"],
            "mean_abs_braking_zone_distance_error_m": physics_report[
                "v2_mean_abs_braking_zone_distance_error_m"
            ],
            "min_robust_corner_lateral_g_margin": physics_report[
                "v2_min_robust_corner_lateral_g_margin"
            ],
            "sustained_corner_reference_control_pass_rate": physics_report[
                "v2_sustained_corner_reference_control_pass_rate"
            ],
            "max_sustained_corner_reference_p95_abs_lateral_error_m": physics_report[
                "v2_max_sustained_corner_reference_p95_abs_lateral_error_m"
            ],
        },
        "manual_telemetry_path": telemetry_summary.get("path") if telemetry_summary is not None else None,
        "observed_sustained_sections": [
            {
                "section": row["section"],
                "rows": row["rows"],
                "manual_review_pass": row["manual_review_pass"],
                "p95_abs_lateral_error_m": row["p95_abs_lateral_error_m"],
                "steering_saturation_rate": row["steering_saturation_rate"],
                "off_track_rate": row["off_track_rate"],
            }
            for row in sustained_rows
        ],
        "observed_sustained_sections_pass": observed_pass,
        "manual_checklist_path": str(manual_checklist),
        "required_manual_commands": [
            "uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start",
            (
                "uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference "
                "--start-section sustained_corner_01 --start-section-lead-in-m 120"
            ),
            (
                "uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference "
                "--start-section sustained_corner_02 --start-section-lead-in-m 120"
            ),
            (
                "uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference "
                "--start-section sustained_corner_03 --start-section-lead-in-m 120"
            ),
        ],
        "blocked_until_manual_approval": [
            "scripted_threshold",
            "scaled_v2_gpu_es",
            "v2_dataset_export",
            "v2_bc",
            "v2_sac",
            "v2_highlights_and_gifs",
        ],
    }


def run_qc(
    *,
    output_root: Path,
    seed: int,
    telemetry_path: Path | None,
    run_scripted: bool,
    scripted_steps: int,
    max_telemetry_files: int = 20,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = output_root / f"qc-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    resolved_telemetry_paths = telemetry_paths_from_input(telemetry_path, max_files=max_telemetry_files)
    telemetry_reports = [summarize_steps(path) for path in resolved_telemetry_paths]
    telemetry_summary = telemetry_reports[0] if telemetry_reports else None
    failure_table = [compact_failure_row(summary) for summary in telemetry_reports]
    scripted_report = scripted_qc(steps=scripted_steps, seed=seed) if run_scripted else None
    physics_report = physics_qc()
    dashboard = write_dashboard(root, resolved_telemetry_paths[0] if resolved_telemetry_paths else None, telemetry_summary)
    checklist = write_manual_checklist(root)
    manual_gate = manual_gate_readiness(
        physics_report=physics_report,
        telemetry_summary=telemetry_summary,
        manual_checklist=checklist,
    )
    report: dict[str, Any] = {
        "run_id": root.name,
        "seed": seed,
        "track": track_qc(),
        "lap_validity": lap_validity_qc(),
        "physics": physics_report,
        "telemetry": telemetry_summary,
        "telemetry_reports": telemetry_reports,
        "failure_table": failure_table,
        "scripted": scripted_report,
        "manual_gate": manual_gate,
    }
    (root / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_lines = [
        "# F1RL QC Report",
        "",
        f"- run: `{root.name}`",
        f"- telemetry: `{resolved_telemetry_paths[0]}`" if resolved_telemetry_paths else "- telemetry: none found",
        f"- telemetry files analyzed: `{len(telemetry_reports)}`",
        f"- dashboard: `{dashboard}`" if dashboard else "- dashboard: not generated",
        f"- manual checklist: `{checklist}`",
        "",
        "## Key Checks",
        f"- checkpoint count: `{report['track']['checkpoint_count']}`",
        f"- boundary segments: `{report['track']['boundary_segment_count']}`",
        f"- observation/action: `{report['track']['observation_dim']}` / `{report['track']['action_dim']}`",
        f"- huge progress jump invalidates lap: `{report['lap_validity']['huge_progress_jump_invalidates_lap']}`",
        f"- wide checkpoint crossing invalidates lap: `{report['lap_validity']['wide_checkpoint_crossing_invalidates_lap']}`",
        "",
        "## Physics V2 Manual Gate",
        f"- status: `{manual_gate['status']}`",
        f"- threshold: `{manual_gate['scripted_threshold_status']}`",
        f"- physics: `{manual_gate['physics_version']}`",
        f"- calibration: `{manual_gate['physics_calibration_id']}`",
        f"- observed sustained sections pass: `{manual_gate['observed_sustained_sections_pass']}`",
    ]
    if telemetry_summary is not None:
        summary_lines.extend(
            [
                "",
                "## Telemetry",
                f"- termination: `{telemetry_summary['termination_reason']}`",
                f"- best progress: `{telemetry_summary['best_progress_m']:.1f}m`",
                f"- valid lap: `{telemetry_summary['valid_lap']}`",
                f"- missed checkpoints: `{telemetry_summary['missed_checkpoint_count']}`",
                f"- max speed: `{telemetry_summary['max_speed_kph']:.1f}kph`",
            ]
        )
        failure = telemetry_summary.get("failure_report", {})
        first_bad = failure.get("first_bad_event")
        terminal = failure.get("terminal_event")
        summary_lines.extend(["", "## Failure Analysis"])
        if first_bad:
            summary_lines.extend(
                [
                    f"- failed section: `{failure.get('failed_section')}`",
                    f"- first bad event: `{first_bad.get('kind')}`",
                    f"- first bad progress: `{float(first_bad.get('progress_m', 0.0)):.1f}m`",
                    f"- first bad speed: `{float(first_bad.get('speed_kph', 0.0)):.1f}kph`",
                    f"- reason: `{first_bad.get('reason')}`",
                    f"- actions before failure: `{failure.get('action_histogram_before_failure')}`",
                ]
            )
        if terminal:
            summary_lines.extend(
                [
                    f"- terminal event: `{terminal.get('kind')}`",
                    f"- terminal progress: `{float(terminal.get('progress_m', 0.0)):.1f}m`",
                    f"- terminal speed: `{float(terminal.get('speed_kph', 0.0)):.1f}kph`",
                ]
            )
        summary_lines.extend(["", "## Section Summary"])
        for section in _section_table_rows(telemetry_summary.get("section_summaries", [])):
            summary_lines.append(
                "- "
                f"`{section['section']}`: entry `{float(section['entry_speed_kph']):.1f}kph`, "
                f"min `{float(section['min_speed_kph']):.1f}kph`, "
                f"exit `{float(section['exit_speed_kph']):.1f}kph`, "
                f"avg brake `{float(section['avg_brake'] or 0.0):.2f}`, "
                f"termination `{section['termination_reason']}`"
            )
    if scripted_report is not None:
        summary_lines.extend(
            [
                "",
                "## Scripted Comparison",
                f"- termination: `{scripted_report['termination_reason']}`",
                f"- progress: `{float(scripted_report['progress_m']):.1f}m`",
                f"- valid lap: `{scripted_report['valid_lap']}`",
            ]
        )
    (root / "qc_report.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"qc_complete run={root}")
    print(f"report={root / 'qc_report.json'}")
    if dashboard:
        print(f"dashboard={dashboard}")
    print(f"manual_checklist={checklist}")
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulator and telemetry QC artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--run-scripted", action="store_true")
    parser.add_argument("--scripted-steps", type=int, default=18000)
    parser.add_argument("--max-telemetry-files", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_qc(
        output_root=args.output_dir,
        seed=args.seed,
        telemetry_path=args.telemetry,
        run_scripted=args.run_scripted,
        scripted_steps=args.scripted_steps,
        max_telemetry_files=args.max_telemetry_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
