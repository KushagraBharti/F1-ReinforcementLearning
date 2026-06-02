"""Quality-control report generation for simulator, telemetry, and manual review."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

from f1rl.calibration import calibration_report
from f1rl.config import ARTIFACTS_DIR
from f1rl.scripted import ScriptedController
from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS, load_steps


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _reward_totals(steps: list[dict[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    for step in steps:
        components = step.get("reward_components", {})
        for key in REWARD_COMPONENT_KEYS:
            totals[key] += float(components.get(key, 0.0))
    return totals


def latest_telemetry_path(root: Path = ARTIFACTS_DIR) -> Path | None:
    candidates = [path for path in root.glob("**/steps.jsonl") if path.is_file()]
    candidates.extend(path for path in root.glob("**/selected_telemetry/*-steps.jsonl") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
    return {
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
    }


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


def scripted_qc(*, steps: int, seed: int) -> dict[str, Any]:
    sim = MonzaSim()
    sim.config.max_steps = steps
    sim.reset(seed=seed)
    controller = ScriptedController()
    last = None
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
        last = result.telemetry
        if result.terminated or result.truncated:
            break
    return {
        "steps": sim.state.elapsed_steps,
        "elapsed_time_s": sim.state.elapsed_steps * sim.config.car.dt,
        "termination_reason": sim.termination_reason,
        "completed_lap": sim.completed_lap,
        "valid_lap": bool(last.valid_lap) if last is not None else False,
        "finish_crossed": bool(last.finish_crossed) if last is not None else False,
        "progress_m": sim.state.monotonic_progress_m,
        "checkpoints_passed": sim.checkpoints_passed,
        "missed_checkpoint_count": sim.missed_checkpoint_count,
    }


def physics_qc() -> dict[str, Any]:
    report = calibration_report()
    targets = report["targets"]
    estimates = report["sim_estimates"]
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
    summary_json = html.escape(json.dumps(telemetry_summary or {}, indent=2))
    source = html.escape(str(telemetry_path))
    body = "\n".join(charts)
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
  </style>
</head>
<body>
  <h1>F1RL Telemetry QC</h1>
  <p>Source: <code>{source}</code></p>
  <h2>Summary</h2>
  <pre>{summary_json}</pre>
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

## Manual Driving
- `uv run f1-manual`
- Confirm the car points forward at spawn.
- Confirm full throttle visually matches the HUD speed.
- Confirm braking, steering, and coast feel plausible.
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


def run_qc(
    *,
    output_root: Path,
    seed: int,
    telemetry_path: Path | None,
    run_scripted: bool,
    scripted_steps: int,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = output_root / f"qc-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    resolved_telemetry = telemetry_path or latest_telemetry_path()
    telemetry_summary = summarize_steps(resolved_telemetry) if resolved_telemetry is not None else None
    report: dict[str, Any] = {
        "run_id": root.name,
        "seed": seed,
        "track": track_qc(),
        "lap_validity": lap_validity_qc(),
        "physics": physics_qc(),
        "telemetry": telemetry_summary,
        "scripted": scripted_qc(steps=scripted_steps, seed=seed) if run_scripted else None,
    }
    (root / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dashboard = write_dashboard(root, resolved_telemetry, telemetry_summary)
    checklist = write_manual_checklist(root)
    summary_lines = [
        "# F1RL QC Report",
        "",
        f"- run: `{root.name}`",
        f"- telemetry: `{resolved_telemetry}`" if resolved_telemetry else "- telemetry: none found",
        f"- dashboard: `{dashboard}`" if dashboard else "- dashboard: not generated",
        f"- manual checklist: `{checklist}`",
        "",
        "## Key Checks",
        f"- checkpoint count: `{report['track']['checkpoint_count']}`",
        f"- boundary segments: `{report['track']['boundary_segment_count']}`",
        f"- observation/action: `{report['track']['observation_dim']}` / `{report['track']['action_dim']}`",
        f"- huge progress jump invalidates lap: `{report['lap_validity']['huge_progress_jump_invalidates_lap']}`",
        f"- wide checkpoint crossing invalidates lap: `{report['lap_validity']['wide_checkpoint_crossing_invalidates_lap']}`",
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
