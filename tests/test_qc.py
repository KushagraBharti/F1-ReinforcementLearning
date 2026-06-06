import json
from pathlib import Path

from f1rl import qc
from f1rl.calibration import reference_trace_features
from f1rl.config import MONZA_LENGTH_METERS
from f1rl.sim import MonzaSim
from f1rl.telemetry import TelemetryWriter


def test_qc_writes_report_and_dashboard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(qc, "ARTIFACTS_DIR", tmp_path)
    sim = MonzaSim()
    sim.reset(seed=1)
    result = sim.step(1)
    writer = TelemetryWriter(tmp_path, mode="test-qc", seed=1, lap_length_m=sim.track.length_m)
    writer.write_step(result.telemetry)
    writer.close_episode(termination_reason=sim.termination_reason, completed_lap=sim.completed_lap)

    root = qc.run_qc(
        output_root=tmp_path,
        seed=1,
        telemetry_path=writer.steps_path,
        run_scripted=False,
        scripted_steps=10,
    )

    assert (root / "qc_report.json").exists()
    assert (root / "qc_report.md").exists()
    assert (root / "manual_qc_checklist.md").exists()
    assert (root / "telemetry_dashboard.html").exists()

    report = json.loads((root / "qc_report.json").read_text(encoding="utf-8"))
    assert report["track"]["checkpoint_count"] == len(sim.track.checkpoints)
    assert report["lap_validity"]["huge_progress_jump_invalidates_lap"] is True
    assert report["lap_validity"]["wide_checkpoint_crossing_invalidates_lap"] is True
    assert report["telemetry"]["steps"] == 1
    assert "section_summaries" in report["telemetry"]
    assert "failure_report" in report["telemetry"]
    assert len(report["telemetry_reports"]) == 1
    assert len(report["failure_table"]) == 1
    assert report["manual_gate"]["status"] == "awaiting_user_manual_approval"
    assert report["manual_gate"]["scripted_threshold_status"] == "unset"
    assert report["manual_gate"]["manual_approval_recorded"] is False
    assert "scripted_threshold" in report["manual_gate"]["blocked_until_manual_approval"]

    checklist = (root / "manual_qc_checklist.md").read_text(encoding="utf-8")
    assert "Physics V2 Manual Handoff" in checklist
    assert "physics_v2.0.10-fastf1-manual-balance-fix" in checklist
    assert "sustained_corner_03" in checklist
    assert "left/right steering now matches" in checklist
    assert "HUD is right-aligned" in checklist
    assert "Do not establish `scripted_threshold`" in checklist


def test_qc_reports_sustained_corner_manual_diagnostics(tmp_path: Path) -> None:
    trace = reference_trace_features()
    target = trace["sustained_corner_targets"][0]
    progress_m = float(target["start_distance_m"]) / float(trace["distance_m"]) * MONZA_LENGTH_METERS + 10.0
    steps_path = tmp_path / "steps.jsonl"
    row = {
        "step_index": 1,
        "sim_time_s": 0.016,
        "speed_kph": 216.0,
        "reference_speed_kph": 215.0,
        "ghost_gap_m": 0.4,
        "monotonic_progress_m": progress_m,
        "racing_line_deviation_m": 1.0,
        "lateral_error_m": 1.0,
        "heading_error_deg": 4.0,
        "steering": 0.42,
        "lateral_g": 5.2,
        "front_slip_angle_deg": 7.0,
        "rear_slip_angle_deg": 6.0,
        "front_lateral_force_n": 24000.0,
        "rear_lateral_force_n": 18000.0,
        "tire_saturation": 0.8,
        "throttle": 0.7,
        "brake": 0.0,
        "off_track": False,
        "collided": False,
        "reward_total": 0.0,
        "reward_components": {},
        "ray_distances_m": [30.0],
        "valid_lap": False,
        "finish_crossed": False,
        "segment_complete": False,
        "checkpoint_index": 1,
        "checkpoints_passed": 1,
        "missed_checkpoint_count": 0,
        "termination_reason": "active",
        "longitudinal_g": 0.0,
    }
    steps_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = qc.summarize_steps(steps_path)
    diagnostics = summary["sustained_corner_diagnostics"]
    first = next(item for item in diagnostics if item["section"] == target["name"])

    assert first["rows"] == 1
    assert first["manual_review_pass"] is True
    assert first["p95_abs_ghost_gap_m"] == 0.4
    assert first["front_minus_rear_slip_p90_deg"] == 1.0
    assert first["front_lateral_force_p90_n"] == 24000.0
