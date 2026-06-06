import json

from f1rl.calibration import (
    REFERENCE_CSV,
    calibration_report,
    load_targets,
    reference_summary_from_csv,
    reference_trace_features,
)
from f1rl.fastf1_calibration import (
    compare_reference,
    summarize_multi_reference,
)
from f1rl.fastf1_calibration import (
    main as fastf1_calibration_main,
)


def test_fastf1_reference_targets_available() -> None:
    targets = load_targets()
    assert 79.0 < targets.lap_time_s < 81.0
    assert 340.0 < targets.max_speed_kph < 355.0
    assert 250.0 < targets.mean_speed_kph < 270.0


def test_default_car_speed_is_near_monza_target() -> None:
    report = calibration_report()
    terminal = report["sim_estimates"]["terminal_speed_kph"]
    assert 340.0 <= terminal <= 360.0


def test_v2_calibration_report_has_error_terms() -> None:
    report = calibration_report()
    trace = report["reference_trace_features"]
    v2 = report["physics_models"]["v2"]
    assert v2["physics_model"] == "v2"
    assert v2["physics_calibration_id"]
    assert "terminal_speed_kph" in v2["estimates"]
    assert len(trace["speed_trace_samples"]) > 10
    assert len(trace["braking_zones"]) >= 5
    assert len(trace["corner_speed_targets"]) >= 5
    assert len(trace["sustained_corner_targets"]) >= 3
    assert len(trace["longitudinal_by_speed_bin"]) >= 5
    assert trace["acceleration"]["braking_decel_p95_mps2"] > 0.0
    assert trace["gear_rpm"]["max_gear"] == 8
    assert "max_speed_error_kph" in v2["error_terms"]
    assert "lateral_g_p95_proxy_error" in v2["error_terms"]
    assert "mean_abs_braking_zone_distance_error_m" in v2["error_terms"]
    assert "min_corner_lateral_g_margin" in v2["error_terms"]
    assert "min_robust_corner_lateral_g_margin" in v2["error_terms"]
    assert "speed_trace_accel_p95_mae_mps2" in v2["error_terms"]
    assert "speed_trace_feasibility" in v2["estimates"]
    assert "sustained_corner_diagnostic" in v2["estimates"]
    sustained = v2["estimates"]["sustained_corner_diagnostic"]
    assert sustained["rows"]
    assert sustained["reference_control_pass_rate"] >= 1.0 / 3.0
    assert sustained["max_reference_p95_abs_lateral_error_m"] < 20.0
    assert sustained["min_reference_lateral_g_p75_margin"] > -2.25
    assert sustained["min_reference_lateral_g_p90_margin"] > -3.0
    assert "sustained_corner_reference_control_pass_rate" in v2["error_terms"]
    assert v2["error_terms"]["gear_match_rate"] == 1.0
    assert v2["error_terms"]["mean_abs_rpm_error"] < 50.0


def test_reference_summary_from_csv_contains_rich_trace_features() -> None:
    summary = reference_summary_from_csv(REFERENCE_CSV)
    features = reference_trace_features(REFERENCE_CSV)

    assert summary["rows"] == features["rows"]
    assert summary["trace_features"]["braking_zones"]
    assert summary["trace_features"]["corner_speed_targets"]
    assert summary["trace_features"]["sustained_corner_targets"]
    assert summary["trace_features"]["gear_rpm"]["shift_points"]


def test_v2_sustained_corner_diagnostic_reports_handling_balance() -> None:
    report = calibration_report()
    diagnostic = report["physics_models"]["v2"]["estimates"]["sustained_corner_diagnostic"]
    rows = diagnostic["rows"]
    reference_runs = [row["reference_run"] for row in rows]

    assert any(run["control_pass"] for run in reference_runs)
    assert max(run["steering_saturation_rate"] for run in reference_runs) < 0.75
    assert min(run["front_slip_angle_p90_deg"] for run in reference_runs) > 0.0
    assert min(run["rear_slip_angle_p90_deg"] for run in reference_runs) > 0.0
    assert min(run["front_lateral_force_p90_n"] for run in reference_runs) > 0.0
    assert min(run["rear_lateral_force_p90_n"] for run in reference_runs) > 0.0
    for row in rows:
        assert {run["target_speed_kph"] for run in row["controlled_speed_runs"]} == {
            150.0,
            180.0,
            200.0,
            220.0,
            230.0,
            240.0,
            250.0,
        }


def test_fastf1_calibration_offline_compare_and_summarize(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"

    assert fastf1_calibration_main(["summarize", str(REFERENCE_CSV), "--output", str(summary_path)]) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["trace_features"]["braking_zones"]

    compare = compare_reference(physics_model="v2")
    assert list(compare["physics_models"]) == ["v2"]
    assert compare["physics_models"]["v2"]["error_terms"]["gear_match_rate"] == 1.0


def test_fastf1_multi_summary_aggregates_section_distributions(tmp_path) -> None:
    summary_a = {
        "source": "lap-a",
        "lap_time_s": 80.0,
        "distance_m": 5740.0,
        "max_speed_kph": 345.0,
        "mean_speed_kph": 258.0,
        "p10_speed_kph": 140.0,
        "p50_speed_kph": 270.0,
        "p90_speed_kph": 335.0,
        "trace_features": {
            "sustained_corner_targets": [
                {
                    "index": 0,
                    "start_distance_m": 2430.0,
                    "end_distance_m": 2570.0,
                    "length_m": 140.0,
                    "mean_speed_kph": 218.0,
                    "min_speed_kph": 202.0,
                    "max_speed_kph": 235.0,
                    "lateral_g_p75": 5.0,
                    "lateral_g_p90": 5.6,
                    "lateral_g_max": 6.0,
                    "curvature_abs_p90_rad_per_m": 0.0016,
                    "radius_p10_m": 620.0,
                }
            ],
            "braking_zones": [
                {
                    "start_distance_m": 610.0,
                    "length_m": 120.0,
                    "start_speed_kph": 335.0,
                    "min_speed_kph": 90.0,
                    "speed_drop_kph": 245.0,
                    "mean_decel_mps2": 24.0,
                }
            ],
            "corner_speed_targets": [
                {
                    "entry_speed_kph": 270.0,
                    "min_speed_kph": 90.0,
                    "exit_speed_kph": 210.0,
                    "lateral_g_p75": 3.5,
                    "lateral_g_p90": 4.0,
                }
            ],
        },
    }
    summary_b = json.loads(json.dumps(summary_a))
    summary_b["source"] = "lap-b"
    summary_b["lap_time_s"] = 79.0
    summary_b["trace_features"]["sustained_corner_targets"][0]["mean_speed_kph"] = 222.0

    path_a = tmp_path / "a" / "summary.json"
    path_b = tmp_path / "b" / "summary.json"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    path_a.write_text(json.dumps(summary_a), encoding="utf-8")
    path_b.write_text(json.dumps(summary_b), encoding="utf-8")

    aggregate = summarize_multi_reference(output_dir=tmp_path)

    assert aggregate["summary_count"] == 2
    assert aggregate["lap_distributions"]["lap_time_s"]["p50"] == 79.5
    sustained = aggregate["section_distributions"]["sustained_high_speed_curves"]["section_01"]
    assert sustained["mean_speed_kph"]["p50"] == 220.0
    assert aggregate["section_distributions"]["braking_zones"]["length_m"]["count"] == 2
