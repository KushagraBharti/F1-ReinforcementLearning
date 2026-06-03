from f1rl.section_analysis import (
    detect_first_bad_event,
    failure_report,
    section_for_progress,
    summarize_sections,
)
from f1rl.track_sections import distance_to_next_braking_gate


def _row(
    step: int,
    progress_m: float,
    *,
    speed_kph: float,
    throttle: float = 1.0,
    brake: float = 0.0,
    termination_reason: str = "active",
) -> dict:
    return {
        "step_index": step,
        "sim_time_s": step / 60.0,
        "monotonic_progress_m": progress_m,
        "speed_kph": speed_kph,
        "throttle": throttle,
        "brake": brake,
        "steering": 0.0,
        "action_id": 1,
        "lateral_error_m": 0.0,
        "heading_error_deg": 0.0,
        "ray_distances_m": [20.0],
        "termination_reason": termination_reason,
        "reward_components": {
            "progress": 1.0,
            "finish": 0.0,
            "collision": 0.0,
            "off_track": 0.0,
            "no_progress": 0.0,
            "lateral": 0.0,
            "track_limit": 0.0,
            "heading": 0.0,
            "speed_target": 0.0,
            "overspeed_action": 0.0,
            "steering_target": 0.0,
            "smoothness": 0.0,
        },
    }


def test_section_for_progress_identifies_chicane_region() -> None:
    assert section_for_progress(966.0).name == "rettifilo_chicane"


def test_distance_to_next_braking_gate_wraps_after_final_gate() -> None:
    assert distance_to_next_braking_gate(500.0) == 20.0
    assert distance_to_next_braking_gate(530.0) == 1420.0
    assert distance_to_next_braking_gate(5270.0) == 1043.0


def test_detect_first_bad_event_finds_throttle_during_brake_demand() -> None:
    steps = [
        _row(1, 480.0, speed_kph=250.0),
        _row(2, 550.0, speed_kph=270.0, throttle=1.0, brake=0.0),
        _row(3, 620.0, speed_kph=280.0, termination_reason="off_track"),
    ]

    event = detect_first_bad_event(steps)

    assert event is not None
    assert event["kind"] == "throttle_during_brake_demand"
    assert event["section"] == "rettifilo_chicane"
    assert event["progress_m"] == 550.0


def test_failure_report_includes_action_distribution_before_failure() -> None:
    steps = [
        _row(1, 480.0, speed_kph=250.0),
        _row(2, 550.0, speed_kph=270.0),
        _row(3, 620.0, speed_kph=280.0, termination_reason="off_track"),
    ]

    report = failure_report(steps)

    assert report["failed_section"] == "rettifilo_chicane"
    assert report["first_bad_event"]["kind"] == "throttle_during_brake_demand"
    assert report["action_histogram_before_failure"]["throttle"] == 2


def test_failure_report_prefers_telemetry_action_name() -> None:
    steps = [
        {**_row(1, 480.0, speed_kph=250.0), "action_id": 2, "action_name": "throttle"},
        {**_row(2, 550.0, speed_kph=270.0), "action_id": 2, "action_name": "throttle"},
        {
            **_row(3, 620.0, speed_kph=280.0, termination_reason="off_track"),
            "action_id": 15,
            "action_name": "brake_left",
        },
    ]

    report = failure_report(steps)

    assert report["first_bad_event"]["action"] == "throttle"
    assert report["action_histogram_before_failure"]["throttle"] == 2


def test_summarize_sections_reports_speed_and_brake_metrics() -> None:
    steps = [
        _row(1, 480.0, speed_kph=220.0, throttle=1.0, brake=0.0),
        _row(2, 560.0, speed_kph=180.0, throttle=0.0, brake=1.0),
        _row(3, 800.0, speed_kph=120.0, throttle=0.5, brake=0.0, termination_reason="off_track"),
    ]

    section = next(item for item in summarize_sections(steps) if item["section"] == "rettifilo_chicane")

    assert section["rows"] == 3
    assert section["entry_speed_kph"] == 220.0
    assert section["min_speed_kph"] == 120.0
    assert section["brake_start_progress_m"] == 560.0
    assert section["termination_reason"] == "off_track"
