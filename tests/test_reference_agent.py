from __future__ import annotations

from pathlib import Path

from f1rl.reference_agent import (
    load_reference_profile,
    parse_timedelta_seconds,
    run_reference_control,
    run_reference_ghost,
)


def test_parse_fastf1_timedelta() -> None:
    assert parse_timedelta_seconds("0 days 00:01:19.662000") == 79.662
    assert parse_timedelta_seconds("00:00:03.500000") == 3.5


def test_reference_profile_loads() -> None:
    profile = load_reference_profile()
    assert profile.lap_time_s > 70.0
    assert profile.distance_max_m > 5000.0
    assert profile.speed_at(0.0) > 250.0
    assert 0.0 <= profile.throttle_at(0.0) <= 1.0
    assert 0.0 <= profile.brake_at(0.0) <= 1.0


def test_reference_ghost_runs_without_telemetry() -> None:
    assert run_reference_ghost(seed=3, telemetry=False) is None


def test_reference_control_short_run_writes_telemetry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("f1rl.reference_agent.ARTIFACTS_DIR", tmp_path)
    root = run_reference_control(seed=3, steps=5, telemetry=True)
    assert root is not None
    assert (root / "steps.jsonl").exists()
    assert (root / "episode_summary.json").exists()
