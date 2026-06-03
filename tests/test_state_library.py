import json
from pathlib import Path

import pytest

from f1rl.config import SimConfig
from f1rl.sim import MonzaSim
from f1rl.state_library import (
    load_state_library,
    snapshots_from_scripted,
    snapshots_from_telemetry,
    write_state_library,
)
from f1rl.state_snapshot import snapshot_from_sim, snapshot_to_dict


def test_sim_can_reset_from_state_snapshot_and_continue() -> None:
    sim = MonzaSim(SimConfig(max_steps=80))
    sim.reset(seed=1)
    for _ in range(12):
        sim.step(1)
    snapshot = snapshot_from_sim(sim, source="unit")

    restored = MonzaSim(SimConfig(max_steps=80))
    obs, info = restored.reset(
        seed=2,
        options={
            "state_snapshot": snapshot_to_dict(snapshot),
            "segment_length_m": 2.0,
            "curriculum_stage": "library-unit",
        },
    )

    assert obs.shape == (restored.observation_dim,)
    assert info["curriculum_stage"] == "library-unit"
    assert info["segment_target_progress_m"] == pytest.approx(snapshot.monotonic_progress_m + 2.0)
    assert restored.state.monotonic_progress_m == pytest.approx(snapshot.monotonic_progress_m)
    assert restored.next_checkpoint_index == snapshot.next_checkpoint_index
    result = restored.step(1)
    assert result.telemetry.curriculum_stage == "library-unit"


def test_state_snapshot_reset_clears_terminal_episode_flags() -> None:
    snapshot = {
        "id": "terminal-segment",
        "source": "unit",
        "step_index": 67,
        "sim_time_s": 1.1,
        "x": 1086.0,
        "y": 885.0,
        "heading_rad": 3.14,
        "speed_mps": 44.0,
        "yaw_rate_rps": 0.0,
        "steering_rad": 0.0,
        "raw_progress_m": 600.0,
        "monotonic_progress_m": 600.0,
        "checkpoint_index": 10,
        "next_checkpoint_index": 11,
        "checkpoints_passed": 10,
        "missed_checkpoint_count": 0,
        "lap_index": 0,
        "valid_lap": True,
        "finish_crossed": True,
        "completed_lap": True,
        "segment_complete": True,
        "last_throttle": 0.0,
        "last_brake": 1.0,
        "last_steer": 0.0,
        "last_action_id": 2,
    }
    sim = MonzaSim(SimConfig(max_steps=80))

    _, info = sim.reset(
        seed=1,
        options={
            "state_snapshot": snapshot,
            "segment_length_m": 50.0,
            "curriculum_stage": "fresh-segment",
        },
    )

    assert info["termination_reason"] == "active"
    assert info["segment_complete"] is False
    assert sim.completed_lap is False
    assert sim.finish_crossed is False


def test_state_library_from_telemetry_file(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "steps.jsonl"
    row = {
        "step_index": 10,
        "sim_time_s": 1.0,
        "x": 100.0,
        "y": 200.0,
        "heading_deg": 45.0,
        "speed_mps": 30.0,
        "yaw_rate_rps": 0.1,
        "steering": 0.25,
        "raw_progress_m": 123.0,
        "monotonic_progress_m": 123.0,
        "checkpoint_index": 2,
        "next_checkpoint_index": 3,
        "checkpoints_passed": 2,
        "missed_checkpoint_count": 0,
        "lap_index": 0,
        "valid_lap": True,
        "finish_crossed": False,
        "segment_complete": False,
        "throttle": 0.5,
        "brake": 0.0,
        "action_id": 1,
    }
    telemetry_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    snapshots = snapshots_from_telemetry(
        telemetry_path,
        sample_every_m=0.0,
        sample_every_steps=1,
        max_snapshots=0,
    )

    assert len(snapshots) == 1
    assert snapshots[0].source_file == str(telemetry_path)
    assert snapshots[0].monotonic_progress_m == 123.0
    assert snapshots[0].last_throttle == 0.5


def test_state_library_from_telemetry_filters_progress_and_speed(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "steps.jsonl"
    rows = [
        {
            "step_index": 1,
            "sim_time_s": 0.1,
            "x": 100.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "speed_kph": 220.0,
            "monotonic_progress_m": 590.0,
            "raw_progress_m": 590.0,
        },
        {
            "step_index": 2,
            "sim_time_s": 0.2,
            "x": 101.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "speed_kph": 160.0,
            "monotonic_progress_m": 600.3,
            "raw_progress_m": 600.3,
        },
        {
            "step_index": 3,
            "sim_time_s": 0.3,
            "x": 102.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "speed_kph": 80.0,
            "monotonic_progress_m": 601.0,
            "raw_progress_m": 601.0,
        },
    ]
    telemetry_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    snapshots = snapshots_from_telemetry(
        telemetry_path,
        sample_every_m=0.0,
        sample_every_steps=1,
        max_snapshots=0,
        min_progress_m=600.0,
        max_progress_m=601.0,
        min_speed_kph=150.0,
        max_speed_kph=170.0,
    )

    assert len(snapshots) == 1
    assert snapshots[0].step_index == 2
    assert snapshots[0].monotonic_progress_m == pytest.approx(600.3)
    assert snapshots[0].speed_mps == pytest.approx(160.0 / 3.6)


def test_scripted_state_library_can_be_written_and_loaded(tmp_path: Path) -> None:
    snapshots = snapshots_from_scripted(
        steps=40,
        seed=3,
        sample_every_m=0.0,
        sample_every_steps=10,
        max_snapshots=0,
    )
    output = write_state_library(tmp_path / "state_library.json", snapshots, source="scripted")
    loaded = load_state_library(output)

    assert output.exists()
    assert len(loaded) >= 2
    assert loaded[0].source == "scripted"
    assert loaded[-1].step_index >= loaded[0].step_index
