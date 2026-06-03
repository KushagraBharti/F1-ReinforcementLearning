from pathlib import Path

from f1rl.action_search import default_schedules, run_action_search, schedules_for_preset
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import snapshot_from_sim


def test_default_schedules_include_both_turn_directions() -> None:
    names = {schedule.name for schedule in default_schedules("expanded", max_steps=20)}

    assert any("soft_left" in name for name in names)
    assert any("soft_right" in name for name in names)
    assert any("throttle_soft_left" in name for name in names)
    assert any("throttle_soft_right" in name for name in names)
    assert any(name.startswith("coastx60__soft_left") for name in names)
    assert any(name.startswith("coastx60__soft_right") for name in names)


def test_default_schedules_include_tiny_exit_actions() -> None:
    names = {schedule.name for schedule in default_schedules("exit_tiny", max_steps=20)}

    assert any("maintenance_tiny_left" in name for name in names)
    assert any("maintenance_tiny_right" in name for name in names)
    assert any("soft_brakex60__maintenance_tiny_left" in name for name in names)


def test_rotation_bridge_preset_is_targeted_and_includes_handoffs() -> None:
    schedules = schedules_for_preset("turnin_power", max_steps=80, schedule_preset="rotation_bridge")
    names = {schedule.name for schedule in schedules}

    assert 0 < len(schedules) < 300
    assert len(schedules) < len(default_schedules("turnin_power", max_steps=80))
    assert any(name.startswith("maintenance_soft_rightx12__maintenance_soft_left") for name in names)
    assert any(name.startswith("soft_brakex12__maintenance_soft_left") for name in names)
    assert any("soft_brake_soft_left" in name for name in names)


def test_action_search_writes_summary_telemetry_and_elite_library(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 500.0, "start_speed_kph": 80.0})
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [snapshot_from_sim(sim, source="unit")],
        source="unit",
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "fake.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real model")
    metadata = {
        "sim_config": {
            "max_steps": 20,
            "action_mode": "discrete",
            "action_set": "expanded",
        }
    }
    (checkpoint.parent.parent / "run_metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

    output_dir = run_action_search(
        state_library=library_path,
        checkpoint=checkpoint,
        output_dir=tmp_path / "action-search",
        target_progress_m=510.0,
        target_max_speed_kph=None,
        max_steps=20,
        seed=4,
        top_k=2,
        max_schedules=3,
        schedule_offset=1,
        metadata_mode="require",
    )

    assert (output_dir / "action_search_summary.json").exists()
    assert (output_dir / "attempts.jsonl").exists()
    assert len(load_state_library(output_dir / "elite_state_library.json")) == 2
    assert len(list((output_dir / "selected_telemetry").glob("*.jsonl"))) == 2


def test_action_search_can_override_action_set_from_metadata(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 500.0, "start_speed_kph": 80.0})
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [snapshot_from_sim(sim, source="unit")],
        source="unit",
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "fake.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real model")
    metadata = {
        "sim_config": {
            "max_steps": 20,
            "action_mode": "discrete",
            "action_set": "expanded",
        }
    }
    (checkpoint.parent.parent / "run_metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

    output_dir = run_action_search(
        state_library=library_path,
        checkpoint=checkpoint,
        output_dir=tmp_path / "action-search-override",
        target_progress_m=510.0,
        target_max_speed_kph=None,
        max_steps=20,
        seed=4,
        top_k=1,
        max_schedules=1,
        metadata_mode="require",
        override_action_set="exit_tiny",
    )

    summary = __import__("json").loads((output_dir / "action_search_summary.json").read_text(encoding="utf-8"))
    assert summary["override_action_set"] == "exit_tiny"
    assert summary["checkpoint_config"]["sim_config"]["action_set"] == "exit_tiny"


def test_action_search_records_schedule_preset(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 500.0, "start_speed_kph": 80.0})
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [snapshot_from_sim(sim, source="unit")],
        source="unit",
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "fake.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real model")
    metadata = {
        "sim_config": {
            "max_steps": 20,
            "action_mode": "discrete",
            "action_set": "turnin_power",
        }
    }
    (checkpoint.parent.parent / "run_metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")

    output_dir = run_action_search(
        state_library=library_path,
        checkpoint=checkpoint,
        output_dir=tmp_path / "action-search-preset",
        target_progress_m=510.0,
        target_max_speed_kph=None,
        max_steps=20,
        seed=4,
        top_k=1,
        max_schedules=2,
        schedule_preset="rotation_bridge",
        metadata_mode="require",
    )

    summary = __import__("json").loads((output_dir / "action_search_summary.json").read_text(encoding="utf-8"))
    assert summary["schedule_preset"] == "rotation_bridge"
