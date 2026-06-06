import json

from f1rl import manual
from f1rl.telemetry import load_steps


def test_manual_headless_v2_ghost_reference_writes_gap_telemetry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(manual, "ARTIFACTS_DIR", tmp_path)

    assert (
        manual.run_manual(
            max_steps=4,
            seed=31415,
            headless=True,
            physics_model="v2",
            ghost_reference=True,
            flying_start=True,
        )
        == 0
    )

    runs = sorted(tmp_path.glob("manual-headless-*-seed31415-*"))
    assert runs
    run_dir = runs[-1]
    summary = json.loads((run_dir / "episode_summary.json").read_text(encoding="utf-8"))
    steps = load_steps(run_dir / "steps.jsonl")

    assert summary["physics_model"] == "v2"
    assert summary["physics_version"].startswith("physics_v2.")
    assert summary["physics_calibration_id"]
    assert summary["avg_ghost_gap_m"] is not None
    assert summary["final_ghost_gap_m"] is not None
    assert steps[0]["physics_model"] == "v2"
    assert steps[0]["reference_progress_m"] is not None
    assert steps[0]["reference_speed_kph"] is not None
    assert steps[0]["ghost_gap_m"] is not None


def test_manual_headless_v2_can_start_before_sustained_corner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(manual, "ARTIFACTS_DIR", tmp_path)

    assert (
        manual.run_manual(
            max_steps=4,
            seed=27182,
            headless=True,
            physics_model="v2",
            ghost_reference=True,
            start_section="sustained_corner_03",
            start_section_lead_in_m=120.0,
        )
        == 0
    )

    runs = sorted(tmp_path.glob("manual-headless-*-seed27182-*"))
    assert runs
    run_dir = runs[-1]
    summary = json.loads((run_dir / "episode_summary.json").read_text(encoding="utf-8"))
    steps = load_steps(run_dir / "steps.jsonl")

    assert steps[0]["monotonic_progress_m"] > 4800.0
    assert steps[0]["reference_progress_m"] is not None
    assert steps[0]["reference_progress_m"] > 4800.0
    assert steps[0]["ghost_gap_m"] is not None
    assert abs(steps[0]["ghost_gap_m"]) < 25.0
    assert steps[0]["speed_kph"] > 180.0
    assert 0.0 < summary["distance_traveled_m"] < 100.0
    assert summary["sector_times_s"][:2] == [None, None]
    assert summary["sector_speed_kph"][:2] == [0.0, 0.0]
