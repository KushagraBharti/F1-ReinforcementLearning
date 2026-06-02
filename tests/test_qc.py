import json
from pathlib import Path

from f1rl import qc
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
