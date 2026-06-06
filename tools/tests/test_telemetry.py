from pathlib import Path

from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS, TelemetryWriter, load_steps


def test_telemetry_jsonl_and_summary(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=3)
    writer = TelemetryWriter(tmp_path, mode="test", seed=3)
    result = sim.step(1)
    writer.write_step(result.telemetry)
    summary = writer.close_episode(termination_reason=sim.termination_reason, completed_lap=sim.completed_lap)
    rows = load_steps(writer.steps_path)
    assert len(rows) == 1
    assert set(rows[0]["reward_components"]) == set(REWARD_COMPONENT_KEYS)
    assert "lateral_g" in rows[0]
    assert "racing_line_deviation_m" in rows[0]
    assert "ghost_gap_m" in rows[0]
    assert set(summary.reward_totals) == set(REWARD_COMPONENT_KEYS)
    assert isinstance(summary.sector_times_s, list)
    assert hasattr(summary, "braking_zone_count")
