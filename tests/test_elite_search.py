from pathlib import Path

from f1rl.elite_search import run_elite_search
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import snapshot_from_sim


def test_elite_search_writes_elite_library_and_selected_telemetry(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 500.0, "start_speed_kph": 80.0})
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [snapshot_from_sim(sim, source="unit")],
        source="unit",
    )

    output_dir = run_elite_search(
        state_library=library_path,
        output_dir=tmp_path / "elite",
        attempts=3,
        max_steps=20,
        seed=4,
        top_k=2,
        segment_length_m=20.0,
        start_min_progress_m=450.0,
        start_max_progress_m=560.0,
        policy="scripted",
    )

    elite_library = output_dir / "elite_state_library.json"
    assert elite_library.exists()
    assert len(load_state_library(elite_library)) == 2
    assert (output_dir / "elite_search_summary.json").exists()
    assert (output_dir / "attempts.jsonl").exists()
    assert len(list((output_dir / "selected_telemetry").glob("*.jsonl"))) == 2
