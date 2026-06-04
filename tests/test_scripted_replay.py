import json
from pathlib import Path

from f1rl.replay import ReplayControls, run_replay, run_replay_paths
from f1rl.scripted import run_scripted


def test_replay_controls_clamp_speed_and_queue_skips() -> None:
    controls = ReplayControls(speed=1.0)

    for _ in range(40):
        controls.speed_up()
    assert controls.speed == 32.0

    for _ in range(80):
        controls.speed_down()
    assert controls.speed == 0.05

    controls.reset_speed()
    assert controls.speed == 1.0

    controls.queue_generation_skip(3)
    assert controls.take_generation_skip()
    assert controls.take_generation_skip()
    assert controls.take_generation_skip()
    assert not controls.take_generation_skip()

    controls.skip_remaining_generations()
    assert controls.take_generation_skip()


def test_scripted_and_replay_smoke() -> None:
    run_scripted(steps=8, seed=11, telemetry=True)
    # Use the newest scripted telemetry from artifacts.
    from f1rl.config import ARTIFACTS_DIR

    steps_path = sorted(ARTIFACTS_DIR.glob("scripted-*/steps.jsonl"))[-1]
    assert run_replay(steps_path, headless=True) == 0


def test_replay_can_load_multiple_paths_headless() -> None:
    run_scripted(steps=8, seed=12, telemetry=True)
    run_scripted(steps=8, seed=13, telemetry=True)

    from f1rl.config import ARTIFACTS_DIR

    paths = sorted(ARTIFACTS_DIR.glob("scripted-*/steps.jsonl"))[-2:]
    assert len(paths) == 2
    assert run_replay_paths(paths, headless=True) == 0
    assert run_replay_paths([paths[0].parent], headless=True, limit=1) == 0
    assert run_replay_paths(paths, headless=True, sort_by="best-progress") == 0


def _write_minimal_trace(path: Path, *, progress_m: float) -> None:
    rows = [
        {
            "sim_time_s": 0.0,
            "x": 100.0,
            "y": 100.0,
            "heading_deg": 0.0,
            "speed_mps": 0.0,
            "speed_kph": 0.0,
            "yaw_rate_rps": 0.0,
            "checkpoint_index": 0,
            "lap_index": 0,
            "raw_progress_m": 0.0,
            "monotonic_progress_m": 0.0,
            "terminated": False,
            "termination_reason": "running",
        },
        {
            "sim_time_s": 0.1,
            "x": 100.0 + progress_m,
            "y": 100.0,
            "heading_deg": 0.0,
            "speed_mps": 1.0,
            "speed_kph": 3.6,
            "yaw_rate_rps": 0.0,
            "checkpoint_index": 0,
            "lap_index": 0,
            "raw_progress_m": progress_m,
            "monotonic_progress_m": progress_m,
            "terminated": True,
            "termination_reason": "max_steps",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_replay_can_group_manifest_traces_by_generation(tmp_path: Path, capsys) -> None:
    traces = []
    for generation in range(2):
        for candidate_index in range(2):
            progress_m = float(generation * 100 + candidate_index)
            trace_path = tmp_path / f"trace-g{generation}-c{candidate_index}.jsonl"
            _write_minimal_trace(trace_path, progress_m=progress_m)
            traces.append(
                {
                    "generation": generation,
                    "candidate_index": candidate_index,
                    "score": progress_m,
                    "best_progress_m": progress_m,
                    "path": str(trace_path),
                }
            )
    (tmp_path / "manifest.json").write_text(json.dumps({"traces": traces}), encoding="utf-8")

    assert run_replay_paths([tmp_path], headless=True, sort_by="best-progress", by_generation=True) == 0

    output = capsys.readouterr().out
    assert "replay_loaded groups=2 traces=4" in output
    assert "generation_group index=1 generation=0 traces=2" in output
    assert "generation_group index=2 generation=1 traces=2" in output
