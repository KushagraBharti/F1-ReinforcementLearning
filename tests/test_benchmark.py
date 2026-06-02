import json

from f1rl import benchmark


def test_random_benchmark_writes_required_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path)

    root = benchmark.run_benchmark(
        policies=["random"],
        episodes=1,
        max_steps=5,
        seed=3,
        checkpoint="latest",
        device="cpu",
        telemetry="selected",
        telemetry_every=1,
    )

    assert (root / "config.json").exists()
    assert (root / "summary.json").exists()
    assert (root / "summary.csv").exists()
    assert (root / "per_episode.jsonl").exists()
    assert (root / "selected_telemetry" / "random-episode-000-steps.jsonl").exists()

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["policies"][0]["policy"] == "random"
    assert "completion_rate" in summary["policies"][0]
    assert "best_progress_m" in summary["policies"][0]
    assert "invalid_lap_rate" in summary["policies"][0]
    assert "avg_missed_checkpoint_count" in summary["policies"][0]
    assert "avg_steps" in summary["policies"][0]

    rows = [
        json.loads(line)
        for line in (root / "per_episode.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["policy"] == "random"
    assert "steps_per_second" in rows[0]


def test_episode_metrics_uses_max_checkpoint_count_when_final_row_wraps() -> None:
    def row(step: int, checkpoint_index: int, checkpoints_passed: int) -> dict:
        return {
            "step_index": step,
            "sim_time_s": float(step),
            "termination_reason": "lap_complete" if step == 3 else "active",
            "valid_lap": True,
            "finish_crossed": step == 3,
            "segment_complete": False,
            "monotonic_progress_m": step * 1000.0,
            "checkpoint_index": checkpoint_index,
            "checkpoints_passed": checkpoints_passed,
            "missed_checkpoint_count": 0,
            "ray_distances_m": [10.0],
            "reward_components": {
                "progress": 1.0,
                "finish": 0.0,
                "collision": 0.0,
                "off_track": 0.0,
                "no_progress": 0.0,
                "smoothness": 0.0,
            },
            "reward_total": 1.0,
            "collided": False,
            "off_track": False,
            "speed_kph": 100.0,
            "lateral_g": 0.0,
            "racing_line_deviation_m": 0.0,
            "ghost_gap_m": 0.0,
            "curriculum_stage": None,
            "segment_target_progress_m": None,
        }

    metrics = benchmark._episode_metrics(
        policy="reference_ghost",
        episode=0,
        seed=7,
        rows=[
            row(1, checkpoint_index=118, checkpoints_passed=118),
            row(2, checkpoint_index=119, checkpoints_passed=119),
            row(3, checkpoint_index=0, checkpoints_passed=0),
        ],
        wall_clock_s=1.0,
    )

    assert metrics["checkpoints_passed"] == 119
    assert metrics["valid_lap"] is True
    assert metrics["completed_lap"] is True
