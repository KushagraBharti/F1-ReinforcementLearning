import json
from pathlib import Path

from f1rl.evolution_postcheck import postcheck_evolution_run
from f1rl.evolution_search import EvolutionGates, EvolutionSearchConfig, run_evolution_search
from f1rl.telemetry import load_steps


def test_evolution_postcheck_replays_saved_attempts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "evolution-run",
        config=EvolutionSearchConfig(
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=93,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "clean_exit"),
            checkpoint_every_generations=0,
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    summary_path = postcheck_evolution_run(output_dir, top_k=2, telemetry_compression="gzip")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))
    postchecked_rows = [
        json.loads(line)
        for line in (output_dir / "postchecked_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["postcheck_count"] >= 2
    assert summary["parity_status"] == "passed"
    assert summary["reason_mismatches"] == 0
    assert manifest["backend"] == "cpu_postcheck"
    assert manifest["trace_count"] == summary["postcheck_count"]
    assert len(postchecked_rows) == summary["postcheck_count"]
    assert all(Path(row["selected_telemetry"]).exists() for row in postchecked_rows)
    assert load_steps(Path(manifest["traces"][0]["path"]))


def test_evolution_postcheck_can_cpu_rerank_a_larger_gpu_pool(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "evolution-rerank-run",
        config=EvolutionSearchConfig(
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=5,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=131,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "clean_exit"),
            checkpoint_every_generations=0,
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts_path = output_dir / "attempts.jsonl"
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in attempts:
        row["lineage"] = {
            "source": "candidate_test",
            "parent_candidate_index": int(row["candidate_index"]),
            "parent_generation": int(row["generation"]),
        }
    best = max(attempts, key=lambda row: float(row["score"]))
    worst = min(attempts, key=lambda row: float(row["score"]))
    assert best["candidate_index"] != worst["candidate_index"]
    worst["score"] = float(best["score"]) + 1_000_000.0
    duplicate = dict(best)
    duplicate["candidate_index"] = max(int(row["candidate_index"]) for row in attempts) + 100
    duplicate["seed"] = int(best["seed"]) + 100_000
    duplicate["score"] = float(best["score"]) - 1.0
    duplicate["lineage"] = {
        "source": "clone_test",
        "parent_candidate_index": int(best["candidate_index"]),
        "parent_generation": int(best["generation"]),
    }
    attempts.append(duplicate)
    attempts_path.write_text(
        "\n".join(json.dumps(row) for row in attempts) + "\n",
        encoding="utf-8",
    )

    summary_path = postcheck_evolution_run(
        output_dir,
        top_k=2,
        candidate_pool_size=len(attempts),
        cpu_rerank=True,
        telemetry_compression="gzip",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    postchecked_rows = [
        json.loads(line)
        for line in (output_dir / "postchecked_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool_rows = [
        json.loads(line)
        for line in (output_dir / "postchecked_pool_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["selection_mode"] == "cpu_rerank"
    assert summary["cpu_rerank_status"] == "passed"
    assert summary["pool_postcheck_count"] == summary["pool_cpu_replay_unique_count"]
    assert summary["pool_cpu_replay_requested_count"] <= len(attempts)
    assert summary["pool_cpu_replay_cache_hits"] >= 1
    assert summary["pool_cpu_replay_unique_count"] < summary["pool_cpu_replay_requested_count"]
    assert summary["pool_score_parity_status"] == "failed"
    assert summary["pool_max_score_delta"] >= 999_000.0
    assert postchecked_rows
    assert pool_rows
    assert int(postchecked_rows[0]["candidate_index"]) == int(best["candidate_index"])
    assert all(int(row["candidate_index"]) != int(worst["candidate_index"]) for row in postchecked_rows)
    assert all(Path(row["selected_telemetry"]).exists() for row in postchecked_rows)
