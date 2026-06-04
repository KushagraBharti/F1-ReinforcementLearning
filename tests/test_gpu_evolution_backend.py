import json
from dataclasses import asdict
from pathlib import Path

from f1rl.evolution_search import EvolutionGates, EvolutionSearchConfig, run_evolution_search
from f1rl.telemetry import load_steps


def _attempt_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gpu_backend_writes_compatible_artifacts_with_cpu_verification(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-evolution",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_verify_top_k=2,
            action_set="straight",
            observation_profile="base",
            max_steps=12,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=31,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "farthest_distance"),
            checkpoint_every_generations=1,
        ),
        gates=EvolutionGates(target_progress_m=8.0),
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert summary["attempt_count"] == 4
    assert generation["backend"] == "gpu"
    assert generation["cpu_verification_count"] >= 2
    assert generation["gpu_candidates_per_second"] > 0.0
    assert any(row.get("gpu_verified") is True for row in attempts)
    assert all("profile_scores" in row for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["trace_count"] == 2
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))


def test_gpu_parity_check_matches_cpu_verified_tiny_search(tmp_path: Path) -> None:
    base = EvolutionSearchConfig(
        action_set="straight",
        observation_profile="base",
        max_steps=10,
        population=3,
        generations=1,
        elite_count=1,
        random_immigrants=0,
        min_phases=1,
        max_phases=2,
        min_phase_steps=2,
        max_phase_steps=4,
        seed=41,
        top_k=1,
        workers=1,
        genome_type="controller",
        scoring_profiles=("max_progress",),
    )
    gates = EvolutionGates(target_progress_m=6.0)
    cpu_dir = run_evolution_search(
        output_dir=tmp_path / "cpu",
        config=base,
        gates=gates,
        start_speed_kph=60.0,
    )
    gpu_dir = run_evolution_search(
        output_dir=tmp_path / "gpu",
        config=EvolutionSearchConfig(
            **{
                **asdict(base),
                "backend": "gpu",
                "gpu_device": "cpu",
                "gpu_dtype": "float64",
                "gpu_parity_check": True,
            }
        ),
        gates=gates,
        start_speed_kph=60.0,
    )

    cpu_attempts = _attempt_rows(cpu_dir / "attempts.jsonl")
    gpu_attempts = _attempt_rows(gpu_dir / "attempts.jsonl")

    assert [row["best_progress_m"] for row in gpu_attempts] == [row["best_progress_m"] for row in cpu_attempts]
    assert all(row["gpu_verified"] is True for row in gpu_attempts)


def test_gpu_backend_supports_phase_genomes(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-phase",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_verify_top_k=1,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=47,
            top_k=1,
            workers=1,
            genome_type="phase",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))

    assert len(attempts) == 3
    assert summary["generation_summaries"][0]["backend"] == "gpu"
    assert all(row["genome"]["kind"] == "phase" for row in attempts)
    assert any(row.get("gpu_verified") is True for row in attempts)


def test_gpu_backend_supports_progress_phase_genomes(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-progress-phase",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_verify_top_k=1,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_progress_m=2.0,
            max_phase_progress_m=5.0,
            seed=53,
            top_k=1,
            workers=1,
            genome_type="progress_phase",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))

    assert len(attempts) == 3
    assert summary["generation_summaries"][0]["backend"] == "gpu"
    assert all(row["genome"]["kind"] == "progress_phase" for row in attempts)
    assert any(row.get("gpu_verified") is True for row in attempts)


def test_gpu_all_candidate_telemetry_replays_to_external_dir(tmp_path: Path) -> None:
    all_trace_dir = tmp_path / "all-candidate-traces"
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-all-telemetry",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_telemetry_mode="all-cpu-replay",
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=37,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
            telemetry_selection="all",
            all_candidate_telemetry_dir=all_trace_dir,
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["backend"] == "gpu"
    assert manifest["gpu_telemetry_mode"] == "all-cpu-replay"
    assert manifest["all_candidate_telemetry_dir"] == str(all_trace_dir)
    assert manifest["trace_count"] == 3
    assert all(Path(trace["path"]).parent == all_trace_dir for trace in manifest["traces"])
    assert all(load_steps(Path(trace["path"])) for trace in manifest["traces"])
