# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from f1rl.config import SimConfig
from f1rl.evolution_backend import _verify_rows_with_cpu
from f1rl.evolution_search import (
    Candidate,
    EvolutionGates,
    EvolutionSearchConfig,
    Genome,
    run_evolution_search,
)
from f1rl.gpu_fast_warp import warp_status
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import snapshot_from_sim
from f1rl.telemetry import load_steps


def _attempt_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gpu_cpu_rerank_reuses_identical_genome_snapshot_replays() -> None:
    sim_config = SimConfig(action_set="straight", observation_profile="base", max_steps=5, no_progress_limit_steps=8)
    sim = MonzaSim(sim_config)
    sim.reset(seed=11)
    snapshot = snapshot_from_sim(sim, source="unit")
    genome = Genome(kind="controller", controller_weights=())
    candidates = [
        Candidate(genome=genome, snapshot_index=0, lineage={"source": "elite"}),
        Candidate(genome=genome, snapshot_index=0, lineage={"source": "clone"}),
    ]
    rows = [
        {
            "candidate_index": 0,
            "generation": 0,
            "seed": 101,
            "score": 0.0,
            "profile_scores": {"max_progress": 0.0},
            "best_progress_m": snapshot.monotonic_progress_m,
            "final_progress_m": snapshot.monotonic_progress_m,
            "termination_reason": "max_steps",
            "valid_lap": False,
        },
        {
            "candidate_index": 1,
            "generation": 0,
            "seed": 202,
            "score": 0.0,
            "profile_scores": {"max_progress": 0.0},
            "best_progress_m": snapshot.monotonic_progress_m,
            "final_progress_m": snapshot.monotonic_progress_m,
            "termination_reason": "max_steps",
            "valid_lap": False,
        },
    ]
    cache: dict[str, dict] = {}

    verified, metrics = _verify_rows_with_cpu(
        rows,
        verify_indices=[0, 1],
        candidates=candidates,
        snapshots=[snapshot],
        sim_config=sim_config,
        gates=EvolutionGates(target_progress_m=snapshot.monotonic_progress_m + 10.0, terminate_at_target_progress=False),
        scoring_profiles=("max_progress",),
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
        replay_cache=cache,
    )

    assert metrics["gpu_cpu_replay_requested_count"] == 2
    assert metrics["gpu_cpu_replay_unique_count"] == 1
    assert metrics["gpu_cpu_replay_cache_hits"] == 1
    assert metrics["gpu_cpu_replay_duplicate_count"] == 1
    assert len(cache) == 1
    assert verified[0]["cpu_replay_cache_hit"] is False
    assert verified[1]["cpu_replay_cache_hit"] is True
    assert verified[1]["candidate_index"] == 1
    assert verified[1]["lineage"] == {"source": "clone"}


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
            max_steps=10,
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
    assert generation["gpu_profile_ranges_enabled"] is False
    assert generation["cpu_verification_count"] >= 2
    assert generation["gpu_cpu_replay_count"] == generation["cpu_verification_count"]
    assert generation["gpu_cpu_replay_seconds"] >= 0.0
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
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


def test_gpu_backend_supports_exact_grid_collision_mode(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-exact-grid",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_verify_top_k=1,
            gpu_collision_mode="exact_grid",
            action_set="straight",
            observation_profile="base",
            max_steps=10,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=61,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]

    assert generation["gpu_collision_mode"] == "exact_grid"
    assert generation["cpu_verification_count"] >= 1
    assert generation["gpu_parity_status"] == "passed"


def test_gpu_graph_engine_reports_cpu_fallback(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-graph-cpu-fallback",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_engine="graph",
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
            seed=73,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]

    assert generation["gpu_engine"] == "graph"
    assert generation["gpu_kernel_backend"] == "pytorch_eager_graph_fallback"
    assert generation["gpu_graph_capture_count"] == 0
    assert generation["gpu_graph_fallback_count"] == 1
    assert generation["gpu_graph_cache_hit_count"] == 0
    assert generation["gpu_graph_cache_miss_count"] == 0
    assert generation["gpu_graph_cache_size"] == 0
    assert generation["gpu_graph_errors"]
    assert generation["gpu_compile_requested"] is False
    assert generation["gpu_parity_status"] == "passed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph cache regression requires CUDA")
def test_gpu_graph_engine_reuses_cuda_graph_cache(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-graph-cache",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_engine="graph",
            gpu_device="cuda",
            gpu_dtype="float32",
            gpu_cpu_replay_top_k=1,
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=2,
            elite_count=1,
            random_immigrants=0,
            seed=79,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    first, second = summary["generation_summaries"]

    assert first["gpu_kernel_backend"] == "pytorch_cuda_graph"
    assert first["gpu_graph_capture_count"] == 1
    assert first["gpu_graph_replay_count"] == 1
    assert first["gpu_graph_cache_hit_count"] == 0
    assert first["gpu_graph_cache_miss_count"] == 1
    assert first["gpu_graph_fallback_count"] == 0
    assert first["gpu_parity_status"] == "passed"

    assert second["gpu_kernel_backend"] == "pytorch_cuda_graph"
    assert second["gpu_graph_capture_count"] == 0
    assert second["gpu_graph_replay_count"] == 1
    assert second["gpu_graph_cache_hit_count"] == 1
    assert second["gpu_graph_cache_miss_count"] == 0
    assert second["gpu_graph_cache_size"] == 1
    assert second["gpu_graph_fallback_count"] == 0
    assert second["gpu_parity_status"] == "passed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph chunk regression requires CUDA")
def test_gpu_graph_engine_replays_cuda_graph_chunks(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-graph-chunks",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_engine="graph",
            gpu_device="cuda",
            gpu_dtype="float32",
            gpu_cpu_replay_top_k=1,
            gpu_static_batch_size=4,
            gpu_chunk_steps=4,
            gpu_collision_mode="exact_grid",
            action_set="straight",
            observation_profile="base",
            max_steps=10,
            population=4,
            generations=2,
            elite_count=1,
            random_immigrants=0,
            seed=83,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=8.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    first, second = summary["generation_summaries"]

    assert first["gpu_kernel_backend"] == "pytorch_cuda_graph_chunked"
    assert first["gpu_graph_steps_per_capture"] == [2, 4]
    assert first["gpu_graph_requested_replay_count"] == 3
    assert first["gpu_graph_capture_count"] == 1
    assert first["gpu_graph_replay_count"] == 3
    assert first["gpu_steps_executed"] == 10
    assert first["gpu_graph_fallback_count"] == 0
    assert first["gpu_parity_status"] == "passed"

    assert second["gpu_kernel_backend"] == "pytorch_cuda_graph_chunked"
    assert second["gpu_graph_capture_count"] == 0
    assert second["gpu_graph_replay_count"] == 3
    assert second["gpu_steps_executed"] == 10
    assert second["gpu_graph_cache_hit_count"] == 1
    assert second["gpu_graph_cache_miss_count"] == 0
    assert second["gpu_graph_fallback_count"] == 0
    assert second["gpu_parity_status"] == "passed"


def test_gpu_backend_static_batch_padding_does_not_emit_fake_attempts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-static-padding",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_verify_top_k=1,
            gpu_static_batch_size=5,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=67,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]

    assert len(attempts) == 3
    assert generation["gpu_batch_size"] == 5
    assert generation["gpu_static_batch_size"] == 5
    assert generation["gpu_static_padding_count"] == 2
    assert generation["gpu_sim_steps"] <= 3 * 8


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


def test_gpu_backend_writes_torch_profile_artifacts(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-profile",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_profile="torch",
            gpu_profile_output=profile_dir,
            gpu_chunk_steps=4,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=57,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=6.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]

    assert generation["gpu_engine"] == "eager"
    assert generation["gpu_profile"] == "torch"
    assert generation["gpu_profile_ranges_enabled"] is True
    assert generation["gpu_host_sync_count"] >= 1
    assert generation["gpu_kernel_launches_per_generation"] >= 0
    assert generation["gpu_profile_cuda_launch_count"] == generation["gpu_kernel_launches_per_generation"]
    assert Path(generation["gpu_profile_trace"]).exists()
    assert Path(generation["gpu_profile_table"]).exists()
    assert Path(generation["gpu_profile_summary"]).exists()
    assert Path(generation["gpu_profile_bottleneck_report"]).exists()
    assert generation["gpu_profile_bottlenecks"]
    profile_summary = json.loads(Path(generation["gpu_profile_summary"]).read_text(encoding="utf-8"))
    assert profile_summary["cuda_launch_count"] == generation["gpu_kernel_launches_per_generation"]
    assert profile_summary["bottlenecks"] == generation["gpu_profile_bottlenecks"]
    assert profile_summary["top_events"]
    assert "Top bottlenecks" in Path(generation["gpu_profile_bottleneck_report"]).read_text(encoding="utf-8")


def test_gpu_backend_writes_nsight_profile_hints(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-nsight",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cpu",
            gpu_dtype="float64",
            gpu_profile="nsight",
            gpu_profile_output=profile_dir,
            action_set="straight",
            observation_profile="base",
            max_steps=6,
            population=2,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=71,
            top_k=1,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress",),
        ),
        gates=EvolutionGates(target_progress_m=5.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    hints_path = Path(generation["gpu_profile_nsight_hints"])

    assert generation["gpu_profile"] == "nsight"
    assert hints_path.exists()
    assert "nsys profile" in hints_path.read_text(encoding="utf-8")


def test_gpu_unimplemented_speed_engines_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GPU fused engine"):
        run_evolution_search(
            output_dir=tmp_path / "gpu-fused",
            config=EvolutionSearchConfig(
                backend="gpu",
                gpu_device="cpu",
                gpu_engine="fused",
                action_set="straight",
                observation_profile="base",
                max_steps=4,
                population=2,
                generations=1,
                elite_count=1,
                random_immigrants=0,
                seed=59,
                top_k=1,
                workers=1,
                genome_type="controller",
            ),
            gates=EvolutionGates(target_progress_m=3.0),
            start_speed_kph=60.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU evolution smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_open_distance_backend_writes_verified_artifacts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-open",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=2,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=79,
            top_k=2,
            workers=1,
            genome_type="phase",
            scoring_profiles=("max_progress", "early_pace"),
        ),
        gates=EvolutionGates(target_progress_m=6.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert generation["backend"] == "gpu"
    assert generation["gpu_engine"] == "fused"
    assert generation["gpu_kernel_backend"] == "warp_open_step"
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_replay_count"] >= 2
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
    assert any(row.get("gpu_verified") is True for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["trace_count"] == 2
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU controller evolution smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_controller_backend_writes_verified_artifacts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-controller",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=2,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=89,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "clean_exit"),
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert all(row["genome"]["kind"] == "controller" for row in attempts)
    assert generation["backend"] == "gpu"
    assert generation["gpu_engine"] == "fused"
    assert generation["gpu_kernel_backend"] == "warp_persistent_controller_open"
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_replay_count"] >= 2
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
    assert any(row.get("gpu_verified") is True for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["trace_count"] == 2
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU V2 evolution smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_v2_controller_backend_uses_persistent_kernel_and_verifies(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-v2-controller",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=2,
            physics_model="v2",
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=109,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "clean_exit"),
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert generation["physics_model"] == "v2"
    assert generation["physics_version"].startswith("physics_v2.")
    assert generation["gpu_engine"] == "fused"
    assert generation["gpu_kernel_backend"] == "warp_persistent_controller_open"
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
    assert generation["gpu_cpu_top_replay_valid_lap_mismatches"] == 0
    assert any(row.get("physics_model") == "v2" for row in attempts)
    assert manifest["physics_model"] == "v2"
    assert manifest["trace_count"] == 2
    assert load_steps(Path(manifest["traces"][0]["path"]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU production-mode smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_production_mode_skips_default_replay_and_telemetry(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-production",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_run_mode="production",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=0,
            gpu_telemetry_mode="none",
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=91,
            top_k=2,
            workers=1,
            genome_type="controller",
            scoring_profiles=("max_progress", "clean_exit"),
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert generation["gpu_run_mode"] == "production"
    assert generation["gpu_attempts_mode"] == "compact"
    assert generation["gpu_kernel_backend"] == "warp_persistent_controller_open"
    assert generation["gpu_cpu_replay_count"] == 0
    assert generation["gpu_parity_status"] == "not_checked"
    assert generation["gpu_telemetry_mode"] == "none"
    assert generation["gpu_backend_total_seconds"] >= generation["gpu_rollout_seconds"]
    assert "gpu_control_program_upload_seconds" in generation
    assert "gpu_result_materialization_seconds" in generation
    assert "timing_attempts_jsonl_write_seconds" in generation
    assert len(attempts) == 4
    assert all(row.get("gpu_row_detail") == "compact" for row in attempts)
    assert all("final_row" not in row for row in attempts)
    assert all("genome" in row and "profile_scores" in row for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["gpu_telemetry_mode"] == "none"
    assert manifest["trace_count"] == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU progress-phase evolution smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_progress_phase_backend_writes_verified_artifacts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-progress-phase",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=2,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phase_progress_m=1.0,
            max_phase_progress_m=4.0,
            seed=97,
            top_k=2,
            workers=1,
            genome_type="progress_phase",
            scoring_profiles=("max_progress", "clean_exit"),
        ),
        gates=EvolutionGates(target_progress_m=506.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert all(row["genome"]["kind"] == "progress_phase" for row in attempts)
    assert generation["backend"] == "gpu"
    assert generation["gpu_engine"] == "fused"
    assert generation["gpu_kernel_backend"] == "warp_open_step"
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_replay_count"] >= 2
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
    assert any(row.get("gpu_verified") is True for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["trace_count"] == 2
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Fused GPU target evolution smoke requires CUDA and Warp CUDA support",
)
def test_gpu_fused_target_backend_writes_verified_artifacts(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "gpu-fused-target",
        config=EvolutionSearchConfig(
            backend="gpu",
            gpu_device="cuda",
            gpu_engine="fused",
            gpu_dtype="float32",
            gpu_static_batch_size=4,
            gpu_collision_mode="exact_grid",
            gpu_cpu_replay_top_k=2,
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=4,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            seed=83,
            top_k=2,
            workers=1,
            genome_type="phase",
            scoring_profiles=("max_progress", "clean_exit"),
        ),
        gates=EvolutionGates(target_progress_m=501.0, terminate_at_target_progress=True),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    attempts = _attempt_rows(output_dir / "attempts.jsonl")
    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation_summaries"][0]
    manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))

    assert len(attempts) == 4
    assert generation["backend"] == "gpu"
    assert generation["gpu_engine"] == "fused"
    assert generation["gpu_kernel_backend"] == "warp_open_step"
    assert generation["gpu_parity_status"] == "passed"
    assert generation["gpu_cpu_replay_count"] >= 2
    assert generation["gpu_cpu_top_replay_reason_mismatches"] == 0
    assert any(row.get("sim_segment_complete") is True for row in attempts)
    assert any(row.get("gpu_verified") is True for row in attempts)
    assert manifest["backend"] == "gpu"
    assert manifest["trace_count"] == 2
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))
