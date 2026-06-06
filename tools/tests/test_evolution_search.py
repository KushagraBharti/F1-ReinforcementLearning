import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from f1rl.evolution_search import (
    CONTROLLER_FEATURE_NAMES,
    EvolutionGates,
    EvolutionSearchConfig,
    Genome,
    PhaseGene,
    ProgressPhaseGene,
    _action_for_progress_delta,
    _controller_controls,
    _effective_gpu_cpu_replay_top_k,
    _max_steps_for_generation,
    _next_population,
    _parent_buckets,
    _parse_max_steps_schedule,
    _plateau_report,
    _reset_options,
    _score_rows,
    _survival_floor_report,
    genome_from_mapping,
    genome_to_dict,
    mutate_genome,
    mutate_genome_with_metadata,
    random_genome,
    run_evolution_search,
)
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import snapshot_from_sim
from f1rl.telemetry import load_steps


def test_random_and_mutated_phase_genomes_stay_valid() -> None:
    rng = np.random.default_rng(1)
    action_names = ["coast", "throttle", "brake"]
    genome = random_genome(
        rng,
        action_names=action_names,
        min_phases=2,
        max_phases=5,
        min_phase_steps=3,
        max_phase_steps=20,
    )
    mutated = mutate_genome(
        genome,
        rng,
        action_names=action_names,
        min_phase_steps=3,
        max_phase_steps=20,
        max_phases=5,
    )

    assert genome.kind == "phase"
    assert 1 <= len(genome.phases) <= 5
    assert 1 <= len(mutated.phases) <= 5
    assert {phase.action for phase in mutated.phases} <= set(action_names)
    assert all(3 <= phase.steps <= 20 for phase in mutated.phases)

    _, metadata = mutate_genome_with_metadata(
        genome,
        rng,
        action_names=action_names,
        min_phase_steps=3,
        max_phase_steps=20,
        max_phases=5,
    )
    assert metadata["mutation_type"].startswith("phase_")
    assert "reset_count" in metadata


def test_gpu_production_defaults_to_small_cpu_rerank_unless_explicitly_disabled() -> None:
    assert (
        _effective_gpu_cpu_replay_top_k(
            gpu_run_mode="production",
            gpu_verify_top_k=4,
            top_k=8,
            gpu_cpu_replay_top_k=None,
            explicit_gpu_cpu_replay_top_k=False,
        )
        == 16
    )
    assert (
        _effective_gpu_cpu_replay_top_k(
            gpu_run_mode="production",
            gpu_verify_top_k=12,
            top_k=4,
            gpu_cpu_replay_top_k=None,
            explicit_gpu_cpu_replay_top_k=False,
        )
        == 16
    )
    assert (
        _effective_gpu_cpu_replay_top_k(
            gpu_run_mode="production",
            gpu_verify_top_k=12,
            top_k=24,
            gpu_cpu_replay_top_k=None,
            explicit_gpu_cpu_replay_top_k=False,
        )
        == 48
    )
    assert (
        _effective_gpu_cpu_replay_top_k(
            gpu_run_mode="production",
            gpu_verify_top_k=4,
            top_k=8,
            gpu_cpu_replay_top_k=0,
            explicit_gpu_cpu_replay_top_k=True,
        )
        == 0
    )
    assert (
        _effective_gpu_cpu_replay_top_k(
            gpu_run_mode="parity",
            gpu_verify_top_k=4,
            top_k=8,
            gpu_cpu_replay_top_k=None,
            explicit_gpu_cpu_replay_top_k=False,
        )
        is None
    )


def test_progress_phase_genome_roundtrips_and_switches_by_distance() -> None:
    genome = Genome(
        kind="progress_phase",
        progress_phases=(
            ProgressPhaseGene(action="brake", progress_m=10.0),
            ProgressPhaseGene(action="coast", progress_m=5.0),
            ProgressPhaseGene(action="throttle", progress_m=20.0),
        ),
    )
    roundtrip = genome_from_mapping(genome_to_dict(genome))

    assert roundtrip.kind == "progress_phase"
    assert _action_for_progress_delta(roundtrip, 0.0) == "brake"
    assert _action_for_progress_delta(roundtrip, 11.0) == "coast"
    assert _action_for_progress_delta(roundtrip, 30.0) == "throttle"


def test_controller_genome_outputs_bounded_controls() -> None:
    weights = [0.0] * (len(CONTROLLER_FEATURE_NAMES) * 3)
    feature_index = {name: index for index, name in enumerate(CONTROLLER_FEATURE_NAMES)}
    weights[0] = 2.0
    weights[len(CONTROLLER_FEATURE_NAMES)] = -2.0
    weights[len(CONTROLLER_FEATURE_NAMES) * 2 + feature_index["target_steer"]] = 1.0
    genome = Genome(kind="controller", controller_weights=tuple(weights))
    features = {name: 0.0 for name in CONTROLLER_FEATURE_NAMES}
    features["bias"] = 1.0
    features["target_steer"] = 0.5

    throttle, brake, steer = _controller_controls(genome, features)

    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0
    assert -1.0 <= steer <= 1.0
    assert throttle > brake


def test_search_features_expose_general_braking_lookahead_context() -> None:
    sim = MonzaSim()
    sim.reset(seed=4, options={"start_progress_m": 5200.0, "start_speed_kph": 210.0})

    features = sim.search_features(segment_start_progress_m=0.0, segment_target_progress_m=5793.0)

    for key in (
        "future_brake_demand",
        "target_speed_drop_norm",
        "brake_gate_proximity",
        "brake_gate_distance_norm",
        "lookahead_abs_max",
        "braking_gate_distance_m",
    ):
        assert key in features
    assert 0.0 <= features["future_brake_demand"] <= 1.0
    assert 0.0 <= features["brake_gate_proximity"] <= 1.0


def test_max_steps_schedule_changes_budget_by_generation() -> None:
    schedule = _parse_max_steps_schedule("0:10000,5:15000,15:25000")
    config = EvolutionSearchConfig(max_steps=600, max_steps_schedule=schedule)

    assert _max_steps_for_generation(config, 0) == 10000
    assert _max_steps_for_generation(config, 4) == 10000
    assert _max_steps_for_generation(config, 5) == 15000
    assert _max_steps_for_generation(config, 14) == 15000
    assert _max_steps_for_generation(config, 15) == 25000


def test_scoring_profiles_return_distinct_finite_scores() -> None:
    rows = [
        {
            "monotonic_progress_m": 500.0,
            "sim_time_s": 1.0,
            "speed_kph": 320.0,
            "lateral_error_m": 1.0,
            "heading_error_deg": 2.0,
            "yaw_rate_rps": 0.1,
            "steering": 0.1,
            "throttle": 0.0,
            "brake": 1.0,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": "running",
        },
        {
            "monotonic_progress_m": 650.0,
            "sim_time_s": 2.0,
            "speed_kph": 210.0,
            "lateral_error_m": 2.0,
            "heading_error_deg": 5.0,
            "yaw_rate_rps": 0.2,
            "steering": 0.2,
            "throttle": 0.2,
            "brake": 0.0,
            "missed_checkpoint_count": 0,
            "segment_complete": True,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": "segment_complete",
        },
    ]

    scores = _score_rows(
        rows,
        start_progress_m=500.0,
        gates=EvolutionGates(target_progress_m=650.0),
        scoring_profiles=(
            "max_progress",
            "brake_zone",
            "risk_seeking",
            "frontier_fast",
            "early_pace",
            "clean_distance",
            "farthest_distance",
            "frontier_recovery",
            "frontier_novelty",
        ),
    )

    assert set(scores) == {
        "max_progress",
        "brake_zone",
        "risk_seeking",
        "frontier_fast",
        "early_pace",
        "clean_distance",
        "farthest_distance",
        "frontier_recovery",
        "frontier_novelty",
    }
    assert all(np.isfinite(value) for value in scores.values())
    assert scores["brake_zone"] != scores["risk_seeking"]


def test_early_pace_profile_rewards_fast_progress_without_steering_flip_penalty() -> None:
    def row(progress_m: float, sim_time_s: float, speed_kph: float, brake: float) -> dict:
        return {
            "monotonic_progress_m": progress_m,
            "sim_time_s": sim_time_s,
            "speed_kph": speed_kph,
            "lateral_error_m": 1.0,
            "heading_error_deg": 2.0,
            "yaw_rate_rps": 0.1,
            "steering": -0.9 if int(progress_m) % 2 else 0.9,
            "throttle": 0.6,
            "brake": brake,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": "running",
        }

    fast_rows = [row(150.0, 1.0, 180.0, 0.05), row(300.0, 2.0, 210.0, 0.05), row(450.0, 3.0, 240.0, 0.05)]
    slow_rows = [row(150.0, 10.0, 35.0, 0.50), row(300.0, 20.0, 45.0, 0.50), row(450.0, 30.0, 55.0, 0.50)]
    gates = EvolutionGates(target_progress_m=1500.0, terminate_at_target_progress=False)

    fast_score = _score_rows(
        fast_rows,
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("early_pace",),
    )["early_pace"]
    slow_score = _score_rows(
        slow_rows,
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("early_pace",),
    )["early_pace"]

    assert fast_score > slow_score


def test_speed_profiles_prefer_faster_valid_lap_over_slow_finish() -> None:
    def lap_rows(elapsed_s: float) -> list[dict]:
        return [
            {
                "monotonic_progress_m": 0.0,
                "sim_time_s": 0.0,
                "speed_kph": 80.0,
                "lateral_error_m": 0.0,
                "heading_error_deg": 0.0,
                "yaw_rate_rps": 0.0,
                "steering": 0.0,
                "throttle": 0.8,
                "brake": 0.0,
                "missed_checkpoint_count": 0,
                "segment_complete": False,
                "completed_lap": False,
                "valid_lap": True,
                "finish_crossed": False,
                "collided": False,
                "off_track": False,
                "termination_reason": "running",
            },
            {
                "monotonic_progress_m": 5812.0,
                "sim_time_s": elapsed_s,
                "speed_kph": 150.0,
                "lateral_error_m": 1.0,
                "heading_error_deg": 2.0,
                "yaw_rate_rps": 0.1,
                "steering": 0.1,
                "throttle": 0.8,
                "brake": 0.0,
                "missed_checkpoint_count": 0,
                "segment_complete": False,
                "completed_lap": False,
                "valid_lap": True,
                "finish_crossed": True,
                "collided": False,
                "off_track": False,
                "termination_reason": "lap_complete",
            },
        ]

    gates = EvolutionGates(target_progress_m=1500.0, terminate_at_target_progress=False)
    faster = _score_rows(
        lap_rows(120.0),
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("fast_valid_lap", "time_attack", "lap_pace"),
    )
    slower = _score_rows(
        lap_rows(180.0),
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("fast_valid_lap", "time_attack", "lap_pace"),
    )

    assert faster["fast_valid_lap"] > slower["fast_valid_lap"]
    assert faster["time_attack"] > slower["time_attack"]
    assert faster["lap_pace"] > slower["lap_pace"]


def test_frontier_recovery_prefers_alive_controlled_exit_over_farther_stall() -> None:
    def row(progress_m: float, speed_kph: float, heading_deg: float, lateral_m: float, reason: str) -> dict:
        return {
            "monotonic_progress_m": progress_m,
            "sim_time_s": progress_m / 70.0,
            "speed_kph": speed_kph,
            "lateral_error_m": lateral_m,
            "heading_error_deg": heading_deg,
            "yaw_rate_rps": 0.1,
            "steering": 0.2,
            "throttle": 0.4,
            "brake": 0.1,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": reason,
        }

    stalled = [row(2200.0, 150.0, -12.0, 7.0, "active"), row(2457.0, 0.0, -36.0, 14.5, "no_progress")]
    recovered = [row(2200.0, 170.0, -8.0, 6.0, "active"), row(2425.0, 175.0, -9.0, 7.5, "active")]
    gates = EvolutionGates(target_progress_m=1500.0, terminate_at_target_progress=False)

    stalled_score = _score_rows(
        stalled,
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("frontier_recovery",),
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )["frontier_recovery"]
    recovered_score = _score_rows(
        recovered,
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("frontier_recovery",),
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )["frontier_recovery"]

    assert recovered_score > stalled_score


def _fake_ranked_row(index: int, *, progress_m: float, pace_kph: float, clean_score: float) -> dict:
    return {
        "candidate_index": index,
        "generation": 0,
        "seed": 100 + index,
        "score": progress_m,
        "profile_scores": {"clean_distance": clean_score},
        "snapshot_index": 0,
        "start_progress_m": 0.0,
        "best_progress_m": progress_m,
        "final_progress_m": progress_m,
        "remaining_m": max(0.0, 1500.0 - progress_m),
        "segment_complete": False,
        "target_reached": False,
        "termination_reason": "collision",
        "pace_kph": pace_kph,
        "genome": genome_to_dict(Genome(phases=(PhaseGene(action="throttle", steps=4),))),
        "lineage": {"source": "unit"},
    }


def test_dynamic_survival_floor_and_parent_buckets_are_performance_based() -> None:
    ranked = [
        _fake_ranked_row(0, progress_m=1400.0, pace_kph=80.0, clean_score=10.0),
        _fake_ranked_row(1, progress_m=1100.0, pace_kph=220.0, clean_score=20.0),
        _fake_ranked_row(2, progress_m=700.0, pace_kph=240.0, clean_score=80.0),
        _fake_ranked_row(3, progress_m=80.0, pace_kph=20.0, clean_score=5.0),
    ]
    config = EvolutionSearchConfig(
        population=8,
        elite_count=2,
        parent_pool_size=4,
        scoring_profiles=("clean_distance",),
        survival_floor_stages_m=(450.0, 1000.0, 1220.0),
        survival_floor_pass_rate=0.50,
    )

    report = _survival_floor_report(ranked, config, active_index=0)
    buckets, summary = _parent_buckets(ranked, config, survival_floor_m=report["next_survival_floor_m"])

    assert report["survival_floor_m"] == 450.0
    assert report["survival_floor_pass_rate"] == 0.75
    assert report["next_survival_floor_m"] == 1220.0
    assert all(row["best_progress_m"] >= 1220.0 for row in buckets["survival_gate"])
    assert set(buckets) == {
        "survival_gate",
        "farthest_distance",
        "far_fast",
        "fastest_pace",
        "cleanest_distance",
        "fast_frontier_score",
    }
    assert summary["parent_survival_floor_m"] == 1220.0


def test_frontier_parent_bucket_keeps_rare_far_candidates_available() -> None:
    ranked = [
        _fake_ranked_row(0, progress_m=2450.0, pace_kph=190.0, clean_score=10.0),
        _fake_ranked_row(1, progress_m=2100.0, pace_kph=210.0, clean_score=20.0),
        _fake_ranked_row(2, progress_m=900.0, pace_kph=240.0, clean_score=80.0),
        _fake_ranked_row(3, progress_m=70.0, pace_kph=20.0, clean_score=5.0),
    ]
    for row in ranked:
        row["profile_scores"]["frontier_recovery"] = float(row["best_progress_m"])
    config = EvolutionSearchConfig(
        population=8,
        elite_count=2,
        parent_pool_size=5,
        scoring_profiles=("frontier_recovery", "clean_distance"),
        frontier_parent_min_progress_m=2000.0,
    )

    buckets, _ = _parent_buckets(ranked, config, survival_floor_m=1000.0)

    assert "frontier_distance" in buckets
    assert "late_frontier_distance" in buckets
    assert all(row["best_progress_m"] >= 2000.0 for row in buckets["frontier_distance"])


def test_plateau_report_activates_when_best_is_flat_and_average_improves() -> None:
    config = EvolutionSearchConfig(
        plateau_generations=3,
        plateau_distance_epsilon_m=8.0,
        plateau_average_improvement_m=50.0,
    )
    previous = [
        {"generation_best_distance_m": 2450.0, "generation_average_distance_m": 700.0},
        {"generation_best_distance_m": 2454.0, "generation_average_distance_m": 760.0},
    ]
    current = {"generation_best_distance_m": 2456.0, "generation_average_distance_m": 820.0}

    report = _plateau_report(previous, current, config)

    assert report["plateau_active"] is True
    assert report["plateau_best_range_m"] == 6.0
    assert report["plateau_average_gain_m"] == 120.0


def test_next_population_uses_more_offspring_smart_immigrants_and_lineage() -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 0.0, "start_speed_kph": 80.0})
    snapshot = snapshot_from_sim(sim, source="unit")
    ranked = [
        _fake_ranked_row(0, progress_m=1200.0, pace_kph=190.0, clean_score=100.0),
        _fake_ranked_row(1, progress_m=1000.0, pace_kph=230.0, clean_score=80.0),
        _fake_ranked_row(2, progress_m=800.0, pace_kph=240.0, clean_score=120.0),
        _fake_ranked_row(3, progress_m=50.0, pace_kph=20.0, clean_score=1.0),
    ]
    config = EvolutionSearchConfig(
        action_set="straight",
        population=8,
        elite_count=1,
        random_immigrants=4,
        min_random_immigrants=2,
        smart_immigrant_fraction=0.50,
        smart_immigrant_current_fraction=1.0,
        adaptive_immigrants=False,
        parent_pool_size=4,
        min_phases=1,
        max_phases=3,
        min_phase_steps=2,
        max_phase_steps=8,
        scoring_profiles=("clean_distance",),
    )

    population, summary = _next_population(
        ranked=ranked,
        best_rows=ranked,
        snapshots=[snapshot],
        config=config,
        rng=np.random.default_rng(99),
        action_names=["coast", "throttle", "brake"],
        created_generation=3,
        survival_floor_m=1000.0,
        quality_pass_rate=0.60,
    )

    sources = Counter(candidate.lineage.get("source") for candidate in population)
    assert len(population) == 8
    assert summary["offspring_count"] == 3
    assert summary["smart_immigrant_count"] == 2
    assert summary["pure_random_immigrant_count"] == 2
    assert sources["offspring"] == 3
    assert sources["smart_immigrant"] == 2
    assert sources["pure_random_immigrant"] == 2
    offspring = next(candidate for candidate in population if candidate.lineage.get("source") == "offspring")
    assert offspring.lineage["parent_generation"] == 0
    assert "source_bucket" in offspring.lineage
    assert "genome_distance_from_parent" in offspring.lineage


def test_plateau_mode_reduces_elites_and_adds_extra_frontier_mutations() -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 0.0, "start_speed_kph": 80.0})
    snapshot = snapshot_from_sim(sim, source="unit")
    ranked = [
        _fake_ranked_row(0, progress_m=2450.0, pace_kph=190.0, clean_score=100.0),
        _fake_ranked_row(1, progress_m=2300.0, pace_kph=220.0, clean_score=80.0),
        _fake_ranked_row(2, progress_m=2200.0, pace_kph=210.0, clean_score=120.0),
        _fake_ranked_row(3, progress_m=800.0, pace_kph=240.0, clean_score=1.0),
    ]
    for row in ranked:
        row["profile_scores"]["frontier_recovery"] = float(row["best_progress_m"])
    config = EvolutionSearchConfig(
        action_set="straight",
        population=8,
        elite_count=4,
        random_immigrants=0,
        plateau_elite_fraction=0.50,
        plateau_extra_mutations=1,
        parent_pool_size=6,
        min_phases=1,
        max_phases=3,
        min_phase_steps=2,
        max_phase_steps=8,
        scoring_profiles=("frontier_recovery", "clean_distance"),
    )

    population, summary = _next_population(
        ranked=ranked,
        best_rows=ranked,
        snapshots=[snapshot],
        config=config,
        rng=np.random.default_rng(123),
        action_names=["coast", "throttle", "brake"],
        created_generation=4,
        survival_floor_m=1000.0,
        quality_pass_rate=0.70,
        plateau={"plateau_active": True},
    )

    assert len(population) == 8
    assert summary["selected_elite_count"] == 4
    assert summary["elite_copy_count"] == 2
    assert summary["plateau_extra_mutation_count"] > 0
    assert any(candidate.lineage.get("plateau_mode") is True for candidate in population)


def test_no_target_termination_keeps_target_as_milestone_only() -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 0.0, "start_speed_kph": 80.0})
    snapshot = snapshot_from_sim(sim, source="unit")
    gates = EvolutionGates(target_progress_m=1500.0, terminate_at_target_progress=False)

    options = _reset_options(snapshot, gates, collect_observation=False)
    closed_options = _reset_options(
        snapshot,
        EvolutionGates(target_progress_m=1500.0),
        collect_observation=False,
    )

    assert "segment_length_m" not in options
    assert closed_options["segment_length_m"] == 1500.0


def test_open_distance_scoring_rewards_progress_past_milestone() -> None:
    def row(progress_m: float):
        return {
            "monotonic_progress_m": progress_m,
            "speed_kph": 250.0,
            "lateral_error_m": 4.0,
            "heading_error_deg": 5.0,
            "yaw_rate_rps": 0.2,
            "steering": 0.1,
            "throttle": 0.6,
            "brake": 0.0,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": "collision",
        }

    gates = EvolutionGates(target_progress_m=1500.0, terminate_at_target_progress=False)
    milestone_score = _score_rows(
        [row(1500.0)],
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("frontier",),
    )["frontier"]
    beyond_score = _score_rows(
        [row(1800.0)],
        start_progress_m=0.0,
        gates=gates,
        scoring_profiles=("frontier",),
    )["frontier"]

    assert beyond_score > milestone_score


def test_evolution_search_writes_streams_checkpoint_bridge_telemetry_and_elite_library(tmp_path: Path) -> None:
    sim = MonzaSim()
    sim.reset(seed=1, options={"start_progress_m": 500.0, "start_speed_kph": 80.0})
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [snapshot_from_sim(sim, source="unit")],
        source="unit",
    )

    output_dir = run_evolution_search(
        output_dir=tmp_path / "evolution",
        config=EvolutionSearchConfig(
            action_set="straight",
            observation_profile="base",
            max_steps=20,
            population=6,
            generations=2,
            elite_count=2,
            random_immigrants=1,
            min_phases=1,
            max_phases=3,
            min_phase_steps=2,
            max_phase_steps=8,
            seed=5,
            top_k=2,
            workers=1,
            scoring_profiles=("max_progress", "clean_exit"),
            checkpoint_every_generations=1,
        ),
        gates=EvolutionGates(target_progress_m=505.0, segment_fail_on_speed_gate_miss=True),
        state_library=library_path,
    )

    summary_path = output_dir / "evolution_summary.json"
    attempts_path = output_dir / "attempts.jsonl"
    elite_library_path = output_dir / "elite_state_library.json"

    assert summary_path.exists()
    assert attempts_path.exists()
    assert (output_dir / "generation_summary.jsonl").exists()
    assert (output_dir / "best_so_far.json").exists()
    assert (output_dir / "population_checkpoint.json").exists()
    assert (output_dir / "ppo_bridge.json").exists()
    assert (output_dir / "next_commands.md").exists()
    assert len(list((output_dir / "top_genomes").glob("*.json"))) == 2
    assert elite_library_path.exists()
    assert len(load_state_library(elite_library_path)) == 2
    assert len(list((output_dir / "selected_telemetry").glob("*.jsonl"))) == 2
    assert (output_dir / "selected_telemetry" / "manifest.json").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = json.loads((output_dir / "population_checkpoint.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "elitist_evolutionary_search"
    assert summary["complete"] is True
    assert summary["attempt_count"] == 12
    assert len(summary["generation_summaries"]) == 2
    assert len(summary["top_attempts"]) == 2
    assert "profile_scores" in summary["top_attempts"][0]
    assert all("lineage" in row for row in attempts)
    assert summary["generation_summaries"][0]["survival_floor_m"] == 450.0
    assert "lineage_source_counts" in summary["generation_summaries"][0]
    assert "next_population" in summary["generation_summaries"][0]
    assert "lineage" in checkpoint["population"][0]
    assert "--survival-floor-stages-m" in checkpoint["resume_command"]


def test_evolution_search_can_start_without_state_library_using_progress_phase(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "evolution-default-start",
        config=EvolutionSearchConfig(
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
            min_phase_progress_m=1.0,
            max_phase_progress_m=5.0,
            seed=8,
            top_k=1,
            workers=1,
            genome_type="progress_phase",
        ),
        gates=EvolutionGates(target_progress_m=4.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    assert summary["state_library"] is None
    assert summary["snapshot_count"] == 1
    assert summary["top_attempts"][0]["genome"]["kind"] == "progress_phase"


def test_evolution_search_can_resume_from_population_checkpoint(tmp_path: Path) -> None:
    config = EvolutionSearchConfig(
        action_set="straight",
        observation_profile="base",
        max_steps=8,
        population=4,
        generations=2,
        elite_count=1,
        random_immigrants=1,
        min_phases=1,
        max_phases=2,
        min_phase_steps=2,
        max_phase_steps=4,
        seed=11,
        top_k=1,
        workers=1,
    )
    gates = EvolutionGates(target_progress_m=5.0)
    output_dir = tmp_path / "resume"

    run_evolution_search(
        output_dir=output_dir,
        config=config,
        gates=gates,
        start_speed_kph=60.0,
        stop_after_generations=1,
    )
    partial_summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output_dir / "population_checkpoint.json").read_text(encoding="utf-8"))
    assert partial_summary["complete"] is False
    assert (output_dir / "population_checkpoint.json").exists()
    assert "--target-progress-m 5.0" in checkpoint["resume_command"]
    assert "--start-speed-kph 60.0" in checkpoint["resume_command"]
    assert checkpoint["config"]["action_set"] == "straight"
    assert "--telemetry-selection top" in checkpoint["resume_command"]

    run_evolution_search(
        output_dir=output_dir,
        config=config,
        gates=gates,
        start_speed_kph=60.0,
        resume=output_dir / "population_checkpoint.json",
    )
    final_summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    assert final_summary["complete"] is True
    assert final_summary["attempt_count"] == 8


def test_evolution_search_can_save_all_candidate_telemetry_for_swarm_replay(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "swarm",
        config=EvolutionSearchConfig(
            action_set="straight",
            observation_profile="base",
            max_steps=8,
            population=3,
            generations=2,
            elite_count=1,
            random_immigrants=1,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=23,
            top_k=1,
            workers=2,
            worker_chunk_size=1,
            telemetry_selection="all",
            telemetry_compression="gzip",
            all_candidate_telemetry_dir=tmp_path / "cold-telemetry",
        ),
        gates=EvolutionGates(target_progress_m=5.0),
        start_speed_kph=60.0,
    )

    telemetry_dir = output_dir / "selected_telemetry"
    cold_dir = tmp_path / "cold-telemetry"
    manifest = json.loads((telemetry_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["telemetry_selection"] == "all"
    assert manifest["telemetry_compression"] == "gzip"
    assert manifest["all_candidate_telemetry_dir"] == str(cold_dir)
    assert manifest["trace_count"] == 6
    assert len(list(cold_dir.glob("*.jsonl.gz"))) == 6
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
    assert load_steps(Path(manifest["traces"][0]["path"]))
    attempts = [
        json.loads(line)
        for line in (output_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(attempts) == 6
    assert all(Path(row["selected_telemetry"]).exists() for row in attempts)


def test_worker_chunked_evaluation_matches_serial_best(tmp_path: Path) -> None:
    config = EvolutionSearchConfig(
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
        seed=17,
        top_k=1,
        workers=1,
    )
    gates = EvolutionGates(target_progress_m=5.0)
    serial = run_evolution_search(
        output_dir=tmp_path / "serial",
        config=config,
        gates=gates,
        start_speed_kph=60.0,
    )
    parallel = run_evolution_search(
        output_dir=tmp_path / "parallel",
        config=EvolutionSearchConfig(**{**asdict(config), "workers": 2, "worker_chunk_size": 2}),
        gates=gates,
        start_speed_kph=60.0,
    )
    serial_summary = json.loads((serial / "evolution_summary.json").read_text(encoding="utf-8"))
    parallel_summary = json.loads((parallel / "evolution_summary.json").read_text(encoding="utf-8"))

    assert serial_summary["top_attempts"][0]["best_progress_m"] == parallel_summary["top_attempts"][0]["best_progress_m"]


def test_genome_constructor_example_is_plain_data() -> None:
    genome = Genome(phases=(PhaseGene(action="coast", steps=4), PhaseGene(action="brake", steps=3)))
    assert genome.kind == "phase"
    assert [phase.action for phase in genome.phases] == ["coast", "brake"]
