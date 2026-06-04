import json
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
    _reset_options,
    _score_rows,
    genome_from_mapping,
    genome_to_dict,
    mutate_genome,
    random_genome,
    run_evolution_search,
)
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import snapshot_from_sim


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
    weights[0] = 2.0
    weights[len(CONTROLLER_FEATURE_NAMES)] = -2.0
    weights[len(CONTROLLER_FEATURE_NAMES) * 2 + 9] = 1.0
    genome = Genome(kind="controller", controller_weights=tuple(weights))
    features = {name: 0.0 for name in CONTROLLER_FEATURE_NAMES}
    features["bias"] = 1.0
    features["target_steer"] = 0.5

    throttle, brake, steer = _controller_controls(genome, features)

    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0
    assert -1.0 <= steer <= 1.0
    assert throttle > brake


def test_scoring_profiles_return_distinct_finite_scores() -> None:
    rows = [
        {
            "monotonic_progress_m": 500.0,
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
        scoring_profiles=("max_progress", "brake_zone", "risk_seeking"),
    )

    assert set(scores) == {"max_progress", "brake_zone", "risk_seeking"}
    assert all(np.isfinite(value) for value in scores.values())
    assert scores["brake_zone"] != scores["risk_seeking"]


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
    assert summary["kind"] == "elitist_evolutionary_search"
    assert summary["complete"] is True
    assert summary["attempt_count"] == 12
    assert len(summary["generation_summaries"]) == 2
    assert len(summary["top_attempts"]) == 2
    assert "profile_scores" in summary["top_attempts"][0]


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
        ),
        gates=EvolutionGates(target_progress_m=5.0),
        start_speed_kph=60.0,
    )

    telemetry_dir = output_dir / "selected_telemetry"
    manifest = json.loads((telemetry_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["telemetry_selection"] == "all"
    assert manifest["trace_count"] == 6
    assert len(list(telemetry_dir.glob("*.jsonl"))) == 6
    assert all(Path(row["path"]).exists() for row in manifest["traces"])
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
