import json
from pathlib import Path

import numpy as np

from f1rl.evolution_search import (
    EvolutionGates,
    EvolutionSearchConfig,
    Genome,
    PhaseGene,
    mutate_genome,
    random_genome,
    run_evolution_search,
)
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import snapshot_from_sim


def test_random_and_mutated_genomes_stay_valid() -> None:
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

    assert 1 <= len(genome.phases) <= 5
    assert 1 <= len(mutated.phases) <= 5
    assert {phase.action for phase in mutated.phases} <= set(action_names)
    assert all(3 <= phase.steps <= 20 for phase in mutated.phases)


def test_evolution_search_writes_summary_attempts_telemetry_and_elite_library(tmp_path: Path) -> None:
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
        ),
        gates=EvolutionGates(target_progress_m=505.0, segment_fail_on_speed_gate_miss=True),
        state_library=library_path,
    )

    summary_path = output_dir / "evolution_summary.json"
    attempts_path = output_dir / "attempts.jsonl"
    elite_library_path = output_dir / "elite_state_library.json"

    assert summary_path.exists()
    assert attempts_path.exists()
    assert elite_library_path.exists()
    assert len(load_state_library(elite_library_path)) == 2
    assert len(list((output_dir / "selected_telemetry").glob("*.jsonl"))) == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["kind"] == "elitist_evolutionary_search"
    assert summary["attempt_count"] == 12
    assert len(summary["generation_summaries"]) == 2
    assert len(summary["top_attempts"]) == 2


def test_evolution_search_can_start_without_state_library(tmp_path: Path) -> None:
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
            seed=8,
            top_k=1,
            workers=1,
        ),
        gates=EvolutionGates(target_progress_m=4.0),
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    assert summary["state_library"] is None
    assert summary["snapshot_count"] == 1
    assert summary["top_attempts"][0]["genome"]["phases"]


def test_genome_constructor_example_is_plain_data() -> None:
    genome = Genome(phases=(PhaseGene(action="coast", steps=4), PhaseGene(action="brake", steps=3)))
    assert [phase.action for phase in genome.phases] == ["coast", "brake"]
