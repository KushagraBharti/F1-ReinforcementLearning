"""Elitist evolutionary search over short driving controllers.

This is deliberately separate from PPO. It searches phase-based action schedules,
keeps the best candidates, mutates/crosses them into the next generation, and
writes elite telemetry/state libraries for curriculum use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import (
    ARTIFACTS_DIR,
    MONZA_LENGTH_METERS,
    SimConfig,
    actions_for_action_set,
    dataclass_to_dict,
)
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import (
    StateSnapshot,
    snapshot_from_mapping,
    snapshot_from_sim,
    snapshot_to_dict,
)


@dataclass(frozen=True, slots=True)
class PhaseGene:
    action: str
    steps: int


@dataclass(frozen=True, slots=True)
class Genome:
    phases: tuple[PhaseGene, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    genome: Genome
    snapshot_index: int


@dataclass(frozen=True, slots=True)
class EvolutionGates:
    target_progress_m: float
    target_min_speed_kph: float | None = None
    target_max_speed_kph: float | None = None
    target_max_lateral_error_m: float | None = None
    target_max_heading_error_deg: float | None = None
    target_max_abs_yaw_rate_rps: float | None = None
    target_max_abs_steering: float | None = None
    segment_fail_on_speed_gate_miss: bool = True
    segment_require_release: bool = False
    segment_release_min_speed_kph: float = 135.0
    segment_release_max_speed_kph: float = 210.0
    segment_release_max_brake: float = 0.1
    segment_release_max_throttle: float = 0.1


@dataclass(frozen=True, slots=True)
class EvolutionSearchConfig:
    action_set: str = "racing"
    observation_profile: str = "racing_v2"
    max_steps: int = 600
    population: int = 256
    generations: int = 20
    elite_count: int = 16
    random_immigrants: int = 16
    min_phases: int = 2
    max_phases: int = 8
    min_phase_steps: int = 4
    max_phase_steps: int = 96
    mutation_rate: float = 0.75
    crossover_rate: float = 0.35
    start_mutation_rate: float = 0.10
    seed: int = 7
    top_k: int = 12
    workers: int = 1


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    return {"phases": [asdict(phase) for phase in genome.phases]}


def genome_from_mapping(value: dict[str, Any]) -> Genome:
    phases = tuple(
        PhaseGene(action=str(phase["action"]), steps=max(1, int(phase["steps"])))
        for phase in value["phases"]
    )
    return Genome(phases=phases)


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "snapshot_index": candidate.snapshot_index,
        "genome": genome_to_dict(candidate.genome),
    }


def _action_for_step(genome: Genome, step_index: int) -> str:
    elapsed = 0
    for phase in genome.phases:
        elapsed += phase.steps
        if step_index < elapsed:
            return phase.action
    return genome.phases[-1].action


def _normalize_genome(genome: Genome, *, action_names: list[str], min_steps: int, max_steps: int) -> Genome:
    valid_actions = set(action_names)
    phases: list[PhaseGene] = []
    for phase in genome.phases:
        action = phase.action if phase.action in valid_actions else action_names[0]
        steps = int(np.clip(phase.steps, min_steps, max_steps))
        if phases and phases[-1].action == action:
            previous = phases.pop()
            phases.append(PhaseGene(action=action, steps=int(np.clip(previous.steps + steps, min_steps, max_steps))))
        else:
            phases.append(PhaseGene(action=action, steps=steps))
    if not phases:
        phases.append(PhaseGene(action=action_names[0], steps=min_steps))
    return Genome(phases=tuple(phases))


def random_genome(
    rng: np.random.Generator,
    *,
    action_names: list[str],
    min_phases: int,
    max_phases: int,
    min_phase_steps: int,
    max_phase_steps: int,
) -> Genome:
    phase_count = int(rng.integers(min_phases, max_phases + 1))
    phases = tuple(
        PhaseGene(
            action=str(action_names[int(rng.integers(0, len(action_names)))]),
            steps=int(rng.integers(min_phase_steps, max_phase_steps + 1)),
        )
        for _ in range(phase_count)
    )
    return _normalize_genome(
        Genome(phases=phases),
        action_names=action_names,
        min_steps=min_phase_steps,
        max_steps=max_phase_steps,
    )


def mutate_genome(
    genome: Genome,
    rng: np.random.Generator,
    *,
    action_names: list[str],
    min_phase_steps: int,
    max_phase_steps: int,
    max_phases: int,
) -> Genome:
    phases = list(genome.phases)
    if not phases:
        return random_genome(
            rng,
            action_names=action_names,
            min_phases=1,
            max_phases=max_phases,
            min_phase_steps=min_phase_steps,
            max_phase_steps=max_phase_steps,
        )

    operation = str(rng.choice(["action", "duration", "insert", "delete", "swap"]))
    if operation == "action":
        index = int(rng.integers(0, len(phases)))
        phases[index] = PhaseGene(
            action=str(action_names[int(rng.integers(0, len(action_names)))]),
            steps=phases[index].steps,
        )
    elif operation == "duration":
        index = int(rng.integers(0, len(phases)))
        delta = int(rng.normal(0.0, max(2.0, (max_phase_steps - min_phase_steps) / 5.0)))
        phases[index] = PhaseGene(action=phases[index].action, steps=phases[index].steps + delta)
    elif operation == "insert" and len(phases) < max_phases:
        index = int(rng.integers(0, len(phases) + 1))
        phases.insert(
            index,
            PhaseGene(
                action=str(action_names[int(rng.integers(0, len(action_names)))]),
                steps=int(rng.integers(min_phase_steps, max_phase_steps + 1)),
            ),
        )
    elif operation == "delete" and len(phases) > 1:
        del phases[int(rng.integers(0, len(phases)))]
    elif operation == "swap" and len(phases) > 1:
        left = int(rng.integers(0, len(phases)))
        right = int(rng.integers(0, len(phases)))
        phases[left], phases[right] = phases[right], phases[left]

    return _normalize_genome(
        Genome(phases=tuple(phases)),
        action_names=action_names,
        min_steps=min_phase_steps,
        max_steps=max_phase_steps,
    )


def crossover_genomes(
    left: Genome,
    right: Genome,
    rng: np.random.Generator,
    *,
    action_names: list[str],
    min_phase_steps: int,
    max_phase_steps: int,
    max_phases: int,
) -> Genome:
    if not left.phases:
        return right
    if not right.phases:
        return left
    left_cut = int(rng.integers(1, len(left.phases) + 1))
    right_cut = int(rng.integers(0, len(right.phases)))
    phases = (*left.phases[:left_cut], *right.phases[right_cut:])
    if len(phases) > max_phases:
        phases = phases[:max_phases]
    return _normalize_genome(
        Genome(phases=tuple(phases)),
        action_names=action_names,
        min_steps=min_phase_steps,
        max_steps=max_phase_steps,
    )


def _default_start_snapshot(*, sim_config: SimConfig, seed: int, start_progress_m: float | None, start_speed_kph: float) -> StateSnapshot:
    sim = MonzaSim(sim_config)
    options: dict[str, Any] = {"start_speed_kph": start_speed_kph}
    if start_progress_m is not None:
        options["start_progress_m"] = start_progress_m
    sim.reset(seed=seed, options=options)
    return snapshot_from_sim(sim, source="evolution_start")


def _reset_options(snapshot: StateSnapshot, gates: EvolutionGates, *, collect_observation: bool) -> dict[str, Any]:
    return {
        "state_snapshot": snapshot_to_dict(snapshot),
        "segment_length_m": max(gates.target_progress_m - snapshot.monotonic_progress_m, 1.0),
        "segment_target_min_speed_kph": gates.target_min_speed_kph,
        "segment_target_max_speed_kph": gates.target_max_speed_kph,
        "segment_target_max_lateral_error_m": gates.target_max_lateral_error_m,
        "segment_target_max_heading_error_deg": gates.target_max_heading_error_deg,
        "segment_target_max_abs_yaw_rate_rps": gates.target_max_abs_yaw_rate_rps,
        "segment_target_max_abs_steering": gates.target_max_abs_steering,
        "segment_fail_on_speed_gate_miss": gates.segment_fail_on_speed_gate_miss,
        "segment_require_release": gates.segment_require_release,
        "segment_release_min_speed_kph": gates.segment_release_min_speed_kph,
        "segment_release_max_speed_kph": gates.segment_release_max_speed_kph,
        "segment_release_max_brake": gates.segment_release_max_brake,
        "segment_release_max_throttle": gates.segment_release_max_throttle,
        "curriculum_stage": "evolution-search",
        "collect_observation": collect_observation,
    }


def _score_rows(rows: list[dict[str, Any]], *, start_progress_m: float, gates: EvolutionGates) -> float:
    if not rows:
        return float("-inf")
    final = rows[-1]
    best_progress_m = max(float(row["monotonic_progress_m"]) for row in rows)
    progress_to_target_m = min(best_progress_m, gates.target_progress_m) - start_progress_m
    remaining_m = max(0.0, gates.target_progress_m - best_progress_m)
    score = progress_to_target_m * 20.0 - remaining_m * 35.0

    if final.get("segment_complete"):
        score += 100_000.0
    if final.get("completed_lap") or final.get("valid_lap") and final.get("finish_crossed"):
        score += 200_000.0

    reason = str(final.get("termination_reason", "unknown"))
    if reason in {"collision", "off_track", "assist_virtual_corridor"}:
        score -= 20_000.0
    elif reason.startswith("segment_") and reason != "segment_complete":
        score -= 7_500.0
    elif reason == "no_progress":
        score -= 3_500.0

    if not final.get("collided") and not final.get("off_track"):
        score += 1_500.0
    score += min(float(final.get("speed_kph", 0.0)), 320.0) * 5.0
    score -= abs(float(final.get("lateral_error_m", 0.0))) * 120.0
    score -= abs(float(final.get("heading_error_deg", 0.0))) * 35.0
    score -= abs(float(final.get("yaw_rate_rps", 0.0))) * 250.0
    score -= abs(float(final.get("steering", 0.0))) * 500.0
    score -= int(final.get("missed_checkpoint_count", 0)) * 2500.0
    return float(score)


def _compact_row(row: Any) -> dict[str, Any]:
    return {
        "monotonic_progress_m": row.monotonic_progress_m,
        "speed_kph": row.speed_kph,
        "lateral_error_m": row.lateral_error_m,
        "heading_error_deg": row.heading_error_deg,
        "yaw_rate_rps": row.yaw_rate_rps,
        "steering": row.steering,
        "missed_checkpoint_count": row.missed_checkpoint_count,
        "segment_complete": row.segment_complete,
        "completed_lap": bool(getattr(row, "completed_lap", False)),
        "valid_lap": row.valid_lap,
        "finish_crossed": row.finish_crossed,
        "collided": row.collided,
        "off_track": row.off_track,
        "termination_reason": row.termination_reason,
        "action_id": row.action_id,
        "action_name": row.action_name,
    }


def _run_candidate(
    *,
    sim_config: SimConfig,
    snapshot: StateSnapshot,
    genome: Genome,
    gates: EvolutionGates,
    seed: int,
    collect_full_telemetry: bool,
) -> tuple[list[dict[str, Any]], MonzaSim]:
    sim = MonzaSim(sim_config)
    sim.reset(seed=seed, options=_reset_options(snapshot, gates, collect_observation=collect_full_telemetry))
    action_specs = {
        name: (index, throttle, brake, steer)
        for index, (name, throttle, brake, steer) in enumerate(actions_for_action_set(sim_config.action_set))
    }
    rows: list[dict[str, Any]] = []
    for step_index in range(sim_config.max_steps):
        action_name = _action_for_step(genome, step_index)
        action_id, throttle, brake, steer = action_specs[action_name]
        result = sim.step_controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            action_id=action_id,
            collect_observation=collect_full_telemetry,
            collect_rays=collect_full_telemetry,
        )
        rows.append(asdict(result.telemetry) if collect_full_telemetry else _compact_row(result.telemetry))
        if result.terminated or result.truncated:
            break
    return rows, sim


def _evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    candidate = Candidate(
        genome=genome_from_mapping(task["genome"]),
        snapshot_index=int(task["snapshot_index"]),
    )
    snapshot = snapshot_from_mapping(task["snapshot"])
    rows, _ = _run_candidate(
        sim_config=task["sim_config"],
        snapshot=snapshot,
        genome=candidate.genome,
        gates=task["gates"],
        seed=int(task["seed"]),
        collect_full_telemetry=False,
    )
    score = _score_rows(rows, start_progress_m=snapshot.monotonic_progress_m, gates=task["gates"])
    final = rows[-1] if rows else {}
    best_progress_m = max([snapshot.monotonic_progress_m, *[float(row["monotonic_progress_m"]) for row in rows]])
    return {
        "candidate_index": int(task["candidate_index"]),
        "generation": int(task["generation"]),
        "seed": int(task["seed"]),
        "score": score,
        "snapshot_index": candidate.snapshot_index,
        "start_snapshot_id": snapshot.id,
        "start_progress_m": snapshot.monotonic_progress_m,
        "start_speed_kph": snapshot.speed_mps * 3.6,
        "genome": genome_to_dict(candidate.genome),
        "best_progress_m": best_progress_m,
        "final_progress_m": final.get("monotonic_progress_m", snapshot.monotonic_progress_m),
        "remaining_m": max(0.0, task["gates"].target_progress_m - best_progress_m),
        "segment_complete": bool(final.get("segment_complete", False)),
        "completed_lap": bool(final.get("completed_lap", False)),
        "valid_lap": bool(final.get("valid_lap", False)),
        "termination_reason": final.get("termination_reason", "empty"),
        "collided": bool(final.get("collided", False)),
        "off_track": bool(final.get("off_track", False)),
        "final_speed_kph": final.get("speed_kph"),
        "final_lateral_error_m": final.get("lateral_error_m"),
        "final_heading_error_deg": final.get("heading_error_deg"),
        "final_yaw_rate_rps": final.get("yaw_rate_rps"),
        "final_steering": final.get("steering"),
        "steps": len(rows),
    }


def _evaluate_population(
    *,
    candidates: list[Candidate],
    snapshots: list[StateSnapshot],
    sim_config: SimConfig,
    gates: EvolutionGates,
    generation: int,
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    tasks = [
        {
            "candidate_index": index,
            "generation": generation,
            "seed": seed + generation * 1_000_000 + index,
            "genome": genome_to_dict(candidate.genome),
            "snapshot_index": candidate.snapshot_index,
            "snapshot": snapshot_to_dict(snapshots[candidate.snapshot_index]),
            "sim_config": sim_config,
            "gates": gates,
        }
        for index, candidate in enumerate(candidates)
    ]
    if workers <= 1:
        return [_evaluate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate_task, tasks))


def _next_population(
    *,
    ranked: list[dict[str, Any]],
    snapshots: list[StateSnapshot],
    config: EvolutionSearchConfig,
    rng: np.random.Generator,
    action_names: list[str],
) -> list[Candidate]:
    elite_rows = ranked[: max(1, min(config.elite_count, len(ranked)))]
    elites = [
        Candidate(
            genome=genome_from_mapping(row["genome"]),
            snapshot_index=int(row["snapshot_index"]),
        )
        for row in elite_rows
    ]
    next_candidates = list(elites)
    immigrant_count = min(config.random_immigrants, max(0, config.population - len(next_candidates)))
    while len(next_candidates) < config.population - immigrant_count:
        parent = elites[int(rng.integers(0, len(elites)))]
        genome = parent.genome
        if len(elites) > 1 and rng.random() < config.crossover_rate:
            other = elites[int(rng.integers(0, len(elites)))]
            genome = crossover_genomes(
                genome,
                other.genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
            )
        if rng.random() < config.mutation_rate:
            genome = mutate_genome(
                genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
            )
        snapshot_index = parent.snapshot_index
        if len(snapshots) > 1 and rng.random() < config.start_mutation_rate:
            snapshot_index = int(rng.integers(0, len(snapshots)))
        next_candidates.append(Candidate(genome=genome, snapshot_index=snapshot_index))
    while len(next_candidates) < config.population:
        next_candidates.append(
            Candidate(
                genome=random_genome(
                    rng,
                    action_names=action_names,
                    min_phases=config.min_phases,
                    max_phases=config.max_phases,
                    min_phase_steps=config.min_phase_steps,
                    max_phase_steps=config.max_phase_steps,
                ),
                snapshot_index=int(rng.integers(0, len(snapshots))),
            )
        )
    return next_candidates


def _generation_summary(generation: int, ranked: list[dict[str, Any]]) -> dict[str, Any]:
    termination_reasons = Counter(str(row["termination_reason"]) for row in ranked)
    completion_count = sum(1 for row in ranked if row["segment_complete"])
    return {
        "generation": generation,
        "population": len(ranked),
        "best_score": ranked[0]["score"] if ranked else None,
        "best_progress_m": ranked[0]["best_progress_m"] if ranked else None,
        "best_remaining_m": ranked[0]["remaining_m"] if ranked else None,
        "completion_count": completion_count,
        "completion_rate": completion_count / max(1, len(ranked)),
        "termination_reasons": dict(termination_reasons),
        "best_attempt": ranked[0] if ranked else None,
    }


def run_evolution_search(
    *,
    output_dir: Path,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    state_library: Path | None = None,
    start_progress_m: float | None = None,
    start_speed_kph: float = 0.0,
    start_min_progress_m: float | None = None,
    start_max_progress_m: float | None = None,
) -> Path:
    action_names = [name for name, _, _, _ in actions_for_action_set(config.action_set)]
    sim_config = SimConfig(
        max_steps=config.max_steps,
        action_mode="discrete",
        action_set=config.action_set,
        observation_profile=config.observation_profile,
    )
    if state_library is None:
        snapshots = [
            _default_start_snapshot(
                sim_config=sim_config,
                seed=config.seed,
                start_progress_m=start_progress_m,
                start_speed_kph=start_speed_kph,
            )
        ]
    else:
        snapshots = load_state_library(state_library)
        if start_min_progress_m is not None:
            snapshots = [
                snapshot
                for snapshot in snapshots
                if snapshot.monotonic_progress_m + 1e-9 >= start_min_progress_m
            ]
        if start_max_progress_m is not None:
            snapshots = [
                snapshot
                for snapshot in snapshots
                if snapshot.monotonic_progress_m - 1e-9 <= start_max_progress_m
            ]
    if not snapshots:
        raise ValueError("No snapshots available for evolutionary search.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_telemetry"
    rng = np.random.default_rng(config.seed)
    population = [
        Candidate(
            genome=random_genome(
                rng,
                action_names=action_names,
                min_phases=config.min_phases,
                max_phases=config.max_phases,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
            ),
            snapshot_index=int(rng.integers(0, len(snapshots))),
        )
        for _ in range(config.population)
    ]

    attempts: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    latest_ranked: list[dict[str, Any]] = []
    for generation in range(config.generations):
        evaluated = _evaluate_population(
            candidates=population,
            snapshots=snapshots,
            sim_config=sim_config,
            gates=gates,
            generation=generation,
            seed=config.seed,
            workers=config.workers,
        )
        ranked = sorted(evaluated, key=lambda row: float(row["score"]), reverse=True)
        latest_ranked = ranked
        attempts.extend(ranked)
        generation_summaries.append(_generation_summary(generation, ranked))
        if generation < config.generations - 1:
            population = _next_population(
                ranked=ranked,
                snapshots=snapshots,
                config=config,
                rng=rng,
                action_names=action_names,
            )

    all_ranked = sorted(attempts, key=lambda row: float(row["score"]), reverse=True)
    top_rows = all_ranked[: config.top_k]
    elite_snapshots: list[StateSnapshot] = []
    for rank, row in enumerate(top_rows):
        snapshot = snapshots[int(row["snapshot_index"])]
        rows, sim = _run_candidate(
            sim_config=sim_config,
            snapshot=snapshot,
            genome=genome_from_mapping(row["genome"]),
            gates=gates,
            seed=int(row["seed"]),
            collect_full_telemetry=True,
        )
        telemetry_path = selected_dir / f"evolution-rank-{rank:03d}-gen-{row['generation']:03d}-candidate-{row['candidate_index']:05d}-steps.jsonl"
        _write_jsonl(telemetry_path, rows)
        row["selected_telemetry"] = str(telemetry_path)
        elite_snapshots.append(snapshot_from_sim(sim, source="evolution_search", source_file=str(telemetry_path)))

    _write_jsonl(output_dir / "attempts.jsonl", attempts)
    elite_library_path = output_dir / "elite_state_library.json"
    write_state_library(
        elite_library_path,
        elite_snapshots,
        source="evolution_search",
        metadata={
            "state_library": str(state_library) if state_library is not None else None,
            "start_progress_m": start_progress_m,
            "start_speed_kph": start_speed_kph,
            "start_min_progress_m": start_min_progress_m,
            "start_max_progress_m": start_max_progress_m,
            "config": asdict(config),
            "gates": asdict(gates),
        },
    )
    summary = {
        "run_id": output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "elitist_evolutionary_search",
        "sim_config": dataclass_to_dict(sim_config),
        "config": asdict(config),
        "gates": asdict(gates),
        "state_library": str(state_library) if state_library is not None else None,
        "snapshot_count": len(snapshots),
        "attempt_count": len(attempts),
        "elite_state_library": str(elite_library_path),
        "generation_summaries": generation_summaries,
        "latest_generation_top": latest_ranked[: config.top_k],
        "top_attempts": top_rows,
    }
    (output_dir / "evolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"evolution-search-{timestamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run elitist evolutionary search over phase-based driving controllers.")
    parser.add_argument("--state-library", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-progress-m", type=float)
    parser.add_argument("--start-speed-kph", type=float, default=0.0)
    parser.add_argument("--start-min-progress-m", type=float)
    parser.add_argument("--start-max-progress-m", type=float)
    parser.add_argument("--target-progress-m", type=float, default=MONZA_LENGTH_METERS)
    parser.add_argument("--target-min-speed-kph", type=float)
    parser.add_argument("--target-max-speed-kph", type=float)
    parser.add_argument("--target-max-lateral-error-m", type=float)
    parser.add_argument("--target-max-heading-error-deg", type=float)
    parser.add_argument("--target-max-abs-yaw-rate-rps", type=float)
    parser.add_argument("--target-max-abs-steering", type=float)
    parser.add_argument("--no-segment-fail-on-speed-gate-miss", action="store_true")
    parser.add_argument("--segment-require-release", action="store_true")
    parser.add_argument("--segment-release-min-speed-kph", type=float, default=135.0)
    parser.add_argument("--segment-release-max-speed-kph", type=float, default=210.0)
    parser.add_argument("--segment-release-max-brake", type=float, default=0.1)
    parser.add_argument("--segment-release-max-throttle", type=float, default=0.1)
    parser.add_argument("--action-set", default="racing")
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--random-immigrants", type=int, default=16)
    parser.add_argument("--min-phases", type=int, default=2)
    parser.add_argument("--max-phases", type=int, default=8)
    parser.add_argument("--min-phase-steps", type=int, default=4)
    parser.add_argument("--max-phase-steps", type=int, default=96)
    parser.add_argument("--mutation-rate", type=float, default=0.75)
    parser.add_argument("--crossover-rate", type=float, default=0.35)
    parser.add_argument("--start-mutation-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = EvolutionSearchConfig(
        action_set=args.action_set,
        observation_profile=args.observation_profile,
        max_steps=max(1, args.max_steps),
        population=max(1, args.population),
        generations=max(1, args.generations),
        elite_count=max(1, args.elite_count),
        random_immigrants=max(0, args.random_immigrants),
        min_phases=max(1, args.min_phases),
        max_phases=max(args.min_phases, args.max_phases),
        min_phase_steps=max(1, args.min_phase_steps),
        max_phase_steps=max(args.min_phase_steps, args.max_phase_steps),
        mutation_rate=float(np.clip(args.mutation_rate, 0.0, 1.0)),
        crossover_rate=float(np.clip(args.crossover_rate, 0.0, 1.0)),
        start_mutation_rate=float(np.clip(args.start_mutation_rate, 0.0, 1.0)),
        seed=args.seed,
        top_k=max(1, args.top_k),
        workers=max(1, args.workers),
    )
    gates = EvolutionGates(
        target_progress_m=args.target_progress_m,
        target_min_speed_kph=args.target_min_speed_kph,
        target_max_speed_kph=args.target_max_speed_kph,
        target_max_lateral_error_m=args.target_max_lateral_error_m,
        target_max_heading_error_deg=args.target_max_heading_error_deg,
        target_max_abs_yaw_rate_rps=args.target_max_abs_yaw_rate_rps,
        target_max_abs_steering=args.target_max_abs_steering,
        segment_fail_on_speed_gate_miss=not bool(args.no_segment_fail_on_speed_gate_miss),
        segment_require_release=bool(args.segment_require_release),
        segment_release_min_speed_kph=args.segment_release_min_speed_kph,
        segment_release_max_speed_kph=args.segment_release_max_speed_kph,
        segment_release_max_brake=args.segment_release_max_brake,
        segment_release_max_throttle=args.segment_release_max_throttle,
    )
    output_dir = args.output_dir or default_output_dir()
    run_root = run_evolution_search(
        output_dir=output_dir,
        config=config,
        gates=gates,
        state_library=args.state_library,
        start_progress_m=args.start_progress_m,
        start_speed_kph=args.start_speed_kph,
        start_min_progress_m=args.start_min_progress_m,
        start_max_progress_m=args.start_max_progress_m,
    )
    print(f"evolution_search_complete run={run_root} summary={run_root / 'evolution_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
