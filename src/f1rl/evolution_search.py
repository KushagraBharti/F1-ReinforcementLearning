"""Elitist evolutionary search for Monza driving behavior.

The search engine is intentionally separate from PPO. It brute-forces candidate
controllers, keeps useful elites, writes resumable artifacts, and produces state
libraries that PPO can later use as curriculum starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

GENOME_TYPES = frozenset({"phase", "progress_phase", "controller"})
TELEMETRY_SELECTIONS = frozenset({"top", "leaders", "all"})
SCORING_PROFILES = frozenset(
    {
        "frontier",
        "max_progress",
        "clean_exit",
        "brake_zone",
        "apex",
        "exit_speed",
        "full_lap_validity",
        "risk_seeking",
    }
)
CONTROLLER_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "speed_norm",
    "target_speed_norm",
    "speed_error_norm",
    "brake_demand",
    "signed_lateral_error_norm",
    "heading_error_norm",
    "yaw_rate_norm",
    "curvature_norm",
    "target_steer",
    "last_throttle",
    "last_brake",
    "last_steer",
    "segment_progress_ratio",
    "lookahead_0",
    "lookahead_1",
    "lookahead_2",
    "lookahead_3",
)
CONTROLLER_OUTPUT_COUNT = 3
CHECKPOINT_NAME = "population_checkpoint.json"


@dataclass(frozen=True, slots=True)
class PhaseGene:
    action: str
    steps: int


@dataclass(frozen=True, slots=True)
class ProgressPhaseGene:
    action: str
    progress_m: float


@dataclass(frozen=True, slots=True)
class Genome:
    phases: tuple[PhaseGene, ...] = ()
    kind: str = "phase"
    progress_phases: tuple[ProgressPhaseGene, ...] = ()
    controller_weights: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    genome: Genome
    snapshot_index: int


@dataclass(frozen=True, slots=True)
class EvolutionGates:
    target_progress_m: float
    terminate_at_target_progress: bool = True
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
    min_phase_progress_m: float = 5.0
    max_phase_progress_m: float = 160.0
    mutation_rate: float = 0.75
    crossover_rate: float = 0.35
    start_mutation_rate: float = 0.10
    seed: int = 7
    top_k: int = 12
    workers: int = 1
    worker_chunk_size: int = 0
    genome_type: str = "phase"
    scoring_profiles: tuple[str, ...] = ("max_progress",)
    telemetry_selection: str = "top"
    checkpoint_every_generations: int = 1
    progress_every_generation: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, default=_json_default) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, default=_json_default) + "\n")


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    last_row: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last_row = json.loads(line)
    return last_row


def _parse_csv(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_config(config: EvolutionSearchConfig) -> None:
    if config.genome_type not in GENOME_TYPES:
        valid = ", ".join(sorted(GENOME_TYPES))
        raise ValueError(f"Unknown genome type {config.genome_type!r}; expected one of: {valid}")
    if config.telemetry_selection not in TELEMETRY_SELECTIONS:
        valid = ", ".join(sorted(TELEMETRY_SELECTIONS))
        raise ValueError(f"Unknown telemetry selection {config.telemetry_selection!r}; expected one of: {valid}")
    unknown_profiles = set(config.scoring_profiles) - SCORING_PROFILES
    if unknown_profiles:
        valid = ", ".join(sorted(SCORING_PROFILES))
        unknown = ", ".join(sorted(unknown_profiles))
        raise ValueError(f"Unknown scoring profile(s): {unknown}; expected one or more of: {valid}")


def _controller_weight_count() -> int:
    return len(CONTROLLER_FEATURE_NAMES) * CONTROLLER_OUTPUT_COUNT


def genome_to_dict(genome: Genome) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": genome.kind}
    if genome.kind == "progress_phase":
        payload["progress_phases"] = [asdict(phase) for phase in genome.progress_phases]
    elif genome.kind == "controller":
        payload["controller_weights"] = [float(value) for value in genome.controller_weights]
        payload["controller_features"] = list(CONTROLLER_FEATURE_NAMES)
    else:
        payload["phases"] = [asdict(phase) for phase in genome.phases]
    return payload


def genome_from_mapping(value: dict[str, Any]) -> Genome:
    kind = str(value.get("kind", "phase"))
    if kind == "progress_phase":
        progress_phases = tuple(
            ProgressPhaseGene(
                action=str(phase["action"]),
                progress_m=max(1e-3, float(phase["progress_m"])),
            )
            for phase in value.get("progress_phases", ())
        )
        return Genome(kind=kind, progress_phases=progress_phases)
    if kind == "controller":
        weights = tuple(float(item) for item in value.get("controller_weights", ()))
        return _normalize_genome(Genome(kind=kind, controller_weights=weights), action_names=[])
    phases = tuple(
        PhaseGene(action=str(phase["action"]), steps=max(1, int(phase["steps"])))
        for phase in value.get("phases", ())
    )
    return Genome(kind="phase", phases=phases)


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "snapshot_index": candidate.snapshot_index,
        "genome": genome_to_dict(candidate.genome),
    }


def candidate_from_mapping(value: dict[str, Any]) -> Candidate:
    return Candidate(
        genome=genome_from_mapping(value["genome"]),
        snapshot_index=int(value["snapshot_index"]),
    )


def _normalize_phase_genome(
    genome: Genome,
    *,
    action_names: list[str],
    min_steps: int,
    max_steps: int,
) -> Genome:
    valid_actions = set(action_names)
    phases: list[PhaseGene] = []
    for phase in genome.phases:
        action = phase.action if phase.action in valid_actions else action_names[0]
        steps = int(np.clip(phase.steps, min_steps, max_steps))
        if phases and phases[-1].action == action:
            previous = phases.pop()
            phases.append(
                PhaseGene(
                    action=action,
                    steps=int(np.clip(previous.steps + steps, min_steps, max_steps)),
                )
            )
        else:
            phases.append(PhaseGene(action=action, steps=steps))
    if not phases and action_names:
        phases.append(PhaseGene(action=action_names[0], steps=min_steps))
    return Genome(kind="phase", phases=tuple(phases))


def _normalize_progress_genome(
    genome: Genome,
    *,
    action_names: list[str],
    min_progress_m: float,
    max_progress_m: float,
) -> Genome:
    valid_actions = set(action_names)
    phases: list[ProgressPhaseGene] = []
    for phase in genome.progress_phases:
        action = phase.action if phase.action in valid_actions else action_names[0]
        progress_m = float(np.clip(phase.progress_m, min_progress_m, max_progress_m))
        if phases and phases[-1].action == action:
            previous = phases.pop()
            phases.append(
                ProgressPhaseGene(
                    action=action,
                    progress_m=float(np.clip(previous.progress_m + progress_m, min_progress_m, max_progress_m)),
                )
            )
        else:
            phases.append(ProgressPhaseGene(action=action, progress_m=progress_m))
    if not phases and action_names:
        phases.append(ProgressPhaseGene(action=action_names[0], progress_m=min_progress_m))
    return Genome(kind="progress_phase", progress_phases=tuple(phases))


def _normalize_controller_genome(genome: Genome) -> Genome:
    weights = list(genome.controller_weights)
    target_count = _controller_weight_count()
    if len(weights) < target_count:
        weights.extend([0.0] * (target_count - len(weights)))
    if len(weights) > target_count:
        weights = weights[:target_count]
    clipped = tuple(float(np.clip(value, -6.0, 6.0)) for value in weights)
    return Genome(kind="controller", controller_weights=clipped)


def _normalize_genome(
    genome: Genome,
    *,
    action_names: list[str],
    min_steps: int = 1,
    max_steps: int = 96,
    min_progress_m: float = 5.0,
    max_progress_m: float = 160.0,
) -> Genome:
    if genome.kind == "progress_phase":
        return _normalize_progress_genome(
            genome,
            action_names=action_names,
            min_progress_m=min_progress_m,
            max_progress_m=max_progress_m,
        )
    if genome.kind == "controller":
        return _normalize_controller_genome(genome)
    return _normalize_phase_genome(
        Genome(kind="phase", phases=genome.phases),
        action_names=action_names,
        min_steps=min_steps,
        max_steps=max_steps,
    )


def random_genome(
    rng: np.random.Generator,
    *,
    action_names: list[str],
    min_phases: int,
    max_phases: int,
    min_phase_steps: int,
    max_phase_steps: int,
    genome_type: str = "phase",
    min_phase_progress_m: float = 5.0,
    max_phase_progress_m: float = 160.0,
) -> Genome:
    if genome_type == "controller":
        weights = rng.normal(0.0, 0.75, _controller_weight_count())
        feature_count = len(CONTROLLER_FEATURE_NAMES)
        weights[0] += 1.0
        weights[feature_count] -= 1.0
        return _normalize_controller_genome(Genome(kind="controller", controller_weights=tuple(weights)))
    phase_count = int(rng.integers(min_phases, max_phases + 1))
    if genome_type == "progress_phase":
        progress_phases = tuple(
            ProgressPhaseGene(
                action=str(action_names[int(rng.integers(0, len(action_names)))]),
                progress_m=float(rng.uniform(min_phase_progress_m, max_phase_progress_m)),
            )
            for _ in range(phase_count)
        )
        return _normalize_progress_genome(
            Genome(kind="progress_phase", progress_phases=progress_phases),
            action_names=action_names,
            min_progress_m=min_phase_progress_m,
            max_progress_m=max_phase_progress_m,
        )
    phases = tuple(
        PhaseGene(
            action=str(action_names[int(rng.integers(0, len(action_names)))]),
            steps=int(rng.integers(min_phase_steps, max_phase_steps + 1)),
        )
        for _ in range(phase_count)
    )
    return _normalize_phase_genome(
        Genome(kind="phase", phases=phases),
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
    min_phase_progress_m: float = 5.0,
    max_phase_progress_m: float = 160.0,
) -> Genome:
    if genome.kind == "controller":
        weights = np.asarray(genome.controller_weights, dtype=np.float64).copy()
        if len(weights) == 0:
            return random_genome(
                rng,
                action_names=action_names,
                min_phases=1,
                max_phases=max_phases,
                min_phase_steps=min_phase_steps,
                max_phase_steps=max_phase_steps,
                genome_type="controller",
            )
        roll = float(rng.random())
        if roll < 0.52:
            mask = rng.random(len(weights)) < 0.42
            if not bool(mask.any()):
                mask[int(rng.integers(0, len(weights)))] = True
            weights[mask] += rng.normal(0.0, 0.58, int(mask.sum()))
        elif roll < 0.82:
            weights += rng.normal(0.0, 0.34, len(weights))
        else:
            reset_count = int(rng.integers(3, min(9, len(weights) + 1)))
            indices = rng.choice(len(weights), size=reset_count, replace=False)
            weights[indices] = rng.normal(0.0, 2.05, reset_count)
        return _normalize_controller_genome(Genome(kind="controller", controller_weights=tuple(weights)))

    if genome.kind == "progress_phase":
        phases = list(genome.progress_phases)
        if not phases:
            return random_genome(
                rng,
                action_names=action_names,
                min_phases=1,
                max_phases=max_phases,
                min_phase_steps=min_phase_steps,
                max_phase_steps=max_phase_steps,
                genome_type="progress_phase",
                min_phase_progress_m=min_phase_progress_m,
                max_phase_progress_m=max_phase_progress_m,
            )
        operation = str(rng.choice(["action", "duration", "insert", "delete", "swap"]))
        if operation == "action":
            index = int(rng.integers(0, len(phases)))
            phases[index] = ProgressPhaseGene(
                action=str(action_names[int(rng.integers(0, len(action_names)))]),
                progress_m=phases[index].progress_m,
            )
        elif operation == "duration":
            index = int(rng.integers(0, len(phases)))
            delta = float(rng.normal(0.0, max(2.0, (max_phase_progress_m - min_phase_progress_m) / 6.0)))
            phases[index] = ProgressPhaseGene(
                action=phases[index].action,
                progress_m=phases[index].progress_m + delta,
            )
        elif operation == "insert" and len(phases) < max_phases:
            index = int(rng.integers(0, len(phases) + 1))
            phases.insert(
                index,
                ProgressPhaseGene(
                    action=str(action_names[int(rng.integers(0, len(action_names)))]),
                    progress_m=float(rng.uniform(min_phase_progress_m, max_phase_progress_m)),
                ),
            )
        elif operation == "delete" and len(phases) > 1:
            del phases[int(rng.integers(0, len(phases)))]
        elif operation == "swap" and len(phases) > 1:
            left = int(rng.integers(0, len(phases)))
            right = int(rng.integers(0, len(phases)))
            phases[left], phases[right] = phases[right], phases[left]
        return _normalize_progress_genome(
            Genome(kind="progress_phase", progress_phases=tuple(phases)),
            action_names=action_names,
            min_progress_m=min_phase_progress_m,
            max_progress_m=max_phase_progress_m,
        )

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

    return _normalize_phase_genome(
        Genome(kind="phase", phases=tuple(phases)),
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
    min_phase_progress_m: float = 5.0,
    max_phase_progress_m: float = 160.0,
) -> Genome:
    if left.kind != right.kind:
        return left if rng.random() < 0.5 else right
    if left.kind == "controller":
        left_weights = np.asarray(left.controller_weights, dtype=np.float64)
        right_weights = np.asarray(right.controller_weights, dtype=np.float64)
        if len(left_weights) != len(right_weights):
            return left
        if rng.random() < 0.55:
            mask = rng.random(len(left_weights)) < 0.5
            weights = np.where(mask, left_weights, right_weights)
        else:
            alpha = rng.uniform(0.25, 0.75, len(left_weights))
            weights = left_weights * alpha + right_weights * (1.0 - alpha)
            weights += rng.normal(0.0, 0.08, len(weights))
        return _normalize_controller_genome(Genome(kind="controller", controller_weights=tuple(weights)))
    if left.kind == "progress_phase":
        if not left.progress_phases:
            return right
        if not right.progress_phases:
            return left
        left_cut = int(rng.integers(1, len(left.progress_phases) + 1))
        right_cut = int(rng.integers(0, len(right.progress_phases)))
        phases = (*left.progress_phases[:left_cut], *right.progress_phases[right_cut:])
        if len(phases) > max_phases:
            phases = phases[:max_phases]
        return _normalize_progress_genome(
            Genome(kind="progress_phase", progress_phases=tuple(phases)),
            action_names=action_names,
            min_progress_m=min_phase_progress_m,
            max_progress_m=max_phase_progress_m,
        )
    if not left.phases:
        return right
    if not right.phases:
        return left
    left_cut = int(rng.integers(1, len(left.phases) + 1))
    right_cut = int(rng.integers(0, len(right.phases)))
    phases = (*left.phases[:left_cut], *right.phases[right_cut:])
    if len(phases) > max_phases:
        phases = phases[:max_phases]
    return _normalize_phase_genome(
        Genome(kind="phase", phases=tuple(phases)),
        action_names=action_names,
        min_steps=min_phase_steps,
        max_steps=max_phase_steps,
    )


def _action_for_step(genome: Genome, step_index: int) -> str:
    elapsed = 0
    for phase in genome.phases:
        elapsed += phase.steps
        if step_index < elapsed:
            return phase.action
    return genome.phases[-1].action


def _action_for_progress_delta(genome: Genome, progress_delta_m: float) -> str:
    elapsed = 0.0
    for phase in genome.progress_phases:
        elapsed += phase.progress_m
        if progress_delta_m < elapsed:
            return phase.action
    return genome.progress_phases[-1].action


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-value))


def _controller_feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in CONTROLLER_FEATURE_NAMES], dtype=np.float64)


def _controller_controls(genome: Genome, features: dict[str, float]) -> tuple[float, float, float]:
    weights = np.asarray(genome.controller_weights, dtype=np.float64)
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    if len(weights) != feature_count * CONTROLLER_OUTPUT_COUNT:
        genome = _normalize_controller_genome(genome)
        weights = np.asarray(genome.controller_weights, dtype=np.float64)
    matrix = weights.reshape((CONTROLLER_OUTPUT_COUNT, feature_count))
    logits = matrix @ _controller_feature_vector(features)
    throttle_raw = _sigmoid(float(logits[0]))
    brake_raw = _sigmoid(float(logits[1]))
    if brake_raw > throttle_raw:
        throttle = throttle_raw * (1.0 - brake_raw)
        brake = brake_raw
    else:
        throttle = throttle_raw
        brake = brake_raw * (1.0 - throttle_raw)
    steer = math.tanh(float(logits[2]))
    return (
        float(np.clip(throttle, 0.0, 1.0)),
        float(np.clip(brake, 0.0, 1.0)),
        float(np.clip(steer, -1.0, 1.0)),
    )


def _default_start_snapshot(
    *,
    sim_config: SimConfig,
    seed: int,
    start_progress_m: float | None,
    start_speed_kph: float,
) -> StateSnapshot:
    sim = MonzaSim(sim_config)
    options: dict[str, Any] = {"start_speed_kph": start_speed_kph}
    if start_progress_m is not None:
        options["start_progress_m"] = start_progress_m
    sim.reset(seed=seed, options=options)
    return snapshot_from_sim(sim, source="evolution_start")


def _reset_options(snapshot: StateSnapshot, gates: EvolutionGates, *, collect_observation: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        "state_snapshot": snapshot_to_dict(snapshot),
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
    if gates.terminate_at_target_progress:
        options["segment_length_m"] = max(gates.target_progress_m - snapshot.monotonic_progress_m, 1.0)
    return options


def _score_profile(
    rows: list[dict[str, Any]],
    *,
    start_progress_m: float,
    gates: EvolutionGates,
    profile: str,
) -> float:
    if not rows:
        return float("-inf")
    final = rows[-1]
    best_progress_m = max(float(row["monotonic_progress_m"]) for row in rows)
    raw_progress_m = best_progress_m - start_progress_m
    if gates.terminate_at_target_progress:
        progress_to_target_m = min(best_progress_m, gates.target_progress_m) - start_progress_m
    else:
        progress_to_target_m = raw_progress_m
    remaining_m = max(0.0, gates.target_progress_m - best_progress_m)
    final_speed_kph = float(final.get("speed_kph", 0.0) or 0.0)
    final_lateral_error_m = abs(float(final.get("lateral_error_m", 0.0) or 0.0))
    final_heading_error_deg = abs(float(final.get("heading_error_deg", 0.0) or 0.0))
    final_yaw_rate_rps = abs(float(final.get("yaw_rate_rps", 0.0) or 0.0))
    final_steering = abs(float(final.get("steering", 0.0) or 0.0))
    missed_checkpoints = int(final.get("missed_checkpoint_count", 0) or 0)
    clean = not final.get("collided") and not final.get("off_track")
    milestone_complete = best_progress_m >= gates.target_progress_m
    segment_complete = bool(final.get("segment_complete", False)) or milestone_complete
    valid_finish = bool(final.get("completed_lap") or (final.get("valid_lap") and final.get("finish_crossed")))
    reason = str(final.get("termination_reason", "unknown"))
    target_span_m = max(gates.target_progress_m - start_progress_m, 1.0)
    progress_ratio = float(np.clip(min(raw_progress_m, target_span_m) / target_span_m, 0.0, 1.0))
    frontier_m = max(0.0, progress_to_target_m - target_span_m * 0.70)
    near_target_m = max(0.0, progress_to_target_m - target_span_m * 0.88)
    beyond_target_m = max(0.0, best_progress_m - gates.target_progress_m)
    stalled = reason == "max_steps" and final_speed_kph < 20.0

    if profile == "frontier":
        collision_penalty = 4_500.0
    elif profile == "risk_seeking":
        collision_penalty = 7_000.0
    else:
        collision_penalty = 24_000.0
    score = progress_to_target_m * 34.0 - remaining_m * 36.0
    score += frontier_m * 65.0 + near_target_m * 120.0 + beyond_target_m * 160.0
    if segment_complete:
        score += 160_000.0
    if valid_finish:
        score += 300_000.0 if profile == "full_lap_validity" else 240_000.0
    if reason in {"collision", "off_track", "assist_virtual_corridor"}:
        score -= collision_penalty
    elif reason.startswith("segment_") and reason != "segment_complete":
        score -= 7_500.0
    elif reason == "no_progress":
        score -= 3_500.0
    if stalled:
        score -= 6_000.0
    if clean:
        score += 3_500.0 + progress_ratio * 3_000.0

    if profile == "frontier":
        score += best_progress_m * 12.0 + min(final_speed_kph, 360.0) * 18.0
        score += frontier_m * 95.0 + near_target_m * 210.0 + beyond_target_m * 320.0
        score -= final_lateral_error_m * 72.0
        score -= final_heading_error_deg * 26.0
        score -= final_yaw_rate_rps * 330.0
        score -= final_steering * 300.0
        score -= missed_checkpoints * 2_500.0
        return float(score)

    if profile == "risk_seeking":
        score += best_progress_m * 9.0 + min(final_speed_kph, 360.0) * 18.0
        score += frontier_m * 70.0 + near_target_m * 150.0 + beyond_target_m * 220.0
        score -= final_lateral_error_m * 52.0
        score -= final_heading_error_deg * 18.0
        return float(score)

    if profile == "clean_exit":
        score += min(final_speed_kph, 320.0) * 4.0
        score -= final_lateral_error_m * 260.0
        score -= final_heading_error_deg * 95.0
        score -= final_yaw_rate_rps * 600.0
        score -= final_steering * 800.0
        score -= missed_checkpoints * 5_000.0
        return float(score)

    if profile == "brake_zone":
        start_speed_kph = float(rows[0].get("speed_kph", final_speed_kph) or final_speed_kph)
        max_brake = max(float(row.get("brake", 0.0) or 0.0) for row in rows)
        avg_throttle = sum(float(row.get("throttle", 0.0) or 0.0) for row in rows) / max(len(rows), 1)
        speed_drop = max(0.0, start_speed_kph - final_speed_kph)
        score += max_brake * 18_000.0 + speed_drop * 120.0
        score -= avg_throttle * 4_000.0
        score -= final_lateral_error_m * 110.0
        return float(score)

    if profile == "apex":
        score += min(final_speed_kph, 240.0) * 3.0
        score -= final_lateral_error_m * 340.0
        score -= final_heading_error_deg * 130.0
        score -= final_yaw_rate_rps * 850.0
        score -= missed_checkpoints * 5_000.0
        return float(score)

    if profile == "exit_speed":
        score += min(final_speed_kph, 340.0) * 18.0
        score -= final_lateral_error_m * 150.0
        score -= final_heading_error_deg * 45.0
        score -= final_steering * 500.0
        return float(score)

    if profile == "full_lap_validity":
        score += best_progress_m * 5.0
        score += 10_000.0 if bool(final.get("valid_lap", False)) else -10_000.0
        score -= missed_checkpoints * 10_000.0
        score -= final_lateral_error_m * 120.0
        score -= final_heading_error_deg * 40.0
        return float(score)

    score += min(final_speed_kph, 320.0) * 5.0
    score -= final_lateral_error_m * 120.0
    score -= final_heading_error_deg * 35.0
    score -= final_yaw_rate_rps * 250.0
    score -= final_steering * 500.0
    score -= missed_checkpoints * 2_500.0
    return float(score)


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    start_progress_m: float,
    gates: EvolutionGates,
    scoring_profiles: tuple[str, ...] = ("max_progress",),
) -> dict[str, float]:
    return {
        profile: _score_profile(rows, start_progress_m=start_progress_m, gates=gates, profile=profile)
        for profile in scoring_profiles
    }


def _compact_row(row: Any) -> dict[str, Any]:
    return {
        "monotonic_progress_m": row.monotonic_progress_m,
        "progress_delta_m": row.progress_delta_m,
        "speed_kph": row.speed_kph,
        "lateral_error_m": row.lateral_error_m,
        "heading_error_deg": row.heading_error_deg,
        "yaw_rate_rps": row.yaw_rate_rps,
        "curvature_rad_per_m": row.curvature_rad_per_m,
        "throttle": row.throttle,
        "brake": row.brake,
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
    keep_step_telemetry: bool = False,
    stream_telemetry_path: Path | None = None,
    sim: MonzaSim | None = None,
) -> tuple[list[dict[str, Any]], MonzaSim]:
    active_sim = sim or MonzaSim(sim_config)
    active_sim.reset(seed=seed, options=_reset_options(snapshot, gates, collect_observation=collect_full_telemetry))
    action_specs = {
        name: (index, throttle, brake, steer)
        for index, (name, throttle, brake, steer) in enumerate(actions_for_action_set(sim_config.action_set))
    }
    rows: list[dict[str, Any]] = []
    start_progress_m = snapshot.monotonic_progress_m
    stream_file = None
    tmp_stream_path: Path | None = None
    completed = False
    try:
        if stream_telemetry_path is not None:
            stream_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_stream_path = stream_telemetry_path.with_name(f"{stream_telemetry_path.name}.tmp")
            stream_file = tmp_stream_path.open("w", encoding="utf-8")
        for step_index in range(sim_config.max_steps):
            if genome.kind == "controller":
                features = active_sim.search_features(
                    segment_start_progress_m=start_progress_m,
                    segment_target_progress_m=gates.target_progress_m,
                )
                throttle, brake, steer = _controller_controls(genome, features)
                action_id = -2
            elif genome.kind == "progress_phase":
                progress_delta_m = max(0.0, active_sim.state.monotonic_progress_m - start_progress_m)
                action_name = _action_for_progress_delta(genome, progress_delta_m)
                action_id, throttle, brake, steer = action_specs[action_name]
            else:
                action_name = _action_for_step(genome, step_index)
                action_id, throttle, brake, steer = action_specs[action_name]
            result = active_sim.step_controls(
                throttle=throttle,
                brake=brake,
                steer=steer,
                action_id=action_id,
                collect_observation=collect_full_telemetry,
                collect_rays=collect_full_telemetry,
                compute_reward=collect_full_telemetry,
            )
            if stream_file is not None or collect_full_telemetry or keep_step_telemetry:
                full_row = asdict(result.telemetry)
                if stream_file is not None:
                    stream_file.write(json.dumps(full_row, default=_json_default) + "\n")
                if collect_full_telemetry or keep_step_telemetry:
                    rows.append(full_row)
                else:
                    rows.append(_compact_row(result.telemetry))
            else:
                rows.append(_compact_row(result.telemetry))
            if result.terminated or result.truncated:
                break
        completed = True
    finally:
        if stream_file is not None:
            stream_file.close()
        if tmp_stream_path is not None:
            if completed:
                assert stream_telemetry_path is not None
                tmp_stream_path.replace(stream_telemetry_path)
            elif tmp_stream_path.exists():
                tmp_stream_path.unlink()
    return rows, active_sim


def _evaluate_task_with_sim(task: dict[str, Any], sim: MonzaSim | None = None) -> dict[str, Any]:
    candidate = Candidate(
        genome=genome_from_mapping(task["genome"]),
        snapshot_index=int(task["snapshot_index"]),
    )
    snapshot = snapshot_from_mapping(task["snapshot"])
    stream_telemetry_path = task.get("stream_telemetry_path")
    stream_path = Path(str(stream_telemetry_path)) if stream_telemetry_path is not None else None
    rows, _ = _run_candidate(
        sim_config=task["sim_config"],
        snapshot=snapshot,
        genome=candidate.genome,
        gates=task["gates"],
        seed=int(task["seed"]),
        collect_full_telemetry=False,
        keep_step_telemetry=bool(task.get("capture_step_telemetry", False)) and stream_path is None,
        stream_telemetry_path=stream_path,
        sim=sim,
    )
    profile_scores = _score_rows(
        rows,
        start_progress_m=snapshot.monotonic_progress_m,
        gates=task["gates"],
        scoring_profiles=tuple(task["scoring_profiles"]),
    )
    primary_profile = str(task["scoring_profiles"][0])
    final = rows[-1] if rows else {}
    best_progress_m = max([snapshot.monotonic_progress_m, *[float(row["monotonic_progress_m"]) for row in rows]])
    target_reached = best_progress_m >= task["gates"].target_progress_m
    sim_segment_complete = bool(final.get("segment_complete", False))
    result = {
        "candidate_index": int(task["candidate_index"]),
        "generation": int(task["generation"]),
        "seed": int(task["seed"]),
        "score": float(profile_scores[primary_profile]),
        "primary_scoring_profile": primary_profile,
        "profile_scores": profile_scores,
        "snapshot_index": candidate.snapshot_index,
        "start_snapshot_id": snapshot.id,
        "start_progress_m": snapshot.monotonic_progress_m,
        "start_speed_kph": snapshot.speed_mps * 3.6,
        "genome": genome_to_dict(candidate.genome),
        "best_progress_m": best_progress_m,
        "final_progress_m": final.get("monotonic_progress_m", snapshot.monotonic_progress_m),
        "remaining_m": max(0.0, task["gates"].target_progress_m - best_progress_m),
        "segment_complete": bool(sim_segment_complete or target_reached),
        "target_reached": bool(target_reached),
        "sim_segment_complete": bool(sim_segment_complete),
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
    if bool(task.get("capture_step_telemetry", False)):
        if stream_path is not None:
            result["selected_telemetry"] = str(stream_path)
        else:
            result["captured_telemetry"] = rows
    return result


def _evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    return _evaluate_task_with_sim(task)


def _evaluate_chunk(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    sim = MonzaSim(tasks[0]["sim_config"])
    return [_evaluate_task_with_sim(task, sim=sim) for task in tasks]


def _chunks(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _resolved_workers(workers: int) -> int:
    if workers == 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, workers)


def _resolved_chunk_size(population: int, workers: int, configured: int) -> int:
    if configured > 0:
        return configured
    if workers <= 1:
        return population
    return max(1, population // max(workers * 4, 1))


def _evaluate_population(
    *,
    candidates: list[Candidate],
    snapshots: list[StateSnapshot],
    sim_config: SimConfig,
    gates: EvolutionGates,
    generation: int,
    seed: int,
    workers: int,
    worker_chunk_size: int,
    scoring_profiles: tuple[str, ...],
    capture_step_telemetry: bool,
    stream_telemetry_dir: Path | None = None,
    executor: ProcessPoolExecutor | None = None,
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
            "scoring_profiles": scoring_profiles,
            "capture_step_telemetry": capture_step_telemetry,
            "stream_telemetry_path": str(
                stream_telemetry_dir
                / f"evolution-all_candidates-gen-{generation:03d}-candidate-{index:05d}-steps.jsonl"
            )
            if capture_step_telemetry and stream_telemetry_dir is not None
            else None,
        }
        for index, candidate in enumerate(candidates)
    ]
    resolved_workers = _resolved_workers(workers)
    if resolved_workers <= 1:
        sim = MonzaSim(sim_config)
        return [_evaluate_task_with_sim(task, sim=sim) for task in tasks]
    chunk_size = _resolved_chunk_size(len(tasks), resolved_workers, worker_chunk_size)
    task_chunks = _chunks(tasks, chunk_size)
    active_executor = executor or ProcessPoolExecutor(max_workers=resolved_workers)
    should_shutdown = executor is None
    try:
        results: list[dict[str, Any]] = []
        for chunk_results in active_executor.map(_evaluate_chunk, task_chunks):
            results.extend(chunk_results)
        return results
    finally:
        if should_shutdown:
            active_executor.shutdown()


def _select_elite_rows(ranked: list[dict[str, Any]], config: EvolutionSearchConfig) -> list[dict[str, Any]]:
    if not ranked:
        return []
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    per_profile = max(1, config.elite_count // max(1, len(config.scoring_profiles)))
    for profile in config.scoring_profiles:
        profile_ranked = sorted(
            ranked,
            key=lambda row: float(row.get("profile_scores", {}).get(profile, float("-inf"))),
            reverse=True,
        )
        for row in profile_ranked[:per_profile]:
            selected[(int(row["generation"]), int(row["candidate_index"]))] = row
    for row in ranked:
        if len(selected) >= config.elite_count:
            break
        selected[(int(row["generation"]), int(row["candidate_index"]))] = row
    rows = list(selected.values())
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows[: max(1, min(config.elite_count, len(ranked)))]


def _rank_biased_index(count: int, rng: np.random.Generator) -> int:
    if count <= 1:
        return 0
    ranks = np.arange(count, 0, -1, dtype=np.float64)
    weights = ranks**3.4
    weights /= weights.sum()
    return int(rng.choice(count, p=weights))


def _next_population(
    *,
    ranked: list[dict[str, Any]],
    snapshots: list[StateSnapshot],
    config: EvolutionSearchConfig,
    rng: np.random.Generator,
    action_names: list[str],
) -> list[Candidate]:
    elite_rows = _select_elite_rows(ranked, config)
    elites = [
        Candidate(
            genome=genome_from_mapping(row["genome"]),
            snapshot_index=int(row["snapshot_index"]),
        )
        for row in elite_rows
    ]
    if not elites:
        elites = [
            Candidate(
                genome=random_genome(
                    rng,
                    action_names=action_names,
                    min_phases=config.min_phases,
                    max_phases=config.max_phases,
                    min_phase_steps=config.min_phase_steps,
                    max_phase_steps=config.max_phase_steps,
                    genome_type=config.genome_type,
                    min_phase_progress_m=config.min_phase_progress_m,
                    max_phase_progress_m=config.max_phase_progress_m,
                ),
                snapshot_index=int(rng.integers(0, len(snapshots))),
            )
        ]
    next_candidates = list(elites)
    immigrant_count = min(config.random_immigrants, max(0, config.population - len(next_candidates)))
    while len(next_candidates) < config.population - immigrant_count:
        parent = elites[_rank_biased_index(len(elites), rng)]
        genome = parent.genome
        if len(elites) > 1 and rng.random() < config.crossover_rate:
            other = elites[_rank_biased_index(len(elites), rng)]
            genome = crossover_genomes(
                genome,
                other.genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
                min_phase_progress_m=config.min_phase_progress_m,
                max_phase_progress_m=config.max_phase_progress_m,
            )
        if rng.random() < config.mutation_rate:
            genome = mutate_genome(
                genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
                min_phase_progress_m=config.min_phase_progress_m,
                max_phase_progress_m=config.max_phase_progress_m,
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
                    genome_type=config.genome_type,
                    min_phase_progress_m=config.min_phase_progress_m,
                    max_phase_progress_m=config.max_phase_progress_m,
                ),
                snapshot_index=int(rng.integers(0, len(snapshots))),
            )
        )
    return next_candidates


def _generation_summary(generation: int, ranked: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    termination_reasons = Counter(str(row["termination_reason"]) for row in ranked)
    completion_count = sum(1 for row in ranked if row["segment_complete"])
    milestone_count = sum(1 for row in ranked if row.get("target_reached", row["segment_complete"]))
    farthest_attempt = max(ranked, key=lambda row: float(row["best_progress_m"]), default=None)
    avg_progress_m = (
        sum(float(row["best_progress_m"]) for row in ranked) / max(1, len(ranked))
        if ranked
        else 0.0
    )
    profile_leaders = {}
    if ranked:
        for profile in ranked[0].get("profile_scores", {}):
            leader = max(ranked, key=lambda row: float(row.get("profile_scores", {}).get(profile, float("-inf"))))
            profile_leaders[profile] = {
                "score": leader.get("profile_scores", {}).get(profile),
                "candidate_index": leader["candidate_index"],
                "best_progress_m": leader["best_progress_m"],
                "termination_reason": leader["termination_reason"],
            }
    return {
        "generation": generation,
        "population": len(ranked),
        "elapsed_s": elapsed_s,
        "candidates_per_second": len(ranked) / max(elapsed_s, 1e-9),
        "best_score": ranked[0]["score"] if ranked else None,
        "best_progress_m": ranked[0]["best_progress_m"] if ranked else None,
        "score_leader_progress_m": ranked[0]["best_progress_m"] if ranked else None,
        "farthest_progress_m": farthest_attempt["best_progress_m"] if farthest_attempt is not None else None,
        "generation_best_distance_m": farthest_attempt["best_progress_m"] if farthest_attempt is not None else None,
        "generation_average_distance_m": avg_progress_m,
        "best_remaining_m": ranked[0]["remaining_m"] if ranked else None,
        "completion_count": completion_count,
        "completion_rate": completion_count / max(1, len(ranked)),
        "milestone_count": milestone_count,
        "milestone_rate": milestone_count / max(1, len(ranked)),
        "termination_reasons": dict(termination_reasons),
        "profile_leaders": profile_leaders,
        "best_attempt": ranked[0] if ranked else None,
        "farthest_attempt": farthest_attempt,
    }


def _merge_best_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    combined = [*existing, *new_rows]
    combined.sort(key=lambda row: float(row["score"]), reverse=True)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in combined:
        key = json.dumps(
            {
                "generation": row["generation"],
                "candidate_index": row["candidate_index"],
                "seed": row["seed"],
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _row_identity(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["generation"]), int(row["candidate_index"]), int(row["seed"])


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        key = _row_identity(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def _load_attempt_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tagged_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    tagged = dict(row)
    tagged["telemetry_selection_reason"] = reason
    return tagged


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in value).strip("-") or "selected"


def _best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(key, float("-inf")) or float("-inf")))


def _select_telemetry_rows(
    *,
    attempts_path: Path,
    top_rows: list[dict[str, Any]],
    latest_ranked: list[dict[str, Any]],
    config: EvolutionSearchConfig,
) -> list[dict[str, Any]]:
    if config.telemetry_selection == "top":
        return [_tagged_row(row, "top_score") for row in top_rows]

    all_rows = _load_attempt_rows(attempts_path)
    if not all_rows:
        all_rows = latest_ranked
    if config.telemetry_selection == "all":
        return [_tagged_row(row, "all_candidates") for row in all_rows]

    selected: list[dict[str, Any]] = [_tagged_row(row, "top_score") for row in top_rows]
    for profile in config.scoring_profiles:
        leader = max(
            all_rows,
            key=lambda row: float(row.get("profile_scores", {}).get(profile, float("-inf"))),
            default=None,
        )
        if leader is not None:
            selected.append(_tagged_row(leader, f"profile_leader:{profile}"))
    for key, reason in (
        ("best_progress_m", "farthest_best_progress"),
        ("final_progress_m", "farthest_final_progress"),
        ("final_speed_kph", "fastest_final_speed"),
    ):
        leader = _best_by(all_rows, key)
        if leader is not None:
            selected.append(_tagged_row(leader, reason))
    moving_rows = [
        row
        for row in all_rows
        if float(row.get("best_progress_m", 0.0) or 0.0) > float(row.get("start_progress_m", 0.0) or 0.0) + 5.0
    ]
    moving = _best_by(moving_rows, "best_progress_m")
    if moving is not None:
        selected.append(_tagged_row(moving, "best_moving_candidate"))
    for reason in ("off_track", "collision", "no_progress"):
        rows = [row for row in all_rows if str(row.get("termination_reason")) == reason]
        leader = _best_by(rows, "best_progress_m")
        if leader is not None:
            selected.append(_tagged_row(leader, f"best_{reason}"))
    return _dedupe_rows(selected)


def _write_generation_genomes(output_dir: Path, generation: int, ranked: list[dict[str, Any]], top_k: int) -> None:
    payload = {
        "generation": generation,
        "top": [
            {
                "rank": rank,
                "candidate_index": row["candidate_index"],
                "score": row["score"],
                "profile_scores": row.get("profile_scores", {}),
                "best_progress_m": row["best_progress_m"],
                "termination_reason": row["termination_reason"],
                "genome": row["genome"],
            }
            for rank, row in enumerate(ranked[:top_k])
        ],
    }
    _write_json(output_dir / "top_genomes" / f"generation_{generation:04d}.json", payload)


def _search_hash(
    *,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    state_library: Path | None,
    start_progress_m: float | None,
    start_speed_kph: float,
    start_min_progress_m: float | None,
    start_max_progress_m: float | None,
) -> str:
    config_payload = asdict(config)
    for runtime_key in (
        "workers",
        "worker_chunk_size",
        "checkpoint_every_generations",
        "progress_every_generation",
    ):
        config_payload.pop(runtime_key, None)
    payload = {
        "config": config_payload,
        "gates": asdict(gates),
        "state_library": str(state_library) if state_library is not None else None,
        "start_progress_m": start_progress_m,
        "start_speed_kph": start_speed_kph,
        "start_min_progress_m": start_min_progress_m,
        "start_max_progress_m": start_max_progress_m,
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_path(path: Path) -> Path:
    if path.is_dir() or path.suffix == "":
        return path / CHECKPOINT_NAME
    return path


def _quote_cli(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or any(char in text for char in ('"', "'", "&", "(", ")")):
        return '"' + text.replace('"', '`"') + '"'
    return text


def _resume_command(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    state_library: Path | None,
    start_progress_m: float | None,
    start_speed_kph: float,
    start_min_progress_m: float | None,
    start_max_progress_m: float | None,
) -> str:
    parts = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "f1rl.evolution_search",
        "--resume",
        _quote_cli(checkpoint_path),
        "--output-dir",
        _quote_cli(output_dir),
        "--target-progress-m",
        str(gates.target_progress_m),
        "--action-set",
        config.action_set,
        "--observation-profile",
        config.observation_profile,
        "--max-steps",
        str(config.max_steps),
        "--population",
        str(config.population),
        "--generations",
        str(config.generations),
        "--elite-count",
        str(config.elite_count),
        "--random-immigrants",
        str(config.random_immigrants),
        "--min-phases",
        str(config.min_phases),
        "--max-phases",
        str(config.max_phases),
        "--min-phase-steps",
        str(config.min_phase_steps),
        "--max-phase-steps",
        str(config.max_phase_steps),
        "--min-phase-progress-m",
        str(config.min_phase_progress_m),
        "--max-phase-progress-m",
        str(config.max_phase_progress_m),
        "--mutation-rate",
        str(config.mutation_rate),
        "--crossover-rate",
        str(config.crossover_rate),
        "--start-mutation-rate",
        str(config.start_mutation_rate),
        "--seed",
        str(config.seed),
        "--top-k",
        str(config.top_k),
        "--workers",
        str(config.workers),
        "--worker-chunk-size",
        str(config.worker_chunk_size),
        "--genome-type",
        config.genome_type,
        "--scoring-profiles",
        ",".join(config.scoring_profiles),
        "--telemetry-selection",
        config.telemetry_selection,
        "--checkpoint-every-generations",
        str(config.checkpoint_every_generations),
    ]
    if not gates.terminate_at_target_progress:
        parts.append("--no-target-termination")
    if config.progress_every_generation:
        parts.append("--progress-every-generation")
    if state_library is not None:
        parts.extend(["--state-library", _quote_cli(state_library)])
    if start_progress_m is not None:
        parts.extend(["--start-progress-m", str(start_progress_m)])
    parts.extend(["--start-speed-kph", str(start_speed_kph)])
    if start_min_progress_m is not None:
        parts.extend(["--start-min-progress-m", str(start_min_progress_m)])
    if start_max_progress_m is not None:
        parts.extend(["--start-max-progress-m", str(start_max_progress_m)])
    if gates.target_min_speed_kph is not None:
        parts.extend(["--target-min-speed-kph", str(gates.target_min_speed_kph)])
    if gates.target_max_speed_kph is not None:
        parts.extend(["--target-max-speed-kph", str(gates.target_max_speed_kph)])
    if gates.target_max_lateral_error_m is not None:
        parts.extend(["--target-max-lateral-error-m", str(gates.target_max_lateral_error_m)])
    if gates.target_max_heading_error_deg is not None:
        parts.extend(["--target-max-heading-error-deg", str(gates.target_max_heading_error_deg)])
    if gates.target_max_abs_yaw_rate_rps is not None:
        parts.extend(["--target-max-abs-yaw-rate-rps", str(gates.target_max_abs_yaw_rate_rps)])
    if gates.target_max_abs_steering is not None:
        parts.extend(["--target-max-abs-steering", str(gates.target_max_abs_steering)])
    if not gates.segment_fail_on_speed_gate_miss:
        parts.append("--no-segment-fail-on-speed-gate-miss")
    if gates.segment_require_release:
        parts.append("--segment-require-release")
    parts.extend(
        [
            "--segment-release-min-speed-kph",
            str(gates.segment_release_min_speed_kph),
            "--segment-release-max-speed-kph",
            str(gates.segment_release_max_speed_kph),
            "--segment-release-max-brake",
            str(gates.segment_release_max_brake),
            "--segment-release-max-throttle",
            str(gates.segment_release_max_throttle),
        ]
    )
    return " ".join(parts)


def _write_checkpoint(
    output_dir: Path,
    *,
    config_hash: str,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    state_library: Path | None,
    start_progress_m: float | None,
    start_speed_kph: float,
    start_min_progress_m: float | None,
    start_max_progress_m: float | None,
    next_generation: int,
    population: list[Candidate],
    rng: np.random.Generator,
    best_rows: list[dict[str, Any]],
    generation_summaries: list[dict[str, Any]],
    attempt_count: int,
) -> None:
    checkpoint_path = output_dir / CHECKPOINT_NAME
    resume_command = _resume_command(
        checkpoint_path,
        output_dir=output_dir,
        config=config,
        gates=gates,
        state_library=state_library,
        start_progress_m=start_progress_m,
        start_speed_kph=start_speed_kph,
        start_min_progress_m=start_min_progress_m,
        start_max_progress_m=start_max_progress_m,
    )
    payload = {
        "schema_version": 1,
        "kind": "evolution_population_checkpoint",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_hash": config_hash,
        "config": asdict(config),
        "gates": asdict(gates),
        "state_library": str(state_library) if state_library is not None else None,
        "start_progress_m": start_progress_m,
        "start_speed_kph": start_speed_kph,
        "start_min_progress_m": start_min_progress_m,
        "start_max_progress_m": start_max_progress_m,
        "next_generation": next_generation,
        "population": [candidate_to_dict(candidate) for candidate in population],
        "rng_state": rng.bit_generator.state,
        "best_rows": best_rows,
        "generation_summaries": generation_summaries,
        "attempt_count": attempt_count,
        "resume_command": resume_command,
    }
    _write_json_atomic(checkpoint_path, payload)


def _load_checkpoint(path: Path, *, expected_hash: str, force: bool) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    actual_hash = str(payload.get("config_hash", ""))
    if actual_hash != expected_hash and not force:
        raise ValueError(
            "Checkpoint config hash mismatch. Use --force-resume only if you intentionally "
            "want to branch from this population with changed settings."
        )
    return payload


def _initial_population(
    *,
    config: EvolutionSearchConfig,
    snapshots: list[StateSnapshot],
    rng: np.random.Generator,
    action_names: list[str],
) -> list[Candidate]:
    return [
        Candidate(
            genome=random_genome(
                rng,
                action_names=action_names,
                min_phases=config.min_phases,
                max_phases=config.max_phases,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                genome_type=config.genome_type,
                min_phase_progress_m=config.min_phase_progress_m,
                max_phase_progress_m=config.max_phase_progress_m,
            ),
            snapshot_index=int(rng.integers(0, len(snapshots))),
        )
        for _ in range(config.population)
    ]


def _load_snapshots(
    *,
    sim_config: SimConfig,
    config: EvolutionSearchConfig,
    state_library: Path | None,
    start_progress_m: float | None,
    start_speed_kph: float,
    start_min_progress_m: float | None,
    start_max_progress_m: float | None,
) -> list[StateSnapshot]:
    if state_library is None:
        return [
            _default_start_snapshot(
                sim_config=sim_config,
                seed=config.seed,
                start_progress_m=start_progress_m,
                start_speed_kph=start_speed_kph,
            )
        ]
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
    return snapshots


def _write_ppo_bridge(
    output_dir: Path,
    *,
    elite_library_path: Path,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    state_library: Path | None,
    top_rows: list[dict[str, Any]],
) -> None:
    segment_length_m = max(
        1.0,
        gates.target_progress_m - min((float(row["start_progress_m"]) for row in top_rows), default=0.0),
    )
    selected_telemetry = top_rows[0].get("selected_telemetry") if top_rows else None
    train_command = (
        "uv run --no-sync python -m f1rl.train "
        "--curriculum segments "
        f"--curriculum-state-library {elite_library_path} "
        f"--curriculum-state-library-segment-length-m {segment_length_m:.1f} "
        f"--curriculum-state-library-target-progress-m {gates.target_progress_m:.1f} "
        f"--action-set {config.action_set} "
        f"--observation-profile {config.observation_profile} "
        "--device auto --require-gpu --vec-env subproc"
    )
    benchmark_command = (
        "uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint latest "
        "--episodes 3 --max-steps 18000 --device auto "
        f"--action-set {config.action_set} --observation-profile {config.observation_profile} "
        "--disable-scaffold-rewards --disable-training-assists"
    )
    replay_command = (
        f"uv run --no-sync python -m f1rl.replay {selected_telemetry}"
        if selected_telemetry is not None
        else "No selected telemetry available."
    )
    bridge = {
        "kind": "evolution_to_ppo_bridge",
        "elite_state_library": str(elite_library_path),
        "source_state_library": str(state_library) if state_library is not None else None,
        "config": asdict(config),
        "gates": asdict(gates),
        "top_attempts": top_rows,
        "recommended_commands": {
            "train_curriculum": train_command,
            "replay_best_evolution_candidate": replay_command,
            "honest_normal_start_benchmark": benchmark_command,
        },
        "final_success_rule": (
            "Evolutionary behavior is discovery only. Final success still requires PPO to "
            "complete a valid normal-start lap near <=80.0s with assists and scaffolds disabled."
        ),
    }
    _write_json(output_dir / "ppo_bridge.json", bridge)
    next_commands = "\n".join(
        [
            "# Next Commands",
            "",
            "Replay the best search candidate:",
            "",
            "```powershell",
            replay_command,
            "```",
            "",
            "Train PPO from the elite state library:",
            "",
            "```powershell",
            train_command,
            "```",
            "",
            "Benchmark honest normal-start PPO after training:",
            "",
            "```powershell",
            benchmark_command,
            "```",
            "",
            "Do not count the evolutionary trajectory as final success.",
            "",
        ]
    )
    (output_dir / "next_commands.md").write_text(next_commands, encoding="utf-8")


def _write_summary(
    output_dir: Path,
    *,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    sim_config: SimConfig,
    state_library: Path | None,
    snapshot_count: int,
    attempt_count: int,
    generation_summaries: list[dict[str, Any]],
    latest_ranked: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    elite_library_path: Path | None,
    complete: bool,
    stopped_early: bool,
) -> None:
    summary = {
        "run_id": output_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "elitist_evolutionary_search",
        "complete": complete,
        "stopped_early": stopped_early,
        "sim_config": dataclass_to_dict(sim_config),
        "config": asdict(config),
        "gates": asdict(gates),
        "state_library": str(state_library) if state_library is not None else None,
        "snapshot_count": snapshot_count,
        "attempt_count": attempt_count,
        "elite_state_library": str(elite_library_path) if elite_library_path is not None else None,
        "generation_summaries": generation_summaries,
        "latest_generation_top": latest_ranked[: config.top_k],
        "top_attempts": top_rows,
    }
    _write_json(output_dir / "evolution_summary.json", summary)


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
    resume: Path | None = None,
    force_resume: bool = False,
    stop_after_generations: int | None = None,
) -> Path:
    config = EvolutionSearchConfig(
        **{
            **asdict(config),
            "scoring_profiles": tuple(config.scoring_profiles),
        }
    )
    _validate_config(config)
    action_names = [name for name, _, _, _ in actions_for_action_set(config.action_set)]
    sim_config = SimConfig(
        max_steps=config.max_steps,
        action_mode="continuous" if config.genome_type == "controller" else "discrete",
        action_set=config.action_set,
        observation_profile=config.observation_profile,
    )
    snapshots = _load_snapshots(
        sim_config=sim_config,
        config=config,
        state_library=state_library,
        start_progress_m=start_progress_m,
        start_speed_kph=start_speed_kph,
        start_min_progress_m=start_min_progress_m,
        start_max_progress_m=start_max_progress_m,
    )
    if not snapshots:
        raise ValueError("No snapshots available for evolutionary search.")

    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "attempts.jsonl"
    generation_summary_path = output_dir / "generation_summary.jsonl"
    selected_dir = output_dir / "selected_telemetry"
    config_hash = _search_hash(
        config=config,
        gates=gates,
        state_library=state_library,
        start_progress_m=start_progress_m,
        start_speed_kph=start_speed_kph,
        start_min_progress_m=start_min_progress_m,
        start_max_progress_m=start_max_progress_m,
    )

    rng = np.random.default_rng(config.seed)
    start_generation = 0
    attempt_count = 0
    best_rows: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    global_best_distance_m = 0.0
    if resume is not None:
        checkpoint = _load_checkpoint(resume, expected_hash=config_hash, force=force_resume)
        start_generation = int(checkpoint["next_generation"])
        population = [candidate_from_mapping(row) for row in checkpoint["population"]]
        rng.bit_generator.state = checkpoint["rng_state"]
        best_rows = list(checkpoint.get("best_rows", []))
        generation_summaries = list(checkpoint.get("generation_summaries", []))
        attempt_count = int(checkpoint.get("attempt_count", start_generation * config.population))
        global_best_distance_m = max(
            [
                float(summary.get("generation_best_distance_m", summary.get("farthest_progress_m", 0.0)) or 0.0)
                for summary in generation_summaries
            ],
            default=0.0,
        )
    else:
        attempts_path.write_text("", encoding="utf-8")
        generation_summary_path.write_text("", encoding="utf-8")
        population = _initial_population(
            config=config,
            snapshots=snapshots,
            rng=rng,
            action_names=action_names,
        )

    latest_ranked: list[dict[str, Any]] = []
    captured_telemetry_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    best_limit = max(config.top_k * 5, config.elite_count * 2, 32)
    resolved_workers = _resolved_workers(config.workers)
    executor: ProcessPoolExecutor | None = None
    if resolved_workers > 1:
        executor = ProcessPoolExecutor(max_workers=resolved_workers)
    try:
        for generation in range(start_generation, config.generations):
            started = time.perf_counter()
            evaluated = _evaluate_population(
                candidates=population,
                snapshots=snapshots,
                sim_config=sim_config,
                gates=gates,
                generation=generation,
                seed=config.seed,
                workers=resolved_workers,
                worker_chunk_size=config.worker_chunk_size,
                scoring_profiles=config.scoring_profiles,
                capture_step_telemetry=config.telemetry_selection == "all",
                stream_telemetry_dir=selected_dir if config.telemetry_selection == "all" else None,
                executor=executor,
            )
            for row in evaluated:
                captured = row.pop("captured_telemetry", None)
                if captured is not None:
                    captured_telemetry_by_key[_row_identity(row)] = captured
            elapsed_s = time.perf_counter() - started
            ranked = sorted(evaluated, key=lambda row: float(row["score"]), reverse=True)
            latest_ranked = ranked
            attempt_count += len(ranked)
            _append_jsonl(attempts_path, ranked)
            summary = _generation_summary(generation, ranked, elapsed_s)
            global_best_distance_m = max(
                global_best_distance_m,
                float(summary.get("generation_best_distance_m", 0.0) or 0.0),
            )
            summary["global_best_distance_m"] = global_best_distance_m
            generation_summaries.append(summary)
            _append_jsonl(generation_summary_path, [summary])
            best_rows = _merge_best_rows(best_rows, ranked, limit=best_limit)
            _write_json(output_dir / "best_so_far.json", best_rows[0] if best_rows else {})
            _write_generation_genomes(output_dir, generation, ranked, config.top_k)

            if config.progress_every_generation:
                best = ranked[0] if ranked else {}
                farthest = summary.get("farthest_attempt") or {}
                print(
                    "evolution_generation "
                    f"generation={generation} population={len(ranked)} "
                    f"leader_progress_m={float(best.get('best_progress_m', 0.0)):.3f} "
                    f"generation_best_distance_m={float(farthest.get('best_progress_m', 0.0)):.3f} "
                    f"generation_average_distance_m={float(summary.get('generation_average_distance_m', 0.0)):.3f} "
                    f"global_best_distance_m={global_best_distance_m:.3f} "
                    f"best_score={float(best.get('score', 0.0)):.3f} "
                    f"completion_rate={summary['completion_rate']:.3f} "
                    f"milestone_rate={summary['milestone_rate']:.3f} "
                    f"candidates_per_second={summary['candidates_per_second']:.3f}",
                    flush=True,
                )

            if generation < config.generations - 1:
                population = _next_population(
                    ranked=ranked,
                    snapshots=snapshots,
                    config=config,
                    rng=rng,
                    action_names=action_names,
                )
            if (
                config.checkpoint_every_generations > 0
                and (generation + 1) % config.checkpoint_every_generations == 0
            ):
                _write_checkpoint(
                    output_dir,
                    config_hash=config_hash,
                    config=config,
                    gates=gates,
                    state_library=state_library,
                    start_progress_m=start_progress_m,
                    start_speed_kph=start_speed_kph,
                    start_min_progress_m=start_min_progress_m,
                    start_max_progress_m=start_max_progress_m,
                    next_generation=generation + 1,
                    population=population,
                    rng=rng,
                    best_rows=best_rows,
                    generation_summaries=generation_summaries,
                    attempt_count=attempt_count,
                )
            if (
                stop_after_generations is not None
                and generation + 1 >= stop_after_generations
                and generation < config.generations - 1
            ):
                _write_checkpoint(
                    output_dir,
                    config_hash=config_hash,
                    config=config,
                    gates=gates,
                    state_library=state_library,
                    start_progress_m=start_progress_m,
                    start_speed_kph=start_speed_kph,
                    start_min_progress_m=start_min_progress_m,
                    start_max_progress_m=start_max_progress_m,
                    next_generation=generation + 1,
                    population=population,
                    rng=rng,
                    best_rows=best_rows,
                    generation_summaries=generation_summaries,
                    attempt_count=attempt_count,
                )
                _write_summary(
                    output_dir,
                    config=config,
                    gates=gates,
                    sim_config=sim_config,
                    state_library=state_library,
                    snapshot_count=len(snapshots),
                    attempt_count=attempt_count,
                    generation_summaries=generation_summaries,
                    latest_ranked=latest_ranked,
                    top_rows=best_rows[: config.top_k],
                    elite_library_path=None,
                    complete=False,
                    stopped_early=True,
                )
                return output_dir
    finally:
        if executor is not None:
            executor.shutdown()

    top_rows = best_rows[: config.top_k]
    top_row_keys = {_row_identity(row) for row in top_rows}
    telemetry_rows = _select_telemetry_rows(
        attempts_path=attempts_path,
        top_rows=top_rows,
        latest_ranked=latest_ranked,
        config=config,
    )
    elite_snapshots: list[StateSnapshot] = []
    telemetry_manifest: list[dict[str, Any]] = []
    top_rows_by_key = {_row_identity(row): row for row in top_rows}
    for rank, row in enumerate(telemetry_rows):
        snapshot = snapshots[int(row["snapshot_index"])]
        key = _row_identity(row)
        rows = captured_telemetry_by_key.get(key)
        sim: MonzaSim | None = None
        existing_telemetry = row.get("selected_telemetry")
        telemetry_path = Path(str(existing_telemetry)) if existing_telemetry is not None else None
        if telemetry_path is not None and telemetry_path.exists():
            final_row = _last_jsonl_row(telemetry_path)
        else:
            rows, sim = _run_candidate(
                sim_config=sim_config,
                snapshot=snapshot,
                genome=genome_from_mapping(row["genome"]),
                gates=gates,
                seed=int(row["seed"]),
                collect_full_telemetry=True,
            )
            reason = _safe_slug(str(row.get("telemetry_selection_reason", "selected")))
            telemetry_path = selected_dir / (
                f"evolution-{reason}-rank-{rank:03d}-gen-{int(row['generation']):03d}-"
                f"candidate-{int(row['candidate_index']):05d}-steps.jsonl"
            )
            _write_jsonl(telemetry_path, rows)
            final_row = rows[-1] if rows else {}
        row["selected_telemetry"] = str(telemetry_path)
        telemetry_manifest.append(
            {
                "rank": rank,
                "selection_reason": row.get("telemetry_selection_reason", "selected"),
                "generation": row["generation"],
                "candidate_index": row["candidate_index"],
                "seed": row["seed"],
                "score": row["score"],
                "profile_scores": row.get("profile_scores", {}),
                "best_progress_m": row["best_progress_m"],
                "final_progress_m": final_row.get("monotonic_progress_m", row.get("final_progress_m")),
                "termination_reason": final_row.get("termination_reason", row.get("termination_reason")),
                "path": str(telemetry_path),
            }
        )
        if key in top_rows_by_key:
            top_rows_by_key[key]["selected_telemetry"] = str(telemetry_path)
        if key in top_row_keys:
            if sim is not None:
                elite_snapshots.append(snapshot_from_sim(sim, source="evolution_search", source_file=str(telemetry_path)))
            elif final_row:
                elite_snapshots.append(
                    snapshot_from_mapping(
                        {
                            **final_row,
                            "source": "evolution_search",
                            "source_file": str(telemetry_path),
                        }
                    )
                )
    _write_json(
        selected_dir / "manifest.json",
        {
            "telemetry_selection": config.telemetry_selection,
            "trace_count": len(telemetry_manifest),
            "traces": telemetry_manifest,
        },
    )

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
    _write_ppo_bridge(
        output_dir,
        elite_library_path=elite_library_path,
        config=config,
        gates=gates,
        state_library=state_library,
        top_rows=top_rows,
    )
    _write_summary(
        output_dir,
        config=config,
        gates=gates,
        sim_config=sim_config,
        state_library=state_library,
        snapshot_count=len(snapshots),
        attempt_count=attempt_count,
        generation_summaries=generation_summaries,
        latest_ranked=latest_ranked,
        top_rows=top_rows,
        elite_library_path=elite_library_path,
        complete=True,
        stopped_early=False,
    )
    _write_checkpoint(
        output_dir,
        config_hash=config_hash,
        config=config,
        gates=gates,
        state_library=state_library,
        start_progress_m=start_progress_m,
        start_speed_kph=start_speed_kph,
        start_min_progress_m=start_min_progress_m,
        start_max_progress_m=start_max_progress_m,
        next_generation=config.generations,
        population=population,
        rng=rng,
        best_rows=best_rows,
        generation_summaries=generation_summaries,
        attempt_count=attempt_count,
    )
    return output_dir


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"evolution-search-{timestamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run elitist evolutionary search over driving controllers.")
    parser.add_argument("--state-library", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--stop-after-generations", type=int)
    parser.add_argument("--start-progress-m", type=float)
    parser.add_argument("--start-speed-kph", type=float, default=0.0)
    parser.add_argument("--start-min-progress-m", type=float)
    parser.add_argument("--start-max-progress-m", type=float)
    parser.add_argument("--target-progress-m", type=float, default=MONZA_LENGTH_METERS)
    parser.add_argument(
        "--no-target-termination",
        action="store_true",
        help=(
            "Treat --target-progress-m as a scoring/completion milestone only. "
            "Do not install a simulator segment target that truncates candidates there."
        ),
    )
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
    parser.add_argument("--min-phase-progress-m", type=float, default=5.0)
    parser.add_argument("--max-phase-progress-m", type=float, default=160.0)
    parser.add_argument("--mutation-rate", type=float, default=0.75)
    parser.add_argument("--crossover-rate", type=float, default=0.35)
    parser.add_argument("--start-mutation-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunk-size", type=int, default=0)
    parser.add_argument("--genome-type", choices=sorted(GENOME_TYPES), default="phase")
    parser.add_argument("--scoring-profiles", default="max_progress")
    parser.add_argument("--telemetry-selection", choices=sorted(TELEMETRY_SELECTIONS), default="top")
    parser.add_argument("--checkpoint-every-generations", type=int, default=1)
    parser.add_argument("--progress-every-generation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resume_path = args.resume
    output_dir = args.output_dir
    if output_dir is None and resume_path is not None:
        output_dir = _checkpoint_path(resume_path).parent
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
        min_phase_progress_m=max(0.1, args.min_phase_progress_m),
        max_phase_progress_m=max(args.min_phase_progress_m, args.max_phase_progress_m),
        mutation_rate=float(np.clip(args.mutation_rate, 0.0, 1.0)),
        crossover_rate=float(np.clip(args.crossover_rate, 0.0, 1.0)),
        start_mutation_rate=float(np.clip(args.start_mutation_rate, 0.0, 1.0)),
        seed=args.seed,
        top_k=max(1, args.top_k),
        workers=max(0, args.workers),
        worker_chunk_size=max(0, args.worker_chunk_size),
        genome_type=args.genome_type,
        scoring_profiles=_parse_csv(args.scoring_profiles),
        telemetry_selection=args.telemetry_selection,
        checkpoint_every_generations=max(0, args.checkpoint_every_generations),
        progress_every_generation=bool(args.progress_every_generation),
    )
    gates = EvolutionGates(
        target_progress_m=args.target_progress_m,
        terminate_at_target_progress=not bool(args.no_target_termination),
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
    run_root = run_evolution_search(
        output_dir=output_dir or default_output_dir(),
        config=config,
        gates=gates,
        state_library=args.state_library,
        start_progress_m=args.start_progress_m,
        start_speed_kph=args.start_speed_kph,
        start_min_progress_m=args.start_min_progress_m,
        start_max_progress_m=args.start_max_progress_m,
        resume=resume_path,
        force_resume=bool(args.force_resume),
        stop_after_generations=args.stop_after_generations,
    )
    print(f"evolution_search_complete run={run_root} summary={run_root / 'evolution_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
