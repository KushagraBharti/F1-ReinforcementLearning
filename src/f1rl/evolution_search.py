"""Elitist evolutionary search for Monza driving behavior.

The search engine is intentionally separate from PPO. It brute-forces candidate
controllers, keeps useful elites, writes resumable artifacts, and produces state
libraries that PPO can later use as curriculum starts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast

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
EVOLUTION_BACKENDS = frozenset({"cpu", "gpu"})
GPU_ENGINES = frozenset({"eager", "graph", "fused"})
GPU_PROFILES = frozenset({"none", "torch", "nsight"})
GPU_FAST_GEOMETRIES = frozenset({"local_window", "grid"})
GPU_COLLISION_MODES = frozenset({"exact_all_segments", "exact_grid", "mask_only_debug"})
GPU_RUN_MODES = frozenset({"parity", "production", "postcheck"})
GPU_ATTEMPTS_MODES = frozenset({"auto", "full", "compact"})
TELEMETRY_SELECTIONS = frozenset({"top", "leaders", "all"})
TELEMETRY_COMPRESSIONS = frozenset({"none", "gzip"})
GPU_DTYPES = frozenset({"float32", "float64"})
GPU_TELEMETRY_MODES = frozenset({"selected", "top", "all-cpu-replay", "none"})
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
        "frontier_fast",
        "early_pace",
        "clean_distance",
        "farthest_distance",
        "frontier_recovery",
        "frontier_novelty",
        "fast_valid_lap",
        "time_attack",
        "lap_pace",
        "fast_frontier",
    }
)
CONTROLLER_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "speed_norm",
    "target_speed_norm",
    "speed_error_norm",
    "brake_demand",
    "future_brake_demand",
    "target_speed_drop_norm",
    "brake_gate_proximity",
    "brake_gate_distance_norm",
    "lookahead_abs_max",
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
CONTROLLER_DOMINANCE_EPSILON = 1.0e-3
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
    lineage: dict[str, Any] = field(default_factory=dict)


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
    backend: str = "cpu"
    action_set: str = "racing"
    observation_profile: str = "racing_v2"
    max_steps: int = 600
    max_steps_schedule: tuple[tuple[int, int], ...] = ()
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
    telemetry_compression: str = "none"
    all_candidate_telemetry_dir: Path | None = None
    checkpoint_every_generations: int = 1
    progress_every_generation: bool = False
    adaptive_survival_floor: bool = True
    survival_floor_stages_m: tuple[float, ...] = (450.0, 1000.0, 1220.0, 1500.0, 2000.0, 2400.0, 3000.0, 4000.0, 5000.0)
    survival_floor_pass_rate: float = 0.30
    parent_pool_size: int = 0
    adaptive_immigrants: bool = True
    min_random_immigrants: int = 2
    smart_immigrant_fraction: float = 0.50
    smart_immigrant_current_fraction: float = 0.90
    frontier_focus_start_m: float = 2200.0
    frontier_focus_end_m: float = 2600.0
    frontier_parent_min_progress_m: float = 2000.0
    survival_floor_leader_jump: bool = True
    late_frontier_trigger_m: float = 4000.0
    plateau_mode: bool = True
    plateau_generations: int = 3
    plateau_distance_epsilon_m: float = 8.0
    plateau_average_improvement_m: float = 80.0
    plateau_elite_fraction: float = 0.50
    plateau_extra_mutations: int = 1
    gpu_device: str = "cuda"
    gpu_engine: str = "eager"
    gpu_run_mode: str = "parity"
    gpu_dtype: str = "float32"
    gpu_batch_size: int | None = None
    gpu_static_batch_size: int | None = None
    gpu_verify_top_k: int = 4
    gpu_cpu_replay_top_k: int | None = None
    gpu_verify_elite_multiplier: int = 2
    gpu_telemetry_mode: str = "selected"
    gpu_parity_check: bool = False
    gpu_fallback_to_cpu: bool = False
    gpu_compile: bool = False
    gpu_profile: str = "none"
    gpu_profile_output: Path | None = None
    gpu_chunk_steps: int = 256
    gpu_disable_early_stop: bool = False
    gpu_fast_geometry: str = "local_window"
    gpu_collision_mode: str = "exact_all_segments"
    gpu_attempts_mode: str = "auto"


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


def _jsonl_text_open(path: Path, mode: str) -> TextIO:
    text_mode = mode if "t" in mode else f"{mode}t"
    opener = gzip.open if path.suffix == ".gz" else Path.open
    return cast(TextIO, opener(path, text_mode, encoding="utf-8"))


def _telemetry_file_suffix(compression: str) -> str:
    if compression == "none":
        return ".jsonl"
    if compression == "gzip":
        return ".jsonl.gz"
    raise ValueError(f"Unknown telemetry compression {compression!r}")


def _tmp_jsonl_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.gz"):
        return path.with_name(f"{path.name.removesuffix('.jsonl.gz')}.tmp.jsonl.gz")
    if path.name.endswith(".jsonl"):
        return path.with_name(f"{path.name.removesuffix('.jsonl')}.tmp.jsonl")
    return path.with_name(f"{path.name}.tmp")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _jsonl_text_open(path, "a") as file:
        for row in rows:
            file.write(json.dumps(row, default=_json_default) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _jsonl_text_open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, default=_json_default) + "\n")


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    last_row: dict[str, Any] = {}
    with _jsonl_text_open(path, "r") as file:
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


def _parse_float_csv(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(float(item) for item in value)
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_max_steps_schedule(value: str | tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if isinstance(value, tuple):
        return tuple((int(generation), int(steps)) for generation, steps in value)
    if isinstance(value, list):
        return tuple((int(generation), int(steps)) for generation, steps in value)
    if not value.strip():
        return ()
    schedule: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        generation_text, separator, steps_text = item.partition(":")
        if separator != ":":
            raise ValueError("max steps schedule entries must use generation:steps, e.g. 0:10000,5:15000")
        schedule.append((int(generation_text.strip()), int(steps_text.strip())))
    return tuple(schedule)


def _parse_gpu_batch_size(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return None if value <= 0 else value
    text = value.strip().lower()
    if text in {"", "auto"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("gpu batch size must be a positive integer or 'auto'")
    return parsed


def _max_steps_for_generation(config: EvolutionSearchConfig, generation: int) -> int:
    max_steps = int(config.max_steps)
    for start_generation, scheduled_steps in config.max_steps_schedule:
        if generation >= start_generation:
            max_steps = int(scheduled_steps)
        else:
            break
    return max(1, max_steps)


def _sim_config_for_generation(config: EvolutionSearchConfig, generation: int) -> SimConfig:
    return SimConfig(
        max_steps=_max_steps_for_generation(config, generation),
        action_mode="continuous" if config.genome_type == "controller" else "discrete",
        action_set=config.action_set,
        observation_profile=config.observation_profile,
    )


def _validate_config(config: EvolutionSearchConfig, gates: EvolutionGates) -> None:
    if config.backend not in EVOLUTION_BACKENDS:
        valid = ", ".join(sorted(EVOLUTION_BACKENDS))
        raise ValueError(f"Unknown evolution backend {config.backend!r}; expected one of: {valid}")
    if config.gpu_engine not in GPU_ENGINES:
        valid = ", ".join(sorted(GPU_ENGINES))
        raise ValueError(f"Unknown GPU engine {config.gpu_engine!r}; expected one of: {valid}")
    if config.gpu_run_mode not in GPU_RUN_MODES:
        valid = ", ".join(sorted(GPU_RUN_MODES))
        raise ValueError(f"Unknown GPU run mode {config.gpu_run_mode!r}; expected one of: {valid}")
    if config.gpu_attempts_mode not in GPU_ATTEMPTS_MODES:
        valid = ", ".join(sorted(GPU_ATTEMPTS_MODES))
        raise ValueError(f"Unknown GPU attempts mode {config.gpu_attempts_mode!r}; expected one of: {valid}")
    if config.gpu_profile not in GPU_PROFILES:
        valid = ", ".join(sorted(GPU_PROFILES))
        raise ValueError(f"Unknown GPU profile mode {config.gpu_profile!r}; expected one of: {valid}")
    if config.gpu_fast_geometry not in GPU_FAST_GEOMETRIES:
        valid = ", ".join(sorted(GPU_FAST_GEOMETRIES))
        raise ValueError(f"Unknown GPU fast geometry mode {config.gpu_fast_geometry!r}; expected one of: {valid}")
    if config.gpu_collision_mode not in GPU_COLLISION_MODES:
        valid = ", ".join(sorted(GPU_COLLISION_MODES))
        raise ValueError(f"Unknown GPU collision mode {config.gpu_collision_mode!r}; expected one of: {valid}")
    if config.genome_type not in GENOME_TYPES:
        valid = ", ".join(sorted(GENOME_TYPES))
        raise ValueError(f"Unknown genome type {config.genome_type!r}; expected one of: {valid}")
    if config.telemetry_selection not in TELEMETRY_SELECTIONS:
        valid = ", ".join(sorted(TELEMETRY_SELECTIONS))
        raise ValueError(f"Unknown telemetry selection {config.telemetry_selection!r}; expected one of: {valid}")
    if config.telemetry_compression not in TELEMETRY_COMPRESSIONS:
        valid = ", ".join(sorted(TELEMETRY_COMPRESSIONS))
        raise ValueError(f"Unknown telemetry compression {config.telemetry_compression!r}; expected one of: {valid}")
    if config.gpu_dtype not in GPU_DTYPES:
        valid = ", ".join(sorted(GPU_DTYPES))
        raise ValueError(f"Unknown GPU dtype {config.gpu_dtype!r}; expected one of: {valid}")
    if config.gpu_telemetry_mode not in GPU_TELEMETRY_MODES:
        valid = ", ".join(sorted(GPU_TELEMETRY_MODES))
        raise ValueError(f"Unknown GPU telemetry mode {config.gpu_telemetry_mode!r}; expected one of: {valid}")
    if config.gpu_batch_size is not None and config.gpu_batch_size <= 0:
        raise ValueError("gpu_batch_size must be positive when set")
    if config.gpu_static_batch_size is not None and config.gpu_static_batch_size <= 0:
        raise ValueError("gpu_static_batch_size must be positive when set")
    if config.gpu_verify_top_k < 0:
        raise ValueError("gpu_verify_top_k must be >= 0")
    if config.gpu_cpu_replay_top_k is not None and config.gpu_cpu_replay_top_k < 0:
        raise ValueError("gpu_cpu_replay_top_k must be >= 0 when set")
    if config.gpu_verify_elite_multiplier < 1:
        raise ValueError("gpu_verify_elite_multiplier must be >= 1")
    if config.gpu_chunk_steps <= 0:
        raise ValueError("gpu_chunk_steps must be > 0")
    if config.backend == "gpu":
        if config.gpu_engine == "fused":
            from f1rl.gpu_fast_warp import warp_status

            status = warp_status()
            if not status.installed:
                raise ValueError(
                    "GPU fused engine requires optional dependency 'warp-lang' before fused rollout work can run. "
                    "Install with `uv sync --extra gpu-fast`, then use --gpu-engine eager or graph until the fused "
                    "rollout implementation is promoted."
                )
            if config.gpu_device != "cuda":
                raise ValueError("GPU fused engine currently requires --gpu-device cuda.")
            if config.gpu_dtype != "float32":
                raise ValueError("GPU fused engine currently requires --gpu-dtype float32.")
            if config.gpu_fast_geometry != "local_window":
                raise ValueError("GPU fused engine currently requires --gpu-fast-geometry local_window.")
            if config.gpu_collision_mode != "exact_grid":
                raise ValueError("GPU fused engine currently requires --gpu-collision-mode exact_grid.")
        if config.gpu_fast_geometry == "grid":
            raise ValueError("GPU grid geometry is planned but not implemented yet; use local_window.")
        if config.telemetry_selection == "all" and config.gpu_telemetry_mode != "all-cpu-replay":
            raise ValueError(
                "--telemetry-selection all with --backend gpu requires "
                "--gpu-telemetry-mode all-cpu-replay so all full telemetry is CPU-replayed explicitly."
            )
    if config.max_steps_schedule:
        previous_generation = -1
        for generation, max_steps in config.max_steps_schedule:
            if generation < 0:
                raise ValueError("max_steps_schedule generations must be >= 0")
            if max_steps <= 0:
                raise ValueError("max_steps_schedule step counts must be > 0")
            if generation <= previous_generation:
                raise ValueError("max_steps_schedule generations must be sorted ascending and unique")
            previous_generation = generation
    unknown_profiles = set(config.scoring_profiles) - SCORING_PROFILES
    if unknown_profiles:
        valid = ", ".join(sorted(SCORING_PROFILES))
        unknown = ", ".join(sorted(unknown_profiles))
        raise ValueError(f"Unknown scoring profile(s): {unknown}; expected one or more of: {valid}")
    if config.parent_pool_size < 0:
        raise ValueError("parent_pool_size must be >= 0")
    if not config.survival_floor_stages_m:
        raise ValueError("survival_floor_stages_m must contain at least one progress floor")
    if any(stage < 0.0 for stage in config.survival_floor_stages_m):
        raise ValueError("survival_floor_stages_m values must be >= 0")
    if tuple(sorted(config.survival_floor_stages_m)) != tuple(config.survival_floor_stages_m):
        raise ValueError("survival_floor_stages_m must be sorted ascending")
    if not 0.0 <= config.survival_floor_pass_rate <= 1.0:
        raise ValueError("survival_floor_pass_rate must be between 0 and 1")
    if config.min_random_immigrants < 0:
        raise ValueError("min_random_immigrants must be >= 0")
    if not 0.0 <= config.smart_immigrant_fraction <= 1.0:
        raise ValueError("smart_immigrant_fraction must be between 0 and 1")
    if not 0.0 <= config.smart_immigrant_current_fraction <= 1.0:
        raise ValueError("smart_immigrant_current_fraction must be between 0 and 1")
    if config.frontier_focus_end_m <= config.frontier_focus_start_m:
        raise ValueError("frontier_focus_end_m must be greater than frontier_focus_start_m")
    if config.frontier_parent_min_progress_m < 0.0:
        raise ValueError("frontier_parent_min_progress_m must be >= 0")
    if config.late_frontier_trigger_m < 0.0:
        raise ValueError("late_frontier_trigger_m must be >= 0")
    if config.plateau_generations < 2:
        raise ValueError("plateau_generations must be >= 2")
    if config.plateau_distance_epsilon_m < 0.0:
        raise ValueError("plateau_distance_epsilon_m must be >= 0")
    if config.plateau_average_improvement_m < 0.0:
        raise ValueError("plateau_average_improvement_m must be >= 0")
    if not 0.0 < config.plateau_elite_fraction <= 1.0:
        raise ValueError("plateau_elite_fraction must be in (0, 1]")
    if config.plateau_extra_mutations < 0:
        raise ValueError("plateau_extra_mutations must be >= 0")


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
        "lineage": candidate.lineage,
    }


def candidate_from_mapping(value: dict[str, Any]) -> Candidate:
    return Candidate(
        genome=genome_from_mapping(value["genome"]),
        snapshot_index=int(value["snapshot_index"]),
        lineage=dict(value.get("lineage", {})),
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
        feature_index = {name: index for index, name in enumerate(CONTROLLER_FEATURE_NAMES)}
        throttle_offset = 0
        brake_offset = feature_count
        steer_offset = feature_count * 2
        weights[throttle_offset + feature_index["bias"]] += 1.0
        weights[brake_offset + feature_index["bias"]] -= 1.0
        for name in ("brake_demand", "future_brake_demand", "target_speed_drop_norm", "brake_gate_proximity"):
            weights[brake_offset + feature_index[name]] += 0.65
            weights[throttle_offset + feature_index[name]] -= 0.45
        weights[steer_offset + feature_index["target_steer"]] += 0.80
        weights[steer_offset + feature_index["heading_error_norm"]] -= 0.25
        weights[steer_offset + feature_index["signed_lateral_error_norm"]] -= 0.20
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


def mutate_genome_with_metadata(
    genome: Genome,
    rng: np.random.Generator,
    *,
    action_names: list[str],
    min_phase_steps: int,
    max_phase_steps: int,
    max_phases: int,
    min_phase_progress_m: float = 5.0,
    max_phase_progress_m: float = 160.0,
    mutation_scale: float = 1.0,
) -> tuple[Genome, dict[str, Any]]:
    mutation_scale = max(0.05, float(mutation_scale))
    metadata: dict[str, Any] = {
        "mutation_type": "none",
        "mutation_sigma": None,
        "reset_count": 0,
        "changed_gene_count": 0,
    }
    if genome.kind == "controller":
        weights = np.asarray(genome.controller_weights, dtype=np.float64).copy()
        if len(weights) == 0:
            random = random_genome(
                rng,
                action_names=action_names,
                min_phases=1,
                max_phases=max_phases,
                min_phase_steps=min_phase_steps,
                max_phase_steps=max_phase_steps,
                genome_type="controller",
            )
            metadata.update(
                {
                    "mutation_type": "controller_random_reset",
                    "reset_count": len(random.controller_weights),
                    "changed_gene_count": len(random.controller_weights),
                }
            )
            return random, metadata
        roll = float(rng.random())
        if roll < 0.52:
            mask = rng.random(len(weights)) < 0.42
            if not bool(mask.any()):
                mask[int(rng.integers(0, len(weights)))] = True
            changed = int(mask.sum())
            sigma = 0.58 * mutation_scale
            weights[mask] += rng.normal(0.0, sigma, changed)
            metadata.update(
                {
                    "mutation_type": "controller_masked_gaussian",
                    "mutation_sigma": sigma,
                    "changed_gene_count": changed,
                }
            )
        elif roll < 0.82:
            sigma = 0.34 * mutation_scale
            weights += rng.normal(0.0, sigma, len(weights))
            metadata.update(
                {
                    "mutation_type": "controller_full_gaussian",
                    "mutation_sigma": sigma,
                    "changed_gene_count": len(weights),
                }
            )
        else:
            reset_count = int(rng.integers(3, min(9, len(weights) + 1)))
            indices = rng.choice(len(weights), size=reset_count, replace=False)
            sigma = 2.05 * mutation_scale
            if mutation_scale < 0.75:
                weights[indices] += rng.normal(0.0, sigma, reset_count)
            else:
                weights[indices] = rng.normal(0.0, sigma, reset_count)
            metadata.update(
                {
                    "mutation_type": "controller_weight_reset",
                    "mutation_sigma": sigma,
                    "reset_count": reset_count,
                    "changed_gene_count": reset_count,
                }
            )
        return _normalize_controller_genome(Genome(kind="controller", controller_weights=tuple(weights))), metadata

    if genome.kind == "progress_phase":
        phases = list(genome.progress_phases)
        if not phases:
            random = random_genome(
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
            metadata.update(
                {
                    "mutation_type": "progress_phase_random_reset",
                    "reset_count": len(random.progress_phases),
                    "changed_gene_count": len(random.progress_phases),
                }
            )
            return random, metadata
        operation = str(rng.choice(["action", "duration", "insert", "delete", "swap"]))
        metadata["mutation_type"] = f"progress_phase_{operation}"
        if operation == "action":
            index = int(rng.integers(0, len(phases)))
            phases[index] = ProgressPhaseGene(
                action=str(action_names[int(rng.integers(0, len(action_names)))]),
                progress_m=phases[index].progress_m,
            )
            metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
        elif operation == "duration":
            index = int(rng.integers(0, len(phases)))
            sigma = max(2.0, (max_phase_progress_m - min_phase_progress_m) / 6.0) * mutation_scale
            delta = float(rng.normal(0.0, sigma))
            phases[index] = ProgressPhaseGene(
                action=phases[index].action,
                progress_m=phases[index].progress_m + delta,
            )
            metadata.update(
                {
                    "mutation_sigma": sigma,
                    "changed_gene_count": 1,
                    "changed_gene_index": index,
                }
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
            metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
        elif operation == "delete" and len(phases) > 1:
            index = int(rng.integers(0, len(phases)))
            del phases[index]
            metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
        elif operation == "swap" and len(phases) > 1:
            left = int(rng.integers(0, len(phases)))
            right = int(rng.integers(0, len(phases)))
            phases[left], phases[right] = phases[right], phases[left]
            metadata.update({"changed_gene_count": 2, "changed_gene_indices": [left, right]})
        return _normalize_progress_genome(
            Genome(kind="progress_phase", progress_phases=tuple(phases)),
            action_names=action_names,
            min_progress_m=min_phase_progress_m,
            max_progress_m=max_phase_progress_m,
        ), metadata

    phases = list(genome.phases)
    if not phases:
        random = random_genome(
            rng,
            action_names=action_names,
            min_phases=1,
            max_phases=max_phases,
            min_phase_steps=min_phase_steps,
            max_phase_steps=max_phase_steps,
        )
        metadata.update(
            {
                "mutation_type": "phase_random_reset",
                "reset_count": len(random.phases),
                "changed_gene_count": len(random.phases),
            }
        )
        return random, metadata

    operation = str(rng.choice(["action", "duration", "insert", "delete", "swap"]))
    metadata["mutation_type"] = f"phase_{operation}"
    if operation == "action":
        index = int(rng.integers(0, len(phases)))
        phases[index] = PhaseGene(
            action=str(action_names[int(rng.integers(0, len(action_names)))]),
            steps=phases[index].steps,
        )
        metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
    elif operation == "duration":
        index = int(rng.integers(0, len(phases)))
        sigma = max(2.0, (max_phase_steps - min_phase_steps) / 5.0) * mutation_scale
        delta = int(rng.normal(0.0, sigma))
        phases[index] = PhaseGene(action=phases[index].action, steps=phases[index].steps + delta)
        metadata.update(
            {
                "mutation_sigma": sigma,
                "changed_gene_count": 1,
                "changed_gene_index": index,
            }
        )
    elif operation == "insert" and len(phases) < max_phases:
        index = int(rng.integers(0, len(phases) + 1))
        phases.insert(
            index,
            PhaseGene(
                action=str(action_names[int(rng.integers(0, len(action_names)))]),
                steps=int(rng.integers(min_phase_steps, max_phase_steps + 1)),
            ),
        )
        metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
    elif operation == "delete" and len(phases) > 1:
        index = int(rng.integers(0, len(phases)))
        del phases[index]
        metadata.update({"changed_gene_count": 1, "changed_gene_index": index})
    elif operation == "swap" and len(phases) > 1:
        left = int(rng.integers(0, len(phases)))
        right = int(rng.integers(0, len(phases)))
        phases[left], phases[right] = phases[right], phases[left]
        metadata.update({"changed_gene_count": 2, "changed_gene_indices": [left, right]})

    return _normalize_phase_genome(
        Genome(kind="phase", phases=tuple(phases)),
        action_names=action_names,
        min_steps=min_phase_steps,
        max_steps=max_phase_steps,
    ), metadata


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
    mutation_scale: float = 1.0,
) -> Genome:
    mutated, _ = mutate_genome_with_metadata(
        genome,
        rng,
        action_names=action_names,
        min_phase_steps=min_phase_steps,
        max_phase_steps=max_phase_steps,
        max_phases=max_phases,
        min_phase_progress_m=min_phase_progress_m,
        max_phase_progress_m=max_phase_progress_m,
        mutation_scale=mutation_scale,
    )
    return mutated


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
    if brake_raw + CONTROLLER_DOMINANCE_EPSILON >= throttle_raw:
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


def _pace_metrics(rows: list[dict[str, Any]], *, start_progress_m: float) -> dict[str, Any]:
    if not rows:
        return {
            "elapsed_s": 0.0,
            "pace_mps": 0.0,
            "pace_kph": 0.0,
            "time_to_300_m": None,
            "time_to_450_m": None,
            "avg_speed_first_300_m": 0.0,
            "avg_speed_first_450_m": 0.0,
            "avg_brake_first_300_m": 0.0,
            "avg_brake_first_450_m": 0.0,
        }

    final = rows[-1]
    best_progress_m = max(float(row["monotonic_progress_m"]) for row in rows)
    raw_progress_m = max(0.0, best_progress_m - start_progress_m)
    elapsed_s = float(final.get("sim_time_s", 0.0) or 0.0)
    if elapsed_s <= 0.0:
        elapsed_s = len(rows) / 60.0
    pace_mps = raw_progress_m / max(elapsed_s, 1e-6)

    def first_time_for(delta_m: float) -> float | None:
        target = start_progress_m + delta_m
        for row in rows:
            if float(row.get("monotonic_progress_m", start_progress_m) or start_progress_m) >= target:
                time_s = float(row.get("sim_time_s", 0.0) or 0.0)
                return time_s if time_s > 0.0 else None
        return None

    def averages_before(delta_m: float) -> tuple[float, float]:
        target = start_progress_m + delta_m
        selected = [
            row
            for row in rows
            if float(row.get("monotonic_progress_m", start_progress_m) or start_progress_m) <= target
        ]
        if not selected:
            selected = rows[:1]
        avg_speed = sum(float(row.get("speed_kph", 0.0) or 0.0) for row in selected) / max(1, len(selected))
        avg_brake = sum(float(row.get("brake", 0.0) or 0.0) for row in selected) / max(1, len(selected))
        return avg_speed, avg_brake

    avg_speed_300, avg_brake_300 = averages_before(300.0)
    avg_speed_450, avg_brake_450 = averages_before(450.0)
    return {
        "elapsed_s": elapsed_s,
        "pace_mps": pace_mps,
        "pace_kph": pace_mps * 3.6,
        "time_to_300_m": first_time_for(300.0),
        "time_to_450_m": first_time_for(450.0),
        "avg_speed_first_300_m": avg_speed_300,
        "avg_speed_first_450_m": avg_speed_450,
        "avg_brake_first_300_m": avg_brake_300,
        "avg_brake_first_450_m": avg_brake_450,
    }


def _row_is_valid_finish(row: dict[str, Any]) -> bool:
    return bool(
        row.get("completed_lap")
        or row.get("valid_lap")
        or row.get("finish_crossed")
        or str(row.get("termination_reason", "")) == "lap_complete"
    )


def _row_elapsed_s(row: dict[str, Any]) -> float:
    elapsed = row.get("elapsed_s")
    if elapsed is None:
        final_row = row.get("final_row")
        if isinstance(final_row, dict):
            elapsed = final_row.get("sim_time_s")
    if elapsed is None:
        elapsed = row.get("sim_time_s")
    if elapsed is None:
        return float("inf")
    try:
        return float(elapsed)
    except (TypeError, ValueError):
        return float("inf")


def _fast_lap_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    valid = 1.0 if _row_is_valid_finish(row) else 0.0
    elapsed = _row_elapsed_s(row)
    elapsed_score = -elapsed if math.isfinite(elapsed) else -1_000_000.0
    return (
        valid,
        elapsed_score,
        float(row.get("pace_kph", 0.0) or 0.0),
        float(row.get("score", 0.0) or 0.0),
        float(row.get("best_progress_m", 0.0) or 0.0),
    )


def _score_profile(
    rows: list[dict[str, Any]],
    *,
    start_progress_m: float,
    gates: EvolutionGates,
    profile: str,
    frontier_focus_start_m: float = 2200.0,
    frontier_focus_end_m: float = 2600.0,
) -> float:
    if not rows:
        return float("-inf")
    final = rows[-1]
    best_row = max(rows, key=lambda row: float(row["monotonic_progress_m"]))
    best_progress_m = float(best_row["monotonic_progress_m"])
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
    best_speed_kph = float(best_row.get("speed_kph", final_speed_kph) or final_speed_kph)
    best_lateral_error_m = abs(float(best_row.get("lateral_error_m", final_lateral_error_m) or final_lateral_error_m))
    best_heading_error_deg = abs(float(best_row.get("heading_error_deg", final_heading_error_deg) or final_heading_error_deg))
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
    pace = _pace_metrics(rows, start_progress_m=start_progress_m)
    elapsed_s = float(pace["elapsed_s"])
    pace_kph = float(pace["pace_kph"])
    avg_speed_first_300_m = float(pace["avg_speed_first_300_m"])
    avg_speed_first_450_m = float(pace["avg_speed_first_450_m"])
    avg_brake_first_450_m = float(pace["avg_brake_first_450_m"])
    time_to_450_m = pace.get("time_to_450_m")
    early_slow_penalty = max(0.0, 125.0 - avg_speed_first_300_m) * 34.0
    if raw_progress_m >= 450.0:
        early_slow_penalty += max(0.0, 145.0 - avg_speed_first_450_m) * 18.0
    early_brake_penalty = max(0.0, avg_brake_first_450_m - 0.32) * 1_800.0
    focus_start_m = min(frontier_focus_start_m, frontier_focus_end_m)
    focus_end_m = max(frontier_focus_start_m, frontier_focus_end_m)
    focus_span_m = max(1.0, focus_end_m - focus_start_m)
    focus_progress_m = float(np.clip(best_progress_m - focus_start_m, 0.0, focus_span_m))
    focus_progress_ratio = focus_progress_m / focus_span_m
    reached_focus = best_progress_m >= focus_start_m
    cleared_focus = best_progress_m >= focus_end_m
    stalled_in_focus = reached_focus and reason == "no_progress"
    focus_lateral_penalty = max(0.0, best_lateral_error_m - 9.0) * 420.0
    focus_heading_penalty = max(0.0, best_heading_error_deg - 18.0) * 520.0
    focus_stop_penalty = max(0.0, 90.0 - final_speed_kph) * 170.0 if reached_focus else 0.0
    late_progress_factor = float(np.clip((best_progress_m - 3000.0) / 2200.0, 0.0, 1.0))
    frontier_quality_factor = float(np.clip((best_progress_m - 1500.0) / 3500.0, 0.0, 1.0))
    lap_progress_ratio = float(np.clip(best_progress_m / MONZA_LENGTH_METERS, 0.0, 1.0))
    demand_rows = [
        row
        for row in rows
        if (
            float(row.get("future_brake_demand", row.get("brake_demand", 0.0)) or 0.0) >= 0.22
            or float(row.get("brake_gate_proximity", 0.0) or 0.0) >= 0.70
            or float(row.get("target_speed_drop_norm", 0.0) or 0.0) >= 0.18
        )
        and float(row.get("speed_kph", 0.0) or 0.0) >= 135.0
    ]
    if demand_rows:
        avg_brake_demand_zone = sum(float(row.get("brake", 0.0) or 0.0) for row in demand_rows) / len(demand_rows)
        avg_throttle_demand_zone = sum(float(row.get("throttle", 0.0) or 0.0) for row in demand_rows) / len(demand_rows)
        max_future_brake_demand = max(
            float(row.get("future_brake_demand", row.get("brake_demand", 0.0)) or 0.0)
            for row in demand_rows
        )
    else:
        avg_brake_demand_zone = 0.0
        avg_throttle_demand_zone = 0.0
        max_future_brake_demand = 0.0
    setup_penalty = frontier_quality_factor * (
        max(0.0, final_lateral_error_m - 12.0) * 620.0
        + max(0.0, final_heading_error_deg - 22.0) * 720.0
        + max(0.0, final_yaw_rate_rps - 0.85) * 2_200.0
    )
    late_setup_penalty = late_progress_factor * (
        max(0.0, final_lateral_error_m - 9.0) * 900.0
        + max(0.0, final_heading_error_deg - 16.0) * 1_050.0
        + max(0.0, 115.0 - final_speed_kph) * 120.0
    )
    brake_demand_penalty = frontier_quality_factor * max_future_brake_demand * (
        max(0.0, 0.32 - avg_brake_demand_zone) * 18_000.0
        + avg_throttle_demand_zone * 7_500.0
    )
    viability_penalty = setup_penalty + late_setup_penalty + brake_demand_penalty
    valid_elapsed_s = elapsed_s if valid_finish and elapsed_s > 0.0 else float("inf")
    valid_lap_speed_bonus = 0.0
    if valid_finish and math.isfinite(valid_elapsed_s):
        valid_lap_speed_bonus = (
            8_500_000.0
            + max(0.0, 220.0 - valid_elapsed_s) * 58_000.0
            + max(0.0, 170.0 - valid_elapsed_s) * 72_000.0
            + max(0.0, 130.0 - valid_elapsed_s) * 120_000.0
            + max(0.0, 100.0 - valid_elapsed_s) * 180_000.0
            + max(0.0, 85.0 - valid_elapsed_s) * 260_000.0
            + min(pace_kph, 380.0) * 16_000.0
            - max(0.0, valid_elapsed_s - 100.0) * 34_000.0
            - max(0.0, valid_elapsed_s - 130.0) * 80_000.0
            - max(0.0, valid_elapsed_s - 170.0) * 140_000.0
        )

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

    if profile == "fast_valid_lap":
        score += raw_progress_m * 20.0 + best_progress_m * 16.0
        score += frontier_m * 70.0 + near_target_m * 160.0 + beyond_target_m * 250.0
        score += min(pace_kph, 380.0) * 4_600.0
        score += min(final_speed_kph, 360.0) * 120.0
        if valid_finish:
            score += valid_lap_speed_bonus
            score -= max(0.0, valid_elapsed_s - 150.0) * 95_000.0
            score -= max(0.0, valid_elapsed_s - 180.0) * 140_000.0
        else:
            score -= remaining_m * 70.0
            if reason == "max_steps":
                score -= 80_000.0 + max(0.0, elapsed_s - 120.0) * 800.0
        if time_to_450_m is not None:
            score += max(0.0, 16.5 - float(time_to_450_m)) * 1_200.0
        score -= early_slow_penalty * 2.40
        score -= early_brake_penalty * 1.60
        score -= viability_penalty * (0.35 if valid_finish else 0.70)
        score -= missed_checkpoints * 12_000.0
        return float(score)

    if profile == "time_attack":
        score += raw_progress_m * 12.0 + best_progress_m * 10.0
        score += frontier_m * 55.0 + near_target_m * 130.0 + beyond_target_m * 190.0
        score += min(pace_kph, 400.0) * 8_800.0 * max(0.20, progress_ratio)
        if valid_finish:
            score += 6_500_000.0
            score += max(0.0, 180.0 - valid_elapsed_s) * 95_000.0
            score += max(0.0, 120.0 - valid_elapsed_s) * 160_000.0
            score += max(0.0, 90.0 - valid_elapsed_s) * 280_000.0
            score -= max(0.0, valid_elapsed_s - 120.0) * 80_000.0
            score -= max(0.0, valid_elapsed_s - 150.0) * 150_000.0
        else:
            score -= remaining_m * 90.0
            if reason in {"max_steps", "no_progress"}:
                score -= 90_000.0
        score -= early_slow_penalty * 2.10
        score -= viability_penalty * 0.45
        score -= missed_checkpoints * 14_000.0
        return float(score)

    if profile == "lap_pace":
        score += raw_progress_m * 24.0 + best_progress_m * 12.0
        score += frontier_m * 60.0 + near_target_m * 135.0 + beyond_target_m * 210.0
        score += min(pace_kph, 390.0) * 3_800.0 * max(0.30, frontier_quality_factor)
        score += min(avg_speed_first_450_m, 340.0) * 150.0
        if valid_finish:
            score += 2_800_000.0 + max(0.0, 190.0 - valid_elapsed_s) * 50_000.0
            score -= max(0.0, valid_elapsed_s - 150.0) * 55_000.0
        if reason == "max_steps":
            score -= max(0.0, elapsed_s - 150.0) * 1_200.0
        score -= early_slow_penalty * 1.80
        score -= viability_penalty * 0.50
        score -= missed_checkpoints * 8_000.0
        return float(score)

    if profile == "fast_frontier":
        progress_pace_factor = 0.18 + lap_progress_ratio * 1.95
        score += raw_progress_m * 18.0 + best_progress_m * 120.0
        score += frontier_m * 70.0 + near_target_m * 130.0 + beyond_target_m * 170.0
        score += min(pace_kph, 360.0) * 13_500.0 * progress_pace_factor
        score += min(final_speed_kph, 360.0) * 1_200.0 * max(0.15, lap_progress_ratio)
        score += min(avg_speed_first_450_m, 340.0) * 160.0
        if best_progress_m >= 3000.0:
            score += 260_000.0 + min(pace_kph, 330.0) * 1_200.0
        if best_progress_m >= 4000.0:
            score += 360_000.0 + min(pace_kph, 330.0) * 1_700.0
        if best_progress_m >= 5000.0:
            score += 520_000.0 + min(pace_kph, 330.0) * 2_400.0
        if valid_finish:
            score -= 650_000.0 + max(0.0, valid_elapsed_s - 150.0) * 90_000.0
        else:
            score -= remaining_m * 28.0
        if best_progress_m >= 2400.0:
            score -= max(0.0, 175.0 - pace_kph) * 14_000.0
        if reason in {"max_steps", "no_progress"} and not valid_finish:
            score -= 65_000.0 + max(0.0, elapsed_s - 130.0) * 900.0
        score -= early_slow_penalty * 1.45
        score -= viability_penalty * 0.38
        score -= missed_checkpoints * 10_000.0
        return float(score)

    if profile == "frontier":
        score += best_progress_m * 12.0 + min(final_speed_kph, 360.0) * 18.0
        score += frontier_m * 95.0 + near_target_m * 210.0 + beyond_target_m * 320.0
        score += min(pace_kph, 300.0) * 16.0
        score -= viability_penalty * 0.42
        score -= final_lateral_error_m * 72.0
        score -= final_heading_error_deg * 26.0
        score -= final_yaw_rate_rps * 330.0
        score -= final_steering * 300.0
        score -= missed_checkpoints * 2_500.0
        return float(score)

    if profile == "frontier_fast":
        score += best_progress_m * 18.0 + min(final_speed_kph, 380.0) * 20.0
        score += min(pace_kph, 310.0) * 86.0
        score += frontier_m * 105.0 + near_target_m * 235.0 + beyond_target_m * 340.0
        if valid_finish:
            score += valid_lap_speed_bonus * 0.18
        if time_to_450_m is not None:
            score += max(0.0, 18.0 - float(time_to_450_m)) * 260.0
        score += min(avg_speed_first_450_m, 320.0) * 28.0
        score -= early_slow_penalty
        score -= early_brake_penalty
        score -= viability_penalty * 0.58
        score -= final_lateral_error_m * 70.0
        score -= final_heading_error_deg * 24.0
        score -= final_yaw_rate_rps * 320.0
        score -= final_steering * 260.0
        score -= missed_checkpoints * 2_500.0
        return float(score)

    if profile == "frontier_recovery":
        score += raw_progress_m * 18.0 + best_progress_m * 10.0
        score += focus_progress_m * 1_250.0
        score += focus_progress_ratio * 28_000.0
        score += min(best_speed_kph, 300.0) * 35.0
        score += min(final_speed_kph, 260.0) * 115.0
        score -= focus_lateral_penalty
        score -= focus_heading_penalty
        score -= focus_stop_penalty
        score -= missed_checkpoints * 3_000.0
        if cleared_focus:
            score += 55_000.0
        if stalled_in_focus:
            score -= 55_000.0
        if reason == "no_progress":
            score -= 18_000.0
        elif reason in {"collision", "off_track", "assist_virtual_corridor"} and reached_focus:
            score -= 10_000.0
        return float(score)

    if profile == "frontier_novelty":
        speed_bucket = min(5, int(max(0.0, final_speed_kph) // 55.0))
        lateral_bucket = min(5, int(best_lateral_error_m // 4.0))
        heading_bucket = min(5, int(best_heading_error_deg // 10.0))
        novelty_hint = (speed_bucket * 1_100.0) + ((5 - lateral_bucket) * 750.0) + ((5 - heading_bucket) * 650.0)
        score += raw_progress_m * 16.0 + focus_progress_m * 900.0 + beyond_target_m * 140.0
        score += novelty_hint
        score += min(pace_kph, 280.0) * 18.0
        score += min(final_speed_kph, 320.0) * 70.0
        score -= focus_lateral_penalty * 0.80
        score -= focus_heading_penalty * 0.80
        score -= focus_stop_penalty * 1.20
        if cleared_focus:
            score += 36_000.0
        if stalled_in_focus:
            score -= 65_000.0
        if reason == "no_progress":
            score -= 20_000.0
        return float(score)

    if profile == "risk_seeking":
        score += best_progress_m * 11.0 + min(final_speed_kph, 380.0) * 24.0
        score += min(pace_kph, 320.0) * 22.0
        score += frontier_m * 70.0 + near_target_m * 150.0 + beyond_target_m * 220.0
        score -= viability_penalty * 0.18
        score -= final_lateral_error_m * 52.0
        score -= final_heading_error_deg * 18.0
        return float(score)

    if profile == "farthest_distance":
        score += raw_progress_m * 31.0 + best_progress_m * 22.0
        score += frontier_m * 75.0 + near_target_m * 160.0 + beyond_target_m * 280.0
        score += min(pace_kph, 300.0) * 20.0
        score -= viability_penalty * 0.22
        score -= missed_checkpoints * 1_600.0
        if reason in {"collision", "off_track", "assist_virtual_corridor"}:
            score -= 1_500.0
        return float(score)

    if profile == "early_pace":
        score += raw_progress_m * 20.0 + min(pace_kph, 320.0) * 125.0
        score += min(avg_speed_first_300_m, 300.0) * 56.0
        score += min(avg_speed_first_450_m, 320.0) * 38.0
        if time_to_450_m is not None:
            score += max(0.0, 20.0 - float(time_to_450_m)) * 420.0
        score += frontier_m * 38.0 + beyond_target_m * 90.0
        score -= early_slow_penalty * 1.55
        score -= early_brake_penalty * 1.35
        score -= final_lateral_error_m * 45.0
        score -= final_heading_error_deg * 15.0
        score -= missed_checkpoints * 1_800.0
        return float(score)

    if profile == "clean_distance":
        score += raw_progress_m * 28.0 + best_progress_m * 15.0
        score += frontier_m * 58.0 + near_target_m * 115.0 + beyond_target_m * 190.0
        score += min(pace_kph, 280.0) * 18.0
        score -= viability_penalty * 0.82
        score -= final_lateral_error_m * 210.0
        score -= final_heading_error_deg * 80.0
        score -= final_yaw_rate_rps * 560.0
        score -= final_steering * 520.0
        score -= missed_checkpoints * 4_500.0
        if clean:
            score += 7_000.0
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
    frontier_focus_start_m: float = 2200.0,
    frontier_focus_end_m: float = 2600.0,
) -> dict[str, float]:
    return {
        profile: _score_profile(
            rows,
            start_progress_m=start_progress_m,
            gates=gates,
            profile=profile,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
        )
        for profile in scoring_profiles
    }


def _compact_row(row: Any, search_features: dict[str, float] | None = None) -> dict[str, Any]:
    compact = {
        "step_index": getattr(row, "step_index", None),
        "sim_time_s": getattr(row, "sim_time_s", None),
        "x": row.x,
        "y": row.y,
        "heading_deg": row.heading_deg,
        "speed_mps": row.speed_mps,
        "monotonic_progress_m": row.monotonic_progress_m,
        "raw_progress_m": row.raw_progress_m,
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
    if search_features:
        for key in (
            "target_speed_kph",
            "near_target_speed_kph",
            "min_future_target_speed_kph",
            "target_speed_drop_kph",
            "target_speed_drop_norm",
            "brake_demand",
            "future_brake_demand",
            "brake_gate_proximity",
            "braking_gate_distance_m",
            "brake_gate_distance_norm",
            "lookahead_abs_max",
        ):
            if key in search_features:
                compact[key] = float(search_features[key])
    return compact


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
            tmp_stream_path = _tmp_jsonl_path(stream_telemetry_path)
            stream_file = _jsonl_text_open(tmp_stream_path, "w")
        for step_index in range(sim_config.max_steps):
            search_features = active_sim.search_features(
                segment_start_progress_m=start_progress_m,
                segment_target_progress_m=gates.target_progress_m,
            )
            if genome.kind == "controller":
                throttle, brake, steer = _controller_controls(genome, search_features)
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
                full_row.update(_compact_row(result.telemetry, search_features))
                if stream_file is not None:
                    stream_file.write(json.dumps(full_row, default=_json_default) + "\n")
                if collect_full_telemetry or keep_step_telemetry:
                    rows.append(full_row)
                else:
                    rows.append(_compact_row(result.telemetry, search_features))
            else:
                rows.append(_compact_row(result.telemetry, search_features))
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
        lineage=dict(task.get("lineage", {})),
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
        frontier_focus_start_m=float(task.get("frontier_focus_start_m", 2200.0)),
        frontier_focus_end_m=float(task.get("frontier_focus_end_m", 2600.0)),
    )
    primary_profile = str(task["scoring_profiles"][0])
    final = rows[-1] if rows else {}
    best_progress_m = max([snapshot.monotonic_progress_m, *[float(row["monotonic_progress_m"]) for row in rows]])
    pace = _pace_metrics(rows, start_progress_m=snapshot.monotonic_progress_m)
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
        "lineage": candidate.lineage,
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
        "final_row": final,
        "max_steps": int(task["sim_config"].max_steps),
        "steps": len(rows),
        **pace,
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
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
    capture_step_telemetry: bool,
    stream_telemetry_dir: Path | None = None,
    telemetry_compression: str = "none",
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    telemetry_suffix = _telemetry_file_suffix(telemetry_compression)
    tasks = [
        {
            "candidate_index": index,
            "generation": generation,
            "seed": seed + generation * 1_000_000 + index,
            "genome": genome_to_dict(candidate.genome),
            "lineage": candidate.lineage,
            "snapshot_index": candidate.snapshot_index,
            "snapshot": snapshot_to_dict(snapshots[candidate.snapshot_index]),
            "sim_config": sim_config,
            "gates": gates,
            "scoring_profiles": scoring_profiles,
            "frontier_focus_start_m": frontier_focus_start_m,
            "frontier_focus_end_m": frontier_focus_end_m,
            "capture_step_telemetry": capture_step_telemetry,
            "stream_telemetry_path": str(
                stream_telemetry_dir
                / f"evolution-all_candidates-gen-{generation:03d}-candidate-{index:05d}-steps{telemetry_suffix}"
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


def _make_gpu_backend(
    *,
    config: EvolutionSearchConfig,
    profile_output_dir: Path | None,
) -> Any:
    from f1rl.evolution_backend import GpuBackendSettings, GpuEvolutionBackend

    return GpuEvolutionBackend(
        settings=GpuBackendSettings(
            device=config.gpu_device,
            engine=config.gpu_engine,
            run_mode=config.gpu_run_mode,
            dtype=config.gpu_dtype,
            batch_size=config.gpu_batch_size,
            static_batch_size=config.gpu_static_batch_size,
            verify_top_k=config.gpu_cpu_replay_top_k
            if config.gpu_cpu_replay_top_k is not None
            else config.gpu_verify_top_k,
            verify_elite_multiplier=config.gpu_verify_elite_multiplier,
            telemetry_mode=config.gpu_telemetry_mode,
            parity_check=config.gpu_parity_check,
            fallback_to_cpu=config.gpu_fallback_to_cpu,
            compile_rollout=config.gpu_compile and config.gpu_engine != "graph",
            profile=config.gpu_profile,
            profile_output_dir=profile_output_dir,
            active_check_interval=config.gpu_chunk_steps,
            disable_early_stop=config.gpu_disable_early_stop,
            fast_geometry=config.gpu_fast_geometry,
            collision_mode=config.gpu_collision_mode,
            attempts_mode=_effective_gpu_attempts_mode(config),
        ),
        feature_names=CONTROLLER_FEATURE_NAMES,
        controller_output_count=CONTROLLER_OUTPUT_COUNT,
    )


def _evaluate_population_gpu(
    *,
    candidates: list[Candidate],
    snapshots: list[StateSnapshot],
    sim_config: SimConfig,
    gates: EvolutionGates,
    generation: int,
    seed: int,
    scoring_profiles: tuple[str, ...],
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
    capture_step_telemetry: bool,
    stream_telemetry_dir: Path | None,
    telemetry_compression: str,
    config: EvolutionSearchConfig,
    profile_output_dir: Path | None,
    backend: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if backend is None:
        backend = _make_gpu_backend(config=config, profile_output_dir=profile_output_dir)
    active_backend = cast(Any, backend)
    return active_backend.evaluate_population(
        candidates=candidates,
        snapshots=snapshots,
        sim_config=sim_config,
        gates=gates,
        generation=generation,
        seed=seed,
        scoring_profiles=scoring_profiles,
        frontier_focus_start_m=frontier_focus_start_m,
        frontier_focus_end_m=frontier_focus_end_m,
        capture_step_telemetry=capture_step_telemetry,
        stream_telemetry_dir=stream_telemetry_dir,
        telemetry_compression=telemetry_compression,
    )


def _select_elite_rows(ranked: list[dict[str, Any]], config: EvolutionSearchConfig) -> list[dict[str, Any]]:
    if not ranked:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (int(row["generation"]), int(row["candidate_index"]))
        if key not in seen:
            seen.add(key)
            selected.append(row)

    valid_rows = [row for row in ranked if _row_is_valid_finish(row)]
    fastest_valid_rows = sorted(
        valid_rows,
        key=lambda row: (
            _row_elapsed_s(row),
            -float(row.get("pace_kph", 0.0) or 0.0),
            -float(row.get("score", 0.0) or 0.0),
        ),
    )
    for row in fastest_valid_rows[: max(1, config.elite_count // 4)]:
        add(row)

    per_profile = max(1, config.elite_count // max(1, len(config.scoring_profiles)))
    for profile in config.scoring_profiles:
        profile_ranked = sorted(
            ranked,
            key=lambda row: float(row.get("profile_scores", {}).get(profile, float("-inf"))),
            reverse=True,
        )
        for row in profile_ranked[:per_profile]:
            add(row)
    for row in ranked:
        if len(selected) >= config.elite_count:
            break
        add(row)
    return selected[: max(1, min(config.elite_count, len(ranked)))]


def _rank_biased_index(count: int, rng: np.random.Generator) -> int:
    if count <= 1:
        return 0
    ranks = np.arange(count, 0, -1, dtype=np.float64)
    weights = ranks**3.4
    weights /= weights.sum()
    return int(rng.choice(count, p=weights))


def _genome_numeric_vector(genome: Genome) -> np.ndarray:
    if genome.kind == "controller":
        return np.asarray(genome.controller_weights, dtype=np.float64)
    if genome.kind == "progress_phase":
        values: list[float] = []
        for phase in genome.progress_phases:
            values.append(float(sum(ord(char) for char in phase.action) % 997) / 997.0)
            values.append(float(phase.progress_m))
        return np.asarray(values, dtype=np.float64)
    values = []
    for phase in genome.phases:
        values.append(float(sum(ord(char) for char in phase.action) % 997) / 997.0)
        values.append(float(phase.steps))
    return np.asarray(values, dtype=np.float64)


def _genome_distance(left: Genome, right: Genome) -> dict[str, Any]:
    if left.kind != right.kind:
        return {"kind_match": False, "l2": None, "max_abs": None, "changed_gene_count": None}
    left_vector = _genome_numeric_vector(left)
    right_vector = _genome_numeric_vector(right)
    length = max(len(left_vector), len(right_vector))
    if length == 0:
        return {"kind_match": True, "l2": 0.0, "max_abs": 0.0, "changed_gene_count": 0}
    left_padded = np.pad(left_vector, (0, length - len(left_vector)))
    right_padded = np.pad(right_vector, (0, length - len(right_vector)))
    delta = left_padded - right_padded
    return {
        "kind_match": True,
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "changed_gene_count": int(np.count_nonzero(np.abs(delta) > 1e-9)),
    }


def _row_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "seed": int(row["seed"]),
        "best_progress_m": float(row.get("best_progress_m", 0.0) or 0.0),
        "score": float(row.get("score", 0.0) or 0.0),
        "source": row.get("lineage", {}).get("source"),
    }


def _survival_floor_report(
    ranked: list[dict[str, Any]],
    config: EvolutionSearchConfig,
    *,
    active_index: int,
) -> dict[str, Any]:
    stages = tuple(float(stage) for stage in config.survival_floor_stages_m)
    active_index = int(np.clip(active_index, 0, len(stages) - 1))
    floor_m = stages[active_index] if config.adaptive_survival_floor else 0.0
    pass_count = sum(1 for row in ranked if float(row.get("best_progress_m", 0.0) or 0.0) >= floor_m)
    pass_rate = pass_count / max(1, len(ranked))
    next_index = active_index
    if config.adaptive_survival_floor and pass_rate >= config.survival_floor_pass_rate:
        next_index = min(active_index + 1, len(stages) - 1)
    leader_progress_m = max(
        (float(row.get("best_progress_m", 0.0) or 0.0) for row in ranked),
        default=0.0,
    )
    leader_floor_index = active_index
    if config.adaptive_survival_floor and config.survival_floor_leader_jump:
        for index, stage in enumerate(stages):
            if leader_progress_m >= stage:
                leader_floor_index = index
        next_index = max(next_index, leader_floor_index)
    next_floor_m = stages[next_index] if config.adaptive_survival_floor else 0.0
    stage_rates = {
        f"{stage:.1f}": sum(1 for row in ranked if float(row.get("best_progress_m", 0.0) or 0.0) >= stage)
        / max(1, len(ranked))
        for stage in stages
    }
    return {
        "survival_floor_m": floor_m,
        "survival_floor_index": active_index,
        "survival_floor_pass_count": pass_count,
        "survival_floor_pass_rate": pass_rate,
        "survival_floor_pass_threshold": config.survival_floor_pass_rate,
        "survival_floor_leader_progress_m": leader_progress_m,
        "survival_floor_leader_index": leader_floor_index,
        "survival_floor_leader_jump": config.survival_floor_leader_jump,
        "next_survival_floor_m": next_floor_m,
        "next_survival_floor_index": next_index,
        "survival_floor_stage_rates": stage_rates,
    }


def _plateau_report(
    generation_summaries: list[dict[str, Any]],
    current_summary: dict[str, Any],
    config: EvolutionSearchConfig,
) -> dict[str, Any]:
    window = [*generation_summaries, current_summary][-config.plateau_generations :]
    if not config.plateau_mode or len(window) < config.plateau_generations:
        return {
            "plateau_active": False,
            "plateau_window": len(window),
            "plateau_best_range_m": None,
            "plateau_average_gain_m": None,
        }
    best_values = [
        float(summary.get("generation_best_distance_m", summary.get("farthest_progress_m", 0.0)) or 0.0)
        for summary in window
    ]
    average_values = [float(summary.get("generation_average_distance_m", 0.0) or 0.0) for summary in window]
    best_range_m = max(best_values) - min(best_values)
    average_gain_m = average_values[-1] - average_values[0]
    active = (
        best_range_m <= config.plateau_distance_epsilon_m
        and average_gain_m >= config.plateau_average_improvement_m
    )
    return {
        "plateau_active": bool(active),
        "plateau_window": len(window),
        "plateau_best_range_m": best_range_m,
        "plateau_average_gain_m": average_gain_m,
        "plateau_distance_epsilon_m": config.plateau_distance_epsilon_m,
        "plateau_average_improvement_m": config.plateau_average_improvement_m,
    }


def _parent_pool_size(config: EvolutionSearchConfig, ranked_count: int) -> int:
    if config.parent_pool_size > 0:
        return min(config.parent_pool_size, ranked_count)
    return min(ranked_count, max(config.elite_count * 3, 16))


def _parent_buckets(
    ranked: list[dict[str, Any]],
    config: EvolutionSearchConfig,
    *,
    survival_floor_m: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not ranked:
        return {}, {"parent_pool_size": 0, "parent_bucket_counts": {}}
    pool_size = _parent_pool_size(config, len(ranked))
    per_bucket = max(1, math.ceil(pool_size / 10))
    leader_progress_m = max(
        (float(row.get("best_progress_m", 0.0) or 0.0) for row in ranked),
        default=0.0,
    )
    late_frontier_active = leader_progress_m >= config.late_frontier_trigger_m
    late_floor_m = max(survival_floor_m, config.frontier_parent_min_progress_m)
    if late_frontier_active:
        late_floor_m = max(late_floor_m, leader_progress_m - 900.0)
    speed_bucket_floor_m = max(450.0, survival_floor_m * 0.50)
    survival_rows = [
        row for row in ranked if float(row.get("best_progress_m", 0.0) or 0.0) >= survival_floor_m
    ]
    frontier_rows = [
        row
        for row in ranked
        if float(row.get("best_progress_m", 0.0) or 0.0) >= late_floor_m
    ]
    speed_rows = [
        row
        for row in ranked
        if float(row.get("best_progress_m", 0.0) or 0.0) >= speed_bucket_floor_m
    ]
    valid_rows = [row for row in ranked if _row_is_valid_finish(row)]
    non_finish_frontier_rows = [row for row in speed_rows if not _row_is_valid_finish(row)]
    buckets = {
        "fastest_valid_lap": sorted(
            valid_rows,
            key=lambda row: (
                _row_elapsed_s(row),
                -float(row.get("pace_kph", 0.0) or 0.0),
                -float(row.get("score", 0.0) or 0.0),
            ),
        )[:per_bucket],
        "fast_valid_lap_score": sorted(
            valid_rows,
            key=lambda row: float(row.get("profile_scores", {}).get("fast_valid_lap", row.get("score", float("-inf")))),
            reverse=True,
        )[:per_bucket],
        "fast_frontier_score": sorted(
            non_finish_frontier_rows,
            key=lambda row: (
                float(row.get("profile_scores", {}).get("fast_frontier", row.get("score", float("-inf")))),
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
                float(row.get("pace_kph", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
        "survival_gate": sorted(survival_rows, key=lambda row: float(row.get("score", float("-inf"))), reverse=True)[
            :per_bucket
        ],
        "late_frontier_distance": sorted(
            frontier_rows,
            key=lambda row: (
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
                float(row.get("pace_kph", float("-inf")) or float("-inf")),
                float(row.get("final_speed_kph", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
        "frontier_distance": sorted(
            frontier_rows,
            key=lambda row: (
                float(row.get("profile_scores", {}).get("frontier_recovery", row.get("score", float("-inf")))),
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
                float(row.get("final_speed_kph", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
        "farthest_distance": sorted(
            ranked,
            key=lambda row: float(row.get("best_progress_m", float("-inf")) or float("-inf")),
            reverse=True,
        )[:per_bucket],
        "far_fast": sorted(
            speed_rows,
            key=lambda row: (
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
                float(row.get("pace_kph", float("-inf")) or float("-inf")),
                float(row.get("final_speed_kph", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
        "fastest_pace": sorted(
            speed_rows,
            key=lambda row: (
                float(row.get("pace_kph", float("-inf")) or float("-inf")),
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
        "cleanest_distance": sorted(
            ranked,
            key=lambda row: (
                float(row.get("profile_scores", {}).get("clean_distance", row.get("score", float("-inf")))),
                float(row.get("best_progress_m", float("-inf")) or float("-inf")),
            ),
            reverse=True,
        )[:per_bucket],
    }
    buckets = {name: rows for name, rows in buckets.items() if rows}
    if not buckets:
        buckets["fallback_top_score"] = ranked[: max(1, min(pool_size, len(ranked)))]
    summary = {
        "parent_pool_size": pool_size,
        "parent_bucket_counts": {name: len(rows) for name, rows in buckets.items()},
        "parent_survival_floor_m": survival_floor_m,
        "parent_late_floor_m": late_floor_m,
        "parent_late_frontier_active": late_frontier_active,
        "parent_speed_bucket_floor_m": speed_bucket_floor_m,
    }
    return buckets, summary


def _sample_parent_row(
    buckets: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any]]:
    names = [name for name, rows in buckets.items() if rows]
    if not names:
        raise ValueError("Cannot sample parent from empty parent buckets")
    bucket_weights = {
        "fastest_valid_lap": 9.5,
        "fast_valid_lap_score": 9.0,
        "fast_frontier_score": 7.2,
        "late_frontier_distance": 4.0,
        "frontier_distance": 3.0,
        "farthest_distance": 1.3,
        "survival_gate": 1.1,
        "far_fast": 4.8,
        "cleanest_distance": 1.3,
        "fastest_pace": 4.2,
        "fallback_top_score": 1.0,
    }
    weights = np.asarray([bucket_weights.get(name, 1.0) for name in names], dtype=np.float64)
    weights /= weights.sum()
    bucket_name = str(rng.choice(names, p=weights))
    rows = buckets[bucket_name]
    return bucket_name, rows[_rank_biased_index(len(rows), rng)]


def _effective_immigrant_counts(
    config: EvolutionSearchConfig,
    *,
    generation: int,
    remaining_slots: int,
    quality_pass_rate: float,
    frontier_best_m: float = 0.0,
) -> dict[str, int]:
    total = min(config.random_immigrants, max(0, remaining_slots))
    if config.adaptive_immigrants and generation >= 2 and total > 0:
        if frontier_best_m >= 5000.0:
            total = max(config.min_random_immigrants, int(round(total * 0.16)))
        elif frontier_best_m >= 4000.0:
            total = max(config.min_random_immigrants, int(round(total * 0.22)))
        elif frontier_best_m >= 3000.0:
            total = max(config.min_random_immigrants, int(round(total * 0.30)))
        elif quality_pass_rate >= 0.60:
            total = max(config.min_random_immigrants, int(round(total * 0.35)))
        elif quality_pass_rate >= 0.40:
            total = max(config.min_random_immigrants, int(round(total * 0.50)))
        elif quality_pass_rate >= 0.25:
            total = max(config.min_random_immigrants, int(round(total * 0.70)))
        total = min(total, remaining_slots)
    smart = int(round(total * config.smart_immigrant_fraction))
    pure = total - smart
    return {"total": total, "smart": smart, "pure": pure}


def _next_population(
    *,
    ranked: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    snapshots: list[StateSnapshot],
    config: EvolutionSearchConfig,
    rng: np.random.Generator,
    action_names: list[str],
    created_generation: int,
    survival_floor_m: float,
    quality_pass_rate: float,
    plateau: dict[str, Any] | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    plateau = plateau or {}
    plateau_active = bool(plateau.get("plateau_active", False))
    frontier_best_m = max((float(row.get("best_progress_m", 0.0) or 0.0) for row in ranked), default=0.0)
    late_frontier_active = frontier_best_m >= config.late_frontier_trigger_m
    elite_rows = _select_elite_rows(ranked, config)
    selected_elite_count = len(elite_rows)
    if plateau_active and elite_rows:
        effective_plateau_elite_fraction = config.plateau_elite_fraction
        if late_frontier_active:
            effective_plateau_elite_fraction = max(effective_plateau_elite_fraction, 0.75)
        plateau_elite_count = max(1, int(math.ceil(len(elite_rows) * effective_plateau_elite_fraction)))
        elite_rows = elite_rows[:plateau_elite_count]
    elites = [
        Candidate(
            genome=genome_from_mapping(row["genome"]),
            snapshot_index=int(row["snapshot_index"]),
            lineage={
                "source": "elite_copy",
                "created_generation": created_generation,
                "parent": _row_reference(row),
                "parent_generation": int(row["generation"]),
                "parent_candidate_index": int(row["candidate_index"]),
                "parent_seed": int(row["seed"]),
                "previous_lineage": row.get("lineage", {}),
            },
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
                lineage={"source": "random_reseed", "created_generation": created_generation},
            )
        ]
    parent_buckets, parent_summary = _parent_buckets(ranked, config, survival_floor_m=survival_floor_m)
    next_candidates = list(elites)
    immigrant_counts = _effective_immigrant_counts(
        config,
        generation=created_generation,
        remaining_slots=max(0, config.population - len(next_candidates)),
        quality_pass_rate=quality_pass_rate,
        frontier_best_m=frontier_best_m,
    )
    reproduction_counts: Counter[str] = Counter({"elite_copy": len(next_candidates)})
    source_bucket_counts: Counter[str] = Counter()
    while len(next_candidates) < config.population - immigrant_counts["total"]:
        source_bucket, parent_row = _sample_parent_row(parent_buckets, rng)
        parent = Candidate(
            genome=genome_from_mapping(parent_row["genome"]),
            snapshot_index=int(parent_row["snapshot_index"]),
            lineage=dict(parent_row.get("lineage", {})),
        )
        parent_genome = parent.genome
        genome = parent.genome
        partner_reference: dict[str, Any] | None = None
        crossover_used = False
        if len(elites) > 1 and rng.random() < config.crossover_rate:
            partner_bucket, partner_row = _sample_parent_row(parent_buckets, rng)
            other = Candidate(
                genome=genome_from_mapping(partner_row["genome"]),
                snapshot_index=int(partner_row["snapshot_index"]),
                lineage=dict(partner_row.get("lineage", {})),
            )
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
            partner_reference = {**_row_reference(partner_row), "source_bucket": partner_bucket}
            crossover_used = True
            reproduction_counts["crossover"] += 1
        mutation_history: list[dict[str, Any]] = []
        parent_is_frontier = (
            float(parent_row.get("best_progress_m", 0.0) or 0.0)
            >= max(config.frontier_parent_min_progress_m, survival_floor_m)
        )
        mutation_rate = 1.0 if plateau_active and parent_is_frontier else config.mutation_rate
        if rng.random() < mutation_rate:
            mutation_scale = 1.0
            if plateau_active and parent_is_frontier:
                mutation_scale = 0.35 if late_frontier_active else 0.75
            genome, mutation_metadata = mutate_genome_with_metadata(
                genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
                min_phase_progress_m=config.min_phase_progress_m,
                max_phase_progress_m=config.max_phase_progress_m,
                mutation_scale=mutation_scale,
            )
            mutation_history.append(mutation_metadata)
            reproduction_counts["mutated_offspring"] += 1
        if plateau_active and parent_is_frontier:
            extra_mutations = min(config.plateau_extra_mutations, 1) if late_frontier_active else config.plateau_extra_mutations
            for _ in range(extra_mutations):
                genome, mutation_metadata = mutate_genome_with_metadata(
                    genome,
                    rng,
                    action_names=action_names,
                    min_phase_steps=config.min_phase_steps,
                    max_phase_steps=config.max_phase_steps,
                    max_phases=config.max_phases,
                    min_phase_progress_m=config.min_phase_progress_m,
                    max_phase_progress_m=config.max_phase_progress_m,
                    mutation_scale=0.25 if late_frontier_active else 0.75,
                )
                mutation_history.append(mutation_metadata)
                reproduction_counts["plateau_extra_mutation"] += 1
        snapshot_index = parent.snapshot_index
        if len(snapshots) > 1 and rng.random() < config.start_mutation_rate:
            snapshot_index = int(rng.integers(0, len(snapshots)))
        distance = _genome_distance(parent_genome, genome)
        source_bucket_counts[source_bucket] += 1
        next_candidates.append(
            Candidate(
                genome=genome,
                snapshot_index=snapshot_index,
                lineage={
                    "source": "offspring",
                    "created_generation": created_generation,
                    "source_bucket": source_bucket,
                    "parent": _row_reference(parent_row),
                    "parent_generation": int(parent_row["generation"]),
                    "parent_candidate_index": int(parent_row["candidate_index"]),
                    "parent_seed": int(parent_row["seed"]),
                    "crossover_used": crossover_used,
                    "crossover_partner": partner_reference,
                    "crossover_partner_generation": partner_reference.get("generation") if partner_reference else None,
                    "crossover_partner_candidate_index": (
                        partner_reference.get("candidate_index") if partner_reference else None
                    ),
                    "crossover_partner_seed": partner_reference.get("seed") if partner_reference else None,
                    "mutation": mutation_history[-1]
                    if mutation_history
                    else {"mutation_type": "none", "mutation_sigma": None, "reset_count": 0},
                    "mutation_history": mutation_history,
                    "mutation_type": mutation_history[-1].get("mutation_type") if mutation_history else "none",
                    "mutation_sigma": mutation_history[-1].get("mutation_sigma") if mutation_history else None,
                    "reset_count": sum(int(item.get("reset_count", 0) or 0) for item in mutation_history),
                    "plateau_mode": plateau_active,
                    "genome_distance_from_parent": distance,
                    "previous_lineage": parent_row.get("lineage", {}),
                },
            )
        )
        reproduction_counts["offspring"] += 1

    current_source_rows = sorted(
        ranked,
        key=_fast_lap_rank_key,
        reverse=True,
    )[: max(len(elite_rows), config.elite_count * 3, 8)]
    if not current_source_rows:
        current_source_rows = elite_rows or ranked
    global_source_rows = sorted(
        best_rows or current_source_rows,
        key=_fast_lap_rank_key,
        reverse=True,
    )
    for _ in range(immigrant_counts["smart"]):
        if len(next_candidates) >= config.population:
            break
        use_current = bool(rng.random() < config.smart_immigrant_current_fraction) or not global_source_rows
        source_rows = current_source_rows if use_current else global_source_rows
        parent_row = source_rows[_rank_biased_index(len(source_rows), rng)]
        parent_genome = genome_from_mapping(parent_row["genome"])
        genome = parent_genome
        mutation_rounds = 1 if late_frontier_active else 1 + int(rng.random() < 0.35)
        mutation_history: list[dict[str, Any]] = []
        for _round in range(mutation_rounds):
            genome, mutation_metadata = mutate_genome_with_metadata(
                genome,
                rng,
                action_names=action_names,
                min_phase_steps=config.min_phase_steps,
                max_phase_steps=config.max_phase_steps,
                max_phases=config.max_phases,
                min_phase_progress_m=config.min_phase_progress_m,
                max_phase_progress_m=config.max_phase_progress_m,
                mutation_scale=0.40 if late_frontier_active else 1.0,
            )
            mutation_history.append(mutation_metadata)
        immigrant_type = "smart_current_elite" if use_current else "smart_global_elite"
        next_candidates.append(
            Candidate(
                genome=genome,
                snapshot_index=int(parent_row["snapshot_index"]),
                lineage={
                    "source": "smart_immigrant",
                    "immigrant_type": immigrant_type,
                    "created_generation": created_generation,
                    "source_bucket": immigrant_type,
                    "parent": _row_reference(parent_row),
                    "parent_generation": int(parent_row["generation"]),
                    "parent_candidate_index": int(parent_row["candidate_index"]),
                    "parent_seed": int(parent_row["seed"]),
                    "mutation_history": mutation_history,
                    "mutation_type": mutation_history[-1].get("mutation_type") if mutation_history else "none",
                    "mutation_sigma": mutation_history[-1].get("mutation_sigma") if mutation_history else None,
                    "reset_count": sum(int(item.get("reset_count", 0) or 0) for item in mutation_history),
                    "genome_distance_from_parent": _genome_distance(parent_genome, genome),
                    "previous_lineage": parent_row.get("lineage", {}),
                },
            )
        )
        reproduction_counts["smart_immigrant"] += 1
        source_bucket_counts[immigrant_type] += 1

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
                lineage={
                    "source": "pure_random_immigrant",
                    "immigrant_type": "pure_random",
                    "created_generation": created_generation,
                    "source_bucket": "pure_random",
                },
            )
        )
        reproduction_counts["pure_random_immigrant"] += 1
        source_bucket_counts["pure_random"] += 1
    summary = {
        **parent_summary,
        "created_generation": created_generation,
        "plateau_mode": plateau_active,
        "plateau_report": plateau,
        "selected_elite_count": selected_elite_count,
        "elite_copy_count": reproduction_counts["elite_copy"],
        "offspring_count": reproduction_counts["offspring"],
        "mutated_offspring_count": reproduction_counts["mutated_offspring"],
        "plateau_extra_mutation_count": reproduction_counts["plateau_extra_mutation"],
        "crossover_count": reproduction_counts["crossover"],
        "smart_immigrant_count": reproduction_counts["smart_immigrant"],
        "pure_random_immigrant_count": reproduction_counts["pure_random_immigrant"],
        "configured_random_immigrants": config.random_immigrants,
        "effective_immigrant_count": immigrant_counts["total"],
        "immigrant_quality_pass_rate": quality_pass_rate,
        "frontier_best_m": frontier_best_m,
        "late_frontier_active": late_frontier_active,
        "source_bucket_selection_counts": dict(source_bucket_counts),
    }
    return next_candidates, summary


def _generation_summary(generation: int, ranked: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    termination_reasons = Counter(str(row["termination_reason"]) for row in ranked)
    lineage_sources = Counter(str(row.get("lineage", {}).get("source", "unknown")) for row in ranked)
    completion_count = sum(1 for row in ranked if row["segment_complete"])
    milestone_count = sum(1 for row in ranked if row.get("target_reached", row["segment_complete"]))
    farthest_attempt = max(ranked, key=lambda row: float(row["best_progress_m"]), default=None)
    avg_progress_m = (
        sum(float(row["best_progress_m"]) for row in ranked) / max(1, len(ranked))
        if ranked
        else 0.0
    )
    avg_pace_kph = (
        sum(float(row.get("pace_kph", 0.0) or 0.0) for row in ranked) / max(1, len(ranked))
        if ranked
        else 0.0
    )
    top_decile_count = max(1, int(math.ceil(len(ranked) * 0.10))) if ranked else 0
    top_decile_distance_rows = sorted(
        ranked,
        key=lambda row: float(row.get("best_progress_m", float("-inf")) or float("-inf")),
        reverse=True,
    )[:top_decile_count]
    top_decile_pace_rows = sorted(
        ranked,
        key=lambda row: float(row.get("pace_kph", float("-inf")) or float("-inf")),
        reverse=True,
    )[:top_decile_count]
    top_decile_distance_m = (
        sum(float(row.get("best_progress_m", 0.0) or 0.0) for row in top_decile_distance_rows)
        / max(1, len(top_decile_distance_rows))
        if top_decile_distance_rows
        else 0.0
    )
    top_decile_pace_kph = (
        sum(float(row.get("pace_kph", 0.0) or 0.0) for row in top_decile_pace_rows)
        / max(1, len(top_decile_pace_rows))
        if top_decile_pace_rows
        else 0.0
    )
    valid_lap_rows = [row for row in ranked if _row_is_valid_finish(row)]
    valid_lap_count = len(valid_lap_rows)
    fastest_valid_lap = min(valid_lap_rows, key=_row_elapsed_s, default=None)
    fastest_valid_lap_s = _row_elapsed_s(fastest_valid_lap) if fastest_valid_lap is not None else None
    average_valid_lap_s = (
        sum(_row_elapsed_s(row) for row in valid_lap_rows) / valid_lap_count
        if valid_lap_rows
        else None
    )
    fastest_valid_lap_pace_kph = (
        float(fastest_valid_lap.get("pace_kph", 0.0) or 0.0) if fastest_valid_lap is not None else None
    )
    progress_threshold_counts = {
        str(threshold): sum(1 for row in ranked if float(row.get("best_progress_m", 0.0) or 0.0) >= threshold)
        for threshold in (100, 300, 450, 650, 1000, 1220, 1500, 2000, 2400, 3000, 4000, 5000, 5500, 5793)
    }
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
        "generation_average_pace_kph": avg_pace_kph,
        "generation_top_decile_distance_m": top_decile_distance_m,
        "generation_top_decile_pace_kph": top_decile_pace_kph,
        "valid_lap_count": valid_lap_count,
        "valid_lap_rate": valid_lap_count / max(1, len(ranked)),
        "fastest_valid_lap_s": fastest_valid_lap_s,
        "average_valid_lap_s": average_valid_lap_s,
        "fastest_valid_lap_pace_kph": fastest_valid_lap_pace_kph,
        "fastest_valid_lap_attempt": fastest_valid_lap,
        "progress_threshold_counts": progress_threshold_counts,
        "best_remaining_m": ranked[0]["remaining_m"] if ranked else None,
        "completion_count": completion_count,
        "completion_rate": completion_count / max(1, len(ranked)),
        "milestone_count": milestone_count,
        "milestone_rate": milestone_count / max(1, len(ranked)),
        "termination_reasons": dict(termination_reasons),
        "lineage_source_counts": dict(lineage_sources),
        "profile_leaders": profile_leaders,
        "best_attempt": ranked[0] if ranked else None,
        "farthest_attempt": farthest_attempt,
    }


def _merge_best_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    combined = [*existing, *new_rows]
    fastest_valid = sorted(
        [row for row in combined if _row_is_valid_finish(row)],
        key=lambda row: (
            _row_elapsed_s(row),
            -float(row.get("pace_kph", 0.0) or 0.0),
            -float(row.get("score", 0.0) or 0.0),
        ),
    )
    top_score = sorted(combined, key=lambda row: float(row["score"]), reverse=True)
    combined = [*fastest_valid[: max(1, limit // 4)], *top_score]
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
    valid_rows = [row for row in all_rows if _row_is_valid_finish(row)]
    fastest_valid = min(valid_rows, key=_row_elapsed_s, default=None)
    if fastest_valid is not None:
        selected.append(_tagged_row(fastest_valid, "fastest_valid_lap"))
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
                "lineage": row.get("lineage", {}),
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


def _effective_gpu_attempts_mode(config: EvolutionSearchConfig) -> str:
    if config.gpu_attempts_mode != "auto":
        return config.gpu_attempts_mode
    if (
        config.backend == "gpu"
        and config.gpu_run_mode == "production"
        and config.gpu_telemetry_mode == "none"
        and not config.gpu_parity_check
    ):
        return "compact"
    return "full"


def _effective_gpu_cpu_replay_top_k(
    *,
    gpu_run_mode: str,
    gpu_verify_top_k: int,
    top_k: int,
    gpu_cpu_replay_top_k: int | None,
    explicit_gpu_cpu_replay_top_k: bool,
) -> int | None:
    if gpu_cpu_replay_top_k is not None:
        return max(0, int(gpu_cpu_replay_top_k))
    if gpu_run_mode == "production" and not explicit_gpu_cpu_replay_top_k:
        # The fused rollout is fast enough to expose long-horizon float32/control
        # sensitivity. Keep production compact, but CPU-rerank a small selected set
        # unless the caller explicitly opts out with --gpu-cpu-replay-top-k 0.
        return max(16, int(gpu_verify_top_k), int(top_k) * 2)
    return None


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
        "--backend",
        config.backend,
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
        "--telemetry-compression",
        config.telemetry_compression,
        "--checkpoint-every-generations",
        str(config.checkpoint_every_generations),
        "--survival-floor-stages-m",
        ",".join(str(stage) for stage in config.survival_floor_stages_m),
        "--survival-floor-pass-rate",
        str(config.survival_floor_pass_rate),
        "--parent-pool-size",
        str(config.parent_pool_size),
        "--min-random-immigrants",
        str(config.min_random_immigrants),
        "--smart-immigrant-fraction",
        str(config.smart_immigrant_fraction),
        "--smart-immigrant-current-fraction",
        str(config.smart_immigrant_current_fraction),
        "--frontier-focus-start-m",
        str(config.frontier_focus_start_m),
        "--frontier-focus-end-m",
        str(config.frontier_focus_end_m),
        "--frontier-parent-min-progress-m",
        str(config.frontier_parent_min_progress_m),
        "--plateau-generations",
        str(config.plateau_generations),
        "--plateau-distance-epsilon-m",
        str(config.plateau_distance_epsilon_m),
        "--plateau-average-improvement-m",
        str(config.plateau_average_improvement_m),
        "--plateau-elite-fraction",
        str(config.plateau_elite_fraction),
        "--plateau-extra-mutations",
        str(config.plateau_extra_mutations),
        "--late-frontier-trigger-m",
        str(config.late_frontier_trigger_m),
    ]
    if config.backend == "gpu":
        parts.extend(
            [
                "--gpu-device",
                config.gpu_device,
                "--gpu-engine",
                config.gpu_engine,
                "--gpu-run-mode",
                config.gpu_run_mode,
                "--gpu-dtype",
                config.gpu_dtype,
                "--gpu-batch-size",
                str(config.gpu_batch_size) if config.gpu_batch_size is not None else "auto",
                "--gpu-static-batch-size",
                str(config.gpu_static_batch_size) if config.gpu_static_batch_size is not None else "auto",
                "--gpu-verify-top-k",
                str(config.gpu_verify_top_k),
                "--gpu-verify-elite-multiplier",
                str(config.gpu_verify_elite_multiplier),
                "--gpu-telemetry-mode",
                config.gpu_telemetry_mode,
                "--gpu-profile",
                config.gpu_profile,
                "--gpu-chunk-steps",
                str(config.gpu_chunk_steps),
                "--gpu-fast-geometry",
                config.gpu_fast_geometry,
                "--gpu-collision-mode",
                config.gpu_collision_mode,
                "--gpu-attempts-mode",
                config.gpu_attempts_mode,
            ]
        )
        if config.gpu_cpu_replay_top_k is not None:
            parts.extend(["--gpu-cpu-replay-top-k", str(config.gpu_cpu_replay_top_k)])
        if config.gpu_profile_output is not None:
            parts.extend(["--gpu-profile-output", _quote_cli(config.gpu_profile_output)])
        if config.gpu_parity_check:
            parts.append("--gpu-parity-check")
        if config.gpu_fallback_to_cpu:
            parts.append("--gpu-fallback-to-cpu")
        if config.gpu_compile:
            parts.append("--gpu-compile")
        if config.gpu_disable_early_stop:
            parts.append("--gpu-disable-early-stop")
    if config.max_steps_schedule:
        parts.extend(
            [
                "--max-steps-schedule",
                ",".join(f"{generation}:{steps}" for generation, steps in config.max_steps_schedule),
            ]
        )
    if not config.adaptive_survival_floor:
        parts.append("--no-adaptive-survival-floor")
    if not config.survival_floor_leader_jump:
        parts.append("--no-survival-floor-leader-jump")
    if not config.adaptive_immigrants:
        parts.append("--no-adaptive-immigrants")
    if not config.plateau_mode:
        parts.append("--no-plateau-mode")
    if not gates.terminate_at_target_progress:
        parts.append("--no-target-termination")
    if config.progress_every_generation:
        parts.append("--progress-every-generation")
    if config.all_candidate_telemetry_dir is not None:
        parts.extend(["--all-candidate-telemetry-dir", _quote_cli(config.all_candidate_telemetry_dir)])
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
    survival_floor_index: int,
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
        "survival_floor_index": survival_floor_index,
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
            lineage={"source": "initial_random", "created_generation": 0},
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
    _validate_config(config, gates)
    action_names = [name for name, _, _, _ in actions_for_action_set(config.action_set)]
    sim_config = _sim_config_for_generation(config, 0)
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
    all_candidate_telemetry_dir = (
        Path(config.all_candidate_telemetry_dir)
        if config.all_candidate_telemetry_dir is not None
        else selected_dir
    )
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
    survival_floor_index = 0
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
        survival_floor_index = int(
            checkpoint.get(
                "survival_floor_index",
                generation_summaries[-1].get("next_survival_floor_index", 0) if generation_summaries else 0,
            )
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
    use_gpu_backend = config.backend == "gpu"
    gpu_backend: Any | None = None
    gpu_profile_output_dir = config.gpu_profile_output or (output_dir / "gpu_profiles")
    if not use_gpu_backend and resolved_workers > 1:
        executor = ProcessPoolExecutor(max_workers=resolved_workers)
    try:
        for generation in range(start_generation, config.generations):
            started = time.perf_counter()
            generation_sim_config = _sim_config_for_generation(config, generation)
            backend_metrics: dict[str, Any] = {"backend": "cpu"}
            evaluation_started = time.perf_counter()
            if use_gpu_backend:
                try:
                    if gpu_backend is None:
                        gpu_backend = _make_gpu_backend(
                            config=config,
                            profile_output_dir=gpu_profile_output_dir,
                        )
                    evaluated, backend_metrics = _evaluate_population_gpu(
                        candidates=population,
                        snapshots=snapshots,
                        sim_config=generation_sim_config,
                        gates=gates,
                        generation=generation,
                        seed=config.seed,
                        scoring_profiles=config.scoring_profiles,
                        frontier_focus_start_m=config.frontier_focus_start_m,
                        frontier_focus_end_m=config.frontier_focus_end_m,
                        capture_step_telemetry=config.telemetry_selection == "all",
                        stream_telemetry_dir=all_candidate_telemetry_dir if config.telemetry_selection == "all" else None,
                        telemetry_compression=config.telemetry_compression,
                        config=config,
                        profile_output_dir=gpu_profile_output_dir,
                        backend=gpu_backend,
                    )
                except Exception as exc:
                    gpu_backend = None
                    if not config.gpu_fallback_to_cpu:
                        raise
                    print(f"evolution_backend_fallback reason={type(exc).__name__}:{exc}", flush=True)
                    evaluated = _evaluate_population(
                        candidates=population,
                        snapshots=snapshots,
                        sim_config=generation_sim_config,
                        gates=gates,
                        generation=generation,
                        seed=config.seed,
                        workers=resolved_workers,
                        worker_chunk_size=config.worker_chunk_size,
                        scoring_profiles=config.scoring_profiles,
                        frontier_focus_start_m=config.frontier_focus_start_m,
                        frontier_focus_end_m=config.frontier_focus_end_m,
                        capture_step_telemetry=config.telemetry_selection == "all",
                        stream_telemetry_dir=all_candidate_telemetry_dir if config.telemetry_selection == "all" else None,
                        telemetry_compression=config.telemetry_compression,
                        executor=executor,
                    )
                    backend_metrics = {"backend": "cpu", "gpu_fallback_reason": f"{type(exc).__name__}: {exc}"}
            else:
                evaluated = _evaluate_population(
                    candidates=population,
                    snapshots=snapshots,
                    sim_config=generation_sim_config,
                    gates=gates,
                    generation=generation,
                    seed=config.seed,
                    workers=resolved_workers,
                    worker_chunk_size=config.worker_chunk_size,
                    scoring_profiles=config.scoring_profiles,
                    frontier_focus_start_m=config.frontier_focus_start_m,
                    frontier_focus_end_m=config.frontier_focus_end_m,
                    capture_step_telemetry=config.telemetry_selection == "all",
                    stream_telemetry_dir=all_candidate_telemetry_dir if config.telemetry_selection == "all" else None,
                    telemetry_compression=config.telemetry_compression,
                    executor=executor,
                )
            population_evaluation_seconds = time.perf_counter() - evaluation_started
            telemetry_extract_started = time.perf_counter()
            for row in evaluated:
                captured = row.pop("captured_telemetry", None)
                if captured is not None:
                    captured_telemetry_by_key[_row_identity(row)] = captured
            captured_telemetry_extract_seconds = time.perf_counter() - telemetry_extract_started
            elapsed_s = time.perf_counter() - started
            sort_started = time.perf_counter()
            ranked = sorted(evaluated, key=lambda row: float(row["score"]), reverse=True)
            selection_sort_seconds = time.perf_counter() - sort_started
            latest_ranked = ranked
            attempt_count += len(ranked)
            attempts_write_started = time.perf_counter()
            _append_jsonl(attempts_path, ranked)
            attempts_write_seconds = time.perf_counter() - attempts_write_started
            summary = _generation_summary(generation, ranked, elapsed_s)
            summary["max_steps"] = generation_sim_config.max_steps
            summary.update(backend_metrics)
            summary.update(
                {
                    "timing_population_evaluation_seconds": population_evaluation_seconds,
                    "timing_captured_telemetry_extract_seconds": captured_telemetry_extract_seconds,
                    "timing_selection_sort_seconds": selection_sort_seconds,
                    "timing_attempts_jsonl_write_seconds": attempts_write_seconds,
                }
            )
            survival_report = _survival_floor_report(ranked, config, active_index=survival_floor_index)
            summary.update(survival_report)
            plateau = _plateau_report(generation_summaries, summary, config)
            summary.update(plateau)
            global_best_distance_m = max(
                global_best_distance_m,
                float(summary.get("generation_best_distance_m", 0.0) or 0.0),
            )
            summary["global_best_distance_m"] = global_best_distance_m
            best_merge_started = time.perf_counter()
            best_rows = _merge_best_rows(best_rows, ranked, limit=best_limit)
            summary["timing_best_row_merge_seconds"] = time.perf_counter() - best_merge_started
            best_artifact_started = time.perf_counter()
            _write_json(output_dir / "best_so_far.json", best_rows[0] if best_rows else {})
            _write_generation_genomes(output_dir, generation, ranked, config.top_k)
            summary["timing_best_artifact_write_seconds"] = time.perf_counter() - best_artifact_started

            if generation < config.generations - 1:
                reproduction_started = time.perf_counter()
                survival_floor_index = int(survival_report["next_survival_floor_index"])
                base_stage_key = f"{float(config.survival_floor_stages_m[0]):.1f}"
                leader_progress_m = float(survival_report.get("survival_floor_leader_progress_m", 0.0) or 0.0)
                frontier_quality_floor_m = max(
                    float(config.survival_floor_stages_m[0]),
                    min(float(survival_report["next_survival_floor_m"]), max(450.0, leader_progress_m - 1000.0)),
                )
                frontier_quality_rate = sum(
                    1
                    for row in ranked
                    if float(row.get("best_progress_m", 0.0) or 0.0) >= frontier_quality_floor_m
                ) / max(1, len(ranked))
                quality_pass_rate = max(
                    float(survival_report["survival_floor_pass_rate"]),
                    float(survival_report["survival_floor_stage_rates"].get(base_stage_key, 0.0)),
                    frontier_quality_rate,
                )
                summary["frontier_quality_floor_m"] = frontier_quality_floor_m
                summary["frontier_quality_pass_rate"] = frontier_quality_rate
                population, reproduction_summary = _next_population(
                    ranked=ranked,
                    best_rows=best_rows,
                    snapshots=snapshots,
                    config=config,
                    rng=rng,
                    action_names=action_names,
                    created_generation=generation + 1,
                    survival_floor_m=float(survival_report["next_survival_floor_m"]),
                    quality_pass_rate=quality_pass_rate,
                    plateau=plateau,
                )
                summary["next_population"] = reproduction_summary
                summary["timing_population_selection_mutation_seconds"] = (
                    time.perf_counter() - reproduction_started
                )
            else:
                summary["timing_population_selection_mutation_seconds"] = 0.0

            generation_summaries.append(summary)
            generation_summary_write_started = time.perf_counter()
            _append_jsonl(generation_summary_path, [summary])
            summary["timing_generation_summary_jsonl_write_seconds"] = (
                time.perf_counter() - generation_summary_write_started
            )

            if config.progress_every_generation:
                best = ranked[0] if ranked else {}
                farthest = summary.get("farthest_attempt") or {}
                next_population_summary = summary.get("next_population", {})
                fastest_valid_lap_s = summary.get("fastest_valid_lap_s")
                average_valid_lap_s = summary.get("average_valid_lap_s")
                timing_summary = (
                    f"timing_eval_s={float(summary.get('timing_population_evaluation_seconds', 0.0)):.3f} "
                    f"timing_gpu_rollout_s={float(summary.get('gpu_rollout_seconds', 0.0) or 0.0):.3f} "
                    f"timing_gpu_control_upload_s={float(summary.get('gpu_control_program_upload_seconds', 0.0) or 0.0):.3f} "
                    f"timing_gpu_materialize_s={float(summary.get('gpu_result_materialization_seconds', 0.0) or 0.0):.3f} "
                    f"timing_cpu_replay_s={float(summary.get('gpu_cpu_replay_seconds', 0.0) or 0.0):.3f} "
                    f"timing_attempt_write_s={float(summary.get('timing_attempts_jsonl_write_seconds', 0.0)):.3f} "
                    f"timing_reproduction_s={float(summary.get('timing_population_selection_mutation_seconds', 0.0)):.3f} "
                    f"timing_artifact_write_s={float(summary.get('timing_best_artifact_write_seconds', 0.0)):.3f} "
                    f"gpu_rollout_fraction={float(summary.get('gpu_rollout_time_fraction', 0.0) or 0.0):.3f} "
                    f"gpu_cpu_replay_fraction={float(summary.get('gpu_cpu_replay_time_fraction', 0.0) or 0.0):.3f}"
                )
                print(
                    "evolution_generation "
                    f"generation={generation} population={len(ranked)} "
                    f"max_steps={generation_sim_config.max_steps} "
                    f"leader_progress_m={float(best.get('best_progress_m', 0.0)):.3f} "
                    f"generation_best_distance_m={float(farthest.get('best_progress_m', 0.0)):.3f} "
                    f"generation_average_distance_m={float(summary.get('generation_average_distance_m', 0.0)):.3f} "
                    f"generation_average_pace_kph={float(summary.get('generation_average_pace_kph', 0.0)):.3f} "
                    f"top_decile_distance_m={float(summary.get('generation_top_decile_distance_m', 0.0)):.3f} "
                    f"top_decile_pace_kph={float(summary.get('generation_top_decile_pace_kph', 0.0)):.3f} "
                    f"valid_laps={int(summary.get('valid_lap_count', 0) or 0)} "
                    f"valid_lap_rate={float(summary.get('valid_lap_rate', 0.0) or 0.0):.3f} "
                    f"fastest_valid_lap_s={float(fastest_valid_lap_s) if fastest_valid_lap_s is not None else 0.0:.3f} "
                    f"average_valid_lap_s={float(average_valid_lap_s) if average_valid_lap_s is not None else 0.0:.3f} "
                    f"global_best_distance_m={global_best_distance_m:.3f} "
                    f"best_score={float(best.get('score', 0.0)):.3f} "
                    f"survival_floor_m={float(summary.get('survival_floor_m', 0.0)):.1f} "
                    f"survival_pass_rate={float(summary.get('survival_floor_pass_rate', 0.0)):.3f} "
                    f"next_survival_floor_m={float(summary.get('next_survival_floor_m', 0.0)):.1f} "
                    f"offspring={int(next_population_summary.get('offspring_count', 0) or 0)} "
                    f"smart_immigrants={int(next_population_summary.get('smart_immigrant_count', 0) or 0)} "
                    f"pure_randoms={int(next_population_summary.get('pure_random_immigrant_count', 0) or 0)} "
                    f"plateau_mode={int(bool(summary.get('plateau_active', False)))} "
                    f"gate_3000={int(summary.get('progress_threshold_counts', {}).get('3000', 0))} "
                    f"gate_4000={int(summary.get('progress_threshold_counts', {}).get('4000', 0))} "
                    f"gate_5000={int(summary.get('progress_threshold_counts', {}).get('5000', 0))} "
                    f"completion_rate={summary['completion_rate']:.3f} "
                    f"milestone_rate={summary['milestone_rate']:.3f} "
                    f"candidates_per_second={summary['candidates_per_second']:.3f} "
                    f"{timing_summary}",
                    flush=True,
                )

            if (
                config.checkpoint_every_generations > 0
                and (generation + 1) % config.checkpoint_every_generations == 0
            ):
                checkpoint_started = time.perf_counter()
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
                    survival_floor_index=survival_floor_index,
                )
                summary["timing_checkpoint_write_seconds"] = time.perf_counter() - checkpoint_started
            else:
                summary["timing_checkpoint_write_seconds"] = 0.0
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
                    survival_floor_index=survival_floor_index,
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

    postprocess_started = time.perf_counter()
    top_rows = best_rows[: config.top_k]
    top_row_keys = {_row_identity(row) for row in top_rows}
    telemetry_select_started = time.perf_counter()
    if config.backend == "gpu" and config.gpu_telemetry_mode == "none":
        telemetry_rows = []
    else:
        telemetry_rows = _select_telemetry_rows(
            attempts_path=attempts_path,
            top_rows=top_rows,
            latest_ranked=latest_ranked,
            config=config,
        )
    telemetry_selection_seconds = time.perf_counter() - telemetry_select_started
    elite_snapshots: list[StateSnapshot] = []
    telemetry_manifest: list[dict[str, Any]] = []
    top_rows_by_key = {_row_identity(row): row for row in top_rows}
    selected_telemetry_write_seconds = 0.0
    selected_telemetry_replay_seconds = 0.0
    for rank, row in enumerate(telemetry_rows):
        snapshot = snapshots[int(row["snapshot_index"])]
        row_sim_config = _sim_config_for_generation(config, int(row["generation"]))
        key = _row_identity(row)
        rows = captured_telemetry_by_key.get(key)
        sim: MonzaSim | None = None
        existing_telemetry = row.get("selected_telemetry")
        telemetry_path = Path(str(existing_telemetry)) if existing_telemetry is not None else None
        if telemetry_path is not None and telemetry_path.exists():
            final_row = dict(row.get("final_row") or {})
            if not final_row:
                final_row = _last_jsonl_row(telemetry_path)
        else:
            telemetry_replay_started = time.perf_counter()
            rows, sim = _run_candidate(
                sim_config=row_sim_config,
                snapshot=snapshot,
                genome=genome_from_mapping(row["genome"]),
                gates=gates,
                seed=int(row["seed"]),
                collect_full_telemetry=True,
            )
            selected_telemetry_replay_seconds += time.perf_counter() - telemetry_replay_started
            reason = _safe_slug(str(row.get("telemetry_selection_reason", "selected")))
            telemetry_suffix = _telemetry_file_suffix(config.telemetry_compression)
            target_telemetry_dir = (
                all_candidate_telemetry_dir
                if config.telemetry_selection == "all" and all_candidate_telemetry_dir is not None
                else selected_dir
            )
            target_telemetry_dir.mkdir(parents=True, exist_ok=True)
            telemetry_path = target_telemetry_dir / (
                f"evolution-{reason}-rank-{rank:03d}-gen-{int(row['generation']):03d}-"
                f"candidate-{int(row['candidate_index']):05d}-steps{telemetry_suffix}"
            )
            telemetry_write_started = time.perf_counter()
            _write_jsonl(telemetry_path, rows)
            selected_telemetry_write_seconds += time.perf_counter() - telemetry_write_started
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
                "final_speed_kph": final_row.get("speed_kph", row.get("final_speed_kph")),
                "final_lateral_error_m": final_row.get("lateral_error_m", row.get("final_lateral_error_m")),
                "final_heading_error_deg": final_row.get("heading_error_deg", row.get("final_heading_error_deg")),
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
    manifest_started = time.perf_counter()
    _write_json(
        selected_dir / "manifest.json",
        {
            "telemetry_selection": config.telemetry_selection,
            "telemetry_compression": config.telemetry_compression,
            "backend": config.backend,
            "gpu_telemetry_mode": config.gpu_telemetry_mode if config.backend == "gpu" else None,
            "all_candidate_telemetry_dir": str(all_candidate_telemetry_dir)
            if config.telemetry_selection == "all"
            else None,
            "trace_count": len(telemetry_manifest),
            "traces": telemetry_manifest,
        },
    )
    manifest_write_seconds = time.perf_counter() - manifest_started

    elite_library_path = output_dir / "elite_state_library.json"
    state_library_started = time.perf_counter()
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
            "config": dataclass_to_dict(config),
            "gates": asdict(gates),
        },
    )
    state_library_write_seconds = time.perf_counter() - state_library_started
    bridge_started = time.perf_counter()
    _write_ppo_bridge(
        output_dir,
        elite_library_path=elite_library_path,
        config=config,
        gates=gates,
        state_library=state_library,
        top_rows=top_rows,
    )
    bridge_write_seconds = time.perf_counter() - bridge_started
    if generation_summaries:
        generation_summaries[-1].update(
            {
                "timing_postprocess_total_seconds": time.perf_counter() - postprocess_started,
                "timing_selected_telemetry_selection_seconds": telemetry_selection_seconds,
                "timing_selected_telemetry_cpu_replay_seconds": selected_telemetry_replay_seconds,
                "timing_selected_telemetry_write_seconds": selected_telemetry_write_seconds,
                "timing_manifest_write_seconds": manifest_write_seconds,
                "timing_elite_state_library_write_seconds": state_library_write_seconds,
                "timing_ppo_bridge_write_seconds": bridge_write_seconds,
            }
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
        survival_floor_index=survival_floor_index,
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
    parser.add_argument("--backend", choices=sorted(EVOLUTION_BACKENDS), default="cpu")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--gpu-engine", choices=sorted(GPU_ENGINES), default="eager")
    parser.add_argument(
        "--gpu-run-mode",
        choices=sorted(GPU_RUN_MODES),
        default="parity",
        help=(
            "GPU evaluation mode. parity keeps conservative replay/telemetry defaults; "
            "production defaults to compact GPU-first search; postcheck keeps GPU artifacts ready for verification."
        ),
    )
    parser.add_argument("--gpu-dtype", choices=sorted(GPU_DTYPES), default="float32")
    parser.add_argument("--gpu-batch-size", default="auto")
    parser.add_argument("--gpu-static-batch-size", default="auto")
    parser.add_argument("--gpu-verify-top-k", type=int, default=4)
    parser.add_argument("--gpu-cpu-replay-top-k", type=int)
    parser.add_argument("--gpu-verify-elite-multiplier", type=int, default=2)
    parser.add_argument("--gpu-telemetry-mode", choices=sorted(GPU_TELEMETRY_MODES), default="selected")
    parser.add_argument("--gpu-parity-check", action="store_true")
    parser.add_argument("--gpu-fallback-to-cpu", action="store_true")
    parser.add_argument("--gpu-compile", action="store_true")
    parser.add_argument("--gpu-profile", choices=sorted(GPU_PROFILES), default="none")
    parser.add_argument("--gpu-profile-output", type=Path)
    parser.add_argument("--gpu-chunk-steps", type=int, default=256)
    parser.add_argument("--gpu-disable-early-stop", action="store_true")
    parser.add_argument("--gpu-fast-geometry", choices=sorted(GPU_FAST_GEOMETRIES), default="local_window")
    parser.add_argument("--gpu-collision-mode", choices=sorted(GPU_COLLISION_MODES), default="exact_all_segments")
    parser.add_argument(
        "--gpu-attempts-mode",
        choices=sorted(GPU_ATTEMPTS_MODES),
        default="auto",
        help=(
            "GPU attempts artifact detail. auto keeps parity/debug full and makes production/no-telemetry compact; "
            "compact omits per-candidate final_row diagnostics from the hot path."
        ),
    )
    parser.add_argument("--action-set", default="racing")
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--max-steps-schedule",
        default="",
        help="Optional comma-separated generation:max_steps schedule, e.g. 0:10000,5:15000,15:25000.",
    )
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
    parser.add_argument("--telemetry-compression", choices=sorted(TELEMETRY_COMPRESSIONS), default="none")
    parser.add_argument(
        "--all-candidate-telemetry-dir",
        type=Path,
        help="Optional external directory for --telemetry-selection all trace files; manifests stay in output-dir.",
    )
    parser.add_argument("--checkpoint-every-generations", type=int, default=1)
    parser.add_argument("--progress-every-generation", action="store_true")
    parser.add_argument("--no-adaptive-survival-floor", action="store_true")
    parser.add_argument("--survival-floor-stages-m", default="450,1000,1220,1500,2000,2400,3000,4000,5000")
    parser.add_argument("--survival-floor-pass-rate", type=float, default=0.30)
    parser.add_argument("--no-survival-floor-leader-jump", action="store_true")
    parser.add_argument("--parent-pool-size", type=int, default=0)
    parser.add_argument("--no-adaptive-immigrants", action="store_true")
    parser.add_argument("--min-random-immigrants", type=int, default=4)
    parser.add_argument("--smart-immigrant-fraction", type=float, default=0.50)
    parser.add_argument("--smart-immigrant-current-fraction", type=float, default=0.90)
    parser.add_argument("--frontier-focus-start-m", type=float, default=2200.0)
    parser.add_argument("--frontier-focus-end-m", type=float, default=2600.0)
    parser.add_argument("--frontier-parent-min-progress-m", type=float, default=2000.0)
    parser.add_argument("--late-frontier-trigger-m", type=float, default=4000.0)
    parser.add_argument("--no-plateau-mode", action="store_true")
    parser.add_argument("--plateau-generations", type=int, default=3)
    parser.add_argument("--plateau-distance-epsilon-m", type=float, default=8.0)
    parser.add_argument("--plateau-average-improvement-m", type=float, default=80.0)
    parser.add_argument("--plateau-elite-fraction", type=float, default=0.50)
    parser.add_argument("--plateau-extra-mutations", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    explicit_gpu_cpu_replay_top_k = any(
        item == "--gpu-cpu-replay-top-k" or item.startswith("--gpu-cpu-replay-top-k=") for item in raw_argv
    )
    explicit_gpu_telemetry_mode = any(
        item == "--gpu-telemetry-mode" or item.startswith("--gpu-telemetry-mode=") for item in raw_argv
    )
    resume_path = args.resume
    output_dir = args.output_dir
    if output_dir is None and resume_path is not None:
        output_dir = _checkpoint_path(resume_path).parent
    gpu_telemetry_mode = args.gpu_telemetry_mode
    gpu_cpu_replay_top_k = _effective_gpu_cpu_replay_top_k(
        gpu_run_mode=args.gpu_run_mode,
        gpu_verify_top_k=max(0, args.gpu_verify_top_k),
        top_k=max(1, args.top_k),
        gpu_cpu_replay_top_k=args.gpu_cpu_replay_top_k,
        explicit_gpu_cpu_replay_top_k=explicit_gpu_cpu_replay_top_k,
    )
    if args.gpu_run_mode == "production":
        if not explicit_gpu_telemetry_mode:
            gpu_telemetry_mode = "none"
    elif args.gpu_run_mode == "postcheck":
        if not explicit_gpu_telemetry_mode:
            gpu_telemetry_mode = "top"

    config = EvolutionSearchConfig(
        backend=args.backend,
        action_set=args.action_set,
        observation_profile=args.observation_profile,
        max_steps=max(1, args.max_steps),
        max_steps_schedule=_parse_max_steps_schedule(args.max_steps_schedule),
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
        telemetry_compression=args.telemetry_compression,
        all_candidate_telemetry_dir=args.all_candidate_telemetry_dir,
        checkpoint_every_generations=max(0, args.checkpoint_every_generations),
        progress_every_generation=bool(args.progress_every_generation),
        adaptive_survival_floor=not bool(args.no_adaptive_survival_floor),
        survival_floor_stages_m=_parse_float_csv(args.survival_floor_stages_m),
        survival_floor_pass_rate=float(np.clip(args.survival_floor_pass_rate, 0.0, 1.0)),
        survival_floor_leader_jump=not bool(args.no_survival_floor_leader_jump),
        parent_pool_size=max(0, args.parent_pool_size),
        adaptive_immigrants=not bool(args.no_adaptive_immigrants),
        min_random_immigrants=max(0, args.min_random_immigrants),
        smart_immigrant_fraction=float(np.clip(args.smart_immigrant_fraction, 0.0, 1.0)),
        smart_immigrant_current_fraction=float(np.clip(args.smart_immigrant_current_fraction, 0.0, 1.0)),
        frontier_focus_start_m=args.frontier_focus_start_m,
        frontier_focus_end_m=args.frontier_focus_end_m,
        frontier_parent_min_progress_m=max(0.0, args.frontier_parent_min_progress_m),
        late_frontier_trigger_m=max(0.0, args.late_frontier_trigger_m),
        plateau_mode=not bool(args.no_plateau_mode),
        plateau_generations=max(2, args.plateau_generations),
        plateau_distance_epsilon_m=max(0.0, args.plateau_distance_epsilon_m),
        plateau_average_improvement_m=max(0.0, args.plateau_average_improvement_m),
        plateau_elite_fraction=float(np.clip(args.plateau_elite_fraction, 0.05, 1.0)),
        plateau_extra_mutations=max(0, args.plateau_extra_mutations),
        gpu_device=args.gpu_device,
        gpu_engine=args.gpu_engine,
        gpu_run_mode=args.gpu_run_mode,
        gpu_dtype=args.gpu_dtype,
        gpu_batch_size=_parse_gpu_batch_size(args.gpu_batch_size),
        gpu_static_batch_size=_parse_gpu_batch_size(args.gpu_static_batch_size),
        gpu_verify_top_k=max(0, args.gpu_verify_top_k),
        gpu_cpu_replay_top_k=gpu_cpu_replay_top_k,
        gpu_verify_elite_multiplier=max(1, args.gpu_verify_elite_multiplier),
        gpu_telemetry_mode=gpu_telemetry_mode,
        gpu_parity_check=bool(args.gpu_parity_check),
        gpu_fallback_to_cpu=bool(args.gpu_fallback_to_cpu),
        gpu_compile=bool(args.gpu_compile),
        gpu_profile=args.gpu_profile,
        gpu_profile_output=args.gpu_profile_output,
        gpu_chunk_steps=max(1, args.gpu_chunk_steps),
        gpu_disable_early_stop=bool(args.gpu_disable_early_stop),
        gpu_fast_geometry=args.gpu_fast_geometry,
        gpu_collision_mode=args.gpu_collision_mode,
        gpu_attempts_mode=args.gpu_attempts_mode,
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
