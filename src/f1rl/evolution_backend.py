# pyright: reportPrivateImportUsage=false
"""Evolution evaluation backend implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from f1rl.config import SimConfig, actions_for_action_set
from f1rl.gpu_batch import GpuControlProgram, GpuMonzaBatch, gpu_rollout_rows
from f1rl.gpu_types import gpu_track_from_cpu, normalize_device, torch_dtype_from_name
from f1rl.state_snapshot import StateSnapshot, snapshot_to_dict
from f1rl.track_model import load_track_spec

_GPU_TRACK_CACHE: dict[tuple[str, int, str, str], Any] = {}


class EvolutionEvaluationBackend(Protocol):
    name: str

    def evaluate_population(
        self,
        *,
        candidates: list[Any],
        snapshots: list[StateSnapshot],
        sim_config: SimConfig,
        gates: Any,
        generation: int,
        seed: int,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        capture_step_telemetry: bool,
        stream_telemetry_dir: Path | None,
        telemetry_compression: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...


@dataclass(frozen=True, slots=True)
class GpuBackendSettings:
    device: str = "cuda"
    dtype: str = "float32"
    batch_size: int | None = None
    verify_top_k: int = 4
    verify_elite_multiplier: int = 2
    telemetry_mode: str = "selected"
    parity_check: bool = False
    fallback_to_cpu: bool = False
    collision_check: bool = True
    collision_chunk_size: int = 2048
    compile_rollout: bool = False
    compile_mode: str = "reduce-overhead"


def _cached_gpu_track(path: Path, *, device: torch.device, dtype: torch.dtype) -> Any:
    resolved = path.resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = -1
    key = (str(resolved), int(mtime_ns), str(device), str(dtype))
    track = _GPU_TRACK_CACHE.get(key)
    if track is None:
        track = gpu_track_from_cpu(load_track_spec(path), device=device, dtype=dtype)
        _GPU_TRACK_CACHE[key] = track
    return track


def _controller_weight_tensor(
    candidates: list[Any],
    *,
    feature_count: int,
    output_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows: list[list[float]] = []
    expected = feature_count * output_count
    for candidate in candidates:
        weights = list(float(value) for value in candidate.genome.controller_weights)
        if len(weights) < expected:
            weights.extend([0.0] * (expected - len(weights)))
        if len(weights) > expected:
            weights = weights[:expected]
        rows.append(weights)
    flat = torch.tensor(rows, device=device, dtype=dtype)
    return flat.reshape((len(candidates), output_count, feature_count))


def _control_program_for_candidates(
    candidates: list[Any],
    *,
    action_set: str,
    feature_count: int,
    output_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GpuControlProgram:
    kinds = {str(candidate.genome.kind) for candidate in candidates}
    if len(kinds) != 1:
        raise ValueError(f"GPU backend requires one genome kind per batch; got {sorted(kinds)}")
    kind = next(iter(kinds))
    if kind == "controller":
        return GpuControlProgram(
            kind=kind,
            action_names=("continuous",),
            controller_weights=_controller_weight_tensor(
                candidates,
                feature_count=feature_count,
                output_count=output_count,
                device=device,
                dtype=dtype,
            ),
        )
    if kind not in {"phase", "progress_phase"}:
        raise ValueError(f"GPU backend does not support genome kind {kind!r}")

    actions = actions_for_action_set(action_set)
    action_names = tuple(spec[0] for spec in actions)
    action_index = {name: index for index, name in enumerate(action_names)}
    action_controls = torch.tensor(
        [[float(throttle), float(brake), float(steer)] for _name, throttle, brake, steer in actions],
        device=device,
        dtype=dtype,
    )

    phase_lists: list[tuple[Any, ...]] = []
    for candidate in candidates:
        phases = candidate.genome.progress_phases if kind == "progress_phase" else candidate.genome.phases
        if not phases:
            raise ValueError(f"GPU {kind} genomes require at least one phase")
        phase_lists.append(tuple(phases))
    phase_count = max(len(phases) for phases in phase_lists)
    ids: list[list[int]] = []
    thresholds: list[list[float]] = []
    for phases in phase_lists:
        candidate_ids: list[int] = []
        candidate_thresholds: list[float] = []
        elapsed = 0.0
        for phase in phases:
            try:
                candidate_ids.append(action_index[str(phase.action)])
            except KeyError as exc:
                valid = ", ".join(action_names)
                raise ValueError(f"Unknown action {phase.action!r} for action set {action_set!r}; expected one of: {valid}") from exc
            elapsed += float(phase.progress_m if kind == "progress_phase" else phase.steps)
            candidate_thresholds.append(elapsed)
        while len(candidate_ids) < phase_count:
            candidate_ids.append(candidate_ids[-1])
            candidate_thresholds.append(float("inf"))
        ids.append(candidate_ids)
        thresholds.append(candidate_thresholds)
    return GpuControlProgram(
        kind=kind,
        action_names=action_names,
        action_controls=action_controls,
        phase_action_ids=torch.tensor(ids, device=device, dtype=torch.int64),
        phase_thresholds=torch.tensor(thresholds, device=device, dtype=dtype),
    )


def _verification_indices(
    rows: list[dict[str, Any]],
    *,
    scoring_profiles: tuple[str, ...],
    verify_top_k: int,
    verify_all: bool,
) -> list[int]:
    if verify_all:
        return list(range(len(rows)))
    count = max(0, int(verify_top_k))
    if count <= 0:
        return []
    selected: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            selected.append(index)

    for index, _row in sorted(enumerate(rows), key=lambda item: float(item[1].get("score", float("-inf"))), reverse=True)[:count]:
        add(index)
    for profile in scoring_profiles:
        ranked = sorted(
            enumerate(rows),
            key=lambda item: float(item[1].get("profile_scores", {}).get(profile, float("-inf"))),
            reverse=True,
        )
        for index, _row in ranked[:count]:
            add(index)
    for index, _row in sorted(
        enumerate(rows),
        key=lambda item: float(item[1].get("best_progress_m", float("-inf"))),
        reverse=True,
    )[:count]:
        add(index)
    return selected


def _task_for_row(
    row: dict[str, Any],
    *,
    candidates: list[Any],
    snapshots: list[StateSnapshot],
    sim_config: SimConfig,
    gates: Any,
    scoring_profiles: tuple[str, ...],
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
) -> dict[str, Any]:
    candidate_index = int(row["candidate_index"])
    candidate = candidates[candidate_index]
    snapshot = snapshots[int(candidate.snapshot_index)]
    from f1rl.evolution_search import genome_to_dict

    return {
        "candidate_index": candidate_index,
        "generation": int(row["generation"]),
        "seed": int(row["seed"]),
        "genome": genome_to_dict(candidate.genome),
        "lineage": candidate.lineage,
        "snapshot_index": candidate.snapshot_index,
        "snapshot": snapshot_to_dict(snapshot),
        "sim_config": sim_config,
        "gates": gates,
        "scoring_profiles": scoring_profiles,
        "frontier_focus_start_m": frontier_focus_start_m,
        "frontier_focus_end_m": frontier_focus_end_m,
        "capture_step_telemetry": False,
        "stream_telemetry_path": None,
    }


def _verify_rows_with_cpu(
    rows: list[dict[str, Any]],
    *,
    verify_indices: list[int],
    candidates: list[Any],
    snapshots: list[StateSnapshot],
    sim_config: SimConfig,
    gates: Any,
    scoring_profiles: tuple[str, ...],
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not verify_indices:
        return rows, {
            "cpu_verification_count": 0,
            "cpu_verification_seconds": 0.0,
            "gpu_rank_cpu_top_overlap": None,
            "gpu_cpu_score_correlation": None,
        }
    started = time.perf_counter()
    verified_rows = list(rows)
    gpu_scores: list[float] = []
    cpu_scores: list[float] = []
    for index in verify_indices:
        gpu_row = rows[index]
        task = _task_for_row(
            gpu_row,
            candidates=candidates,
            snapshots=snapshots,
            sim_config=sim_config,
            gates=gates,
            scoring_profiles=scoring_profiles,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
        )
        from f1rl.evolution_search import _evaluate_task_with_sim

        cpu_row = _evaluate_task_with_sim(task)
        gpu_score = float(gpu_row.get("score", 0.0) or 0.0)
        cpu_score = float(cpu_row.get("score", 0.0) or 0.0)
        gpu_scores.append(gpu_score)
        cpu_scores.append(cpu_score)
        cpu_row["backend"] = "gpu"
        cpu_row["gpu_verified"] = True
        cpu_row["gpu_score"] = gpu_score
        cpu_row["gpu_profile_scores"] = gpu_row.get("profile_scores", {})
        cpu_row["cpu_verified_score"] = cpu_score
        cpu_row["backend_parity_error"] = {
            "score_delta": cpu_score - gpu_score,
            "best_progress_delta_m": float(cpu_row.get("best_progress_m", 0.0) or 0.0)
            - float(gpu_row.get("best_progress_m", 0.0) or 0.0),
            "final_progress_delta_m": float(cpu_row.get("final_progress_m", 0.0) or 0.0)
            - float(gpu_row.get("final_progress_m", 0.0) or 0.0),
            "gpu_termination_reason": gpu_row.get("termination_reason"),
            "cpu_termination_reason": cpu_row.get("termination_reason"),
        }
        verified_rows[index] = cpu_row
    elapsed = time.perf_counter() - started
    correlation = None
    if len(gpu_scores) >= 2 and np.std(gpu_scores) > 1e-9 and np.std(cpu_scores) > 1e-9:
        correlation = float(np.corrcoef(gpu_scores, cpu_scores)[0, 1])
    gpu_top = {
        int(row["candidate_index"])
        for row in sorted(rows, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[
            : max(1, min(len(verify_indices), 10))
        ]
    }
    cpu_verified = [verified_rows[index] for index in verify_indices]
    cpu_top = {
        int(row["candidate_index"])
        for row in sorted(cpu_verified, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[
            : max(1, min(len(verify_indices), 10))
        ]
    }
    overlap = len(gpu_top & cpu_top) / max(1, len(gpu_top | cpu_top))
    return verified_rows, {
        "cpu_verification_count": len(verify_indices),
        "cpu_verification_seconds": elapsed,
        "gpu_rank_cpu_top_overlap": overlap,
        "gpu_cpu_score_correlation": correlation,
    }


class GpuEvolutionBackend:
    name = "gpu"

    def __init__(
        self,
        *,
        settings: GpuBackendSettings,
        feature_names: tuple[str, ...],
        controller_output_count: int,
    ) -> None:
        self.settings = settings
        self.feature_names = feature_names
        self.controller_output_count = controller_output_count

    def evaluate_population(
        self,
        *,
        candidates: list[Any],
        snapshots: list[StateSnapshot],
        sim_config: SimConfig,
        gates: Any,
        generation: int,
        seed: int,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        capture_step_telemetry: bool,
        stream_telemetry_dir: Path | None,
        telemetry_compression: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del stream_telemetry_dir, telemetry_compression
        if capture_step_telemetry and self.settings.telemetry_mode != "all-cpu-replay":
            raise ValueError(
                "GPU backend cannot silently stream all-candidate step telemetry. "
                "Use --gpu-telemetry-mode all-cpu-replay or --backend cpu."
            )
        device = normalize_device(self.settings.device)
        dtype = torch_dtype_from_name(self.settings.dtype)
        track = _cached_gpu_track(sim_config.track_path, device=device, dtype=dtype)
        batch_size = self.settings.batch_size or len(candidates)
        batch_size = max(1, int(batch_size))
        all_rows: list[dict[str, Any]] = []
        rollout_seconds = 0.0
        sim_steps = 0
        steps_executed = 0
        compile_enabled = False
        compile_errors: list[str] = []
        from f1rl.evolution_search import genome_to_dict

        for start in range(0, len(candidates), batch_size):
            chunk_candidates = candidates[start : start + batch_size]
            chunk_snapshots = [snapshots[int(candidate.snapshot_index)] for candidate in chunk_candidates]
            control_program = _control_program_for_candidates(
                chunk_candidates,
                action_set=sim_config.action_set,
                feature_count=len(self.feature_names),
                output_count=self.controller_output_count,
                device=device,
                dtype=dtype,
            )
            batch = GpuMonzaBatch(
                track=track,
                sim_config=sim_config,
                feature_names=self.feature_names,
                collision_check=self.settings.collision_check,
                collision_chunk_size=self.settings.collision_chunk_size,
                compile_rollout=self.settings.compile_rollout,
                compile_mode=self.settings.compile_mode,
            )
            compile_enabled = compile_enabled or batch.compile_enabled
            if batch.compile_error is not None:
                compile_errors.append(batch.compile_error)
            batch.reset(
                chunk_snapshots,
                target_progress_m=float(gates.target_progress_m),
                terminate_at_target=bool(gates.terminate_at_target_progress),
            )
            result = batch.rollout(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
            rollout_seconds += result.rollout_seconds
            sim_steps += result.sim_steps
            steps_executed = max(steps_executed, result.steps_executed)
            all_rows.extend(
                gpu_rollout_rows(
                    result=result,
                    candidates=chunk_candidates,
                    snapshots=snapshots,
                    generation=generation,
                    seed=seed,
                    scoring_profiles=scoring_profiles,
                    target_progress_m=float(gates.target_progress_m),
                    max_steps=sim_config.max_steps,
                    genome_to_dict=genome_to_dict,
                    candidate_index_offset=start,
                )
            )

        verify_all = self.settings.telemetry_mode == "all-cpu-replay" or bool(self.settings.parity_check)
        verify_indices = _verification_indices(
            all_rows,
            scoring_profiles=scoring_profiles,
            verify_top_k=self.settings.verify_top_k,
            verify_all=verify_all,
        )
        verified_rows, verification_metrics = _verify_rows_with_cpu(
            all_rows,
            verify_indices=verify_indices,
            candidates=candidates,
            snapshots=snapshots,
            sim_config=sim_config,
            gates=gates,
            scoring_profiles=scoring_profiles,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
        )
        metrics = {
            "backend": "gpu",
            "gpu_device": str(device),
            "gpu_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "gpu_dtype": self.settings.dtype,
            "gpu_batch_size": batch_size,
            "gpu_rollout_seconds": rollout_seconds,
            "gpu_candidates_per_second": len(candidates) / max(rollout_seconds, 1e-9),
            "gpu_steps_per_second": sim_steps / max(rollout_seconds, 1e-9),
            "gpu_steps_executed": steps_executed,
            "gpu_sim_steps": sim_steps,
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9 if device.type == "cuda" else 0.0,
            "gpu_telemetry_mode": self.settings.telemetry_mode,
            "gpu_verify_top_k": self.settings.verify_top_k,
            "gpu_parity_check": self.settings.parity_check,
            "gpu_compile_requested": self.settings.compile_rollout,
            "gpu_compile_enabled": compile_enabled,
            "gpu_compile_mode": self.settings.compile_mode if self.settings.compile_rollout else None,
            "gpu_compile_errors": compile_errors,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **verification_metrics,
        }
        return verified_rows, metrics
