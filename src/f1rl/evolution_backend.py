# pyright: reportPrivateImportUsage=false
"""Evolution evaluation backend implementations."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from f1rl.config import SimConfig, actions_for_action_set
from f1rl.gpu_batch import (
    GpuCapturedChunkRollout,
    GpuCapturedRollout,
    GpuControlProgram,
    GpuMonzaBatch,
    gpu_rollout_rows,
)
from f1rl.gpu_types import gpu_track_from_cpu, normalize_device, torch_dtype_from_name
from f1rl.state_snapshot import StateSnapshot, snapshot_to_dict
from f1rl.track_model import load_track_spec

_GPU_TRACK_CACHE: dict[tuple[str, int, str, str], Any] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_torch_profile(
    profiler: Any,
    *,
    output_dir: Path,
    generation: int,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"generation-{generation:04d}-torch-profile.json"
    table_path = output_dir / f"generation-{generation:04d}-torch-profile.txt"
    summary_path = output_dir / f"generation-{generation:04d}-torch-profile-summary.json"
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages()
    max_cuda_time = max((float(getattr(event, "self_cuda_time_total", 0.0)) for event in averages), default=0.0)
    sort_key = "self_cuda_time_total" if device.type == "cuda" and max_cuda_time > 0.0 else "self_cpu_time_total"
    table = averages.table(sort_by=sort_key, row_limit=30)
    table_path.write_text(table, encoding="utf-8")
    sorted_events = sorted(averages, key=lambda item: float(getattr(item, sort_key, 0.0)), reverse=True)
    top_events: list[dict[str, Any]] = []
    for event in sorted_events[:20]:
        top_events.append(
            {
                "key": str(getattr(event, "key", "")),
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "self_cuda_time_total_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                "cpu_memory_usage": int(getattr(event, "cpu_memory_usage", 0)),
                "cuda_memory_usage": int(getattr(event, "cuda_memory_usage", 0)),
            }
        )
    launch_events = [event for event in averages if "cudaLaunchKernel" in str(getattr(event, "key", ""))]
    cuda_launch_count = sum(int(getattr(event, "count", 0)) for event in launch_events)
    cuda_launch_self_cpu_time_us = sum(float(getattr(event, "self_cpu_time_total", 0.0)) for event in launch_events)
    cuda_launch_self_cuda_time_us = sum(float(getattr(event, "self_cuda_time_total", 0.0)) for event in launch_events)
    total_cuda_time_us = sum(float(getattr(event, "self_cuda_time_total", 0.0)) for event in averages)
    total_cpu_time_us = sum(float(getattr(event, "self_cpu_time_total", 0.0)) for event in averages)
    total_profile_time_us = total_cuda_time_us if device.type == "cuda" and total_cuda_time_us > 0.0 else total_cpu_time_us
    bottlenecks: list[dict[str, Any]] = []
    for rank, event in enumerate(sorted_events[:3], start=1):
        event_time_us = float(getattr(event, sort_key, 0.0))
        bottlenecks.append(
            {
                "rank": rank,
                "event": str(getattr(event, "key", "")),
                "count": int(getattr(event, "count", 0)),
                "metric": sort_key,
                "time_us": event_time_us,
                "percent_of_profile_time": event_time_us / max(total_profile_time_us, 1e-9) * 100.0,
            }
        )
    bottleneck_report_path = output_dir / f"generation-{generation:04d}-bottlenecks.md"
    bottleneck_lines = [
        "# GPU Evolution Torch Profiler Bottlenecks",
        "",
        f"- generation: `{generation}`",
        f"- device: `{device}`",
        f"- sort metric: `{sort_key}`",
        f"- cuda launch count: `{cuda_launch_count}`",
        f"- total profiled CPU self time: `{total_cpu_time_us:.3f}us`",
        f"- total profiled CUDA self time: `{total_cuda_time_us:.3f}us`",
        "",
        "Top bottlenecks:",
        "",
    ]
    for item in bottlenecks:
        bottleneck_lines.append(
            f"{item['rank']}. `{item['event']}` - count `{item['count']}`, "
            f"{item['metric']} `{item['time_us']:.3f}us`, "
            f"{item['percent_of_profile_time']:.2f}% of profiled self time"
        )
    bottleneck_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- High `cudaLaunchKernel` count indicates Python/driver launch overhead pressure.",
            "- High `gpu_rollout_*` ranges identify the hottest rollout phase when profile ranges are enabled.",
            "- Compare this report across eager, graph, and future fused backends before claiming speed wins.",
            "",
        ]
    )
    bottleneck_report_path.write_text("\n".join(bottleneck_lines), encoding="utf-8")
    payload = {
        "generation": generation,
        "device": str(device),
        "sort_key": sort_key,
        "trace_path": trace_path,
        "table_path": table_path,
        "bottleneck_report_path": bottleneck_report_path,
        "cuda_launch_count": cuda_launch_count,
        "cuda_launch_self_cpu_time_us": cuda_launch_self_cpu_time_us,
        "cuda_launch_self_cuda_time_us": cuda_launch_self_cuda_time_us,
        "total_cpu_self_time_us": total_cpu_time_us,
        "total_cuda_self_time_us": total_cuda_time_us,
        "bottlenecks": bottlenecks,
        "top_events": top_events,
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return {
        "gpu_profile_trace": str(trace_path),
        "gpu_profile_table": str(table_path),
        "gpu_profile_summary": str(summary_path),
        "gpu_profile_bottleneck_report": str(bottleneck_report_path),
        "gpu_kernel_launches_per_generation": cuda_launch_count,
        "gpu_profile_cuda_launch_count": cuda_launch_count,
        "gpu_profile_cuda_launch_self_cpu_time_us": cuda_launch_self_cpu_time_us,
        "gpu_profile_cuda_launch_self_cuda_time_us": cuda_launch_self_cuda_time_us,
        "gpu_profile_bottlenecks": bottlenecks,
        "gpu_profile_top_events": top_events[:5],
    }


def _write_nsight_hints(
    *,
    output_dir: Path,
    generation: int,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hints_path = output_dir / f"generation-{generation:04d}-nsight-hints.md"
    payload = "\n".join(
        [
            "# Nsight Systems GPU Evolution Profiling",
            "",
            f"- generation: `{generation}`",
            f"- device: `{device}`",
            "",
            "Run the same evolution command under Nsight Systems from the project root, for example:",
            "",
            "```powershell",
            "nsys profile --trace=cuda,nvtx,osrt --capture-range=none "
            "--stats=true --force-overwrite=true --output artifacts\\gpu-nsight\\evolution-gpu "
            "uv run --no-sync python -m f1rl.evolution_search <same args>",
            "```",
            "",
            "Useful NVTX / profiler ranges in the PyTorch backend:",
            "",
            "- `gpu_rollout_controls`",
            "- `gpu_rollout_step`",
            "- `gpu_rollout_scoring`",
            "- `gpu_rollout_profile_scores`",
            "",
            "Inspect launch gaps, host synchronization, allocation events, and whether rollout work is "
            "dominated by small per-step kernels.",
            "",
        ]
    )
    hints_path.write_text(payload, encoding="utf-8")
    return {"gpu_profile_nsight_hints": str(hints_path)}


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
    engine: str = "eager"
    run_mode: str = "parity"
    dtype: str = "float32"
    batch_size: int | None = None
    static_batch_size: int | None = None
    verify_top_k: int = 4
    verify_elite_multiplier: int = 2
    telemetry_mode: str = "selected"
    parity_check: bool = False
    fallback_to_cpu: bool = False
    collision_check: bool = True
    collision_chunk_size: int = 2048
    compile_rollout: bool = False
    compile_mode: str = "reduce-overhead"
    profile: str = "none"
    profile_output_dir: Path | None = None
    active_check_interval: int = 256
    disable_early_stop: bool = False
    fast_geometry: str = "local_window"
    collision_mode: str = "exact_all_segments"
    attempts_mode: str = "full"


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


def _optional_tensor_shape(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    return (tuple(int(value) for value in tensor.shape), str(tensor.dtype), str(tensor.device))


def _control_program_shape_key(program: GpuControlProgram) -> tuple[Any, ...]:
    return (
        program.kind,
        program.action_names,
        _optional_tensor_shape(program.controller_weights),
        _optional_tensor_shape(program.action_controls),
        _optional_tensor_shape(program.phase_action_ids),
        _optional_tensor_shape(program.phase_thresholds),
    )


def _gates_key(gates: Any) -> tuple[Any, ...]:
    return (
        float(getattr(gates, "target_progress_m", 0.0)),
        bool(getattr(gates, "terminate_at_target_progress", True)),
        getattr(gates, "target_min_speed_kph", None),
        getattr(gates, "target_max_speed_kph", None),
        getattr(gates, "target_max_lateral_error_m", None),
        getattr(gates, "target_max_heading_error_deg", None),
        getattr(gates, "target_max_abs_yaw_rate_rps", None),
        getattr(gates, "target_max_abs_steering", None),
        bool(getattr(gates, "segment_fail_on_speed_gate_miss", True)),
        bool(getattr(gates, "segment_require_release", False)),
        float(getattr(gates, "segment_release_min_speed_kph", 0.0)),
        float(getattr(gates, "segment_release_max_speed_kph", 0.0)),
        float(getattr(gates, "segment_release_max_brake", 0.0)),
        float(getattr(gates, "segment_release_max_throttle", 0.0)),
    )


def _graph_cache_key(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sim_config: SimConfig,
    gates: Any,
    scoring_profiles: tuple[str, ...],
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
    control_program: GpuControlProgram,
    collision_check: bool,
    collision_mode: str,
    graph_steps_per_capture: int,
    graph_replay_count: int,
) -> tuple[Any, ...]:
    return (
        str(device),
        str(dtype),
        int(batch_size),
        repr(sim_config),
        _gates_key(gates),
        tuple(scoring_profiles),
        float(frontier_focus_start_m),
        float(frontier_focus_end_m),
        _control_program_shape_key(control_program),
        bool(collision_check),
        str(collision_mode),
        int(graph_steps_per_capture),
        int(graph_replay_count),
    )


def _apply_padded_lane_mask(batch: GpuMonzaBatch, *, real_count: int, padded_count: int, device: torch.device) -> None:
    if padded_count <= 0:
        return
    assert batch.state is not None
    padded_mask = torch.arange(real_count + padded_count, device=device) >= real_count
    batch.state.alive.copy_(torch.where(padded_mask, torch.zeros_like(batch.state.alive), batch.state.alive))


def _new_gpu_batch(
    *,
    track: Any,
    sim_config: SimConfig,
    feature_names: tuple[str, ...],
    settings: GpuBackendSettings,
) -> GpuMonzaBatch:
    return GpuMonzaBatch(
        track=track,
        sim_config=sim_config,
        feature_names=feature_names,
        collision_check=settings.collision_check and settings.collision_mode != "mask_only_debug",
        collision_chunk_size=settings.collision_chunk_size,
        collision_mode=settings.collision_mode,
        compile_rollout=settings.compile_rollout and settings.engine != "graph",
        compile_mode=settings.compile_mode,
        active_check_interval=settings.active_check_interval,
        disable_early_stop=settings.disable_early_stop,
        profile_ranges=settings.profile != "none" and settings.engine != "graph",
    )


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


def _cpu_replay_cache_key(task: dict[str, Any]) -> str:
    payload = {
        "genome": task["genome"],
        "snapshot": task["snapshot"],
        "sim_config": repr(task["sim_config"]),
        "gates": _gates_key(task["gates"]),
        "scoring_profiles": tuple(task["scoring_profiles"]),
        "frontier_focus_start_m": float(task.get("frontier_focus_start_m", 2200.0)),
        "frontier_focus_end_m": float(task.get("frontier_focus_end_m", 2600.0)),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _retarget_cached_cpu_row(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    retargeted = copy.deepcopy(row)
    retargeted.update(
        {
            "candidate_index": int(task["candidate_index"]),
            "generation": int(task["generation"]),
            "seed": int(task["seed"]),
            "genome": task["genome"],
            "lineage": task["lineage"],
            "snapshot_index": int(task["snapshot_index"]),
        }
    )
    return retargeted


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
    replay_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not verify_indices:
        return rows, {
            "cpu_verification_count": 0,
            "cpu_verification_seconds": 0.0,
            "gpu_cpu_replay_count": 0,
            "gpu_cpu_replay_seconds": 0.0,
            "cpu_verification_max_progress_delta_m": None,
            "cpu_verification_max_final_progress_delta_m": None,
            "cpu_verification_max_score_delta": None,
            "cpu_verification_reason_mismatches": None,
            "cpu_verification_valid_lap_mismatches": None,
            "gpu_parity_status": "not_checked",
            "gpu_cpu_top_replay_max_progress_delta_m": None,
            "gpu_cpu_top_replay_max_final_progress_delta_m": None,
            "gpu_cpu_top_replay_max_score_delta": None,
            "gpu_cpu_top_replay_reason_mismatches": None,
            "gpu_cpu_top_replay_valid_lap_mismatches": None,
            "gpu_rank_cpu_top_overlap": None,
            "gpu_cpu_score_correlation": None,
            "gpu_cpu_replay_requested_count": 0,
            "gpu_cpu_replay_unique_count": 0,
            "gpu_cpu_replay_cache_hits": 0,
            "gpu_cpu_replay_cache_misses": 0,
            "gpu_cpu_replay_duplicate_count": 0,
            "gpu_cpu_replay_cache_size": len(replay_cache) if replay_cache is not None else 0,
        }
    started = time.perf_counter()
    verified_rows = list(rows)
    gpu_scores: list[float] = []
    cpu_scores: list[float] = []
    best_progress_deltas: list[float] = []
    final_progress_deltas: list[float] = []
    score_deltas: list[float] = []
    reason_mismatches = 0
    valid_lap_mismatches = 0
    cache_hits = 0
    cache_misses = 0
    active_cache = replay_cache if replay_cache is not None else {}
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
        cache_key = _cpu_replay_cache_key(task)
        cached_row = active_cache.get(cache_key)
        cache_hit = cached_row is not None
        if cached_row is None:
            from f1rl.evolution_search import _evaluate_task_with_sim

            cpu_row = _evaluate_task_with_sim(task)
            active_cache[cache_key] = copy.deepcopy(cpu_row)
            cache_misses += 1
        else:
            cpu_row = _retarget_cached_cpu_row(cached_row, task)
            cache_hits += 1
        gpu_score = float(gpu_row.get("score", 0.0) or 0.0)
        cpu_score = float(cpu_row.get("score", 0.0) or 0.0)
        gpu_scores.append(gpu_score)
        cpu_scores.append(cpu_score)
        score_delta = cpu_score - gpu_score
        best_progress_delta = float(cpu_row.get("best_progress_m", 0.0) or 0.0) - float(
            gpu_row.get("best_progress_m", 0.0) or 0.0
        )
        final_progress_delta = float(cpu_row.get("final_progress_m", 0.0) or 0.0) - float(
            gpu_row.get("final_progress_m", 0.0) or 0.0
        )
        best_progress_deltas.append(best_progress_delta)
        final_progress_deltas.append(final_progress_delta)
        score_deltas.append(score_delta)
        if gpu_row.get("termination_reason") != cpu_row.get("termination_reason"):
            reason_mismatches += 1
        if bool(gpu_row.get("valid_lap")) != bool(cpu_row.get("valid_lap")):
            valid_lap_mismatches += 1
        cpu_row["backend"] = "gpu"
        cpu_row["gpu_verified"] = True
        cpu_row["gpu_score"] = gpu_score
        cpu_row["gpu_profile_scores"] = gpu_row.get("profile_scores", {})
        cpu_row["cpu_verified_score"] = cpu_score
        cpu_row["cpu_replay_cache_hit"] = cache_hit
        cpu_row["backend_parity_error"] = {
            "score_delta": score_delta,
            "best_progress_delta_m": best_progress_delta,
            "final_progress_delta_m": final_progress_delta,
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
    max_best_progress_delta = max((abs(value) for value in best_progress_deltas), default=0.0)
    max_final_progress_delta = max((abs(value) for value in final_progress_deltas), default=0.0)
    max_score_delta = max((abs(value) for value in score_deltas), default=0.0)
    parity_status = (
        "passed"
        if max_best_progress_delta <= 0.5
        and max_final_progress_delta <= 0.5
        and reason_mismatches == 0
        and valid_lap_mismatches == 0
        else "failed"
    )
    return verified_rows, {
        "cpu_verification_count": len(verify_indices),
        "cpu_verification_seconds": elapsed,
        "gpu_cpu_replay_count": len(verify_indices),
        "gpu_cpu_replay_seconds": elapsed,
        "cpu_verification_max_progress_delta_m": max_best_progress_delta,
        "cpu_verification_max_final_progress_delta_m": max_final_progress_delta,
        "cpu_verification_max_score_delta": max_score_delta,
        "cpu_verification_reason_mismatches": reason_mismatches,
        "cpu_verification_valid_lap_mismatches": valid_lap_mismatches,
        "gpu_parity_status": parity_status,
        "gpu_cpu_top_replay_max_progress_delta_m": max_best_progress_delta,
        "gpu_cpu_top_replay_max_final_progress_delta_m": max_final_progress_delta,
        "gpu_cpu_top_replay_max_score_delta": max_score_delta,
        "gpu_cpu_top_replay_reason_mismatches": reason_mismatches,
        "gpu_cpu_top_replay_valid_lap_mismatches": valid_lap_mismatches,
        "gpu_rank_cpu_top_overlap": overlap,
        "gpu_cpu_score_correlation": correlation,
        "gpu_cpu_replay_requested_count": len(verify_indices),
        "gpu_cpu_replay_unique_count": cache_misses,
        "gpu_cpu_replay_cache_hits": cache_hits,
        "gpu_cpu_replay_cache_misses": cache_misses,
        "gpu_cpu_replay_duplicate_count": cache_hits,
        "gpu_cpu_replay_cache_size": len(active_cache),
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
        self._captured_rollouts: dict[tuple[Any, ...], tuple[GpuMonzaBatch, GpuCapturedRollout]] = {}
        self._captured_chunk_rollouts: dict[tuple[Any, ...], tuple[GpuMonzaBatch, GpuCapturedChunkRollout]] = {}
        self._cpu_replay_cache: dict[str, dict[str, Any]] = {}

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
        evaluate_started = time.perf_counter()
        device = normalize_device(self.settings.device)
        dtype = torch_dtype_from_name(self.settings.dtype)
        track_started = time.perf_counter()
        track = _cached_gpu_track(sim_config.track_path, device=device, dtype=dtype)
        track_setup_seconds = time.perf_counter() - track_started
        batch_size = self.settings.static_batch_size or self.settings.batch_size or len(candidates)
        batch_size = max(1, int(batch_size))
        all_rows: list[dict[str, Any]] = []
        rollout_seconds = 0.0
        sim_steps = 0
        steps_executed = 0
        host_sync_count = 0
        static_padding_count = 0
        compile_enabled = False
        compile_errors: list[str] = []
        kernel_backends: set[str] = set()
        graph_capture_count = 0
        graph_replay_count = 0
        graph_fallback_count = 0
        graph_cache_hit_count = 0
        graph_cache_miss_count = 0
        graph_errors: list[str] = []
        graph_replay_seconds = 0.0
        graph_capture_seconds = 0.0
        graph_steps_per_capture_values: set[int] = set()
        graph_requested_replay_count = 0
        control_upload_seconds = 0.0
        batch_reset_seconds = 0.0
        rollout_wall_seconds = 0.0
        result_materialization_seconds = 0.0
        from f1rl.evolution_search import genome_to_dict

        profiler: Any | None = None
        if self.settings.profile == "torch":
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                acc_events=True,
            )
        profile_context = profiler if profiler is not None else contextlib.nullcontext()
        with profile_context:
            for start in range(0, len(candidates), batch_size):
                chunk_candidates = candidates[start : start + batch_size]
                real_chunk_count = len(chunk_candidates)
                padded_candidates = chunk_candidates
                if self.settings.static_batch_size is not None and real_chunk_count < batch_size:
                    pad_count = batch_size - real_chunk_count
                    static_padding_count += pad_count
                    padded_candidates = [*chunk_candidates, *([chunk_candidates[-1]] * pad_count)]
                chunk_snapshots = [snapshots[int(candidate.snapshot_index)] for candidate in padded_candidates]
                control_started = time.perf_counter()
                control_program = _control_program_for_candidates(
                    padded_candidates,
                    action_set=sim_config.action_set,
                    feature_count=len(self.feature_names),
                    output_count=self.controller_output_count,
                    device=device,
                    dtype=dtype,
                )
                control_upload_seconds += time.perf_counter() - control_started
                padded_count = len(padded_candidates) - real_chunk_count
                if self.settings.engine == "graph" and device.type == "cuda":
                    graph_steps_per_capture = min(
                        max(1, int(self.settings.active_check_interval)),
                        max(1, int(sim_config.max_steps)),
                    )
                    graph_chunk_replay_count, graph_tail_steps = divmod(
                        max(1, int(sim_config.max_steps)),
                        graph_steps_per_capture,
                    )
                    use_chunked_graph = graph_tail_steps > 0 or graph_chunk_replay_count > 1
                    if graph_chunk_replay_count == 0:
                        graph_chunk_replay_count = 1
                        graph_steps_per_capture = max(1, graph_tail_steps)
                        graph_tail_steps = 0
                    graph_steps_per_capture_values.add(graph_steps_per_capture)
                    if graph_tail_steps > 0:
                        graph_steps_per_capture_values.add(graph_tail_steps)
                    graph_requested_replay_count += graph_chunk_replay_count + (1 if graph_tail_steps > 0 else 0)
                    cache_key = _graph_cache_key(
                        device=device,
                        dtype=dtype,
                        batch_size=len(padded_candidates),
                        sim_config=sim_config,
                        gates=gates,
                        scoring_profiles=scoring_profiles,
                        frontier_focus_start_m=frontier_focus_start_m,
                        frontier_focus_end_m=frontier_focus_end_m,
                        control_program=control_program,
                        collision_check=self.settings.collision_check,
                        collision_mode=self.settings.collision_mode,
                        graph_steps_per_capture=graph_steps_per_capture,
                        graph_replay_count=graph_chunk_replay_count,
                    )
                    cached = (
                        self._captured_chunk_rollouts.get(cache_key)
                        if use_chunked_graph
                        else self._captured_rollouts.get(cache_key)
                    )
                    try:
                        if cached is None:
                            graph_cache_miss_count += 1
                            batch = _new_gpu_batch(
                                track=track,
                                sim_config=sim_config,
                                feature_names=self.feature_names,
                                settings=self.settings,
                            )
                            compile_enabled = compile_enabled or batch.compile_enabled
                            if batch.compile_error is not None:
                                compile_errors.append(batch.compile_error)
                            reset_started = time.perf_counter()
                            batch.reset(
                                chunk_snapshots,
                                target_progress_m=float(gates.target_progress_m),
                                terminate_at_target=bool(gates.terminate_at_target_progress),
                            )
                            _apply_padded_lane_mask(
                                batch,
                                real_count=real_chunk_count,
                                padded_count=padded_count,
                                device=device,
                            )
                            batch_reset_seconds += time.perf_counter() - reset_started
                            if use_chunked_graph:
                                captured = batch.capture_cuda_graph_chunk(
                                    control_program=control_program,
                                    gates=gates,
                                    scoring_profiles=scoring_profiles,
                                    frontier_focus_start_m=frontier_focus_start_m,
                                    frontier_focus_end_m=frontier_focus_end_m,
                                    chunk_steps=graph_steps_per_capture,
                                    replay_count=graph_chunk_replay_count,
                                    tail_steps=graph_tail_steps,
                                )
                            else:
                                captured = batch.capture_cuda_graph(
                                    control_program=control_program,
                                    gates=gates,
                                    scoring_profiles=scoring_profiles,
                                    frontier_focus_start_m=frontier_focus_start_m,
                                    frontier_focus_end_m=frontier_focus_end_m,
                                )
                            batch.state = captured.input_state
                            batch._last_segment_idx = captured.input_last_segment_idx
                            batch.segment_start_progress_m = captured.input_segment_start_progress_m
                            batch.segment_target_progress_m = captured.input_segment_target_progress_m
                            reset_started = time.perf_counter()
                            batch.reset(
                                chunk_snapshots,
                                target_progress_m=float(gates.target_progress_m),
                                terminate_at_target=bool(gates.terminate_at_target_progress),
                            )
                            _apply_padded_lane_mask(
                                batch,
                                real_count=real_chunk_count,
                                padded_count=padded_count,
                                device=device,
                            )
                            batch_reset_seconds += time.perf_counter() - reset_started
                            rollout_started = time.perf_counter()
                            if use_chunked_graph:
                                captured_chunk = cast(GpuCapturedChunkRollout, captured)
                                result = batch.replay_cuda_graph_chunks(
                                    captured_chunk,
                                    snapshots=None,
                                    control_program=control_program,
                                    gates=gates,
                                )
                            else:
                                captured_full = cast(GpuCapturedRollout, captured)
                                result = batch.replay_cuda_graph(
                                    captured_full,
                                    snapshots=None,
                                    control_program=control_program,
                                    gates=gates,
                                )
                            rollout_wall_seconds += time.perf_counter() - rollout_started
                            graph_capture_count += 1
                            graph_capture_seconds += captured.capture_seconds
                            result = replace(result, rollout_seconds=captured.capture_seconds + result.rollout_seconds)
                            if use_chunked_graph:
                                self._captured_chunk_rollouts[cache_key] = (
                                    batch,
                                    cast(GpuCapturedChunkRollout, captured),
                                )
                            else:
                                self._captured_rollouts[cache_key] = (batch, cast(GpuCapturedRollout, captured))
                        else:
                            graph_cache_hit_count += 1
                            batch, captured = cached
                            batch.state = captured.input_state
                            batch._last_segment_idx = captured.input_last_segment_idx
                            batch.segment_start_progress_m = captured.input_segment_start_progress_m
                            batch.segment_target_progress_m = captured.input_segment_target_progress_m
                            reset_started = time.perf_counter()
                            batch.reset(
                                chunk_snapshots,
                                target_progress_m=float(gates.target_progress_m),
                                terminate_at_target=bool(gates.terminate_at_target_progress),
                            )
                            _apply_padded_lane_mask(
                                batch,
                                real_count=real_chunk_count,
                                padded_count=padded_count,
                                device=device,
                            )
                            batch_reset_seconds += time.perf_counter() - reset_started
                            rollout_started = time.perf_counter()
                            if use_chunked_graph:
                                captured_chunk = cast(GpuCapturedChunkRollout, captured)
                                result = batch.replay_cuda_graph_chunks(
                                    captured_chunk,
                                    snapshots=None,
                                    control_program=control_program,
                                    gates=gates,
                                )
                            else:
                                captured_full = cast(GpuCapturedRollout, captured)
                                result = batch.replay_cuda_graph(
                                    captured_full,
                                    snapshots=None,
                                    control_program=control_program,
                                    gates=gates,
                                )
                            rollout_wall_seconds += time.perf_counter() - rollout_started
                        graph_replay_count += max(1, int(result.graph_replay_count or 1))
                    except Exception as exc:  # pragma: no cover - CUDA graph support is platform/operator dependent.
                        batch = _new_gpu_batch(
                            track=track,
                            sim_config=sim_config,
                            feature_names=self.feature_names,
                            settings=self.settings,
                        )
                        compile_enabled = compile_enabled or batch.compile_enabled
                        if batch.compile_error is not None:
                            compile_errors.append(batch.compile_error)
                        reset_started = time.perf_counter()
                        batch.reset(
                            chunk_snapshots,
                            target_progress_m=float(gates.target_progress_m),
                            terminate_at_target=bool(gates.terminate_at_target_progress),
                        )
                        _apply_padded_lane_mask(
                            batch,
                            real_count=real_chunk_count,
                            padded_count=padded_count,
                            device=device,
                        )
                        batch_reset_seconds += time.perf_counter() - reset_started
                        rollout_started = time.perf_counter()
                        fallback = batch.rollout(
                            control_program=control_program,
                            gates=gates,
                            scoring_profiles=scoring_profiles,
                            frontier_focus_start_m=frontier_focus_start_m,
                            frontier_focus_end_m=frontier_focus_end_m,
                        )
                        result = replace(
                            fallback,
                            kernel_backend="pytorch_eager_graph_fallback",
                            graph_error=f"{type(exc).__name__}: {exc}",
                        )
                        rollout_wall_seconds += time.perf_counter() - rollout_started
                else:
                    batch = _new_gpu_batch(
                        track=track,
                        sim_config=sim_config,
                        feature_names=self.feature_names,
                        settings=self.settings,
                    )
                    compile_enabled = compile_enabled or batch.compile_enabled
                    if batch.compile_error is not None:
                        compile_errors.append(batch.compile_error)
                    reset_started = time.perf_counter()
                    batch.reset(
                        chunk_snapshots,
                        target_progress_m=float(gates.target_progress_m),
                        terminate_at_target=bool(gates.terminate_at_target_progress),
                    )
                    _apply_padded_lane_mask(
                        batch,
                        real_count=real_chunk_count,
                        padded_count=padded_count,
                        device=device,
                    )
                    batch_reset_seconds += time.perf_counter() - reset_started
                    rollout_started = time.perf_counter()
                    if self.settings.engine == "graph":
                        result = batch.rollout_cuda_graph(
                            control_program=control_program,
                            gates=gates,
                            scoring_profiles=scoring_profiles,
                            frontier_focus_start_m=frontier_focus_start_m,
                            frontier_focus_end_m=frontier_focus_end_m,
                        )
                    elif self.settings.engine == "fused":
                        result = batch.rollout_warp_open(
                            control_program=control_program,
                            gates=gates,
                            scoring_profiles=scoring_profiles,
                            frontier_focus_start_m=frontier_focus_start_m,
                            frontier_focus_end_m=frontier_focus_end_m,
                        )
                    else:
                        result = batch.rollout(
                            control_program=control_program,
                            gates=gates,
                            scoring_profiles=scoring_profiles,
                            frontier_focus_start_m=frontier_focus_start_m,
                            frontier_focus_end_m=frontier_focus_end_m,
                        )
                    rollout_wall_seconds += time.perf_counter() - rollout_started
                kernel_backends.add(result.kernel_backend)
                if result.kernel_backend in {"pytorch_cuda_graph", "pytorch_cuda_graph_chunked"}:
                    graph_replay_seconds += float(result.graph_replay_seconds or 0.0)
                if result.graph_error is not None:
                    graph_fallback_count += 1
                    if result.graph_error not in graph_errors:
                        graph_errors.append(result.graph_error)
                rollout_seconds += result.rollout_seconds
                sim_steps += result.sim_steps
                host_sync_count += result.host_sync_count
                steps_executed = max(steps_executed, result.steps_executed)
                materialize_started = time.perf_counter()
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
                        physics_model=sim_config.physics_model,
                        physics_version=sim_config.physics_version,
                        physics_calibration_id=sim_config.physics_calibration_id,
                        candidate_index_offset=start,
                        row_detail=self.settings.attempts_mode,
                    )
                )
                result_materialization_seconds += time.perf_counter() - materialize_started
        profile_metrics: dict[str, Any] = {}
        if profiler is not None and self.settings.profile_output_dir is not None:
            profile_metrics = _write_torch_profile(
                profiler,
                output_dir=self.settings.profile_output_dir,
                generation=generation,
                device=device,
            )
        elif self.settings.profile == "nsight" and self.settings.profile_output_dir is not None:
            profile_metrics = _write_nsight_hints(
                output_dir=self.settings.profile_output_dir,
                generation=generation,
                device=device,
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
            replay_cache=self._cpu_replay_cache,
        )
        backend_total_seconds = time.perf_counter() - evaluate_started
        replay_seconds = float(verification_metrics.get("gpu_cpu_replay_seconds", 0.0) or 0.0)
        non_rollout_seconds = max(0.0, backend_total_seconds - rollout_seconds)
        metrics = {
            "backend": "gpu",
            "gpu_engine": self.settings.engine,
            "gpu_run_mode": self.settings.run_mode,
            "gpu_kernel_backend": ",".join(sorted(kernel_backends)) if kernel_backends else "not_run",
            "gpu_graph_capture_count": graph_capture_count,
            "gpu_graph_replay_count": graph_replay_count,
            "gpu_graph_fallback_count": graph_fallback_count,
            "gpu_graph_cache_hit_count": graph_cache_hit_count,
            "gpu_graph_cache_miss_count": graph_cache_miss_count,
            "gpu_graph_cache_size": len(self._captured_rollouts) + len(self._captured_chunk_rollouts),
            "gpu_graph_steps_per_capture": sorted(graph_steps_per_capture_values) or None,
            "gpu_graph_requested_replay_count": graph_requested_replay_count if graph_requested_replay_count > 0 else None,
            "gpu_graph_errors": graph_errors,
            "gpu_graph_capture_seconds": graph_capture_seconds if graph_capture_count > 0 else None,
            "gpu_graph_replay_seconds": graph_replay_seconds if graph_replay_count > 0 else None,
            "gpu_device": str(device),
            "gpu_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "gpu_dtype": self.settings.dtype,
            "gpu_batch_size": batch_size,
            "gpu_static_batch_size": self.settings.static_batch_size,
            "gpu_static_padding_count": static_padding_count,
            "gpu_backend_total_seconds": backend_total_seconds,
            "gpu_track_setup_seconds": track_setup_seconds,
            "gpu_control_program_upload_seconds": control_upload_seconds,
            "gpu_batch_reset_seconds": batch_reset_seconds,
            "gpu_rollout_wall_seconds": rollout_wall_seconds,
            "gpu_rollout_seconds": rollout_seconds,
            "gpu_non_rollout_backend_seconds": non_rollout_seconds,
            "gpu_result_materialization_seconds": result_materialization_seconds,
            "gpu_rollout_time_fraction": rollout_seconds / max(backend_total_seconds, 1e-9),
            "gpu_cpu_replay_time_fraction": replay_seconds / max(backend_total_seconds, 1e-9),
            "gpu_materialization_time_fraction": result_materialization_seconds / max(backend_total_seconds, 1e-9),
            "gpu_candidates_per_second": len(candidates) / max(rollout_seconds, 1e-9),
            "gpu_backend_candidates_per_second": len(candidates) / max(backend_total_seconds, 1e-9),
            "gpu_steps_per_second": sim_steps / max(rollout_seconds, 1e-9),
            "gpu_active_steps_per_second": sim_steps / max(rollout_seconds, 1e-9),
            "gpu_steps_executed": steps_executed,
            "gpu_sim_steps": sim_steps,
            "gpu_kernel_launches_per_generation": None,
            "gpu_host_sync_count": host_sync_count,
            "gpu_profile": self.settings.profile,
            "gpu_profile_ranges_enabled": self.settings.profile != "none" and self.settings.engine != "graph",
            "gpu_fast_geometry": self.settings.fast_geometry,
            "gpu_collision_mode": self.settings.collision_mode,
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9 if device.type == "cuda" else 0.0,
            "gpu_telemetry_mode": self.settings.telemetry_mode,
            "gpu_attempts_mode": self.settings.attempts_mode,
            "gpu_verify_top_k": self.settings.verify_top_k,
            "gpu_parity_check": self.settings.parity_check,
            "gpu_compile_requested": self.settings.compile_rollout,
            "gpu_compile_enabled": compile_enabled,
            "gpu_compile_mode": self.settings.compile_mode if self.settings.compile_rollout else None,
            "gpu_compile_errors": compile_errors,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **verification_metrics,
            **profile_metrics,
        }
        return verified_rows, metrics
