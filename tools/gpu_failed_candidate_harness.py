# pyright: reportPrivateImportUsage=false
"""Focused GPU/CPU failed-candidate parity harness.

This is intentionally diagnostic code. It loads selected rows from a completed
GPU production run's postcheck output, then replays exact GPU batch internals in
four split modes:

- TT: torch controller features/controls + torch step
- WT: warp controller features/controls + torch step
- TW: torch controller features/controls + warp step
- WW: warp controller features/controls + warp step

The split isolates whether drift starts in controller features/actions or in the
Warp step/projection/collision path.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from f1rl.evolution_backend import (
    GpuBackendSettings,
    _cached_gpu_track,
    _control_program_for_candidates,
    _new_gpu_batch,
)
from f1rl.evolution_search import (
    CONTROLLER_FEATURE_NAMES,
    Candidate,
    EvolutionGates,
    EvolutionSearchConfig,
    _load_snapshots,
    _sim_config_for_generation,
    genome_from_mapping,
)
from f1rl.gpu_batch import GpuMonzaBatch
from f1rl.gpu_fused_warp import (
    controller_controls_warp_batch,
    controller_feature_ids_tensor,
    open_step_warp_batch,
    search_features_warp_batch,
)
from f1rl.gpu_scoring import TERMINATION_ID_TO_REASON

DEFAULT_CANDIDATES = ((4, 793), (4, 277), (3, 578), (4, 0), (4, 120), (4, 122))
COMBOS = {
    "TT": ("torch", "torch"),
    "WT": ("warp", "torch"),
    "TW": ("torch", "warp"),
    "WW": ("warp", "warp"),
}
MATERIAL_TOLERANCES = {
    "x": 0.25,
    "y": 0.25,
    "heading_deg": 0.5,
    "speed_kph": 0.5,
    "yaw_rate_rps": 0.02,
    "steering": 0.02,
    "throttle": 0.01,
    "brake": 0.01,
    "steer_cmd": 0.01,
    "raw_progress_m": 0.5,
    "monotonic_progress_m": 0.5,
    "lateral_error_m": 0.5,
    "heading_error_deg": 0.5,
    "progress_delta_m": 0.5,
}
MATERIAL_EXACT_KEYS = (
    "checkpoint_index",
    "next_checkpoint_index",
    "checkpoints_passed",
    "missed_checkpoint_count",
    "alive",
    "terminated",
    "truncated",
    "reason",
    "valid_lap",
    "collided",
    "off_track",
)


@dataclass(slots=True)
class LoadedRun:
    run_dir: Path
    checkpoint: dict[str, Any]
    config: EvolutionSearchConfig
    gates: EvolutionGates
    postchecked_rows: dict[tuple[int, int], dict[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _parse_candidate_ids(values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    parsed: list[tuple[int, int]] = []
    for value in values:
        if ":" in value:
            gen_s, idx_s = value.split(":", 1)
            parsed.append((int(gen_s), int(idx_s)))
        else:
            parsed.append((4, int(value)))
    return tuple(parsed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_run(run_dir: Path) -> LoadedRun:
    checkpoint = json.loads((run_dir / "population_checkpoint.json").read_text())
    config_payload = dict(checkpoint["config"])
    config_payload["scoring_profiles"] = tuple(config_payload["scoring_profiles"])
    config_payload["max_steps_schedule"] = tuple(tuple(x) for x in config_payload.get("max_steps_schedule", ()))
    config_payload["survival_floor_stages_m"] = tuple(config_payload.get("survival_floor_stages_m", ()))
    config = EvolutionSearchConfig(**config_payload)
    gates = EvolutionGates(**checkpoint["gates"])
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "postchecked_attempts.jsonl"):
        rows[(int(row["generation"]), int(row["candidate_index"]))] = row
    return LoadedRun(
        run_dir=run_dir,
        checkpoint=checkpoint,
        config=config,
        gates=gates,
        postchecked_rows=rows,
    )


def _make_candidate(row: dict[str, Any]) -> Candidate:
    return Candidate(
        genome=genome_from_mapping(row["genome"]),
        snapshot_index=int(row["snapshot_index"]),
        lineage=dict(row.get("lineage", {})),
    )


def _make_batch(
    *,
    run: LoadedRun,
    row: dict[str, Any],
    sim_config: Any,
    snapshots: list[Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GpuMonzaBatch, Any]:
    settings = GpuBackendSettings(
        device=str(device),
        engine="eager",
        run_mode="parity",
        dtype="float32",
        verify_top_k=0,
        telemetry_mode="none",
        collision_mode="exact_grid",
        fast_geometry="local_window",
        attempts_mode="compact",
    )
    track = _cached_gpu_track(sim_config.track_path, device=device, dtype=dtype)
    batch = _new_gpu_batch(
        track=track,
        sim_config=sim_config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        settings=settings,
    )
    candidate = _make_candidate(row)
    batch.reset(
        [snapshots[candidate.snapshot_index]],
        target_progress_m=float(run.gates.target_progress_m),
        terminate_at_target=bool(run.gates.terminate_at_target_progress),
    )
    control_program = _control_program_for_candidates(
        [candidate],
        action_set=sim_config.action_set,
        feature_count=len(CONTROLLER_FEATURE_NAMES),
        output_count=3,
        device=device,
        dtype=dtype,
    )
    return batch, control_program


def _feature_controls(
    batch: GpuMonzaBatch,
    control_program: Any,
    *,
    gates: EvolutionGates,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    assert batch.state is not None
    assert batch.segment_start_progress_m is not None
    assert batch._last_segment_idx is not None
    assert batch._continuous_action_id is not None
    assert control_program.controller_weights is not None
    if backend == "warp":
        feature_ids = controller_feature_ids_tensor(CONTROLLER_FEATURE_NAMES, device=batch.track.device)
        features, diagnostics = search_features_warp_batch(
            batch.state,
            batch.track,
            batch.sim_config,
            feature_names=CONTROLLER_FEATURE_NAMES,
            segment_start_progress_m=batch.segment_start_progress_m,
            segment_target_progress_m=float(gates.target_progress_m),
            braking_gates_m=batch.braking_gates_m,
            lookahead_m=batch.lookahead_m,
            local_projection_window_px=batch.local_projection_window_px,
            previous_segment_idx=batch._last_segment_idx,
            feature_ids=feature_ids,
        )
        throttle, brake, steer = controller_controls_warp_batch(control_program.controller_weights, features)
    else:
        features, diagnostics = batch._search_features(
            batch.state,
            batch.track,
            batch.sim_config,
            feature_names=CONTROLLER_FEATURE_NAMES,
            segment_start_progress_m=batch.segment_start_progress_m,
            segment_target_progress_m=float(gates.target_progress_m),
            braking_gates_m=batch.braking_gates_m,
            lookahead_m=batch.lookahead_m,
            local_projection_window_px=batch.local_projection_window_px,
            previous_segment_idx=batch._last_segment_idx,
            zero_feature=batch._zero_float,
            row_index=batch._row_index,
        )
        throttle, brake, steer = batch._controller_controls(control_program.controller_weights, features)
    batch._last_segment_idx = diagnostics["segment_idx"]
    return throttle, brake, steer, batch._continuous_action_id, diagnostics, features


def _to_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().reshape(-1)[0].item())


def _to_int(tensor: torch.Tensor) -> int:
    return int(tensor.detach().cpu().reshape(-1)[0].item())


def _to_bool(tensor: torch.Tensor) -> bool:
    return bool(tensor.detach().cpu().reshape(-1)[0].item())


def _reason_from_batch(batch: GpuMonzaBatch) -> str:
    assert batch.state is not None
    return TERMINATION_ID_TO_REASON.get(_to_int(batch.state.termination_reason_id), "unknown")


def _trace_row(
    *,
    step: int,
    batch: GpuMonzaBatch,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    features: torch.Tensor,
    collided: torch.Tensor,
    off_track: torch.Tensor,
) -> dict[str, Any]:
    assert batch.state is not None
    assert batch._last_segment_idx is not None
    row = {
        "step": step,
        "x": _to_float(batch.state.x),
        "y": _to_float(batch.state.y),
        "heading_rad": _to_float(batch.state.heading_rad),
        "speed_mps": _to_float(batch.state.speed_mps),
        "yaw_rate_rps": _to_float(batch.state.yaw_rate_rps),
        "steering": _to_float(batch.state.steering),
        "throttle": _to_float(throttle),
        "brake": _to_float(brake),
        "steer_cmd": _to_float(steer),
        "raw_progress_m": _to_float(batch.state.raw_progress_m),
        "monotonic_progress_m": _to_float(batch.state.monotonic_progress_m),
        "last_raw_progress_px": _to_float(batch.state.last_raw_progress_px),
        "lateral_error_m": _to_float(diagnostics["lateral_error_m"]),
        "signed_lateral_error_m": _to_float(diagnostics.get("signed_lateral_error_m", diagnostics["lateral_error_m"])),
        "heading_error_rad": _to_float(diagnostics["heading_error_rad"]),
        "progress_delta_m": _to_float(diagnostics.get("progress_delta_m", torch.zeros_like(batch.state.speed_mps))),
        "segment_idx": _to_int(batch._last_segment_idx),
        "checkpoint_index": _to_int(batch.state.checkpoint_index),
        "next_checkpoint_index": _to_int(batch.state.next_checkpoint_index),
        "checkpoints_passed": _to_int(batch.state.checkpoints_passed),
        "missed_checkpoint_count": _to_int(batch.state.missed_checkpoint_count),
        "elapsed_steps": _to_int(batch.state.elapsed_steps),
        "no_progress_steps": _to_int(batch.state.no_progress_steps),
        "alive": _to_bool(batch.state.alive),
        "terminated": _to_bool(batch.state.terminated),
        "truncated": _to_bool(batch.state.truncated),
        "reason": _reason_from_batch(batch),
        "valid_lap": _to_bool(batch.state.valid_lap),
        "collided": _to_bool(collided),
        "off_track": _to_bool(off_track),
        "feature_max_abs": float(torch.max(torch.abs(features.detach())).cpu().item()),
    }
    for name in (
        "target_speed_kph",
        "near_target_speed_kph",
        "min_future_target_speed_kph",
        "target_speed_drop_kph",
        "brake_demand",
        "future_brake_demand",
        "brake_gate_proximity",
        "braking_gate_distance_m",
        "brake_gate_distance_norm",
        "lookahead_abs_max",
        "target_steer",
    ):
        if name in diagnostics:
            row[name] = _to_float(diagnostics[name])
    return row


def _material_gpu_row(
    *,
    step: int,
    batch: GpuMonzaBatch,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    collided: torch.Tensor,
    off_track: torch.Tensor,
) -> dict[str, Any]:
    assert batch.state is not None
    return {
        "step": step,
        "x": _to_float(batch.state.x),
        "y": _to_float(batch.state.y),
        "heading_deg": _to_float(batch.state.heading_rad) * 57.29577951308232,
        "speed_kph": _to_float(batch.state.speed_mps) * 3.6,
        "yaw_rate_rps": _to_float(batch.state.yaw_rate_rps),
        "steering": _to_float(batch.state.steering),
        "throttle": _to_float(throttle),
        "brake": _to_float(brake),
        "steer_cmd": _to_float(steer),
        "raw_progress_m": _to_float(batch.state.raw_progress_m),
        "monotonic_progress_m": _to_float(batch.state.monotonic_progress_m),
        "lateral_error_m": _to_float(diagnostics["lateral_error_m"]),
        "heading_error_deg": _to_float(diagnostics["heading_error_rad"]) * 57.29577951308232,
        "progress_delta_m": _to_float(diagnostics.get("progress_delta_m", torch.zeros_like(batch.state.speed_mps))),
        "checkpoint_index": _to_int(batch.state.checkpoint_index),
        "next_checkpoint_index": _to_int(batch.state.next_checkpoint_index),
        "checkpoints_passed": _to_int(batch.state.checkpoints_passed),
        "missed_checkpoint_count": _to_int(batch.state.missed_checkpoint_count),
        "alive": _to_bool(batch.state.alive),
        "terminated": _to_bool(batch.state.terminated),
        "truncated": _to_bool(batch.state.truncated),
        "reason": _reason_from_batch(batch),
        "valid_lap": _to_bool(batch.state.valid_lap),
        "collided": _to_bool(collided),
        "off_track": _to_bool(off_track),
    }


def _cpu_material_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(row.get("step_index", row.get("step", 0))),
        "x": float(row.get("x", 0.0) or 0.0),
        "y": float(row.get("y", 0.0) or 0.0),
        "heading_deg": float(row.get("heading_deg", 0.0) or 0.0),
        "speed_kph": float(row.get("speed_kph", 0.0) or 0.0),
        "yaw_rate_rps": float(row.get("yaw_rate_rps", 0.0) or 0.0),
        "throttle": float(row.get("throttle", 0.0) or 0.0),
        "brake": float(row.get("brake", 0.0) or 0.0),
        "steer_cmd": float(row.get("steering", 0.0) or 0.0),
        "raw_progress_m": float(row.get("raw_progress_m", 0.0) or 0.0),
        "monotonic_progress_m": float(row.get("monotonic_progress_m", 0.0) or 0.0),
        "lateral_error_m": float(row.get("lateral_error_m", 0.0) or 0.0),
        "heading_error_deg": float(row.get("heading_error_deg", 0.0) or 0.0),
        "progress_delta_m": float(row.get("progress_delta_m", 0.0) or 0.0),
        "checkpoint_index": int(row.get("checkpoint_index", 0) or 0),
        "next_checkpoint_index": int(row.get("next_checkpoint_index", 0) or 0),
        "checkpoints_passed": int(row.get("checkpoints_passed", 0) or 0),
        "missed_checkpoint_count": int(row.get("missed_checkpoint_count", 0) or 0),
        "alive": not bool(row.get("terminated", False) or row.get("truncated", False)),
        "terminated": bool(row.get("terminated", False)),
        "truncated": bool(row.get("truncated", False)),
        "reason": str(row.get("termination_reason", "active")),
        "valid_lap": bool(row.get("valid_lap", False)),
        "collided": bool(row.get("collided", False)),
        "off_track": bool(row.get("off_track", False)),
    }


def _material_diff(
    reference: dict[str, Any],
    actual: dict[str, Any],
    *,
    compare_valid_lap: bool = True,
) -> dict[str, Any] | None:
    diffs: dict[str, Any] = {}
    for key, tolerance in MATERIAL_TOLERANCES.items():
        if key not in reference or key not in actual:
            continue
        left = float(reference.get(key, 0.0) or 0.0)
        right = float(actual.get(key, 0.0) or 0.0)
        if key == "heading_deg":
            delta = abs(((left - right + 180.0) % 360.0) - 180.0)
        else:
            delta = abs(left - right)
        if delta > tolerance:
            diffs[key] = {
                "reference": reference.get(key),
                "actual": actual.get(key),
                "delta": delta,
                "tol": tolerance,
            }
    for key in MATERIAL_EXACT_KEYS:
        if key == "valid_lap" and not compare_valid_lap:
            continue
        if reference.get(key) != actual.get(key):
            diffs[key] = {"reference": reference.get(key), "actual": actual.get(key)}
    if not diffs:
        return None
    return {
        "step": int(reference.get("step", actual.get("step", -1))),
        "diffs": diffs,
        "reference": reference,
        "actual": actual,
    }


def _run_gpu_step(
    batch: GpuMonzaBatch,
    control_program: Any,
    *,
    gates: EvolutionGates,
    combo: str,
    step: int,
) -> dict[str, Any] | None:
    control_backend, step_backend = COMBOS[combo]
    active = batch._active()
    if step > 0 and not _to_bool(torch.any(active)):
        return None
    throttle, brake, steer, _action_id, diagnostics, _features = _feature_controls(
        batch,
        control_program,
        gates=gates,
        backend=control_backend,
    )
    if step_backend == "warp":
        state = batch.state
        assert state is not None
        assert batch._last_segment_idx is not None
        step_diagnostics, collided, off_track, _telemetry_valid_lap = open_step_warp_batch(
            state,
            last_segment_idx=batch._last_segment_idx,
            active=active,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=batch.params,
            track=batch.track,
            sim_config=batch.sim_config,
            segment_target_progress_m=batch.segment_target_progress_m,
            gates=gates,
            collision_check=batch.collision_check,
            collision_mode=batch.collision_mode,
        )
    else:
        step_diagnostics, collided, off_track, _telemetry_valid_lap = batch._step(
            throttle=throttle,
            brake=brake,
            steer=steer,
            active=active,
            gates=gates,
        )
    combined = dict(diagnostics)
    combined.update(step_diagnostics)
    return _material_gpu_row(
        step=step,
        batch=batch,
        throttle=throttle,
        brake=brake,
        steer=steer,
        diagnostics=combined,
        collided=collided,
        off_track=off_track,
    )


def _run_combo_trace(
    *,
    run: LoadedRun,
    row: dict[str, Any],
    combo: str,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    control_backend, step_backend = COMBOS[combo]
    gen = int(row["generation"])
    device = torch.device("cuda")
    dtype = torch.float32
    sim_config = _sim_config_for_generation(run.config, gen)
    snapshots = _load_snapshots(
        sim_config=sim_config,
        config=run.config,
        state_library=None,
        start_progress_m=run.checkpoint.get("start_progress_m"),
        start_speed_kph=float(run.checkpoint.get("start_speed_kph") or 0.0),
        start_min_progress_m=None,
        start_max_progress_m=None,
    )
    batch, control_program = _make_batch(
        run=run,
        row=row,
        sim_config=sim_config,
        snapshots=snapshots,
        device=device,
        dtype=dtype,
    )
    steps_to_run = sim_config.max_steps if max_steps is None else min(int(max_steps), sim_config.max_steps)
    trace: list[dict[str, Any]] = []
    with torch.inference_mode():
        for step in range(steps_to_run):
            active = batch._active()
            if step > 0 and not _to_bool(torch.any(active)):
                break
            throttle, brake, steer, _action_id, diagnostics, features = _feature_controls(
                batch,
                control_program,
                gates=run.gates,
                backend=control_backend,
            )
            if step_backend == "warp":
                state = batch.state
                assert state is not None
                assert batch._last_segment_idx is not None
                step_diagnostics, collided, off_track, _telemetry_valid_lap = open_step_warp_batch(
                    state,
                    last_segment_idx=batch._last_segment_idx,
                    active=active,
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    params=batch.params,
                    track=batch.track,
                    sim_config=batch.sim_config,
                    segment_target_progress_m=batch.segment_target_progress_m,
                    gates=run.gates,
                    collision_check=batch.collision_check,
                    collision_mode=batch.collision_mode,
                )
            else:
                step_diagnostics, collided, off_track, _telemetry_valid_lap = batch._step(
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    active=active,
                    gates=run.gates,
                )
            combined = dict(diagnostics)
            combined.update(step_diagnostics)
            trace.append(
                _trace_row(
                    step=step,
                    batch=batch,
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    diagnostics=combined,
                    features=features,
                    collided=collided,
                    off_track=off_track,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return trace


def _first_diff(
    reference: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    progress_tol: float = 0.05,
    state_tol: float = 0.02,
    control_tol: float = 1.0e-4,
) -> dict[str, Any] | None:
    numeric_tolerances = {
        "x": state_tol,
        "y": state_tol,
        "heading_rad": 1.0e-4,
        "speed_mps": 1.0e-3,
        "yaw_rate_rps": 1.0e-4,
        "steering": 1.0e-4,
        "throttle": control_tol,
        "brake": control_tol,
        "steer_cmd": control_tol,
        "raw_progress_m": progress_tol,
        "monotonic_progress_m": progress_tol,
        "lateral_error_m": state_tol,
        "signed_lateral_error_m": state_tol,
        "heading_error_rad": 1.0e-4,
        "progress_delta_m": progress_tol,
    }
    exact_keys = (
        "segment_idx",
        "checkpoint_index",
        "next_checkpoint_index",
        "checkpoints_passed",
        "missed_checkpoint_count",
        "alive",
        "terminated",
        "truncated",
        "reason",
        "valid_lap",
        "collided",
        "off_track",
    )
    max_len = max(len(reference), len(actual))
    for index in range(max_len):
        if index >= len(reference) or index >= len(actual):
            return {
                "step": index,
                "kind": "length",
                "reference_present": index < len(reference),
                "actual_present": index < len(actual),
            }
        ref = reference[index]
        got = actual[index]
        diffs: dict[str, Any] = {}
        for key, tol in numeric_tolerances.items():
            delta = abs(float(ref.get(key, 0.0)) - float(got.get(key, 0.0)))
            if delta > tol:
                diffs[key] = {"reference": ref.get(key), "actual": got.get(key), "delta": delta, "tol": tol}
        for key in exact_keys:
            if ref.get(key) != got.get(key):
                diffs[key] = {"reference": ref.get(key), "actual": got.get(key)}
        if diffs:
            return {"step": index, "kind": "value", "diffs": diffs, "reference": ref, "actual": got}
    return None


def _summary(row: dict[str, Any], traces: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "cpu_postcheck": {
            "steps": row.get("steps"),
            "termination_reason": row.get("termination_reason"),
            "final_progress_m": row.get("final_progress_m"),
            "best_progress_m": row.get("best_progress_m"),
            "score": row.get("score"),
            "gpu_termination_reason": row.get("backend_parity_error", {}).get("gpu_termination_reason"),
            "gpu_final_progress_m": row.get("gpu_final_progress_m"),
            "gpu_score": row.get("gpu_score"),
        },
        "combo_finals": {},
        "first_diffs_vs_TT": {},
    }
    for combo, trace in traces.items():
        final = trace[-1] if trace else {}
        result["combo_finals"][combo] = {
            "steps": len(trace),
            "termination_reason": final.get("reason"),
            "final_progress_m": final.get("monotonic_progress_m"),
            "final_speed_kph": float(final.get("speed_mps", 0.0)) * 3.6,
            "lateral_error_m": final.get("lateral_error_m"),
            "heading_error_deg": float(final.get("heading_error_rad", 0.0)) * 57.29577951308232,
            "collided": final.get("collided"),
            "off_track": final.get("off_track"),
        }
    reference = traces["TT"]
    for combo in ("WT", "TW", "WW"):
        result["first_diffs_vs_TT"][combo] = _first_diff(reference, traces[combo])
    return result


def _context_for_row(
    run: LoadedRun,
    row: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, list[Any]]:
    sim_config = _sim_config_for_generation(run.config, int(row["generation"]))
    snapshots = _load_snapshots(
        sim_config=sim_config,
        config=run.config,
        state_library=None,
        start_progress_m=run.checkpoint.get("start_progress_m"),
        start_speed_kph=float(run.checkpoint.get("start_speed_kph") or 0.0),
        start_min_progress_m=None,
        start_max_progress_m=None,
    )
    del device, dtype
    return sim_config, snapshots


def _new_trace_batch(
    run: LoadedRun,
    row: dict[str, Any],
    *,
    sim_config: Any,
    snapshots: list[Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GpuMonzaBatch, Any]:
    return _make_batch(
        run=run,
        row=row,
        sim_config=sim_config,
        snapshots=snapshots,
        device=device,
        dtype=dtype,
    )


def _first_gpu_combo_diff(
    *,
    run: LoadedRun,
    row: dict[str, Any],
    combo: str,
    max_steps: int | None,
) -> dict[str, Any]:
    device = torch.device("cuda")
    dtype = torch.float32
    sim_config, snapshots = _context_for_row(run, row, device=device, dtype=dtype)
    ref_batch, ref_program = _new_trace_batch(
        run,
        row,
        sim_config=sim_config,
        snapshots=snapshots,
        device=device,
        dtype=dtype,
    )
    actual_batch, actual_program = _new_trace_batch(
        run,
        row,
        sim_config=sim_config,
        snapshots=snapshots,
        device=device,
        dtype=dtype,
    )
    steps_to_run = sim_config.max_steps if max_steps is None else min(int(max_steps), sim_config.max_steps)
    last_ref: dict[str, Any] | None = None
    last_actual: dict[str, Any] | None = None
    for step in range(steps_to_run):
        ref = _run_gpu_step(ref_batch, ref_program, gates=run.gates, combo="TT", step=step)
        actual = _run_gpu_step(actual_batch, actual_program, gates=run.gates, combo=combo, step=step)
        if ref is None or actual is None:
            return {
                "comparison": f"TT_vs_{combo}",
                "status": "length_mismatch" if ref is not None or actual is not None else "no_material_diff",
                "step": step,
                "reference_present": ref is not None,
                "actual_present": actual is not None,
                "last_reference": last_ref,
                "last_actual": last_actual,
            }
        last_ref = ref
        last_actual = actual
        diff = _material_diff(ref, actual)
        if diff is not None:
            diff["comparison"] = f"TT_vs_{combo}"
            diff["status"] = "material_diff"
            return diff
    return {
        "comparison": f"TT_vs_{combo}",
        "status": "no_material_diff",
        "steps_checked": steps_to_run,
        "last_reference": last_ref,
        "last_actual": last_actual,
    }


def _cpu_trace_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    telemetry_path = row.get("selected_telemetry")
    if telemetry_path is None:
        return []
    path = Path(str(telemetry_path))
    if not path.exists():
        return []
    return [_cpu_material_row(item) for item in _read_jsonl(path)]


def _first_cpu_tt_diff(
    *,
    run: LoadedRun,
    row: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    cpu_rows = _cpu_trace_rows(row)
    if not cpu_rows:
        return {"comparison": "CPU_vs_TT", "status": "missing_cpu_trace"}
    device = torch.device("cuda")
    dtype = torch.float32
    sim_config, snapshots = _context_for_row(run, row, device=device, dtype=dtype)
    batch, program = _new_trace_batch(
        run,
        row,
        sim_config=sim_config,
        snapshots=snapshots,
        device=device,
        dtype=dtype,
    )
    steps_to_run = min(len(cpu_rows), sim_config.max_steps if max_steps is None else int(max_steps))
    last_cpu: dict[str, Any] | None = None
    last_gpu: dict[str, Any] | None = None
    for step in range(steps_to_run):
        gpu = _run_gpu_step(batch, program, gates=run.gates, combo="TT", step=step)
        if gpu is None:
            return {
                "comparison": "CPU_vs_TT",
                "status": "length_mismatch",
                "step": step,
                "cpu_present": True,
                "gpu_present": False,
                "last_cpu": last_cpu,
                "last_gpu": last_gpu,
            }
        cpu = cpu_rows[step]
        last_cpu = cpu
        last_gpu = gpu
        diff = _material_diff(cpu, gpu, compare_valid_lap=False)
        if diff is not None:
            diff["comparison"] = "CPU_vs_TT"
            diff["status"] = "material_diff"
            return diff
    if len(cpu_rows) != steps_to_run:
        return {
            "comparison": "CPU_vs_TT",
            "status": "not_checked_to_cpu_end",
            "steps_checked": steps_to_run,
            "cpu_steps": len(cpu_rows),
            "last_cpu": last_cpu,
            "last_gpu": last_gpu,
        }
    extra_gpu = _run_gpu_step(batch, program, gates=run.gates, combo="TT", step=steps_to_run)
    if extra_gpu is not None:
        return {
            "comparison": "CPU_vs_TT",
            "status": "length_mismatch",
            "step": steps_to_run,
            "cpu_present": False,
            "gpu_present": True,
            "last_cpu": last_cpu,
            "last_gpu": last_gpu,
            "extra_gpu": extra_gpu,
        }
    return {
        "comparison": "CPU_vs_TT",
        "status": "no_material_diff",
        "steps_checked": steps_to_run,
        "last_cpu": last_cpu,
        "last_gpu": last_gpu,
    }


def _first_divergence_summary(
    *,
    run: LoadedRun,
    row: dict[str, Any],
    max_steps: int | None,
    combos: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "cpu_postcheck": {
            "steps": row.get("steps"),
            "termination_reason": row.get("termination_reason"),
            "final_progress_m": row.get("final_progress_m"),
            "best_progress_m": row.get("best_progress_m"),
            "score": row.get("score"),
            "gpu_termination_reason": row.get("backend_parity_error", {}).get("gpu_termination_reason"),
            "gpu_score": row.get("gpu_score"),
        },
        "first_divergences": {
            "CPU_vs_TT": _first_cpu_tt_diff(run=run, row=row, max_steps=max_steps),
        },
    }
    for combo in combos:
        if combo == "TT":
            continue
        result["first_divergences"][f"TT_vs_{combo}"] = _first_gpu_combo_diff(
            run=run,
            row=row,
            combo=combo,
            max_steps=max_steps,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--write-traces", type=Path, default=None)
    parser.add_argument("--first-divergence", action="store_true")
    parser.add_argument("--combo", action="append", choices=tuple(COMBOS), default=[])
    args = parser.parse_args()

    run = _load_run(args.run_dir)
    requested = _parse_candidate_ids(args.candidate) if args.candidate else DEFAULT_CANDIDATES
    combos = tuple(args.combo) if args.combo else ("WT", "TW", "WW")
    for key in requested:
        try:
            row = run.postchecked_rows[key]
        except KeyError:
            print(json.dumps({"missing_candidate": {"generation": key[0], "candidate_index": key[1]}}), flush=True)
            continue
        if args.first_divergence:
            summary = _first_divergence_summary(run=run, row=row, max_steps=args.max_steps, combos=combos)
            print(json.dumps(summary, default=_json_default, sort_keys=True), flush=True)
            continue
        traces = {
            combo: _run_combo_trace(run=run, row=row, combo=combo, max_steps=args.max_steps)
            for combo in ("TT", "WT", "TW", "WW")
        }
        summary = _summary(row, traces)
        print(json.dumps(summary, default=_json_default, sort_keys=True), flush=True)
        if args.write_traces is not None:
            args.write_traces.mkdir(parents=True, exist_ok=True)
            trace_path = args.write_traces / f"gen-{key[0]:03d}-candidate-{key[1]:05d}-gpu-split-traces.json"
            trace_path.write_text(json.dumps(traces, default=_json_default), encoding="utf-8")


if __name__ == "__main__":
    main()
