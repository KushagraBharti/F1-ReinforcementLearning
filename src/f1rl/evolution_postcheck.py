# pyright: reportPrivateUsage=false
"""Deferred CPU replay verification for evolution search artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
from typing import Any

from f1rl.evolution_search import (
    CHECKPOINT_NAME,
    EvolutionGates,
    EvolutionSearchConfig,
    _load_snapshots,
    _row_elapsed_s,
    _row_identity,
    _row_is_valid_finish,
    _run_candidate,
    _safe_slug,
    _score_rows,
    _sim_config_for_generation,
    _telemetry_file_suffix,
    _write_json,
    _write_jsonl,
    genome_from_mapping,
)
from f1rl.state_snapshot import StateSnapshot, snapshot_to_dict


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, default=str) + "\n")


def _config_from_mapping(payload: dict[str, Any]) -> EvolutionSearchConfig:
    allowed = {field.name for field in fields(EvolutionSearchConfig)}
    data = {key: value for key, value in payload.items() if key in allowed}
    data["scoring_profiles"] = tuple(data.get("scoring_profiles", ()))
    data["max_steps_schedule"] = tuple(
        (int(item[0]), int(item[1])) for item in data.get("max_steps_schedule", ())
    )
    data["survival_floor_stages_m"] = tuple(float(item) for item in data.get("survival_floor_stages_m", ()))
    for path_key in ("all_candidate_telemetry_dir", "gpu_profile_output"):
        if data.get(path_key) is not None:
            data[path_key] = Path(str(data[path_key]))
    return EvolutionSearchConfig(**data)


def _gates_from_mapping(payload: dict[str, Any]) -> EvolutionGates:
    allowed = {field.name for field in fields(EvolutionGates)}
    data = {key: value for key, value in payload.items() if key in allowed}
    return EvolutionGates(**data)


def _row_genome_snapshot_key(row: dict[str, Any]) -> str:
    payload = {
        "genome": row.get("genome"),
        "snapshot_index": int(row.get("snapshot_index", 0)),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _row_lineage_branch_key(row: dict[str, Any]) -> str:
    lineage = dict(row.get("lineage", {}) or {})
    parent = dict(lineage.get("parent", {}) or {})
    payload = {
        "source": lineage.get("source"),
        "source_bucket": lineage.get("source_bucket"),
        "parent_generation": lineage.get("parent_generation", parent.get("generation")),
        "parent_candidate_index": lineage.get("parent_candidate_index", parent.get("candidate_index")),
        "parent_seed": lineage.get("parent_seed", parent.get("seed")),
        "snapshot_index": int(row.get("snapshot_index", 0)),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _select_postcheck_rows(
    rows: list[dict[str, Any]],
    *,
    scoring_profiles: tuple[str, ...],
    top_k: int,
    candidate_pool_size: int | None = None,
) -> list[dict[str, Any]]:
    if candidate_pool_size is None:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()

        def add_standard(row: dict[str, Any], reason: str) -> None:
            key = _row_identity(row)
            if key in seen:
                return
            seen.add(key)
            tagged = dict(row)
            tagged["postcheck_selection_reason"] = reason
            selected.append(tagged)

        for row in sorted(rows, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[:top_k]:
            add_standard(row, "top_score")
        for profile in scoring_profiles:
            leader = max(
                rows,
                key=lambda item: float(item.get("profile_scores", {}).get(profile, float("-inf"))),
                default=None,
            )
            if leader is not None:
                add_standard(leader, f"profile_leader:{profile}")
        farthest = max(rows, key=lambda item: float(item.get("best_progress_m", float("-inf"))), default=None)
        if farthest is not None:
            add_standard(farthest, "farthest_best_progress")
        valid_rows = [row for row in rows if _row_is_valid_finish(row)]
        fastest_valid = min(valid_rows, key=_row_elapsed_s, default=None)
        if fastest_valid is not None:
            add_standard(fastest_valid, "fastest_valid_lap")
        return selected

    pool_limit = max(top_k, int(candidate_pool_size))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    lineage_counts: Counter[str] = Counter()

    def add_pool(row: dict[str, Any], reason: str, *, force: bool = False) -> None:
        if len(selected) >= pool_limit:
            return
        key = _row_identity(row)
        if key in seen:
            return
        branch_key = _row_lineage_branch_key(row)
        if not force and lineage_counts[branch_key] >= 2:
            return
        seen.add(key)
        lineage_counts[branch_key] += 1
        tagged = dict(row)
        tagged["postcheck_selection_reason"] = reason
        selected.append(tagged)

    score_rows = sorted(rows, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
    score_take = min(pool_limit, max(top_k * 2, pool_limit // 3))
    for rank, row in enumerate(score_rows[:score_take]):
        reason = "top_score" if rank < top_k else "top_score_pool"
        add_pool(row, reason)
    for profile in scoring_profiles:
        for rank, leader in enumerate(
            sorted(
                rows,
                key=lambda item, active_profile=profile: float(
                    item.get("profile_scores", {}).get(active_profile, float("-inf"))
                ),
                reverse=True,
            )[: max(1, min(3, top_k))]
        ):
            add_pool(leader, f"profile_leader:{profile}" if rank == 0 else f"profile_pool:{profile}")
    for key, reason in (
        ("best_progress_m", "farthest_best_progress"),
        ("final_progress_m", "farthest_final_progress"),
        ("final_speed_kph", "fastest_final_speed"),
    ):
        for rank, leader in enumerate(
            sorted(rows, key=lambda item, active_key=key: float(item.get(active_key, float("-inf"))), reverse=True)[
                : max(1, min(3, top_k))
            ]
        ):
            add_pool(leader, reason if rank == 0 else f"{reason}_pool")
    valid_rows = [row for row in rows if _row_is_valid_finish(row)]
    for rank, row in enumerate(sorted(valid_rows, key=_row_elapsed_s)[: max(1, min(3, top_k))]):
        add_pool(row, "fastest_valid_lap" if rank == 0 else "fastest_valid_lap_pool", force=True)
    best_progress = max((float(row.get("best_progress_m", 0.0) or 0.0) for row in rows), default=0.0)
    near_valid_rows = [
        row
        for row in rows
        if float(row.get("best_progress_m", 0.0) or 0.0) >= max(450.0, best_progress - 250.0)
    ]
    for row in sorted(
        near_valid_rows,
        key=lambda item: (
            float(item.get("best_progress_m", 0.0) or 0.0),
            float(item.get("final_speed_kph", 0.0) or 0.0),
        ),
        reverse=True,
    )[: max(1, min(top_k, pool_limit // 4))]:
        add_pool(row, "near_valid_or_frontier_pool")
    for row in score_rows:
        add_pool(row, "unique_genome_lineage_fill")
        if len(selected) >= pool_limit:
            break
    if len(selected) < min(top_k, len(rows)):
        for row in score_rows:
            add_pool(row, "fallback_top_score", force=True)
            if len(selected) >= min(pool_limit, len(rows)):
                break
    return selected


def _select_cpu_rerank_rows(
    rows: list[dict[str, Any]],
    *,
    scoring_profiles: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    seen_genomes: set[str] = set()

    def add(row: dict[str, Any], reason: str) -> None:
        key = _row_identity(row)
        if key in seen:
            return
        genome_key = _row_genome_snapshot_key(row)
        if genome_key in seen_genomes:
            return
        seen.add(key)
        seen_genomes.add(genome_key)
        tagged = dict(row)
        tagged["postcheck_selection_reason"] = reason
        selected.append(tagged)

    for row in sorted(rows, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[:top_k]:
        add(row, "cpu_rerank_top_score")
    for profile in scoring_profiles:
        leader = max(
            rows,
            key=lambda item: float(item.get("profile_scores", {}).get(profile, float("-inf"))),
            default=None,
        )
        if leader is not None:
            add(leader, f"cpu_rerank_profile_leader:{profile}")
    farthest = max(rows, key=lambda item: float(item.get("best_progress_m", float("-inf"))), default=None)
    if farthest is not None:
        add(farthest, "cpu_rerank_farthest_best_progress")
    valid_rows = [row for row in rows if _row_is_valid_finish(row)]
    fastest_valid = min(valid_rows, key=_row_elapsed_s, default=None)
    if fastest_valid is not None:
        add(fastest_valid, "cpu_rerank_fastest_valid_lap")
    return selected


def _row_has_clean_state_parity(row: dict[str, Any]) -> bool:
    delta = dict(row.get("backend_parity_error", {}) or {})
    return (
        abs(float(delta.get("best_progress_delta_m", 0.0) or 0.0)) <= 0.5
        and abs(float(delta.get("final_progress_delta_m", 0.0) or 0.0)) <= 0.5
        and not bool(delta.get("reason_mismatch", False))
        and not bool(delta.get("valid_lap_mismatch", False))
    )


def _pool_row_replay_key(
    row: dict[str, Any],
    *,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    snapshots_by_generation: dict[int, list[StateSnapshot]],
) -> str:
    generation = int(row["generation"])
    snapshot = snapshots_by_generation[generation][int(row["snapshot_index"])]
    return _postcheck_replay_cache_key(row, config=config, gates=gates, snapshot=snapshot)


def _snapshot_cache_key(row: dict[str, Any]) -> int:
    return int(row.get("generation", 0))


def _gates_payload(gates: EvolutionGates) -> dict[str, Any]:
    return {field.name: getattr(gates, field.name) for field in fields(EvolutionGates)}


def _postcheck_replay_cache_key(
    row: dict[str, Any],
    *,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    snapshot: StateSnapshot,
) -> str:
    sim_config = _sim_config_for_generation(config, int(row["generation"]))
    payload = {
        "genome": row["genome"],
        "snapshot": snapshot_to_dict(snapshot),
        "sim_config": repr(sim_config),
        "gates": _gates_payload(gates),
        "scoring_profiles": tuple(config.scoring_profiles),
        "frontier_focus_start_m": float(config.frontier_focus_start_m),
        "frontier_focus_end_m": float(config.frontier_focus_end_m),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("backend_parity_error", {}) for row in rows]
    max_best_delta = max((abs(float(delta.get("best_progress_delta_m", 0.0) or 0.0)) for delta in deltas), default=0.0)
    max_final_delta = max(
        (abs(float(delta.get("final_progress_delta_m", 0.0) or 0.0)) for delta in deltas),
        default=0.0,
    )
    max_score_delta = max((abs(float(delta.get("score_delta", 0.0) or 0.0)) for delta in deltas), default=0.0)
    reason_mismatches = sum(1 for delta in deltas if bool(delta.get("reason_mismatch", False)))
    valid_lap_mismatches = sum(1 for delta in deltas if bool(delta.get("valid_lap_mismatch", False)))
    parity_status = (
        "passed"
        if max_best_delta <= 0.5
        and max_final_delta <= 0.5
        and reason_mismatches == 0
        and valid_lap_mismatches == 0
        else "failed"
    )
    score_parity_status = "passed" if max_score_delta <= 1.0 else "failed"
    return {
        "parity_status": parity_status,
        "score_parity_status": score_parity_status,
        "max_best_progress_delta_m": max_best_delta,
        "max_final_progress_delta_m": max_final_delta,
        "max_score_delta": max_score_delta,
        "reason_mismatches": reason_mismatches,
        "valid_lap_mismatches": valid_lap_mismatches,
    }


def _replay_row(
    row: dict[str, Any],
    *,
    config: EvolutionSearchConfig,
    gates: EvolutionGates,
    snapshots_by_generation: dict[int, list[StateSnapshot]],
    telemetry_dir: Path,
    telemetry_compression: str,
    rank: int,
    collect_full_telemetry: bool = True,
    write_telemetry: bool = True,
    replay_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = int(row["generation"])
    sim_config = _sim_config_for_generation(config, generation)
    snapshots = snapshots_by_generation[generation]
    snapshot = snapshots[int(row["snapshot_index"])]
    started = time.perf_counter()
    cache_hit = False
    cache_key = (
        _postcheck_replay_cache_key(row, config=config, gates=gates, snapshot=snapshot)
        if replay_cache is not None and not collect_full_telemetry and not write_telemetry
        else None
    )
    cached_row = replay_cache.get(cache_key) if replay_cache is not None and cache_key is not None else None
    if cached_row is not None:
        cache_hit = True
        cpu_row = copy.deepcopy(cached_row)
        replay_seconds = 0.0
    else:
        telemetry_rows, _sim = _run_candidate(
            sim_config=sim_config,
            snapshot=snapshot,
            genome=genome_from_mapping(row["genome"]),
            gates=gates,
            seed=int(row["seed"]),
            collect_full_telemetry=collect_full_telemetry,
        )
        replay_seconds = time.perf_counter() - started
        profile_scores = _score_rows(
            telemetry_rows,
            start_progress_m=snapshot.monotonic_progress_m,
            gates=gates,
            scoring_profiles=config.scoring_profiles,
            frontier_focus_start_m=config.frontier_focus_start_m,
            frontier_focus_end_m=config.frontier_focus_end_m,
        )
        final = telemetry_rows[-1] if telemetry_rows else {}
        best_progress_m = max(
            [snapshot.monotonic_progress_m, *[float(item["monotonic_progress_m"]) for item in telemetry_rows]]
        )
        primary_profile = config.scoring_profiles[0]
        cpu_score = float(profile_scores[primary_profile])
        final_progress_m = float(final.get("monotonic_progress_m", snapshot.monotonic_progress_m) or 0.0)
        reason = str(row.get("postcheck_selection_reason", "selected"))
        suffix = _telemetry_file_suffix(telemetry_compression)
        telemetry_path: Path | None = None
        if write_telemetry:
            telemetry_path = telemetry_dir / (
                f"postcheck-{_safe_slug(reason)}-rank-{rank:03d}-gen-{generation:03d}-"
                f"candidate-{int(row['candidate_index']):05d}-steps{suffix}"
            )
            _write_jsonl(telemetry_path, telemetry_rows)
        cpu_row = {
            "candidate_index": int(row["candidate_index"]),
            "generation": generation,
            "seed": int(row["seed"]),
            "score": cpu_score,
            "primary_scoring_profile": primary_profile,
            "profile_scores": profile_scores,
            "snapshot_index": int(row["snapshot_index"]),
            "start_snapshot_id": snapshot.id,
            "start_progress_m": snapshot.monotonic_progress_m,
            "start_speed_kph": snapshot.speed_mps * 3.6,
            "genome": row["genome"],
            "lineage": row.get("lineage", {}),
            "best_progress_m": best_progress_m,
            "final_progress_m": final_progress_m,
            "remaining_m": max(0.0, gates.target_progress_m - best_progress_m),
            "segment_complete": bool(final.get("segment_complete", False) or best_progress_m >= gates.target_progress_m),
            "target_reached": bool(best_progress_m >= gates.target_progress_m),
            "sim_segment_complete": bool(final.get("segment_complete", False)),
            "completed_lap": bool(final.get("completed_lap", False)),
            "valid_lap": bool(final.get("valid_lap", False)),
            "finish_crossed": bool(final.get("finish_crossed", False)),
            "termination_reason": final.get("termination_reason", "empty"),
            "collided": bool(final.get("collided", False)),
            "off_track": bool(final.get("off_track", False)),
            "final_speed_kph": final.get("speed_kph"),
            "final_lateral_error_m": final.get("lateral_error_m"),
            "final_heading_error_deg": final.get("heading_error_deg"),
            "final_yaw_rate_rps": final.get("yaw_rate_rps"),
            "final_steering": final.get("steering"),
            "final_row": final,
            "max_steps": int(sim_config.max_steps),
            "steps": len(telemetry_rows),
            "elapsed_s": final.get("sim_time_s", math.inf),
            "backend": "cpu_postcheck",
            "postcheck_selection_reason": reason,
        }
        if telemetry_path is not None:
            cpu_row["selected_telemetry"] = str(telemetry_path)
        if replay_cache is not None and cache_key is not None:
            replay_cache[cache_key] = copy.deepcopy(cpu_row)
    cpu_row.update(
        {
            "candidate_index": int(row["candidate_index"]),
            "generation": generation,
            "seed": int(row["seed"]),
            "snapshot_index": int(row["snapshot_index"]),
            "start_snapshot_id": snapshot.id,
            "genome": row["genome"],
            "lineage": row.get("lineage", {}),
            "postcheck_selection_reason": str(row.get("postcheck_selection_reason", "selected")),
            "cpu_replay_cache_hit": cache_hit,
        }
    )
    primary_profile = config.scoring_profiles[0]
    cpu_score = float(cpu_row["score"])
    gpu_score = float(row.get("score", 0.0) or 0.0)
    final_progress_m = float(cpu_row.get("final_progress_m", snapshot.monotonic_progress_m) or 0.0)
    gpu_final_progress_m = float(row.get("final_progress_m", 0.0) or 0.0)
    reason = str(row.get("postcheck_selection_reason", "selected"))
    telemetry_path_raw = cpu_row.get("selected_telemetry")
    telemetry_path = Path(str(telemetry_path_raw)) if telemetry_path_raw else None
    cpu_row["gpu_score"] = gpu_score
    cpu_row["gpu_profile_scores"] = row.get("profile_scores", {})
    cpu_row["cpu_verified_score"] = cpu_score
    delta = {
        "score_delta": cpu_score - gpu_score,
        "best_progress_delta_m": float(cpu_row.get("best_progress_m", 0.0) or 0.0)
        - float(row.get("best_progress_m", 0.0) or 0.0),
        "final_progress_delta_m": final_progress_m - gpu_final_progress_m,
        "gpu_termination_reason": row.get("termination_reason"),
        "cpu_termination_reason": cpu_row.get("termination_reason"),
        "reason_mismatch": row.get("termination_reason") != cpu_row.get("termination_reason"),
        "valid_lap_mismatch": bool(row.get("valid_lap")) != bool(cpu_row.get("valid_lap")),
        "cpu_replay_seconds": replay_seconds,
    }
    cpu_row["backend_parity_error"] = delta
    manifest_row = {
        "rank": rank,
        "selection_reason": reason,
        "generation": generation,
        "candidate_index": int(row["candidate_index"]),
        "seed": int(row["seed"]),
        "score": cpu_score,
        "gpu_score": gpu_score,
        "profile_scores": cpu_row.get("profile_scores", {}),
        "gpu_profile_scores": row.get("profile_scores", {}),
        "best_progress_m": cpu_row.get("best_progress_m"),
        "gpu_best_progress_m": row.get("best_progress_m"),
        "final_progress_m": final_progress_m,
        "gpu_final_progress_m": row.get("final_progress_m"),
        "termination_reason": cpu_row.get("termination_reason"),
        "gpu_termination_reason": row.get("termination_reason"),
        "path": str(telemetry_path) if telemetry_path is not None else None,
    }
    return cpu_row, manifest_row


def _replay_compact_pool_worker(payload: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    row = dict(payload["row"])
    rank = int(payload["rank"])
    cpu_row, manifest_row = _replay_row(
        row,
        config=payload["config"],
        gates=payload["gates"],
        snapshots_by_generation=payload["snapshots_by_generation"],
        telemetry_dir=Path(str(payload["telemetry_dir"])),
        telemetry_compression=str(payload["telemetry_compression"]),
        rank=rank,
        collect_full_telemetry=False,
        write_telemetry=False,
        replay_cache=None,
    )
    return rank, cpu_row, manifest_row


def postcheck_evolution_run(
    run_dir: Path,
    *,
    top_k: int = 8,
    candidate_pool_size: int | None = None,
    cpu_rerank: bool = False,
    workers: int = 1,
    telemetry_compression: str = "gzip",
) -> Path:
    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / CHECKPOINT_NAME
    attempts_path = run_dir / "attempts.jsonl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for postcheck: {checkpoint_path}")
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts for postcheck: {attempts_path}")
    checkpoint = _load_json(checkpoint_path)
    config = _config_from_mapping(dict(checkpoint.get("config", {})))
    gates = _gates_from_mapping(dict(checkpoint.get("gates", {})))
    state_library_raw = checkpoint.get("state_library")
    state_library = Path(str(state_library_raw)) if state_library_raw is not None else None
    attempts = _load_jsonl(attempts_path)
    if not attempts:
        raise ValueError(f"No attempts found in {attempts_path}")
    selected = _select_postcheck_rows(
        attempts,
        scoring_profiles=config.scoring_profiles,
        top_k=max(1, int(top_k)),
        candidate_pool_size=candidate_pool_size,
    )
    snapshots_by_generation: dict[int, list[StateSnapshot]] = {}
    for row in selected:
        generation = _snapshot_cache_key(row)
        if generation not in snapshots_by_generation:
            sim_config = _sim_config_for_generation(config, generation)
            snapshots_by_generation[generation] = _load_snapshots(
                sim_config=sim_config,
                config=config,
                state_library=state_library,
                start_progress_m=checkpoint.get("start_progress_m"),
                start_speed_kph=float(checkpoint.get("start_speed_kph", 0.0) or 0.0),
                start_min_progress_m=checkpoint.get("start_min_progress_m"),
                start_max_progress_m=checkpoint.get("start_max_progress_m"),
            )

    telemetry_dir = run_dir / "selected_telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pool_cpu_rows: list[dict[str, Any]] = []
    pool_manifest_rows: list[dict[str, Any]] = []
    pool_attempts_path = run_dir / "postchecked_pool_attempts.jsonl"
    pool_progress_path = run_dir / "postcheck_pool_progress.jsonl"
    pool_requested_count = len(selected)
    pool_duplicate_skips = 0
    pool_worker_count = max(1, int(workers))
    if cpu_rerank:
        pool_attempts_path.write_text("", encoding="utf-8")
        pool_progress_path.write_text("", encoding="utf-8")
        unique_rows: list[tuple[int, dict[str, Any], str]] = []
        seen_replay_keys: dict[str, int] = {}
        for rank, row in enumerate(selected):
            replay_key = _pool_row_replay_key(
                row,
                config=config,
                gates=gates,
                snapshots_by_generation=snapshots_by_generation,
            )
            duplicate_of = seen_replay_keys.get(replay_key)
            if duplicate_of is not None:
                pool_duplicate_skips += 1
                progress = {
                    "event": "duplicate_skip",
                    "rank": rank,
                    "duplicate_of_rank": duplicate_of,
                    "generation": int(row["generation"]),
                    "candidate_index": int(row["candidate_index"]),
                    "selection_reason": row.get("postcheck_selection_reason"),
                    "replay_key": replay_key,
                    "cache_hit": True,
                }
                _append_jsonl_row(pool_progress_path, progress)
                print(
                    "postcheck_pool_duplicate_skip "
                    f"rank={rank} duplicate_of_rank={duplicate_of} "
                    f"generation={int(row['generation'])} candidate={int(row['candidate_index'])}",
                    flush=True,
                )
                continue
            seen_replay_keys[replay_key] = rank
            unique_rows.append((rank, row, replay_key))

        print(
            "postcheck_pool_start "
            f"requested={pool_requested_count} unique={len(unique_rows)} "
            f"duplicate_skips={pool_duplicate_skips} workers={pool_worker_count}",
            flush=True,
        )

        def record_pool_result(rank: int, row: dict[str, Any], replay_key: str, cpu_row: dict[str, Any]) -> None:
            pool_cpu_rows.append(cpu_row)
            _append_jsonl_row(pool_attempts_path, cpu_row)
            delta = dict(cpu_row.get("backend_parity_error", {}) or {})
            progress = {
                "event": "replay_complete",
                "completed_count": len(pool_cpu_rows),
                "requested_count": pool_requested_count,
                "unique_count": len(unique_rows),
                "rank": rank,
                "generation": int(row["generation"]),
                "candidate_index": int(row["candidate_index"]),
                "selection_reason": row.get("postcheck_selection_reason"),
                "replay_key": replay_key,
                "cache_hit": False,
                "cpu_replay_seconds": float(delta.get("cpu_replay_seconds", 0.0) or 0.0),
                "score_delta": float(delta.get("score_delta", 0.0) or 0.0),
                "best_progress_delta_m": float(delta.get("best_progress_delta_m", 0.0) or 0.0),
                "final_progress_delta_m": float(delta.get("final_progress_delta_m", 0.0) or 0.0),
                "reason_mismatch": bool(delta.get("reason_mismatch", False)),
                "valid_lap_mismatch": bool(delta.get("valid_lap_mismatch", False)),
            }
            _append_jsonl_row(pool_progress_path, progress)
            print(
                "postcheck_pool_replay_complete "
                f"done={len(pool_cpu_rows)}/{len(unique_rows)} rank={rank} "
                f"generation={int(row['generation'])} candidate={int(row['candidate_index'])} "
                f"seconds={progress['cpu_replay_seconds']:.3f} "
                f"reason_mismatch={int(progress['reason_mismatch'])} "
                f"valid_lap_mismatch={int(progress['valid_lap_mismatch'])}",
                flush=True,
            )

        if pool_worker_count > 1 and len(unique_rows) > 1:
            with ProcessPoolExecutor(max_workers=pool_worker_count) as executor:
                future_map = {
                    executor.submit(
                        _replay_compact_pool_worker,
                        {
                            "rank": rank,
                            "row": row,
                            "config": config,
                            "gates": gates,
                            "snapshots_by_generation": snapshots_by_generation,
                            "telemetry_dir": telemetry_dir,
                            "telemetry_compression": telemetry_compression,
                        },
                    ): (rank, row, replay_key)
                    for rank, row, replay_key in unique_rows
                }
                for future in as_completed(future_map):
                    rank, row, replay_key = future_map[future]
                    _result_rank, cpu_row, _manifest_row = future.result()
                    record_pool_result(rank, row, replay_key, cpu_row)
        else:
            for rank, row, replay_key in unique_rows:
                cpu_row, _manifest_row = _replay_row(
                    row,
                    config=config,
                    gates=gates,
                    snapshots_by_generation=snapshots_by_generation,
                    telemetry_dir=telemetry_dir,
                    telemetry_compression=telemetry_compression,
                    rank=rank,
                    collect_full_telemetry=False,
                    write_telemetry=False,
                )
                record_pool_result(rank, row, replay_key, cpu_row)
    else:
        for rank, row in enumerate(selected):
            cpu_row, manifest_row = _replay_row(
                row,
                config=config,
                gates=gates,
                snapshots_by_generation=snapshots_by_generation,
                telemetry_dir=telemetry_dir,
                telemetry_compression=telemetry_compression,
                rank=rank,
                collect_full_telemetry=True,
                write_telemetry=True,
            )
            pool_cpu_rows.append(cpu_row)
            pool_manifest_rows.append(manifest_row)

    pool_metrics = _parity_metrics(pool_cpu_rows)
    pool_cache_hits = pool_duplicate_skips
    pool_unique_replays = len(pool_cpu_rows)
    cpu_rows = pool_cpu_rows
    manifest_rows = pool_manifest_rows
    cpu_rerank_rejected_mismatch_count = 0
    if cpu_rerank:
        clean_pool_rows = [row for row in pool_cpu_rows if _row_has_clean_state_parity(row)]
        cpu_rerank_rejected_mismatch_count = len(pool_cpu_rows) - len(clean_pool_rows)
        selection_pool_rows = clean_pool_rows if clean_pool_rows else pool_cpu_rows
        rows_by_key = {_row_identity(row): row for row in selected}
        reranked = _select_cpu_rerank_rows(
            selection_pool_rows,
            scoring_profiles=config.scoring_profiles,
            top_k=max(1, int(top_k)),
        )
        cpu_rows = []
        manifest_rows = []
        for rank, cpu_selected in enumerate(reranked):
            source = rows_by_key[_row_identity(cpu_selected)]
            tagged_source = dict(source)
            tagged_source["postcheck_selection_reason"] = cpu_selected.get(
                "postcheck_selection_reason", "cpu_rerank_selected"
            )
            cpu_row, manifest_row = _replay_row(
                tagged_source,
                config=config,
                gates=gates,
                snapshots_by_generation=snapshots_by_generation,
                telemetry_dir=telemetry_dir,
                telemetry_compression=telemetry_compression,
                rank=rank,
                collect_full_telemetry=True,
                write_telemetry=True,
            )
            cpu_rows.append(cpu_row)
            manifest_rows.append(manifest_row)

    selected_metrics = _parity_metrics(cpu_rows)
    elapsed = time.perf_counter() - started
    _write_jsonl(run_dir / "postchecked_attempts.jsonl", cpu_rows)
    manifest = {
        "kind": "evolution_postcheck_manifest",
        "source_run": str(run_dir),
        "telemetry_selection": "postcheck",
        "telemetry_compression": telemetry_compression,
        "backend": "cpu_postcheck",
        "trace_count": len(manifest_rows),
        "traces": manifest_rows,
    }
    _write_json(telemetry_dir / "manifest.json", manifest)
    summary = {
        "kind": "evolution_postcheck_summary",
        "source_run": str(run_dir),
        "attempts_path": str(attempts_path),
        "postchecked_attempts": str(run_dir / "postchecked_attempts.jsonl"),
        "selected_telemetry_manifest": str(telemetry_dir / "manifest.json"),
        "selection_mode": "cpu_rerank" if cpu_rerank else "gpu_selected",
        "candidate_pool_size": candidate_pool_size,
        "postcheck_pool_progress": str(pool_progress_path) if cpu_rerank else None,
        "postchecked_pool_attempts": str(pool_attempts_path) if cpu_rerank else None,
        "pool_postcheck_count": len(pool_cpu_rows),
        "pool_cpu_replay_requested_count": pool_requested_count,
        "pool_cpu_replay_unique_count": pool_unique_replays,
        "pool_cpu_replay_cache_hits": pool_cache_hits,
        "pool_cpu_replay_duplicate_count": pool_duplicate_skips,
        "pool_cpu_replay_cache_size": pool_unique_replays,
        "pool_cpu_replay_workers": pool_worker_count if cpu_rerank else 1,
        "cpu_rerank_clean_pool_count": len(pool_cpu_rows) - cpu_rerank_rejected_mismatch_count,
        "cpu_rerank_rejected_mismatch_count": cpu_rerank_rejected_mismatch_count,
        "pool_parity_status": pool_metrics["parity_status"],
        "pool_score_parity_status": pool_metrics["score_parity_status"],
        "pool_max_best_progress_delta_m": pool_metrics["max_best_progress_delta_m"],
        "pool_max_final_progress_delta_m": pool_metrics["max_final_progress_delta_m"],
        "pool_max_score_delta": pool_metrics["max_score_delta"],
        "pool_reason_mismatches": pool_metrics["reason_mismatches"],
        "pool_valid_lap_mismatches": pool_metrics["valid_lap_mismatches"],
        "cpu_rerank_status": "passed" if cpu_rerank and cpu_rows else ("not_requested" if not cpu_rerank else "failed"),
        "postcheck_count": len(cpu_rows),
        "postcheck_seconds": elapsed,
        "parity_status": selected_metrics["parity_status"],
        "score_parity_status": selected_metrics["score_parity_status"],
        "max_best_progress_delta_m": selected_metrics["max_best_progress_delta_m"],
        "max_final_progress_delta_m": selected_metrics["max_final_progress_delta_m"],
        "max_score_delta": selected_metrics["max_score_delta"],
        "reason_mismatches": selected_metrics["reason_mismatches"],
        "valid_lap_mismatches": selected_metrics["valid_lap_mismatches"],
        "selected": manifest_rows,
    }
    _write_json(run_dir / "postcheck_summary.json", summary)
    return run_dir / "postcheck_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-replay selected attempts from an evolution search run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        help="Replay this many GPU top-score rows before optional CPU reranking.",
    )
    parser.add_argument(
        "--cpu-rerank",
        action="store_true",
        help="CPU-replay a candidate pool, rerank by CPU score/profile/progress, and write telemetry only for winners.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel CPU workers for the compact deferred pool replay pass.",
    )
    parser.add_argument("--telemetry-compression", choices=("none", "gzip"), default="gzip")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = postcheck_evolution_run(
        args.run_dir,
        top_k=max(1, args.top_k),
        candidate_pool_size=args.candidate_pool_size,
        cpu_rerank=bool(args.cpu_rerank),
        workers=max(1, int(args.workers)),
        telemetry_compression=args.telemetry_compression,
    )
    summary = _load_json(summary_path)
    print(
        "evolution_postcheck_complete "
        f"run={args.run_dir} "
        f"summary={summary_path} "
        f"count={summary['postcheck_count']} "
        f"selection_mode={summary['selection_mode']} "
        f"parity_status={summary['parity_status']} "
        f"reason_mismatches={summary['reason_mismatches']} "
        f"valid_lap_mismatches={summary['valid_lap_mismatches']} "
        f"pool_parity_status={summary['pool_parity_status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
