"""Export and inspect CPU-verified ES transition datasets for learned policies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import (
    DATASETS_DIR,
    LEARNED_POLICY_V1_FEATURES,
    SimConfig,
    actions_for_action_set,
    dataclass_to_dict,
)
from f1rl.evolution_postcheck import _config_from_mapping, _gates_from_mapping
from f1rl.evolution_search import (
    EvolutionGates,
    EvolutionSearchConfig,
    _action_for_progress_delta,
    _action_for_step,
    _compact_row,
    _controller_controls,
    _default_start_snapshot,
    _reset_options,
    _sim_config_for_generation,
    genome_from_mapping,
)
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import StateSnapshot, snapshot_to_dict

DATASET_SCHEMA_VERSION = 1
ACTION_SCHEMA = ("throttle", "brake", "steer")
NUMERIC_KEYS = (
    "reward",
    "lap_time_s",
    "sim_time_s",
    "progress_m",
    "progress_delta_m",
    "speed_kph",
    "throttle",
    "brake",
    "steer",
    "heading_error_deg",
    "lateral_error_m",
)


def observation_feature_schema(observation_profile: str) -> dict[str, Any]:
    if observation_profile == "learned_policy_v1":
        return {
            "profile": "learned_policy_v1",
            "base_profile": "racing_v2",
            "ordered_blocks": [
                {
                    "name": "racing_v2",
                    "description": "Existing public MonzaSim racing_v2 observation block.",
                },
                {
                    "name": "learned_policy_v1_append",
                    "features": list(LEARNED_POLICY_V1_FEATURES),
                },
            ],
        }
    return {"profile": observation_profile}


class TransitionDataset:
    def __init__(self, root: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.root = root
        self.manifest = manifest
        self.arrays = arrays

    @property
    def obs(self) -> np.ndarray:
        return self.arrays["obs"]

    @property
    def actions(self) -> np.ndarray:
        return self.arrays["action"]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _jsonl_open(path: Path, mode: str = "wt") -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    bridge_path = run_dir / "ppo_bridge.json"
    if bridge_path.exists():
        return _read_json(bridge_path)
    summary_path = run_dir / "evolution_summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    checkpoint_path = run_dir / "population_checkpoint.json"
    if checkpoint_path.exists():
        payload = _read_json(checkpoint_path)
        if "top_attempts" not in payload and isinstance(payload.get("best_rows"), list):
            payload = dict(payload)
            payload["top_attempts"] = list(payload["best_rows"])
        if "config" in payload and "gates" in payload:
            return payload
    raise FileNotFoundError(
        f"Missing ppo_bridge.json, evolution_summary.json, or usable population_checkpoint.json in {run_dir}"
    )


def _row_identity(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row.get("generation", 0)), int(row.get("candidate_index", 0)), int(row.get("seed", 0))


def _lineage_root(lineage: Any) -> str:
    current = lineage if isinstance(lineage, dict) else {}
    last_source = str(current.get("source", "unknown"))
    while isinstance(current.get("previous_lineage"), dict):
        current = current["previous_lineage"]
        last_source = str(current.get("source", last_source))
    parent = current.get("parent")
    if isinstance(parent, dict):
        return f"{last_source}:{parent.get('generation')}:{parent.get('candidate_index')}"
    return last_source


def _genome_hash(row: dict[str, Any]) -> str:
    return _sha256_json(row.get("genome", {}))[:24]


def _candidate_bucket(row: dict[str, Any]) -> str:
    reason = str(row.get("termination_reason", ""))
    progress = float(row.get("best_progress_m", row.get("final_progress_m", 0.0)) or 0.0)
    valid = bool(row.get("completed_lap") or row.get("valid_lap") or row.get("finish_crossed") or reason == "lap_complete")
    if valid:
        return "valid_lap"
    if progress >= 5500.0:
        return "near_valid_5500m"
    if progress >= 5000.0:
        return "near_valid_5000m"
    if progress >= 4000.0:
        return "late_frontier"
    if progress >= 2000.0:
        return "mid_frontier"
    return "early_failure"


def _elapsed_s(row: dict[str, Any]) -> float:
    value = row.get("elapsed_s")
    if value is None and isinstance(row.get("final_row"), dict):
        value = row["final_row"].get("sim_time_s")
    if value is None:
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _selection_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    bucket_order = {
        "valid_lap": 0,
        "near_valid_5500m": 1,
        "near_valid_5000m": 2,
        "late_frontier": 3,
        "mid_frontier": 4,
        "early_failure": 5,
    }
    bucket = _candidate_bucket(row)
    elapsed = _elapsed_s(row)
    elapsed_key = elapsed if math.isfinite(elapsed) else 1_000_000.0
    return (
        bucket_order[bucket],
        elapsed_key,
        -float(row.get("best_progress_m", 0.0) or 0.0),
        -float(row.get("score", 0.0) or 0.0),
    )


def _load_top_genome_rows(run_dir: Path, *, max_per_generation: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top_dir = run_dir / "top_genomes"
    if not top_dir.exists():
        return rows
    for path in sorted(top_dir.glob("generation_*.json")):
        payload = _read_json(path)
        generation = int(payload.get("generation", 0))
        for row in list(payload.get("top", []))[:max_per_generation]:
            item = dict(row)
            item.setdefault("generation", generation)
            if "seed" not in item:
                item["seed"] = generation * 1_000_000 + int(item.get("candidate_index", 0))
            rows.append(item)
    return rows


def _load_attempt_rows(run_dir: Path) -> list[dict[str, Any]]:
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        return []
    return _read_jsonl(attempts_path)


def _load_source_rows(run_dir: Path, *, max_per_generation: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("postchecked_summary_top_attempts.jsonl", "postchecked_attempts.jsonl"):
        rows.extend(_read_jsonl(run_dir / name))
    payload = _load_run_payload(run_dir)
    rows.extend(dict(row) for row in payload.get("top_attempts", []))
    rows.extend(_load_top_genome_rows(run_dir, max_per_generation=max_per_generation))
    rows.extend(_load_attempt_rows(run_dir))
    best_path = run_dir / "best_so_far.json"
    if best_path.exists():
        rows.append(_read_json(best_path))

    selected: list[dict[str, Any]] = []
    seen_identity: set[tuple[int, int, int]] = set()
    seen_genomes: set[str] = set()
    for row in sorted(rows, key=_selection_key):
        if "genome" not in row:
            continue
        identity = _row_identity(row)
        genome_hash = _genome_hash(row)
        if identity in seen_identity or genome_hash in seen_genomes:
            continue
        seen_identity.add(identity)
        seen_genomes.add(genome_hash)
        selected.append(row)
    return selected


def _balanced_source_rows(rows: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    if max_candidates <= 0:
        return []
    target_counts = {
        "valid_lap": max(1, int(max_candidates * 0.70)),
        "near_valid_5500m": max(1, int(max_candidates * 0.08)),
        "near_valid_5000m": max(1, int(max_candidates * 0.08)),
        "late_frontier": max(1, int(max_candidates * 0.06)),
        "mid_frontier": max(1, int(max_candidates * 0.06)),
        "early_failure": 1,
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int, int]] = set()
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(_candidate_bucket(row), []).append(row)

    for bucket, target_count in target_counts.items():
        for row in by_bucket.get(bucket, [])[:target_count]:
            if len(selected) >= max_candidates:
                return selected
            identity = _row_identity(row)
            if identity in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(identity)

    for row in rows:
        if len(selected) >= max_candidates:
            break
        identity = _row_identity(row)
        if identity in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(identity)
    return selected


def _source_snapshot_for_row(
    row: dict[str, Any],
    *,
    sim_config: SimConfig,
    config: EvolutionSearchConfig,
) -> StateSnapshot:
    start_progress = row.get("start_progress_m")
    start_speed = float(row.get("start_speed_kph", 80.0) or 80.0)
    return _default_start_snapshot(
        sim_config=sim_config,
        seed=int(config.seed),
        start_progress_m=float(start_progress) if start_progress is not None else None,
        start_speed_kph=start_speed,
    )


def _run_transition_replay(
    *,
    sim_config: SimConfig,
    snapshot: StateSnapshot,
    genome_payload: dict[str, Any],
    gates: EvolutionGates,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sim = MonzaSim(sim_config)
    sim.reset(seed=seed, options=_reset_options(snapshot, gates, collect_observation=True))
    genome = genome_from_mapping(genome_payload)
    action_specs = {
        name: (index, throttle, brake, steer)
        for index, (name, throttle, brake, steer) in enumerate(actions_for_action_set(sim.config.action_set))
    }
    start_progress_m = snapshot.monotonic_progress_m
    obs_rows: list[np.ndarray] = []
    next_obs_rows: list[np.ndarray] = []
    action_rows: list[tuple[float, float, float]] = []
    scalars: dict[str, list[Any]] = {
        "reward": [],
        "done": [],
        "terminated": [],
        "truncated": [],
        "terminal_reason": [],
        "valid_lap": [],
        "completed_lap": [],
        "lap_time_s": [],
        "sim_time_s": [],
        "progress_m": [],
        "progress_delta_m": [],
        "speed_kph": [],
        "throttle": [],
        "brake": [],
        "steer": [],
        "heading_error_deg": [],
        "lateral_error_m": [],
        "checkpoint_index": [],
    }
    rows: list[dict[str, Any]] = []
    for step_index in range(sim_config.max_steps):
        obs = sim.observation()
        search_features = sim.search_features(
            segment_start_progress_m=start_progress_m,
            segment_target_progress_m=gates.target_progress_m,
        )
        if genome.kind == "controller":
            throttle, brake, steer = _controller_controls(genome, search_features)
            action_id = -2
        elif genome.kind == "progress_phase":
            progress_delta_m = max(0.0, sim.state.monotonic_progress_m - start_progress_m)
            action_name = _action_for_progress_delta(genome, progress_delta_m)
            action_id, throttle, brake, steer = action_specs[action_name]
        else:
            action_name = _action_for_step(genome, step_index)
            action_id, throttle, brake, steer = action_specs[action_name]
        result = sim.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=action_id)
        telemetry = asdict(result.telemetry)
        telemetry.update(_compact_row(result.telemetry, search_features))
        rows.append(telemetry)
        done = bool(result.terminated or result.truncated)
        lap_time_s = float(result.telemetry.sim_time_s) if result.telemetry.termination_reason == "lap_complete" else np.nan

        obs_rows.append(obs)
        next_obs_rows.append(result.observation)
        action_rows.append((float(throttle), float(brake), float(steer)))
        scalars["reward"].append(float(result.reward))
        scalars["done"].append(done)
        scalars["terminated"].append(bool(result.terminated))
        scalars["truncated"].append(bool(result.truncated))
        scalars["terminal_reason"].append(result.telemetry.termination_reason if done else "active")
        scalars["valid_lap"].append(bool(result.telemetry.valid_lap))
        scalars["completed_lap"].append(bool(sim.completed_lap))
        scalars["lap_time_s"].append(lap_time_s)
        scalars["sim_time_s"].append(float(result.telemetry.sim_time_s))
        scalars["progress_m"].append(float(result.telemetry.monotonic_progress_m))
        scalars["progress_delta_m"].append(float(result.telemetry.progress_delta_m))
        scalars["speed_kph"].append(float(result.telemetry.speed_kph))
        scalars["throttle"].append(float(result.telemetry.throttle))
        scalars["brake"].append(float(result.telemetry.brake))
        scalars["steer"].append(float(result.telemetry.steering))
        scalars["heading_error_deg"].append(float(result.telemetry.heading_error_deg))
        scalars["lateral_error_m"].append(float(result.telemetry.lateral_error_m))
        scalars["checkpoint_index"].append(int(result.telemetry.checkpoint_index))
        if done:
            break

    arrays: dict[str, np.ndarray] = {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "next_obs": np.asarray(next_obs_rows, dtype=np.float32),
        "action": np.asarray(action_rows, dtype=np.float32),
        "reward": np.asarray(scalars["reward"], dtype=np.float32),
        "done": np.asarray(scalars["done"], dtype=np.bool_),
        "terminated": np.asarray(scalars["terminated"], dtype=np.bool_),
        "truncated": np.asarray(scalars["truncated"], dtype=np.bool_),
        "terminal_reason": np.asarray(scalars["terminal_reason"], dtype="<U64"),
        "valid_lap": np.asarray(scalars["valid_lap"], dtype=np.bool_),
        "completed_lap": np.asarray(scalars["completed_lap"], dtype=np.bool_),
        "lap_time_s": np.asarray(scalars["lap_time_s"], dtype=np.float32),
        "sim_time_s": np.asarray(scalars["sim_time_s"], dtype=np.float32),
        "progress_m": np.asarray(scalars["progress_m"], dtype=np.float32),
        "progress_delta_m": np.asarray(scalars["progress_delta_m"], dtype=np.float32),
        "speed_kph": np.asarray(scalars["speed_kph"], dtype=np.float32),
        "throttle": np.asarray(scalars["throttle"], dtype=np.float32),
        "brake": np.asarray(scalars["brake"], dtype=np.float32),
        "steer": np.asarray(scalars["steer"], dtype=np.float32),
        "heading_error_deg": np.asarray(scalars["heading_error_deg"], dtype=np.float32),
        "lateral_error_m": np.asarray(scalars["lateral_error_m"], dtype=np.float32),
        "checkpoint_index": np.asarray(scalars["checkpoint_index"], dtype=np.int32),
    }
    final = rows[-1] if rows else {}
    summary = {
        "steps": int(arrays["obs"].shape[0]),
        "termination_reason": final.get("termination_reason", "empty"),
        "valid_lap": bool(final.get("valid_lap", False)),
        "completed_lap": bool(final.get("termination_reason") == "lap_complete"),
        "finish_crossed": bool(final.get("finish_crossed", False)),
        "lap_time_s": float(final.get("sim_time_s", math.inf)) if final.get("termination_reason") == "lap_complete" else None,
        "best_progress_m": max([snapshot.monotonic_progress_m, *[float(row["monotonic_progress_m"]) for row in rows]]),
        "final_progress_m": float(final.get("monotonic_progress_m", snapshot.monotonic_progress_m) or 0.0),
    }
    return arrays, summary


def _append_candidate_arrays(
    target: dict[str, list[np.ndarray]],
    arrays: dict[str, np.ndarray],
    *,
    source_candidate_id: int,
) -> None:
    count = int(arrays["obs"].shape[0])
    for key, values in arrays.items():
        target.setdefault(key, []).append(values)
    target.setdefault("source_candidate_id", []).append(np.full(count, source_candidate_id, dtype=np.int32))


def _concat_parts(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in parts.items() if values}


def _write_shard(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
    return {"path": str(path.name), "rows": int(arrays["obs"].shape[0]), "sha256": _sha256_file(path)}


def export_dataset(
    *,
    run_dir: Path,
    selected_telemetry: Path | None,
    output_dir: Path,
    observation_profile: str,
    physics_model: str,
    max_candidates: int,
    max_per_generation: int,
    balanced_buckets: bool,
    source_json: Path | None = None,
) -> Path:
    if physics_model != "v1":
        raise ValueError("Current learned-policy dataset export only supports physics_model='v1'.")
    payload = _load_run_payload(run_dir)
    config = _config_from_mapping(payload["config"])
    gates = _gates_from_mapping(payload["gates"])
    if source_json is not None:
        source_path_arg = source_json
        if not source_path_arg.is_absolute() and not source_path_arg.exists():
            source_path_arg = run_dir / source_path_arg
        source_rows = [_read_json(source_path_arg)]
    else:
        all_source_rows = _load_source_rows(run_dir, max_per_generation=max_per_generation)
        source_rows = (
            _balanced_source_rows(all_source_rows, max_candidates=max_candidates)
            if balanced_buckets
            else all_source_rows[:max_candidates]
        )
    if not source_rows:
        raise ValueError(f"No replayable ES candidates found in {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "transition_shards"
    reports_dir = output_dir / "reports"
    source_path = output_dir / "source_candidates.jsonl"
    candidate_summaries: list[dict[str, Any]] = []
    parts: dict[str, list[np.ndarray]] = {}
    sim_config_hashes: set[str] = set()
    track_hashes: set[str] = set()
    sim_config_payloads: list[dict[str, Any]] = []

    with source_path.open("w", encoding="utf-8") as source_file:
        for source_id, row in enumerate(source_rows):
            generation = int(row.get("generation", 0))
            sim_config = _sim_config_for_generation(config, generation)
            sim_config.observation_profile = observation_profile
            snapshot = _source_snapshot_for_row(row, sim_config=sim_config, config=config)
            arrays, replay_summary = _run_transition_replay(
                sim_config=sim_config,
                snapshot=snapshot,
                genome_payload=row["genome"],
                gates=gates,
                seed=int(row.get("seed", config.seed)),
            )
            _append_candidate_arrays(parts, arrays, source_candidate_id=source_id)
            sim_config_payload = dataclass_to_dict(sim_config)
            sim_config_payloads.append(sim_config_payload)
            sim_hash = _sha256_json(sim_config_payload)
            track_hash = _sha256_file(Path(sim_config.track_path))
            sim_config_hashes.add(sim_hash)
            track_hashes.add(track_hash)
            bucket = _candidate_bucket({**row, **replay_summary})
            summary = {
                "source_candidate_id": source_id,
                "source_run_id": run_dir.name,
                "source_generation": generation,
                "source_candidate_index": int(row.get("candidate_index", -1)),
                "source_seed": int(row.get("seed", config.seed)),
                "source_selection_reason": str(row.get("postcheck_selection_reason", row.get("telemetry_selection_reason", bucket))),
                "source_bucket": bucket,
                "source_profile_scores": row.get("profile_scores", {}),
                "genome_hash": _genome_hash(row),
                "lineage_root": _lineage_root(row.get("lineage", {})),
                "physics_model": physics_model,
                "sim_config_hash": sim_hash,
                "track_hash": track_hash,
                "snapshot": snapshot_to_dict(snapshot),
                **replay_summary,
            }
            candidate_summaries.append(summary)
            source_file.write(json.dumps(summary, default=_json_default) + "\n")

    arrays = _concat_parts(parts)
    if arrays["obs"].shape[0] == 0:
        raise ValueError("No transitions were exported.")

    shard = _write_shard(shard_dir / "shard_0000.npz", arrays)
    valid_laps = [row for row in candidate_summaries if row["termination_reason"] == "lap_complete"]
    valid_times = [float(row["lap_time_s"]) for row in valid_laps if row.get("lap_time_s") is not None]
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": output_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "f1rl_es_transition_dataset",
        "source_run_list": [str(run_dir)],
        "source_artifact_paths": {
            "run_dir": str(run_dir),
            "selected_telemetry": str(selected_telemetry) if selected_telemetry is not None else None,
            "source_candidates": str(source_path.name),
            "source_json": str(source_json) if source_json is not None else None,
        },
        "postcheck_status": "cpu_replayed_export",
        "physics_model": physics_model,
        "sim_config": sim_config_payloads[0] if sim_config_payloads else None,
        "sim_config_hashes": sorted(sim_config_hashes),
        "track_hashes": sorted(track_hashes),
        "reward_config": sim_config_payloads[0].get("reward") if sim_config_payloads else None,
        "assist_config": sim_config_payloads[0].get("assist") if sim_config_payloads else None,
        "observation_profile": observation_profile,
        "observation_dim": int(arrays["obs"].shape[1]),
        "observation_feature_schema": observation_feature_schema(observation_profile),
        "action_schema": list(ACTION_SCHEMA),
        "normalization_strategy": "dataset_mean_std",
        "exporter_git_commit": _git_commit(),
        "candidate_selection_counts_by_bucket": dict(Counter(row["source_bucket"] for row in candidate_summaries)),
        "total_transitions": int(arrays["obs"].shape[0]),
        "total_source_candidates": len(candidate_summaries),
        "valid_lap_count": len(valid_laps),
        "fastest_source_lap": min(valid_times) if valid_times else None,
        "mean_valid_lap_time": float(np.mean(valid_times)) if valid_times else None,
        "average_progress_m": float(np.mean([row["best_progress_m"] for row in candidate_summaries])),
        "median_progress_m": float(np.median([row["best_progress_m"] for row in candidate_summaries])),
        "max_progress_m": float(max(row["best_progress_m"] for row in candidate_summaries)),
        "storage_size_bytes": int(sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())),
        "shards": [shard],
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    _write_json(reports_dir / "dataset_report.json", build_report(output_dir))
    return output_dir / "dataset_manifest.json"


def load_dataset(dataset: Path) -> TransitionDataset:
    root = dataset
    if dataset.is_file():
        root = dataset.parent
    manifest = _read_json(root / "dataset_manifest.json")
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    for shard in manifest["shards"]:
        with np.load(root / "transition_shards" / shard["path"], allow_pickle=False) as loaded:
            for key in loaded.files:
                arrays_by_key.setdefault(key, []).append(loaded[key])
    arrays = _concat_parts(arrays_by_key)
    return TransitionDataset(root=root, manifest=manifest, arrays=arrays)


def build_report(dataset: Path) -> dict[str, Any]:
    data = load_dataset(dataset)
    arrays = data.arrays
    terminal_counts = Counter(str(value) for value in arrays["terminal_reason"] if str(value) != "active")
    source_counts = Counter(int(value) for value in arrays["source_candidate_id"])
    action = arrays["action"]
    progress = arrays["progress_m"]
    completed_source_count = len({int(source) for source in np.unique(arrays["source_candidate_id"][arrays["completed_lap"]])})
    done = arrays["done"].astype(bool)
    lap_times = arrays["lap_time_s"][np.isfinite(arrays["lap_time_s"])]
    return {
        "dataset": str(data.root),
        "total_transitions": int(arrays["obs"].shape[0]),
        "observation_dim": int(arrays["obs"].shape[1]),
        "source_candidate_count": len(source_counts),
        "terminal_reason_distribution": dict(terminal_counts),
        "done_count": int(done.sum()),
        "completed_lap_transition_count": int(arrays["completed_lap"].sum()),
        "completed_lap_source_count": completed_source_count,
        "fastest_lap_time_s": float(np.min(lap_times)) if lap_times.size else None,
        "progress": {
            "mean_m": float(np.mean(progress)),
            "median_m": float(np.median(progress)),
            "max_m": float(np.max(progress)),
        },
        "actions": {
            "mean": np.mean(action, axis=0).tolist(),
            "std": np.std(action, axis=0).tolist(),
            "min": np.min(action, axis=0).tolist(),
            "max": np.max(action, axis=0).tolist(),
            "simultaneous_throttle_brake_rate": float(np.mean((action[:, 0] > 0.05) & (action[:, 1] > 0.05))),
        },
        "braking_zone_action_checks": {
            "mean_brake_when_braking": float(np.mean(action[action[:, 1] > 0.05, 1])) if np.any(action[:, 1] > 0.05) else 0.0,
            "braking_fraction": float(np.mean(action[:, 1] > 0.05)),
        },
    }


def _parse_export(args: argparse.Namespace) -> int:
    manifest = export_dataset(
        run_dir=Path(args.run_dir),
        selected_telemetry=Path(args.selected_telemetry) if args.selected_telemetry else None,
        output_dir=Path(args.output_dir),
        observation_profile=args.observation_profile,
        physics_model=args.physics_model,
        max_candidates=args.max_candidates,
        max_per_generation=args.max_per_generation,
        balanced_buckets=args.balanced_buckets,
        source_json=Path(args.source_json) if args.source_json else None,
    )
    print(f"dataset_export_complete manifest={manifest}")
    return 0


def _parse_report(args: argparse.Namespace) -> int:
    report = build_report(Path(args.dataset))
    reports_dir = Path(args.dataset)
    if reports_dir.is_file():
        reports_dir = reports_dir.parent
    output = reports_dir / "reports" / "dataset_report.json"
    _write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and report ES transition datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="CPU-replay ES candidates into transition shards.")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--selected-telemetry")
    export.add_argument("--output-dir", default=str(DATASETS_DIR / "v1-es-policy-dataset"))
    export.add_argument("--observation-profile", default="racing_v2")
    export.add_argument("--physics-model", default="v1")
    export.add_argument("--max-candidates", type=int, default=64)
    export.add_argument("--max-per-generation", type=int, default=4)
    export.add_argument(
        "--source-json",
        help="Export exactly one candidate from a JSON row, such as best_so_far.json; relative paths resolve from the run dir.",
    )
    export.add_argument(
        "--balanced-buckets",
        action="store_true",
        help="Reserve candidate slots for valid, near-valid, frontier, and failure buckets before filling by score.",
    )
    export.set_defaults(func=_parse_export)

    report = subparsers.add_parser("report", help="Summarize a transition dataset.")
    report.add_argument("dataset")
    report.set_defaults(func=_parse_report)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
