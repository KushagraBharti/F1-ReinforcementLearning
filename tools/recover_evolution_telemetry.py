"""Regenerate all-candidate telemetry for an interrupted evolution run.

This is an artifact recovery utility. It rebuilds selected_telemetry from
attempts.jsonl and population_checkpoint.json, then swaps the recovered
directory into place only after all expected traces are written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from f1rl.evolution_search import (
    EvolutionGates,
    EvolutionSearchConfig,
    _evaluate_chunk,
    _load_snapshots,
    _sim_config_for_generation,
)
from f1rl.state_snapshot import snapshot_to_dict

T = TypeVar("T")


def _tuple_config(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _config_from_checkpoint(payload: dict[str, Any]) -> EvolutionSearchConfig:
    config = dict(payload["config"])
    for key in ("scoring_profiles", "survival_floor_stages_m"):
        if key in config:
            config[key] = _tuple_config(config[key])
    if "max_steps_schedule" in config:
        config["max_steps_schedule"] = tuple((int(row[0]), int(row[1])) for row in config["max_steps_schedule"])
    return EvolutionSearchConfig(**config)


def _chunks(items: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _load_attempts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    final_row = dict(row.get("final_row") or {})
    return {
        "rank": int(row["generation"]) * 100_000 + int(row["candidate_index"]),
        "selection_reason": "all_candidates_recovered",
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "seed": int(row["seed"]),
        "score": row["score"],
        "profile_scores": row.get("profile_scores", {}),
        "best_progress_m": row["best_progress_m"],
        "final_progress_m": final_row.get("monotonic_progress_m", row.get("final_progress_m")),
        "final_speed_kph": final_row.get("speed_kph", row.get("final_speed_kph")),
        "final_lateral_error_m": final_row.get("lateral_error_m", row.get("final_lateral_error_m")),
        "final_heading_error_deg": final_row.get("heading_error_deg", row.get("final_heading_error_deg")),
        "termination_reason": final_row.get("termination_reason", row.get("termination_reason")),
        "path": row["selected_telemetry"],
    }


def recover_telemetry(
    run_dir: Path,
    *,
    workers: int,
    chunk_size: int,
    replace: bool,
) -> Path:
    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / "population_checkpoint.json"
    attempts_path = run_dir / "attempts.jsonl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempts: {attempts_path}")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    config = _config_from_checkpoint(checkpoint)
    gates = EvolutionGates(**dict(checkpoint["gates"]))
    state_library = checkpoint.get("state_library")
    snapshots = _load_snapshots(
        sim_config=_sim_config_for_generation(config, 0),
        config=config,
        state_library=Path(state_library) if state_library else None,
        start_progress_m=checkpoint.get("start_progress_m"),
        start_speed_kph=float(checkpoint.get("start_speed_kph", 0.0) or 0.0),
        start_min_progress_m=checkpoint.get("start_min_progress_m"),
        start_max_progress_m=checkpoint.get("start_max_progress_m"),
    )
    attempts = _load_attempts(attempts_path)
    expected_count = int(checkpoint.get("attempt_count", len(attempts)))
    if len(attempts) != expected_count:
        raise ValueError(f"attempt count mismatch: attempts.jsonl has {len(attempts)}, checkpoint has {expected_count}")

    recovered_dir = run_dir / "selected_telemetry_recovered"
    if recovered_dir.exists():
        shutil.rmtree(recovered_dir)
    recovered_dir.mkdir(parents=True)

    tasks_by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        generation = int(row["generation"])
        candidate_index = int(row["candidate_index"])
        snapshot_index = int(row["snapshot_index"])
        telemetry_path = recovered_dir / (
            f"evolution-all_candidates-gen-{generation:03d}-candidate-{candidate_index:05d}-steps.jsonl"
        )
        row["selected_telemetry"] = str(run_dir / "selected_telemetry" / telemetry_path.name)
        tasks_by_generation[generation].append(
            {
                "candidate_index": candidate_index,
                "generation": generation,
                "seed": int(row["seed"]),
                "genome": row["genome"],
                "lineage": row.get("lineage", {}),
                "snapshot_index": snapshot_index,
                "snapshot": snapshot_to_dict(snapshots[snapshot_index]),
                "sim_config": _sim_config_for_generation(config, generation),
                "gates": gates,
                "scoring_profiles": tuple(config.scoring_profiles),
                "frontier_focus_start_m": config.frontier_focus_start_m,
                "frontier_focus_end_m": config.frontier_focus_end_m,
                "capture_step_telemetry": True,
                "stream_telemetry_path": str(telemetry_path),
            }
        )

    started = time.perf_counter()
    regenerated_rows: list[dict[str, Any]] = []
    resolved_workers = max(1, workers if workers > 0 else (os.cpu_count() or 2) - 1)
    resolved_chunk_size = max(1, chunk_size)
    print(
        "recover_start "
        f"run={run_dir} attempts={len(attempts)} workers={resolved_workers} chunk_size={resolved_chunk_size}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        for generation in sorted(tasks_by_generation):
            generation_tasks = tasks_by_generation[generation]
            generation_started = time.perf_counter()
            chunked = list(_chunks(generation_tasks, resolved_chunk_size))
            for chunk_rows in executor.map(_evaluate_chunk, chunked):
                regenerated_rows.extend(chunk_rows)
            elapsed = time.perf_counter() - generation_started
            print(
                "recover_generation "
                f"generation={generation} traces={len(generation_tasks)} elapsed_s={elapsed:.2f} "
                f"traces_per_second={len(generation_tasks) / max(elapsed, 1e-9):.3f}",
                flush=True,
            )

    trace_files = sorted(recovered_dir.glob("*.jsonl"))
    if len(trace_files) != len(attempts):
        raise RuntimeError(f"recovered trace count mismatch: expected {len(attempts)}, found {len(trace_files)}")

    manifest_rows = sorted((_manifest_row(row) for row in regenerated_rows), key=lambda row: (row["generation"], row["candidate_index"]))
    manifest_path = recovered_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "telemetry_selection": "all",
                "trace_count": len(manifest_rows),
                "recovered_from": str(run_dir),
                "traces": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if replace:
        selected_dir = run_dir / "selected_telemetry"
        backup_dir = run_dir / "selected_telemetry_partial_before_recovery"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if selected_dir.exists():
            selected_dir.replace(backup_dir)
        recovered_dir.replace(selected_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        final_dir = selected_dir
    else:
        final_dir = recovered_dir

    elapsed = time.perf_counter() - started
    print(
        "recover_complete "
        f"output={final_dir} traces={len(trace_files)} elapsed_s={elapsed:.2f}",
        flush=True,
    )
    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover selected_telemetry for an interrupted evolution run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recover_telemetry(
        args.run_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        replace=bool(args.replace),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
