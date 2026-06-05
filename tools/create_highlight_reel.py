"""Build curated full-generation replay highlights from ES runs.

The CPU source run already saved all candidate telemetry to cold storage, so the
CPU highlight is copied. The GPU source run saved compact attempt rows, so this
script CPU-replays the selected generations and writes replayable telemetry.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1rl.evolution_postcheck import _config_from_mapping, _gates_from_mapping
from f1rl.evolution_search import (
    _load_snapshots,
    _run_candidate,
    _score_rows,
    _sim_config_for_generation,
    genome_from_mapping,
)
from f1rl.state_snapshot import snapshot_from_mapping, snapshot_to_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "highlights" / "full-generation-reel-20260605"

CPU_RUN_NAME = "evolution-speed-150x60-20260604-0501"
CPU_RUN_DIR = REPO_ROOT / "artifacts" / "runs" / CPU_RUN_NAME
CPU_MANIFEST = CPU_RUN_DIR / "selected_telemetry" / "manifest.json"
CPU_GENERATIONS = (0, 5, 10, 29, 49, 59)

GPU_RUN_NAME = "gpu-speed-speedprofiles-2000x150-25k-20260605"
GPU_RUN_DIR = REPO_ROOT / "artifacts" / "runs" / GPU_RUN_NAME
GPU_SUMMARY = GPU_RUN_DIR / "evolution_summary.json"
GPU_ATTEMPTS = GPU_RUN_DIR / "attempts.jsonl"
GPU_GENERATIONS = (0, 14, 20, 61, 72, 143)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_fast_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".jsonl.gz"):
        tmp_path = path.with_name(f"{path.name.removesuffix('.jsonl.gz')}.tmp.jsonl.gz")
        with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=1) as file:
            for row in rows:
                file.write(json.dumps(row, default=_json_default) + "\n")
        tmp_path.replace(path)
        return
    if path.name.endswith(".jsonl"):
        tmp_path = path.with_name(f"{path.name.removesuffix('.jsonl')}.tmp.jsonl")
        with tmp_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, default=_json_default) + "\n")
        tmp_path.replace(path)
        return
    raise ValueError(f"unsupported telemetry path suffix: {path}")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _resolve_manifest_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _parse_generations(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not value.strip():
        return default
    generations = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not generations:
        raise ValueError("generation list cannot be empty")
    return generations


def _manifest_payload(
    *,
    kind: str,
    source_run: str,
    source_dir: Path,
    generations: tuple[int, ...],
    traces: list[dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    counts = Counter(int(row["generation"]) for row in traces)
    return {
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run": source_run,
        "source_dir": str(source_dir.resolve()),
        "generation_count": len(generations),
        "generations": list(generations),
        "trace_count": len(traces),
        "trace_count_by_generation": {str(gen): counts.get(gen, 0) for gen in generations},
        "notes": notes,
        "traces": traces,
    }


def _write_generation_manifests(
    *,
    run_dir: Path,
    kind: str,
    source_run: str,
    source_dir: Path,
    generations: tuple[int, ...],
    traces: list[dict[str, Any]],
    notes: str,
) -> None:
    traces_by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        traces_by_generation[int(row["generation"])].append(row)

    for generation in generations:
        generation_dir = run_dir / f"gen-{generation:03d}"
        generation_traces = sorted(
            traces_by_generation.get(generation, ()),
            key=lambda row: int(row.get("candidate_index", 0)),
        )
        _write_json(
            generation_dir / "manifest.json",
            _manifest_payload(
                kind=f"{kind}_generation",
                source_run=source_run,
                source_dir=source_dir,
                generations=(generation,),
                traces=generation_traces,
                notes=notes,
            ),
        )


def _copy_cpu_generation_reel(
    *,
    output_dir: Path,
    generations: tuple[int, ...],
    force: bool,
) -> list[dict[str, Any]]:
    manifest = _load_json(CPU_MANIFEST)
    wanted = set(generations)
    source_rows = [
        dict(row)
        for row in manifest.get("traces", [])
        if int(row.get("generation", -1)) in wanted
    ]
    source_rows.sort(key=lambda row: (int(row["generation"]), int(row["candidate_index"])))

    expected = len(generations) * 150
    if len(source_rows) != expected:
        raise RuntimeError(f"expected {expected} CPU traces, found {len(source_rows)}")

    run_dir = output_dir / "cpu-es-150x60"
    traces: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        source_path = Path(str(row["path"]))
        if not source_path.exists():
            raise FileNotFoundError(f"CPU telemetry not found: {source_path}")
        generation = int(row["generation"])
        destination = run_dir / f"gen-{generation:03d}" / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if force or not destination.exists() or destination.stat().st_size != source_path.stat().st_size:
            shutil.copy2(source_path, destination)

        manifest_row = dict(row)
        manifest_row.update(
            {
                "path": _repo_relative(destination),
                "source_path": str(source_path.resolve()),
                "source_run": CPU_RUN_NAME,
                "highlight_run": "cpu-es-150x60",
                "highlight_kind": "full_generation",
            }
        )
        traces.append(manifest_row)
        if index % 150 == 0:
            print(f"cpu copied generation {generation:03d} ({index}/{len(source_rows)})", flush=True)

    _write_generation_manifests(
        run_dir=run_dir,
        kind="f1rl_cpu_es_full_generation_highlight_reel",
        source_run=CPU_RUN_NAME,
        source_dir=CPU_RUN_DIR,
        generations=generations,
        traces=traces,
        notes="Copied full selected CPU ES generations from cold all-candidate telemetry.",
    )
    _write_json(
        run_dir / "manifest.json",
        _manifest_payload(
            kind="f1rl_cpu_es_full_generation_highlight_reel",
            source_run=CPU_RUN_NAME,
            source_dir=CPU_RUN_DIR,
            generations=generations,
            traces=traces,
            notes="Copied full selected CPU ES generations from cold all-candidate telemetry.",
        ),
    )
    return traces


def _collect_gpu_attempt_rows(
    *,
    generations: tuple[int, ...],
    limit_per_generation: int | None,
) -> list[dict[str, Any]]:
    wanted = set(generations)
    per_generation_counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    expected_per_generation = limit_per_generation if limit_per_generation is not None else 2000

    with GPU_ATTEMPTS.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            generation = int(row.get("generation", -1))
            if generation not in wanted:
                continue
            if limit_per_generation is not None and per_generation_counts[generation] >= limit_per_generation:
                continue
            rows.append(row)
            per_generation_counts[generation] += 1
            if all(per_generation_counts.get(item, 0) >= expected_per_generation for item in wanted):
                break

    missing = {
        generation: per_generation_counts.get(generation, 0)
        for generation in generations
        if per_generation_counts.get(generation, 0) != expected_per_generation
    }
    if missing:
        raise RuntimeError(f"unexpected GPU attempt counts: {missing}")
    rows.sort(key=lambda row: (int(row["generation"]), int(row["candidate_index"])))
    return rows


def _snapshot_payloads_for_gpu_rows(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    config = _config_from_mapping(summary["config"])
    state_library_value = summary.get("state_library")
    state_library = Path(str(state_library_value)) if state_library_value else None
    if state_library is not None and not state_library.is_absolute():
        state_library = (GPU_RUN_DIR / state_library).resolve()

    rows_by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_generation[int(row["generation"])].append(row)

    snapshots_by_generation: dict[int, list[dict[str, Any]]] = {}
    for generation, generation_rows in rows_by_generation.items():
        first = generation_rows[0]
        start_progress_m = first.get("start_progress_m")
        start_speed_kph = first.get("start_speed_kph")
        if start_speed_kph is None:
            start_speed_kph = summary.get("config", {}).get("start_speed_kph")
        if start_speed_kph is None:
            raise RuntimeError(f"could not infer start_speed_kph for GPU generation {generation}")

        sim_config = _sim_config_for_generation(config, generation)
        snapshots = _load_snapshots(
            sim_config=sim_config,
            config=config,
            state_library=state_library,
            start_progress_m=float(start_progress_m) if start_progress_m is not None else None,
            start_speed_kph=float(start_speed_kph),
            start_min_progress_m=summary.get("config", {}).get("start_min_progress_m"),
            start_max_progress_m=summary.get("config", {}).get("start_max_progress_m"),
        )
        snapshots_by_generation[generation] = [snapshot_to_dict(snapshot) for snapshot in snapshots]
    return snapshots_by_generation


def _gpu_worker(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload["row"]
    output_path = Path(payload["output_path"])
    config = _config_from_mapping(payload["config"])
    gates = _gates_from_mapping(payload["gates"])
    generation = int(row["generation"])
    candidate_index = int(row["candidate_index"])
    snapshots = [snapshot_from_mapping(item) for item in payload["snapshots"]]
    snapshot_index = int(row.get("snapshot_index", 0))
    if snapshot_index < 0 or snapshot_index >= len(snapshots):
        raise RuntimeError(
            f"snapshot_index {snapshot_index} out of range for gen {generation} candidate {candidate_index}"
        )

    sim_config = _sim_config_for_generation(config, generation)
    rows, _sim = _run_candidate(
        sim_config=sim_config,
        snapshot=snapshots[snapshot_index],
        genome=genome_from_mapping(row["genome"]),
        gates=gates,
        seed=int(row["seed"]),
        collect_full_telemetry=False,
        keep_step_telemetry=True,
    )
    if not rows:
        raise RuntimeError(f"empty CPU replay for gen {generation} candidate {candidate_index}")
    _write_fast_jsonl_gz(output_path, rows)

    profile_scores = _score_rows(
        rows,
        start_progress_m=float(row.get("start_progress_m", 0.0) or 0.0),
        gates=gates,
        scoring_profiles=tuple(config.scoring_profiles),
        frontier_focus_start_m=float(config.frontier_focus_start_m),
        frontier_focus_end_m=float(config.frontier_focus_end_m),
    )
    primary_profile = str(row.get("primary_scoring_profile") or (config.scoring_profiles[0] if config.scoring_profiles else "max_progress"))
    final = rows[-1]
    best_progress_m = max(float(item.get("monotonic_progress_m", 0.0)) for item in rows)
    score = float(profile_scores.get(primary_profile, next(iter(profile_scores.values()), 0.0)))

    manifest_row = {
        "generation": generation,
        "candidate_index": candidate_index,
        "seed": int(row["seed"]),
        "score": score,
        "profile_scores": profile_scores,
        "primary_scoring_profile": primary_profile,
        "best_progress_m": best_progress_m,
        "final_progress_m": float(final.get("monotonic_progress_m", 0.0)),
        "final_speed_kph": float(final.get("speed_kph", 0.0)),
        "final_lateral_error_m": float(final.get("lateral_error_m", 0.0)),
        "final_heading_error_deg": float(final.get("heading_error_deg", 0.0)),
        "termination_reason": str(final.get("termination_reason", "")),
        "completed_lap": bool(final.get("completed_lap", False)),
        "valid_lap": bool(final.get("valid_lap", False)),
        "finish_crossed": bool(final.get("finish_crossed", False)),
        "steps": len(rows),
        "elapsed_s": float(final.get("sim_time_s", 0.0)),
        "path": _repo_relative(output_path),
        "source_run": GPU_RUN_NAME,
        "source_backend": row.get("backend", "gpu"),
        "source_gpu_score": row.get("gpu_score", row.get("score")),
        "source_gpu_profile_scores": row.get("profile_scores", {}),
        "source_gpu_best_progress_m": row.get("best_progress_m"),
        "source_gpu_termination_reason": row.get("termination_reason"),
        "snapshot_index": snapshot_index,
        "start_progress_m": row.get("start_progress_m"),
        "start_speed_kph": row.get("start_speed_kph"),
        "highlight_run": "gpu-es-2000x150",
        "highlight_kind": "full_generation_cpu_replay",
    }
    return manifest_row


def _gpu_existing_manifest_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = dict(payload["row"])
    output_path = Path(payload["output_path"])
    profile_scores = dict(row.get("profile_scores", {}))
    primary_profile = str(row.get("primary_scoring_profile") or next(iter(profile_scores), "max_progress"))
    return {
        "generation": int(row["generation"]),
        "candidate_index": int(row["candidate_index"]),
        "seed": int(row["seed"]),
        "score": float(row.get("score", 0.0)),
        "profile_scores": profile_scores,
        "primary_scoring_profile": primary_profile,
        "best_progress_m": float(row.get("best_progress_m", 0.0)),
        "final_progress_m": float(row.get("final_progress_m", 0.0)),
        "final_speed_kph": float(row.get("final_speed_kph", 0.0)),
        "final_lateral_error_m": float(row.get("final_lateral_error_m", 0.0)),
        "final_heading_error_deg": float(row.get("final_heading_error_deg", 0.0)),
        "termination_reason": str(row.get("termination_reason", "")),
        "completed_lap": bool(row.get("completed_lap", False)),
        "valid_lap": bool(row.get("valid_lap", False)),
        "finish_crossed": bool(row.get("finish_crossed", False)),
        "steps": int(row.get("steps", 0)),
        "elapsed_s": float(row.get("elapsed_s", 0.0)),
        "path": _repo_relative(output_path),
        "source_run": GPU_RUN_NAME,
        "source_backend": row.get("backend", "gpu"),
        "source_gpu_score": row.get("gpu_score", row.get("score")),
        "source_gpu_profile_scores": profile_scores,
        "source_gpu_best_progress_m": row.get("best_progress_m"),
        "source_gpu_termination_reason": row.get("termination_reason"),
        "snapshot_index": int(row.get("snapshot_index", 0)),
        "start_progress_m": row.get("start_progress_m"),
        "start_speed_kph": row.get("start_speed_kph"),
        "highlight_run": "gpu-es-2000x150",
        "highlight_kind": "full_generation_cpu_replay",
        "resumed_from_existing_file": True,
    }


def _load_existing_trace_rows(run_dir: Path) -> dict[Path, dict[str, Any]]:
    rows_by_path: dict[Path, dict[str, Any]] = {}
    manifest_paths = [run_dir / "manifest.json", *sorted(run_dir.glob("gen-*/manifest.json"))]
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        payload = _load_json(manifest_path)
        for row in payload.get("traces", []):
            row_path = row.get("path")
            if not row_path:
                continue
            rows_by_path[_resolve_manifest_path(str(row_path))] = dict(row)
    return rows_by_path


def _build_gpu_generation_reel(
    *,
    output_dir: Path,
    generations: tuple[int, ...],
    workers: int,
    force: bool,
    limit_per_generation: int | None,
) -> list[dict[str, Any]]:
    summary = _load_json(GPU_SUMMARY)
    source_rows = _collect_gpu_attempt_rows(
        generations=generations,
        limit_per_generation=limit_per_generation,
    )
    snapshot_payloads = _snapshot_payloads_for_gpu_rows(summary=summary, rows=source_rows)

    run_dir = output_dir / "gpu-es-2000x150"
    existing_rows_by_path = _load_existing_trace_rows(run_dir)
    tasks: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for row in source_rows:
        generation = int(row["generation"])
        candidate_index = int(row["candidate_index"])
        output_path = (
            run_dir
            / f"gen-{generation:03d}"
            / f"gpu-es-2000x150-gen-{generation:03d}-candidate-{candidate_index:05d}-steps.jsonl.gz"
        )
        task = {
            "row": row,
            "output_path": str(output_path),
            "config": summary["config"],
            "gates": summary["gates"],
            "snapshots": snapshot_payloads[generation],
        }
        if not force and output_path.exists() and output_path.stat().st_size > 0:
            traces.append(existing_rows_by_path.get(output_path.resolve(), _gpu_existing_manifest_row(task)))
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(task)

    if tasks:
        print(
            f"gpu replaying {len(tasks)} candidates with {workers} worker(s); "
            f"{len(traces)} existing files reused",
            flush=True,
        )
        completed = 0
        started_at = time.monotonic()
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_gpu_worker, task) for task in tasks]
            for future in as_completed(futures):
                traces.append(future.result())
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    rate = completed / elapsed
                    print(
                        f"gpu replayed {completed}/{len(tasks)} "
                        f"({rate:.2f} candidates/s)",
                        flush=True,
                    )
    else:
        print(f"gpu reused {len(traces)} existing replay files", flush=True)

    traces.sort(key=lambda row: (int(row["generation"]), int(row["candidate_index"])))
    notes = "CPU-replayed full selected GPU ES generations from compact GPU attempt rows."
    if limit_per_generation is not None:
        notes += f" Debug build capped at {limit_per_generation} trace(s) per generation."
    _write_generation_manifests(
        run_dir=run_dir,
        kind="f1rl_gpu_es_full_generation_highlight_reel",
        source_run=GPU_RUN_NAME,
        source_dir=GPU_RUN_DIR,
        generations=generations,
        traces=traces,
        notes=notes,
    )
    _write_json(
        run_dir / "manifest.json",
        _manifest_payload(
            kind="f1rl_gpu_es_full_generation_highlight_reel",
            source_run=GPU_RUN_NAME,
            source_dir=GPU_RUN_DIR,
            generations=generations,
            traces=traces,
            notes=notes,
        ),
    )
    return traces


def _write_reel_readme(
    *,
    output_dir: Path,
    cpu_generations: tuple[int, ...],
    gpu_generations: tuple[int, ...],
    cpu_count: int,
    gpu_count: int,
) -> None:
    cpu_generation_limit = max(1, cpu_count // max(1, len(cpu_generations)))
    gpu_generation_limit = max(1, gpu_count // max(1, len(gpu_generations)))
    readme = f"""# Full-Generation Highlight Reel

Created: {datetime.now(UTC).isoformat()}

This reel preserves curated full generations from the main CPU ES and GPU ES runs.

## Contents

- `cpu-es-150x60`: generations {", ".join(str(item) for item in cpu_generations)} ({cpu_count} traces)
- `gpu-es-2000x150`: generations {", ".join(str(item) for item in gpu_generations)} ({gpu_count} traces)

Each run folder has a replay manifest, and each `gen-XXX` folder has its own manifest.

## Replay

```powershell
uv run --no-sync python -m f1rl.replay artifacts\\highlights\\full-generation-reel-20260605\\cpu-es-150x60 --by-generation --generation-limit {cpu_generation_limit}
uv run --no-sync python -m f1rl.replay artifacts\\highlights\\full-generation-reel-20260605\\gpu-es-2000x150 --by-generation --generation-limit {gpu_generation_limit}
```

For smoother preview, reduce `--generation-limit`:

```powershell
uv run --no-sync python -m f1rl.replay artifacts\\highlights\\full-generation-reel-20260605\\gpu-es-2000x150 --by-generation --generation-limit 20 --sort-by score
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def build_highlight_reel(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    cpu_generations = _parse_generations(args.cpu_generations, CPU_GENERATIONS)
    gpu_generations = _parse_generations(args.gpu_generations, GPU_GENERATIONS)

    cpu_traces: list[dict[str, Any]] = []
    gpu_traces: list[dict[str, Any]] = []

    if args.only in {"all", "cpu"}:
        cpu_traces = _copy_cpu_generation_reel(
            output_dir=output_dir,
            generations=cpu_generations,
            force=bool(args.force),
        )
    elif (output_dir / "cpu-es-150x60" / "manifest.json").exists():
        cpu_traces = _load_json(output_dir / "cpu-es-150x60" / "manifest.json").get("traces", [])

    if args.only in {"all", "gpu"}:
        gpu_traces = _build_gpu_generation_reel(
            output_dir=output_dir,
            generations=gpu_generations,
            workers=int(args.workers),
            force=bool(args.force),
            limit_per_generation=args.limit_per_generation,
        )
    elif (output_dir / "gpu-es-2000x150" / "manifest.json").exists():
        gpu_traces = _load_json(output_dir / "gpu-es-2000x150" / "manifest.json").get("traces", [])

    _write_json(
        output_dir / "manifest.json",
        {
            "kind": "f1rl_full_generation_highlight_reel",
            "created_at": datetime.now(UTC).isoformat(),
            "output_dir": str(output_dir),
            "cpu": {
                "source_run": CPU_RUN_NAME,
                "path": _repo_relative(output_dir / "cpu-es-150x60"),
                "generations": list(cpu_generations),
                "trace_count": len(cpu_traces),
            },
            "gpu": {
                "source_run": GPU_RUN_NAME,
                "path": _repo_relative(output_dir / "gpu-es-2000x150"),
                "generations": list(gpu_generations),
                "trace_count": len(gpu_traces),
            },
            "replay_note": "Replay the cpu-es-150x60 or gpu-es-2000x150 subdirectories; each has a manifest.json.",
        },
    )
    _write_reel_readme(
        output_dir=output_dir,
        cpu_generations=cpu_generations,
        gpu_generations=gpu_generations,
        cpu_count=len(cpu_traces),
        gpu_count=len(gpu_traces),
    )
    print(f"wrote highlight reel manifest: {output_dir / 'manifest.json'}", flush=True)
    print(f"cpu traces: {len(cpu_traces)}", flush=True)
    print(f"gpu traces: {len(gpu_traces)}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only", choices=("all", "cpu", "gpu"), default="all")
    parser.add_argument("--cpu-generations", default=None)
    parser.add_argument("--gpu-generations", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-generation", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.limit_per_generation is not None and args.limit_per_generation <= 0:
        parser.error("--limit-per-generation must be positive")
    build_highlight_reel(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
