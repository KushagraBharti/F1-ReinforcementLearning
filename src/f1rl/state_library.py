"""Build and load state libraries for successful-state curricula."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from f1rl.config import ARTIFACTS_DIR, SimConfig
from f1rl.scripted import ScriptedController
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import (
    StateSnapshot,
    snapshot_from_mapping,
    snapshot_from_sim,
    snapshot_from_telemetry_row,
    snapshot_to_dict,
)
from f1rl.telemetry import load_steps


def load_state_library(path: Path) -> list[StateSnapshot]:
    data = json.loads(path.read_text(encoding="utf-8"))
    snapshots = data["snapshots"] if isinstance(data, dict) else data
    return [snapshot_from_mapping(snapshot) for snapshot in snapshots]


def write_state_library(
    path: Path,
    snapshots: Iterable[StateSnapshot],
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    rows = [snapshot_to_dict(snapshot) for snapshot in snapshots]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "track": "monza",
        "snapshot_count": len(rows),
        "metadata": metadata or {},
        "snapshots": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _telemetry_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.jsonl") if item.is_file())
    raise FileNotFoundError(f"Telemetry path does not exist: {path}")


def snapshots_from_telemetry(
    path: Path,
    *,
    sample_every_m: float,
    sample_every_steps: int,
    max_snapshots: int,
) -> list[StateSnapshot]:
    snapshots: list[StateSnapshot] = []
    for telemetry_path in _telemetry_paths(path):
        rows = load_steps(telemetry_path)
        next_progress_m: float | None = None
        for index, row in enumerate(rows):
            progress_m = float(row.get("monotonic_progress_m", 0.0))
            if next_progress_m is None:
                next_progress_m = progress_m
            step_due = sample_every_steps > 0 and index % sample_every_steps == 0
            progress_due = sample_every_m > 0.0 and progress_m + 1e-9 >= next_progress_m
            if step_due or progress_due:
                snapshot = snapshot_from_telemetry_row(row, source_file=str(telemetry_path))
                snapshots.append(snapshot)
                if sample_every_m > 0.0:
                    next_progress_m = progress_m + sample_every_m
            if max_snapshots > 0 and len(snapshots) >= max_snapshots:
                return snapshots
    return snapshots


def snapshots_from_scripted(
    *,
    steps: int,
    seed: int,
    sample_every_m: float,
    sample_every_steps: int,
    max_snapshots: int,
) -> list[StateSnapshot]:
    sim = MonzaSim(SimConfig(max_steps=steps))
    sim.reset(seed=seed)
    controller = ScriptedController()
    snapshots: list[StateSnapshot] = [snapshot_from_sim(sim, source="scripted")]
    next_progress_m = sample_every_m
    for step_index in range(1, steps + 1):
        throttle, brake, steer = controller.controls(sim)
        result = sim.step_controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            action_id=-10,
            collect_observation=False,
            collect_rays=False,
        )
        progress_m = float(sim.state.monotonic_progress_m)
        step_due = sample_every_steps > 0 and step_index % sample_every_steps == 0
        progress_due = sample_every_m > 0.0 and progress_m + 1e-9 >= next_progress_m
        if step_due or progress_due or result.terminated or result.truncated:
            snapshots.append(snapshot_from_sim(sim, source="scripted"))
            if sample_every_m > 0.0:
                next_progress_m = progress_m + sample_every_m
        if max_snapshots > 0 and len(snapshots) >= max_snapshots:
            break
        if result.terminated or result.truncated:
            break
    return snapshots


def default_output_path(source: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"state-library-{source}-{timestamp}" / "state_library.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Monza simulator state libraries for curriculum starts.")
    parser.add_argument("--source", choices=["scripted", "telemetry"], default="scripted")
    parser.add_argument("--telemetry", type=Path, help="Telemetry JSONL file or directory for --source telemetry.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-every-m", type=float, default=100.0)
    parser.add_argument("--sample-every-steps", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or default_output_path(args.source)
    if args.source == "scripted":
        snapshots = snapshots_from_scripted(
            steps=args.steps,
            seed=args.seed,
            sample_every_m=args.sample_every_m,
            sample_every_steps=args.sample_every_steps,
            max_snapshots=args.max_snapshots,
        )
        metadata = {
            "steps": args.steps,
            "seed": args.seed,
            "sample_every_m": args.sample_every_m,
            "sample_every_steps": args.sample_every_steps,
        }
    else:
        if args.telemetry is None:
            raise ValueError("--telemetry is required when --source telemetry.")
        snapshots = snapshots_from_telemetry(
            args.telemetry,
            sample_every_m=args.sample_every_m,
            sample_every_steps=args.sample_every_steps,
            max_snapshots=args.max_snapshots,
        )
        metadata = {
            "telemetry": str(args.telemetry),
            "sample_every_m": args.sample_every_m,
            "sample_every_steps": args.sample_every_steps,
        }
    write_state_library(output, snapshots, source=args.source, metadata=metadata)
    print(f"state_library_complete path={output} snapshots={len(snapshots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
