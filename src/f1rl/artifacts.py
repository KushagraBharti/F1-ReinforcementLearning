"""Helpers for artifact directories and checkpoint discovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from f1rl.constants import ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, RENDERS_DIR


@dataclass(slots=True)
class RunPaths:
    run_id: str
    root: Path
    checkpoints: Path
    metrics: Path
    logs: Path
    renders: Path


def ensure_artifact_dirs() -> None:
    for directory in (ARTIFACTS_DIR, CHECKPOINTS_DIR, METRICS_DIR, LOGS_DIR, RENDERS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def new_run_paths(prefix: str) -> RunPaths:
    ensure_artifact_dirs()
    run_id = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root = ARTIFACTS_DIR / run_id
    checkpoints = root / "checkpoints"
    metrics = root / "metrics"
    logs = root / "logs"
    renders = root / "renders"
    for directory in (root, checkpoints, metrics, logs, renders):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, root=root, checkpoints=checkpoints, metrics=metrics, logs=logs, renders=renders)


def checkpoint_dirs(base_dir: Path | None = None) -> list[Path]:
    base = base_dir or ARTIFACTS_DIR
    if not base.exists():
        return []
    candidates: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_dir():
            continue
        if path.name.startswith("checkpoint_"):
            candidates.append(path)
            continue
        if (path / "rllib_checkpoint.json").exists() or (path / "algorithm_state.pkl").exists():
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.stat().st_mtime)


def latest_checkpoint(base_dir: Path | None = None) -> Path | None:
    candidates = checkpoint_dirs(base_dir)
    return candidates[-1] if candidates else None


def resolve_checkpoint(checkpoint: str, base_dir: Path | None = None) -> Path:
    if checkpoint == "latest":
        latest = latest_checkpoint(base_dir)
        if latest is None:
            raise FileNotFoundError("No checkpoint directories found under artifacts.")
        return latest
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    return path
