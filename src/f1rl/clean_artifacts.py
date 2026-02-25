"""Artifact cleanup command with retention policy."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from f1rl.constants import ARTIFACTS_DIR, RENDERS_DIR

RUN_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<stamp>\d{8}-\d{6})$")
PROTECTED_MARKERS = (".pin", "KEEP", ".keep")


@dataclass(slots=True, frozen=True)
class RunRecord:
    path: Path
    prefix: str
    timestamp: datetime
    protected: bool


@dataclass(slots=True)
class RetentionPolicy:
    keep_runs_per_prefix: int = 3
    keep_days: int = 14
    keep_render_files: int = 400
    include_prefixes: tuple[str, ...] = ()


@dataclass(slots=True)
class CleanupSummary:
    dry_run: bool
    run_dirs_seen: int
    run_dirs_kept: int
    run_dirs_removed: int
    render_files_seen: int
    render_files_kept: int
    render_files_removed: int
    removed_paths: list[str]


def _is_protected(path: Path) -> bool:
    return any((path / marker).exists() for marker in PROTECTED_MARKERS)


def _parse_run_record(path: Path) -> RunRecord | None:
    match = RUN_PATTERN.match(path.name)
    if match is None:
        return None
    stamp = match.group("stamp")
    try:
        ts = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return RunRecord(
        path=path,
        prefix=match.group("prefix"),
        timestamp=ts,
        protected=_is_protected(path),
    )


def discover_run_records(artifacts_dir: Path) -> list[RunRecord]:
    if not artifacts_dir.exists():
        return []
    records: list[RunRecord] = []
    for child in artifacts_dir.iterdir():
        if not child.is_dir():
            continue
        record = _parse_run_record(child)
        if record is not None:
            records.append(record)
    return records


def _group_by_prefix(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.prefix, []).append(record)
    for prefix in grouped:
        grouped[prefix].sort(key=lambda item: item.timestamp, reverse=True)
    return grouped


def plan_run_cleanup(
    records: list[RunRecord],
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> tuple[list[RunRecord], list[RunRecord]]:
    now = now or datetime.now(tz=UTC)
    grouped = _group_by_prefix(records)
    keep_set: set[Path] = set()

    for prefix, runs in grouped.items():
        if policy.include_prefixes and prefix not in policy.include_prefixes:
            keep_set.update(run.path for run in runs)
            continue

        protected = [run for run in runs if run.protected]
        keep_set.update(run.path for run in protected)

        by_count = runs[: max(0, policy.keep_runs_per_prefix)]
        keep_set.update(run.path for run in by_count)

        max_age = timedelta(days=max(0, policy.keep_days))
        by_age = [run for run in runs if (now - run.timestamp) <= max_age]
        keep_set.update(run.path for run in by_age)

    keep = [record for record in records if record.path in keep_set]
    remove = [record for record in records if record.path not in keep_set]
    return keep, remove


def _prune_renders(
    renders_dir: Path,
    policy: RetentionPolicy,
    dry_run: bool,
    now: datetime | None = None,
) -> tuple[list[Path], list[Path]]:
    now = now or datetime.now(tz=UTC)
    if not renders_dir.exists():
        return [], []

    image_files = [p for p in renders_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    image_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    keep: set[Path] = set(image_files[: max(0, policy.keep_render_files)])

    max_age = timedelta(days=max(0, policy.keep_days))
    for file in image_files:
        modified = datetime.fromtimestamp(file.stat().st_mtime, tz=UTC)
        if now - modified <= max_age:
            keep.add(file)

    kept = [path for path in image_files if path in keep]
    remove = [path for path in image_files if path not in keep]
    if not dry_run:
        for path in remove:
            path.unlink(missing_ok=True)
    return kept, remove


def clean_artifacts(
    artifacts_dir: Path,
    policy: RetentionPolicy,
    dry_run: bool = True,
    now: datetime | None = None,
) -> CleanupSummary:
    records = discover_run_records(artifacts_dir)
    keep, remove = plan_run_cleanup(records, policy=policy, now=now)

    removed_paths: list[str] = []
    if not dry_run:
        for record in remove:
            shutil.rmtree(record.path, ignore_errors=True)
            removed_paths.append(str(record.path))

    render_keep, render_remove = _prune_renders(
        renders_dir=artifacts_dir / RENDERS_DIR.name,
        policy=policy,
        dry_run=dry_run,
        now=now,
    )
    if not dry_run:
        removed_paths.extend(str(path) for path in render_remove)

    return CleanupSummary(
        dry_run=dry_run,
        run_dirs_seen=len(records),
        run_dirs_kept=len(keep),
        run_dirs_removed=len(remove),
        render_files_seen=len(render_keep) + len(render_remove),
        render_files_kept=len(render_keep),
        render_files_removed=len(render_remove),
        removed_paths=removed_paths,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean old artifacts using retention policy.")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), help="Artifacts directory root.")
    parser.add_argument("--keep-runs-per-prefix", type=int, default=3, help="Runs to keep per run-prefix.")
    parser.add_argument("--keep-days", type=int, default=14, help="Always keep runs/files newer than this age.")
    parser.add_argument("--keep-render-files", type=int, default=400, help="Max recent render files to keep.")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional run prefix filter (repeatable). If omitted, all prefixes are managed.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply deletions. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    return parser.parse_args(argv)


def _summary_dict(summary: CleanupSummary) -> dict[str, object]:
    return {
        "dry_run": summary.dry_run,
        "run_dirs_seen": summary.run_dirs_seen,
        "run_dirs_kept": summary.run_dirs_kept,
        "run_dirs_removed": summary.run_dirs_removed,
        "render_files_seen": summary.render_files_seen,
        "render_files_kept": summary.render_files_kept,
        "render_files_removed": summary.render_files_removed,
        "removed_paths": summary.removed_paths,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy = RetentionPolicy(
        keep_runs_per_prefix=args.keep_runs_per_prefix,
        keep_days=args.keep_days,
        keep_render_files=args.keep_render_files,
        include_prefixes=tuple(args.prefix),
    )
    summary = clean_artifacts(
        artifacts_dir=Path(args.artifacts_dir),
        policy=policy,
        dry_run=not args.apply,
    )
    payload = _summary_dict(summary)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            "clean_artifacts_complete "
            f"dry_run={summary.dry_run} "
            f"run_seen={summary.run_dirs_seen} run_kept={summary.run_dirs_kept} run_removed={summary.run_dirs_removed} "
            f"renders_seen={summary.render_files_seen} renders_kept={summary.render_files_kept} "
            f"renders_removed={summary.render_files_removed}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
