from datetime import UTC, datetime
from pathlib import Path

from f1rl.clean_artifacts import RetentionPolicy, discover_run_records, plan_run_cleanup


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_retention_keeps_recent_and_latest_per_prefix(tmp_path: Path) -> None:
    artifacts = _mkdir(tmp_path / "artifacts")
    _mkdir(artifacts / "train-smoke-20260101-000000")
    _mkdir(artifacts / "train-smoke-20260102-000000")
    _mkdir(artifacts / "train-smoke-20260103-000000")
    _mkdir(artifacts / "eval-20260103-010000")

    records = discover_run_records(artifacts)
    policy = RetentionPolicy(keep_runs_per_prefix=1, keep_days=0, keep_render_files=10)
    keep, remove = plan_run_cleanup(records, policy=policy, now=datetime(2026, 1, 10, tzinfo=UTC))

    keep_names = {record.path.name for record in keep}
    remove_names = {record.path.name for record in remove}
    assert "train-smoke-20260103-000000" in keep_names
    assert "eval-20260103-010000" in keep_names
    assert "train-smoke-20260101-000000" in remove_names
    assert "train-smoke-20260102-000000" in remove_names


def test_protected_run_is_never_removed(tmp_path: Path) -> None:
    artifacts = _mkdir(tmp_path / "artifacts")
    old_run = _mkdir(artifacts / "train-smoke-20250101-000000")
    (old_run / ".pin").write_text("keep", encoding="utf-8")
    _mkdir(artifacts / "train-smoke-20250102-000000")

    records = discover_run_records(artifacts)
    policy = RetentionPolicy(keep_runs_per_prefix=0, keep_days=0, keep_render_files=10)
    keep, remove = plan_run_cleanup(records, policy=policy, now=datetime(2026, 1, 10, tzinfo=UTC))

    keep_names = {record.path.name for record in keep}
    remove_names = {record.path.name for record in remove}
    assert "train-smoke-20250101-000000" in keep_names
    assert "train-smoke-20250102-000000" in remove_names
