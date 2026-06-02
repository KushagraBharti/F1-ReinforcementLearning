import importlib.util
import json
import os
from pathlib import Path

import pytest

from f1rl import policy_io, train


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_smoke_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    checkpoint = train.run_training(
        timesteps=64,
        seed=5,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=32,
        eval_episodes=1,
        telemetry="selected",
        run_name="smoke",
    )
    root = checkpoint.parents[1]
    assert checkpoint.exists()
    assert (root / "initial_model.zip").exists()
    assert (root / "checkpoints" / "initial_model.zip").exists()
    assert (root / "final_model.zip").exists()
    assert (root / "best_model.zip").exists()
    assert (root / "eval" / "eval_metrics.jsonl").exists()
    assert any((root / "eval" / "selected_telemetry").glob("*.jsonl"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["device"] == "cpu"
    assert metadata["vec_env"] == "dummy"
    assert metadata["scratch_initialization"] is True
    assert metadata["initial_checkpoint"].endswith("initial_model.zip")
    assert metadata["training_fps"] > 0.0
    eval_rows = [
        json.loads(line)
        for line in (root / "eval" / "eval_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert eval_rows[0]["timesteps"] == 0
    assert eval_rows[0]["phase"] == "initial_scratch"
    assert eval_rows[0]["scratch_initial_policy"] is True


def test_latest_checkpoint_finds_named_runs(tmp_path: Path) -> None:
    older = tmp_path / "named-run-a" / "checkpoints" / "older.zip"
    newer = tmp_path / "named-run-b" / "checkpoints" / "newer.zip"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    os.utime(older, (1.0, 1.0))
    os.utime(newer, (2.0, 2.0))
    assert policy_io.latest_checkpoint(tmp_path) == newer


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_resume_checkpoint_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    base_checkpoint = train.run_training(
        timesteps=64,
        seed=6,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="base",
    )
    resumed_checkpoint = train.run_training(
        timesteps=64,
        seed=7,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="resumed",
        resume_checkpoint=base_checkpoint,
    )
    root = resumed_checkpoint.parents[1]
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["scratch_initialization"] is False
    assert metadata["resume_checkpoint"] == str(base_checkpoint)
