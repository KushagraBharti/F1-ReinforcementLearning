"""Policy loading/saving boundary shared by PPO now and future policy types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from f1rl.config import ARTIFACTS_DIR


def latest_checkpoint(root: Path = ARTIFACTS_DIR) -> Path:
    candidates = sorted(root.glob("train-*/checkpoints/*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No checkpoints found under artifacts/train-*/checkpoints.")
    return candidates[-1]


def load_sb3_ppo(checkpoint: Path | str, *, env: Any | None = None, device: str = "auto") -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - exercised before train extra is installed
        raise RuntimeError("stable-baselines3 is required. Run `uv sync --active --all-extras --all-packages`.") from exc
    resolved = latest_checkpoint() if str(checkpoint) == "latest" else Path(checkpoint)
    return PPO.load(resolved, env=env, device=device)
