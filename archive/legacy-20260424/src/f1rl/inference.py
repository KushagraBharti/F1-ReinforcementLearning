"""Torch-native inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f1rl.artifacts import owning_run_dir, resolve_checkpoint
from f1rl.torch_agent import LoadedTorchPolicy, load_torch_policy


def load_inference_policy(checkpoint: str | Path, *, device: str = "auto") -> LoadedTorchPolicy:
    return load_torch_policy(checkpoint, device=device)


def training_env_config_for_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint(str(checkpoint))
    metadata_path = checkpoint_path / "torch_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        env_config = metadata.get("env_config", {})
        if isinstance(env_config, dict):
            return env_config

    run_dir = owning_run_dir(checkpoint_path)
    run_metadata_path = run_dir / "run_metadata.json"
    if run_metadata_path.exists():
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        env_config = metadata.get("env_config", {})
        if isinstance(env_config, dict):
            return env_config
    return {}


def default_eval_env_config(*, steps: int, render: bool, headless: bool, export_frames: bool, frame_prefix: str) -> dict[str, Any]:
    return {
        "max_steps": max(200, steps),
        "render_mode": "human" if render and not headless else "rgb_array" if export_frames else None,
        "headless": headless,
        "render": {
            "enabled": render or export_frames,
            "export_frames": export_frames,
            "frame_prefix": frame_prefix,
            "draw_hud": render or export_frames,
        },
    }


def deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

