"""Lightweight RLModule export/load helpers for inference-only workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.artifacts import owning_run_dir, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv

INFERENCE_METADATA_FILENAME = "inference_artifact.json"
INFERENCE_MODULE_DIRNAME = "module"


@dataclass(slots=True)
class LoadedInferencePolicy:
    checkpoint_path: Path
    artifact_dir: Path
    module: Any
    env_config: dict[str, Any]
    model_config: dict[str, Any]
    backend: str = "artifact"

    def compute_action(self, obs: np.ndarray, explore: bool = False) -> np.ndarray:
        import torch
        from ray.rllib.core.columns import Columns

        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        with torch.no_grad():
            output = self.module.forward_inference({Columns.OBS: torch.from_numpy(obs_batch)})

            if Columns.ACTIONS in output:
                action_tensor = output[Columns.ACTIONS]
            else:
                dist_cls = self.module.get_inference_action_dist_cls()
                dist = dist_cls.from_logits(output[Columns.ACTION_DIST_INPUTS])
                if not explore and hasattr(dist, "to_deterministic"):
                    dist = dist.to_deterministic()
                action_tensor = dist.sample()
        return np.asarray(action_tensor.detach().cpu().numpy()[0], dtype=np.float32)


def export_inference_artifact(
    *,
    algo: Any,
    checkpoint_path: Path,
    artifact_dir: Path,
    env_config: dict[str, Any],
    model_config: dict[str, Any],
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    module_dir = artifact_dir / INFERENCE_MODULE_DIRNAME
    if module_dir.exists():
        shutil.rmtree(module_dir)

    module = algo.get_module()
    module.save_to_path(str(module_dir))

    env = F1RaceEnv(env_config)
    try:
        metadata = {
            "format_version": 1,
            "algorithm": "PPO",
            "framework": "torch",
            "checkpoint": str(checkpoint_path),
            "env_config": env_config,
            "model_config": model_config,
            "observation_space": {
                "shape": list(env.observation_space.shape or ()),
                "dtype": str(env.observation_space.dtype),
            },
            "action_space": {
                "shape": list(env.action_space.shape or ()),
                "dtype": str(env.action_space.dtype),
            },
        }
    finally:
        env.close()

    metadata_path = artifact_dir / INFERENCE_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def inference_artifact_dir_for_checkpoint(checkpoint: str | Path) -> Path:
    checkpoint_path = resolve_checkpoint(str(checkpoint))
    return owning_run_dir(checkpoint_path) / "inference"


def load_inference_metadata(checkpoint: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint(str(checkpoint))
    artifact_dir = owning_run_dir(checkpoint_path) / "inference"
    metadata_path = artifact_dir / INFERENCE_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing lightweight inference artifact for checkpoint {checkpoint_path}. "
            "Train a new run or use legacy algorithm restore mode."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return checkpoint_path, artifact_dir, metadata


def build_inference_module(*, env_config: dict[str, Any], model_config: dict[str, Any]) -> Any:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module import RLModuleSpec

    env = F1RaceEnv(env_config)
    try:
        base_config = (
            PPOConfig()
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
            .environment(env=F1RaceEnv, env_config=env_config)
        )
        default_spec = base_config.get_default_rl_module_spec()
        module_spec = RLModuleSpec(
            module_class=default_spec.module_class,
            observation_space=env.observation_space,
            action_space=env.action_space,
            inference_only=True,
            model_config=model_config,
            catalog_class=default_spec.catalog_class,
        )
        return module_spec.build()
    finally:
        env.close()


def load_inference_policy(checkpoint: str | Path) -> LoadedInferencePolicy:
    checkpoint_path, artifact_dir, metadata = load_inference_metadata(checkpoint)
    module_dir = artifact_dir / INFERENCE_MODULE_DIRNAME
    module = build_inference_module(
        env_config=metadata["env_config"],
        model_config=metadata["model_config"],
    )
    module.restore_from_path(str(module_dir))
    if hasattr(module, "eval"):
        module.eval()

    return LoadedInferencePolicy(
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        module=module,
        env_config=metadata["env_config"],
        model_config=metadata["model_config"],
    )


def default_eval_env_config(*, steps: int, render: bool, headless: bool, export_frames: bool, frame_prefix: str) -> dict:
    config = EnvConfig()
    config.max_steps = max(200, steps)
    config.render_mode = "human" if render and not headless else "rgb_array" if export_frames else None
    config.render.enabled = render or export_frames
    config.headless = headless
    config.render.export_frames = export_frames
    config.render.frame_prefix = frame_prefix
    config.render.draw_hud = render or export_frames
    return to_dict(config)


def deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
