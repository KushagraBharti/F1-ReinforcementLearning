"""Policy loading/saving boundary shared by PPO now and future policy types."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from f1rl.config import (
    ARTIFACTS_DIR,
    REPO_ROOT,
    AssistConfig,
    CarParams,
    RewardConfig,
    SensorConfig,
    SimConfig,
    dataclass_to_dict,
)
from f1rl.hardware import torch_device


@dataclass(slots=True)
class PpoEvalConfig:
    checkpoint_path: Path
    run_root: Path | None
    metadata_path: Path | None
    metadata: dict[str, Any]
    sim_config: SimConfig
    vecnormalize_path: Path | None
    metadata_loaded: bool
    metadata_mode: str
    config_source: str

    def report(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "run_root": str(self.run_root) if self.run_root is not None else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path is not None else None,
            "metadata_loaded": self.metadata_loaded,
            "metadata_mode": self.metadata_mode,
            "config_source": self.config_source,
            "vecnormalize_path": str(self.vecnormalize_path) if self.vecnormalize_path is not None else None,
            "sim_config": dataclass_to_dict(self.sim_config),
        }


def latest_checkpoint(root: Path = ARTIFACTS_DIR) -> Path:
    candidates = sorted(root.glob("*/checkpoints/*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No checkpoints found under artifacts/*/checkpoints.")
    return candidates[-1]


def resolve_checkpoint_path(checkpoint: Path | str, *, root: Path = ARTIFACTS_DIR) -> Path:
    if str(checkpoint) == "latest":
        return latest_checkpoint(root)
    path = Path(checkpoint)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "best_model.zip",
            path / "final_model.zip",
            path / "checkpoints" / "best_model.zip",
            path / "checkpoints" / "final_model.zip",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        checkpoint_candidates = sorted(path.glob("checkpoints/*.zip"), key=lambda item: item.stat().st_mtime)
        if checkpoint_candidates:
            return checkpoint_candidates[-1]
        zip_candidates = sorted(path.glob("*.zip"), key=lambda item: item.stat().st_mtime)
        if zip_candidates:
            return zip_candidates[-1]
    raise FileNotFoundError(f"PPO checkpoint does not exist or contains no .zip model: {checkpoint}")


def infer_run_root(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    if (checkpoint_path.parent / "run_metadata.json").is_file():
        return checkpoint_path.parent
    for parent in checkpoint_path.parents:
        if (parent / "run_metadata.json").is_file():
            return parent
        if parent == ARTIFACTS_DIR or parent.parent == parent:
            break
    return None


def _resolve_metadata_path(checkpoint_path: Path) -> Path | None:
    run_root = infer_run_root(checkpoint_path)
    if run_root is None:
        return None
    metadata_path = run_root / "run_metadata.json"
    return metadata_path if metadata_path.is_file() else None


def _resolve_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _dataclass_kwargs(cls: type[Any], values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in valid}


def sim_config_from_metadata(metadata: dict[str, Any], *, max_steps: int | None = None) -> SimConfig:
    raw_config = dict(metadata.get("sim_config") or {})
    car = CarParams(**_dataclass_kwargs(CarParams, raw_config.get("car")))
    sensors = SensorConfig(**_dataclass_kwargs(SensorConfig, raw_config.get("sensors")))
    reward = RewardConfig(**_dataclass_kwargs(RewardConfig, raw_config.get("reward")))
    assist = AssistConfig(**_dataclass_kwargs(AssistConfig, raw_config.get("assist")))
    kwargs = _dataclass_kwargs(SimConfig, raw_config)
    kwargs.pop("car", None)
    kwargs.pop("sensors", None)
    kwargs.pop("reward", None)
    kwargs.pop("assist", None)
    if "track_path" in kwargs:
        kwargs["track_path"] = _resolve_path(kwargs["track_path"])
    if "car_image" in kwargs:
        kwargs["car_image"] = _resolve_path(kwargs["car_image"])
    if "lookahead_m" in kwargs:
        kwargs["lookahead_m"] = tuple(float(value) for value in kwargs["lookahead_m"])
    sim_config = SimConfig(car=car, sensors=sensors, reward=reward, assist=assist, **kwargs)
    if max_steps is not None:
        sim_config = replace(sim_config, max_steps=max_steps)
    return sim_config


def _candidate_vecnormalize_paths(checkpoint_path: Path, metadata: dict[str, Any]) -> list[Path]:
    run_root = infer_run_root(checkpoint_path)
    candidates: list[Path] = []
    for key in ("final_vecnormalize", "vec_normalize_path"):
        resolved = _resolve_path(metadata.get(key))
        if resolved is not None:
            candidates.append(resolved)
    if run_root is not None:
        if checkpoint_path.name == "best_model.zip":
            candidates.append(run_root / "best_vecnormalize.pkl")
        candidates.append(run_root / "vecnormalize.pkl")
        candidates.append(run_root / "best_vecnormalize.pkl")
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def resolve_vecnormalize_path(checkpoint_path: Path, metadata: dict[str, Any]) -> Path | None:
    for candidate in _candidate_vecnormalize_paths(checkpoint_path, metadata):
        if candidate.is_file():
            return candidate
    return None


def resolve_ppo_eval_config(
    checkpoint: Path | str,
    *,
    max_steps: int,
    fallback_config: SimConfig | None = None,
    metadata_mode: str = "auto",
    root: Path = ARTIFACTS_DIR,
) -> PpoEvalConfig:
    if metadata_mode not in {"auto", "require", "ignore"}:
        raise ValueError("metadata_mode must be one of: auto, require, ignore.")
    checkpoint_path = resolve_checkpoint_path(checkpoint, root=root)
    metadata_path = None if metadata_mode == "ignore" else _resolve_metadata_path(checkpoint_path)
    if metadata_mode == "require" and metadata_path is None:
        raise FileNotFoundError(f"No run_metadata.json found for checkpoint: {checkpoint_path}")
    metadata: dict[str, Any] = {}
    metadata_loaded = False
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sim_config = sim_config_from_metadata(metadata, max_steps=max_steps)
        metadata_loaded = True
        config_source = "run_metadata"
    else:
        sim_config = replace(fallback_config or SimConfig(), max_steps=max_steps)
        config_source = "explicit_args"
    return PpoEvalConfig(
        checkpoint_path=checkpoint_path,
        run_root=infer_run_root(checkpoint_path),
        metadata_path=metadata_path,
        metadata=metadata,
        sim_config=sim_config,
        vecnormalize_path=resolve_vecnormalize_path(checkpoint_path, metadata) if metadata_loaded else None,
        metadata_loaded=metadata_loaded,
        metadata_mode=metadata_mode,
        config_source=config_source,
    )


def validate_model_spaces(model: Any, *, observation_space: spaces.Space[Any], action_space: spaces.Space[Any]) -> None:
    model_observation_space = getattr(model, "observation_space", None)
    model_action_space = getattr(model, "action_space", None)
    if model_observation_space is None or model_action_space is None:
        return
    if tuple(getattr(model_observation_space, "shape", ())) != tuple(getattr(observation_space, "shape", ())):
        raise ValueError(
            "Model observation shape does not match eval environment: "
            f"model={getattr(model_observation_space, 'shape', None)} "
            f"env={getattr(observation_space, 'shape', None)}. "
            "Load the matching run metadata or pass the correct observation profile."
        )
    if isinstance(model_action_space, spaces.Discrete) and isinstance(action_space, spaces.Discrete):
        if int(model_action_space.n) != int(action_space.n):
            raise ValueError(f"Model action count {model_action_space.n} does not match env action count {action_space.n}.")
        return
    if isinstance(model_action_space, spaces.MultiDiscrete) and isinstance(action_space, spaces.MultiDiscrete):
        if not np.array_equal(model_action_space.nvec, action_space.nvec):
            raise ValueError(f"Model MultiDiscrete nvec {model_action_space.nvec} does not match env nvec {action_space.nvec}.")
        return
    if isinstance(model_action_space, spaces.Box) and isinstance(action_space, spaces.Box):
        if tuple(model_action_space.shape) != tuple(action_space.shape):
            raise ValueError(f"Model action shape {model_action_space.shape} does not match env action shape {action_space.shape}.")
        return
    if type(model_action_space) is not type(action_space):
        raise ValueError(
            "Model action space type does not match eval environment: "
            f"model={type(model_action_space).__name__} env={type(action_space).__name__}."
        )


def load_vecnormalize_stats(config: PpoEvalConfig) -> Any | None:
    if config.vecnormalize_path is None:
        return None
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - exercised before train extra is installed
        raise RuntimeError("stable-baselines3 is required to load VecNormalize stats.") from exc
    from f1rl.env import MonzaEnv

    vec_env = DummyVecEnv([lambda: MonzaEnv(config.sim_config)])
    vec_normalize = VecNormalize.load(str(config.vecnormalize_path), vec_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def normalize_observation(obs: Any, vec_normalize: Any | None) -> Any:
    if vec_normalize is None:
        return obs
    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 1:
        return vec_normalize.normalize_obs(obs_array.reshape(1, -1))[0]
    return vec_normalize.normalize_obs(obs_array)


def close_vecnormalize(vec_normalize: Any | None) -> None:
    if vec_normalize is not None:
        vec_normalize.close()


def load_sb3_ppo(checkpoint: Path | str, *, env: Any | None = None, device: str = "auto") -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - exercised before train extra is installed
        raise RuntimeError("stable-baselines3 is required. Run `uv sync --active --all-extras --all-packages`.") from exc
    resolved = resolve_checkpoint_path(checkpoint)
    return PPO.load(resolved, env=env, device=torch_device(device))
