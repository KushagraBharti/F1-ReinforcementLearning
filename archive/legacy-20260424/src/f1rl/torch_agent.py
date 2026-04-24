"""Torch-native PPO policy and checkpoint helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from f1rl.artifacts import owning_run_dir, resolve_checkpoint
from f1rl.hardware import get_torch_runtime_info

CHECKPOINT_STATE_FILENAME = "torch_checkpoint.pt"
CHECKPOINT_METADATA_FILENAME = "torch_metadata.json"


class ActorCriticPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = obs_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(in_features, width), nn.Tanh()))
            in_features = width
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_features, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.75))
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        return self.actor_mean(features), self.value_head(features).squeeze(-1)

    def distribution(self, observations: torch.Tensor) -> tuple[TransformedDistribution, torch.Tensor]:
        mean, value = self(observations)
        std = self.actor_log_std.exp().expand_as(mean)
        base = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        return dist, value

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution(observations)
        if deterministic:
            mean, _ = self(observations)
            actions = torch.tanh(mean)
            log_prob = dist.log_prob(actions)
        else:
            actions = dist.rsample()
            log_prob = dist.log_prob(actions)
        return actions, log_prob, value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution(observations)
        log_prob = dist.log_prob(actions)
        entropy = dist.base_dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass(slots=True)
class PPOHyperParams:
    horizon: int = 128
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class LoadedTorchPolicy:
    checkpoint_path: Path
    device: str
    env_config: dict[str, Any]
    metadata: dict[str, Any]
    model: ActorCriticPolicy

    def compute_actions(self, observations: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            actions, _, _ = self.model.act(observations.to(self.device), deterministic=deterministic)
        return actions

    def compute_action(self, observation: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        return self.compute_actions(observation.unsqueeze(0), deterministic=deterministic)[0]


def checkpoint_files(path: str | Path) -> tuple[Path, Path]:
    checkpoint_path = resolve_checkpoint(str(path))
    return checkpoint_path / CHECKPOINT_STATE_FILENAME, checkpoint_path / CHECKPOINT_METADATA_FILENAME


def resolve_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device(device)


def save_torch_checkpoint(
    checkpoint_dir: Path,
    *,
    model: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer | None,
    env_config: dict[str, Any],
    training_metadata: dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / CHECKPOINT_STATE_FILENAME
    metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "training_metadata": training_metadata,
    }
    torch.save(payload, state_path)
    runtime = get_torch_runtime_info()
    metadata = {
        "format_version": 1,
        "backend": "torch_native",
        "env_config": env_config,
        "policy": {
            "obs_dim": model.backbone[0].in_features if len(model.backbone) > 0 else model.actor_mean.in_features,
            "action_dim": model.actor_mean.out_features,
            "hidden_sizes": [module.out_features for module in model.backbone if isinstance(module, nn.Linear)],
        },
        "torch": {
            "version": runtime.torch_version,
            "cuda_available": runtime.cuda_available,
            "cuda_version": runtime.cuda_version,
            "device_name": runtime.device_name,
        },
        "training_metadata": training_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return state_path


def load_torch_policy(checkpoint: str | Path, *, device: str = "auto") -> LoadedTorchPolicy:
    state_path, metadata_path = checkpoint_files(checkpoint)
    if not state_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing torch-native checkpoint files under {resolve_checkpoint(str(checkpoint))}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resolved = resolve_torch_device(device)
    policy_cfg = metadata["policy"]
    model = ActorCriticPolicy(
        obs_dim=int(policy_cfg["obs_dim"]),
        action_dim=int(policy_cfg["action_dim"]),
        hidden_sizes=tuple(int(width) for width in policy_cfg["hidden_sizes"]),
    ).to(resolved)
    payload = torch.load(state_path, map_location=resolved)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return LoadedTorchPolicy(
        checkpoint_path=resolve_checkpoint(str(checkpoint)),
        device=str(resolved),
        env_config=metadata["env_config"],
        metadata=metadata,
        model=model,
    )


def checkpoint_run_dir(checkpoint: str | Path) -> Path:
    return owning_run_dir(resolve_checkpoint(str(checkpoint)))
