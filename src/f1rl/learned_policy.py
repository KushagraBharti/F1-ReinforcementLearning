# pyright: reportPrivateImportUsage=false
"""Shared neural policy components for BC, SAC, and CPU policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6
DOMINANCE_EPSILON = 1.0e-3
CONTROL_MODES = frozenset({"independent", "dominance"})


@dataclass(slots=True)
class PolicyNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_observations(cls, obs: np.ndarray) -> PolicyNormalizer:
        mean = obs.astype(np.float64).mean(axis=0).astype(np.float32)
        std = obs.astype(np.float64).std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-4).astype(np.float32)
        return cls(mean=mean, std=std)

    @classmethod
    def identity(cls, obs_dim: int) -> PolicyNormalizer:
        return cls(mean=np.zeros(obs_dim, dtype=np.float32), std=np.ones(obs_dim, dtype=np.float32))

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        return ((obs.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=obs.dtype, device=obs.device)
        std = torch.as_tensor(self.std, dtype=obs.dtype, device=obs.device)
        return (obs - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyNormalizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def _mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(previous, hidden))
        layers.append(nn.ReLU())
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class SACActor(nn.Module):
    """Gaussian actor with squashed continuous car controls.

    The network's squashed raw action is in [-1, 1]^3. It is mapped to the
    simulator control surface as throttle in [0, 1], brake in [0, 1], and steer
    in [-1, 1].
    """

    def __init__(
        self,
        obs_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (256, 256),
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        control_mode: str = "independent",
    ) -> None:
        super().__init__()
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown SACActor control_mode={control_mode!r}; expected one of {sorted(CONTROL_MODES)}")
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.control_mode = control_mode
        if self.hidden_sizes:
            self.trunk = _mlp(self.obs_dim, self.hidden_sizes, self.hidden_sizes[-1])
            feature_dim = self.hidden_sizes[-1]
        else:
            self.trunk = nn.Identity()
            feature_dim = self.obs_dim
        self.mean_head = nn.Linear(feature_dim, 3)
        self.log_std_head = nn.Linear(feature_dim, 3)

    @staticmethod
    def _independent_raw_to_controls(raw_action: torch.Tensor) -> torch.Tensor:
        throttle = (raw_action[..., 0:1] + 1.0) * 0.5
        brake = (raw_action[..., 1:2] + 1.0) * 0.5
        steer = raw_action[..., 2:3]
        return torch.cat((throttle, brake, steer), dim=-1).clamp(
            torch.tensor([0.0, 0.0, -1.0], dtype=raw_action.dtype, device=raw_action.device),
            torch.tensor([1.0, 1.0, 1.0], dtype=raw_action.dtype, device=raw_action.device),
        )

    @staticmethod
    def _dominance_raw_to_controls(raw_action: torch.Tensor) -> torch.Tensor:
        throttle_raw = (raw_action[..., 0:1] + 1.0) * 0.5
        brake_raw = (raw_action[..., 1:2] + 1.0) * 0.5
        brake_dominates = brake_raw + DOMINANCE_EPSILON >= throttle_raw
        throttle = torch.where(brake_dominates, throttle_raw * (1.0 - brake_raw), throttle_raw)
        brake = torch.where(brake_dominates, brake_raw, brake_raw * (1.0 - throttle_raw))
        steer = raw_action[..., 2:3]
        return torch.cat((throttle, brake, steer), dim=-1).clamp(
            torch.tensor([0.0, 0.0, -1.0], dtype=raw_action.dtype, device=raw_action.device),
            torch.tensor([1.0, 1.0, 1.0], dtype=raw_action.dtype, device=raw_action.device),
        )

    def raw_to_controls(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self.control_mode == "dominance":
            return self._dominance_raw_to_controls(raw_action)
        return self._independent_raw_to_controls(raw_action)

    @staticmethod
    def controls_to_raw(action: torch.Tensor) -> torch.Tensor:
        throttle = action[..., 0:1].clamp(0.0, 1.0) * 2.0 - 1.0
        brake = action[..., 1:2].clamp(0.0, 1.0) * 2.0 - 1.0
        steer = action[..., 2:3].clamp(-1.0, 1.0)
        return torch.cat((throttle, brake, steer), dim=-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return self.raw_to_controls(torch.tanh(mean))

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        raw = torch.tanh(z)
        action = self.raw_to_controls(raw)
        log_prob = normal.log_prob(z) - torch.log(1.0 - raw.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def bc_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    action_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_dim = functional.smooth_l1_loss(pred, target, reduction="none")
    if action_weights is not None:
        per_dim = per_dim * action_weights.reshape(1, -1)
    if weights is not None:
        per_dim = per_dim * weights.reshape(-1, 1)
    return per_dim.mean()


def checkpoint_payload(
    *,
    actor: SACActor,
    normalizer: PolicyNormalizer,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "f1rl_learned_policy_checkpoint",
        "actor_class": "SACActor",
        "actor_kwargs": {
            "obs_dim": actor.obs_dim,
            "hidden_sizes": list(actor.hidden_sizes),
            "log_std_min": actor.log_std_min,
            "log_std_max": actor.log_std_max,
            "control_mode": actor.control_mode,
        },
        "actor_state_dict": actor.state_dict(),
        "normalizer": normalizer.to_dict(),
        "metadata": metadata,
    }


def save_policy_checkpoint(
    path: Path,
    *,
    actor: SACActor,
    normalizer: PolicyNormalizer,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(actor=actor, normalizer=normalizer, metadata=metadata), path)


def load_policy_checkpoint(path: Path, *, device: str | torch.device = "cpu") -> tuple[SACActor, PolicyNormalizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    kwargs = dict(payload["actor_kwargs"])
    kwargs["hidden_sizes"] = tuple(int(v) for v in kwargs["hidden_sizes"])
    actor = SACActor(**kwargs).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    normalizer = PolicyNormalizer.from_dict(payload["normalizer"])
    metadata = dict(payload.get("metadata", {}))
    return actor, normalizer, metadata
