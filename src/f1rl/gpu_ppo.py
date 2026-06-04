# pyright: reportPrivateImportUsage=false
"""Opt-in GPU-native PPO trainer.

This module deliberately does not replace the SB3/Gymnasium path in ``train.py``.
It keeps rollout collection, policy inference, rewards, and buffers in PyTorch so
the batched GPU simulator can be used without per-step NumPy round-trips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from f1rl.config import (
    ARTIFACTS_DIR,
    MONZA_LENGTH_METERS,
    SimConfig,
    build_sim_config,
    dataclass_to_dict,
)
from f1rl.gpu_batch import GpuMonzaBatch
from f1rl.gpu_observation import observation_batch
from f1rl.gpu_scoring import TERMINATION_REASON_TO_ID
from f1rl.gpu_track import ray_distances_batch, sensor_angles_tensor
from f1rl.gpu_types import (
    GpuCarBatch,
    car_batch_from_snapshots,
    gpu_track_from_cpu,
    normalize_device,
    torch_dtype_from_name,
)
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import StateSnapshot, snapshot_from_sim
from f1rl.track_model import load_track_spec


@dataclass(frozen=True, slots=True)
class GpuPPOConfig:
    output_dir: Path | None = None
    device: str = "cuda"
    dtype: str = "float32"
    require_gpu: bool = False
    seed: int = 7
    timesteps: int = 1024
    n_envs: int = 64
    n_steps: int = 128
    batch_size: int = 1024
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    max_steps: int = 6000
    observation_profile: str = "racing_v2"
    continuous_action_scheme: str = "drive_brake"
    start_speed_kph: float = 0.0
    start_progress_m: float | None = None
    target_progress_m: float = MONZA_LENGTH_METERS
    terminate_at_target: bool = False
    ray_chunk_size: int = 256
    collision_chunk_size: int = 512
    compile_policy: bool = False
    cpu_eval_episodes: int = 1
    cpu_eval_max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class GpuPPOGates:
    target_progress_m: float
    terminate_at_target_progress: bool
    target_min_speed_kph: float | None = None
    target_max_speed_kph: float | None = None
    target_max_lateral_error_m: float | None = None
    target_max_heading_error_deg: float | None = None
    target_max_abs_yaw_rate_rps: float | None = None
    target_max_abs_steering: float | None = None
    segment_fail_on_speed_gate_miss: bool = False
    segment_require_release: bool = False
    segment_release_min_speed_kph: float = 135.0
    segment_release_max_speed_kph: float = 210.0
    segment_release_max_brake: float = 0.1
    segment_release_max_throttle: float = 0.1


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(obs))
        value = self.critic(obs).squeeze(-1)
        return mean, value


class PolicyRunner:
    def __init__(self, model: ActorCritic, *, compile_policy: bool) -> None:
        self.model = model
        self.module: Any = model
        self.compile_requested = bool(compile_policy)
        self.compile_enabled = False
        self.compile_error: str | None = None
        if self.compile_requested:
            try:
                self.module = torch.compile(model, mode="reduce-overhead")
                self.compile_enabled = True
            except Exception as exc:  # pragma: no cover - local compiler support varies.
                self.compile_error = f"{type(exc).__name__}: {exc}"
                self.module = model

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self.module(obs)
        except Exception as exc:
            if not self.compile_enabled:
                raise
            self.compile_enabled = False
            self.compile_error = f"{type(exc).__name__}: {exc}"
            self.module = self.model
            return self.model(obs)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _continuous_controls(actions: torch.Tensor, scheme: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drive = torch.clamp(actions[:, 0], min=-1.0, max=1.0)
    steer = torch.clamp(actions[:, 1], min=-1.0, max=1.0)
    if scheme == "exclusive_throttle_bias":
        throttle = torch.where(drive >= 0.0, 0.25 + 0.75 * drive, torch.zeros_like(drive))
        brake = torch.where(drive < 0.0, -drive, torch.zeros_like(drive))
    elif scheme == "throttle_bias":
        throttle = 0.5 + 0.5 * drive
        brake = torch.clamp(-drive, min=0.0, max=1.0)
    else:
        throttle = torch.clamp(drive, min=0.0, max=1.0)
        brake = torch.clamp(-drive, min=0.0, max=1.0)
    return throttle, brake, steer


def _normal_start_snapshots(
    *,
    sim_config: SimConfig,
    count: int,
    seed: int,
    start_speed_kph: float,
    start_progress_m: float | None,
) -> list[StateSnapshot]:
    sim = MonzaSim(sim_config)
    snapshots: list[StateSnapshot] = []
    for index in range(count):
        options: dict[str, Any] = {
            "start_speed_kph": float(start_speed_kph),
            "collect_observation": False,
        }
        if start_progress_m is not None:
            options["start_progress_m"] = float(start_progress_m)
        sim.reset(seed=seed + index, options=options)
        snapshots.append(snapshot_from_sim(sim, source="gpu_ppo_start", source_file=None))
    return snapshots


class GpuPPOBatchEnv:
    def __init__(
        self,
        *,
        sim_config: SimConfig,
        n_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        start_speed_kph: float,
        start_progress_m: float | None,
        target_progress_m: float,
        terminate_at_target: bool,
        ray_chunk_size: int,
        collision_chunk_size: int,
    ) -> None:
        self.sim_config = sim_config
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.start_speed_kph = start_speed_kph
        self.start_progress_m = start_progress_m
        self.ray_chunk_size = ray_chunk_size
        self.track = gpu_track_from_cpu(load_track_spec(sim_config.track_path), device=device, dtype=dtype)
        self.gates = GpuPPOGates(
            target_progress_m=float(target_progress_m),
            terminate_at_target_progress=bool(terminate_at_target),
        )
        self.batch = GpuMonzaBatch(
            track=self.track,
            sim_config=sim_config,
            feature_names=(),
            collision_check=True,
            collision_chunk_size=collision_chunk_size,
        )
        self.sensor_angles = sensor_angles_tensor(
            count=sim_config.sensors.count,
            spread_deg=sim_config.sensors.spread_deg,
            forward_bias=sim_config.sensors.forward_bias,
            device=device,
            dtype=dtype,
        )
        self.episode_returns = torch.zeros(n_envs, device=device, dtype=dtype)
        self.episode_lengths = torch.zeros(n_envs, device=device, dtype=torch.int64)
        self.episode_count = 0
        self._pending_episode_summaries: list[dict[str, Any]] = []

    def reset(self) -> torch.Tensor:
        snapshots = _normal_start_snapshots(
            sim_config=self.sim_config,
            count=self.n_envs,
            seed=self.seed,
            start_speed_kph=self.start_speed_kph,
            start_progress_m=self.start_progress_m,
        )
        self.batch.reset(
            snapshots,
            target_progress_m=self.gates.target_progress_m,
            terminate_at_target=self.gates.terminate_at_target_progress,
        )
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        return self.observe()

    def observe(self) -> torch.Tensor:
        if self.batch.state is None:
            raise RuntimeError("GpuPPOBatchEnv.reset() must be called before observe().")
        return observation_batch(
            self.batch.state,
            self.track,
            self.sim_config,
            sensor_angles=self.sensor_angles,
            ray_chunk_size=self.ray_chunk_size,
        )

    def _reset_done(self, done: torch.Tensor) -> None:
        if self.batch.state is None or self.batch.segment_start_progress_m is None:
            raise RuntimeError("GpuPPOBatchEnv.reset() must be called before step().")
        indices = torch.nonzero(done, as_tuple=False).flatten()
        if indices.numel() == 0:
            return
        cpu_indices = [int(value) for value in indices.detach().cpu().tolist()]
        snapshots = _normal_start_snapshots(
            sim_config=self.sim_config,
            count=len(cpu_indices),
            seed=self.seed + self.episode_count * self.n_envs + 10_000,
            start_speed_kph=self.start_speed_kph,
            start_progress_m=self.start_progress_m,
        )
        replacement = car_batch_from_snapshots(
            snapshots,
            meters_per_pixel=float(self.track.meters_per_pixel.detach().cpu()),
            device=self.device,
            dtype=self.dtype,
        )
        state = self.batch.state
        for field in fields(GpuCarBatch):
            getattr(state, field.name)[indices] = getattr(replacement, field.name)
        self.batch.segment_start_progress_m[indices] = replacement.monotonic_progress_m
        if self.batch.segment_target_progress_m is not None:
            self.batch.segment_target_progress_m[indices] = (
                replacement.monotonic_progress_m
                + torch.clamp(
                    torch.as_tensor(self.gates.target_progress_m, device=self.device, dtype=self.dtype)
                    - replacement.monotonic_progress_m,
                    min=1.0,
                )
            )
        self.episode_returns[indices] = 0.0
        self.episode_lengths[indices] = 0

    def _reward(
        self,
        *,
        diagnostics: dict[str, torch.Tensor],
        throttle: torch.Tensor,
        brake: torch.Tensor,
        steer: torch.Tensor,
        previous_steer: torch.Tensor,
        active: torch.Tensor,
        collided: torch.Tensor,
        off_track: torch.Tensor,
    ) -> torch.Tensor:
        assert self.batch.state is not None
        state = self.batch.state
        reward = diagnostics["progress_delta_m"] * float(self.sim_config.reward.progress_scale)
        lateral_excess = torch.clamp(
            torch.abs(diagnostics["lateral_error_m"]) - float(self.sim_config.reward.lateral_deadzone_m),
            min=0.0,
        )
        reward -= float(self.sim_config.reward.lateral_penalty_scale) * lateral_excess
        rays = ray_distances_batch(
            state.x,
            state.y,
            state.heading_rad,
            self.track,
            sensor_angles=self.sensor_angles,
            range_m=self.sim_config.sensors.range_m,
            chunk_size=self.ray_chunk_size,
        )
        min_ray = torch.min(rays, dim=1).values if rays.shape[1] else torch.full_like(state.speed_mps, self.sim_config.reward.track_limit_safe_ray_m)
        speed_factor = torch.maximum(torch.ones_like(state.speed_mps), (state.speed_mps * 3.6) / 100.0)
        track_limit_excess = torch.clamp(float(self.sim_config.reward.track_limit_safe_ray_m) - min_ray, min=0.0)
        reward -= float(self.sim_config.reward.track_limit_penalty_scale) * track_limit_excess * speed_factor
        heading_error_deg = torch.abs(diagnostics["heading_error_rad"] * (180.0 / math.pi))
        heading_excess = torch.clamp(heading_error_deg - float(self.sim_config.reward.heading_deadzone_deg), min=0.0)
        reward -= float(self.sim_config.reward.heading_penalty_scale) * heading_excess * speed_factor
        reward -= float(self.sim_config.reward.smoothness_penalty) * torch.abs(steer - previous_steer)
        finish = active & (state.segment_complete | state.completed_lap)
        no_progress = active & (state.termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"])
        reward += torch.where(finish, torch.full_like(reward, self.sim_config.reward.finish_bonus), torch.zeros_like(reward))
        reward += torch.where(collided, torch.full_like(reward, self.sim_config.reward.collision_penalty), torch.zeros_like(reward))
        reward += torch.where(off_track, torch.full_like(reward, self.sim_config.reward.off_track_penalty), torch.zeros_like(reward))
        reward += torch.where(no_progress, torch.full_like(reward, self.sim_config.reward.no_progress_penalty), torch.zeros_like(reward))
        del throttle, brake
        return torch.where(active, reward, torch.zeros_like(reward))

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch.state is None:
            raise RuntimeError("GpuPPOBatchEnv.reset() must be called before step().")
        active = self.batch._active()
        previous_steer = self.batch.state.last_steer.clone()
        throttle, brake, steer = _continuous_controls(actions, self.sim_config.continuous_action_scheme)
        diagnostics, collided, off_track, _telemetry_valid_lap = self.batch._step(
            throttle=throttle,
            brake=brake,
            steer=steer,
            active=active,
            gates=self.gates,
        )
        diagnostics["heading_error_deg"] = diagnostics["heading_error_rad"] * (180.0 / math.pi)
        reward = self._reward(
            diagnostics=diagnostics,
            throttle=throttle,
            brake=brake,
            steer=steer,
            previous_steer=previous_steer,
            active=active,
            collided=collided,
            off_track=off_track,
        )
        done = self.batch.state.terminated | self.batch.state.truncated
        self.episode_returns += reward
        self.episode_lengths += active.to(dtype=torch.int64)
        if bool(done.any().detach().cpu().item()):
            done_indices = [int(value) for value in torch.nonzero(done, as_tuple=False).flatten().detach().cpu().tolist()]
            for index in done_indices:
                self._pending_episode_summaries.append(
                    {
                        "episode": self.episode_count,
                        "env_index": index,
                        "return": float(self.episode_returns[index].detach().cpu()),
                        "length": int(self.episode_lengths[index].detach().cpu()),
                        "final_progress_m": float(self.batch.state.monotonic_progress_m[index].detach().cpu()),
                        "best_progress_m": float(self.batch.state.monotonic_progress_m[index].detach().cpu()),
                        "terminated": bool(self.batch.state.terminated[index].detach().cpu()),
                        "truncated": bool(self.batch.state.truncated[index].detach().cpu()),
                        "completed_lap": bool(self.batch.state.completed_lap[index].detach().cpu()),
                        "valid_lap": bool(self.batch.state.valid_lap[index].detach().cpu()),
                    }
                )
                self.episode_count += 1
            self._reset_done(done)
        return self.observe(), reward.to(dtype=torch.float32), done

    def drain_episode_summaries(self) -> list[dict[str, Any]]:
        summaries = self._pending_episode_summaries
        self._pending_episode_summaries = []
        return summaries


def _dist(model: ActorCritic, mean: torch.Tensor) -> Normal:
    std = torch.exp(model.log_std).expand_as(mean)
    return Normal(mean, std)


def _sample_action(policy: PolicyRunner, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, value = policy(obs)
    dist = _dist(policy.model, mean)
    raw_action = dist.sample()
    log_prob = dist.log_prob(raw_action).sum(dim=1)
    return raw_action, log_prob, value


def _evaluate_actions(
    policy: PolicyRunner,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, value = policy(obs)
    dist = _dist(policy.model, mean)
    log_prob = dist.log_prob(actions).sum(dim=1)
    entropy = dist.entropy().sum(dim=1)
    return log_prob, entropy, value


def train_gpu_ppo(config: GpuPPOConfig) -> Path:
    device = normalize_device(config.device)
    if config.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set, but the resolved GPU PPO device is not CUDA.")
    dtype = torch_dtype_from_name(config.dtype)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    run_dir = config.output_dir or ARTIFACTS_DIR / f"gpu-ppo-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_config = build_sim_config(
        max_steps=config.max_steps,
        action_mode="continuous",
        continuous_action_scheme=config.continuous_action_scheme,
        observation_profile=config.observation_profile,
    )
    env = GpuPPOBatchEnv(
        sim_config=sim_config,
        n_envs=max(1, config.n_envs),
        device=device,
        dtype=dtype,
        seed=config.seed,
        start_speed_kph=config.start_speed_kph,
        start_progress_m=config.start_progress_m,
        target_progress_m=config.target_progress_m,
        terminate_at_target=config.terminate_at_target,
        ray_chunk_size=config.ray_chunk_size,
        collision_chunk_size=config.collision_chunk_size,
    )
    obs = env.reset().to(device=device)
    obs_dim = int(obs.shape[1])
    model = ActorCritic(obs_dim, max(16, config.hidden_size)).to(device=device)
    policy = PolicyRunner(model, compile_policy=config.compile_policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    rollout_size = max(1, config.n_envs) * max(1, config.n_steps)
    updates = max(1, math.ceil(max(1, config.timesteps) / rollout_size))
    episode_summaries: list[dict[str, Any]] = []
    update_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()

    for update in range(updates):
        obs_buf = torch.empty((config.n_steps, config.n_envs, obs_dim), device=device, dtype=torch.float32)
        action_buf = torch.empty((config.n_steps, config.n_envs, 2), device=device, dtype=torch.float32)
        logprob_buf = torch.empty((config.n_steps, config.n_envs), device=device, dtype=torch.float32)
        reward_buf = torch.empty((config.n_steps, config.n_envs), device=device, dtype=torch.float32)
        done_buf = torch.empty((config.n_steps, config.n_envs), device=device, dtype=torch.bool)
        value_buf = torch.empty((config.n_steps, config.n_envs), device=device, dtype=torch.float32)
        for step in range(config.n_steps):
            obs_buf[step] = obs
            with torch.no_grad():
                raw_action, log_prob, value = _sample_action(policy, obs)
            env_action = torch.clamp(raw_action, min=-1.0, max=1.0)
            next_obs, reward, done = env.step(env_action)
            action_buf[step] = raw_action
            logprob_buf[step] = log_prob
            reward_buf[step] = reward
            done_buf[step] = done
            value_buf[step] = value
            obs = next_obs.to(device=device)
            episode_summaries.extend(env.drain_episode_summaries())
        with torch.no_grad():
            _next_mean, next_value = policy(obs)
        advantages = torch.zeros_like(reward_buf)
        last_gae = torch.zeros(config.n_envs, device=device, dtype=torch.float32)
        for step in reversed(range(config.n_steps)):
            if step == config.n_steps - 1:
                next_values = next_value
            else:
                next_values = value_buf[step + 1]
            next_nonterminal = (~done_buf[step]).to(dtype=torch.float32)
            delta = reward_buf[step] + config.gamma * next_values * next_nonterminal - value_buf[step]
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + value_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = action_buf.reshape((-1, 2))
        b_logprobs = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / torch.clamp(b_advantages.std(), min=1e-8)
        batch_size = b_obs.shape[0]
        minibatch_size = min(max(1, config.batch_size), batch_size)
        last_loss = torch.tensor(0.0, device=device)
        last_policy_loss = torch.tensor(0.0, device=device)
        last_value_loss = torch.tensor(0.0, device=device)
        last_entropy = torch.tensor(0.0, device=device)
        for _epoch in range(max(1, config.n_epochs)):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = permutation[start : start + minibatch_size]
                new_logprob, entropy, new_value = _evaluate_actions(policy, b_obs[mb], b_actions[mb])
                log_ratio = new_logprob - b_logprobs[mb]
                ratio = torch.exp(log_ratio)
                unclipped = -b_advantages[mb] * ratio
                clipped = -b_advantages[mb] * torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()
                value_clipped = b_values[mb] + torch.clamp(new_value - b_values[mb], -config.clip_range, config.clip_range)
                value_loss_unclipped = (new_value - b_returns[mb]) ** 2
                value_loss_clipped = (value_clipped - b_returns[mb]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                last_loss = loss.detach()
                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy = entropy_loss.detach()
        update_summaries.append(
            {
                "update": update,
                "timesteps": (update + 1) * rollout_size,
                "mean_reward": float(reward_buf.mean().detach().cpu()),
                "mean_return": float(b_returns.mean().detach().cpu()),
                "loss": float(last_loss.detach().cpu()),
                "policy_loss": float(last_policy_loss.detach().cpu()),
                "value_loss": float(last_value_loss.detach().cpu()),
                "entropy": float(last_entropy.detach().cpu()),
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    policy_path = run_dir / "policy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "hidden_size": max(16, config.hidden_size),
            "config": asdict(config),
            "sim_config": dataclass_to_dict(sim_config),
        },
        policy_path,
    )
    training_summary = {
        "backend": "gpu_ppo",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": config.dtype,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "compile_policy_requested": policy.compile_requested,
        "compile_policy_enabled": policy.compile_enabled,
        "compile_policy_error": policy.compile_error,
        "timesteps_requested": config.timesteps,
        "timesteps_collected": updates * rollout_size,
        "updates": updates,
        "rollout_size": rollout_size,
        "elapsed_seconds": elapsed,
        "steps_per_second": (updates * rollout_size) / max(elapsed, 1e-9),
        "episode_count": len(episode_summaries),
        "policy_path": str(policy_path),
        "updates_log": update_summaries,
        "episodes": episode_summaries[-100:],
    }
    _write_json(run_dir / "training_summary.json", training_summary)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "config": asdict(config),
            "sim_config": dataclass_to_dict(sim_config),
            "training_summary": training_summary,
        },
    )
    if config.cpu_eval_episodes > 0:
        cpu_eval = evaluate_policy_on_cpu(
            model=model,
            sim_config=sim_config,
            device=device,
            output_dir=run_dir,
            episodes=config.cpu_eval_episodes,
            max_steps=config.cpu_eval_max_steps or config.max_steps,
            seed=config.seed + 50_000,
            start_speed_kph=config.start_speed_kph,
            start_progress_m=config.start_progress_m,
        )
        _write_json(run_dir / "cpu_eval_summary.json", cpu_eval)
    return run_dir


def evaluate_policy_on_cpu(
    *,
    model: ActorCritic,
    sim_config: SimConfig,
    device: torch.device,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    start_speed_kph: float,
    start_progress_m: float | None,
) -> dict[str, Any]:
    model.eval()
    summaries: list[dict[str, Any]] = []
    sim = MonzaSim(sim_config)
    for episode in range(episodes):
        options: dict[str, Any] = {
            "start_speed_kph": float(start_speed_kph),
            "collect_observation": True,
        }
        if start_progress_m is not None:
            options["start_progress_m"] = float(start_progress_m)
        obs, _info = sim.reset(seed=seed + episode, options=options)
        total_reward = 0.0
        telemetry_path = output_dir / f"cpu_eval_episode_{episode:03d}.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as file:
            for _step in range(max_steps):
                obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, _value = model(obs_tensor)
                action = torch.clamp(mean.squeeze(0), min=-1.0, max=1.0).detach().cpu().numpy()
                result = sim.step_continuous(action)
                total_reward += float(result.reward)
                file.write(json.dumps(asdict(result.telemetry), default=_json_default) + "\n")
                obs = result.observation
                if result.terminated or result.truncated:
                    break
        summaries.append(
            {
                "episode": episode,
                "return": total_reward,
                "steps": int(sim.state.elapsed_steps),
                "final_progress_m": float(sim.state.monotonic_progress_m),
                "final_speed_kph": float(sim.state.speed_mps * 3.6),
                "termination_reason": sim.termination_reason,
                "completed_lap": bool(sim.completed_lap),
                "valid_lap": bool(sim.valid_lap),
                "telemetry_path": str(telemetry_path),
            }
        )
    best_progress = max((summary["final_progress_m"] for summary in summaries), default=0.0)
    return {
        "backend": "cpu_replay",
        "episodes": summaries,
        "best_progress_m": best_progress,
        "completed_laps": sum(1 for summary in summaries if summary["completed_lap"]),
        "valid_laps": sum(1 for summary in summaries if summary["valid_lap"] and summary["completed_lap"]),
    }


def default_output_dir() -> Path:
    return ARTIFACTS_DIR / f"gpu-ppo-{time.strftime('%Y%m%d-%H%M%S')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run opt-in GPU-native PPO over the batched Monza simulator.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--continuous-action-scheme", default="drive_brake")
    parser.add_argument("--start-speed-kph", type=float, default=0.0)
    parser.add_argument("--start-progress-m", type=float)
    parser.add_argument("--target-progress-m", type=float, default=MONZA_LENGTH_METERS)
    parser.add_argument("--terminate-at-target", action="store_true")
    parser.add_argument("--ray-chunk-size", type=int, default=256)
    parser.add_argument("--collision-chunk-size", type=int, default=512)
    parser.add_argument("--compile-policy", action="store_true")
    parser.add_argument("--cpu-eval-episodes", type=int, default=1)
    parser.add_argument("--cpu-eval-max-steps", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = train_gpu_ppo(
        GpuPPOConfig(
            output_dir=args.output_dir or default_output_dir(),
            device=args.device,
            dtype=args.dtype,
            require_gpu=bool(args.require_gpu),
            seed=args.seed,
            timesteps=max(1, args.timesteps),
            n_envs=max(1, args.n_envs),
            n_steps=max(1, args.n_steps),
            batch_size=max(1, args.batch_size),
            n_epochs=max(1, args.n_epochs),
            learning_rate=max(1e-8, args.learning_rate),
            gamma=float(np.clip(args.gamma, 0.0, 1.0)),
            gae_lambda=float(np.clip(args.gae_lambda, 0.0, 1.0)),
            clip_range=max(0.0, args.clip_range),
            ent_coef=max(0.0, args.ent_coef),
            vf_coef=max(0.0, args.vf_coef),
            max_grad_norm=max(0.0, args.max_grad_norm),
            hidden_size=max(16, args.hidden_size),
            max_steps=max(1, args.max_steps),
            observation_profile=args.observation_profile,
            continuous_action_scheme=args.continuous_action_scheme,
            start_speed_kph=max(0.0, args.start_speed_kph),
            start_progress_m=args.start_progress_m,
            target_progress_m=args.target_progress_m,
            terminate_at_target=bool(args.terminate_at_target),
            ray_chunk_size=max(1, args.ray_chunk_size),
            collision_chunk_size=max(1, args.collision_chunk_size),
            compile_policy=bool(args.compile_policy),
            cpu_eval_episodes=max(0, args.cpu_eval_episodes),
            cpu_eval_max_steps=args.cpu_eval_max_steps,
        )
    )
    print(f"gpu_ppo_complete run={run_dir} summary={run_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
