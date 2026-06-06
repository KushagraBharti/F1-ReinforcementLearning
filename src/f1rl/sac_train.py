# pyright: reportPrivateImportUsage=false
"""Project-native PyTorch SAC fine-tuning for learned Monza policies."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from f1rl.config import LEARNED_DIR, SimConfig
from f1rl.es_dataset import load_dataset
from f1rl.learned_policy import (
    PolicyNormalizer,
    QNetwork,
    SACActor,
    hard_update,
    load_policy_checkpoint,
    save_policy_checkpoint,
    soft_update,
)
from f1rl.policy_eval import evaluate_policy
from f1rl.policy_swarm_eval import run_policy_swarm_eval
from f1rl.sim import MonzaSim


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, *, seed: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, 3), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.source = np.zeros(capacity, dtype=np.int32)
        self.priority = np.ones(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        *,
        source: int,
        priority: np.ndarray | None = None,
    ) -> None:
        for idx in range(obs.shape[0]):
            row_priority = float(priority[idx]) if priority is not None else 1.0
            self.add(obs[idx], action[idx], float(reward[idx]), next_obs[idx], bool(done[idx]), source=source, priority=row_priority)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        source: int,
        priority: float = 1.0,
    ) -> None:
        self.obs[self.index] = obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.done[self.index] = float(done)
        self.source[self.index] = int(source)
        self.priority[self.index] = max(float(priority), 1e-6)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        priorities = self.priority[: self.size].astype(np.float64)
        probabilities = priorities / priorities.sum() if priorities.sum() > 0.0 else None
        indices = self.rng.choice(self.size, size=batch_size, replace=self.size < batch_size, p=probabilities)
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            "action": torch.as_tensor(self.action[indices], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(self.reward[indices, None], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[indices, None], dtype=torch.float32, device=device),
            "source": torch.as_tensor(self.source[indices, None], dtype=torch.int32, device=device),
        }

    def source_mix(self) -> dict[str, int]:
        values, counts = np.unique(self.source[: self.size], return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def _load_source_candidate_rows(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "source_candidates.jsonl"
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["source_candidate_id"])] = row
    return rows


def _dataset_transition_priorities(
    data_root: Path,
    arrays: dict[str, np.ndarray],
    *,
    fastest_weight: float,
    valid_time_power: float,
    late_progress_threshold_m: float,
    late_progress_weight: float,
    frontier_weight: float,
) -> np.ndarray:
    rows = _load_source_candidate_rows(data_root)
    if not rows:
        return np.ones(arrays["obs"].shape[0], dtype=np.float32)
    valid_times = [
        float(row["lap_time_s"])
        for row in rows.values()
        if str(row.get("termination_reason")) == "lap_complete" and row.get("lap_time_s") is not None
    ]
    fastest = min(valid_times) if valid_times else None
    source_weights: dict[int, float] = {}
    for source_id, row in rows.items():
        weight = 1.0
        lap_time = row.get("lap_time_s")
        if fastest is not None and lap_time is not None and str(row.get("termination_reason")) == "lap_complete":
            weight *= max((fastest / max(float(lap_time), 1e-6)) ** max(valid_time_power, 0.0), 0.05)
            if float(lap_time) <= fastest + 1e-6:
                weight *= max(fastest_weight, 0.0)
        elif str(row.get("source_bucket", "")).startswith("near_valid") or float(row.get("best_progress_m", 0.0) or 0.0) >= 5000.0:
            weight *= max(frontier_weight, 0.0)
        else:
            weight *= 0.35
        source_weights[source_id] = max(weight, 1e-6)
    source_ids = arrays["source_candidate_id"].astype(np.int64)
    priorities = np.asarray([source_weights.get(int(source_id), 1.0) for source_id in source_ids], dtype=np.float32)
    priorities[arrays["progress_m"].astype(np.float32) >= late_progress_threshold_m] *= max(late_progress_weight, 0.0)
    return np.maximum(priorities, 1e-6).astype(np.float32)


def _shape_rewards(
    arrays: dict[str, np.ndarray],
    *,
    time_penalty_per_step: float,
    speed_reward_scale: float,
    valid_finish_bonus: float,
) -> np.ndarray:
    reward = arrays["reward"].astype(np.float32).copy()
    if time_penalty_per_step != 0.0:
        reward -= np.float32(time_penalty_per_step)
    if speed_reward_scale != 0.0:
        reward += arrays["speed_kph"].astype(np.float32) / np.float32(400.0) * np.float32(speed_reward_scale)
    if valid_finish_bonus != 0.0:
        reward += arrays["valid_lap"].astype(np.float32) * arrays["done"].astype(np.float32) * np.float32(valid_finish_bonus)
    return reward.astype(np.float32)


def _shape_online_reward(
    base_reward: float,
    *,
    speed_kph: float,
    valid_lap: bool,
    done: bool,
    time_penalty_per_step: float,
    speed_reward_scale: float,
    valid_finish_bonus: float,
) -> float:
    reward = float(base_reward) - float(time_penalty_per_step)
    reward += (float(speed_kph) / 400.0) * float(speed_reward_scale)
    if done and valid_lap:
        reward += float(valid_finish_bonus)
    return reward


def _make_envs(
    n_envs: int,
    *,
    observation_profile: str,
    physics_model: str,
    max_steps: int,
    seed: int,
    start_position_noise_m: float,
    start_heading_noise_deg: float,
    start_speed_noise_kph: float,
) -> tuple[list[MonzaSim], list[np.ndarray]]:
    envs: list[MonzaSim] = []
    observations: list[np.ndarray] = []
    for idx in range(n_envs):
        sim = MonzaSim(
            SimConfig(
                max_steps=max_steps,
                action_mode="continuous",
                action_set="racing",
                observation_profile=observation_profile,
                physics_model=physics_model,
            )
        )
        obs, _ = sim.reset(
            seed=seed + idx,
            options={
                "start_speed_kph": 80.0,
                "start_progress_m": 0.0,
                "position_noise_m": start_position_noise_m,
                "heading_noise_deg": start_heading_noise_deg,
                "speed_noise_kph": start_speed_noise_kph,
            },
        )
        envs.append(sim)
        observations.append(obs)
    return envs, observations


def _reset_training_env(
    sim: MonzaSim,
    *,
    seed: int,
    start_position_noise_m: float,
    start_heading_noise_deg: float,
    start_speed_noise_kph: float,
) -> np.ndarray:
    obs, _ = sim.reset(
        seed=seed,
        options={
            "start_speed_kph": 80.0,
            "start_progress_m": 0.0,
            "position_noise_m": start_position_noise_m,
            "heading_noise_deg": start_heading_noise_deg,
            "speed_noise_kph": start_speed_noise_kph,
        },
    )
    return obs


def _sample_policy_actions(
    actor: SACActor,
    normalizer: PolicyNormalizer,
    observations: list[np.ndarray],
    *,
    device: torch.device,
    deterministic: bool,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    obs_t = normalizer.normalize_tensor(torch.as_tensor(obs, dtype=torch.float32, device=device))
    with torch.no_grad():
        if deterministic:
            actions = actor.deterministic_action(obs_t)
        else:
            actions, _, _ = actor.sample_action(obs_t)
    result = actions.detach().cpu().numpy().astype(np.float32)
    if noise_std > 0.0:
        result += rng.normal(0.0, noise_std, size=result.shape).astype(np.float32)
        result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
        result[:, 1] = np.clip(result[:, 1], 0.0, 1.0)
        result[:, 2] = np.clip(result[:, 2], -1.0, 1.0)
    return result


def _sac_update(
    *,
    replay: ReplayBuffer,
    actor: SACActor,
    q1: QNetwork,
    q2: QNetwork,
    q1_target: QNetwork,
    q2_target: QNetwork,
    normalizer: PolicyNormalizer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    tau: float,
    target_entropy: float,
    bc_loss_weight: float,
    freeze_alpha: bool,
    device: torch.device,
) -> dict[str, float]:
    batch = replay.sample(batch_size, device=device)
    obs = normalizer.normalize_tensor(batch["obs"])
    next_obs = normalizer.normalize_tensor(batch["next_obs"])
    action = batch["action"]
    reward = batch["reward"]
    done = batch["done"]
    alpha = log_alpha.exp()
    with torch.no_grad():
        next_action, next_log_prob, _ = actor.sample_action(next_obs)
        target_q = torch.min(q1_target(next_obs, next_action), q2_target(next_obs, next_action))
        q_backup = reward + gamma * (1.0 - done) * (target_q - alpha * next_log_prob)
    q1_value = q1(obs, action)
    q2_value = q2(obs, action)
    critic_loss = functional.mse_loss(q1_value, q_backup) + functional.mse_loss(q2_value, q_backup)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    sampled_action, log_prob, _ = actor.sample_action(obs)
    q_value = torch.min(q1(obs, sampled_action), q2(obs, sampled_action))
    deterministic_action = actor.deterministic_action(obs)
    es_source_weights = (batch["source"] == 0).to(torch.float32)
    if torch.any(es_source_weights > 0.0):
        es_source_weights = es_source_weights / es_source_weights.mean().clamp_min(1e-6)
        bc_loss = functional.smooth_l1_loss(deterministic_action, action, reduction="none")
        bc_loss = (bc_loss * es_source_weights).mean()
    else:
        bc_loss = deterministic_action.sum() * 0.0
    actor_loss = (alpha.detach() * log_prob - q_value).mean() + bc_loss_weight * bc_loss
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    if not freeze_alpha:
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()

    soft_update(q1_target, q1, tau)
    soft_update(q2_target, q2, tau)
    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "alpha_loss": float(alpha_loss.detach().cpu()),
        "alpha": float(log_alpha.exp().detach().cpu()),
        "q_mean": float(q_value.detach().mean().cpu()),
        "log_prob_mean": float(log_prob.detach().mean().cpu()),
        "bc_loss": float(bc_loss.detach().cpu()),
    }


def train_sac(
    *,
    dataset: Path,
    bc_checkpoint: Path | None,
    output_dir: Path,
    device: str,
    timesteps: int,
    n_envs: int,
    max_steps: int,
    batch_size: int,
    hidden_size: int,
    lr: float,
    gamma: float,
    tau: float,
    updates_per_step: float,
    bc_loss_weight: float,
    rollout_deterministic: bool,
    rollout_noise_std: float,
    initial_alpha: float,
    target_entropy: float,
    freeze_alpha: bool,
    control_mode: str,
    rollout_start_position_noise_m: float,
    rollout_start_heading_noise_deg: float,
    rollout_start_speed_noise_kph: float,
    es_fastest_weight: float,
    es_valid_time_power: float,
    es_late_progress_weight: float,
    es_frontier_weight: float,
    reward_time_penalty_per_step: float,
    reward_speed_scale: float,
    reward_valid_finish_bonus: float,
    eval_every: int,
    swarm_every: int,
    swarm_size: int,
    physics_model: str,
    seed: int,
) -> Path:
    data = load_dataset(dataset)
    dataset_physics_model = str(data.manifest.get("physics_model", "v1"))
    if physics_model == "dataset":
        physics_model = dataset_physics_model
    if physics_model not in {"v1", "v2"}:
        raise ValueError("physics_model must be 'dataset', 'v1', or 'v2'")
    if dataset_physics_model in {"v1", "v2"} and physics_model != dataset_physics_model:
        raise ValueError(
            f"SAC physics_model={physics_model!r} does not match dataset physics_model={dataset_physics_model!r}."
        )
    obs_dim = int(data.arrays["obs"].shape[1])
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    if bc_checkpoint is not None:
        actor, normalizer, bc_metadata = load_policy_checkpoint(bc_checkpoint, device=torch_device)
    else:
        actor = SACActor(obs_dim, hidden_sizes=(hidden_size, hidden_size), control_mode=control_mode).to(torch_device)
        normalizer = PolicyNormalizer.from_observations(data.arrays["obs"])
        bc_metadata = {}
    q1 = QNetwork(obs_dim, hidden_sizes=(hidden_size, hidden_size)).to(torch_device)
    q2 = QNetwork(obs_dim, hidden_sizes=(hidden_size, hidden_size)).to(torch_device)
    q1_target = QNetwork(obs_dim, hidden_sizes=(hidden_size, hidden_size)).to(torch_device)
    q2_target = QNetwork(obs_dim, hidden_sizes=(hidden_size, hidden_size)).to(torch_device)
    hard_update(q1_target, q1)
    hard_update(q2_target, q2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=lr)
    log_alpha = torch.full(
        (),
        math.log(max(float(initial_alpha), 1e-8)),
        dtype=torch.float32,
        device=torch_device,
        requires_grad=True,
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    capacity = max(int(data.arrays["obs"].shape[0] + timesteps * max(n_envs, 1) + 4096), batch_size * 8)
    replay = ReplayBuffer(capacity, obs_dim, seed=seed)
    shaped_es_reward = _shape_rewards(
        data.arrays,
        time_penalty_per_step=reward_time_penalty_per_step,
        speed_reward_scale=reward_speed_scale,
        valid_finish_bonus=reward_valid_finish_bonus,
    )
    es_priorities = _dataset_transition_priorities(
        data.root,
        data.arrays,
        fastest_weight=es_fastest_weight,
        valid_time_power=es_valid_time_power,
        late_progress_threshold_m=4800.0,
        late_progress_weight=es_late_progress_weight,
        frontier_weight=es_frontier_weight,
    )
    replay.add_batch(
        data.arrays["obs"],
        data.arrays["action"],
        shaped_es_reward,
        data.arrays["next_obs"],
        data.arrays["done"],
        source=0,
        priority=es_priorities,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": str(data.root),
        "dataset_manifest": data.manifest,
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint is not None else None,
        "bc_metadata": bc_metadata,
        "observation_profile": data.manifest.get("observation_profile"),
        "observation_dim": obs_dim,
        "observation_feature_schema": data.manifest.get("observation_feature_schema"),
        "physics_model": physics_model,
        "physics_version": data.manifest.get("physics_version"),
        "physics_calibration_id": data.manifest.get("physics_calibration_id"),
        "device": str(torch_device),
        "timesteps": timesteps,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "lr": lr,
        "gamma": gamma,
        "tau": tau,
        "updates_per_step": updates_per_step,
        "bc_loss_weight": bc_loss_weight,
        "rollout_deterministic": rollout_deterministic,
        "rollout_noise_std": rollout_noise_std,
        "initial_alpha": initial_alpha,
        "target_entropy": target_entropy,
        "freeze_alpha": freeze_alpha,
        "control_mode": actor.control_mode,
        "rollout_start_position_noise_m": rollout_start_position_noise_m,
        "rollout_start_heading_noise_deg": rollout_start_heading_noise_deg,
        "rollout_start_speed_noise_kph": rollout_start_speed_noise_kph,
        "es_replay_priority": {
            "fastest_weight": es_fastest_weight,
            "valid_time_power": es_valid_time_power,
            "late_progress_weight": es_late_progress_weight,
            "frontier_weight": es_frontier_weight,
        },
        "reward_shaping": {
            "time_penalty_per_step": reward_time_penalty_per_step,
            "speed_reward_scale": reward_speed_scale,
            "valid_finish_bonus": reward_valid_finish_bonus,
        },
        "eval_every": eval_every,
        "swarm_every": swarm_every,
        "swarm_size": swarm_size,
        "rollout_backend": "cpu_monzasim",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (output_dir / "replay_buffer_manifest.json").write_text(
        json.dumps(
            {
                "capacity": capacity,
                "initial_es_transitions": int(data.arrays["obs"].shape[0]),
                "source_mix": replay.source_mix(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _checkpoint_metadata(
        *,
        stage: str,
        step: int | None = None,
        updates: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "stage": stage,
            "config": config,
            "physics_model": config["physics_model"],
            "physics_version": config["physics_version"],
            "physics_calibration_id": config["physics_calibration_id"],
        }
        if step is not None:
            metadata["step"] = step
        if updates is not None:
            metadata["updates"] = updates
        if metrics is not None:
            metadata["metrics"] = metrics
        return metadata

    envs, observations = _make_envs(
        n_envs,
        observation_profile=data.manifest["observation_profile"],
        physics_model=physics_model,
        max_steps=max_steps,
        seed=seed,
        start_position_noise_m=rollout_start_position_noise_m,
        start_heading_noise_deg=rollout_start_heading_noise_deg,
        start_speed_noise_kph=rollout_start_speed_noise_kph,
    )
    episode_returns = [0.0 for _ in envs]
    episode_progress = [0.0 for _ in envs]
    recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
    metrics_path = output_dir / "metrics.jsonl"
    best_lap = math.inf
    best_policy = output_dir / "best_policy.pt"
    updates = 0
    last_metrics: dict[str, float] = {}
    rollout_rng = np.random.default_rng(seed + 900_000)

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        initial_checkpoint_path = output_dir / "checkpoints" / "sac_step_00000000.pt"
        initial_metadata = _checkpoint_metadata(stage="sac", step=0, updates=updates, metrics=last_metrics)
        save_policy_checkpoint(initial_checkpoint_path, actor=actor, normalizer=normalizer, metadata=initial_metadata)
        initial_eval_dir = output_dir / "cpu_eval" / "step_00000000"
        initial_eval_summary = evaluate_policy(
            policy=initial_checkpoint_path,
            output_dir=initial_eval_dir,
            episodes=1,
            deterministic=True,
            normal_start=True,
            observation_profile=data.manifest["observation_profile"],
            physics_model=physics_model,
            write_telemetry="gzip",
            max_steps=max_steps,
            seed=seed + 49_000,
            device=device,
        )
        initial_fastest = initial_eval_summary.get("fastest_valid_lap_s")
        if initial_fastest is not None and float(initial_fastest) < best_lap:
            best_lap = float(initial_fastest)
            save_policy_checkpoint(
                best_policy,
                actor=actor,
                normalizer=normalizer,
                metadata=initial_metadata | {"eval_summary": initial_eval_summary},
            )
        metrics_file.write(
            json.dumps(
                {
                    "step": 0,
                    "updates": updates,
                    "replay_size": replay.size,
                    "mean_episode_return": 0.0,
                    "mean_episode_progress_m": 0.0,
                    "eval_fastest_valid_lap_s": initial_fastest,
                    "eval_valid_lap_count": initial_eval_summary.get("valid_lap_count"),
                    **last_metrics,
                }
            )
            + "\n"
        )
        metrics_file.flush()

        for step in range(0, timesteps, max(1, n_envs)):
            actor.eval()
            actions = _sample_policy_actions(
                actor,
                normalizer,
                observations,
                device=torch_device,
                deterministic=rollout_deterministic,
                noise_std=rollout_noise_std,
                rng=rollout_rng,
            )
            for env_index, sim in enumerate(envs):
                obs = observations[env_index]
                action = actions[env_index]
                result = sim.step_controls(throttle=float(action[0]), brake=float(action[1]), steer=float(action[2]), action_id=-30)
                done = bool(result.terminated or result.truncated)
                shaped_reward = _shape_online_reward(
                    float(result.reward),
                    speed_kph=float(result.telemetry.speed_kph),
                    valid_lap=bool(result.telemetry.termination_reason == "lap_complete" and result.telemetry.valid_lap),
                    done=done,
                    time_penalty_per_step=reward_time_penalty_per_step,
                    speed_reward_scale=reward_speed_scale,
                    valid_finish_bonus=reward_valid_finish_bonus,
                )
                replay.add(obs, action, shaped_reward, result.observation, done, source=1, priority=1.0)
                episode_returns[env_index] += shaped_reward
                episode_progress[env_index] = max(episode_progress[env_index], float(result.telemetry.monotonic_progress_m))
                observations[env_index] = result.observation
                if done:
                    recent_episodes.append(
                        {
                            "return": episode_returns[env_index],
                            "best_progress_m": episode_progress[env_index],
                            "termination_reason": result.telemetry.termination_reason,
                            "valid_lap": bool(result.telemetry.termination_reason == "lap_complete" and result.telemetry.valid_lap),
                            "lap_time_s": float(result.telemetry.sim_time_s)
                            if result.telemetry.termination_reason == "lap_complete"
                            else None,
                        }
                    )
                    observations[env_index] = _reset_training_env(
                        sim,
                        seed=seed + step + env_index + 1,
                        start_position_noise_m=rollout_start_position_noise_m,
                        start_heading_noise_deg=rollout_start_heading_noise_deg,
                        start_speed_noise_kph=rollout_start_speed_noise_kph,
                    )
                    episode_returns[env_index] = 0.0
                    episode_progress[env_index] = 0.0
            actor.train()
            update_count = max(1, int(round(n_envs * updates_per_step)))
            for _ in range(update_count):
                if replay.size < batch_size:
                    break
                last_metrics = _sac_update(
                    replay=replay,
                    actor=actor,
                    q1=q1,
                    q2=q2,
                    q1_target=q1_target,
                    q2_target=q2_target,
                    normalizer=normalizer,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    log_alpha=log_alpha,
                    alpha_optimizer=alpha_optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                    target_entropy=target_entropy,
                    bc_loss_weight=bc_loss_weight,
                    freeze_alpha=freeze_alpha,
                    device=torch_device,
                )
                updates += 1
            if (step + n_envs) % max(eval_every, 1) < n_envs or step + n_envs >= timesteps:
                checkpoint_path = output_dir / "checkpoints" / f"sac_step_{step + n_envs:08d}.pt"
                metadata = _checkpoint_metadata(
                    stage="sac",
                    step=step + n_envs,
                    updates=updates,
                    metrics=last_metrics,
                )
                save_policy_checkpoint(checkpoint_path, actor=actor, normalizer=normalizer, metadata=metadata)
                eval_dir = output_dir / "cpu_eval" / f"step_{step + n_envs:08d}"
                eval_summary = evaluate_policy(
                    policy=checkpoint_path,
                    output_dir=eval_dir,
                    episodes=1,
                    deterministic=True,
                    normal_start=True,
                    observation_profile=data.manifest["observation_profile"],
                    physics_model=physics_model,
                    write_telemetry="gzip",
                    max_steps=max_steps,
                    seed=seed + 50_000 + step,
                    device=device,
                )
                fastest = eval_summary.get("fastest_valid_lap_s")
                if fastest is not None and float(fastest) < best_lap:
                    best_lap = float(fastest)
                    save_policy_checkpoint(best_policy, actor=actor, normalizer=normalizer, metadata=metadata | {"eval_summary": eval_summary})
                if swarm_every > 0 and swarm_size > 0 and ((step + n_envs) % swarm_every < n_envs or step + n_envs >= timesteps):
                    run_policy_swarm_eval(
                        policy_dir=checkpoint_path,
                        output_dir=output_dir / "checkpoint_swarms" / f"step_{step + n_envs:08d}",
                        swarm_size=swarm_size,
                        checkpoints="best",
                        observation_profile=data.manifest["observation_profile"],
                        physics_model=physics_model,
                        deterministic=True,
                        max_steps=max_steps,
                        seed=seed + 700_000 + step,
                        device=device,
                        full_telemetry_limit=swarm_size,
                        start_position_noise_m=0.0,
                        start_heading_noise_deg=0.0,
                        start_speed_noise_kph=0.0,
                    )
                mean_return = float(np.mean([row["return"] for row in recent_episodes])) if recent_episodes else 0.0
                mean_progress = float(np.mean([row["best_progress_m"] for row in recent_episodes])) if recent_episodes else 0.0
                row = {
                    "step": step + n_envs,
                    "updates": updates,
                    "replay_size": replay.size,
                    "mean_episode_return": mean_return,
                    "mean_episode_progress_m": mean_progress,
                    "eval_fastest_valid_lap_s": fastest,
                    "eval_valid_lap_count": eval_summary.get("valid_lap_count"),
                    **last_metrics,
                }
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
    final_policy = output_dir / "final_policy.pt"
    save_policy_checkpoint(final_policy, actor=actor, normalizer=normalizer, metadata=_checkpoint_metadata(stage="sac"))
    if not best_policy.exists():
        save_policy_checkpoint(best_policy, actor=actor, normalizer=normalizer, metadata=_checkpoint_metadata(stage="sac"))
    print(f"sac_train_complete best_policy={best_policy} best_lap_s={best_lap if math.isfinite(best_lap) else None}")
    return best_policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC from BC initialization and ES replay prefill.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bc-checkpoint")
    parser.add_argument("--output-dir", default=str(LEARNED_DIR / "v1-sac"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    parser.add_argument("--bc-loss-weight", type=float, default=0.0)
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--rollout-noise-std", type=float, default=0.0)
    parser.add_argument("--initial-alpha", type=float, default=1.0)
    parser.add_argument("--target-entropy", type=float, default=-3.0)
    parser.add_argument("--freeze-alpha", action="store_true")
    parser.add_argument("--control-mode", choices=("independent", "dominance"), default="independent")
    parser.add_argument("--rollout-start-position-noise-m", type=float, default=0.0)
    parser.add_argument("--rollout-start-heading-noise-deg", type=float, default=0.0)
    parser.add_argument("--rollout-start-speed-noise-kph", type=float, default=0.0)
    parser.add_argument("--es-fastest-weight", type=float, default=1.0)
    parser.add_argument("--es-valid-time-power", type=float, default=0.0)
    parser.add_argument("--es-late-progress-weight", type=float, default=1.0)
    parser.add_argument("--es-frontier-weight", type=float, default=1.0)
    parser.add_argument("--reward-time-penalty-per-step", type=float, default=0.0)
    parser.add_argument("--reward-speed-scale", type=float, default=0.0)
    parser.add_argument("--reward-valid-finish-bonus", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=512)
    parser.add_argument("--swarm-every", type=int, default=0)
    parser.add_argument("--swarm-size", type=int, default=0)
    parser.add_argument("--physics-model", choices=("dataset", "v1", "v2"), default="dataset")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_sac(
        dataset=Path(args.dataset),
        bc_checkpoint=Path(args.bc_checkpoint) if args.bc_checkpoint else None,
        output_dir=Path(args.output_dir),
        device=args.device,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        updates_per_step=args.updates_per_step,
        bc_loss_weight=args.bc_loss_weight,
        rollout_deterministic=args.deterministic_rollout,
        rollout_noise_std=args.rollout_noise_std,
        initial_alpha=args.initial_alpha,
        target_entropy=args.target_entropy,
        freeze_alpha=args.freeze_alpha,
        control_mode=args.control_mode,
        rollout_start_position_noise_m=args.rollout_start_position_noise_m,
        rollout_start_heading_noise_deg=args.rollout_start_heading_noise_deg,
        rollout_start_speed_noise_kph=args.rollout_start_speed_noise_kph,
        es_fastest_weight=args.es_fastest_weight,
        es_valid_time_power=args.es_valid_time_power,
        es_late_progress_weight=args.es_late_progress_weight,
        es_frontier_weight=args.es_frontier_weight,
        reward_time_penalty_per_step=args.reward_time_penalty_per_step,
        reward_speed_scale=args.reward_speed_scale,
        reward_valid_finish_bonus=args.reward_valid_finish_bonus,
        eval_every=args.eval_every,
        swarm_every=args.swarm_every,
        swarm_size=args.swarm_size,
        physics_model=args.physics_model,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
