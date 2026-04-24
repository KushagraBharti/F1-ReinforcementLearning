"""Torch-native PPO rollout and update helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from f1rl.torch_agent import ActorCriticPolicy, PPOHyperParams
from f1rl.torch_runtime import TERMINATION_REASONS, TorchSimBatch


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    telemetry_summary: dict[str, Any]


def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizon, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((num_envs,), dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(horizon)):
        if step == horizon - 1:
            next_non_terminal = 1.0 - dones[step].to(rewards.dtype)
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[step + 1].to(rewards.dtype)
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[step] = last_adv
    returns = advantages + values
    return advantages, returns


def collect_rollout(
    simulator: TorchSimBatch,
    policy: ActorCriticPolicy,
    ppo: PPOHyperParams,
    *,
    deterministic: bool = False,
) -> RolloutBatch:
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    checkpoint_events: list[dict[str, float | int]] = []

    obs = simulator.compute_observations()
    for _ in range(ppo.horizon):
        with torch.no_grad():
            action, log_prob, value = policy.act(obs, deterministic=deterministic)
        step = simulator.step(action)
        checkpoint_events.extend(step.checkpoint_events)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(step.rewards)
        dones.append(step.terminated | step.truncated)
        values.append(value)
        obs = step.observations

    with torch.no_grad():
        _, _, next_value = policy.act(obs, deterministic=True)

    obs_tensor = torch.stack(observations)
    action_tensor = torch.stack(actions)
    log_prob_tensor = torch.stack(log_probs)
    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    value_tensor = torch.stack(values)
    advantages, returns = _compute_gae(
        reward_tensor,
        done_tensor,
        value_tensor,
        next_value,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    snapshot = simulator.snapshot()
    summary = {
        "checkpoint_events": checkpoint_events,
        "avg_speed_kph": float(snapshot["speed_kph"].mean().item()),
        "avg_distance_travelled": float(snapshot["distance_travelled"].mean().item()),
        "moving_cars": int(
            (
                torch.abs(snapshot["speed_kph"]) >= simulator.telemetry_thresholds.moving_speed_threshold_kph
            ).sum().item()
        ),
        "alive_cars": int((~(snapshot["terminated"] | snapshot["truncated"])).sum().item()),
        "failure_histogram": _failure_histogram(snapshot["termination_code"]),
    }
    return RolloutBatch(
        observations=obs_tensor,
        actions=action_tensor,
        log_probs=log_prob_tensor,
        rewards=reward_tensor,
        dones=done_tensor,
        values=value_tensor,
        next_value=next_value,
        advantages=advantages,
        returns=returns,
        telemetry_summary=summary,
    )


def _failure_histogram(reason_codes: torch.Tensor) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for code in reason_codes.detach().cpu().tolist():
        reason = TERMINATION_REASONS[int(code)]
        histogram[reason] = histogram.get(reason, 0) + 1
    return histogram


def ppo_update(
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    ppo: PPOHyperParams,
) -> dict[str, float]:
    observations = rollout.observations.reshape(-1, rollout.observations.shape[-1])
    actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
    old_log_probs = rollout.log_probs.reshape(-1)
    advantages = rollout.advantages.reshape(-1)
    returns = rollout.returns.reshape(-1)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_batches = 0

    batch_size = observations.shape[0]
    minibatch_size = min(ppo.minibatch_size, batch_size)
    for _ in range(ppo.epochs):
        permutation = torch.randperm(batch_size, device=observations.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            batch_obs = observations[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]

            new_log_probs, entropy, values = policy.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (batch_returns - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + ppo.value_coef * value_loss - ppo.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_loss.item())
            total_batches += 1

    return {
        "policy_loss": total_policy_loss / max(total_batches, 1),
        "value_loss": total_value_loss / max(total_batches, 1),
        "entropy": total_entropy / max(total_batches, 1),
    }

