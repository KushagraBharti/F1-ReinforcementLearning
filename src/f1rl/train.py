"""Stable-Baselines3 PPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import (
    ACTION_SETS,
    ARTIFACTS_DIR,
    CONTINUOUS_ACTION_SCHEMES,
    OBSERVATION_PROFILES,
    ActionSpec,
    actions_for_action_set,
    build_sim_config,
    dataclass_to_dict,
    scaffold_rewards_enabled,
    training_assists_enabled,
)
from f1rl.curriculum import (
    DEFAULT_SEGMENT_STAGES,
    PREFIX_START_STAGES,
    CurriculumConfig,
    CurriculumSampler,
    CurriculumStage,
    chicane_skill_stages,
)
from f1rl.env import MonzaEnv
from f1rl.hardware import compute_policy, torch_device
from f1rl.sim import MonzaSim


def _action_to_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _make_env(
    max_steps: int,
    seed: int,
    curriculum: CurriculumConfig | None = None,
    reward_overrides: dict[str, float | None] | None = None,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    assist_overrides: dict[str, bool | float | None] | None = None,
):
    def factory() -> MonzaEnv:
        env = MonzaEnv(
            build_sim_config(
                max_steps=max_steps,
                action_mode=action_mode,
                action_set=action_set,
                continuous_action_scheme=continuous_action_scheme,
                observation_profile=observation_profile,
                launch_guard_progress_m=launch_guard_progress_m,
                launch_guard_min_speed_kph=launch_guard_min_speed_kph,
                launch_guard_throttle=launch_guard_throttle,
                reward_overrides=reward_overrides,
                assist_overrides=assist_overrides,
            ),
            curriculum=curriculum,
        )
        env.reset(seed=seed)
        return env

    return factory


def _load_vecnormalize_with_reward_fallback(
    vec_normalize_cls: Any,
    vec_normalize_path: Path,
    env: Any,
    *,
    normalize_reward: bool,
    normalize_reward_gamma: float,
    normalize_reward_clip: float,
) -> tuple[Any, str]:
    try:
        return vec_normalize_cls.load(str(vec_normalize_path), env), "exact"
    except AssertionError:
        with vec_normalize_path.open("rb") as file:
            saved_vec_normalize = pickle.load(file)
        if getattr(saved_vec_normalize, "norm_obs", True):
            raise
        old_shape = getattr(getattr(saved_vec_normalize, "observation_space", None), "shape", None)
        new_shape = getattr(getattr(env, "observation_space", None), "shape", None)
        if old_shape == new_shape:
            raise
        vec_normalize = vec_normalize_cls(
            env,
            norm_obs=False,
            norm_reward=normalize_reward,
            clip_reward=normalize_reward_clip,
            gamma=normalize_reward_gamma,
        )
        vec_normalize.ret_rms = saved_vec_normalize.ret_rms
        vec_normalize.epsilon = getattr(saved_vec_normalize, "epsilon", vec_normalize.epsilon)
        vec_normalize.old_reward = np.array([])
        return vec_normalize, f"reward_stats_only_observation_shape_changed:{old_shape}->{new_shape}"


def _build_curriculum_config(
    *,
    mode: str,
    stage_count: int | None,
    stage_start_index: int,
    promotion_resets: int,
    normal_start_probability: float,
    preset: str = "default",
    focus_start_progress_m: float | None = None,
    focus_window_m: float = 0.0,
    focus_segment_length_m: float = 500.0,
    focus_min_speed_kph: float = 120.0,
    focus_max_speed_kph: float = 220.0,
    focus_position_noise_m: float = 0.5,
    focus_heading_noise_deg: float = 2.0,
    focus_speed_noise_kph: float = 5.0,
    state_library_path: Path | None = None,
    state_library_segment_length_m: float = 900.0,
    state_library_position_noise_m: float = 0.0,
    state_library_heading_noise_deg: float = 0.0,
    state_library_speed_noise_kph: float = 0.0,
    chicane: str = "rettifilo",
) -> CurriculumConfig:
    if state_library_path is not None:
        if mode != "segments":
            raise ValueError("--curriculum-state-library requires --curriculum segments.")
        if preset == "chicane-skill":
            stages = tuple(
                CurriculumStage(
                    stage.name,
                    stage.segment_length_m,
                    stage.min_speed_kph,
                    stage.max_speed_kph,
                    state_library_position_noise_m,
                    state_library_heading_noise_deg,
                    state_library_speed_noise_kph,
                    stage.start_min_progress_m,
                    stage.start_max_progress_m,
                    stage.target_progress_m,
                    stage.target_max_speed_kph,
                )
                for stage in chicane_skill_stages(chicane)
            )
            if stage_start_index < 0 or stage_start_index >= len(stages):
                raise ValueError(
                    f"--curriculum-start-stage-index must be between 0 and {len(stages) - 1}."
                )
            stages = stages[stage_start_index:]
            if stage_count is not None:
                if stage_count < 1:
                    raise ValueError("--curriculum-stage-count must be at least 1.")
                stages = stages[: min(stage_count, len(stages))]
        else:
            stages = (
                CurriculumStage(
                    "state-library",
                    state_library_segment_length_m,
                    0.0,
                    0.0,
                    state_library_position_noise_m,
                    state_library_heading_noise_deg,
                    state_library_speed_noise_kph,
                ),
            )
        return CurriculumConfig(
            mode=mode,
            stages=stages,
            promotion_resets=promotion_resets,
            normal_start_probability=normal_start_probability,
            start_mode="state_library",
            state_library_path=state_library_path,
        )
    if mode != "segments":
        return CurriculumConfig(mode=mode)
    if focus_start_progress_m is not None:
        stages = (
            CurriculumStage(
                "focus-window",
                focus_segment_length_m,
                focus_min_speed_kph,
                focus_max_speed_kph,
                focus_position_noise_m,
                focus_heading_noise_deg,
                focus_speed_noise_kph,
            ),
        )
        return CurriculumConfig(
            mode=mode,
            stages=stages,
            promotion_resets=promotion_resets,
            normal_start_probability=normal_start_probability,
            focus_start_progress_m=focus_start_progress_m,
            focus_window_m=focus_window_m,
            start_mode="focus",
        )
    if preset == "chicane-skill":
        source_stages = chicane_skill_stages(chicane)
        start_mode = "progress_range"
    elif preset == "prefix-start":
        source_stages = PREFIX_START_STAGES
        start_mode = "normal"
    else:
        source_stages = DEFAULT_SEGMENT_STAGES
        start_mode = "random_checkpoint"
    if stage_start_index < 0 or stage_start_index >= len(source_stages):
        raise ValueError(
            f"--curriculum-start-stage-index must be between 0 and {len(source_stages) - 1}."
        )
    stages = source_stages[stage_start_index:]
    if stage_count is not None:
        if stage_count < 1:
            raise ValueError("--curriculum-stage-count must be at least 1.")
        stages = stages[: min(stage_count, len(stages))]
    return CurriculumConfig(
        mode=mode,
        stages=stages,
        promotion_resets=promotion_resets,
        normal_start_probability=normal_start_probability,
        start_mode=start_mode,
    )


def _segment_eval_curriculum_config(config: CurriculumConfig) -> CurriculumConfig:
    """Use the same segment stages for diagnostics, but never sample normal starts."""
    return CurriculumConfig(
        mode=config.mode,
        stages=config.stages,
        promotion_resets=config.promotion_resets,
        normal_start_probability=0.0,
        focus_start_progress_m=config.focus_start_progress_m,
        focus_window_m=config.focus_window_m,
        start_mode=config.start_mode,
        state_library_path=config.state_library_path,
    )


def _full_lap_selection_score(summary: dict[str, Any]) -> float:
    """Rank checkpoints by the real goal: valid full-lap completion, then normal-start progress."""
    full_lap_episodes = summary.get("full_lap_episodes") or summary.get("episodes") or []
    completed_times = [
        float(row.get("elapsed_time_s", 9999.0))
        for row in full_lap_episodes
        if row.get("completed_lap") and row.get("valid_lap")
    ]
    completion_rate = float(summary.get("completion_rate", 0.0))
    mean_progress = float(summary.get("mean_best_progress_m", 0.0))
    score = completion_rate * 1_000_000.0 + mean_progress
    if completed_times:
        mean_time_s = sum(completed_times) / len(completed_times)
        score += max(0.0, 1_000.0 - mean_time_s) * 1_000.0
    score += float(summary.get("segment_completion_rate", 0.0))
    score += float(summary.get("mean_segment_progress_delta_m", 0.0)) * 0.001
    return score


def _reward_overrides_from_args(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "progress_scale": args.reward_progress_scale,
        "finish_bonus": args.reward_finish_bonus,
        "collision_penalty": args.reward_collision_penalty,
        "off_track_penalty": args.reward_off_track_penalty,
        "no_progress_penalty": args.reward_no_progress_penalty,
        "lateral_deadzone_m": args.reward_lateral_deadzone_m,
        "lateral_penalty_scale": args.reward_lateral_penalty_scale,
        "track_limit_safe_ray_m": args.reward_track_limit_safe_ray_m,
        "track_limit_penalty_scale": args.reward_track_limit_penalty_scale,
        "heading_deadzone_deg": args.reward_heading_deadzone_deg,
        "heading_penalty_scale": args.reward_heading_penalty_scale,
        "speed_target_min_kph": args.reward_speed_target_min_kph,
        "speed_target_max_kph": args.reward_speed_target_max_kph,
        "speed_target_heading_scale": args.reward_speed_target_heading_scale,
        "speed_target_deadzone_kph": args.reward_speed_target_deadzone_kph,
        "speed_target_penalty_scale": args.reward_speed_target_penalty_scale,
        "overspeed_throttle_penalty_scale": args.reward_overspeed_throttle_penalty_scale,
        "overspeed_brake_reward_scale": args.reward_overspeed_brake_reward_scale,
        "steering_target_deadzone": args.reward_steering_target_deadzone,
        "steering_target_penalty_scale": args.reward_steering_target_penalty_scale,
        "smoothness_penalty": args.reward_smoothness_penalty,
        "scaffold_scale": args.reward_scaffold_scale,
        "scaffold_brake_reward_scale": args.reward_scaffold_brake_reward_scale,
        "scaffold_no_throttle_penalty_scale": args.reward_scaffold_no_throttle_penalty_scale,
        "scaffold_turn_in_speed_penalty_scale": args.reward_scaffold_turn_in_speed_penalty_scale,
        "scaffold_apex_clean_reward_scale": args.reward_scaffold_apex_clean_reward_scale,
        "scaffold_exit_alignment_reward_scale": args.reward_scaffold_exit_alignment_reward_scale,
        "scaffold_exit_speed_reward_scale": args.reward_scaffold_exit_speed_reward_scale,
    }


def _assist_overrides_from_args(args: argparse.Namespace) -> dict[str, bool | float | None]:
    return {
        "enabled": args.assist_enabled,
        "overspeed_turn_in_terminate": args.assist_overspeed_turn_in_terminate,
        "overspeed_turn_in_margin_kph": args.assist_overspeed_turn_in_margin_kph,
        "overspeed_turn_in_penalty": args.assist_overspeed_turn_in_penalty,
        "throttle_brake_demand_penalty_scale": args.assist_throttle_brake_demand_penalty_scale,
        "no_brake_penalty": args.assist_no_brake_penalty,
        "no_brake_min_brake": args.assist_no_brake_min_brake,
        "virtual_corridor_m": args.assist_virtual_corridor_m,
        "virtual_corridor_penalty": args.assist_virtual_corridor_penalty,
        "virtual_corridor_terminate": args.assist_virtual_corridor_terminate,
    }


def _infer_discrete_action_set_from_model(model: Any) -> str | None:
    action_space = getattr(model, "action_space", None)
    action_count = getattr(action_space, "n", None)
    if action_count is None:
        return None
    matches = [name for name, actions in ACTION_SETS.items() if len(actions) == int(action_count)]
    return matches[0] if len(matches) == 1 else None


def _closest_source_action(target_action: ActionSpec, source_actions: tuple[ActionSpec, ...]) -> int:
    _, target_throttle, target_brake, target_steer = target_action
    best_index = 0
    best_score = float("inf")
    for index, (_, throttle, brake, steer) in enumerate(source_actions):
        score = (
            (target_throttle - throttle) ** 2
            + (target_brake - brake) ** 2
            + 0.5 * (target_steer - steer) ** 2
        )
        if score < best_score:
            best_index = index
            best_score = score
    return best_index


def _copy_discrete_action_head(
    *,
    key: str,
    target_tensor: Any,
    source_tensor: Any,
    source_action_set: str | None,
    target_action_set: str,
    new_action_bias_penalty: float = -4.0,
) -> tuple[Any, list[dict[str, Any]]] | None:
    if key not in {"action_net.weight", "action_net.bias"}:
        return None
    if source_action_set is None:
        return None
    source_actions = actions_for_action_set(source_action_set)
    target_actions = actions_for_action_set(target_action_set)
    if source_tensor.shape[0] != len(source_actions) or target_tensor.shape[0] != len(target_actions):
        return None
    if key == "action_net.weight" and not (
        source_tensor.ndim == 2
        and target_tensor.ndim == 2
        and source_tensor.shape[1] == target_tensor.shape[1]
    ):
        return None
    if key == "action_net.bias" and not (source_tensor.ndim == 1 and target_tensor.ndim == 1):
        return None

    exact_index_by_action = {action: index for index, action in enumerate(source_actions)}
    expanded = target_tensor.detach().clone()
    row_report: list[dict[str, Any]] = []
    for target_index, target_action in enumerate(target_actions):
        exact_source_index = exact_index_by_action.get(target_action)
        source_index = (
            exact_source_index
            if exact_source_index is not None
            else _closest_source_action(target_action, source_actions)
        )
        if key == "action_net.weight":
            expanded[target_index, :] = source_tensor[source_index, :].detach().to(target_tensor.device)
        else:
            bias_offset = 0.0 if exact_source_index is not None else new_action_bias_penalty
            expanded[target_index] = source_tensor[source_index].detach().to(target_tensor.device) + bias_offset
        row_report.append(
            {
                "target_index": target_index,
                "target_action": target_action[0],
                "source_index": int(source_index),
                "source_action": source_actions[source_index][0],
                "mode": "exact" if exact_source_index is not None else "nearest_with_bias_penalty",
                "bias_penalty": 0.0 if exact_source_index is not None else new_action_bias_penalty,
            }
        )
    return expanded, row_report


def _copy_compatible_policy_weights(
    target_model: Any,
    source_model: Any,
    *,
    source_action_set: str | None = None,
    target_action_set: str = "legacy",
) -> list[dict[str, Any]]:
    """Copy matching SB3 MlpPolicy tensors, expanding observation/action heads when needed."""
    target_state = target_model.policy.state_dict()
    source_state = source_model.policy.state_dict()
    resolved_source_action_set = source_action_set or _infer_discrete_action_set_from_model(source_model)
    report: list[dict[str, Any]] = []
    updates: dict[str, Any] = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            updates[key] = source_tensor.detach().to(target_tensor.device).clone()
            report.append(
                {
                    "key": key,
                    "mode": "copied",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        action_head_copy = _copy_discrete_action_head(
            key=key,
            target_tensor=target_tensor,
            source_tensor=source_tensor,
            source_action_set=resolved_source_action_set,
            target_action_set=target_action_set,
        )
        if action_head_copy is not None:
            expanded_action_head, row_report = action_head_copy
            updates[key] = expanded_action_head
            report.append(
                {
                    "key": key,
                    "mode": "expanded_discrete_action_head",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                    "source_action_set": resolved_source_action_set,
                    "target_action_set": target_action_set,
                    "rows": row_report,
                }
            )
            continue
        if (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] <= target_tensor.shape[1]
        ):
            expanded = target_tensor.detach().clone()
            expanded.zero_()
            expanded[:, : source_tensor.shape[1]] = source_tensor.detach().to(target_tensor.device)
            updates[key] = expanded
            report.append(
                {
                    "key": key,
                    "mode": "expanded_input",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                    "copied_input_columns": int(source_tensor.shape[1]),
                }
            )
    target_state.update(updates)
    target_model.policy.load_state_dict(target_state)
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _episode_metrics(*, steps: list[dict[str, Any]], policy: str, episode: int, seed: int) -> dict[str, Any]:
    if not steps:
        return {
            "policy": policy,
            "episode": episode,
            "seed": seed,
            "steps": 0,
            "completed_lap": False,
            "valid_lap": False,
            "segment_complete": False,
            "termination_reason": "empty",
            "total_reward": 0.0,
            "final_progress_m": 0.0,
            "best_progress_m": 0.0,
        }
    reward_totals: dict[str, float] = {}
    for step in steps:
        for key, value in step["reward_components"].items():
            reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
    final = steps[-1]
    best_progress_m = float(max(step["monotonic_progress_m"] for step in steps))
    start_progress_m = float(steps[0]["monotonic_progress_m"])
    segment_target_progress_m = final.get("segment_target_progress_m")
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "steps": len(steps),
        "completed_lap": bool(final["termination_reason"] == "lap_complete"),
        "valid_lap": bool(final.get("valid_lap", False)),
        "segment_complete": bool(final.get("segment_complete", False)),
        "termination_reason": final["termination_reason"],
        "elapsed_time_s": float(final["sim_time_s"]),
        "total_reward": float(sum(step["reward_total"] for step in steps)),
        "reward_totals": reward_totals,
        "final_progress_m": float(final["monotonic_progress_m"]),
        "best_progress_m": best_progress_m,
        "segment_progress_delta_m": max(0.0, best_progress_m - start_progress_m),
        "segment_target_progress_m": segment_target_progress_m,
        "segment_remaining_m": (
            max(0.0, float(segment_target_progress_m) - best_progress_m)
            if segment_target_progress_m is not None
            else None
        ),
        "checkpoints_passed": int(final.get("checkpoints_passed", final["checkpoint_index"])),
        "missed_checkpoint_count": int(final.get("missed_checkpoint_count", 0)),
        "avg_speed_kph": float(sum(step["speed_kph"] for step in steps) / len(steps)),
        "max_speed_kph": float(max(step["speed_kph"] for step in steps)),
        "crashed": bool(final["collided"]),
        "off_track": bool(final["off_track"]),
        "curriculum_stage": final.get("curriculum_stage"),
    }


def _curriculum_stage_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        stage = str(row.get("curriculum_stage") or "normal")
        grouped.setdefault(stage, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for stage, rows in grouped.items():
        completion_count = sum(1 for row in rows if row.get("segment_complete"))
        reasons = sorted({str(row.get("termination_reason") or "unknown") for row in rows})
        summary[stage] = {
            "episodes": len(rows),
            "segment_completion_rate": completion_count / max(len(rows), 1),
            "mean_segment_progress_delta_m": sum(float(row.get("segment_progress_delta_m", 0.0)) for row in rows)
            / max(len(rows), 1),
            "mean_best_progress_m": sum(float(row.get("best_progress_m", 0.0)) for row in rows) / max(len(rows), 1),
            "termination_reasons": {
                str(reason): sum(1 for row in rows if row.get("termination_reason") == reason)
                for reason in reasons
            },
        }
    return summary


def run_model_rollouts(
    model: Any,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    telemetry_dir: Path | None = None,
    telemetry_mode: str = "none",
    policy_name: str = "ppo",
    curriculum: CurriculumConfig | None = None,
    reward_overrides: dict[str, float | None] | None = None,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    assist_overrides: dict[str, bool | float | None] | None = None,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    curriculum_sampler: CurriculumSampler | None = None
    if curriculum is not None and curriculum.enabled:
        curriculum_sampler = CurriculumSampler(curriculum)
    for episode in range(episodes):
        sim = MonzaSim(
            build_sim_config(
                max_steps=max_steps,
                action_mode=action_mode,
                action_set=action_set,
                continuous_action_scheme=continuous_action_scheme,
                observation_profile=observation_profile,
                launch_guard_progress_m=launch_guard_progress_m,
                launch_guard_min_speed_kph=launch_guard_min_speed_kph,
                launch_guard_throttle=launch_guard_throttle,
                reward_overrides=reward_overrides,
                assist_overrides=assist_overrides,
            )
        )
        reset_options = curriculum_sampler.sample_options(seed + episode) if curriculum_sampler is not None else None
        obs, _ = sim.reset(seed=seed + episode, options=reset_options)
        rows: list[dict[str, Any]] = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            if action_mode == "continuous":
                result = sim.step_continuous(np.asarray(action, dtype=np.float32))
            elif action_mode == "multidiscrete":
                result = sim.step_multidiscrete(np.asarray(action, dtype=np.int64))
            else:
                result = sim.step(_action_to_int(action))
            obs = result.observation
            row = asdict(result.telemetry)
            rows.append(row)
            if result.terminated or result.truncated:
                break
        metrics.append(_episode_metrics(steps=rows, policy=policy_name, episode=episode, seed=seed + episode))
        should_write = telemetry_mode == "all" or (telemetry_mode == "selected" and episode == 0)
        if should_write and telemetry_dir is not None:
            _write_jsonl(telemetry_dir / f"{policy_name}-episode-{episode:03d}-steps.jsonl", rows)
    return metrics


class TrainingEvalCallback:
    def __init__(
        self,
        *,
        run_root: Path,
        eval_every: int,
        eval_episodes: int,
        max_steps: int,
        seed: int,
        telemetry: str,
        telemetry_every: int,
        save_best: bool,
        curriculum: CurriculumConfig,
        reward_overrides: dict[str, float | None] | None = None,
        action_mode: str = "discrete",
        action_set: str = "legacy",
        continuous_action_scheme: str = "drive_brake",
        observation_profile: str = "base",
        launch_guard_progress_m: float = 0.0,
        launch_guard_min_speed_kph: float = 0.0,
        launch_guard_throttle: float = 0.22,
        assist_overrides: dict[str, bool | float | None] | None = None,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is required for callbacks.") from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: TrainingEvalCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

        self.callback = _Callback(self)
        self.run_root = run_root
        self.eval_every = max(eval_every, 1)
        self.eval_episodes = max(eval_episodes, 1)
        self.max_steps = max_steps
        self.seed = seed
        self.telemetry = telemetry
        self.telemetry_every = max(telemetry_every, 1)
        self.save_best = save_best
        self.curriculum = curriculum
        self.reward_overrides = reward_overrides
        self.action_mode = action_mode
        self.action_set = action_set
        self.continuous_action_scheme = continuous_action_scheme
        self.observation_profile = observation_profile
        self.launch_guard_progress_m = launch_guard_progress_m
        self.launch_guard_min_speed_kph = launch_guard_min_speed_kph
        self.launch_guard_throttle = launch_guard_throttle
        self.assist_overrides = assist_overrides
        self.last_eval = 0
        self.best_score = float("-inf")
        self.eval_root = run_root / "eval"
        self.eval_root.mkdir(parents=True, exist_ok=True)
        self.eval_metrics_path = self.eval_root / "eval_metrics.jsonl"
        self.selected_telemetry_dir = self.eval_root / "selected_telemetry"

    def on_step(self, callback: Any) -> bool:
        if callback.num_timesteps - self.last_eval < self.eval_every:
            return True
        self.last_eval = int(callback.num_timesteps)
        self.evaluate(callback.model, timesteps=int(callback.num_timesteps), phase="train")
        return True

    def evaluate(self, model: Any, *, timesteps: int, phase: str) -> dict[str, Any]:
        telemetry_mode = "none"
        if self.telemetry == "all" or (self.telemetry == "selected" and timesteps % self.telemetry_every == 0):
            telemetry_mode = self.telemetry
        policy_suffix = f"{phase}_{timesteps:08d}"
        metrics = run_model_rollouts(
            model,
            episodes=self.eval_episodes,
            max_steps=self.max_steps,
            seed=self.seed + timesteps,
            telemetry_dir=self.selected_telemetry_dir,
            telemetry_mode=telemetry_mode,
            policy_name=f"ppo_full_lap_{policy_suffix}",
            reward_overrides=self.reward_overrides,
            action_mode=self.action_mode,
            action_set=self.action_set,
            continuous_action_scheme=self.continuous_action_scheme,
            observation_profile=self.observation_profile,
            launch_guard_progress_m=self.launch_guard_progress_m,
            launch_guard_min_speed_kph=self.launch_guard_min_speed_kph,
            launch_guard_throttle=self.launch_guard_throttle,
            assist_overrides=self.assist_overrides,
        )
        mean_reward = sum(row["total_reward"] for row in metrics) / len(metrics)
        mean_progress = sum(row["best_progress_m"] for row in metrics) / len(metrics)
        completion_rate = sum(1 for row in metrics if row["completed_lap"]) / len(metrics)
        segment_metrics: list[dict[str, Any]] = []
        if self.curriculum.enabled:
            segment_eval_curriculum = _segment_eval_curriculum_config(self.curriculum)
            segment_metrics = run_model_rollouts(
                model,
                episodes=self.eval_episodes,
                max_steps=self.max_steps,
                seed=self.seed + timesteps + 50000,
                telemetry_dir=self.selected_telemetry_dir,
                telemetry_mode=telemetry_mode,
                policy_name=f"ppo_curriculum_segment_{policy_suffix}",
                curriculum=segment_eval_curriculum,
                reward_overrides=self.reward_overrides,
                action_mode=self.action_mode,
                action_set=self.action_set,
                continuous_action_scheme=self.continuous_action_scheme,
                observation_profile=self.observation_profile,
                launch_guard_progress_m=self.launch_guard_progress_m,
                launch_guard_min_speed_kph=self.launch_guard_min_speed_kph,
                launch_guard_throttle=self.launch_guard_throttle,
                assist_overrides=self.assist_overrides,
            )
        segment_source = segment_metrics or metrics
        segment_rate = sum(1 for row in segment_source if row["segment_complete"]) / len(segment_source)
        mean_segment_progress = sum(row["best_progress_m"] for row in segment_source) / len(segment_source)
        mean_segment_delta = sum(row["segment_progress_delta_m"] for row in segment_source) / len(segment_source)
        summary = {
            "timesteps": int(timesteps),
            "phase": phase,
            "scratch_initial_policy": phase == "initial_scratch",
            "mean_reward": mean_reward,
            "mean_best_progress_m": mean_progress,
            "completion_rate": completion_rate,
            "segment_completion_rate": segment_rate,
            "mean_segment_best_progress_m": mean_segment_progress,
            "mean_segment_progress_delta_m": mean_segment_delta,
            "episodes": metrics,
            "full_lap_episodes": metrics,
            "curriculum_segment_episodes": segment_metrics,
            "curriculum_stage_metrics": _curriculum_stage_metrics(segment_metrics) if segment_metrics else {},
        }
        score = _full_lap_selection_score(summary)
        summary["selection_score"] = score
        summary["selection_priority"] = "valid full-lap completion, then normal-start best progress"
        with self.eval_metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary) + "\n")
        if self.save_best and score > self.best_score:
            self.best_score = score
            model.save(self.run_root / "best_model.zip")
            vec_normalize = model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.run_root / "best_vecnormalize.pkl"))
            (self.eval_root / "best_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def run_training(
    *,
    timesteps: int,
    seed: int,
    n_envs: int,
    max_steps: int,
    device: str,
    checkpoint_every: int,
    require_gpu: bool = False,
    eval_every: int = 0,
    eval_episodes: int = 1,
    telemetry: str = "none",
    telemetry_every: int = 1,
    run_name: str | None = None,
    curriculum: str = "none",
    curriculum_preset: str = "default",
    curriculum_stage_count: int | None = None,
    curriculum_start_stage_index: int = 0,
    curriculum_promotion_resets: int = 300,
    curriculum_normal_start_probability: float = 0.0,
    curriculum_focus_start_progress_m: float | None = None,
    curriculum_focus_window_m: float = 0.0,
    curriculum_focus_segment_length_m: float = 500.0,
    curriculum_focus_min_speed_kph: float = 120.0,
    curriculum_focus_max_speed_kph: float = 220.0,
    curriculum_focus_position_noise_m: float = 0.5,
    curriculum_focus_heading_noise_deg: float = 2.0,
    curriculum_focus_speed_noise_kph: float = 5.0,
    curriculum_state_library: Path | None = None,
    curriculum_state_library_segment_length_m: float = 900.0,
    curriculum_state_library_position_noise_m: float = 0.0,
    curriculum_state_library_heading_noise_deg: float = 0.0,
    curriculum_state_library_speed_noise_kph: float = 0.0,
    curriculum_chicane: str = "rettifilo",
    vec_env: str = "dummy",
    save_best: bool = True,
    benchmark_throughput: bool = False,
    resume_checkpoint: Path | None = None,
    initialize_from_checkpoint: Path | None = None,
    reward_overrides: dict[str, float | None] | None = None,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    assist_overrides: dict[str, bool | float | None] | None = None,
    normalize_reward: bool = False,
    normalize_reward_gamma: float = 0.99,
    normalize_reward_clip: float = 10.0,
    vec_normalize_path: Path | None = None,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    ent_coef: float = 0.02,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    reward_scaffold_final_scale: float | None = None,
    reward_scaffold_schedule_timesteps: int | None = None,
) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
            VecNormalize,
        )
    except ImportError as exc:
        raise RuntimeError("Training requires stable-baselines3. Run `uv sync --active --all-extras --all-packages`.") from exc

    resolved_device = torch_device(device)
    if require_gpu and resolved_device != "cuda":
        raise RuntimeError(f"GPU required but resolved device is {resolved_device!r}.")
    policy = compute_policy(device)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{run_name}-{timestamp}" if run_name else f"train-{timestamp}"
    run_root = ARTIFACTS_DIR / run_id
    checkpoint_dir = run_root / "checkpoints"
    log_dir = run_root / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = _build_curriculum_config(
        mode=curriculum,
        preset=curriculum_preset,
        stage_count=curriculum_stage_count,
        stage_start_index=curriculum_start_stage_index,
        promotion_resets=curriculum_promotion_resets,
        normal_start_probability=curriculum_normal_start_probability,
        focus_start_progress_m=curriculum_focus_start_progress_m,
        focus_window_m=curriculum_focus_window_m,
        focus_segment_length_m=curriculum_focus_segment_length_m,
        focus_min_speed_kph=curriculum_focus_min_speed_kph,
        focus_max_speed_kph=curriculum_focus_max_speed_kph,
        focus_position_noise_m=curriculum_focus_position_noise_m,
        focus_heading_noise_deg=curriculum_focus_heading_noise_deg,
        focus_speed_noise_kph=curriculum_focus_speed_noise_kph,
        state_library_path=curriculum_state_library,
        state_library_segment_length_m=curriculum_state_library_segment_length_m,
        state_library_position_noise_m=curriculum_state_library_position_noise_m,
        state_library_heading_noise_deg=curriculum_state_library_heading_noise_deg,
        state_library_speed_noise_kph=curriculum_state_library_speed_noise_kph,
        chicane=curriculum_chicane,
    )
    vec_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    env = make_vec_env(
        _make_env(
            max_steps,
            seed,
            curriculum_config,
            reward_overrides,
            action_mode,
            action_set,
            continuous_action_scheme,
            observation_profile,
            launch_guard_progress_m,
            launch_guard_min_speed_kph,
            launch_guard_throttle,
            assist_overrides,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
    )
    env = VecMonitor(env)
    vec_normalize_load_mode: str | None = None
    if vec_normalize_path is not None:
        if not vec_normalize_path.exists():
            raise FileNotFoundError(f"VecNormalize stats do not exist: {vec_normalize_path}")
        env, vec_normalize_load_mode = _load_vecnormalize_with_reward_fallback(
            VecNormalize,
            vec_normalize_path,
            env,
            normalize_reward=normalize_reward,
            normalize_reward_gamma=normalize_reward_gamma,
            normalize_reward_clip=normalize_reward_clip,
        )
        env.training = True
        env.norm_obs = False
        env.norm_reward = normalize_reward
        env.clip_reward = normalize_reward_clip
        env.gamma = normalize_reward_gamma
    elif normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=normalize_reward_clip,
            gamma=normalize_reward_gamma,
        )
        vec_normalize_load_mode = "fresh"
    ppo_hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
    }
    if resume_checkpoint is not None and initialize_from_checkpoint is not None:
        raise ValueError("--resume-checkpoint and --initialize-from-checkpoint are mutually exclusive.")
    transfer_initialization = False
    transfer_weight_report: list[dict[str, Any]] | None = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        model = PPO.load(
            resume_checkpoint,
            env=env,
            device=resolved_device,
            tensorboard_log=str(log_dir),
            print_system_info=False,
            **ppo_hyperparams,
        )
        scratch_initialization = False
    elif initialize_from_checkpoint is not None:
        if not initialize_from_checkpoint.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {initialize_from_checkpoint}")
        source_model = PPO.load(
            initialize_from_checkpoint,
            device=resolved_device,
            print_system_info=False,
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=resolved_device,
            tensorboard_log=str(log_dir),
            **ppo_hyperparams,
        )
        transfer_weight_report = _copy_compatible_policy_weights(
            model,
            source_model,
            target_action_set=action_set,
        )
        scratch_initialization = False
        transfer_initialization = True
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=resolved_device,
            tensorboard_log=str(log_dir),
            **ppo_hyperparams,
        )
        scratch_initialization = True
    initial_checkpoint_path = checkpoint_dir / "initial_model.zip"
    initial_root_path = run_root / "initial_model.zip"
    model.save(initial_checkpoint_path)
    model.save(initial_root_path)
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=max(checkpoint_every // max(n_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_monza",
        )
    ]
    if reward_scaffold_final_scale is not None:
        from stable_baselines3.common.callbacks import BaseCallback

        class ScaffoldScheduleCallback(BaseCallback):
            def __init__(self, *, initial_scale: float, final_scale: float, schedule_timesteps: int) -> None:
                super().__init__()
                self.initial_scale = initial_scale
                self.final_scale = final_scale
                self.schedule_timesteps = max(int(schedule_timesteps), 1)

            def _on_step(self) -> bool:
                fraction = min(float(self.num_timesteps) / self.schedule_timesteps, 1.0)
                scale = self.initial_scale + (self.final_scale - self.initial_scale) * fraction
                self.training_env.env_method("set_reward_scaffold_scale", float(scale))
                return True

        initial_scale_value = (reward_overrides or {}).get("scaffold_scale")
        initial_scale = 1.0 if initial_scale_value is None else float(initial_scale_value)
        callbacks.append(
            ScaffoldScheduleCallback(
                initial_scale=initial_scale,
                final_scale=reward_scaffold_final_scale,
                schedule_timesteps=reward_scaffold_schedule_timesteps or timesteps,
            )
        )
    eval_callback: TrainingEvalCallback | None = None
    if eval_every > 0:
        eval_callback = TrainingEvalCallback(
            run_root=run_root,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 1000,
            telemetry=telemetry,
            telemetry_every=telemetry_every,
            save_best=save_best,
            curriculum=curriculum_config,
            reward_overrides=reward_overrides,
            action_mode=action_mode,
            action_set=action_set,
            continuous_action_scheme=continuous_action_scheme,
            observation_profile=observation_profile,
            launch_guard_progress_m=launch_guard_progress_m,
            launch_guard_min_speed_kph=launch_guard_min_speed_kph,
            launch_guard_throttle=launch_guard_throttle,
            assist_overrides=assist_overrides,
        )
        callbacks.append(
            eval_callback.callback
        )
    metadata_scaffold_initial_value = (reward_overrides or {}).get("scaffold_scale")
    metadata_scaffold_initial_scale = (
        1.0 if metadata_scaffold_initial_value is None else float(metadata_scaffold_initial_value)
    )
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "device": resolved_device,
        "require_gpu": require_gpu,
        "vec_env": vec_env,
        "telemetry": telemetry,
        "telemetry_every": telemetry_every,
        "curriculum": curriculum_config.mode,
        "curriculum_preset": curriculum_preset,
        "curriculum_stage_count": len(curriculum_config.stages) if curriculum_config.enabled else 0,
        "curriculum_start_stage_index": curriculum_start_stage_index,
        "curriculum_promotion_resets": curriculum_config.promotion_resets,
        "curriculum_config": CurriculumSampler(curriculum_config).to_dict(),
        "curriculum_state_library": str(curriculum_state_library) if curriculum_state_library is not None else None,
        "curriculum_chicane": curriculum_chicane,
        "compute_policy": policy,
        "benchmark_throughput": benchmark_throughput,
        "scratch_initialization": scratch_initialization,
        "transfer_initialization": transfer_initialization,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "initialize_from_checkpoint": (
            str(initialize_from_checkpoint) if initialize_from_checkpoint is not None else None
        ),
        "transfer_weight_report": transfer_weight_report,
        "vec_normalize_path": str(vec_normalize_path) if vec_normalize_path is not None else None,
        "vec_normalize_load_mode": vec_normalize_load_mode,
        "normalize_reward": normalize_reward,
        "normalize_reward_gamma": normalize_reward_gamma,
        "normalize_reward_clip": normalize_reward_clip,
        "initial_checkpoint": str(initial_checkpoint_path),
        "ppo_hyperparams": ppo_hyperparams,
        "scaffold_rewards_enabled": scaffold_rewards_enabled(
            build_sim_config(max_steps=max_steps, reward_overrides=reward_overrides).reward
        ),
        "training_assists_enabled": training_assists_enabled(
            build_sim_config(max_steps=max_steps, assist_overrides=assist_overrides).assist
        ),
        "assist_config": dataclass_to_dict(build_sim_config(max_steps=max_steps, assist_overrides=assist_overrides).assist),
        "scaffold_reward_schedule": {
            "initial_scale": metadata_scaffold_initial_scale,
            "final_scale": reward_scaffold_final_scale,
            "schedule_timesteps": reward_scaffold_schedule_timesteps or timesteps,
        }
        if reward_scaffold_final_scale is not None
        else None,
        "sim_config": dataclass_to_dict(
            build_sim_config(
                max_steps=max_steps,
                action_mode=action_mode,
                action_set=action_set,
                continuous_action_scheme=continuous_action_scheme,
                observation_profile=observation_profile,
                launch_guard_progress_m=launch_guard_progress_m,
                launch_guard_min_speed_kph=launch_guard_min_speed_kph,
                launch_guard_throttle=launch_guard_throttle,
                reward_overrides=reward_overrides,
                assist_overrides=assist_overrides,
            )
        ),
    }
    metadata_path = run_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if eval_callback is not None:
        if scratch_initialization:
            initial_phase = "initial_scratch"
        elif transfer_initialization:
            initial_phase = "initial_transfer"
        else:
            initial_phase = "initial_resume"
        eval_callback.evaluate(model, timesteps=0, phase=initial_phase)
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks), tb_log_name="ppo")
    elapsed = time.perf_counter() - started
    final_checkpoint_path = checkpoint_dir / "final_model.zip"
    final_root_path = run_root / "final_model.zip"
    model.save(final_checkpoint_path)
    model.save(final_root_path)
    vec_normalize = model.get_vec_normalize_env()
    final_vecnormalize_path: Path | None = None
    if vec_normalize is not None:
        final_vecnormalize_path = run_root / "vecnormalize.pkl"
        vec_normalize.save(str(final_vecnormalize_path))
    env.close()
    metadata.update(
        {
            "wall_clock_s": elapsed,
            "training_fps": float(timesteps / max(elapsed, 1e-9)),
            "env_steps_per_second": float((timesteps * max(n_envs, 1)) / max(elapsed, 1e-9)),
            "final_checkpoint": str(final_checkpoint_path),
            "final_vecnormalize": str(final_vecnormalize_path) if final_vecnormalize_path is not None else None,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "train_complete "
        f"run={run_root} checkpoint={final_checkpoint_path} device={resolved_device} "
        f"vec_env={vec_env} fps={metadata['training_fps']:.1f}"
    )
    return final_checkpoint_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the simplified Monza simulator.")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.set_defaults(save_best=True)
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--telemetry", choices=["none", "selected", "all"], default="none")
    parser.add_argument("--telemetry-every", type=int, default=1)
    parser.add_argument("--run-name")
    parser.add_argument("--curriculum", choices=["none", "segments"], default="none")
    parser.add_argument("--curriculum-preset", choices=["default", "prefix-start", "chicane-skill"], default="default")
    parser.add_argument("--curriculum-chicane", choices=["rettifilo", "roggia"], default="rettifilo")
    parser.add_argument("--curriculum-stage-count", type=int)
    parser.add_argument("--curriculum-start-stage-index", type=int, default=0)
    parser.add_argument("--curriculum-promotion-resets", type=int, default=300)
    parser.add_argument("--curriculum-normal-start-probability", type=float, default=0.0)
    parser.add_argument("--curriculum-focus-start-progress-m", type=float)
    parser.add_argument("--curriculum-focus-window-m", type=float, default=0.0)
    parser.add_argument("--curriculum-focus-segment-length-m", type=float, default=500.0)
    parser.add_argument("--curriculum-focus-min-speed-kph", type=float, default=120.0)
    parser.add_argument("--curriculum-focus-max-speed-kph", type=float, default=220.0)
    parser.add_argument("--curriculum-focus-position-noise-m", type=float, default=0.5)
    parser.add_argument("--curriculum-focus-heading-noise-deg", type=float, default=2.0)
    parser.add_argument("--curriculum-focus-speed-noise-kph", type=float, default=5.0)
    parser.add_argument("--curriculum-state-library", type=Path)
    parser.add_argument("--curriculum-state-library-segment-length-m", type=float, default=900.0)
    parser.add_argument("--curriculum-state-library-position-noise-m", type=float, default=0.0)
    parser.add_argument("--curriculum-state-library-heading-noise-deg", type=float, default=0.0)
    parser.add_argument("--curriculum-state-library-speed-noise-kph", type=float, default=0.0)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--action-mode", choices=["discrete", "continuous", "multidiscrete"], default="discrete")
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="legacy")
    parser.add_argument("--continuous-action-scheme", choices=sorted(CONTINUOUS_ACTION_SCHEMES), default="drive_brake")
    parser.add_argument("--observation-profile", choices=sorted(OBSERVATION_PROFILES), default="base")
    parser.add_argument("--launch-guard-progress-m", type=float, default=0.0)
    parser.add_argument("--launch-guard-min-speed-kph", type=float, default=0.0)
    parser.add_argument("--launch-guard-throttle", type=float, default=0.22)
    parser.add_argument("--assist-enabled", action="store_true")
    parser.add_argument("--assist-overspeed-turn-in-terminate", action="store_true")
    parser.add_argument("--assist-overspeed-turn-in-margin-kph", type=float)
    parser.add_argument("--assist-overspeed-turn-in-penalty", type=float)
    parser.add_argument("--assist-throttle-brake-demand-penalty-scale", type=float)
    parser.add_argument("--assist-no-brake-penalty", type=float)
    parser.add_argument("--assist-no-brake-min-brake", type=float)
    parser.add_argument("--assist-virtual-corridor-m", type=float)
    parser.add_argument("--assist-virtual-corridor-penalty", type=float)
    parser.add_argument("--assist-virtual-corridor-terminate", action="store_true")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--normalize-reward-gamma", type=float, default=0.99)
    parser.add_argument("--normalize-reward-clip", type=float, default=10.0)
    parser.add_argument("--vec-normalize-path", type=Path)
    parser.add_argument("--benchmark-throughput", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--initialize-from-checkpoint", type=Path)
    parser.add_argument("--reward-progress-scale", type=float)
    parser.add_argument("--reward-finish-bonus", type=float)
    parser.add_argument("--reward-collision-penalty", type=float)
    parser.add_argument("--reward-off-track-penalty", type=float)
    parser.add_argument("--reward-no-progress-penalty", type=float)
    parser.add_argument("--reward-lateral-deadzone-m", type=float)
    parser.add_argument("--reward-lateral-penalty-scale", type=float)
    parser.add_argument("--reward-track-limit-safe-ray-m", type=float)
    parser.add_argument("--reward-track-limit-penalty-scale", type=float)
    parser.add_argument("--reward-heading-deadzone-deg", type=float)
    parser.add_argument("--reward-heading-penalty-scale", type=float)
    parser.add_argument("--reward-speed-target-min-kph", type=float)
    parser.add_argument("--reward-speed-target-max-kph", type=float)
    parser.add_argument("--reward-speed-target-heading-scale", type=float)
    parser.add_argument("--reward-speed-target-deadzone-kph", type=float)
    parser.add_argument("--reward-speed-target-penalty-scale", type=float)
    parser.add_argument("--reward-overspeed-throttle-penalty-scale", type=float)
    parser.add_argument("--reward-overspeed-brake-reward-scale", type=float)
    parser.add_argument("--reward-steering-target-deadzone", type=float)
    parser.add_argument("--reward-steering-target-penalty-scale", type=float)
    parser.add_argument("--reward-smoothness-penalty", type=float)
    parser.add_argument("--reward-scaffold-scale", type=float)
    parser.add_argument("--reward-scaffold-brake-reward-scale", type=float)
    parser.add_argument("--reward-scaffold-no-throttle-penalty-scale", type=float)
    parser.add_argument("--reward-scaffold-turn-in-speed-penalty-scale", type=float)
    parser.add_argument("--reward-scaffold-apex-clean-reward-scale", type=float)
    parser.add_argument("--reward-scaffold-exit-alignment-reward-scale", type=float)
    parser.add_argument("--reward-scaffold-exit-speed-reward-scale", type=float)
    parser.add_argument("--reward-scaffold-final-scale", type=float)
    parser.add_argument("--reward-scaffold-schedule-timesteps", type=int)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--sde-sample-freq", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        require_gpu=args.require_gpu,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
        run_name=args.run_name,
        curriculum=args.curriculum,
        curriculum_preset=args.curriculum_preset,
        curriculum_stage_count=args.curriculum_stage_count,
        curriculum_start_stage_index=args.curriculum_start_stage_index,
        curriculum_promotion_resets=args.curriculum_promotion_resets,
        curriculum_normal_start_probability=args.curriculum_normal_start_probability,
        curriculum_focus_start_progress_m=args.curriculum_focus_start_progress_m,
        curriculum_focus_window_m=args.curriculum_focus_window_m,
        curriculum_focus_segment_length_m=args.curriculum_focus_segment_length_m,
        curriculum_focus_min_speed_kph=args.curriculum_focus_min_speed_kph,
        curriculum_focus_max_speed_kph=args.curriculum_focus_max_speed_kph,
        curriculum_focus_position_noise_m=args.curriculum_focus_position_noise_m,
        curriculum_focus_heading_noise_deg=args.curriculum_focus_heading_noise_deg,
        curriculum_focus_speed_noise_kph=args.curriculum_focus_speed_noise_kph,
        curriculum_state_library=args.curriculum_state_library,
        curriculum_state_library_segment_length_m=args.curriculum_state_library_segment_length_m,
        curriculum_state_library_position_noise_m=args.curriculum_state_library_position_noise_m,
        curriculum_state_library_heading_noise_deg=args.curriculum_state_library_heading_noise_deg,
        curriculum_state_library_speed_noise_kph=args.curriculum_state_library_speed_noise_kph,
        curriculum_chicane=args.curriculum_chicane,
        vec_env=args.vec_env,
        save_best=args.save_best,
        benchmark_throughput=args.benchmark_throughput,
        resume_checkpoint=args.resume_checkpoint,
        initialize_from_checkpoint=args.initialize_from_checkpoint,
        reward_overrides=_reward_overrides_from_args(args),
        action_mode=args.action_mode,
        action_set=args.action_set,
        continuous_action_scheme=args.continuous_action_scheme,
        observation_profile=args.observation_profile,
        launch_guard_progress_m=args.launch_guard_progress_m,
        launch_guard_min_speed_kph=args.launch_guard_min_speed_kph,
        launch_guard_throttle=args.launch_guard_throttle,
        assist_overrides=_assist_overrides_from_args(args),
        normalize_reward=args.normalize_reward,
        normalize_reward_gamma=args.normalize_reward_gamma,
        normalize_reward_clip=args.normalize_reward_clip,
        vec_normalize_path=args.vec_normalize_path,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        reward_scaffold_final_scale=args.reward_scaffold_final_scale,
        reward_scaffold_schedule_timesteps=args.reward_scaffold_schedule_timesteps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
