"""Headless benchmark harness for random, scripted, PPO, and reference policies."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import (
    ACTION_MODES,
    ACTION_SETS,
    ARTIFACTS_DIR,
    CONTINUOUS_ACTION_SCHEMES,
    OBSERVATION_PROFILES,
    PHYSICS_MODELS,
    SimConfig,
    build_sim_config,
    dataclass_to_dict,
    disable_scaffold_rewards,
    disable_training_assists,
    multidiscrete_action_nvec,
)
from f1rl.env import MonzaEnv
from f1rl.policy_io import (
    close_vecnormalize,
    load_sb3_ppo,
    load_vecnormalize_stats,
    normalize_observation,
    resolve_ppo_eval_config,
    validate_model_spaces,
)
from f1rl.reference_agent import load_reference_profile, run_reference_ghost
from f1rl.scripted import ScriptedController
from f1rl.sim import MonzaSim


def _action_to_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _episode_metrics(
    *,
    policy: str,
    episode: int,
    seed: int,
    rows: list[dict[str, Any]],
    wall_clock_s: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "policy": policy,
            "episode": episode,
            "seed": seed,
            "termination_reason": "empty",
            "completed_lap": False,
            "valid_lap": False,
            "steps": 0,
            "wall_clock_s": wall_clock_s,
            "steps_per_second": 0.0,
        }
    final = rows[-1]
    reward_totals: dict[str, float] = {}
    for row in rows:
        for key, value in row["reward_components"].items():
            reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
    ray_mins = [min(row["ray_distances_m"]) for row in rows if row["ray_distances_m"]]
    best_progress_m = float(max(row["monotonic_progress_m"] for row in rows))
    start_progress_m = float(rows[0]["monotonic_progress_m"])
    segment_target_progress_m = final.get("segment_target_progress_m")
    checkpoints_passed = max(
        int(row.get("checkpoints_passed", row.get("checkpoint_index", 0)) or 0) for row in rows
    )
    missed_checkpoint_count = max(int(row.get("missed_checkpoint_count", 0) or 0) for row in rows)
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "termination_reason": final["termination_reason"],
        "completed_lap": bool(final["termination_reason"] == "lap_complete"),
        "valid_lap": bool(final.get("valid_lap", False)),
        "finish_crossed": bool(final.get("finish_crossed", False)),
        "segment_complete": bool(final.get("segment_complete", False)),
        "elapsed_time_s": float(final["sim_time_s"]),
        "lap_time_s": float(final["sim_time_s"]) if final["termination_reason"] == "lap_complete" else None,
        "steps": len(rows),
        "total_reward": float(sum(row["reward_total"] for row in rows)),
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
        "checkpoints_passed": checkpoints_passed,
        "checkpoint_count": 120,
        "missed_checkpoint_count": missed_checkpoint_count,
        "crashed": bool(final["collided"]),
        "off_track": bool(final["off_track"]),
        "no_progress": final["termination_reason"] == "no_progress",
        "max_steps": final["termination_reason"] == "max_steps",
        "avg_speed_kph": float(sum(row["speed_kph"] for row in rows) / len(rows)),
        "max_speed_kph": float(max(row["speed_kph"] for row in rows)),
        "avg_lateral_g": float(sum(abs(row["lateral_g"]) for row in rows) / len(rows)),
        "max_lateral_g": float(max(abs(row["lateral_g"]) for row in rows)),
        "min_ray_distance_m": float(min(ray_mins)) if ray_mins else None,
        "avg_min_ray_distance_m": float(sum(ray_mins) / len(ray_mins)) if ray_mins else None,
        "avg_racing_line_deviation_m": float(
            sum(abs(row["racing_line_deviation_m"]) for row in rows) / len(rows)
        ),
        "final_ghost_gap_m": final.get("ghost_gap_m"),
        "curriculum_stage": final.get("curriculum_stage"),
        "wall_clock_s": wall_clock_s,
        "steps_per_second": float(len(rows) / max(wall_clock_s, 1e-9)),
    }


def _should_write_telemetry(mode: str, episode: int, telemetry_every: int) -> bool:
    return mode == "all" or (mode == "selected" and episode % max(telemetry_every, 1) == 0)


def _run_sim_policy(
    *,
    policy: str,
    episode: int,
    seed: int,
    max_steps: int,
    telemetry_dir: Path,
    telemetry: str,
    telemetry_every: int,
    model: Any | None = None,
    sim_config: SimConfig | None = None,
    vec_normalize: Any | None = None,
    ppo_config_source: str | None = None,
    ppo_deterministic: bool = True,
) -> dict[str, Any]:
    resolved_config = replace(sim_config or SimConfig(), max_steps=max_steps)
    sim = MonzaSim(resolved_config)
    obs, _ = sim.reset(seed=seed)
    obs_for_model = normalize_observation(obs, vec_normalize)
    rng = np.random.default_rng(seed)
    scripted = ScriptedController()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(max_steps):
        if policy == "random":
            if resolved_config.action_mode == "continuous":
                result = sim.step_continuous(rng.uniform(-1.0, 1.0, size=2))
            elif resolved_config.action_mode == "multidiscrete":
                nvec = multidiscrete_action_nvec()
                result = sim.step_multidiscrete(
                    [int(rng.integers(0, nvec[0])), int(rng.integers(0, nvec[1]))]
                )
            else:
                result = sim.step(int(rng.integers(0, sim.action_dim)))
            obs = result.observation
        elif policy == "scripted":
            throttle, brake, steer = scripted.controls(sim)
            result = sim.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=-10)
            obs = result.observation
        elif policy == "ppo" and model is not None:
            action, _ = model.predict(obs_for_model, deterministic=ppo_deterministic)
            if resolved_config.action_mode == "continuous":
                result = sim.step_continuous(np.asarray(action, dtype=np.float32))
            elif resolved_config.action_mode == "multidiscrete":
                result = sim.step_multidiscrete(np.asarray(action, dtype=np.int64))
            else:
                result = sim.step(_action_to_int(action))
            obs = result.observation
            obs_for_model = normalize_observation(obs, vec_normalize)
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        rows.append(asdict(result.telemetry))
        if result.terminated or result.truncated:
            break
    wall_clock_s = time.perf_counter() - started
    if _should_write_telemetry(telemetry, episode, telemetry_every):
        _write_jsonl(telemetry_dir / f"{policy}-episode-{episode:03d}-steps.jsonl", rows)
    metrics = _episode_metrics(policy=policy, episode=episode, seed=seed, rows=rows, wall_clock_s=wall_clock_s)
    if policy == "ppo":
        metrics["ppo_config_source"] = ppo_config_source
        metrics["ppo_action_mode"] = resolved_config.action_mode
        metrics["ppo_action_set"] = resolved_config.action_set
        metrics["ppo_observation_profile"] = resolved_config.observation_profile
        metrics["ppo_continuous_action_scheme"] = resolved_config.continuous_action_scheme
        metrics["ppo_deterministic"] = ppo_deterministic
    metrics["physics_model"] = resolved_config.physics_model
    metrics["physics_version"] = resolved_config.physics_version
    metrics["physics_calibration_id"] = resolved_config.physics_calibration_id
    return metrics


def _run_reference_policy(
    *,
    episode: int,
    seed: int,
    telemetry_dir: Path,
    telemetry: str,
    telemetry_every: int,
) -> dict[str, Any]:
    profile = load_reference_profile()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if _should_write_telemetry(telemetry, episode, telemetry_every):
        root = run_reference_ghost(seed=seed, telemetry=True)
        if root is not None:
            source = root / "steps.jsonl"
            target = telemetry_dir / f"reference_ghost-episode-{episode:03d}-steps.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with target.open("r", encoding="utf-8") as file:
                rows = [json.loads(line) for line in file if line.strip()]
    wall_clock_s = time.perf_counter() - started
    if rows:
        return _episode_metrics(policy="reference_ghost", episode=episode, seed=seed, rows=rows, wall_clock_s=wall_clock_s)
    return {
        "policy": "reference_ghost",
        "episode": episode,
        "seed": seed,
        "termination_reason": "lap_complete",
        "completed_lap": True,
        "valid_lap": True,
        "finish_crossed": True,
        "segment_complete": False,
        "elapsed_time_s": profile.lap_time_s,
        "lap_time_s": profile.lap_time_s,
        "steps": len(profile.time_s),
        "total_reward": 0.0,
        "reward_totals": {},
        "final_progress_m": 5793.0,
        "best_progress_m": 5793.0,
        "segment_progress_delta_m": 5793.0,
        "segment_target_progress_m": None,
        "segment_remaining_m": None,
        "checkpoints_passed": 119,
        "checkpoint_count": 120,
        "missed_checkpoint_count": 0,
        "crashed": False,
        "off_track": False,
        "no_progress": False,
        "max_steps": False,
        "avg_speed_kph": float(np.mean(profile.speed_kph)),
        "max_speed_kph": float(np.max(profile.speed_kph)),
        "avg_lateral_g": None,
        "max_lateral_g": None,
        "min_ray_distance_m": None,
        "avg_min_ray_distance_m": None,
        "avg_racing_line_deviation_m": 0.0,
        "final_ghost_gap_m": 0.0,
        "curriculum_stage": None,
        "wall_clock_s": wall_clock_s,
        "steps_per_second": float(len(profile.time_s) / max(wall_clock_s, 1e-9)),
    }


def _aggregate(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(policy_rows)
    if episodes == 0:
        return {}
    return {
        "policy": policy_rows[0]["policy"],
        "episodes": episodes,
        "completion_rate": sum(1 for row in policy_rows if row["completed_lap"]) / episodes,
        "valid_lap_rate": sum(1 for row in policy_rows if row.get("valid_lap")) / episodes,
        "finish_crossed_rate": sum(1 for row in policy_rows if row.get("finish_crossed")) / episodes,
        "invalid_lap_rate": sum(1 for row in policy_rows if not row.get("valid_lap")) / episodes,
        "segment_completion_rate": sum(1 for row in policy_rows if row.get("segment_complete")) / episodes,
        "crash_rate": sum(1 for row in policy_rows if row.get("crashed") or row.get("off_track")) / episodes,
        "avg_reward": sum(float(row.get("total_reward") or 0.0) for row in policy_rows) / episodes,
        "avg_progress_m": sum(float(row.get("best_progress_m") or 0.0) for row in policy_rows) / episodes,
        "best_progress_m": max(float(row.get("best_progress_m") or 0.0) for row in policy_rows),
        "avg_checkpoints_passed": sum(float(row.get("checkpoints_passed") or 0.0) for row in policy_rows)
        / episodes,
        "avg_missed_checkpoint_count": sum(
            float(row.get("missed_checkpoint_count") or 0.0) for row in policy_rows
        )
        / episodes,
        "max_missed_checkpoint_count": max(int(row.get("missed_checkpoint_count") or 0) for row in policy_rows),
        "avg_steps": sum(float(row.get("steps") or 0.0) for row in policy_rows) / episodes,
        "max_steps_observed": max(int(row.get("steps") or 0) for row in policy_rows),
        "avg_elapsed_time_s": sum(float(row.get("elapsed_time_s") or 0.0) for row in policy_rows)
        / episodes,
        "avg_segment_progress_delta_m": sum(
            float(row.get("segment_progress_delta_m") or 0.0) for row in policy_rows
        )
        / episodes,
        "avg_steps_per_second": sum(float(row.get("steps_per_second") or 0.0) for row in policy_rows) / episodes,
        "best_lap_time_s": min(
            (float(row["lap_time_s"]) for row in policy_rows if row.get("lap_time_s") is not None),
            default=None,
        ),
        "termination_reasons": {
            reason: sum(1 for row in policy_rows if row["termination_reason"] == reason)
            for reason in sorted({row["termination_reason"] for row in policy_rows})
        },
    }


def run_benchmark(
    *,
    policies: list[str],
    episodes: int,
    max_steps: int,
    seed: int,
    checkpoint: str,
    device: str,
    telemetry: str,
    telemetry_every: int,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    physics_model: str = "v1",
    metadata_mode: str = "auto",
    disable_scaffold: bool = False,
    disable_assists: bool = False,
    ppo_deterministic: bool = True,
) -> Path:
    run_id = f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}"
    run_root = ARTIFACTS_DIR / run_id
    telemetry_dir = run_root / "selected_telemetry"
    run_root.mkdir(parents=True, exist_ok=True)
    fallback_config = build_sim_config(
        max_steps=max_steps,
        physics_model=physics_model,
        action_mode=action_mode,
        action_set=action_set,
        continuous_action_scheme=continuous_action_scheme,
        observation_profile=observation_profile,
        launch_guard_progress_m=launch_guard_progress_m,
        launch_guard_min_speed_kph=launch_guard_min_speed_kph,
        launch_guard_throttle=launch_guard_throttle,
    )
    ppo_config = None
    model = None
    vec_normalize = None
    if "ppo" in policies:
        ppo_config = resolve_ppo_eval_config(
            checkpoint,
            max_steps=max_steps,
            fallback_config=fallback_config,
            metadata_mode=metadata_mode,
        )
        if disable_scaffold:
            disable_scaffold_rewards(ppo_config.sim_config)
        if disable_assists:
            disable_training_assists(ppo_config.sim_config)
        validation_env = MonzaEnv(ppo_config.sim_config)
        try:
            vec_normalize = load_vecnormalize_stats(ppo_config)
            model = load_sb3_ppo(ppo_config.checkpoint_path, device=device)
            validate_model_spaces(
                model,
                observation_space=validation_env.observation_space,
                action_space=validation_env.action_space,
            )
        finally:
            validation_env.close()
    effective_config = ppo_config.sim_config if ppo_config is not None else fallback_config
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "policies": policies,
                "episodes": episodes,
                "max_steps": max_steps,
                "seed": seed,
                "checkpoint": checkpoint,
                "resolved_checkpoint": str(ppo_config.checkpoint_path) if ppo_config is not None else None,
                "device": device,
                "telemetry": telemetry,
                "telemetry_every": telemetry_every,
                "physics_model": effective_config.physics_model,
                "physics_version": effective_config.physics_version,
                "physics_calibration_id": effective_config.physics_calibration_id,
                "metadata_mode": metadata_mode,
                "disable_scaffold_rewards": disable_scaffold,
                "disable_training_assists": disable_assists,
                "ppo_deterministic": ppo_deterministic,
                "requested_sim_config": {
                    "physics_model": physics_model,
                    "action_mode": action_mode,
                    "action_set": action_set,
                    "continuous_action_scheme": continuous_action_scheme,
                    "observation_profile": observation_profile,
                    "launch_guard_progress_m": launch_guard_progress_m,
                    "launch_guard_min_speed_kph": launch_guard_min_speed_kph,
                    "launch_guard_throttle": launch_guard_throttle,
                },
                "effective_sim_config": dataclass_to_dict(effective_config),
                "ppo_eval_config": ppo_config.report() if ppo_config is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    try:
        for policy in policies:
            for episode in range(episodes):
                episode_seed = seed + episode
                if policy == "reference_ghost":
                    row = _run_reference_policy(
                        episode=episode,
                        seed=episode_seed,
                        telemetry_dir=telemetry_dir,
                        telemetry=telemetry,
                        telemetry_every=telemetry_every,
                    )
                else:
                    row = _run_sim_policy(
                        policy=policy,
                        episode=episode,
                        seed=episode_seed,
                        max_steps=max_steps,
                        telemetry_dir=telemetry_dir,
                        telemetry=telemetry,
                        telemetry_every=telemetry_every,
                        model=model,
                        sim_config=effective_config,
                        vec_normalize=vec_normalize if policy == "ppo" else None,
                        ppo_config_source=ppo_config.config_source if ppo_config is not None else None,
                        ppo_deterministic=ppo_deterministic,
                    )
                rows.append(row)
    finally:
        close_vecnormalize(vec_normalize)

    _write_jsonl(run_root / "per_episode.jsonl", rows)
    summary_rows = [_aggregate([row for row in rows if row["policy"] == policy]) for policy in policies]
    summary = {
        "run_id": run_id,
        "policies": summary_rows,
        "episodes": rows,
        "physics_model": effective_config.physics_model,
        "physics_version": effective_config.physics_version,
        "physics_calibration_id": effective_config.physics_calibration_id,
        "ppo_eval_config": ppo_config.report() if ppo_config is not None else None,
        "disable_scaffold_rewards": disable_scaffold,
        "disable_training_assists": disable_assists,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"benchmark_complete run={run_root}")
    return run_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Monza policies headlessly.")
    parser.add_argument("--policies", nargs="+", choices=["random", "scripted", "ppo", "reference_ghost"], required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--telemetry", choices=["none", "selected", "all"], default="selected")
    parser.add_argument("--telemetry-every", type=int, default=1)
    parser.add_argument("--action-mode", choices=sorted(ACTION_MODES), default="discrete")
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="legacy")
    parser.add_argument("--continuous-action-scheme", choices=sorted(CONTINUOUS_ACTION_SCHEMES), default="drive_brake")
    parser.add_argument("--observation-profile", choices=sorted(OBSERVATION_PROFILES), default="base")
    parser.add_argument("--launch-guard-progress-m", type=float, default=0.0)
    parser.add_argument("--launch-guard-min-speed-kph", type=float, default=0.0)
    parser.add_argument("--launch-guard-throttle", type=float, default=0.22)
    parser.add_argument("--physics-model", choices=sorted(PHYSICS_MODELS), default="v1")
    parser.add_argument("--metadata-mode", choices=["auto", "require", "ignore"], default="auto")
    parser.add_argument("--disable-scaffold-rewards", action="store_true")
    parser.add_argument("--disable-training-assists", action="store_true")
    parser.set_defaults(ppo_deterministic=True)
    parser.add_argument("--ppo-deterministic", dest="ppo_deterministic", action="store_true")
    parser.add_argument("--ppo-stochastic", dest="ppo_deterministic", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(
        policies=args.policies,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
        action_mode=args.action_mode,
        action_set=args.action_set,
        continuous_action_scheme=args.continuous_action_scheme,
        observation_profile=args.observation_profile,
        launch_guard_progress_m=args.launch_guard_progress_m,
        launch_guard_min_speed_kph=args.launch_guard_min_speed_kph,
        launch_guard_throttle=args.launch_guard_throttle,
        physics_model=args.physics_model,
        metadata_mode=args.metadata_mode,
        disable_scaffold=args.disable_scaffold_rewards,
        disable_assists=args.disable_training_assists,
        ppo_deterministic=args.ppo_deterministic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
