"""Autonomous staged training campaign with swarm diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from f1rl.artifacts import new_run_paths, owning_run_dir, resolve_checkpoint
from f1rl.benchmark import run_benchmark
from f1rl.clean_artifacts import RetentionPolicy, clean_artifacts
from f1rl.constants import ARTIFACTS_DIR, CHAMPIONS_DIR
from f1rl.stages import (
    STAGE_ORDER,
    compare_aggregate_for_stage,
    get_stage_config,
    infer_stage_from_aggregate,
    logical_car_target,
    next_stage,
    stage_gate_met,
)
from f1rl.swarm import parse_args as parse_swarm_args
from f1rl.swarm import run_swarm


@dataclass(slots=True, frozen=True)
class CandidateSpec:
    family: str
    name: str
    env_overrides: dict[str, Any] = field(default_factory=dict)
    ppo_overrides: dict[str, Any] = field(default_factory=dict)
    logical_cars: int | None = None
    vector_mode: str | None = None
    num_env_runners: int | None = None
    num_envs_per_env_runner: int | None = None
    iterations: int | None = None


@dataclass(slots=True)
class CampaignState:
    stage: str
    family_index: int = 0
    family_no_promotion_waves: int = 0
    family_rotations_without_promotion: int = 0
    wave_index: int = 0
    rotation_index: int = 0
    hybrid_enabled: bool = False


FAMILY_ORDER = (
    "reward",
    "termination",
    "observations",
    "action_control",
    "curriculum",
    "ppo",
    "performance",
)

TRAIN_CHECKPOINT_TOKEN = "checkpoint="


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded staged PPO campaign.")
    parser.add_argument("--hours", type=float, default=7.0, help="Wall-clock budget for the campaign.")
    parser.add_argument(
        "--stage",
        choices=["auto", *STAGE_ORDER],
        default="auto",
        help="Force a starting stage or infer it from the baseline benchmark.",
    )
    parser.add_argument(
        "--start-family",
        choices=FAMILY_ORDER,
        default=None,
        help="Optional experiment family to start from instead of beginning at reward.",
    )
    parser.add_argument("--baseline-checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--start-cars", type=int, default=32)
    parser.add_argument("--max-cars", type=int, default=128)
    parser.add_argument("--max-waves", type=int, default=9999)
    parser.add_argument("--max-candidates-per-wave", type=int, default=3)
    parser.add_argument("--holdout-every", type=int, default=3)
    parser.add_argument("--keep-runs-per-prefix", type=int, default=3)
    parser.add_argument("--keep-days", type=int, default=7)
    parser.add_argument("--swarm-clips", action="store_true", help="Export early/mid/late swarm GIFs for each candidate.")
    parser.add_argument("--benchmark-clips", action="store_true", help="Export benchmark GIFs for each candidate.")
    return parser.parse_args(argv)


def _git_commit_hash() -> str:
    git_dir = Path(".git")
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"
    head_value = head_path.read_text(encoding="utf-8").strip()
    if head_value.startswith("ref: "):
        ref_path = git_dir / head_value.removeprefix("ref: ").strip()
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip() or "unknown"
        return "unknown"
    return head_value or "unknown"


def _run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def _extract_checkpoint(stdout: str) -> str:
    match = re.search(r"checkpoint=(.+?)(?:\s+use_gpu=|\s*$)", stdout, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    raise RuntimeError(f"Unable to parse checkpoint from training output:\n{stdout}")


def _logical_cars_for_stage(stage: str, start_cars: int, max_cars: int) -> int:
    return min(max_cars, max(start_cars, logical_car_target(stage)))


def _curriculum_candidates(stage: str) -> list[CandidateSpec]:
    if stage == "competence":
        return [
            CandidateSpec(
                family="curriculum",
                name="easy_sector_a",
                env_overrides={"spawn_goal_range": [1, 10]},
            ),
            CandidateSpec(
                family="curriculum",
                name="easy_sector_b",
                env_overrides={"spawn_goal_range": [10, 20]},
            ),
            CandidateSpec(
                family="curriculum",
                name="easy_sector_jittered",
                env_overrides={
                    "spawn_goal_range": [1, 20],
                    "spawn_position_jitter_px": 8.0,
                    "spawn_heading_jitter_deg": 6.0,
                },
            ),
        ]
    if stage == "lap":
        return [
            CandidateSpec(
                family="curriculum",
                name="lap_random_sectors",
                env_overrides={"spawn_goal_range": [1, 60], "spawn_heading_jitter_deg": 4.0},
            ),
            CandidateSpec(
                family="curriculum",
                name="lap_track_wide_random",
                env_overrides={
                    "spawn_goal_range": [1, 90],
                    "spawn_position_jitter_px": 6.0,
                    "spawn_heading_jitter_deg": 6.0,
                },
            ),
            CandidateSpec(
                family="curriculum",
                name="lap_full_random_start",
                env_overrides={
                    "spawn_goal_range": [1, 118],
                    "spawn_position_jitter_px": 8.0,
                    "spawn_heading_jitter_deg": 8.0,
                },
            ),
        ]
    return [
        CandidateSpec(
            family="curriculum",
            name="stability_jittered_start",
            env_overrides={"spawn_position_jitter_px": 10.0, "spawn_heading_jitter_deg": 8.0},
        ),
        CandidateSpec(
            family="curriculum",
            name="stability_randomized_goals",
            env_overrides={
                "spawn_goal_range": [1, 118],
                "spawn_position_jitter_px": 10.0,
                "spawn_heading_jitter_deg": 10.0,
            },
        ),
        CandidateSpec(
            family="curriculum",
            name="stability_broad_random",
            env_overrides={"spawn_goal_range": [1, 118], "spawn_heading_jitter_deg": 12.0},
        ),
    ]


def candidate_library(stage: str, logical_cars: int, max_cars: int) -> dict[str, list[CandidateSpec]]:
    doubled_cars = min(max_cars, max(logical_cars, logical_cars * 2))
    return {
        "reward": [
            CandidateSpec(
                family="reward",
                name="reward_progress_bias",
                env_overrides={"reward": {"progress_reward": 3.2}},
            ),
            CandidateSpec(
                family="reward",
                name="reward_safety_bias",
                env_overrides={
                    "reward": {
                        "collision_penalty": -10.0,
                        "centerline_penalty_weight": 0.026,
                    }
                },
            ),
            CandidateSpec(
                family="reward",
                name="reward_alignment_bias",
                env_overrides={
                    "reward": {
                        "alignment_weight": 0.03,
                        "forward_speed_weight": 0.04,
                    }
                },
            ),
        ],
        "termination": [
            CandidateSpec(
                family="termination",
                name="term_fast_no_progress",
                env_overrides={"no_progress_limit_steps": max(30, get_stage_config(stage).no_progress_limit_steps - 15)},
            ),
            CandidateSpec(
                family="termination",
                name="term_tighter_heading",
                env_overrides={"heading_away_limit_steps": max(12, get_stage_config(stage).heading_away_limit_steps - 12)},
            ),
            CandidateSpec(
                family="termination",
                name="term_tighter_wall",
                env_overrides={"wall_hugging_limit_steps": max(10, get_stage_config(stage).wall_hugging_limit_steps - 10)},
            ),
        ],
        "observations": [
            CandidateSpec(
                family="observations",
                name="obs_sensor_11_forward",
                env_overrides={"dynamics": {"sensor_count": 11, "sensor_spread_deg": 150.0, "sensor_forward_bias": 2.0}},
            ),
            CandidateSpec(
                family="observations",
                name="obs_sensor_13_forward",
                env_overrides={"dynamics": {"sensor_count": 13, "sensor_spread_deg": 160.0, "sensor_forward_bias": 2.2}},
            ),
            CandidateSpec(
                family="observations",
                name="obs_sensor_11_narrow",
                env_overrides={"dynamics": {"sensor_count": 11, "sensor_spread_deg": 125.0, "sensor_forward_bias": 1.8}},
            ),
        ],
        "action_control": [
            CandidateSpec(
                family="action_control",
                name="control_sharper_steer",
                env_overrides={"dynamics": {"steering_response": 0.48, "max_steer_angle_deg": 32.0}},
            ),
            CandidateSpec(
                family="action_control",
                name="control_more_grip",
                env_overrides={"dynamics": {"lateral_grip": 0.95, "high_speed_steer_reduction": 0.52}},
            ),
            CandidateSpec(
                family="action_control",
                name="control_brake_bias",
                env_overrides={"dynamics": {"brake_deceleration_rate": 0.6, "coast_deceleration_rate": 0.05}},
            ),
        ],
        "curriculum": _curriculum_candidates(stage),
        "ppo": [
            CandidateSpec(
                family="ppo",
                name="ppo_entropy_up",
                ppo_overrides={"entropy_coeff": 0.003},
            ),
            CandidateSpec(
                family="ppo",
                name="ppo_gamma_lambda",
                ppo_overrides={"gamma": 0.997, "lambda_": 0.98},
            ),
            CandidateSpec(
                family="ppo",
                name="ppo_larger_batches",
                ppo_overrides={"train_batch_size_per_learner": 768, "minibatch_size": 256},
            ),
        ],
        "performance": [
            CandidateSpec(
                family="performance",
                name="perf_async_scale",
                logical_cars=doubled_cars,
                vector_mode="async",
            ),
            CandidateSpec(
                family="performance",
                name="perf_more_runners",
                logical_cars=doubled_cars,
                num_env_runners=4,
                vector_mode="async",
            ),
            CandidateSpec(
                family="performance",
                name="perf_faster_rollout",
                logical_cars=doubled_cars,
                vector_mode="async",
                ppo_overrides={"rollout_fragment_length": 128, "train_batch_size_per_learner": 1024},
            ),
        ],
    }


def _diagnostic_collapse(stage: str, candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, str]:
    if stage == "competence":
        if float(candidate.get("first_checkpoint_rate", 0.0)) + 0.2 < float(baseline.get("first_checkpoint_rate", 0.0)):
            return True, "diagnostic first-checkpoint collapse"
        if float(candidate.get("avg_checkpoints_reached", 0.0)) + 1.5 < float(baseline.get("avg_checkpoints_reached", 0.0)):
            return True, "diagnostic checkpoint-reach collapse"
        if float(candidate.get("collision_rate", 0.0)) > float(baseline.get("collision_rate", 0.0)) + 0.25:
            return True, "diagnostic collision regression"
        return False, ""

    if float(candidate.get("completion_rate", 0.0)) + 0.2 < float(baseline.get("completion_rate", 0.0)):
        return True, "diagnostic completion collapse"
    if float(candidate.get("collision_rate", 0.0)) > float(baseline.get("collision_rate", 0.0)) + 0.25:
        return True, "diagnostic collision regression"
    return False, ""


def _train_candidate(
    *,
    candidate: CandidateSpec,
    stage: str,
    device: str,
    logical_cars: int,
    iterations_override: int | None = None,
) -> str:
    stage_cfg = get_stage_config(stage)
    cmd = [
        sys.executable,
        "-m",
        "f1rl.train",
        "--mode",
        stage_cfg.training_mode,
        "--iterations",
        str(iterations_override or candidate.iterations or (2 if stage == "competence" else 3)),
        "--device",
        device,
        "--swarm-stage",
        stage,
        "--logical-cars",
        str(candidate.logical_cars or logical_cars),
        "--run-tag",
        candidate.name,
    ]
    if candidate.vector_mode is not None:
        cmd.extend(["--vector-mode", candidate.vector_mode])
    if candidate.num_env_runners is not None:
        cmd.extend(["--num-env-runners", str(candidate.num_env_runners)])
    if candidate.num_envs_per_env_runner is not None:
        cmd.extend(["--num-envs-per-env-runner", str(candidate.num_envs_per_env_runner)])
    if candidate.env_overrides:
        cmd.extend(["--env-config-json", json.dumps(candidate.env_overrides)])
    if candidate.ppo_overrides:
        cmd.extend(["--ppo-config-json", json.dumps(candidate.ppo_overrides)])
    if device == "gpu":
        cmd.append("--require-gpu")
    stdout = _run_command(cmd)
    return _extract_checkpoint(stdout)


def _baseline_checkpoint(candidate_checkpoint: str | None, fallback: str) -> str:
    return candidate_checkpoint or fallback


def _load_current_champion_checkpoint(default_checkpoint: str) -> str:
    champion_path = CHAMPIONS_DIR / "current.json"
    if not champion_path.exists():
        return default_checkpoint
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    return str(champion.get("checkpoint", default_checkpoint))


def _load_current_champion_summary() -> dict[str, Any] | None:
    champion_path = CHAMPIONS_DIR / "current.json"
    if not champion_path.exists():
        return None
    return json.loads(champion_path.read_text(encoding="utf-8"))


def _load_training_metrics(checkpoint_path: str) -> dict[str, float]:
    run_dir = owning_run_dir(Path(checkpoint_path))
    metrics_path = run_dir / "metrics" / "metrics.csv"
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    latest = rows[-1]
    metrics: dict[str, float] = {}
    for key in (
        "timesteps_total",
        "policy_entropy",
        "value_loss",
        "kl_value",
        "sample_time_s",
        "learner_update_time_s",
        "sync_weights_time_s",
        "samples_per_second",
        "iteration_wall_time_s",
    ):
        value = latest.get(key)
        if value in (None, ""):
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _wave_candidate_limit(args: argparse.Namespace, *, remaining_seconds: float) -> int:
    hard_cap = max(1, args.max_candidates_per_wave)
    if remaining_seconds < 15 * 60 or args.hours <= 0.05:
        return min(1, hard_cap)
    if remaining_seconds < 45 * 60 or args.hours <= 0.25:
        return min(2, hard_cap)
    return min(3, hard_cap)


def _wave_iterations(stage: str, *, remaining_seconds: float) -> int | None:
    if remaining_seconds < 20 * 60:
        return 1
    if stage == "competence":
        return 2
    return 3


def _swarm_steps(stage: str, *, remaining_seconds: float) -> int | None:
    if remaining_seconds < 15 * 60:
        return 60 if stage == "competence" else 90
    if remaining_seconds < 45 * 60:
        return 120 if stage == "competence" else 180
    return None


def _select_family_candidates(
    candidates: list[CandidateSpec],
    *,
    limit: int,
    offset: int,
) -> list[CandidateSpec]:
    if not candidates or limit <= 0:
        return []
    start = offset % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    return ordered[:limit]


def run_campaign(args: argparse.Namespace) -> int:
    run_paths = new_run_paths(prefix="campaign")
    started = time.time()
    deadline = started + max(args.hours, 0.0) * 3600.0
    baseline_checkpoint = _load_current_champion_checkpoint(args.baseline_checkpoint)
    champion_summary = _load_current_champion_summary()
    if champion_summary is not None and str(champion_summary.get("checkpoint")) == str(resolve_checkpoint(baseline_checkpoint)):
        baseline_summary = champion_summary
        baseline_summary_path = CHAMPIONS_DIR / "current.json"
    else:
        baseline_summary_path, baseline_summary = run_benchmark(
            baseline_checkpoint,
            profile_name="quick",
            seed=101,
            export_clips=False,
            promote_if_best=False,
            seed_tier="promotion",
        )

    stage = infer_stage_from_aggregate(baseline_summary["aggregate"]) if args.stage == "auto" else args.stage
    state = CampaignState(stage=stage)
    if args.start_family is not None:
        state.family_index = FAMILY_ORDER.index(args.start_family)
    incumbent_checkpoint = str(resolve_checkpoint(baseline_checkpoint))
    incumbent_promotion_summary = baseline_summary
    incumbent_diagnostic_summary = (
        run_benchmark(
            incumbent_checkpoint,
            profile_name=get_stage_config(stage).benchmark_profile,
            seed=101,
            export_clips=False,
            promote_if_best=False,
            seed_tier="diagnostic",
            rotation_index=0,
        )[1]
        if champion_summary is None
        else baseline_summary
    )

    manifest: dict[str, Any] = {
        "source_commit": _git_commit_hash(),
        "started_at_unix": started,
        "budget_hours": args.hours,
        "baseline_checkpoint": incumbent_checkpoint,
        "baseline_summary_path": str(baseline_summary_path),
        "waves": [],
    }
    manifest_path = run_paths.root / "campaign_manifest.json"
    manifest["current_stage"] = state.stage
    manifest["current_incumbent_checkpoint"] = incumbent_checkpoint
    manifest["updated_at_unix"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    while (time.time() < deadline or (state.wave_index == 0 and args.max_waves > 0)) and state.wave_index < args.max_waves:
        stage_cfg = get_stage_config(state.stage)
        logical_cars = _logical_cars_for_stage(state.stage, args.start_cars, args.max_cars)
        family = FAMILY_ORDER[state.family_index]
        library = candidate_library(state.stage, logical_cars, args.max_cars)
        remaining_seconds = max(0.0, deadline - time.time())
        candidate_limit = _wave_candidate_limit(args, remaining_seconds=remaining_seconds)
        iterations_override = _wave_iterations(state.stage, remaining_seconds=remaining_seconds)
        swarm_steps = _swarm_steps(state.stage, remaining_seconds=remaining_seconds)
        candidates = _select_family_candidates(
            library[family],
            limit=candidate_limit,
            offset=state.wave_index,
        )
        wave_record: dict[str, Any] = {
            "wave_index": state.wave_index,
            "stage": state.stage,
            "family": family,
            "rotation_index": state.rotation_index,
            "logical_cars": logical_cars,
            "candidate_limit": candidate_limit,
            "iterations_override": iterations_override,
            "swarm_steps": swarm_steps,
            "candidates": [],
            "kept": None,
            "reason": None,
            "promoted_to_champion": False,
        }
        best_candidate_record: dict[str, Any] | None = None
        best_candidate_summary = incumbent_promotion_summary
        best_checkpoint = incumbent_checkpoint
        promoted_this_wave = False

        for candidate in candidates:
            checkpoint_path = _train_candidate(
                candidate=candidate,
                stage=state.stage,
                device=args.device,
                logical_cars=logical_cars,
                iterations_override=iterations_override,
            )
            training_metrics = _load_training_metrics(checkpoint_path)
            promotion_summary_path, promotion_summary = run_benchmark(
                checkpoint_path,
                profile_name=stage_cfg.benchmark_profile,
                seed=101,
                export_clips=args.benchmark_clips,
                promote_if_best=False,
                seed_tier="promotion",
            )
            diagnostic_summary_path, diagnostic_summary = run_benchmark(
                checkpoint_path,
                profile_name=stage_cfg.benchmark_profile,
                seed=101,
                export_clips=False,
                promote_if_best=False,
                seed_tier="diagnostic",
                rotation_index=state.rotation_index,
            )
            holdout_summary_path: str | None = None
            holdout_summary: dict[str, Any] | None = None
            if args.holdout_every > 0 and (state.wave_index + 1) % args.holdout_every == 0:
                holdout_summary_file, holdout_summary = run_benchmark(
                    checkpoint_path,
                    profile_name=stage_cfg.benchmark_profile,
                    seed=101,
                    export_clips=False,
                    promote_if_best=False,
                    seed_tier="holdout",
                )
                holdout_summary_path = str(holdout_summary_file)

            swarm_args = parse_swarm_args(
                [
                    "--policy",
                    "checkpoint",
                    "--checkpoint",
                    checkpoint_path,
                    "--cars",
                    str(logical_cars),
                    "--swarm-stage",
                    state.stage,
                    *(["--steps", str(swarm_steps)] if swarm_steps is not None else []),
                    "--headless",
                    *(["--export-clips"] if args.swarm_clips else []),
                ]
            )
            swarm_summary_path, swarm_summary = run_swarm(swarm_args)

            better, compare_reason = compare_aggregate_for_stage(
                state.stage,
                promotion_summary["aggregate"],
                best_candidate_summary["aggregate"],
            )
            collapsed, collapse_reason = _diagnostic_collapse(
                state.stage,
                diagnostic_summary["aggregate"],
                incumbent_diagnostic_summary["aggregate"],
            )
            keep = bool(better and not collapsed)
            candidate_record = {
                "name": candidate.name,
                "family": candidate.family,
                "checkpoint": checkpoint_path,
                "env_overrides": candidate.env_overrides,
                "ppo_overrides": candidate.ppo_overrides,
                "training_metrics": training_metrics,
                "promotion_summary_path": str(promotion_summary_path),
                "diagnostic_summary_path": str(diagnostic_summary_path),
                "holdout_summary_path": holdout_summary_path,
                "swarm_summary_path": str(swarm_summary_path),
                "compare_reason": compare_reason,
                "diagnostic_collapse": collapse_reason if collapsed else None,
                "kept": keep,
                "promotion_aggregate": promotion_summary["aggregate"],
                "diagnostic_aggregate": diagnostic_summary["aggregate"],
                "holdout_aggregate": holdout_summary["aggregate"] if holdout_summary is not None else None,
                "swarm_aggregate": swarm_summary["aggregate"],
            }
            wave_record["candidates"].append(candidate_record)

            if keep:
                promoted_this_wave = True
                best_candidate_record = candidate_record
                best_candidate_summary = promotion_summary
                best_checkpoint = checkpoint_path

        if promoted_this_wave and best_candidate_record is not None:
            incumbent_checkpoint = best_checkpoint
            incumbent_promotion_summary = best_candidate_summary
            incumbent_diagnostic_summary = run_benchmark(
                incumbent_checkpoint,
                profile_name=stage_cfg.benchmark_profile,
                seed=101,
                export_clips=False,
                promote_if_best=False,
                seed_tier="diagnostic",
                rotation_index=state.rotation_index,
            )[1]
            promoted_summary = run_benchmark(
                incumbent_checkpoint,
                profile_name=stage_cfg.benchmark_profile,
                seed=101,
                export_clips=False,
                promote_if_best=True,
                seed_tier="promotion",
            )[1]
            wave_record["kept"] = best_candidate_record["name"]
            wave_record["reason"] = best_candidate_record["compare_reason"]
            wave_record["promoted_to_champion"] = bool(promoted_summary.get("promoted", False))
            state.family_no_promotion_waves = 0
            if stage_gate_met(state.stage, incumbent_promotion_summary["aggregate"]):
                next_stage_name = next_stage(state.stage)
                if next_stage_name != state.stage:
                    wave_record["stage_advanced_to"] = next_stage_name
                    state.stage = next_stage_name
        else:
            wave_record["reason"] = "no candidate beat incumbent"
            state.family_no_promotion_waves += 1
            if state.family_no_promotion_waves >= 3:
                old_family = family
                state.family_index = (state.family_index + 1) % len(FAMILY_ORDER)
                state.family_no_promotion_waves = 0
                if state.family_index == 0:
                    state.family_rotations_without_promotion += 1
                    if state.family_rotations_without_promotion >= 2:
                        state.hybrid_enabled = True
                wave_record["family_advanced_from"] = old_family
                wave_record["family_advanced_to"] = FAMILY_ORDER[state.family_index]

        wave_record["incumbent_checkpoint"] = incumbent_checkpoint
        wave_record["incumbent_aggregate"] = incumbent_promotion_summary["aggregate"]
        wave_record["hybrid_enabled"] = state.hybrid_enabled
        manifest["waves"].append(wave_record)
        manifest["current_stage"] = state.stage
        manifest["current_incumbent_checkpoint"] = incumbent_checkpoint
        manifest["updated_at_unix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        clean_artifacts(
            ARTIFACTS_DIR,
            RetentionPolicy(
                keep_runs_per_prefix=args.keep_runs_per_prefix,
                keep_days=args.keep_days,
            ),
            dry_run=False,
        )

        state.wave_index += 1
        state.rotation_index += 1

    manifest["completed_at_unix"] = time.time()
    manifest["final_stage"] = state.stage
    manifest["final_incumbent_checkpoint"] = incumbent_checkpoint
    manifest["final_incumbent_aggregate"] = incumbent_promotion_summary["aggregate"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"campaign_complete waves={state.wave_index} stage={state.stage} "
        f"checkpoint={incumbent_checkpoint} manifest={manifest_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
