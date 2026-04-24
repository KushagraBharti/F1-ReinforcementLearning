"""Benchmark and champion workflow for torch-native checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from f1rl.artifacts import new_run_paths, owning_run_dir, resolve_checkpoint
from f1rl.constants import CHAMPIONS_DIR
from f1rl.inference import deep_merge_dicts, training_env_config_for_checkpoint
from f1rl.stages import resolve_seed_list
from f1rl.torch_runtime import TERMINATION_REASONS, TelemetryThresholds, TorchSimBatch
from f1rl.torch_agent import load_torch_policy


@dataclass(slots=True, frozen=True)
class BenchmarkProfile:
    name: str
    episodes: int
    max_steps: int
    clip_length: int


BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "quick": BenchmarkProfile(name="quick", episodes=3, max_steps=450, clip_length=30),
    "standard": BenchmarkProfile(name="standard", episodes=5, max_steps=750, clip_length=45),
}


@dataclass(slots=True)
class EpisodeBenchmark:
    seed: int
    completed_lap: bool
    collided: bool
    lap_count: int
    steps: int
    lap_time_steps: int | None
    total_reward: float
    avg_speed: float
    checkpoints_reached: int
    max_checkpoint_streak: int
    progress_ratio: float
    reached_first_checkpoint: bool
    reached_three_checkpoints: bool
    clean_100_step_survived: bool
    clean_300_step_survived: bool
    reverse_events: int
    negative_speed_fraction: float
    longest_reverse_streak: int
    stall_fraction: float
    steering_oscillation: float
    steering_saturation_fraction: float
    min_wall_margin: float
    mean_abs_heading_error: float
    termination_reason: str | None


def get_benchmark_profile(name: str) -> BenchmarkProfile:
    try:
        return BENCHMARK_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark profile: {name}") from exc


def build_benchmark_env_config(profile: BenchmarkProfile, export_clips: bool) -> dict[str, Any]:
    return {
        "max_steps": profile.max_steps,
        "max_laps": 1,
        "headless": True,
        "render_mode": None,
        "render": {
            "enabled": False,
            "export_frames": export_clips,
            "draw_hud": True,
            "frame_prefix": f"benchmark-{profile.name}",
        },
    }


def _slice_clip_windows(frames: list[np.ndarray], clip_length: int) -> dict[str, list[np.ndarray]]:
    if not frames:
        return {}
    total_frames = len(frames)
    actual_clip_length = min(clip_length, total_frames)

    def bounded_window(center_index: int) -> list[np.ndarray]:
        start = max(0, min(total_frames - actual_clip_length, center_index - actual_clip_length // 2))
        end = start + actual_clip_length
        return frames[start:end]

    return {
        "early": frames[:actual_clip_length],
        "mid": bounded_window(total_frames // 2),
        "late": frames[-actual_clip_length:],
    }


def _write_gif(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    imageio.mimsave(path, [np.asarray(frame) for frame in frames], duration=1 / 12)


def _aggregate_episode_metrics(
    *,
    seed: int,
    simulator: TorchSimBatch,
    total_reward: float,
    step_count: int,
    collided: bool,
    progress_total: int,
    max_checkpoint_streak: int,
    reverse_events: int,
    negative_speed_steps: int,
    longest_reverse_streak: int,
    speed_samples: list[float],
    steering_actions: list[float],
    stall_steps: int,
    min_wall_margin: float,
    heading_error_samples: list[float],
    termination_reason: str | None,
) -> EpisodeBenchmark:
    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    steering_oscillation = (
        float(np.mean(np.abs(np.diff(np.asarray(steering_actions, dtype=np.float32))))) if len(steering_actions) > 1 else 0.0
    )
    steering_saturation_fraction = (
        float(np.mean(np.abs(np.asarray(steering_actions, dtype=np.float32)) >= 0.9)) if steering_actions else 0.0
    )
    progress_ratio = float(progress_total / max(simulator.track.definition.num_goals, 1))
    completed_lap = simulator.lap_count[0].item() >= 1 and not collided
    return EpisodeBenchmark(
        seed=seed,
        completed_lap=completed_lap,
        collided=collided,
        lap_count=int(simulator.lap_count[0].item()),
        steps=step_count,
        lap_time_steps=step_count if completed_lap else None,
        total_reward=float(total_reward),
        avg_speed=avg_speed,
        checkpoints_reached=int(progress_total),
        max_checkpoint_streak=int(max_checkpoint_streak),
        progress_ratio=progress_ratio,
        reached_first_checkpoint=progress_total >= 1,
        reached_three_checkpoints=progress_total >= 3,
        clean_100_step_survived=step_count >= 100 and not collided,
        clean_300_step_survived=step_count >= 300 and not collided,
        reverse_events=reverse_events,
        negative_speed_fraction=float(negative_speed_steps / max(step_count, 1)),
        longest_reverse_streak=int(longest_reverse_streak),
        stall_fraction=float(stall_steps / max(step_count, 1)),
        steering_oscillation=steering_oscillation,
        steering_saturation_fraction=steering_saturation_fraction,
        min_wall_margin=float(min_wall_margin),
        mean_abs_heading_error=float(np.mean(heading_error_samples)) if heading_error_samples else 0.0,
        termination_reason=termination_reason,
    )


def run_benchmark(
    checkpoint: str,
    *,
    profile_name: str,
    seed: int,
    export_clips: bool,
    promote_if_best: bool,
    seeds: list[int] | None = None,
    seed_tier: str | None = None,
    rotation_index: int = 0,
) -> tuple[Path, dict[str, Any]]:
    profile = get_benchmark_profile(profile_name)
    checkpoint_path = resolve_checkpoint(checkpoint)
    policy = load_torch_policy(checkpoint_path)
    run_paths = new_run_paths(prefix=f"benchmark-{profile.name}")
    env_config = deep_merge_dicts(
        training_env_config_for_checkpoint(checkpoint_path),
        build_benchmark_env_config(profile, export_clips),
    )
    resolved_seeds = (
        list(seeds)
        if seeds is not None
        else resolve_seed_list(profile.name, seed_tier, rotation_index)
        if seed_tier is not None
        else [seed + episode_idx for episode_idx in range(profile.episodes)]
    )
    episodes: list[EpisodeBenchmark] = []
    episode_frames: list[np.ndarray] = []
    failure_heatmap: dict[str, int] = {}

    for episode_idx, episode_seed in enumerate(resolved_seeds):
        simulator = TorchSimBatch(env_config, num_cars=1, device=policy.device, telemetry_thresholds=TelemetryThresholds())
        observations = simulator.reset(seed=episode_seed)
        total_reward = 0.0
        step_count = 0
        collided = False
        progress_total = 0
        current_checkpoint_streak = 0
        max_checkpoint_streak = 0
        reverse_events = 0
        negative_speed_steps = 0
        current_reverse_streak = 0
        longest_reverse_streak = 0
        speed_samples: list[float] = []
        steering_actions: list[float] = []
        stall_steps = 0
        heading_error_samples: list[float] = []
        min_wall_margin = 1.0
        termination_reason: str | None = None
        while step_count < profile.max_steps:
            action = policy.compute_actions(observations, deterministic=True)
            steering_actions.append(float(action[0, 0].item()))
            step = simulator.step(action)
            observations = step.observations
            total_reward += float(step.rewards[0].item())
            step_count += 1
            progress_delta = int(step.telemetry["progress_delta"][0].item())
            progress_total += max(progress_delta, 0)
            if progress_delta > 0:
                current_checkpoint_streak += progress_delta
                max_checkpoint_streak = max(max_checkpoint_streak, current_checkpoint_streak)
            elif progress_delta < 0:
                current_checkpoint_streak = 0
                reverse_events += 1

            speed = float(step.telemetry["speed"][0].item())
            speed_samples.append(speed)
            stall_steps += int(abs(speed) < simulator.config.reward.stall_speed_threshold)
            heading_error_samples.append(abs(float(step.telemetry["heading_error_norm"][0].item())))
            min_wall_margin = min(min_wall_margin, float(step.telemetry["min_wall_distance_ratio"][0].item()))
            if speed < 0.0:
                negative_speed_steps += 1
                current_reverse_streak += 1
                longest_reverse_streak = max(longest_reverse_streak, current_reverse_streak)
            else:
                current_reverse_streak = 0

            if bool((step.terminated | step.truncated)[0].item()):
                code = int(step.telemetry["termination_code"][0].item())
                termination_reason = TERMINATION_REASONS[code]
                collided = termination_reason == "collision"
                failure_heatmap[str(int(step.telemetry["current_checkpoint_index"][0].item()))] = (
                    failure_heatmap.get(str(int(step.telemetry["current_checkpoint_index"][0].item())), 0) + 1
                )
                break

        episodes.append(
            _aggregate_episode_metrics(
                seed=episode_seed,
                simulator=simulator,
                total_reward=total_reward,
                step_count=step_count,
                collided=collided,
                progress_total=progress_total,
                max_checkpoint_streak=max_checkpoint_streak,
                reverse_events=reverse_events,
                negative_speed_steps=negative_speed_steps,
                longest_reverse_streak=longest_reverse_streak,
                speed_samples=speed_samples,
                steering_actions=steering_actions,
                stall_steps=stall_steps,
                min_wall_margin=min_wall_margin,
                heading_error_samples=heading_error_samples,
                termination_reason=termination_reason,
            )
        )

    clip_paths: dict[str, str] = {}
    if export_clips:
        for label, frames in _slice_clip_windows(episode_frames, profile.clip_length).items():
            output_path = run_paths.renders / f"{label}.gif"
            _write_gif(output_path, frames)
            if output_path.exists():
                clip_paths[label] = str(output_path)

    summary = aggregate_benchmark_summary(
        checkpoint_path=checkpoint_path,
        profile=profile,
        episodes=episodes,
        clip_paths=clip_paths,
    )
    summary["backend"] = "torch_native"
    summary["seed_tier"] = seed_tier
    summary["seed_rotation_index"] = rotation_index
    summary["seeds"] = resolved_seeds
    summary["failure_heatmap"] = failure_heatmap
    summary["runtime"] = TorchSimBatch(env_config, num_cars=1, device=policy.device).runtime_metadata(
        policy_device=policy.device,
        renderer_backend="headless",
    )
    promoted, comparison_reason = maybe_promote_benchmark(summary, checkpoint_path, run_paths.root, promote_if_best)
    summary["promoted"] = promoted
    summary["comparison_reason"] = comparison_reason

    summary_path = run_paths.root / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary_path, summary


def aggregate_benchmark_summary(
    *,
    checkpoint_path: Path,
    profile: BenchmarkProfile,
    episodes: list[EpisodeBenchmark],
    clip_paths: dict[str, str],
) -> dict[str, Any]:
    completion_rate = float(np.mean([episode.completed_lap for episode in episodes])) if episodes else 0.0
    collision_rate = float(np.mean([episode.collided for episode in episodes])) if episodes else 0.0
    progress_ratio = float(np.mean([episode.progress_ratio for episode in episodes])) if episodes else 0.0
    avg_reward = float(np.mean([episode.total_reward for episode in episodes])) if episodes else 0.0
    avg_speed = float(np.mean([episode.avg_speed for episode in episodes])) if episodes else 0.0
    avg_checkpoints_reached = float(np.mean([episode.checkpoints_reached for episode in episodes])) if episodes else 0.0
    max_checkpoints_reached = max((episode.checkpoints_reached for episode in episodes), default=0)
    avg_checkpoint_streak = float(np.mean([episode.max_checkpoint_streak for episode in episodes])) if episodes else 0.0
    best_checkpoint_streak = max((episode.max_checkpoint_streak for episode in episodes), default=0)
    first_checkpoint_rate = float(np.mean([episode.reached_first_checkpoint for episode in episodes])) if episodes else 0.0
    three_checkpoint_rate = float(np.mean([episode.reached_three_checkpoints for episode in episodes])) if episodes else 0.0
    clean_100_step_survival_rate = float(np.mean([episode.clean_100_step_survived for episode in episodes])) if episodes else 0.0
    clean_300_step_survival_rate = float(np.mean([episode.clean_300_step_survived for episode in episodes])) if episodes else 0.0
    avg_negative_speed_fraction = float(np.mean([episode.negative_speed_fraction for episode in episodes])) if episodes else 0.0
    avg_reverse_events = float(np.mean([episode.reverse_events for episode in episodes])) if episodes else 0.0
    avg_longest_reverse_streak = float(np.mean([episode.longest_reverse_streak for episode in episodes])) if episodes else 0.0
    avg_stall_fraction = float(np.mean([episode.stall_fraction for episode in episodes])) if episodes else 0.0
    avg_steering_oscillation = float(np.mean([episode.steering_oscillation for episode in episodes])) if episodes else 0.0
    avg_steering_saturation_fraction = float(np.mean([episode.steering_saturation_fraction for episode in episodes])) if episodes else 0.0
    avg_wall_margin_min = float(np.mean([episode.min_wall_margin for episode in episodes])) if episodes else 0.0
    avg_heading_error_abs = float(np.mean([episode.mean_abs_heading_error for episode in episodes])) if episodes else 0.0
    completed_lap_times = [episode.lap_time_steps for episode in episodes if episode.lap_time_steps is not None]
    failure_histogram: dict[str, int] = {}
    for episode in episodes:
        reason = episode.termination_reason or "unknown"
        failure_histogram[reason] = failure_histogram.get(reason, 0) + 1

    return {
        "checkpoint": str(checkpoint_path),
        "profile": profile.name,
        "episodes": [asdict(episode) for episode in episodes],
        "aggregate": {
            "completion_rate": completion_rate,
            "collision_rate": collision_rate,
            "first_checkpoint_rate": first_checkpoint_rate,
            "three_checkpoint_rate": three_checkpoint_rate,
            "avg_checkpoints_reached": avg_checkpoints_reached,
            "max_checkpoints_reached": max_checkpoints_reached,
            "avg_checkpoint_streak": avg_checkpoint_streak,
            "best_checkpoint_streak": best_checkpoint_streak,
            "clean_100_step_survival_rate": clean_100_step_survival_rate,
            "clean_300_step_survival_rate": clean_300_step_survival_rate,
            "avg_progress_ratio": progress_ratio,
            "avg_reward": avg_reward,
            "avg_speed": avg_speed,
            "avg_negative_speed_fraction": avg_negative_speed_fraction,
            "avg_reverse_events": avg_reverse_events,
            "avg_longest_reverse_streak": avg_longest_reverse_streak,
            "avg_stall_fraction": avg_stall_fraction,
            "avg_steering_oscillation": avg_steering_oscillation,
            "avg_steering_saturation_fraction": avg_steering_saturation_fraction,
            "avg_wall_margin_min": avg_wall_margin_min,
            "avg_heading_error_abs": avg_heading_error_abs,
            "avg_lap_time_steps": float(np.mean(completed_lap_times)) if completed_lap_times else None,
            "best_lap_time_steps": min(completed_lap_times) if completed_lap_times else None,
            "lap_time_std_steps": float(np.std(completed_lap_times)) if len(completed_lap_times) > 1 else 0.0,
        },
        "failure_histogram": failure_histogram,
        "clip_paths": clip_paths,
    }


def compare_benchmark_summaries(candidate: dict[str, Any], champion: dict[str, Any]) -> tuple[bool, str]:
    cand = candidate["aggregate"]
    champ = champion["aggregate"]

    if cand.get("completion_rate", 0.0) > champ.get("completion_rate", 0.0):
        return True, "higher completion rate"
    if cand.get("completion_rate", 0.0) < champ.get("completion_rate", 0.0):
        return False, "lower completion rate"
    cand_lap = cand.get("avg_lap_time_steps")
    champ_lap = champ.get("avg_lap_time_steps")
    if cand_lap is not None and champ_lap is not None and cand_lap != champ_lap:
        return cand_lap < champ_lap, "lap time comparison"
    if cand_lap is not None and champ_lap is None:
        return True, "candidate completed laps while champion did not"
    if cand_lap is None and champ_lap is not None:
        return False, "champion completed laps while candidate did not"
    if cand.get("collision_rate", 1.0) != champ.get("collision_rate", 1.0):
        return cand.get("collision_rate", 1.0) < champ.get("collision_rate", 1.0), "collision rate comparison"
    if cand.get("first_checkpoint_rate", 0.0) != champ.get("first_checkpoint_rate", 0.0):
        return cand.get("first_checkpoint_rate", 0.0) > champ.get("first_checkpoint_rate", 0.0), "first checkpoint comparison"
    if cand.get("three_checkpoint_rate", 0.0) != champ.get("three_checkpoint_rate", 0.0):
        return cand.get("three_checkpoint_rate", 0.0) > champ.get("three_checkpoint_rate", 0.0), "three checkpoint comparison"
    if cand.get("avg_checkpoints_reached", 0.0) != champ.get("avg_checkpoints_reached", 0.0):
        return cand.get("avg_checkpoints_reached", 0.0) > champ.get("avg_checkpoints_reached", 0.0), "checkpoint reach comparison"
    if cand.get("clean_100_step_survival_rate", 0.0) != champ.get("clean_100_step_survival_rate", 0.0):
        return cand.get("clean_100_step_survival_rate", 0.0) > champ.get("clean_100_step_survival_rate", 0.0), "clean 100-step survival comparison"
    if cand.get("avg_negative_speed_fraction", 1.0) != champ.get("avg_negative_speed_fraction", 1.0):
        return cand.get("avg_negative_speed_fraction", 1.0) < champ.get("avg_negative_speed_fraction", 1.0), "negative speed comparison"
    if cand.get("avg_longest_reverse_streak", 1.0) != champ.get("avg_longest_reverse_streak", 1.0):
        return cand.get("avg_longest_reverse_streak", 1.0) < champ.get("avg_longest_reverse_streak", 1.0), "reverse streak comparison"
    if cand.get("avg_steering_oscillation", 1.0) != champ.get("avg_steering_oscillation", 1.0):
        return cand.get("avg_steering_oscillation", 1.0) < champ.get("avg_steering_oscillation", 1.0), "steering oscillation comparison"
    if cand.get("avg_stall_fraction", 1.0) != champ.get("avg_stall_fraction", 1.0):
        return cand.get("avg_stall_fraction", 1.0) < champ.get("avg_stall_fraction", 1.0), "stall fraction comparison"
    if cand.get("avg_progress_ratio", 0.0) != champ.get("avg_progress_ratio", 0.0):
        return cand.get("avg_progress_ratio", 0.0) > champ.get("avg_progress_ratio", 0.0), "progress ratio comparison"
    return cand.get("avg_reward", 0.0) > champ.get("avg_reward", 0.0), "reward tiebreak"


def _pin_run(path: Path) -> None:
    (path / ".pin").write_text("promoted\n", encoding="utf-8")


def maybe_promote_benchmark(
    summary: dict[str, Any],
    checkpoint_path: Path,
    benchmark_run_dir: Path,
    promote_if_best: bool,
) -> tuple[bool, str]:
    if not promote_if_best:
        return False, "promotion disabled"

    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    champion_path = CHAMPIONS_DIR / "current.json"
    if champion_path.exists():
        champion = json.loads(champion_path.read_text(encoding="utf-8"))
        champion_aggregate = champion["aggregate"]
        candidate_aggregate = summary["aggregate"]
        if champion_aggregate.get("avg_speed", 0.0) > 0.0 and candidate_aggregate.get("avg_speed", 0.0) <= 0.0:
            return False, "failed positive speed floor"
        if champion_aggregate.get("avg_progress_ratio", 0.0) > 0.0 and candidate_aggregate.get("avg_progress_ratio", 0.0) <= 0.0:
            return False, "failed positive progress floor"
        if champion_aggregate.get("avg_speed", 0.0) <= 0.0 and candidate_aggregate.get("avg_speed", 0.0) > 0.0:
            reason = "cleared positive speed floor"
        elif champion_aggregate.get("avg_progress_ratio", 0.0) <= 0.0 and candidate_aggregate.get("avg_progress_ratio", 0.0) > 0.0:
            reason = "cleared positive progress floor"
        else:
            is_better, reason = compare_benchmark_summaries(summary, champion)
            if not is_better:
                return False, f"not better than champion: {reason}"
    else:
        reason = "first benchmark champion"
    owning_run = owning_run_dir(checkpoint_path)
    _pin_run(owning_run)
    _pin_run(benchmark_run_dir)
    champion_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return True, reason


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a saved checkpoint and compare against champion.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--profile", choices=sorted(BENCHMARK_PROFILES), default="quick")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seed-tier", choices=["promotion", "diagnostic", "holdout"], default=None)
    parser.add_argument("--rotation-index", type=int, default=0)
    parser.add_argument("--no-clips", action="store_true", help="Disable GIF export for sampled windows.")
    parser.add_argument("--promote-if-best", action="store_true", help="Pin and promote if this beats champion.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path, summary = run_benchmark(
        args.checkpoint,
        profile_name=args.profile,
        seed=args.seed,
        export_clips=not args.no_clips,
        promote_if_best=args.promote_if_best,
        seed_tier=args.seed_tier,
        rotation_index=args.rotation_index,
    )
    print(
        f"benchmark_complete checkpoint={summary['checkpoint']} profile={summary['profile']} "
        f"completion_rate={summary['aggregate']['completion_rate']:.3f} "
        f"collision_rate={summary['aggregate']['collision_rate']:.3f} "
        f"backend={summary['backend']} summary={summary_path} promoted={summary['promoted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
