"""Benchmark and champion workflow for trained checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from f1rl.artifacts import new_run_paths, owning_run_dir, resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.constants import CHAMPIONS_DIR
from f1rl.env import F1RaceEnv
from f1rl.inference import deep_merge_dicts, default_eval_env_config, load_inference_policy


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
    progress_ratio: float
    reverse_events: int
    negative_speed_fraction: float
    longest_reverse_streak: int
    stall_fraction: float
    steering_oscillation: float


def get_benchmark_profile(name: str) -> BenchmarkProfile:
    try:
        return BENCHMARK_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark profile: {name}") from exc


def build_benchmark_env_config(profile: BenchmarkProfile, export_clips: bool) -> dict[str, Any]:
    config = EnvConfig.from_dict(
        default_eval_env_config(
            steps=profile.max_steps,
            render=False,
            headless=True,
            export_frames=export_clips,
            frame_prefix=f"benchmark-{profile.name}",
        )
    )
    config.max_steps = profile.max_steps
    config.max_laps = 1
    config.render.draw_hud = True
    return to_dict(config)


def _training_env_config_for_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    run_dir = owning_run_dir(checkpoint_path)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = metadata.get("env_config", {})
    if isinstance(env_config, dict):
        return env_config
    return {}


def _clip_ranges(profile: BenchmarkProfile) -> dict[str, range]:
    clip_len = profile.clip_length
    centers = {
        "early": clip_len,
        "mid": profile.max_steps // 2,
        "late": max(clip_len, profile.max_steps - clip_len),
    }
    ranges: dict[str, range] = {}
    for label, center in centers.items():
        start = max(0, center - clip_len // 2)
        end = min(profile.max_steps, start + clip_len)
        ranges[label] = range(start, end)
    return ranges


def _write_gif(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    imageio.mimsave(path, [np.asarray(frame) for frame in frames], duration=1 / 12)


def _aggregate_episode_metrics(
    *,
    seed: int,
    env: F1RaceEnv,
    total_reward: float,
    step_count: int,
    collided: bool,
    progress_total: int,
    reverse_events: int,
    negative_speed_steps: int,
    longest_reverse_streak: int,
    speed_samples: list[float],
    steering_actions: list[float],
    stall_steps: int,
) -> EpisodeBenchmark:
    avg_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    steering_oscillation = (
        float(np.mean(np.abs(np.diff(np.asarray(steering_actions, dtype=np.float32)))))
        if len(steering_actions) > 1
        else 0.0
    )
    progress_ratio = float(progress_total / max(env.track.num_goals, 1))
    completed_lap = env.car.lap_count >= 1 and not collided
    return EpisodeBenchmark(
        seed=seed,
        completed_lap=completed_lap,
        collided=collided,
        lap_count=int(env.car.lap_count),
        steps=step_count,
        lap_time_steps=step_count if completed_lap else None,
        total_reward=float(total_reward),
        avg_speed=avg_speed,
        progress_ratio=progress_ratio,
        reverse_events=reverse_events,
        negative_speed_fraction=float(negative_speed_steps / max(step_count, 1)),
        longest_reverse_streak=int(longest_reverse_streak),
        stall_fraction=float(stall_steps / max(step_count, 1)),
        steering_oscillation=steering_oscillation,
    )


def run_benchmark(
    checkpoint: str,
    *,
    profile_name: str,
    seed: int,
    export_clips: bool,
    promote_if_best: bool,
) -> tuple[Path, dict[str, Any]]:
    profile = get_benchmark_profile(profile_name)
    checkpoint_path = resolve_checkpoint(checkpoint)
    policy = load_inference_policy(checkpoint_path)
    run_paths = new_run_paths(prefix=f"benchmark-{profile.name}")
    env = F1RaceEnv(
        deep_merge_dicts(
            _training_env_config_for_checkpoint(checkpoint_path),
            build_benchmark_env_config(profile, export_clips),
        )
    )
    clip_ranges = _clip_ranges(profile)
    clip_frames: dict[str, list[np.ndarray]] = {label: [] for label in clip_ranges}
    episodes: list[EpisodeBenchmark] = []

    try:
        for episode_idx in range(profile.episodes):
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            step_count = 0
            collided = False
            progress_total = 0
            reverse_events = 0
            negative_speed_steps = 0
            current_reverse_streak = 0
            longest_reverse_streak = 0
            speed_samples: list[float] = []
            steering_actions: list[float] = []
            stall_steps = 0

            while step_count < profile.max_steps:
                action = policy.compute_action(obs, explore=False)
                steering_actions.append(float(action[0]))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                collided = collided or bool(info["collided"])
                progress_total += max(int(info["progress_delta"]), 0)
                reverse_events += int(info["progress_delta"] < 0)
                speed = float(info["speed"])
                speed_samples.append(speed)
                stall_steps += int(float(abs(speed)) < env.config.reward.stall_speed_threshold)

                if speed < 0.0:
                    negative_speed_steps += 1
                    current_reverse_streak += 1
                    longest_reverse_streak = max(longest_reverse_streak, current_reverse_streak)
                else:
                    current_reverse_streak = 0

                if export_clips and episode_idx == 0:
                    for label, clip_range in clip_ranges.items():
                        if step_count - 1 in clip_range:
                            frame = env.render()
                            if frame is not None:
                                clip_frames[label].append(frame)

                if terminated or truncated:
                    break

            episodes.append(
                _aggregate_episode_metrics(
                    seed=episode_seed,
                    env=env,
                    total_reward=total_reward,
                    step_count=step_count,
                    collided=collided,
                    progress_total=progress_total,
                    reverse_events=reverse_events,
                    negative_speed_steps=negative_speed_steps,
                    longest_reverse_streak=longest_reverse_streak,
                    speed_samples=speed_samples,
                    steering_actions=steering_actions,
                    stall_steps=stall_steps,
                )
            )
    finally:
        env.close()

    clip_paths: dict[str, str] = {}
    if export_clips:
        for label, frames in clip_frames.items():
            output_path = run_paths.renders / f"{label}.gif"
            _write_gif(output_path, frames)
            clip_paths[label] = str(output_path)

    summary = aggregate_benchmark_summary(
        checkpoint_path=checkpoint_path,
        profile=profile,
        episodes=episodes,
        clip_paths=clip_paths,
    )
    summary["backend"] = policy.backend
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
    avg_negative_speed_fraction = (
        float(np.mean([episode.negative_speed_fraction for episode in episodes])) if episodes else 0.0
    )
    avg_reverse_events = float(np.mean([episode.reverse_events for episode in episodes])) if episodes else 0.0
    avg_longest_reverse_streak = (
        float(np.mean([episode.longest_reverse_streak for episode in episodes])) if episodes else 0.0
    )
    avg_stall_fraction = float(np.mean([episode.stall_fraction for episode in episodes])) if episodes else 0.0
    avg_steering_oscillation = (
        float(np.mean([episode.steering_oscillation for episode in episodes])) if episodes else 0.0
    )
    completed_lap_times = [episode.lap_time_steps for episode in episodes if episode.lap_time_steps is not None]

    return {
        "checkpoint": str(checkpoint_path),
        "profile": profile.name,
        "episodes": [asdict(episode) for episode in episodes],
        "aggregate": {
            "completion_rate": completion_rate,
            "collision_rate": collision_rate,
            "avg_progress_ratio": progress_ratio,
            "avg_reward": avg_reward,
            "avg_speed": avg_speed,
            "avg_negative_speed_fraction": avg_negative_speed_fraction,
            "avg_reverse_events": avg_reverse_events,
            "avg_longest_reverse_streak": avg_longest_reverse_streak,
            "avg_stall_fraction": avg_stall_fraction,
            "avg_steering_oscillation": avg_steering_oscillation,
            "avg_lap_time_steps": float(np.mean(completed_lap_times)) if completed_lap_times else None,
            "best_lap_time_steps": min(completed_lap_times) if completed_lap_times else None,
        },
        "clip_paths": clip_paths,
    }


def compare_benchmark_summaries(candidate: dict[str, Any], champion: dict[str, Any]) -> tuple[bool, str]:
    cand = candidate["aggregate"]
    champ = champion["aggregate"]

    if cand["completion_rate"] > champ["completion_rate"]:
        return True, "higher completion rate"
    if cand["completion_rate"] < champ["completion_rate"]:
        return False, "lower completion rate"

    cand_lap = cand["avg_lap_time_steps"]
    champ_lap = champ["avg_lap_time_steps"]
    if cand_lap is not None and champ_lap is not None and cand_lap != champ_lap:
        return cand_lap < champ_lap, "lap time comparison"
    if cand_lap is not None and champ_lap is None:
        return True, "candidate completed laps while champion did not"
    if cand_lap is None and champ_lap is not None:
        return False, "champion completed laps while candidate did not"

    if cand["collision_rate"] != champ["collision_rate"]:
        return cand["collision_rate"] < champ["collision_rate"], "collision rate comparison"

    if cand["avg_negative_speed_fraction"] != champ["avg_negative_speed_fraction"]:
        return cand["avg_negative_speed_fraction"] < champ["avg_negative_speed_fraction"], "negative speed comparison"

    if cand["avg_longest_reverse_streak"] != champ["avg_longest_reverse_streak"]:
        return cand["avg_longest_reverse_streak"] < champ["avg_longest_reverse_streak"], "reverse streak comparison"

    if cand["avg_steering_oscillation"] != champ["avg_steering_oscillation"]:
        return (
            cand["avg_steering_oscillation"] < champ["avg_steering_oscillation"],
            "steering oscillation comparison",
        )

    if cand["avg_stall_fraction"] != champ["avg_stall_fraction"]:
        return cand["avg_stall_fraction"] < champ["avg_stall_fraction"], "stall fraction comparison"

    if cand["avg_progress_ratio"] != champ["avg_progress_ratio"]:
        return cand["avg_progress_ratio"] > champ["avg_progress_ratio"], "progress ratio comparison"

    return cand["avg_reward"] > champ["avg_reward"], "reward tiebreak"


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

        if champion_aggregate["avg_speed"] > 0.0 and candidate_aggregate["avg_speed"] <= 0.0:
            return False, "failed positive speed floor"
        if champion_aggregate["avg_progress_ratio"] > 0.0 and candidate_aggregate["avg_progress_ratio"] <= 0.0:
            return False, "failed positive progress floor"

        if champion_aggregate["avg_speed"] <= 0.0 and candidate_aggregate["avg_speed"] > 0.0:
            reason = "cleared positive speed floor"
        elif champion_aggregate["avg_progress_ratio"] <= 0.0 and candidate_aggregate["avg_progress_ratio"] > 0.0:
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
