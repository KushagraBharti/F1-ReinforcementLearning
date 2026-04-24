import numpy as np

from f1rl.benchmark import (
    EpisodeBenchmark,
    _slice_clip_windows,
    aggregate_benchmark_summary,
    build_benchmark_env_config,
    compare_benchmark_summaries,
    get_benchmark_profile,
)
from f1rl.inference import deep_merge_dicts
from f1rl.swarm import _base_env_config


def _summary(
    *,
    completion_rate: float,
    collision_rate: float,
    avg_lap_time_steps: float | None,
    avg_progress_ratio: float,
    avg_reward: float = 0.0,
    first_checkpoint_rate: float = 0.0,
    three_checkpoint_rate: float = 0.0,
    avg_checkpoints_reached: float = 0.0,
    clean_100_step_survival_rate: float = 0.0,
    avg_negative_speed_fraction: float = 0.0,
    avg_longest_reverse_streak: float = 0.0,
    avg_stall_fraction: float = 0.0,
    avg_steering_oscillation: float = 0.0,
) -> dict:
    return {
        "aggregate": {
            "completion_rate": completion_rate,
            "collision_rate": collision_rate,
            "avg_lap_time_steps": avg_lap_time_steps,
            "first_checkpoint_rate": first_checkpoint_rate,
            "three_checkpoint_rate": three_checkpoint_rate,
            "avg_checkpoints_reached": avg_checkpoints_reached,
            "clean_100_step_survival_rate": clean_100_step_survival_rate,
            "avg_progress_ratio": avg_progress_ratio,
            "avg_reward": avg_reward,
            "avg_negative_speed_fraction": avg_negative_speed_fraction,
            "avg_longest_reverse_streak": avg_longest_reverse_streak,
            "avg_stall_fraction": avg_stall_fraction,
            "avg_steering_oscillation": avg_steering_oscillation,
        }
    }


def test_completion_rate_dominates_comparison() -> None:
    candidate = _summary(
        completion_rate=1.0,
        collision_rate=0.4,
        avg_lap_time_steps=200.0,
        avg_progress_ratio=0.8,
    )
    champion = _summary(
        completion_rate=0.5,
        collision_rate=0.0,
        avg_lap_time_steps=150.0,
        avg_progress_ratio=0.9,
    )
    is_better, reason = compare_benchmark_summaries(candidate, champion)
    assert is_better is True
    assert reason == "higher completion rate"


def test_lap_time_breaks_completion_tie() -> None:
    candidate = _summary(
        completion_rate=1.0,
        collision_rate=0.0,
        avg_lap_time_steps=180.0,
        avg_progress_ratio=1.0,
    )
    champion = _summary(
        completion_rate=1.0,
        collision_rate=0.0,
        avg_lap_time_steps=220.0,
        avg_progress_ratio=1.0,
    )
    is_better, reason = compare_benchmark_summaries(candidate, champion)
    assert is_better is True
    assert reason == "lap time comparison"


def test_negative_speed_breaks_tie_before_reward() -> None:
    candidate = _summary(
        completion_rate=0.0,
        collision_rate=1.0,
        avg_lap_time_steps=None,
        avg_progress_ratio=0.0,
        avg_negative_speed_fraction=0.1,
    )
    champion = _summary(
        completion_rate=0.0,
        collision_rate=1.0,
        avg_lap_time_steps=None,
        avg_progress_ratio=0.0,
        avg_negative_speed_fraction=0.4,
    )
    is_better, reason = compare_benchmark_summaries(candidate, champion)
    assert is_better is True
    assert reason == "negative speed comparison"


def test_aggregate_summary_includes_checkpoint_progress_metrics() -> None:
    summary = aggregate_benchmark_summary(
        checkpoint_path="checkpoint",
        profile=get_benchmark_profile("quick"),
        episodes=[
            EpisodeBenchmark(
                seed=1,
                completed_lap=False,
                collided=True,
                lap_count=0,
                steps=50,
                lap_time_steps=None,
                total_reward=1.0,
                avg_speed=2.0,
                checkpoints_reached=1,
                max_checkpoint_streak=1,
                progress_ratio=0.1,
                reached_first_checkpoint=True,
                reached_three_checkpoints=False,
                clean_100_step_survived=False,
                clean_300_step_survived=False,
                reverse_events=0,
                negative_speed_fraction=0.0,
                longest_reverse_streak=0,
                stall_fraction=0.0,
                steering_oscillation=0.1,
                steering_saturation_fraction=0.0,
                min_wall_margin=0.3,
                mean_abs_heading_error=0.2,
                termination_reason="collision",
            ),
            EpisodeBenchmark(
                seed=2,
                completed_lap=False,
                collided=True,
                lap_count=0,
                steps=50,
                lap_time_steps=None,
                total_reward=2.0,
                avg_speed=3.0,
                checkpoints_reached=4,
                max_checkpoint_streak=4,
                progress_ratio=0.4,
                reached_first_checkpoint=True,
                reached_three_checkpoints=True,
                clean_100_step_survived=False,
                clean_300_step_survived=False,
                reverse_events=0,
                negative_speed_fraction=0.0,
                longest_reverse_streak=0,
                stall_fraction=0.0,
                steering_oscillation=0.2,
                steering_saturation_fraction=0.0,
                min_wall_margin=0.4,
                mean_abs_heading_error=0.1,
                termination_reason="collision",
            ),
        ],
        clip_paths={},
    )

    aggregate = summary["aggregate"]
    assert aggregate["first_checkpoint_rate"] == 1.0
    assert aggregate["three_checkpoint_rate"] == 0.5
    assert aggregate["avg_checkpoints_reached"] == 2.5
    assert aggregate["max_checkpoints_reached"] == 4
    assert aggregate["avg_checkpoint_streak"] == 2.5
    assert aggregate["best_checkpoint_streak"] == 4


def test_clip_windows_are_based_on_actual_episode_length() -> None:
    frames = [np.full((2, 2, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

    clips = _slice_clip_windows(frames, clip_length=4)

    assert set(clips) == {"early", "mid", "late"}
    assert [int(frame[0, 0, 0]) for frame in clips["early"]] == [0, 1, 2, 3]
    assert [int(frame[0, 0, 0]) for frame in clips["mid"]] == [3, 4, 5, 6]
    assert [int(frame[0, 0, 0]) for frame in clips["late"]] == [6, 7, 8, 9]


def test_clip_windows_fall_back_to_available_frames() -> None:
    frames = [np.full((1, 1, 3), fill_value=i, dtype=np.uint8) for i in range(3)]

    clips = _slice_clip_windows(frames, clip_length=10)

    assert [int(frame[0, 0, 0]) for frame in clips["early"]] == [0, 1, 2]
    assert [int(frame[0, 0, 0]) for frame in clips["mid"]] == [0, 1, 2]
    assert [int(frame[0, 0, 0]) for frame in clips["late"]] == [0, 1, 2]


def test_benchmark_env_config_does_not_overwrite_training_sensor_count() -> None:
    training_env = {"dynamics": {"sensor_count": 11}}
    merged = deep_merge_dicts(
        training_env,
        build_benchmark_env_config(get_benchmark_profile("quick"), export_clips=False),
    )

    assert merged["dynamics"]["sensor_count"] == 11


def test_swarm_env_config_does_not_overwrite_training_sensor_count() -> None:
    class Args:
        steps = 60
        swarm_stage = "competence"
        render = False
        headless = True

    merged = _base_env_config(Args(), base_config={"dynamics": {"sensor_count": 11}})

    assert merged["dynamics"]["sensor_count"] == 11
