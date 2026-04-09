from f1rl.benchmark import compare_benchmark_summaries


def _summary(
    *,
    completion_rate: float,
    collision_rate: float,
    avg_lap_time_steps: float | None,
    avg_progress_ratio: float,
    avg_reward: float = 0.0,
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
