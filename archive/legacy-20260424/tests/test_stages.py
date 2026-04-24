from f1rl.stages import (
    compare_aggregate_for_stage,
    resolve_seed_list,
    stage_gate_met,
)


def test_resolve_seed_list_rotates_diagnostic_pool() -> None:
    first = resolve_seed_list("quick", "diagnostic", rotation_index=0)
    second = resolve_seed_list("quick", "diagnostic", rotation_index=1)

    assert len(first) == len(second) == len(resolve_seed_list("quick", "promotion"))
    assert first != second


def test_competence_gate_requires_checkpoint_and_survival_metrics() -> None:
    aggregate = {
        "first_checkpoint_rate": 1.0,
        "three_checkpoint_rate": 0.95,
        "avg_checkpoints_reached": 12.0,
        "clean_100_step_survival_rate": 0.8,
    }
    assert stage_gate_met("competence", aggregate) is True


def test_compare_aggregate_for_stage_prefers_completion_for_lap_stage() -> None:
    candidate = {
        "completion_rate": 0.2,
        "best_lap_time_steps": 400,
        "avg_lap_time_steps": 450,
        "max_checkpoints_reached": 40,
        "collision_rate": 0.6,
    }
    champion = {
        "completion_rate": 0.0,
        "best_lap_time_steps": None,
        "avg_lap_time_steps": None,
        "max_checkpoints_reached": 35,
        "collision_rate": 0.7,
    }
    is_better, reason = compare_aggregate_for_stage("lap", candidate, champion)
    assert is_better is True
    assert reason == "completion rate"
