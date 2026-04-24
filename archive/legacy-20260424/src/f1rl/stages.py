"""Stage, seed, and promotion policy helpers for the autonomous campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class StageConfig:
    name: str
    logical_cars: int
    max_steps: int
    no_progress_limit_steps: int
    reverse_speed_limit_steps: int
    reverse_progress_limit: int
    heading_away_limit_steps: int
    wall_hugging_limit_steps: int
    checkpoint_deadlines: tuple[tuple[int, int], ...]
    benchmark_profile: str
    training_mode: str


STAGE_ORDER = ("competence", "lap", "stability", "performance")

STAGES: dict[str, StageConfig] = {
    "competence": StageConfig(
        name="competence",
        logical_cars=32,
        max_steps=420,
        no_progress_limit_steps=140,
        reverse_speed_limit_steps=48,
        reverse_progress_limit=4,
        heading_away_limit_steps=0,
        wall_hugging_limit_steps=0,
        checkpoint_deadlines=(),
        benchmark_profile="quick",
        training_mode="smoke",
    ),
    "lap": StageConfig(
        name="lap",
        logical_cars=64,
        max_steps=540,
        no_progress_limit_steps=90,
        reverse_speed_limit_steps=36,
        reverse_progress_limit=3,
        heading_away_limit_steps=70,
        wall_hugging_limit_steps=45,
        checkpoint_deadlines=((1, 80), (3, 160), (8, 320), (15, 500)),
        benchmark_profile="quick",
        training_mode="benchmark",
    ),
    "stability": StageConfig(
        name="stability",
        logical_cars=128,
        max_steps=900,
        no_progress_limit_steps=120,
        reverse_speed_limit_steps=42,
        reverse_progress_limit=4,
        heading_away_limit_steps=90,
        wall_hugging_limit_steps=60,
        checkpoint_deadlines=((1, 90), (3, 170), (8, 330), (18, 620)),
        benchmark_profile="standard",
        training_mode="benchmark",
    ),
    "performance": StageConfig(
        name="performance",
        logical_cars=128,
        max_steps=1200,
        no_progress_limit_steps=150,
        reverse_speed_limit_steps=50,
        reverse_progress_limit=4,
        heading_away_limit_steps=110,
        wall_hugging_limit_steps=70,
        checkpoint_deadlines=((1, 90), (3, 170), (8, 330), (18, 620)),
        benchmark_profile="standard",
        training_mode="performance",
    ),
}


SEED_TIERS: dict[str, dict[str, tuple[int, ...]]] = {
    "quick": {
        "promotion": (101, 102, 103),
        "holdout": (901, 902, 903),
        "diagnostic_pool": tuple(range(201, 241)),
    },
    "standard": {
        "promotion": (111, 112, 113, 114, 115),
        "holdout": (911, 912, 913, 914, 915),
        "diagnostic_pool": tuple(range(301, 361)),
    },
}


def get_stage_config(stage: str) -> StageConfig:
    if stage == "auto":
        return STAGES["competence"]
    try:
        return STAGES[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported swarm stage: {stage}") from exc


def build_stage_env_overrides(stage: str) -> dict[str, Any]:
    cfg = get_stage_config(stage)
    return {
        "max_steps": cfg.max_steps,
        "no_progress_limit_steps": cfg.no_progress_limit_steps,
        "reverse_speed_limit_steps": cfg.reverse_speed_limit_steps,
        "reverse_progress_limit": cfg.reverse_progress_limit,
        "heading_away_limit_steps": cfg.heading_away_limit_steps,
        "wall_hugging_limit_steps": cfg.wall_hugging_limit_steps,
        "checkpoint_deadlines": [list(item) for item in cfg.checkpoint_deadlines],
    }


def logical_car_target(stage: str) -> int:
    return get_stage_config(stage).logical_cars


def resolve_seed_list(profile: str, tier: str, rotation_index: int = 0) -> list[int]:
    try:
        seed_config = SEED_TIERS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark profile for seed tier: {profile}") from exc

    if tier == "diagnostic":
        promotion_count = len(seed_config["promotion"])
        pool = seed_config["diagnostic_pool"]
        start = (rotation_index * promotion_count) % len(pool)
        return [pool[(start + idx) % len(pool)] for idx in range(promotion_count)]
    try:
        return list(seed_config[tier])
    except KeyError as exc:
        raise ValueError(f"Unsupported seed tier: {tier}") from exc


def infer_stage_from_aggregate(aggregate: dict[str, Any]) -> str:
    completion_rate = float(aggregate.get("completion_rate", 0.0) or 0.0)
    avg_lap_time = aggregate.get("avg_lap_time_steps")
    if completion_rate >= 0.8 and avg_lap_time is not None:
        return "performance"
    if completion_rate >= 0.5:
        return "stability"
    if completion_rate > 0.0 or avg_lap_time is not None:
        return "lap"
    return "competence"


def stage_gate_met(stage: str, aggregate: dict[str, Any]) -> bool:
    completion_rate = float(aggregate.get("completion_rate", 0.0) or 0.0)
    if stage == "competence":
        return (
            float(aggregate.get("first_checkpoint_rate", 0.0) or 0.0) >= 1.0
            and float(aggregate.get("three_checkpoint_rate", 0.0) or 0.0) >= 0.95
            and float(aggregate.get("avg_checkpoints_reached", 0.0) or 0.0) >= 12.0
            and float(aggregate.get("clean_100_step_survival_rate", 0.0) or 0.0) >= 0.8
        )
    if stage == "lap":
        return completion_rate > 0.0
    if stage == "stability":
        return completion_rate >= 0.8
    if stage == "performance":
        return False
    raise ValueError(f"Unsupported stage: {stage}")


def next_stage(stage: str) -> str:
    idx = STAGE_ORDER.index(stage)
    if idx >= len(STAGE_ORDER) - 1:
        return stage
    return STAGE_ORDER[idx + 1]


def compare_aggregate_for_stage(stage: str, candidate: dict[str, Any], champion: dict[str, Any]) -> tuple[bool, str]:
    if stage == "competence":
        ordered_metrics: tuple[tuple[str, bool, str], ...] = (
            ("first_checkpoint_rate", True, "first checkpoint rate"),
            ("three_checkpoint_rate", True, "three checkpoint rate"),
            ("avg_checkpoints_reached", True, "average checkpoints reached"),
            ("max_checkpoints_reached", True, "max checkpoints reached"),
            ("clean_100_step_survival_rate", True, "clean 100-step survival"),
            ("collision_rate", False, "collision rate"),
            ("avg_progress_ratio", True, "progress ratio"),
        )
    elif stage == "lap":
        ordered_metrics = (
            ("completion_rate", True, "completion rate"),
            ("best_lap_time_steps", False, "best lap time"),
            ("avg_lap_time_steps", False, "average lap time"),
            ("max_checkpoints_reached", True, "max checkpoints reached"),
            ("collision_rate", False, "collision rate"),
        )
    elif stage == "stability":
        ordered_metrics = (
            ("completion_rate", True, "completion rate"),
            ("collision_rate", False, "collision rate"),
            ("lap_time_std_steps", False, "lap-time variance"),
            ("avg_lap_time_steps", False, "average lap time"),
        )
    elif stage == "performance":
        if float(candidate.get("completion_rate", 0.0) or 0.0) < 0.8:
            return False, "failed performance completion floor"
        ordered_metrics = (
            ("best_lap_time_steps", False, "best lap time"),
            ("avg_lap_time_steps", False, "average lap time"),
            ("collision_rate", False, "collision rate"),
        )
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    for key, higher_is_better, label in ordered_metrics:
        cand_value = candidate.get(key)
        champ_value = champion.get(key)
        if cand_value is None and champ_value is None:
            continue
        if cand_value is None:
            return False, f"missing {label}"
        if champ_value is None:
            return True, f"candidate has {label}"
        cand_float = float(cand_value)
        champ_float = float(champ_value)
        if cand_float == champ_float:
            continue
        if higher_is_better:
            return cand_float > champ_float, label
        return cand_float < champ_float, label

    return float(candidate.get("avg_reward", 0.0) or 0.0) > float(champion.get("avg_reward", 0.0) or 0.0), "reward tiebreak"
