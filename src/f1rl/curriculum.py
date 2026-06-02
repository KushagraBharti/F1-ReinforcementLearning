"""Curriculum reset sampling for segment-based PPO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CurriculumStage:
    name: str
    segment_length_m: float
    min_speed_kph: float
    max_speed_kph: float
    position_noise_m: float
    heading_noise_deg: float
    speed_noise_kph: float


DEFAULT_SEGMENT_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("A0-short-control", 120.0, 10.0, 30.0, 0.0, 0.0, 1.0),
    CurriculumStage("A1-short-low-speed", 240.0, 20.0, 55.0, 0.2, 1.0, 2.0),
    CurriculumStage("B-medium-control", 500.0, 35.0, 85.0, 0.5, 2.0, 4.0),
    CurriculumStage("C-long", 1200.0, 70.0, 140.0, 1.0, 3.0, 7.0),
    CurriculumStage("D-random-checkpoint", 1800.0, 80.0, 200.0, 1.2, 4.0, 8.0),
    CurriculumStage("E-flying-lap", 5793.0, 240.0, 325.0, 0.8, 2.0, 8.0),
    CurriculumStage("F-normal-lap", 5793.0, 0.0, 20.0, 0.5, 2.0, 3.0),
)


@dataclass(slots=True)
class CurriculumConfig:
    mode: str = "none"
    stages: tuple[CurriculumStage, ...] = DEFAULT_SEGMENT_STAGES
    promotion_resets: int = 300

    @property
    def enabled(self) -> bool:
        return self.mode == "segments"


class CurriculumSampler:
    def __init__(self, config: CurriculumConfig | None = None, *, checkpoint_count: int = 120) -> None:
        self.config = config or CurriculumConfig()
        self.checkpoint_count = max(int(checkpoint_count), 1)
        self.reset_count = 0

    def sample_options(self, seed: int | None = None) -> dict[str, Any]:
        if not self.config.enabled:
            return {}
        rng = np.random.default_rng(seed)
        stage_index = min(self.reset_count // max(self.config.promotion_resets, 1), len(self.config.stages) - 1)
        stage = self.config.stages[stage_index]
        self.reset_count += 1
        checkpoint = int(rng.integers(0, self.checkpoint_count))
        speed_kph = float(rng.uniform(stage.min_speed_kph, stage.max_speed_kph))
        return {
            "start_checkpoint": checkpoint,
            "start_speed_kph": speed_kph,
            "segment_length_m": stage.segment_length_m,
            "position_noise_m": stage.position_noise_m,
            "heading_noise_deg": stage.heading_noise_deg,
            "speed_noise_kph": stage.speed_noise_kph,
            "curriculum_stage": stage.name,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "promotion_resets": self.config.promotion_resets,
            "stages": [asdict(stage) for stage in self.config.stages],
        }
