"""Curriculum reset sampling for segment-based PPO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import MONZA_LENGTH_METERS
from f1rl.state_library import load_state_library
from f1rl.state_snapshot import snapshot_to_dict


@dataclass(slots=True)
class CurriculumStage:
    name: str
    segment_length_m: float
    min_speed_kph: float
    max_speed_kph: float
    position_noise_m: float
    heading_noise_deg: float
    speed_noise_kph: float
    start_min_progress_m: float | None = None
    start_max_progress_m: float | None = None
    target_progress_m: float | None = None
    target_max_speed_kph: float | None = None


DEFAULT_SEGMENT_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("A0-short-control", 120.0, 10.0, 30.0, 0.0, 0.0, 1.0),
    CurriculumStage("A1-short-low-speed", 240.0, 20.0, 55.0, 0.2, 1.0, 2.0),
    CurriculumStage("B-medium-control", 500.0, 35.0, 85.0, 0.5, 2.0, 4.0),
    CurriculumStage("C-long", 1200.0, 70.0, 140.0, 1.0, 3.0, 7.0),
    CurriculumStage("D-random-checkpoint", 1800.0, 80.0, 200.0, 1.2, 4.0, 8.0),
    CurriculumStage("E-flying-lap", 5793.0, 240.0, 325.0, 0.8, 2.0, 8.0),
    CurriculumStage("F-normal-lap", 5793.0, 0.0, 20.0, 0.5, 2.0, 3.0),
)

PREFIX_START_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage("P0-launch-240m", 240.0, 0.0, 8.0, 0.0, 0.0, 1.0),
    CurriculumStage("P1-start-500m", 500.0, 0.0, 10.0, 0.0, 0.0, 1.5),
    CurriculumStage("P2-start-900m", 900.0, 0.0, 12.0, 0.0, 0.0, 2.0),
    CurriculumStage("P3-start-1400m", 1400.0, 0.0, 15.0, 0.0, 0.0, 2.0),
    CurriculumStage("P4-start-2200m", 2200.0, 0.0, 18.0, 0.0, 0.0, 2.5),
    CurriculumStage("P5-start-3500m", 3500.0, 0.0, 20.0, 0.0, 0.0, 3.0),
    CurriculumStage("P6-normal-lap", 5793.0, 0.0, 20.0, 0.0, 0.0, 3.0),
)

CHICANE_SKILL_STAGES: dict[str, tuple[CurriculumStage, ...]] = {
    "rettifilo": (
        CurriculumStage("rettifilo-approach-brake", 280.0, 120.0, 240.0, 0.4, 1.5, 5.0, 450.0, 560.0, 720.0, 190.0),
        CurriculumStage("rettifilo-turn-in", 280.0, 85.0, 170.0, 0.4, 2.0, 5.0, 650.0, 760.0, 930.0, 160.0),
        CurriculumStage("rettifilo-apex", 260.0, 70.0, 145.0, 0.3, 2.0, 4.0, 760.0, 900.0, 1080.0, 165.0),
        CurriculumStage("rettifilo-exit", 320.0, 95.0, 190.0, 0.4, 2.0, 5.0, 900.0, 1080.0, 1220.0, 220.0),
        CurriculumStage("rettifilo-post-exit", 360.0, 140.0, 260.0, 0.5, 2.0, 6.0, 1080.0, 1250.0, 1450.0, 285.0),
        CurriculumStage("rettifilo-full-chicane", 760.0, 160.0, 300.0, 0.4, 2.0, 8.0, 450.0, 560.0, 1220.0, 225.0),
    ),
    "roggia": (
        CurriculumStage("roggia-approach-brake", 300.0, 140.0, 280.0, 0.4, 1.5, 6.0, 1850.0, 2020.0, 2140.0, 205.0),
        CurriculumStage("roggia-turn-in", 300.0, 95.0, 180.0, 0.4, 2.0, 5.0, 2050.0, 2180.0, 2320.0, 170.0),
        CurriculumStage("roggia-apex", 260.0, 85.0, 160.0, 0.3, 2.0, 4.0, 2180.0, 2360.0, 2420.0, 175.0),
        CurriculumStage("roggia-exit", 340.0, 110.0, 210.0, 0.4, 2.0, 5.0, 2320.0, 2480.0, 2650.0, 235.0),
        CurriculumStage("roggia-post-exit", 420.0, 150.0, 270.0, 0.5, 2.0, 6.0, 2450.0, 2650.0, 2900.0, 290.0),
        CurriculumStage("roggia-full-chicane", 800.0, 170.0, 310.0, 0.4, 2.0, 8.0, 1850.0, 2020.0, 2650.0, 240.0),
    ),
}


def chicane_skill_stages(chicane: str) -> tuple[CurriculumStage, ...]:
    try:
        return CHICANE_SKILL_STAGES[chicane]
    except KeyError as exc:
        valid = ", ".join(sorted(CHICANE_SKILL_STAGES))
        raise ValueError(f"Unknown chicane {chicane!r}; expected one of: {valid}") from exc


@dataclass(slots=True)
class CurriculumConfig:
    mode: str = "none"
    stages: tuple[CurriculumStage, ...] = DEFAULT_SEGMENT_STAGES
    promotion_resets: int = 300
    normal_start_probability: float = 0.0
    focus_start_progress_m: float | None = None
    focus_window_m: float = 0.0
    start_mode: str = "random_checkpoint"
    state_library_path: Path | None = None

    @property
    def enabled(self) -> bool:
        return self.mode == "segments"


class CurriculumSampler:
    def __init__(self, config: CurriculumConfig | None = None, *, checkpoint_count: int = 120) -> None:
        self.config = config or CurriculumConfig()
        self.checkpoint_count = max(int(checkpoint_count), 1)
        self.reset_count = 0
        self.state_snapshots = (
            load_state_library(self.config.state_library_path) if self.config.state_library_path is not None else []
        )
        if self.config.start_mode == "state_library" and not self.state_snapshots:
            raise ValueError("state_library start mode requires a non-empty state library.")

    def sample_options(self, seed: int | None = None) -> dict[str, Any]:
        if not self.config.enabled:
            return {}
        rng = np.random.default_rng(seed)
        if self.config.start_mode != "normal" and rng.random() < max(
            0.0, min(float(self.config.normal_start_probability), 1.0)
        ):
            self.reset_count += 1
            return {"curriculum_stage": "normal-start-mix"}
        stage_index = min(self.reset_count // max(self.config.promotion_resets, 1), len(self.config.stages) - 1)
        stage = self.config.stages[stage_index]
        self.reset_count += 1
        focus_start_progress_m = self.config.focus_start_progress_m
        segment_length_m = stage.segment_length_m
        if focus_start_progress_m is not None:
            half_window_m = max(float(self.config.focus_window_m), 0.0) * 0.5
            start_progress_m = float(focus_start_progress_m)
            if half_window_m > 0.0:
                start_progress_m += float(rng.uniform(-half_window_m, half_window_m))
            start_location: dict[str, Any] = {"start_progress_m": max(start_progress_m, 0.0)}
        elif self.config.start_mode == "state_library":
            eligible_snapshots = self._snapshots_for_stage(stage)
            if not eligible_snapshots:
                raise ValueError(f"No state-library snapshots match curriculum stage {stage.name!r}.")
            snapshot = eligible_snapshots[int(rng.integers(0, len(eligible_snapshots)))]
            start_location = {"state_snapshot": snapshot_to_dict(snapshot)}
            if stage.target_progress_m is not None:
                segment_length_m = max(float(stage.target_progress_m) - snapshot.monotonic_progress_m, 1.0)
        elif self.config.start_mode == "normal":
            start_location = {}
        elif stage.start_min_progress_m is not None and stage.start_max_progress_m is not None:
            start_progress_m = float(rng.uniform(stage.start_min_progress_m, stage.start_max_progress_m))
            start_location = {"start_progress_m": start_progress_m}
            if stage.target_progress_m is not None:
                segment_length_m = max(float(stage.target_progress_m) - start_progress_m, 1.0)
        else:
            checkpoint = int(rng.integers(0, self.checkpoint_count))
            start_location = {"start_checkpoint": checkpoint}
        speed_kph = float(rng.uniform(stage.min_speed_kph, stage.max_speed_kph))
        if self.config.start_mode == "state_library":
            speed_kph = None
        return {
            **start_location,
            **({} if speed_kph is None else {"start_speed_kph": speed_kph}),
            "segment_length_m": segment_length_m,
            **(
                {}
                if stage.target_max_speed_kph is None
                else {"segment_target_max_speed_kph": float(stage.target_max_speed_kph)}
            ),
            "position_noise_m": stage.position_noise_m,
            "heading_noise_deg": stage.heading_noise_deg,
            "speed_noise_kph": stage.speed_noise_kph,
            "curriculum_stage": stage.name,
        }

    def _snapshots_for_stage(self, stage: CurriculumStage):
        if stage.start_min_progress_m is None or stage.start_max_progress_m is None:
            return self.state_snapshots
        start_m = float(stage.start_min_progress_m)
        end_m = float(stage.start_max_progress_m)
        return [
            snapshot
            for snapshot in self.state_snapshots
            if start_m <= snapshot.monotonic_progress_m % MONZA_LENGTH_METERS <= end_m
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "promotion_resets": self.config.promotion_resets,
            "normal_start_probability": self.config.normal_start_probability,
            "focus_start_progress_m": self.config.focus_start_progress_m,
            "focus_window_m": self.config.focus_window_m,
            "start_mode": self.config.start_mode,
            "state_library_path": str(self.config.state_library_path) if self.config.state_library_path is not None else None,
            "state_library_count": len(self.state_snapshots),
            "stages": [asdict(stage) for stage in self.config.stages],
        }
