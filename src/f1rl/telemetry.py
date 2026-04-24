"""Telemetry schema and JSONL persistence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REWARD_COMPONENT_KEYS = ("progress", "finish", "collision", "off_track", "no_progress", "smoothness")


@dataclass(slots=True)
class StepTelemetry:
    step_index: int
    sim_time_s: float
    x: float
    y: float
    heading_deg: float
    speed_mps: float
    speed_kph: float
    yaw_rate_rps: float
    throttle: float
    brake: float
    steering: float
    action_id: int
    raw_progress_m: float
    monotonic_progress_m: float
    progress_delta_m: float
    lateral_error_m: float
    heading_error_deg: float
    checkpoint_index: int
    lap_index: int
    ray_distances_m: list[float]
    collided: bool
    off_track: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    reward_total: float
    reward_components: dict[str, float]


@dataclass(slots=True)
class EpisodeSummary:
    run_id: str
    mode: str
    seed: int
    termination_reason: str
    completed_lap: bool
    elapsed_time_s: float
    lap_time_s: float | None
    checkpoints_reached: int
    distance_traveled_m: float
    avg_speed_kph: float
    max_speed_kph: float
    collision_count: int
    off_track_count: int
    reward_totals: dict[str, float]


class TelemetryWriter:
    def __init__(self, root: Path, *, mode: str, seed: int) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{mode}-{timestamp}"
        self.root = root / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.root / "steps.jsonl"
        self.summary_path = self.root / "episode_summary.json"
        self.mode = mode
        self.seed = seed
        self._steps: list[StepTelemetry] = []

    def write_step(self, step: StepTelemetry) -> None:
        self._steps.append(step)
        with self.steps_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(step)) + "\n")

    def close_episode(self, *, termination_reason: str, completed_lap: bool) -> EpisodeSummary:
        elapsed = self._steps[-1].sim_time_s if self._steps else 0.0
        speeds = [step.speed_kph for step in self._steps]
        reward_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        for step in self._steps:
            for key in REWARD_COMPONENT_KEYS:
                reward_totals[key] += float(step.reward_components.get(key, 0.0))
        summary = EpisodeSummary(
            run_id=self.run_id,
            mode=self.mode,
            seed=self.seed,
            termination_reason=termination_reason,
            completed_lap=completed_lap,
            elapsed_time_s=elapsed,
            lap_time_s=elapsed if completed_lap else None,
            checkpoints_reached=max((step.checkpoint_index for step in self._steps), default=0),
            distance_traveled_m=self._steps[-1].monotonic_progress_m if self._steps else 0.0,
            avg_speed_kph=float(sum(speeds) / len(speeds)) if speeds else 0.0,
            max_speed_kph=float(max(speeds)) if speeds else 0.0,
            collision_count=sum(1 for step in self._steps if step.collided),
            off_track_count=sum(1 for step in self._steps if step.off_track),
            reward_totals=reward_totals,
        )
        self.summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary


def load_steps(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
