"""Telemetry schema and JSONL persistence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

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
    acceleration_mps2: float
    longitudinal_g: float
    lateral_g: float
    curvature_rad_per_m: float
    throttle: float
    brake: float
    steering: float
    throttle_delta: float
    brake_delta: float
    steering_delta: float
    action_id: int
    raw_progress_m: float
    monotonic_progress_m: float
    progress_delta_m: float
    lateral_error_m: float
    racing_line_deviation_m: float
    heading_error_deg: float
    reference_progress_m: float | None
    reference_speed_kph: float | None
    ghost_gap_m: float | None
    checkpoint_index: int
    next_checkpoint_index: int
    checkpoints_passed: int
    missed_checkpoint_count: int
    lap_index: int
    valid_lap: bool
    finish_crossed: bool
    segment_complete: bool
    curriculum_stage: str | None
    segment_target_progress_m: float | None
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
    valid_lap: bool
    finish_crossed: bool
    segment_complete: bool
    curriculum_stage: str | None
    segment_target_progress_m: float | None
    elapsed_time_s: float
    lap_time_s: float | None
    checkpoints_reached: int
    checkpoints_passed: int
    missed_checkpoint_count: int
    distance_traveled_m: float
    avg_speed_kph: float
    max_speed_kph: float
    collision_count: int
    off_track_count: int
    reward_totals: dict[str, float]
    sector_times_s: list[float | None]
    sector_speed_kph: list[float]
    braking_zone_count: int
    braking_zones: list[dict[str, float]]
    avg_abs_racing_line_deviation_m: float
    max_abs_racing_line_deviation_m: float
    avg_lateral_g: float
    max_lateral_g: float
    avg_longitudinal_g: float
    min_longitudinal_g: float
    steering_smoothness_avg: float
    steering_smoothness_max: float
    throttle_smoothness_avg: float
    brake_smoothness_avg: float
    avg_ghost_gap_m: float | None
    final_ghost_gap_m: float | None
    corner_count: int
    corner_summaries: list[dict[str, float]]


def _sector_times(
    steps: list[StepTelemetry],
    *,
    lap_length_m: float | None,
    sector_count: int = 3,
) -> tuple[list[float | None], list[float]]:
    if not steps:
        return [None] * sector_count, [0.0] * sector_count
    total_distance = max(step.monotonic_progress_m for step in steps)
    if total_distance <= 1e-6:
        return [None] * sector_count, [0.0] * sector_count
    lap_length = lap_length_m or max(steps[-1].monotonic_progress_m, total_distance)
    thresholds = [lap_length * (idx + 1) / sector_count for idx in range(sector_count)]
    times: list[float | None] = []
    speeds: list[float] = []
    previous_time = 0.0
    previous_distance = 0.0
    for threshold in thresholds:
        crossing = next((step for step in steps if step.monotonic_progress_m >= threshold), None)
        if crossing is None:
            times.append(None)
            speeds.append(0.0)
            continue
        sector_time = max(crossing.sim_time_s - previous_time, 0.0)
        sector_distance = max(threshold - previous_distance, 0.0)
        times.append(sector_time)
        speeds.append(float((sector_distance / max(sector_time, 1e-6)) * 3.6))
        previous_time = crossing.sim_time_s
        previous_distance = threshold
    return times, speeds


def _contiguous_zones(
    steps: list[StepTelemetry],
    *,
    active,
    min_steps: int,
) -> list[list[StepTelemetry]]:
    zones: list[list[StepTelemetry]] = []
    current: list[StepTelemetry] = []
    for step in steps:
        if active(step):
            current.append(step)
        elif current:
            if len(current) >= min_steps:
                zones.append(current)
            current = []
    if len(current) >= min_steps:
        zones.append(current)
    return zones


def _braking_zones(steps: list[StepTelemetry]) -> list[dict[str, float]]:
    zones = _contiguous_zones(steps, active=lambda step: step.brake > 0.05 or step.longitudinal_g < -0.15, min_steps=3)
    summaries: list[dict[str, float]] = []
    for zone in zones:
        summaries.append(
            {
                "start_time_s": zone[0].sim_time_s,
                "end_time_s": zone[-1].sim_time_s,
                "start_progress_m": zone[0].monotonic_progress_m,
                "end_progress_m": zone[-1].monotonic_progress_m,
                "entry_speed_kph": zone[0].speed_kph,
                "min_speed_kph": min(step.speed_kph for step in zone),
                "max_brake": max(step.brake for step in zone),
                "peak_decel_g": abs(min(step.longitudinal_g for step in zone)),
            }
        )
    return summaries


def _corner_summaries(steps: list[StepTelemetry]) -> list[dict[str, float]]:
    zones = _contiguous_zones(
        steps,
        active=lambda step: abs(step.curvature_rad_per_m) > 0.006 or abs(step.lateral_g) > 0.55,
        min_steps=4,
    )
    summaries: list[dict[str, float]] = []
    for zone in zones:
        summaries.append(
            {
                "start_time_s": zone[0].sim_time_s,
                "end_time_s": zone[-1].sim_time_s,
                "start_progress_m": zone[0].monotonic_progress_m,
                "end_progress_m": zone[-1].monotonic_progress_m,
                "entry_speed_kph": zone[0].speed_kph,
                "apex_speed_kph": min(step.speed_kph for step in zone),
                "exit_speed_kph": zone[-1].speed_kph,
                "peak_lateral_g": max(abs(step.lateral_g) for step in zone),
                "max_deviation_m": max(abs(step.racing_line_deviation_m) for step in zone),
            }
        )
    return summaries


class TelemetryWriter:
    def __init__(self, root: Path, *, mode: str, seed: int, lap_length_m: float | None = None) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{mode}-{timestamp}"
        self.root = root / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.root / "steps.jsonl"
        self.summary_path = self.root / "episode_summary.json"
        self.mode = mode
        self.seed = seed
        self.lap_length_m = lap_length_m
        self._steps: list[StepTelemetry] = []
        self._steps_file: TextIO = self.steps_path.open("a", encoding="utf-8")

    def write_step(self, step: StepTelemetry) -> None:
        self._steps.append(step)
        self._steps_file.write(json.dumps(asdict(step)) + "\n")

    def close_episode(self, *, termination_reason: str, completed_lap: bool) -> EpisodeSummary:
        if not self._steps_file.closed:
            self._steps_file.flush()
            self._steps_file.close()
        elapsed = self._steps[-1].sim_time_s if self._steps else 0.0
        speeds = [step.speed_kph for step in self._steps]
        abs_deviations = [abs(step.racing_line_deviation_m) for step in self._steps]
        lateral_g_values = [abs(step.lateral_g) for step in self._steps]
        longitudinal_g_values = [step.longitudinal_g for step in self._steps]
        steering_deltas = [abs(step.steering_delta) for step in self._steps]
        throttle_deltas = [abs(step.throttle_delta) for step in self._steps]
        brake_deltas = [abs(step.brake_delta) for step in self._steps]
        ghost_gaps = [step.ghost_gap_m for step in self._steps if step.ghost_gap_m is not None]
        sector_times, sector_speeds = _sector_times(self._steps, lap_length_m=self.lap_length_m)
        braking_zones = _braking_zones(self._steps)
        corner_summaries = _corner_summaries(self._steps)
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
            valid_lap=bool(self._steps[-1].valid_lap) if self._steps else False,
            finish_crossed=any(step.finish_crossed for step in self._steps),
            segment_complete=any(step.segment_complete for step in self._steps),
            curriculum_stage=next((step.curriculum_stage for step in reversed(self._steps) if step.curriculum_stage), None),
            segment_target_progress_m=next(
                (
                    step.segment_target_progress_m
                    for step in reversed(self._steps)
                    if step.segment_target_progress_m is not None
                ),
                None,
            ),
            elapsed_time_s=elapsed,
            lap_time_s=elapsed if completed_lap else None,
            checkpoints_reached=max((step.checkpoint_index for step in self._steps), default=0),
            checkpoints_passed=max((step.checkpoints_passed for step in self._steps), default=0),
            missed_checkpoint_count=max((step.missed_checkpoint_count for step in self._steps), default=0),
            distance_traveled_m=self._steps[-1].monotonic_progress_m if self._steps else 0.0,
            avg_speed_kph=float(sum(speeds) / len(speeds)) if speeds else 0.0,
            max_speed_kph=float(max(speeds)) if speeds else 0.0,
            collision_count=sum(1 for step in self._steps if step.collided),
            off_track_count=sum(1 for step in self._steps if step.off_track),
            reward_totals=reward_totals,
            sector_times_s=sector_times,
            sector_speed_kph=sector_speeds,
            braking_zone_count=len(braking_zones),
            braking_zones=braking_zones,
            avg_abs_racing_line_deviation_m=float(sum(abs_deviations) / len(abs_deviations)) if abs_deviations else 0.0,
            max_abs_racing_line_deviation_m=float(max(abs_deviations)) if abs_deviations else 0.0,
            avg_lateral_g=float(sum(lateral_g_values) / len(lateral_g_values)) if lateral_g_values else 0.0,
            max_lateral_g=float(max(lateral_g_values)) if lateral_g_values else 0.0,
            avg_longitudinal_g=float(sum(longitudinal_g_values) / len(longitudinal_g_values)) if longitudinal_g_values else 0.0,
            min_longitudinal_g=float(min(longitudinal_g_values)) if longitudinal_g_values else 0.0,
            steering_smoothness_avg=float(sum(steering_deltas) / len(steering_deltas)) if steering_deltas else 0.0,
            steering_smoothness_max=float(max(steering_deltas)) if steering_deltas else 0.0,
            throttle_smoothness_avg=float(sum(throttle_deltas) / len(throttle_deltas)) if throttle_deltas else 0.0,
            brake_smoothness_avg=float(sum(brake_deltas) / len(brake_deltas)) if brake_deltas else 0.0,
            avg_ghost_gap_m=float(sum(ghost_gaps) / len(ghost_gaps)) if ghost_gaps else None,
            final_ghost_gap_m=float(ghost_gaps[-1]) if ghost_gaps else None,
            corner_count=len(corner_summaries),
            corner_summaries=corner_summaries,
        )
        self.summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary


def load_steps(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
