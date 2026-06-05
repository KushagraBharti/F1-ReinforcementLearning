# pyright: reportPrivateImportUsage=false
"""GPU scoring accumulators for batched evolution rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from f1rl.config import MONZA_LENGTH_METERS, SimConfig
from f1rl.gpu_types import GpuCarBatch

TERMINATION_REASON_TO_ID = {
    "active": 0,
    "segment_complete": 1,
    "segment_release_gate_failed": 2,
    "segment_min_speed_gate_failed": 3,
    "segment_speed_gate_failed": 4,
    "segment_lateral_gate_failed": 5,
    "segment_heading_gate_failed": 6,
    "segment_yaw_rate_gate_failed": 7,
    "segment_steering_gate_failed": 8,
    "lap_complete": 9,
    "collision": 10,
    "off_track": 11,
    "no_progress": 12,
    "max_steps": 13,
}
TERMINATION_ID_TO_REASON = {value: key for key, value in TERMINATION_REASON_TO_ID.items()}


@dataclass(slots=True)
class GpuScoreAccumulator:
    start_progress_m: torch.Tensor
    start_speed_kph: torch.Tensor
    start_speed_for_score_kph: torch.Tensor
    best_progress_m: torch.Tensor
    best_speed_kph: torch.Tensor
    best_lateral_error_m: torch.Tensor
    best_heading_error_deg: torch.Tensor
    final_x: torch.Tensor
    final_y: torch.Tensor
    final_heading_deg: torch.Tensor
    final_speed_mps: torch.Tensor
    final_speed_kph: torch.Tensor
    final_raw_progress_m: torch.Tensor
    final_progress_m: torch.Tensor
    final_lateral_error_m: torch.Tensor
    final_heading_error_deg: torch.Tensor
    final_yaw_rate_rps: torch.Tensor
    final_curvature_rad_per_m: torch.Tensor
    final_steering: torch.Tensor
    final_throttle: torch.Tensor
    final_brake: torch.Tensor
    final_step_index: torch.Tensor
    final_sim_time_s: torch.Tensor
    final_missed_checkpoint_count: torch.Tensor
    final_valid_lap: torch.Tensor
    final_finish_crossed: torch.Tensor
    final_completed_lap: torch.Tensor
    final_segment_complete: torch.Tensor
    final_collided: torch.Tensor
    final_off_track: torch.Tensor
    final_terminated: torch.Tensor
    final_truncated: torch.Tensor
    final_termination_reason_id: torch.Tensor
    final_target_speed_kph: torch.Tensor
    final_near_target_speed_kph: torch.Tensor
    final_min_future_target_speed_kph: torch.Tensor
    final_target_speed_drop_kph: torch.Tensor
    final_target_speed_drop_norm: torch.Tensor
    final_brake_demand: torch.Tensor
    final_future_brake_demand: torch.Tensor
    final_brake_gate_proximity: torch.Tensor
    final_braking_gate_distance_m: torch.Tensor
    final_brake_gate_distance_norm: torch.Tensor
    final_lookahead_abs_max: torch.Tensor
    time_to_300_m: torch.Tensor
    time_to_450_m: torch.Tensor
    speed_sum_first_300_m: torch.Tensor
    speed_count_first_300_m: torch.Tensor
    speed_sum_first_450_m: torch.Tensor
    speed_count_first_450_m: torch.Tensor
    brake_sum_first_300_m: torch.Tensor
    brake_count_first_300_m: torch.Tensor
    brake_sum_first_450_m: torch.Tensor
    brake_count_first_450_m: torch.Tensor
    max_brake: torch.Tensor
    throttle_sum: torch.Tensor
    row_count: torch.Tensor
    demand_brake_sum: torch.Tensor
    demand_throttle_sum: torch.Tensor
    demand_count: torch.Tensor
    demand_max_future_brake_demand: torch.Tensor


@dataclass(frozen=True, slots=True)
class GpuScoringDiagnostics:
    lateral_error_m: torch.Tensor
    heading_error_deg: torch.Tensor
    future_brake_demand: torch.Tensor
    brake_gate_proximity: torch.Tensor
    target_speed_drop_norm: torch.Tensor
    telemetry_valid_lap: torch.Tensor
    target_speed_kph: torch.Tensor
    near_target_speed_kph: torch.Tensor
    min_future_target_speed_kph: torch.Tensor
    target_speed_drop_kph: torch.Tensor
    brake_demand: torch.Tensor
    braking_gate_distance_m: torch.Tensor
    brake_gate_distance_norm: torch.Tensor
    lookahead_abs_max: torch.Tensor

    @classmethod
    def from_mapping(
        cls,
        diagnostics: Mapping[str, torch.Tensor],
        *,
        fallback_valid_lap: torch.Tensor,
    ) -> GpuScoringDiagnostics:
        return cls(
            lateral_error_m=diagnostics["lateral_error_m"],
            heading_error_deg=diagnostics["heading_error_deg"],
            future_brake_demand=diagnostics["future_brake_demand"],
            brake_gate_proximity=diagnostics["brake_gate_proximity"],
            target_speed_drop_norm=diagnostics["target_speed_drop_norm"],
            telemetry_valid_lap=diagnostics.get("telemetry_valid_lap", fallback_valid_lap),
            target_speed_kph=diagnostics["target_speed_kph"],
            near_target_speed_kph=diagnostics["near_target_speed_kph"],
            min_future_target_speed_kph=diagnostics["min_future_target_speed_kph"],
            target_speed_drop_kph=diagnostics["target_speed_drop_kph"],
            brake_demand=diagnostics["brake_demand"],
            braking_gate_distance_m=diagnostics["braking_gate_distance_m"],
            brake_gate_distance_norm=diagnostics["brake_gate_distance_norm"],
            lookahead_abs_max=diagnostics["lookahead_abs_max"],
        )


def create_score_accumulator(state: GpuCarBatch) -> GpuScoreAccumulator:
    speed_kph = state.speed_mps * 3.6
    zeros = torch.zeros_like(state.speed_mps)
    neg_ones = torch.full_like(state.speed_mps, -1.0)
    return GpuScoreAccumulator(
        start_progress_m=state.monotonic_progress_m.clone(),
        start_speed_kph=speed_kph.clone(),
        start_speed_for_score_kph=zeros.clone(),
        best_progress_m=state.monotonic_progress_m.clone(),
        best_speed_kph=speed_kph.clone(),
        best_lateral_error_m=zeros.clone(),
        best_heading_error_deg=zeros.clone(),
        final_x=state.x.clone(),
        final_y=state.y.clone(),
        final_heading_deg=state.heading_rad * (180.0 / torch.pi),
        final_speed_mps=state.speed_mps.clone(),
        final_speed_kph=speed_kph.clone(),
        final_raw_progress_m=state.raw_progress_m.clone(),
        final_progress_m=state.monotonic_progress_m.clone(),
        final_lateral_error_m=zeros.clone(),
        final_heading_error_deg=zeros.clone(),
        final_yaw_rate_rps=state.yaw_rate_rps.clone(),
        final_curvature_rad_per_m=zeros.clone(),
        final_steering=state.last_steer.clone(),
        final_throttle=state.last_throttle.clone(),
        final_brake=state.last_brake.clone(),
        final_step_index=state.elapsed_steps.clone(),
        final_sim_time_s=zeros.clone(),
        final_missed_checkpoint_count=state.missed_checkpoint_count.clone(),
        final_valid_lap=state.valid_lap.clone(),
        final_finish_crossed=state.finish_crossed.clone(),
        final_completed_lap=state.completed_lap.clone(),
        final_segment_complete=state.segment_complete.clone(),
        final_collided=torch.zeros_like(state.alive),
        final_off_track=torch.zeros_like(state.alive),
        final_terminated=state.terminated.clone(),
        final_truncated=state.truncated.clone(),
        final_termination_reason_id=state.termination_reason_id.clone(),
        final_target_speed_kph=zeros.clone(),
        final_near_target_speed_kph=zeros.clone(),
        final_min_future_target_speed_kph=zeros.clone(),
        final_target_speed_drop_kph=zeros.clone(),
        final_target_speed_drop_norm=zeros.clone(),
        final_brake_demand=zeros.clone(),
        final_future_brake_demand=zeros.clone(),
        final_brake_gate_proximity=zeros.clone(),
        final_braking_gate_distance_m=zeros.clone(),
        final_brake_gate_distance_norm=zeros.clone(),
        final_lookahead_abs_max=zeros.clone(),
        time_to_300_m=neg_ones.clone(),
        time_to_450_m=neg_ones.clone(),
        speed_sum_first_300_m=zeros.clone(),
        speed_count_first_300_m=zeros.clone(),
        speed_sum_first_450_m=zeros.clone(),
        speed_count_first_450_m=zeros.clone(),
        brake_sum_first_300_m=zeros.clone(),
        brake_count_first_300_m=zeros.clone(),
        brake_sum_first_450_m=zeros.clone(),
        brake_count_first_450_m=zeros.clone(),
        max_brake=zeros.clone(),
        throttle_sum=zeros.clone(),
        row_count=zeros.clone(),
        demand_brake_sum=zeros.clone(),
        demand_throttle_sum=zeros.clone(),
        demand_count=zeros.clone(),
        demand_max_future_brake_demand=zeros.clone(),
    )


def _where(mask: torch.Tensor, new_value: torch.Tensor, old_value: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, new_value, old_value)


def update_score_accumulator(
    acc: GpuScoreAccumulator,
    *,
    state: GpuCarBatch,
    diagnostics: dict[str, torch.Tensor],
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    active: torch.Tensor,
    collided: torch.Tensor,
    off_track: torch.Tensor,
    config: SimConfig,
) -> None:
    update_score_accumulator_static(
        acc,
        state=state,
        diagnostics=GpuScoringDiagnostics.from_mapping(diagnostics, fallback_valid_lap=state.valid_lap),
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=active,
        collided=collided,
        off_track=off_track,
        config=config,
    )


def update_score_accumulator_static(
    acc: GpuScoreAccumulator,
    *,
    state: GpuCarBatch,
    diagnostics: GpuScoringDiagnostics,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    active: torch.Tensor,
    collided: torch.Tensor,
    off_track: torch.Tensor,
    config: SimConfig,
    zero: torch.Tensor | None = None,
) -> None:
    speed_kph = state.speed_mps * 3.6
    if zero is None:
        zero = torch.zeros_like(speed_kph)
    lateral_error_m = diagnostics.lateral_error_m
    heading_error_deg = diagnostics.heading_error_deg
    curvature = state.yaw_rate_rps / torch.clamp(state.speed_mps, min=1e-6)
    sim_time_s = state.elapsed_steps.to(dtype=state.dtype) * float(config.car.dt)
    best_mask = active & (state.monotonic_progress_m >= acc.best_progress_m)
    acc.best_progress_m = _where(best_mask, state.monotonic_progress_m, acc.best_progress_m)
    acc.best_speed_kph = _where(best_mask, speed_kph, acc.best_speed_kph)
    acc.best_lateral_error_m = _where(best_mask, torch.abs(lateral_error_m), acc.best_lateral_error_m)
    acc.best_heading_error_deg = _where(best_mask, torch.abs(heading_error_deg), acc.best_heading_error_deg)

    progress_delta_m = torch.clamp(state.monotonic_progress_m - acc.start_progress_m, min=0.0)
    first_300 = active & (progress_delta_m <= 300.0)
    first_450 = active & (progress_delta_m <= 450.0)
    acc.speed_sum_first_300_m += torch.where(first_300, speed_kph, zero)
    acc.speed_count_first_300_m += first_300.to(dtype=state.dtype)
    acc.speed_sum_first_450_m += torch.where(first_450, speed_kph, zero)
    acc.speed_count_first_450_m += first_450.to(dtype=state.dtype)
    acc.brake_sum_first_300_m += torch.where(first_300, brake, zero)
    acc.brake_count_first_300_m += first_300.to(dtype=state.dtype)
    acc.brake_sum_first_450_m += torch.where(first_450, brake, zero)
    acc.brake_count_first_450_m += first_450.to(dtype=state.dtype)
    reached_300 = active & (acc.time_to_300_m < 0.0) & (progress_delta_m >= 300.0)
    reached_450 = active & (acc.time_to_450_m < 0.0) & (progress_delta_m >= 450.0)
    acc.time_to_300_m = _where(reached_300, sim_time_s, acc.time_to_300_m)
    acc.time_to_450_m = _where(reached_450, sim_time_s, acc.time_to_450_m)

    demand = (
        (diagnostics.future_brake_demand >= 0.22)
        | (diagnostics.brake_gate_proximity >= 0.70)
        | (diagnostics.target_speed_drop_norm >= 0.18)
    ) & (speed_kph >= 135.0) & active
    acc.demand_brake_sum += torch.where(demand, brake, zero)
    acc.demand_throttle_sum += torch.where(demand, throttle, zero)
    acc.demand_count += demand.to(dtype=state.dtype)
    acc.demand_max_future_brake_demand = torch.maximum(
        acc.demand_max_future_brake_demand,
        torch.where(demand, diagnostics.future_brake_demand, zero),
    )
    acc.max_brake = torch.maximum(acc.max_brake, torch.where(active, brake, zero))
    acc.throttle_sum += torch.where(active, throttle, zero)
    acc.row_count += active.to(dtype=state.dtype)
    acc.start_speed_for_score_kph = torch.where(
        (acc.row_count <= 1.0) & active,
        speed_kph,
        acc.start_speed_for_score_kph,
    )

    final_mask = active
    acc.final_x = _where(final_mask, state.x, acc.final_x)
    acc.final_y = _where(final_mask, state.y, acc.final_y)
    acc.final_heading_deg = _where(final_mask, state.heading_rad * (180.0 / torch.pi), acc.final_heading_deg)
    acc.final_speed_mps = _where(final_mask, state.speed_mps, acc.final_speed_mps)
    acc.final_speed_kph = _where(final_mask, speed_kph, acc.final_speed_kph)
    acc.final_raw_progress_m = _where(final_mask, state.raw_progress_m, acc.final_raw_progress_m)
    acc.final_progress_m = _where(final_mask, state.monotonic_progress_m, acc.final_progress_m)
    acc.final_lateral_error_m = _where(final_mask, lateral_error_m, acc.final_lateral_error_m)
    acc.final_heading_error_deg = _where(final_mask, heading_error_deg, acc.final_heading_error_deg)
    acc.final_yaw_rate_rps = _where(final_mask, state.yaw_rate_rps, acc.final_yaw_rate_rps)
    acc.final_curvature_rad_per_m = _where(final_mask, curvature, acc.final_curvature_rad_per_m)
    acc.final_steering = _where(final_mask, steer, acc.final_steering)
    acc.final_throttle = _where(final_mask, throttle, acc.final_throttle)
    acc.final_brake = _where(final_mask, brake, acc.final_brake)
    acc.final_step_index = torch.where(final_mask, state.elapsed_steps, acc.final_step_index)
    acc.final_sim_time_s = _where(final_mask, sim_time_s, acc.final_sim_time_s)
    acc.final_missed_checkpoint_count = torch.where(
        final_mask,
        state.missed_checkpoint_count,
        acc.final_missed_checkpoint_count,
    )
    acc.final_valid_lap = torch.where(final_mask, diagnostics.telemetry_valid_lap, acc.final_valid_lap)
    acc.final_finish_crossed = torch.where(final_mask, state.finish_crossed, acc.final_finish_crossed)
    acc.final_completed_lap = torch.where(final_mask, state.completed_lap, acc.final_completed_lap)
    acc.final_segment_complete = torch.where(final_mask, state.segment_complete, acc.final_segment_complete)
    acc.final_collided = torch.where(final_mask, collided, acc.final_collided)
    acc.final_off_track = torch.where(final_mask, off_track, acc.final_off_track)
    acc.final_terminated = torch.where(final_mask, state.terminated, acc.final_terminated)
    acc.final_truncated = torch.where(final_mask, state.truncated, acc.final_truncated)
    acc.final_termination_reason_id = torch.where(
        final_mask,
        state.termination_reason_id,
        acc.final_termination_reason_id,
    )
    for target_name, source_value in (
        ("final_target_speed_kph", diagnostics.target_speed_kph),
        ("final_near_target_speed_kph", diagnostics.near_target_speed_kph),
        ("final_min_future_target_speed_kph", diagnostics.min_future_target_speed_kph),
        ("final_target_speed_drop_kph", diagnostics.target_speed_drop_kph),
        ("final_target_speed_drop_norm", diagnostics.target_speed_drop_norm),
        ("final_brake_demand", diagnostics.brake_demand),
        ("final_future_brake_demand", diagnostics.future_brake_demand),
        ("final_brake_gate_proximity", diagnostics.brake_gate_proximity),
        ("final_braking_gate_distance_m", diagnostics.braking_gate_distance_m),
        ("final_brake_gate_distance_norm", diagnostics.brake_gate_distance_norm),
        ("final_lookahead_abs_max", diagnostics.lookahead_abs_max),
    ):
        setattr(acc, target_name, _where(final_mask, source_value, getattr(acc, target_name)))


def _avg(sum_tensor: torch.Tensor, count_tensor: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    return torch.where(count_tensor > 0.0, sum_tensor / torch.clamp(count_tensor, min=1.0), fallback)


def _profile_base_terms(
    acc: GpuScoreAccumulator,
    *,
    target_progress_m: float,
    terminate_at_target_progress: bool,
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
) -> dict[str, Any]:
    best_progress_m = acc.best_progress_m
    raw_progress_m = best_progress_m - acc.start_progress_m
    target = torch.full_like(best_progress_m, target_progress_m)
    progress_to_target_m = torch.where(
        torch.full_like(best_progress_m, terminate_at_target_progress, dtype=torch.bool),
        torch.minimum(best_progress_m, target) - acc.start_progress_m,
        raw_progress_m,
    )
    remaining_m = torch.clamp(target - best_progress_m, min=0.0)
    final_speed_kph = acc.final_speed_kph
    final_lateral_error_m = torch.abs(acc.final_lateral_error_m)
    final_heading_error_deg = torch.abs(acc.final_heading_error_deg)
    final_yaw_rate_rps = torch.abs(acc.final_yaw_rate_rps)
    final_steering = torch.abs(acc.final_steering)
    missed_checkpoints = acc.final_missed_checkpoint_count.to(dtype=best_progress_m.dtype)
    clean = ~(acc.final_collided | acc.final_off_track)
    milestone_complete = best_progress_m >= target
    segment_complete = acc.final_segment_complete | milestone_complete
    valid_finish = acc.final_completed_lap | (acc.final_valid_lap & acc.final_finish_crossed)
    target_span_m = torch.clamp(target - acc.start_progress_m, min=1.0)
    progress_ratio = torch.clamp(torch.minimum(raw_progress_m, target_span_m) / target_span_m, min=0.0, max=1.0)
    frontier_m = torch.clamp(progress_to_target_m - target_span_m * 0.70, min=0.0)
    near_target_m = torch.clamp(progress_to_target_m - target_span_m * 0.88, min=0.0)
    beyond_target_m = torch.clamp(best_progress_m - target, min=0.0)
    elapsed_s = torch.where(acc.final_sim_time_s > 0.0, acc.final_sim_time_s, acc.row_count / 60.0)
    pace_kph = raw_progress_m / torch.clamp(elapsed_s, min=1e-6) * 3.6
    avg_speed_first_300_m = _avg(acc.speed_sum_first_300_m, acc.speed_count_first_300_m, acc.final_speed_kph)
    avg_speed_first_450_m = _avg(acc.speed_sum_first_450_m, acc.speed_count_first_450_m, acc.final_speed_kph)
    avg_brake_first_450_m = _avg(acc.brake_sum_first_450_m, acc.brake_count_first_450_m, acc.final_brake)
    early_slow_penalty = torch.clamp(125.0 - avg_speed_first_300_m, min=0.0) * 34.0
    early_slow_penalty += torch.where(
        raw_progress_m >= 450.0,
        torch.clamp(145.0 - avg_speed_first_450_m, min=0.0) * 18.0,
        torch.zeros_like(raw_progress_m),
    )
    early_brake_penalty = torch.clamp(avg_brake_first_450_m - 0.32, min=0.0) * 1_800.0
    focus_start_m = min(frontier_focus_start_m, frontier_focus_end_m)
    focus_end_m = max(frontier_focus_start_m, frontier_focus_end_m)
    focus_span_m = max(1.0, focus_end_m - focus_start_m)
    focus_progress_m = torch.clamp(best_progress_m - focus_start_m, min=0.0, max=focus_span_m)
    focus_progress_ratio = focus_progress_m / focus_span_m
    reached_focus = best_progress_m >= focus_start_m
    cleared_focus = best_progress_m >= focus_end_m
    reason_collisionish = (
        (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["collision"])
        | (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["off_track"])
    )
    stalled_in_focus = reached_focus & (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"])
    focus_lateral_penalty = torch.clamp(acc.best_lateral_error_m - 9.0, min=0.0) * 420.0
    focus_heading_penalty = torch.clamp(acc.best_heading_error_deg - 18.0, min=0.0) * 520.0
    focus_stop_penalty = torch.where(
        reached_focus,
        torch.clamp(90.0 - final_speed_kph, min=0.0) * 170.0,
        torch.zeros_like(final_speed_kph),
    )
    late_progress_factor = torch.clamp((best_progress_m - 3000.0) / 2200.0, min=0.0, max=1.0)
    frontier_quality_factor = torch.clamp((best_progress_m - 1500.0) / 3500.0, min=0.0, max=1.0)
    lap_progress_ratio = torch.clamp(best_progress_m / MONZA_LENGTH_METERS, min=0.0, max=1.0)
    avg_brake_demand_zone = _avg(acc.demand_brake_sum, acc.demand_count, torch.zeros_like(best_progress_m))
    avg_throttle_demand_zone = _avg(acc.demand_throttle_sum, acc.demand_count, torch.zeros_like(best_progress_m))
    max_future_brake_demand = acc.demand_max_future_brake_demand
    setup_penalty = frontier_quality_factor * (
        torch.clamp(final_lateral_error_m - 12.0, min=0.0) * 620.0
        + torch.clamp(final_heading_error_deg - 22.0, min=0.0) * 720.0
        + torch.clamp(final_yaw_rate_rps - 0.85, min=0.0) * 2_200.0
    )
    late_setup_penalty = late_progress_factor * (
        torch.clamp(final_lateral_error_m - 9.0, min=0.0) * 900.0
        + torch.clamp(final_heading_error_deg - 16.0, min=0.0) * 1_050.0
        + torch.clamp(115.0 - final_speed_kph, min=0.0) * 120.0
    )
    brake_demand_penalty = frontier_quality_factor * max_future_brake_demand * (
        torch.clamp(0.32 - avg_brake_demand_zone, min=0.0) * 18_000.0
        + avg_throttle_demand_zone * 7_500.0
    )
    viability_penalty = setup_penalty + late_setup_penalty + brake_demand_penalty
    valid_elapsed_s = torch.where(valid_finish & (elapsed_s > 0.0), elapsed_s, torch.full_like(elapsed_s, torch.inf))
    valid_lap_speed_bonus = torch.where(
        valid_finish & torch.isfinite(valid_elapsed_s),
        8_500_000.0
        + torch.clamp(220.0 - valid_elapsed_s, min=0.0) * 58_000.0
        + torch.clamp(170.0 - valid_elapsed_s, min=0.0) * 72_000.0
        + torch.clamp(130.0 - valid_elapsed_s, min=0.0) * 120_000.0
        + torch.clamp(100.0 - valid_elapsed_s, min=0.0) * 180_000.0
        + torch.clamp(85.0 - valid_elapsed_s, min=0.0) * 260_000.0
        + torch.minimum(pace_kph, torch.full_like(pace_kph, 380.0)) * 16_000.0
        - torch.clamp(valid_elapsed_s - 100.0, min=0.0) * 34_000.0
        - torch.clamp(valid_elapsed_s - 130.0, min=0.0) * 80_000.0
        - torch.clamp(valid_elapsed_s - 170.0, min=0.0) * 140_000.0,
        torch.zeros_like(elapsed_s),
    )
    return locals()


def score_profiles_batch(
    acc: GpuScoreAccumulator,
    *,
    profiles: tuple[str, ...],
    target_progress_m: float,
    terminate_at_target_progress: bool,
    frontier_focus_start_m: float,
    frontier_focus_end_m: float,
) -> dict[str, torch.Tensor]:
    terms = _profile_base_terms(
        acc,
        target_progress_m=target_progress_m,
        terminate_at_target_progress=terminate_at_target_progress,
        frontier_focus_start_m=frontier_focus_start_m,
        frontier_focus_end_m=frontier_focus_end_m,
    )
    scores: dict[str, torch.Tensor] = {}
    for profile in profiles:
        collision_penalty = 4_500.0 if profile == "frontier" else 7_000.0 if profile == "risk_seeking" else 24_000.0
        score = terms["progress_to_target_m"] * 34.0 - terms["remaining_m"] * 36.0
        score += terms["frontier_m"] * 65.0 + terms["near_target_m"] * 120.0 + terms["beyond_target_m"] * 160.0
        score += torch.where(terms["segment_complete"], torch.full_like(score, 160_000.0), torch.zeros_like(score))
        score += torch.where(
            terms["valid_finish"],
            torch.full_like(score, 300_000.0 if profile == "full_lap_validity" else 240_000.0),
            torch.zeros_like(score),
        )
        segment_failure = (acc.final_termination_reason_id >= 2) & (acc.final_termination_reason_id <= 8)
        score -= torch.where(terms["reason_collisionish"], torch.full_like(score, collision_penalty), torch.zeros_like(score))
        score -= torch.where(segment_failure, torch.full_like(score, 7_500.0), torch.zeros_like(score))
        score -= torch.where(
            acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"],
            torch.full_like(score, 3_500.0),
            torch.zeros_like(score),
        )
        stalled = (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["max_steps"]) & (terms["final_speed_kph"] < 20.0)
        score -= torch.where(stalled, torch.full_like(score, 6_000.0), torch.zeros_like(score))
        score += torch.where(terms["clean"], 3_500.0 + terms["progress_ratio"] * 3_000.0, torch.zeros_like(score))

        time_to_450_valid = acc.time_to_450_m >= 0.0
        if profile == "fast_valid_lap":
            score += terms["raw_progress_m"] * 20.0 + terms["best_progress_m"] * 16.0
            score += terms["frontier_m"] * 70.0 + terms["near_target_m"] * 160.0 + terms["beyond_target_m"] * 250.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 380.0)) * 4_600.0
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 360.0)) * 120.0
            score += torch.where(
                terms["valid_finish"],
                terms["valid_lap_speed_bonus"]
                - torch.clamp(terms["valid_elapsed_s"] - 150.0, min=0.0) * 95_000.0
                - torch.clamp(terms["valid_elapsed_s"] - 180.0, min=0.0) * 140_000.0,
                -terms["remaining_m"] * 70.0
                - torch.where(
                    acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["max_steps"],
                    80_000.0 + torch.clamp(terms["elapsed_s"] - 120.0, min=0.0) * 800.0,
                    torch.zeros_like(score),
                ),
            )
            score += torch.where(time_to_450_valid, torch.clamp(16.5 - acc.time_to_450_m, min=0.0) * 1_200.0, torch.zeros_like(score))
            score -= terms["early_slow_penalty"] * 2.40
            score -= terms["early_brake_penalty"] * 1.60
            score -= terms["viability_penalty"] * torch.where(terms["valid_finish"], torch.full_like(score, 0.35), torch.full_like(score, 0.70))
            score -= terms["missed_checkpoints"] * 12_000.0
        elif profile == "time_attack":
            score += terms["raw_progress_m"] * 12.0 + terms["best_progress_m"] * 10.0
            score += terms["frontier_m"] * 55.0 + terms["near_target_m"] * 130.0 + terms["beyond_target_m"] * 190.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 400.0)) * 8_800.0 * torch.maximum(
                torch.full_like(score, 0.20),
                terms["progress_ratio"],
            )
            score += torch.where(
                terms["valid_finish"],
                6_500_000.0
                + torch.clamp(180.0 - terms["valid_elapsed_s"], min=0.0) * 95_000.0
                + torch.clamp(120.0 - terms["valid_elapsed_s"], min=0.0) * 160_000.0
                + torch.clamp(90.0 - terms["valid_elapsed_s"], min=0.0) * 280_000.0
                - torch.clamp(terms["valid_elapsed_s"] - 120.0, min=0.0) * 80_000.0
                - torch.clamp(terms["valid_elapsed_s"] - 150.0, min=0.0) * 150_000.0,
                -terms["remaining_m"] * 90.0
                - torch.where(
                    (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["max_steps"])
                    | (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"]),
                    torch.full_like(score, 90_000.0),
                    torch.zeros_like(score),
                ),
            )
            score -= terms["early_slow_penalty"] * 2.10
            score -= terms["viability_penalty"] * 0.45
            score -= terms["missed_checkpoints"] * 14_000.0
        elif profile == "lap_pace":
            score += terms["raw_progress_m"] * 24.0 + terms["best_progress_m"] * 12.0
            score += terms["frontier_m"] * 60.0 + terms["near_target_m"] * 135.0 + terms["beyond_target_m"] * 210.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 390.0)) * 3_800.0 * torch.maximum(
                torch.full_like(score, 0.30),
                terms["frontier_quality_factor"],
            )
            score += torch.minimum(terms["avg_speed_first_450_m"], torch.full_like(score, 340.0)) * 150.0
            score += torch.where(
                terms["valid_finish"],
                2_800_000.0
                + torch.clamp(190.0 - terms["valid_elapsed_s"], min=0.0) * 50_000.0
                - torch.clamp(terms["valid_elapsed_s"] - 150.0, min=0.0) * 55_000.0,
                torch.zeros_like(score),
            )
            score -= torch.where(
                acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["max_steps"],
                torch.clamp(terms["elapsed_s"] - 150.0, min=0.0) * 1_200.0,
                torch.zeros_like(score),
            )
            score -= terms["early_slow_penalty"] * 1.80
            score -= terms["viability_penalty"] * 0.50
            score -= terms["missed_checkpoints"] * 8_000.0
        elif profile == "fast_frontier":
            progress_pace_factor = 0.18 + terms["lap_progress_ratio"] * 1.95
            score += terms["raw_progress_m"] * 18.0 + terms["best_progress_m"] * 120.0
            score += terms["frontier_m"] * 70.0 + terms["near_target_m"] * 130.0 + terms["beyond_target_m"] * 170.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 360.0)) * 13_500.0 * progress_pace_factor
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 360.0)) * 1_200.0 * torch.maximum(
                torch.full_like(score, 0.15),
                terms["lap_progress_ratio"],
            )
            score += torch.minimum(terms["avg_speed_first_450_m"], torch.full_like(score, 340.0)) * 160.0
            score += torch.where(terms["best_progress_m"] >= 3000.0, 260_000.0 + torch.minimum(terms["pace_kph"], torch.full_like(score, 330.0)) * 1_200.0, torch.zeros_like(score))
            score += torch.where(terms["best_progress_m"] >= 4000.0, 360_000.0 + torch.minimum(terms["pace_kph"], torch.full_like(score, 330.0)) * 1_700.0, torch.zeros_like(score))
            score += torch.where(terms["best_progress_m"] >= 5000.0, 520_000.0 + torch.minimum(terms["pace_kph"], torch.full_like(score, 330.0)) * 2_400.0, torch.zeros_like(score))
            score -= torch.where(
                terms["valid_finish"],
                650_000.0 + torch.clamp(terms["valid_elapsed_s"] - 150.0, min=0.0) * 90_000.0,
                terms["remaining_m"] * 28.0,
            )
            score -= torch.where(terms["best_progress_m"] >= 2400.0, torch.clamp(175.0 - terms["pace_kph"], min=0.0) * 14_000.0, torch.zeros_like(score))
            score -= torch.where(
                ((acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["max_steps"])
                 | (acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"]))
                & ~terms["valid_finish"],
                65_000.0 + torch.clamp(terms["elapsed_s"] - 130.0, min=0.0) * 900.0,
                torch.zeros_like(score),
            )
            score -= terms["early_slow_penalty"] * 1.45
            score -= terms["viability_penalty"] * 0.38
            score -= terms["missed_checkpoints"] * 10_000.0
        elif profile == "frontier":
            score += terms["best_progress_m"] * 12.0 + torch.minimum(terms["final_speed_kph"], torch.full_like(score, 360.0)) * 18.0
            score += terms["frontier_m"] * 95.0 + terms["near_target_m"] * 210.0 + terms["beyond_target_m"] * 320.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 300.0)) * 16.0
            score -= terms["viability_penalty"] * 0.42
            score -= terms["final_lateral_error_m"] * 72.0
            score -= terms["final_heading_error_deg"] * 26.0
            score -= terms["final_yaw_rate_rps"] * 330.0
            score -= terms["final_steering"] * 300.0
            score -= terms["missed_checkpoints"] * 2_500.0
        elif profile == "frontier_fast":
            score += terms["best_progress_m"] * 18.0 + torch.minimum(terms["final_speed_kph"], torch.full_like(score, 380.0)) * 20.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 310.0)) * 86.0
            score += terms["frontier_m"] * 105.0 + terms["near_target_m"] * 235.0 + terms["beyond_target_m"] * 340.0
            score += torch.where(terms["valid_finish"], terms["valid_lap_speed_bonus"] * 0.18, torch.zeros_like(score))
            score += torch.where(time_to_450_valid, torch.clamp(18.0 - acc.time_to_450_m, min=0.0) * 260.0, torch.zeros_like(score))
            score += torch.minimum(terms["avg_speed_first_450_m"], torch.full_like(score, 320.0)) * 28.0
            score -= terms["early_slow_penalty"]
            score -= terms["early_brake_penalty"]
            score -= terms["viability_penalty"] * 0.58
            score -= terms["final_lateral_error_m"] * 70.0
            score -= terms["final_heading_error_deg"] * 24.0
            score -= terms["final_yaw_rate_rps"] * 320.0
            score -= terms["final_steering"] * 260.0
            score -= terms["missed_checkpoints"] * 2_500.0
        elif profile == "frontier_recovery":
            score += terms["raw_progress_m"] * 18.0 + terms["best_progress_m"] * 10.0
            score += terms["focus_progress_m"] * 1_250.0 + terms["focus_progress_ratio"] * 28_000.0
            score += torch.minimum(acc.best_speed_kph, torch.full_like(score, 300.0)) * 35.0
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 260.0)) * 115.0
            score -= terms["focus_lateral_penalty"] + terms["focus_heading_penalty"] + terms["focus_stop_penalty"]
            score -= terms["missed_checkpoints"] * 3_000.0
            score += torch.where(terms["cleared_focus"], torch.full_like(score, 55_000.0), torch.zeros_like(score))
            score -= torch.where(terms["stalled_in_focus"], torch.full_like(score, 55_000.0), torch.zeros_like(score))
            score -= torch.where(
                acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"],
                torch.full_like(score, 18_000.0),
                torch.zeros_like(score),
            )
            score -= torch.where(terms["reason_collisionish"] & terms["reached_focus"], torch.full_like(score, 10_000.0), torch.zeros_like(score))
        elif profile == "frontier_novelty":
            speed_bucket = torch.clamp(torch.floor(torch.clamp(terms["final_speed_kph"], min=0.0) / 55.0), max=5.0)
            lateral_bucket = torch.clamp(torch.floor(acc.best_lateral_error_m / 4.0), max=5.0)
            heading_bucket = torch.clamp(torch.floor(acc.best_heading_error_deg / 10.0), max=5.0)
            novelty_hint = speed_bucket * 1_100.0 + (5.0 - lateral_bucket) * 750.0 + (5.0 - heading_bucket) * 650.0
            score += terms["raw_progress_m"] * 16.0 + terms["focus_progress_m"] * 900.0 + terms["beyond_target_m"] * 140.0
            score += novelty_hint
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 280.0)) * 18.0
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 320.0)) * 70.0
            score -= terms["focus_lateral_penalty"] * 0.80 + terms["focus_heading_penalty"] * 0.80 + terms["focus_stop_penalty"] * 1.20
            score += torch.where(terms["cleared_focus"], torch.full_like(score, 36_000.0), torch.zeros_like(score))
            score -= torch.where(terms["stalled_in_focus"], torch.full_like(score, 65_000.0), torch.zeros_like(score))
            score -= torch.where(
                acc.final_termination_reason_id == TERMINATION_REASON_TO_ID["no_progress"],
                torch.full_like(score, 20_000.0),
                torch.zeros_like(score),
            )
        elif profile == "risk_seeking":
            score += terms["best_progress_m"] * 11.0 + torch.minimum(terms["final_speed_kph"], torch.full_like(score, 380.0)) * 24.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 320.0)) * 22.0
            score += terms["frontier_m"] * 70.0 + terms["near_target_m"] * 150.0 + terms["beyond_target_m"] * 220.0
            score -= terms["viability_penalty"] * 0.18
            score -= terms["final_lateral_error_m"] * 52.0
            score -= terms["final_heading_error_deg"] * 18.0
        elif profile == "farthest_distance":
            score += terms["raw_progress_m"] * 31.0 + terms["best_progress_m"] * 22.0
            score += terms["frontier_m"] * 75.0 + terms["near_target_m"] * 160.0 + terms["beyond_target_m"] * 280.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 300.0)) * 20.0
            score -= terms["viability_penalty"] * 0.22
            score -= terms["missed_checkpoints"] * 1_600.0
            score -= torch.where(terms["reason_collisionish"], torch.full_like(score, 1_500.0), torch.zeros_like(score))
        elif profile == "early_pace":
            score += terms["raw_progress_m"] * 20.0 + torch.minimum(terms["pace_kph"], torch.full_like(score, 320.0)) * 125.0
            score += torch.minimum(terms["avg_speed_first_300_m"], torch.full_like(score, 300.0)) * 56.0
            score += torch.minimum(terms["avg_speed_first_450_m"], torch.full_like(score, 320.0)) * 38.0
            score += torch.where(time_to_450_valid, torch.clamp(20.0 - acc.time_to_450_m, min=0.0) * 420.0, torch.zeros_like(score))
            score += terms["frontier_m"] * 38.0 + terms["beyond_target_m"] * 90.0
            score -= terms["early_slow_penalty"] * 1.55
            score -= terms["early_brake_penalty"] * 1.35
            score -= terms["final_lateral_error_m"] * 45.0
            score -= terms["final_heading_error_deg"] * 15.0
            score -= terms["missed_checkpoints"] * 1_800.0
        elif profile == "clean_distance":
            score += terms["raw_progress_m"] * 28.0 + terms["best_progress_m"] * 15.0
            score += terms["frontier_m"] * 58.0 + terms["near_target_m"] * 115.0 + terms["beyond_target_m"] * 190.0
            score += torch.minimum(terms["pace_kph"], torch.full_like(score, 280.0)) * 18.0
            score -= terms["viability_penalty"] * 0.82
            score -= terms["final_lateral_error_m"] * 210.0
            score -= terms["final_heading_error_deg"] * 80.0
            score -= terms["final_yaw_rate_rps"] * 560.0
            score -= terms["final_steering"] * 520.0
            score -= terms["missed_checkpoints"] * 4_500.0
            score += torch.where(terms["clean"], torch.full_like(score, 7_000.0), torch.zeros_like(score))
        elif profile == "clean_exit":
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 320.0)) * 4.0
            score -= terms["final_lateral_error_m"] * 260.0
            score -= terms["final_heading_error_deg"] * 95.0
            score -= terms["final_yaw_rate_rps"] * 600.0
            score -= terms["final_steering"] * 800.0
            score -= terms["missed_checkpoints"] * 5_000.0
        elif profile == "brake_zone":
            avg_throttle = acc.throttle_sum / torch.clamp(acc.row_count, min=1.0)
            speed_drop = torch.clamp(acc.start_speed_for_score_kph - terms["final_speed_kph"], min=0.0)
            score += acc.max_brake * 18_000.0 + speed_drop * 120.0
            score -= avg_throttle * 4_000.0
            score -= terms["final_lateral_error_m"] * 110.0
        elif profile == "apex":
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 240.0)) * 3.0
            score -= terms["final_lateral_error_m"] * 340.0
            score -= terms["final_heading_error_deg"] * 130.0
            score -= terms["final_yaw_rate_rps"] * 850.0
            score -= terms["missed_checkpoints"] * 5_000.0
        elif profile == "exit_speed":
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 340.0)) * 18.0
            score -= terms["final_lateral_error_m"] * 150.0
            score -= terms["final_heading_error_deg"] * 45.0
            score -= terms["final_steering"] * 500.0
        elif profile == "full_lap_validity":
            score += terms["best_progress_m"] * 5.0
            score += torch.where(acc.final_valid_lap, torch.full_like(score, 10_000.0), torch.full_like(score, -10_000.0))
            score -= terms["missed_checkpoints"] * 10_000.0
            score -= terms["final_lateral_error_m"] * 120.0
            score -= terms["final_heading_error_deg"] * 40.0
        else:
            score += torch.minimum(terms["final_speed_kph"], torch.full_like(score, 320.0)) * 5.0
            score -= terms["final_lateral_error_m"] * 120.0
            score -= terms["final_heading_error_deg"] * 35.0
            score -= terms["final_yaw_rate_rps"] * 250.0
            score -= terms["final_steering"] * 500.0
            score -= terms["missed_checkpoints"] * 2_500.0
        scores[profile] = score
    return scores


def termination_reason_strings(ids: torch.Tensor) -> list[str]:
    return [TERMINATION_ID_TO_REASON.get(int(value), "unknown") for value in ids.detach().cpu().tolist()]
