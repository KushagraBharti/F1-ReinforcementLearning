import numpy as np
import torch

from f1rl.evolution_search import EvolutionGates, _score_rows
from f1rl.gpu_scoring import TERMINATION_REASON_TO_ID, GpuScoreAccumulator, score_profiles_batch


def _tensor(values, *, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


def _bool(values):
    return torch.tensor(values, dtype=torch.bool)


def _int(values):
    return torch.tensor(values, dtype=torch.int64)


def _accumulator_from_terminal_rows(rows: list[dict]) -> GpuScoreAccumulator:
    start = _tensor([0.0 for _ in rows])
    final_progress = _tensor([row["monotonic_progress_m"] for row in rows])
    elapsed = _tensor([row["sim_time_s"] for row in rows])
    speed = _tensor([row["speed_kph"] for row in rows])
    brake = _tensor([row["brake"] for row in rows])
    throttle = _tensor([row["throttle"] for row in rows])
    lateral = _tensor([row["lateral_error_m"] for row in rows])
    heading = _tensor([row["heading_error_deg"] for row in rows])
    yaw = _tensor([row["yaw_rate_rps"] for row in rows])
    steer = _tensor([row["steering"] for row in rows])
    missed = _int([row["missed_checkpoint_count"] for row in rows])
    valid = _bool([row["valid_lap"] for row in rows])
    finish = _bool([row["finish_crossed"] for row in rows])
    completed = _bool([row["completed_lap"] for row in rows])
    segment = _bool([row["segment_complete"] for row in rows])
    collided = _bool([row["collided"] for row in rows])
    off_track = _bool([row["off_track"] for row in rows])
    reasons = _int([TERMINATION_REASON_TO_ID[row["termination_reason"]] for row in rows])
    zeros = torch.zeros_like(final_progress)
    ones = torch.ones_like(final_progress)
    return GpuScoreAccumulator(
        start_progress_m=start,
        start_speed_kph=speed,
        start_speed_for_score_kph=speed,
        best_progress_m=final_progress,
        best_speed_kph=speed,
        best_lateral_error_m=torch.abs(lateral),
        best_heading_error_deg=torch.abs(heading),
        final_x=zeros,
        final_y=zeros,
        final_heading_deg=zeros,
        final_speed_mps=speed / 3.6,
        final_speed_kph=speed,
        final_raw_progress_m=final_progress,
        final_progress_m=final_progress,
        final_lateral_error_m=lateral,
        final_heading_error_deg=heading,
        final_yaw_rate_rps=yaw,
        final_curvature_rad_per_m=zeros,
        final_steering=steer,
        final_throttle=throttle,
        final_brake=brake,
        final_step_index=_int([int(row["sim_time_s"] * 60.0) for row in rows]),
        final_sim_time_s=elapsed,
        final_missed_checkpoint_count=missed,
        final_valid_lap=valid,
        final_finish_crossed=finish,
        final_completed_lap=completed,
        final_segment_complete=segment,
        final_collided=collided,
        final_off_track=off_track,
        final_terminated=collided | off_track,
        final_truncated=segment | completed,
        final_termination_reason_id=reasons,
        final_target_speed_kph=zeros,
        final_near_target_speed_kph=zeros,
        final_min_future_target_speed_kph=zeros,
        final_target_speed_drop_kph=zeros,
        final_target_speed_drop_norm=zeros,
        final_brake_demand=zeros,
        final_future_brake_demand=zeros,
        final_brake_gate_proximity=zeros,
        final_braking_gate_distance_m=zeros,
        final_brake_gate_distance_norm=zeros,
        final_lookahead_abs_max=zeros,
        time_to_300_m=torch.where(final_progress >= 300.0, elapsed, torch.full_like(elapsed, -1.0)),
        time_to_450_m=torch.where(final_progress >= 450.0, elapsed, torch.full_like(elapsed, -1.0)),
        speed_sum_first_300_m=speed,
        speed_count_first_300_m=ones,
        speed_sum_first_450_m=speed,
        speed_count_first_450_m=ones,
        brake_sum_first_300_m=brake,
        brake_count_first_300_m=ones,
        brake_sum_first_450_m=brake,
        brake_count_first_450_m=ones,
        max_brake=brake,
        throttle_sum=throttle,
        row_count=ones,
        demand_brake_sum=zeros,
        demand_throttle_sum=zeros,
        demand_count=zeros,
        demand_max_future_brake_demand=zeros,
    )


def test_gpu_scoring_profiles_match_cpu_for_terminal_rows() -> None:
    rows = [
        {
            "monotonic_progress_m": 650.0,
            "sim_time_s": 5.0,
            "speed_kph": 210.0,
            "lateral_error_m": 2.0,
            "heading_error_deg": 5.0,
            "yaw_rate_rps": 0.2,
            "steering": 0.2,
            "throttle": 0.2,
            "brake": 0.0,
            "missed_checkpoint_count": 0,
            "segment_complete": True,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": False,
            "off_track": False,
            "termination_reason": "segment_complete",
        },
        {
            "monotonic_progress_m": 1800.0,
            "sim_time_s": 12.0,
            "speed_kph": 250.0,
            "lateral_error_m": 4.0,
            "heading_error_deg": 6.0,
            "yaw_rate_rps": 0.1,
            "steering": 0.1,
            "throttle": 0.7,
            "brake": 0.0,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": False,
            "valid_lap": True,
            "finish_crossed": False,
            "collided": True,
            "off_track": False,
            "termination_reason": "collision",
        },
    ]
    profiles = ("max_progress", "frontier_fast", "early_pace", "farthest_distance", "clean_exit")
    gates = EvolutionGates(target_progress_m=650.0, terminate_at_target_progress=False)
    acc = _accumulator_from_terminal_rows(rows)
    gpu_scores = score_profiles_batch(
        acc,
        profiles=profiles,
        target_progress_m=gates.target_progress_m,
        terminate_at_target_progress=gates.terminate_at_target_progress,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    for index, row in enumerate(rows):
        cpu_scores = _score_rows([row], start_progress_m=0.0, gates=gates, scoring_profiles=profiles)
        for profile in profiles:
            assert np.isclose(float(gpu_scores[profile][index]), cpu_scores[profile], rtol=1e-7, atol=1e-5)
