# pyright: reportPrivateImportUsage=false

from dataclasses import fields

import numpy as np
import pytest
import torch

from f1rl.config import SimConfig
from f1rl.evolution_search import SCORING_PROFILES, EvolutionGates, _score_rows
from f1rl.gpu_fast_warp import warp_status
from f1rl.gpu_fused_warp import score_profiles_warp_batch, update_score_accumulator_warp_batch
from f1rl.gpu_scoring import (
    TERMINATION_REASON_TO_ID,
    GpuScoreAccumulator,
    GpuScoringDiagnostics,
    create_score_accumulator,
    score_profiles_batch,
    update_score_accumulator,
    update_score_accumulator_static,
)
from f1rl.gpu_types import GpuCarBatch


def _tensor(values, *, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


def _bool(values):
    return torch.tensor(values, dtype=torch.bool)


def _int(values):
    return torch.tensor(values, dtype=torch.int64)


def _gpu_state() -> GpuCarBatch:
    zeros = _tensor([0.0, 0.0])
    false = _bool([False, False])
    return GpuCarBatch(
        x=_tensor([100.0, 120.0]),
        y=_tensor([200.0, 210.0]),
        heading_rad=_tensor([0.1, -0.2]),
        speed_mps=_tensor([42.0, 51.0]),
        yaw_rate_rps=_tensor([0.05, -0.02]),
        steering=_tensor([0.03, -0.04]),
        raw_progress_m=_tensor([12.0, 14.0]),
        monotonic_progress_m=_tensor([12.0, 14.0]),
        last_raw_progress_px=_tensor([120.0, 140.0]),
        checkpoint_index=_int([0, 0]),
        next_checkpoint_index=_int([1, 1]),
        checkpoints_passed=_int([0, 0]),
        missed_checkpoint_count=_int([0, 1]),
        lap_index=_int([0, 0]),
        elapsed_steps=_int([1, 1]),
        no_progress_steps=_int([0, 0]),
        alive=_bool([True, True]),
        terminated=false.clone(),
        truncated=false.clone(),
        termination_reason_id=_int([TERMINATION_REASON_TO_ID["active"], TERMINATION_REASON_TO_ID["collision"]]),
        valid_lap=_bool([True, False]),
        finish_crossed=false.clone(),
        completed_lap=false.clone(),
        segment_complete=false.clone(),
        segment_release_observed=false.clone(),
        last_throttle=zeros.clone(),
        last_brake=zeros.clone(),
        last_steer=zeros.clone(),
    )


def _state_to_cuda_float32(state: GpuCarBatch) -> GpuCarBatch:
    values = {}
    for field in fields(GpuCarBatch):
        tensor = getattr(state, field.name)
        if tensor.dtype == torch.bool:
            values[field.name] = tensor.to(device="cuda", dtype=torch.bool)
        elif tensor.is_floating_point():
            values[field.name] = tensor.to(device="cuda", dtype=torch.float32)
        else:
            values[field.name] = tensor.to(device="cuda", dtype=torch.int64)
    return GpuCarBatch(**values)


def _assert_accumulators_equal(left: GpuScoreAccumulator, right: GpuScoreAccumulator) -> None:
    for field in fields(GpuScoreAccumulator):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if left_value.dtype == torch.bool or not left_value.is_floating_point():
            assert torch.equal(left_value, right_value), field.name
        else:
            assert torch.allclose(left_value, right_value), field.name


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


def _accumulator_to_cuda_float32(acc: GpuScoreAccumulator) -> GpuScoreAccumulator:
    values = {}
    for field in fields(GpuScoreAccumulator):
        tensor = getattr(acc, field.name)
        if tensor.dtype == torch.bool:
            values[field.name] = tensor.to(device="cuda", dtype=torch.bool)
        elif tensor.is_floating_point():
            values[field.name] = tensor.to(device="cuda", dtype=torch.float32)
        else:
            values[field.name] = tensor.to(device="cuda", dtype=torch.int64)
    return GpuScoreAccumulator(**values)


def test_static_score_accumulator_update_matches_mapping_api() -> None:
    state = _gpu_state()
    diagnostics = {
        "lateral_error_m": _tensor([1.2, -2.5]),
        "heading_error_deg": _tensor([3.0, -4.0]),
        "future_brake_demand": _tensor([0.3, 0.1]),
        "brake_gate_proximity": _tensor([0.1, 0.8]),
        "target_speed_drop_norm": _tensor([0.2, 0.0]),
        "telemetry_valid_lap": _bool([True, False]),
        "target_speed_kph": _tensor([180.0, 170.0]),
        "near_target_speed_kph": _tensor([175.0, 165.0]),
        "min_future_target_speed_kph": _tensor([150.0, 145.0]),
        "target_speed_drop_kph": _tensor([20.0, 15.0]),
        "brake_demand": _tensor([0.4, 0.2]),
        "braking_gate_distance_m": _tensor([80.0, 70.0]),
        "brake_gate_distance_norm": _tensor([0.5, 0.4]),
        "lookahead_abs_max": _tensor([0.08, 0.12]),
    }
    throttle = _tensor([0.7, 0.2])
    brake = _tensor([0.1, 0.8])
    steer = _tensor([0.05, -0.25])
    active = _bool([True, True])
    collided = _bool([False, True])
    off_track = _bool([False, False])

    mapping_acc = create_score_accumulator(state)
    static_acc = create_score_accumulator(state)
    update_score_accumulator(
        mapping_acc,
        state=state,
        diagnostics=diagnostics,
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=active,
        collided=collided,
        off_track=off_track,
        config=SimConfig(),
    )
    update_score_accumulator_static(
        static_acc,
        state=state,
        diagnostics=GpuScoringDiagnostics.from_mapping(diagnostics, fallback_valid_lap=state.valid_lap),
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=active,
        collided=collided,
        off_track=off_track,
        config=SimConfig(),
    )

    _assert_accumulators_equal(mapping_acc, static_acc)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp accumulator parity requires CUDA and Warp CUDA support",
)
def test_warp_score_accumulator_update_matches_torch_reference() -> None:
    state = _state_to_cuda_float32(_gpu_state())
    diagnostics_mapping = {
        "lateral_error_m": _tensor([1.2, -2.5], dtype=torch.float32).cuda(),
        "heading_error_deg": _tensor([3.0, -4.0], dtype=torch.float32).cuda(),
        "future_brake_demand": _tensor([0.3, 0.1], dtype=torch.float32).cuda(),
        "brake_gate_proximity": _tensor([0.1, 0.8], dtype=torch.float32).cuda(),
        "target_speed_drop_norm": _tensor([0.2, 0.0], dtype=torch.float32).cuda(),
        "telemetry_valid_lap": _bool([True, False]).cuda(),
        "target_speed_kph": _tensor([180.0, 170.0], dtype=torch.float32).cuda(),
        "near_target_speed_kph": _tensor([175.0, 165.0], dtype=torch.float32).cuda(),
        "min_future_target_speed_kph": _tensor([150.0, 145.0], dtype=torch.float32).cuda(),
        "target_speed_drop_kph": _tensor([20.0, 15.0], dtype=torch.float32).cuda(),
        "brake_demand": _tensor([0.4, 0.2], dtype=torch.float32).cuda(),
        "braking_gate_distance_m": _tensor([80.0, 70.0], dtype=torch.float32).cuda(),
        "brake_gate_distance_norm": _tensor([0.5, 0.4], dtype=torch.float32).cuda(),
        "lookahead_abs_max": _tensor([0.08, 0.12], dtype=torch.float32).cuda(),
    }
    diagnostics = GpuScoringDiagnostics.from_mapping(diagnostics_mapping, fallback_valid_lap=state.valid_lap)
    throttle = _tensor([0.7, 0.2], dtype=torch.float32).cuda()
    brake = _tensor([0.1, 0.8], dtype=torch.float32).cuda()
    steer = _tensor([0.05, -0.25], dtype=torch.float32).cuda()
    active = _bool([True, False]).cuda()
    collided = _bool([False, True]).cuda()
    off_track = _bool([False, False]).cuda()

    torch_acc = create_score_accumulator(state)
    warp_acc = create_score_accumulator(state)
    update_score_accumulator_static(
        torch_acc,
        state=state,
        diagnostics=diagnostics,
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=active,
        collided=collided,
        off_track=off_track,
        config=SimConfig(),
    )
    update_score_accumulator_warp_batch(
        warp_acc,
        state=state,
        diagnostics=diagnostics,
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=active,
        collided=collided,
        off_track=off_track,
        config=SimConfig(),
    )
    torch.cuda.synchronize()

    _assert_accumulators_equal(torch_acc, warp_acc)


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp score-profile parity requires CUDA and Warp CUDA support",
)
def test_warp_score_profiles_match_torch_reference() -> None:
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
        {
            "monotonic_progress_m": 5300.0,
            "sim_time_s": 92.0,
            "speed_kph": 300.0,
            "lateral_error_m": 1.0,
            "heading_error_deg": 2.0,
            "yaw_rate_rps": 0.03,
            "steering": 0.03,
            "throttle": 0.9,
            "brake": 0.0,
            "missed_checkpoint_count": 0,
            "segment_complete": False,
            "completed_lap": True,
            "valid_lap": True,
            "finish_crossed": True,
            "collided": False,
            "off_track": False,
            "termination_reason": "lap_complete",
        },
    ]
    profiles = tuple(sorted(SCORING_PROFILES))
    acc = _accumulator_to_cuda_float32(_accumulator_from_terminal_rows(rows))

    expected = score_profiles_batch(
        acc,
        profiles=profiles,
        target_progress_m=650.0,
        terminate_at_target_progress=False,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    actual = score_profiles_warp_batch(
        acc,
        profiles=profiles,
        target_progress_m=650.0,
        terminate_at_target_progress=False,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    torch.cuda.synchronize()

    for profile in profiles:
        assert torch.allclose(actual[profile], expected[profile], rtol=5e-5, atol=64.0), profile


def test_warp_core_score_profiles_reject_unsupported_profiles() -> None:
    acc = _accumulator_from_terminal_rows(
        [
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
            }
        ]
    )

    with pytest.raises(ValueError, match="does not support"):
        score_profiles_warp_batch(
            acc,
            profiles=("unknown_future_profile",),
            target_progress_m=650.0,
            terminate_at_target_progress=False,
            frontier_focus_start_m=2200.0,
            frontier_focus_end_m=2600.0,
        )
