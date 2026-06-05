# pyright: reportPrivateImportUsage=false

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from f1rl.config import SimConfig
from f1rl.evolution_search import CONTROLLER_FEATURE_NAMES, CONTROLLER_OUTPUT_COUNT
from f1rl.geometry import (
    project_point_to_polyline,
    sample_polyline_at,
    segment_intersects_any,
    wrap_radians,
)
from f1rl.gpu_batch import GpuControlProgram, GpuMonzaBatch
from f1rl.gpu_fast_warp import warp_status
from f1rl.gpu_fused_warp import (
    open_step_bookkeeping_warp_batch,
    open_step_warp_batch,
    point_is_drivable_warp_batch,
    segments_intersect_any_warp_grid_batch,
    track_errors_warp_local_batch,
)
from f1rl.gpu_scoring import TERMINATION_REASON_TO_ID
from f1rl.gpu_track import (
    distance_to_next_braking_gate_batch,
    lookahead_heading_errors_batch,
    point_is_drivable_batch,
    sample_centerline_at_batch,
    segments_intersect_any_batch,
    segments_intersect_any_grid_batch,
    track_errors_batch,
)
from f1rl.gpu_types import GpuCarBatch, gpu_track_from_cpu
from f1rl.state_snapshot import StateSnapshot
from f1rl.track_model import load_track_spec


def test_gpu_track_projection_matches_cpu_geometry() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    progress_m = np.asarray([0.0, 520.0, 2140.0, 4740.0, 5260.0], dtype=np.float64)
    progress_px = progress_m / track.meters_per_pixel
    points = sample_polyline_at(track.centerline, track.centerline_s, progress_px.astype(np.float32)).astype(np.float64)
    points[:, 1] += np.asarray([0.0, 2.0, -3.0, 4.0, -2.0], dtype=np.float64)

    expected_progress = []
    expected_lateral = []
    expected_heading_error = []
    headings = []
    for point, previous_px in zip(points, progress_px, strict=True):
        raw_px, lateral_px, tangent, _projection = project_point_to_polyline(
            point,
            track.centerline,
            track.centerline_s,
            previous_progress=float(previous_px),
            window=500.0 / track.meters_per_pixel,
        )
        heading = tangent - 0.05
        expected_progress.append(raw_px)
        expected_lateral.append(lateral_px)
        expected_heading_error.append(wrap_radians(tangent - heading))
        headings.append(heading)

    errors = track_errors_batch(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(headings, dtype=torch.float64),
        gpu_track,
        previous_progress_px=torch.tensor(progress_px, dtype=torch.float64),
        window_px=500.0 / track.meters_per_pixel,
    )

    assert np.allclose(errors["raw_progress_px"].numpy(), expected_progress, atol=1e-3)
    assert np.allclose(errors["lateral_error_m"].numpy(), np.asarray(expected_lateral) * track.meters_per_pixel, atol=1e-4)
    assert np.allclose(errors["heading_error_rad"].numpy(), expected_heading_error, atol=1e-6)


def test_gpu_projection_accepts_cached_segment_indices() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    progress_m = np.asarray([0.0, 520.0, 2140.0, 3250.0, 4740.0, 5260.0], dtype=np.float64)
    progress_px = progress_m / track.meters_per_pixel
    points = sample_polyline_at(track.centerline, track.centerline_s, progress_px.astype(np.float32)).astype(np.float64)
    points[:, 0] += np.asarray([0.0, 1.5, -2.0, 2.5, -1.0, 3.0], dtype=np.float64)
    points[:, 1] += np.asarray([0.0, -2.5, 1.0, -1.5, 2.0, -3.0], dtype=np.float64)
    headings = torch.zeros(len(points), dtype=torch.float64)
    previous_progress = torch.tensor(progress_px, dtype=torch.float64)
    cached_segments = torch.searchsorted(gpu_track.centerline_cumdist_px, previous_progress, right=True) - 1
    cached_segments = torch.clamp(cached_segments, 0, gpu_track.centerline_xy.shape[0] - 1)

    from_progress = track_errors_batch(
        torch.tensor(points, dtype=torch.float64),
        headings,
        gpu_track,
        previous_progress_px=previous_progress,
        window_px=500.0 / track.meters_per_pixel,
    )
    from_segment_cache = track_errors_batch(
        torch.tensor(points, dtype=torch.float64),
        headings,
        gpu_track,
        previous_progress_px=previous_progress,
        previous_segment_idx=cached_segments,
        window_px=500.0 / track.meters_per_pixel,
    )

    for key in ("raw_progress_px", "lateral_error_m", "signed_lateral_error_m", "heading_error_rad"):
        assert torch.allclose(from_segment_cache[key], from_progress[key])
    assert torch.equal(from_segment_cache["segment_idx"], from_progress["segment_idx"])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp local projection parity requires CUDA and Warp CUDA support",
)
def test_warp_local_projection_matches_torch_cached_projection() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    progress_m = np.asarray([0.0, 520.0, 2140.0, 3250.0, 4740.0, 5260.0], dtype=np.float32)
    progress_px = progress_m / np.float32(track.meters_per_pixel)
    points = sample_polyline_at(track.centerline, track.centerline_s, progress_px).astype(np.float32)
    points[:, 0] += np.asarray([0.0, 1.5, -2.0, 2.5, -1.0, 3.0], dtype=np.float32)
    points[:, 1] += np.asarray([0.0, -2.5, 1.0, -1.5, 2.0, -3.0], dtype=np.float32)
    points_gpu = torch.tensor(points, device="cuda", dtype=torch.float32)
    headings = torch.linspace(-0.25, 0.35, len(points), device="cuda", dtype=torch.float32)
    previous_progress = torch.tensor(progress_px, device="cuda", dtype=torch.float32)
    cached_segments = torch.searchsorted(gpu_track.centerline_cumdist_px, previous_progress, right=True) - 1
    cached_segments = torch.clamp(cached_segments, 0, gpu_track.centerline_xy.shape[0] - 1)
    window_px = torch.tensor(500.0 / track.meters_per_pixel, device="cuda", dtype=torch.float32)

    expected = track_errors_batch(
        points_gpu,
        headings,
        gpu_track,
        previous_progress_px=previous_progress,
        previous_segment_idx=cached_segments,
        window_px=window_px,
    )
    actual = track_errors_warp_local_batch(
        points_gpu,
        headings,
        gpu_track,
        previous_progress_px=previous_progress,
        previous_segment_idx=cached_segments,
        window_px=window_px,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual["segment_idx"], expected["segment_idx"])
    for key in ("raw_progress_px", "lateral_error_m", "signed_lateral_error_m", "heading_error_rad", "projection_x", "projection_y"):
        assert torch.allclose(actual[key], expected[key], atol=2e-4, rtol=2e-5), key


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp local projection parity requires CUDA and Warp CUDA support",
)
def test_warp_local_projection_matches_torch_randomized_near_centerline() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    rng = np.random.default_rng(202)
    progress_px = rng.uniform(0.0, track.length_px, size=128).astype(np.float32)
    points = sample_polyline_at(track.centerline, track.centerline_s, progress_px).astype(np.float32)
    points += rng.normal(0.0, 4.0, size=points.shape).astype(np.float32)
    headings = rng.uniform(-np.pi, np.pi, size=len(points)).astype(np.float32)
    points_gpu = torch.tensor(points, device="cuda", dtype=torch.float32)
    headings_gpu = torch.tensor(headings, device="cuda", dtype=torch.float32)
    previous_progress = torch.tensor(progress_px, device="cuda", dtype=torch.float32)
    cached_segments = torch.searchsorted(gpu_track.centerline_cumdist_px, previous_progress, right=True) - 1
    cached_segments = torch.clamp(cached_segments, 0, gpu_track.centerline_xy.shape[0] - 1)
    window_px = torch.tensor(500.0 / track.meters_per_pixel, device="cuda", dtype=torch.float32)

    expected = track_errors_batch(
        points_gpu,
        headings_gpu,
        gpu_track,
        previous_progress_px=previous_progress,
        previous_segment_idx=cached_segments,
        window_px=window_px,
    )
    actual = track_errors_warp_local_batch(
        points_gpu,
        headings_gpu,
        gpu_track,
        previous_progress_px=previous_progress,
        previous_segment_idx=cached_segments,
        window_px=window_px,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual["segment_idx"], expected["segment_idx"])
    for key in ("raw_progress_px", "lateral_error_m", "signed_lateral_error_m", "heading_error_rad", "projection_x", "projection_y"):
        assert torch.allclose(actual[key], expected[key], atol=3e-4, rtol=3e-5), key


def test_gpu_track_precomputes_local_projection_and_boundary_grid() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)

    assert gpu_track.local_projection_indices.shape[0] == len(track.centerline) - 1
    assert 1 <= gpu_track.local_projection_indices.shape[1] < len(track.centerline) - 1
    assert gpu_track.boundary_grid_indices.shape[0] == gpu_track.boundary_grid_width * gpu_track.boundary_grid_height
    assert gpu_track.boundary_grid_counts.shape[0] == gpu_track.boundary_grid_indices.shape[0]
    assert int(torch.max(gpu_track.boundary_grid_counts).item()) > 0


def test_gpu_centerline_sampling_supports_vectorized_lookaheads() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    base_px = torch.tensor([0.0, 520.0, 2140.0], dtype=torch.float64) / float(track.meters_per_pixel)
    lookahead_px = torch.tensor([40.0, 90.0, 160.0, 280.0], dtype=torch.float64) / float(track.meters_per_pixel)

    points, tangents = sample_centerline_at_batch(gpu_track, base_px[:, None] + lookahead_px[None, :])
    expected_points = []
    expected_tangents = []
    for lookahead in lookahead_px:
        point, tangent = sample_centerline_at_batch(gpu_track, base_px + lookahead)
        expected_points.append(point)
        expected_tangents.append(tangent)

    assert points.shape == (3, 4, 2)
    assert tangents.shape == (3, 4)
    assert torch.allclose(points, torch.stack(expected_points, dim=1))
    assert torch.allclose(tangents, torch.stack(expected_tangents, dim=1))


def test_gpu_lookahead_heading_errors_accept_cached_tensor() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    progress_m = torch.tensor([0.0, 520.0, 2140.0], dtype=torch.float64)
    heading = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    lookahead = torch.tensor([40.0, 90.0, 160.0, 280.0], dtype=torch.float64)

    from_tuple = lookahead_heading_errors_batch(
        raw_progress_m=progress_m,
        heading_rad=heading,
        lookahead_m=tuple(float(value) for value in lookahead.tolist()),
        track=gpu_track,
    )
    from_tensor = lookahead_heading_errors_batch(
        raw_progress_m=progress_m,
        heading_rad=heading,
        lookahead_m=lookahead,
        track=gpu_track,
    )

    assert from_tensor.shape == (3, 4)
    assert torch.allclose(from_tensor, from_tuple)


def test_gpu_drivable_mask_matches_track_spec() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    centerline_points = sample_polyline_at(
        track.centerline,
        track.centerline_s,
        np.asarray([0.0, 500.0, 2000.0, 4500.0], dtype=np.float32) / track.meters_per_pixel,
    )
    points = np.vstack((centerline_points, np.asarray([[-10.0, -10.0], [99999.0, 99999.0]], dtype=np.float32)))
    expected = [track.point_is_drivable(float(x), float(y)) for x, y in points]

    actual = point_is_drivable_batch(
        torch.tensor(points[:, 0], dtype=torch.float64),
        torch.tensor(points[:, 1], dtype=torch.float64),
        gpu_track,
    )

    assert actual.tolist() == expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp drivable-mask parity requires CUDA and Warp CUDA support",
)
def test_warp_drivable_mask_matches_torch_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    centerline_points = sample_polyline_at(
        track.centerline,
        track.centerline_s,
        np.asarray([0.0, 500.0, 2000.0, 4500.0], dtype=np.float32) / track.meters_per_pixel,
    ).astype(np.float32)
    points = np.vstack(
        (
            centerline_points,
            np.asarray([[-10.1, -10.2], [99999.3, 99999.4], [0.2, 0.3]], dtype=np.float32),
        )
    )
    x = torch.tensor(points[:, 0], device="cuda", dtype=torch.float32)
    y = torch.tensor(points[:, 1], device="cuda", dtype=torch.float32)

    expected = point_is_drivable_batch(x, y, gpu_track)
    actual = point_is_drivable_warp_batch(x, y, gpu_track)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp drivable-mask parity requires CUDA and Warp CUDA support",
)
def test_warp_drivable_mask_matches_torch_randomized_points() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    rng = np.random.default_rng(404)
    height, width = track.drivable_mask.shape
    x_values = rng.uniform(-20.0, float(width + 20), size=512).astype(np.float32)
    y_values = rng.uniform(-20.0, float(height + 20), size=512).astype(np.float32)
    x = torch.tensor(x_values, device="cuda", dtype=torch.float32)
    y = torch.tensor(y_values, device="cuda", dtype=torch.float32)

    expected = point_is_drivable_batch(x, y, gpu_track)
    actual = point_is_drivable_warp_batch(x, y, gpu_track)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_gpu_segment_intersection_matches_cpu_geometry() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    start_x, start_y = [float(value) for value in track.start_pose[:2]]
    movements = np.asarray(
        [
            track.boundary_segments[0],
            [start_x, start_y, start_x + 1.0, start_y],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = [segment_intersects_any(segment, track.boundary_segments) for segment in movements]

    actual = segments_intersect_any_batch(
        torch.tensor(movements, dtype=torch.float64),
        gpu_track.boundary_segments,
        chunk_size=128,
    )

    assert actual.tolist() == expected


def test_gpu_grid_segment_intersection_matches_exact_step_movements() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    rng = np.random.default_rng(123)
    centerline_progress = rng.uniform(0.0, track.length_px, size=64).astype(np.float32)
    points = sample_polyline_at(track.centerline, track.centerline_s, centerline_progress)
    offsets = rng.normal(0.0, 5.0, size=points.shape).astype(np.float32)
    start = points + offsets
    delta = rng.normal(0.0, 1.5, size=points.shape).astype(np.float32)
    end = start + delta
    movements = np.column_stack((start[:, 0], start[:, 1], end[:, 0], end[:, 1])).astype(np.float64)

    exact = segments_intersect_any_batch(
        torch.tensor(movements, dtype=torch.float64),
        gpu_track.boundary_segments,
        chunk_size=128,
    )
    grid = segments_intersect_any_grid_batch(torch.tensor(movements, dtype=torch.float64), gpu_track)

    assert grid.tolist() == exact.tolist()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp grid collision parity requires CUDA and Warp CUDA support",
)
def test_warp_grid_segment_intersection_matches_torch_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    start_x, start_y = [float(value) for value in track.start_pose[:2]]
    movements = np.asarray(
        [
            track.boundary_segments[0],
            [start_x, start_y, start_x + 1.0, start_y],
            [0.0, 0.0, 1.0, 0.0],
            [start_x - 2.0, start_y - 2.0, start_x + 2.0, start_y + 2.0],
        ],
        dtype=np.float32,
    )
    movements_gpu = torch.tensor(movements, device="cuda", dtype=torch.float32)

    expected = segments_intersect_any_grid_batch(movements_gpu, gpu_track)
    actual = segments_intersect_any_warp_grid_batch(movements_gpu, gpu_track)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp grid collision parity requires CUDA and Warp CUDA support",
)
def test_warp_grid_segment_intersection_matches_torch_randomized_step_movements() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    rng = np.random.default_rng(505)
    centerline_progress = rng.uniform(0.0, track.length_px, size=128).astype(np.float32)
    points = sample_polyline_at(track.centerline, track.centerline_s, centerline_progress)
    offsets = rng.normal(0.0, 5.0, size=points.shape).astype(np.float32)
    start = points + offsets
    delta = rng.normal(0.0, 1.5, size=points.shape).astype(np.float32)
    end = start + delta
    movements = np.column_stack((start[:, 0], start[:, 1], end[:, 0], end[:, 1])).astype(np.float32)
    movements_gpu = torch.tensor(movements, device="cuda", dtype=torch.float32)

    expected = segments_intersect_any_grid_batch(movements_gpu, gpu_track)
    actual = segments_intersect_any_warp_grid_batch(movements_gpu, gpu_track)
    exact = segments_intersect_any_batch(movements_gpu, gpu_track.boundary_segments, chunk_size=128)
    torch.cuda.synchronize()

    assert torch.equal(expected, exact)
    assert torch.equal(actual, expected)


def test_gpu_braking_gate_distance_tolerates_gate_roundoff() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cpu"), dtype=torch.float64)
    gates = torch.tensor([520.0, 1950.0, 2620.0, 3250.0, 4560.0, 5260.0], dtype=torch.float64)

    actual = distance_to_next_braking_gate_batch(
        torch.tensor([520.0, 520.0000003646167, 519.9999996, 520.01], dtype=torch.float64),
        gates_m=gates,
        length_m=gpu_track.length_m,
    )

    assert actual[:3].tolist() == [0.0, 0.0, 0.0]
    assert float(actual[3]) > 1400.0


def _clone_car_batch(state: GpuCarBatch) -> GpuCarBatch:
    return GpuCarBatch(**{field.name: getattr(state, field.name).clone() for field in fields(GpuCarBatch)})


def _bookkeeping_state(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    monotonic_progress_m: torch.Tensor,
    raw_progress_px: torch.Tensor,
    elapsed_steps: torch.Tensor,
    next_checkpoint_index: torch.Tensor,
    checkpoints_passed: torch.Tensor,
    no_progress_steps: torch.Tensor,
    lap_index: torch.Tensor,
) -> GpuCarBatch:
    zeros = torch.zeros(batch_size, device=device, dtype=dtype)
    false = torch.zeros(batch_size, device=device, dtype=torch.bool)
    return GpuCarBatch(
        x=zeros.clone(),
        y=zeros.clone(),
        heading_rad=zeros.clone(),
        speed_mps=torch.full((batch_size,), 50.0, device=device, dtype=dtype),
        yaw_rate_rps=zeros.clone(),
        steering=zeros.clone(),
        raw_progress_m=monotonic_progress_m.clone(),
        monotonic_progress_m=monotonic_progress_m.clone(),
        last_raw_progress_px=raw_progress_px.clone(),
        checkpoint_index=torch.zeros(batch_size, device=device, dtype=torch.int64),
        next_checkpoint_index=next_checkpoint_index.clone(),
        checkpoints_passed=checkpoints_passed.clone(),
        missed_checkpoint_count=torch.zeros(batch_size, device=device, dtype=torch.int64),
        lap_index=lap_index.clone(),
        elapsed_steps=elapsed_steps.clone(),
        no_progress_steps=no_progress_steps.clone(),
        alive=torch.ones(batch_size, device=device, dtype=torch.bool),
        terminated=false.clone(),
        truncated=false.clone(),
        termination_reason_id=torch.zeros(batch_size, device=device, dtype=torch.int64),
        valid_lap=torch.ones(batch_size, device=device, dtype=torch.bool),
        finish_crossed=false.clone(),
        completed_lap=false.clone(),
        segment_complete=false.clone(),
        segment_release_observed=false.clone(),
        last_throttle=zeros.clone(),
        last_brake=zeros.clone(),
        last_steer=zeros.clone(),
    )


def _torch_open_bookkeeping_reference(
    state: GpuCarBatch,
    *,
    last_segment_idx: torch.Tensor,
    active: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    old_progress_m: torch.Tensor,
    old_raw_progress_px: torch.Tensor,
    old_lap_index: torch.Tensor,
    raw_progress_px: torch.Tensor,
    segment_idx: torch.Tensor,
    lateral_error_m: torch.Tensor,
    collided: torch.Tensor,
    drivable: torch.Tensor,
    physical_finish: torch.Tensor,
    gpu_track,
    sim_config: SimConfig,
) -> dict[str, torch.Tensor]:
    last_segment_idx.copy_(torch.where(active, segment_idx, last_segment_idx))
    state.last_throttle = torch.where(active, throttle, state.last_throttle)
    state.last_brake = torch.where(active, brake, state.last_brake)
    state.last_steer = torch.where(active, steer, state.last_steer)
    raw_delta_px = raw_progress_px - old_raw_progress_px
    raw_delta_px = torch.where(raw_delta_px < -0.5 * gpu_track.length_px, raw_delta_px + gpu_track.length_px, raw_delta_px)
    raw_delta_px = torch.where(raw_delta_px > 0.5 * gpu_track.length_px, raw_delta_px - gpu_track.length_px, raw_delta_px)
    progress_delta_m = torch.clamp(raw_delta_px * gpu_track.meters_per_pixel, min=0.0)
    progress_delta_m = torch.where(
        progress_delta_m > sim_config.local_projection_window_m,
        torch.zeros_like(progress_delta_m),
        progress_delta_m,
    )
    state.last_raw_progress_px = torch.where(active, raw_progress_px, state.last_raw_progress_px)
    state.raw_progress_m = torch.where(active, raw_progress_px * gpu_track.meters_per_pixel, state.raw_progress_m)
    state.monotonic_progress_m = torch.where(active, old_progress_m + progress_delta_m, state.monotonic_progress_m)

    spacing = gpu_track.checkpoint_spacing_m
    checkpoint_count = max(gpu_track.checkpoint_count, 1)
    skipped = active & (progress_delta_m > spacing * 1.75)
    skipped_count = torch.clamp(torch.floor(progress_delta_m / spacing).to(torch.int64) - 1, min=1)
    state.missed_checkpoint_count += torch.where(
        skipped,
        skipped_count,
        torch.zeros_like(state.missed_checkpoint_count),
    )
    state.valid_lap = torch.where(skipped, torch.zeros_like(state.valid_lap), state.valid_lap)

    next_idx = state.next_checkpoint_index
    threshold = spacing * next_idx.to(dtype=gpu_track.dtype)
    due = active & (next_idx < checkpoint_count) & (state.monotonic_progress_m + 1e-6 >= threshold)
    expected = (old_progress_m <= threshold) & (threshold <= state.monotonic_progress_m + spacing * 0.75)
    lateral_ok = torch.abs(lateral_error_m) <= float(sim_config.checkpoint_lateral_limit_m)
    invalid = due & ~(expected & lateral_ok)
    state.missed_checkpoint_count += invalid.to(dtype=torch.int64)
    state.valid_lap = torch.where(invalid, torch.zeros_like(state.valid_lap), state.valid_lap)
    state.checkpoints_passed = torch.where(due, next_idx, state.checkpoints_passed)
    state.next_checkpoint_index = torch.where(due, next_idx + 1, next_idx)
    state.checkpoint_index = torch.remainder(
        torch.floor(state.monotonic_progress_m / spacing).to(torch.int64),
        checkpoint_count,
    )

    no_progress = active & (progress_delta_m <= 1e-4)
    state.no_progress_steps = torch.where(
        no_progress,
        state.no_progress_steps + 1,
        torch.where(active, torch.zeros_like(state.no_progress_steps), state.no_progress_steps),
    )

    target_lap_progress_m = (old_lap_index.to(dtype=gpu_track.dtype) + 1.0) * gpu_track.length_m
    near_finish = old_progress_m >= target_lap_progress_m - spacing * 2.0
    virtual_finish = (old_progress_m < target_lap_progress_m) & (target_lap_progress_m <= state.monotonic_progress_m)
    crossed_finish = active & near_finish & (physical_finish | virtual_finish)
    state.finish_crossed |= crossed_finish
    telemetry_valid_lap = (
        state.valid_lap
        & (state.missed_checkpoint_count == 0)
        & (state.checkpoints_passed >= max(gpu_track.checkpoint_count - 1, 0))
    )
    lap_complete = (
        active
        & crossed_finish
        & telemetry_valid_lap
        & (state.monotonic_progress_m >= target_lap_progress_m)
        & ~state.terminated
    )
    state.lap_index = torch.where(lap_complete, state.lap_index + 1, state.lap_index)
    state.completed_lap |= lap_complete
    state.truncated |= lap_complete
    state.termination_reason_id = torch.where(
        lap_complete,
        torch.full_like(state.termination_reason_id, TERMINATION_REASON_TO_ID["lap_complete"]),
        state.termination_reason_id,
    )

    collided = active & collided
    off_track = active & ~drivable
    state.terminated |= collided
    state.termination_reason_id = torch.where(
        collided,
        torch.full_like(state.termination_reason_id, TERMINATION_REASON_TO_ID["collision"]),
        state.termination_reason_id,
    )
    state.terminated |= off_track
    state.termination_reason_id = torch.where(
        off_track,
        torch.full_like(state.termination_reason_id, TERMINATION_REASON_TO_ID["off_track"]),
        state.termination_reason_id,
    )
    no_progress_terminated = active & (state.no_progress_steps >= sim_config.no_progress_limit_steps)
    state.terminated |= no_progress_terminated
    state.termination_reason_id = torch.where(
        no_progress_terminated,
        torch.full_like(state.termination_reason_id, TERMINATION_REASON_TO_ID["no_progress"]),
        state.termination_reason_id,
    )
    max_steps = active & (state.elapsed_steps >= sim_config.max_steps)
    state.truncated |= max_steps
    state.termination_reason_id = torch.where(
        max_steps,
        torch.full_like(state.termination_reason_id, TERMINATION_REASON_TO_ID["max_steps"]),
        state.termination_reason_id,
    )
    state.alive = torch.where(state.terminated, torch.zeros_like(state.alive), state.alive)
    return {
        "progress_delta_m": progress_delta_m,
        "collided": collided,
        "off_track": off_track,
        "telemetry_valid_lap": telemetry_valid_lap,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp open-step bookkeeping parity requires CUDA and Warp CUDA support",
)
def test_warp_open_step_bookkeeping_matches_torch_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=8, no_progress_limit_steps=2)
    spacing = float(gpu_track.checkpoint_spacing_m.detach().cpu().item())
    meters_per_pixel = float(gpu_track.meters_per_pixel.detach().cpu().item())
    length_m = float(gpu_track.length_m.detach().cpu().item())
    old_progress_values = torch.tensor(
        [
            100.0,
            200.0,
            300.0,
            400.0,
            spacing * 2.0,
            length_m - 1.0,
            600.0,
            700.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    progress_deltas = torch.tensor(
        [
            5.0,
            0.0,
            5.0,
            5.0,
            spacing * 2.1,
            2.0,
            5.0,
            5.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    raw_progress_values = torch.remainder(old_progress_values + progress_deltas, gpu_track.length_m) / meters_per_pixel
    old_raw_progress = torch.remainder(old_progress_values, gpu_track.length_m) / meters_per_pixel
    batch_size = int(old_progress_values.shape[0])
    next_checkpoint_index = torch.full((batch_size,), gpu_track.checkpoint_count + 2, device="cuda", dtype=torch.int64)
    next_checkpoint_index[4] = 2
    checkpoints_passed = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
    checkpoints_passed[5] = max(gpu_track.checkpoint_count - 1, 0)
    elapsed_steps = torch.ones(batch_size, device="cuda", dtype=torch.int64)
    elapsed_steps[6] = sim_config.max_steps
    no_progress_steps = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
    no_progress_steps[1] = sim_config.no_progress_limit_steps - 1
    lap_index = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
    base_state = _bookkeeping_state(
        batch_size=batch_size,
        device=torch.device("cuda"),
        dtype=torch.float32,
        monotonic_progress_m=old_progress_values,
        raw_progress_px=old_raw_progress,
        elapsed_steps=elapsed_steps,
        next_checkpoint_index=next_checkpoint_index,
        checkpoints_passed=checkpoints_passed,
        no_progress_steps=no_progress_steps,
        lap_index=lap_index,
    )
    torch_state = _clone_car_batch(base_state)
    warp_state = _clone_car_batch(base_state)
    last_segment_idx_torch = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
    last_segment_idx_warp = last_segment_idx_torch.clone()
    active = torch.tensor([True, True, True, True, True, True, True, False], device="cuda", dtype=torch.bool)
    throttle = torch.linspace(0.1, 0.8, batch_size, device="cuda", dtype=torch.float32)
    brake = torch.linspace(0.0, 0.3, batch_size, device="cuda", dtype=torch.float32)
    steer = torch.linspace(-0.2, 0.2, batch_size, device="cuda", dtype=torch.float32)
    segment_idx = torch.arange(batch_size, device="cuda", dtype=torch.int64) + 3
    lateral_error = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
    lateral_error[4] = sim_config.checkpoint_lateral_limit_m + 1.0
    heading_error = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
    collided = torch.tensor([False, False, True, False, False, False, False, False], device="cuda", dtype=torch.bool)
    drivable = torch.tensor([True, True, False, False, True, True, True, True], device="cuda", dtype=torch.bool)
    physical_finish = torch.tensor([False, False, False, False, False, True, False, False], device="cuda", dtype=torch.bool)

    expected = _torch_open_bookkeeping_reference(
        torch_state,
        last_segment_idx=last_segment_idx_torch,
        active=active,
        throttle=throttle,
        brake=brake,
        steer=steer,
        old_progress_m=old_progress_values,
        old_raw_progress_px=old_raw_progress,
        old_lap_index=lap_index,
        raw_progress_px=raw_progress_values,
        segment_idx=segment_idx,
        lateral_error_m=lateral_error,
        collided=collided,
        drivable=drivable,
        physical_finish=physical_finish,
        gpu_track=gpu_track,
        sim_config=sim_config,
    )
    actual = open_step_bookkeeping_warp_batch(
        warp_state,
        last_segment_idx=last_segment_idx_warp,
        active=active,
        throttle=throttle,
        brake=brake,
        steer=steer,
        old_progress_m=old_progress_values,
        old_raw_progress_px=old_raw_progress,
        old_lap_index=lap_index,
        raw_progress_px=raw_progress_values,
        segment_idx=segment_idx,
        lateral_error_m=lateral_error,
        heading_error_rad=heading_error,
        collided=collided,
        drivable=drivable,
        physical_finish=physical_finish,
        track=gpu_track,
        local_projection_window_m=sim_config.local_projection_window_m,
        checkpoint_lateral_limit_m=sim_config.checkpoint_lateral_limit_m,
        no_progress_limit_steps=sim_config.no_progress_limit_steps,
        max_steps=sim_config.max_steps,
    )
    torch.cuda.synchronize()

    for key, expected_value in expected.items():
        assert torch.equal(actual[key], expected_value), key
    assert torch.equal(last_segment_idx_warp, last_segment_idx_torch)
    for field in fields(GpuCarBatch):
        actual_value = getattr(warp_state, field.name)
        expected_value = getattr(torch_state, field.name)
        if actual_value.dtype == torch.bool or not actual_value.is_floating_point():
            assert torch.equal(actual_value, expected_value), field.name
        else:
            assert torch.allclose(actual_value, expected_value), field.name


def _assert_car_batches_match(actual: GpuCarBatch, expected: GpuCarBatch, *, atol: float = 2e-2) -> None:
    for field in fields(GpuCarBatch):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if actual_value.dtype == torch.bool or not actual_value.is_floating_point():
            assert torch.equal(actual_value, expected_value), field.name
        else:
            assert torch.allclose(actual_value, expected_value, rtol=1e-5, atol=atol), field.name


def _snapshots_for_progresses(track, progresses_m: list[float]) -> list[StateSnapshot]:
    progress_px = np.asarray(progresses_m, dtype=np.float32) / np.float32(track.meters_per_pixel)
    points = sample_polyline_at(track.centerline, track.centerline_s, progress_px)
    ahead = sample_polyline_at(track.centerline, track.centerline_s, progress_px + np.float32(2.0))
    headings = np.arctan2(ahead[:, 1] - points[:, 1], ahead[:, 0] - points[:, 0])
    checkpoint_count = len(track.checkpoints)
    spacing_m = track.length_m / max(checkpoint_count, 1)
    snapshots: list[StateSnapshot] = []
    for index, progress_m in enumerate(progresses_m):
        checkpoint_index = int(progress_m // spacing_m)
        snapshots.append(
            StateSnapshot(
                id=f"warp-step:{index}",
                source="test",
                source_file=None,
                step_index=0,
                sim_time_s=0.0,
                x=float(points[index, 0]),
                y=float(points[index, 1]),
                heading_rad=float(headings[index]),
                speed_mps=float(45.0 + index * 3.0),
                yaw_rate_rps=0.0,
                steering_rad=0.0,
                raw_progress_m=float(progress_m),
                monotonic_progress_m=float(progress_m),
                checkpoint_index=checkpoint_index,
                next_checkpoint_index=checkpoint_index + 1,
                checkpoints_passed=checkpoint_index,
                missed_checkpoint_count=0,
                lap_index=0,
                valid_lap=True,
                finish_crossed=False,
                completed_lap=False,
                segment_complete=False,
                last_throttle=0.0,
                last_brake=0.0,
                last_steer=0.0,
                last_action_id=-1,
            )
        )
    return snapshots


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp target-step parity requires CUDA and Warp CUDA support",
)
@pytest.mark.parametrize(
    ("case_name", "gate_overrides", "throttle_value", "expected_reason"),
    [
        ("segment_complete", {}, 0.0, "segment_complete"),
        ("max_speed_gate", {"target_max_speed_kph": 80.0}, 0.0, "segment_speed_gate_failed"),
        ("min_speed_gate", {"target_min_speed_kph": 500.0}, 0.0, "segment_min_speed_gate_failed"),
        (
            "release_gate",
            {
                "segment_require_release": True,
                "segment_release_min_speed_kph": 0.0,
                "segment_release_max_speed_kph": 500.0,
                "segment_release_max_brake": 0.1,
                "segment_release_max_throttle": 0.1,
            },
            0.8,
            "segment_release_gate_failed",
        ),
    ],
)
def test_warp_target_step_matches_gpu_batch_target_reference(
    case_name: str,
    gate_overrides: dict[str, float | bool],
    throttle_value: float,
    expected_reason: str,
) -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=24, no_progress_limit_steps=8)
    snapshots = [
        replace(snapshot, speed_mps=120.0)
        for snapshot in _snapshots_for_progresses(track, [500.0, 500.0, 500.0, 500.0])
    ]
    torch_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
    )
    warp_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
    )
    target_progress_m = 501.0
    torch_batch.reset(snapshots, target_progress_m=target_progress_m, terminate_at_target=True)
    warp_batch.reset(snapshots, target_progress_m=target_progress_m, terminate_at_target=True)
    assert torch_batch.state is not None
    assert warp_batch.state is not None
    assert torch_batch._last_segment_idx is not None
    assert warp_batch._last_segment_idx is not None
    assert warp_batch.segment_target_progress_m is not None
    gate_values = {
        "target_progress_m": target_progress_m,
        "terminate_at_target_progress": True,
        "segment_fail_on_speed_gate_miss": True,
        "segment_require_release": False,
    }
    gate_values.update(gate_overrides)
    gates = SimpleNamespace(**gate_values)
    throttle = torch.full((len(snapshots),), throttle_value, device="cuda", dtype=torch.float32)
    brake = torch.zeros_like(throttle)
    steer = torch.zeros_like(throttle)

    expected_errors, expected_collided, expected_off_track, expected_valid_lap = torch_batch._step(
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=torch_batch._active(),
        gates=gates,
    )
    actual_errors, actual_collided, actual_off_track, actual_valid_lap = open_step_warp_batch(
        warp_batch.state,
        last_segment_idx=warp_batch._last_segment_idx,
        active=warp_batch._active(),
        throttle=throttle,
        brake=brake,
        steer=steer,
        params=warp_batch.params,
        track=gpu_track,
        sim_config=sim_config,
        segment_target_progress_m=warp_batch.segment_target_progress_m,
        gates=gates,
        collision_check=True,
        collision_mode="exact_grid",
    )
    torch.cuda.synchronize()

    assert case_name
    expected_reason_id = TERMINATION_REASON_TO_ID[expected_reason]
    assert torch.equal(actual_collided, expected_collided)
    assert torch.equal(actual_off_track, expected_off_track)
    assert torch.equal(actual_valid_lap, expected_valid_lap)
    assert torch.equal(warp_batch._last_segment_idx, torch_batch._last_segment_idx)
    assert torch.all(warp_batch.state.truncated)
    assert torch.all(warp_batch.state.termination_reason_id == expected_reason_id)
    for key in ("raw_progress_px", "lateral_error_m", "signed_lateral_error_m", "heading_error_rad", "progress_delta_m"):
        assert torch.allclose(actual_errors[key], expected_errors[key], rtol=1e-5, atol=2e-2), key
    assert torch.equal(actual_errors["segment_idx"], expected_errors["segment_idx"])
    assert torch.equal(actual_errors["telemetry_valid_lap"], expected_errors["telemetry_valid_lap"])
    _assert_car_batches_match(warp_batch.state, torch_batch.state)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp open-step parity requires CUDA and Warp CUDA support",
)
def test_warp_open_step_matches_gpu_batch_step_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=24, no_progress_limit_steps=8)
    snapshots = _snapshots_for_progresses(track, [50.0, 520.0, 1210.0, 2450.0, 4200.0, 5200.0])
    torch_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
    )
    warp_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
    )
    torch_batch.reset(snapshots, target_progress_m=1500.0, terminate_at_target=False)
    warp_batch.reset(snapshots, target_progress_m=1500.0, terminate_at_target=False)
    assert torch_batch.state is not None
    assert warp_batch.state is not None
    assert torch_batch._last_segment_idx is not None
    assert warp_batch._last_segment_idx is not None
    throttle = torch.tensor([0.6, 0.8, 0.2, 0.4, 0.9, 0.7], device="cuda", dtype=torch.float32)
    brake = torch.tensor([0.0, 0.0, 0.5, 0.1, 0.0, 0.0], device="cuda", dtype=torch.float32)
    steer = torch.tensor([0.0, -0.15, 0.25, -0.2, 0.18, -0.1], device="cuda", dtype=torch.float32)
    gates = SimpleNamespace(segment_require_release=False)

    expected_errors, expected_collided, expected_off_track, expected_valid_lap = torch_batch._step(
        throttle=throttle,
        brake=brake,
        steer=steer,
        active=torch_batch._active(),
        gates=gates,
    )
    actual_errors, actual_collided, actual_off_track, actual_valid_lap = open_step_warp_batch(
        warp_batch.state,
        last_segment_idx=warp_batch._last_segment_idx,
        active=warp_batch._active(),
        throttle=throttle,
        brake=brake,
        steer=steer,
        params=warp_batch.params,
        track=gpu_track,
        sim_config=sim_config,
        collision_check=True,
        collision_mode="exact_grid",
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_collided, expected_collided)
    assert torch.equal(actual_off_track, expected_off_track)
    assert torch.equal(actual_valid_lap, expected_valid_lap)
    assert torch.equal(warp_batch._last_segment_idx, torch_batch._last_segment_idx)
    for key in ("raw_progress_px", "lateral_error_m", "signed_lateral_error_m", "heading_error_rad", "progress_delta_m"):
        assert torch.allclose(actual_errors[key], expected_errors[key], rtol=1e-5, atol=2e-2), key
    assert torch.equal(actual_errors["segment_idx"], expected_errors["segment_idx"])
    assert torch.equal(actual_errors["telemetry_valid_lap"], expected_errors["telemetry_valid_lap"])
    _assert_car_batches_match(warp_batch.state, torch_batch.state)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp open rollout parity requires CUDA and Warp CUDA support",
)
def test_warp_open_rollout_matches_gpu_batch_rollout_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=6, no_progress_limit_steps=8)
    snapshots = _snapshots_for_progresses(track, [50.0, 520.0, 1210.0, 2450.0, 4200.0, 5200.0])
    torch_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    warp_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    torch_batch.reset(snapshots, target_progress_m=1500.0, terminate_at_target=False)
    warp_batch.reset(snapshots, target_progress_m=1500.0, terminate_at_target=False)
    batch_size = len(snapshots)
    action_controls = torch.tensor(
        [
            [0.65, 0.00, 0.00],
            [0.30, 0.45, 0.18],
            [0.80, 0.00, -0.12],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    phase_action_ids = torch.tensor(
        [
            [0, 2],
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 0],
            [0, 1],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    phase_thresholds = torch.full((batch_size, 2), 3, device="cuda", dtype=torch.int64)
    phase_thresholds[:, 0] = 0
    program = GpuControlProgram(
        kind="phase",
        action_names=("power", "brake_turn", "turn"),
        action_controls=action_controls,
        phase_action_ids=phase_action_ids,
        phase_thresholds=phase_thresholds,
    )
    gates = SimpleNamespace(
        target_progress_m=1500.0,
        terminate_at_target_progress=False,
        segment_require_release=False,
    )
    profiles = ("max_progress", "early_pace", "clean_exit", "farthest_distance")

    expected = torch_batch.rollout(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    actual = warp_batch.rollout_warp_open(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    torch.cuda.synchronize()

    assert actual.kernel_backend == "warp_open_step"
    assert actual.steps_executed == expected.steps_executed
    assert actual.sim_steps == expected.sim_steps
    assert torch.equal(actual.final_action_id, expected.final_action_id)
    for profile in profiles:
        assert torch.allclose(actual.profile_scores[profile], expected.profile_scores[profile], rtol=1e-4, atol=128.0), profile
    assert warp_batch.state is not None
    assert torch_batch.state is not None
    _assert_car_batches_match(warp_batch.state, torch_batch.state, atol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Persistent Warp controller rollout parity requires CUDA and Warp CUDA support",
)
def test_warp_persistent_controller_rollout_matches_gpu_batch_rollout_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=10, no_progress_limit_steps=12)
    snapshots = _snapshots_for_progresses(track, [500.0, 520.0, 545.0, 575.0])
    torch_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    warp_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    torch_batch.reset(snapshots, target_progress_m=900.0, terminate_at_target=False)
    warp_batch.reset(snapshots, target_progress_m=900.0, terminate_at_target=False)
    generator = torch.Generator(device="cuda").manual_seed(129)
    weights = torch.randn(
        (len(snapshots), CONTROLLER_OUTPUT_COUNT, len(CONTROLLER_FEATURE_NAMES)),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ) * 0.35
    program = GpuControlProgram(kind="controller", controller_weights=weights)
    gates = SimpleNamespace(
        target_progress_m=900.0,
        terminate_at_target_progress=False,
        segment_require_release=False,
    )
    profiles = ("max_progress", "early_pace", "clean_exit", "farthest_distance")

    expected = torch_batch.rollout(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    actual = warp_batch.rollout_warp_open(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    torch.cuda.synchronize()

    assert actual.kernel_backend == "warp_persistent_controller_open"
    assert actual.steps_executed == expected.steps_executed
    assert actual.sim_steps == expected.sim_steps
    assert torch.equal(actual.final_action_id, expected.final_action_id)
    for profile in profiles:
        assert torch.allclose(actual.profile_scores[profile], expected.profile_scores[profile], rtol=1e-4, atol=256.0), profile
    assert warp_batch.state is not None
    assert torch_batch.state is not None
    _assert_car_batches_match(warp_batch.state, torch_batch.state, atol=7.5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp progress-phase rollout parity requires CUDA and Warp CUDA support",
)
def test_warp_progress_phase_rollout_matches_gpu_batch_rollout_reference() -> None:
    track = load_track_spec()
    gpu_track = gpu_track_from_cpu(track, device=torch.device("cuda"), dtype=torch.float32)
    sim_config = SimConfig(max_steps=8, no_progress_limit_steps=8)
    snapshots = _snapshots_for_progresses(track, [500.0, 520.0, 545.0, 575.0])
    torch_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    warp_batch = GpuMonzaBatch(
        track=gpu_track,
        sim_config=sim_config,
        feature_names=(),
        collision_mode="exact_grid",
        disable_early_stop=True,
    )
    torch_batch.reset(snapshots, target_progress_m=900.0, terminate_at_target=False)
    warp_batch.reset(snapshots, target_progress_m=900.0, terminate_at_target=False)
    action_controls = torch.tensor(
        [
            [0.80, 0.00, 0.00],
            [0.10, 0.55, -0.20],
            [0.35, 0.00, 0.18],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    phase_action_ids = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 1],
            [2, 0, 1],
            [1, 2, 0],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    phase_thresholds = torch.tensor(
        [
            [2.0, 5.0, 9.0],
            [1.0, 3.0, 6.0],
            [0.5, 4.0, 8.0],
            [2.5, 5.5, 10.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    program = GpuControlProgram(
        kind="progress_phase",
        action_names=("power", "brake_turn", "turn"),
        action_controls=action_controls,
        phase_action_ids=phase_action_ids,
        phase_thresholds=phase_thresholds,
    )
    gates = SimpleNamespace(
        target_progress_m=900.0,
        terminate_at_target_progress=False,
        segment_require_release=False,
    )
    profiles = ("max_progress", "early_pace", "clean_exit", "farthest_distance")

    expected = torch_batch.rollout(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    actual = warp_batch.rollout_warp_open(
        control_program=program,
        gates=gates,
        scoring_profiles=profiles,
        frontier_focus_start_m=2200.0,
        frontier_focus_end_m=2600.0,
    )
    torch.cuda.synchronize()

    assert actual.kernel_backend == "warp_open_step"
    assert actual.steps_executed == expected.steps_executed
    assert actual.sim_steps == expected.sim_steps
    assert torch.equal(actual.final_action_id, expected.final_action_id)
    for profile in profiles:
        assert torch.allclose(actual.profile_scores[profile], expected.profile_scores[profile], rtol=1e-4, atol=128.0), profile
    assert warp_batch.state is not None
    assert torch_batch.state is not None
    _assert_car_batches_match(warp_batch.state, torch_batch.state, atol=5e-2)
