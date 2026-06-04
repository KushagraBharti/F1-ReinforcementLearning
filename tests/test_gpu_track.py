import numpy as np
import torch

from f1rl.geometry import (
    project_point_to_polyline,
    sample_polyline_at,
    segment_intersects_any,
    wrap_radians,
)
from f1rl.gpu_track import (
    distance_to_next_braking_gate_batch,
    point_is_drivable_batch,
    segments_intersect_any_batch,
    track_errors_batch,
)
from f1rl.gpu_types import gpu_track_from_cpu
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
