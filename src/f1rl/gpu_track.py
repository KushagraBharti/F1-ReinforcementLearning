# pyright: reportPrivateImportUsage=false
"""Batched PyTorch track geometry helpers."""

from __future__ import annotations

import math

import torch

from f1rl.gpu_types import GpuTrackTensors


def wrap_radians_batch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def project_to_centerline_batch(
    points_xy: torch.Tensor,
    track: GpuTrackTensors,
    *,
    previous_progress_px: torch.Tensor | None = None,
    previous_segment_idx: torch.Tensor | None = None,
    window_px: float | torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project batched points onto the closed Monza centerline."""

    if previous_progress_px is not None and window_px is not None:
        prev = torch.remainder(previous_progress_px, track.length_px)
        if previous_segment_idx is None:
            base_idx = torch.searchsorted(track.centerline_cumdist_px, prev, right=True) - 1
        else:
            base_idx = previous_segment_idx.to(device=points_xy.device, dtype=torch.int64)
        base_idx = torch.clamp(base_idx, 0, track.centerline_xy.shape[0] - 1)
        candidate_idx = track.local_projection_indices[base_idx]
        starts = track.centerline_xy[candidate_idx]
        vec = track.centerline_segment_vec[candidate_idx]
        seg_len2 = track.centerline_segment_len2[candidate_idx]
        seg_len = track.centerline_segment_len[candidate_idx]
        cumdist = track.centerline_cumdist_px[candidate_idx]
        mid = track.centerline_segment_mid_px[candidate_idx]
        rel = points_xy[:, None, :] - starts
        t = torch.sum(rel * vec, dim=-1) / seg_len2
        t = torch.clamp(t, 0.0, 1.0)
        projections = starts + t[..., None] * vec
        deltas = points_xy[:, None, :] - projections
        distances = torch.sqrt(torch.sum(deltas * deltas, dim=-1))
        progress = cumdist + seg_len * t
        scores = distances.clone()
        valid = seg_len2 > 1e-9
        mask = valid
        window = (
            torch.as_tensor(window_px, device=points_xy.device, dtype=points_xy.dtype)
            if not isinstance(window_px, torch.Tensor)
            else window_px.to(device=points_xy.device, dtype=points_xy.dtype)
        )
        wrapped_delta = torch.abs(
            torch.remainder(mid - prev[:, None] + track.length_px * 0.5, track.length_px) - track.length_px * 0.5
        )
        local_mask = mask & (wrapped_delta <= window)
        has_local = torch.any(local_mask, dim=1, keepdim=True)
        mask = torch.where(has_local, local_mask, mask)
        signed_delta = torch.remainder(progress - prev[:, None] + track.length_px * 0.5, track.length_px) - track.length_px * 0.5
        scores = scores + torch.clamp(-signed_delta, min=0.0) * 0.05
        inf = torch.full_like(scores, torch.inf)
        masked_scores = torch.where(mask, scores, inf)
        local_segment_idx = torch.argmin(masked_scores, dim=1)
        row_idx = row_index if row_index is not None else torch.arange(points_xy.shape[0], device=points_xy.device)
        segment_idx = candidate_idx[row_idx, local_segment_idx]
        raw_px = progress[row_idx, local_segment_idx]
        raw_px = torch.remainder(raw_px, track.length_px)
        lateral_px = distances[row_idx, local_segment_idx]
        projection = projections[row_idx, local_segment_idx]
        seg_vec = track.centerline_segment_vec[segment_idx]
        tangent = torch.atan2(-seg_vec[:, 1], seg_vec[:, 0])
        return raw_px, lateral_px, tangent, projection, segment_idx

    starts = track.centerline_xy
    vec = track.centerline_segment_vec
    rel = points_xy[:, None, :] - starts[None, :, :]
    t = torch.sum(rel * vec[None, :, :], dim=-1) / track.centerline_segment_len2[None, :]
    t = torch.clamp(t, 0.0, 1.0)
    projections = starts[None, :, :] + t[..., None] * vec[None, :, :]
    deltas = points_xy[:, None, :] - projections
    distances = torch.sqrt(torch.sum(deltas * deltas, dim=-1))
    progress = track.centerline_cumdist_px[None, :] + track.centerline_segment_len[None, :] * t
    scores = distances.clone()
    valid = track.centerline_segment_len2 > 1e-9
    mask = valid[None, :].expand_as(scores)

    if previous_progress_px is not None and window_px is not None:
        prev = torch.remainder(previous_progress_px, track.length_px)
        window = (
            torch.as_tensor(window_px, device=points_xy.device, dtype=points_xy.dtype)
            if not isinstance(window_px, torch.Tensor)
            else window_px.to(device=points_xy.device, dtype=points_xy.dtype)
        )
        wrapped_delta = torch.abs(
            torch.remainder(track.centerline_segment_mid_px[None, :] - prev[:, None] + track.length_px * 0.5, track.length_px)
            - track.length_px * 0.5
        )
        local_mask = mask & (wrapped_delta <= window)
        has_local = torch.any(local_mask, dim=1, keepdim=True)
        mask = torch.where(has_local, local_mask, mask)
        signed_delta = torch.remainder(progress - prev[:, None] + track.length_px * 0.5, track.length_px) - track.length_px * 0.5
        scores = scores + torch.clamp(-signed_delta, min=0.0) * 0.05

    inf = torch.full_like(scores, torch.inf)
    masked_scores = torch.where(mask, scores, inf)
    segment_idx = torch.argmin(masked_scores, dim=1)
    gather_idx = segment_idx[:, None]
    raw_px = torch.gather(progress, 1, gather_idx).squeeze(1)
    raw_px = torch.remainder(raw_px, track.length_px)
    lateral_px = torch.gather(distances, 1, gather_idx).squeeze(1)
    row_idx = row_index if row_index is not None else torch.arange(points_xy.shape[0], device=points_xy.device)
    projection = projections[row_idx, segment_idx]
    seg_vec = track.centerline_segment_vec[segment_idx]
    tangent = torch.atan2(-seg_vec[:, 1], seg_vec[:, 0])
    return raw_px, lateral_px, tangent, projection, segment_idx


def signed_lateral_error_batch(
    points_xy: torch.Tensor,
    projection_xy: torch.Tensor,
    tangent_heading_rad: torch.Tensor,
    lateral_px: torch.Tensor,
) -> torch.Tensor:
    offset = points_xy - projection_xy
    tangent = torch.stack((torch.cos(tangent_heading_rad), -torch.sin(tangent_heading_rad)), dim=1)
    cross = tangent[:, 0] * offset[:, 1] - tangent[:, 1] * offset[:, 0]
    signed = torch.sign(cross) * lateral_px
    return torch.where((lateral_px <= 1e-6) | (torch.abs(cross) <= 1e-9), torch.zeros_like(signed), signed)


def track_errors_batch(
    points_xy: torch.Tensor,
    heading_rad: torch.Tensor,
    track: GpuTrackTensors,
    *,
    previous_progress_px: torch.Tensor | None = None,
    previous_segment_idx: torch.Tensor | None = None,
    window_px: float | torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    raw_px, lateral_px, tangent, projection, segment_idx = project_to_centerline_batch(
        points_xy,
        track,
        previous_progress_px=previous_progress_px,
        previous_segment_idx=previous_segment_idx,
        window_px=window_px,
        row_index=row_index,
    )
    signed_lateral_px = signed_lateral_error_batch(points_xy, projection, tangent, lateral_px)
    heading_error = wrap_radians_batch(tangent - heading_rad)
    return {
        "raw_progress_px": raw_px,
        "lateral_error_m": lateral_px * track.meters_per_pixel,
        "signed_lateral_error_m": signed_lateral_px * track.meters_per_pixel,
        "heading_error_rad": heading_error,
        "segment_idx": segment_idx,
        "projection_x": projection[:, 0],
        "projection_y": projection[:, 1],
    }


def sample_centerline_at_batch(track: GpuTrackTensors, distances_px: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wrapped = torch.remainder(distances_px, track.length_px)
    indices = torch.searchsorted(track.centerline_cumdist_px, wrapped, right=True) - 1
    indices = torch.clamp(indices, 0, track.centerline_xy.shape[0] - 1)
    start = track.centerline_xy[indices]
    vec = track.centerline_segment_vec[indices]
    seg_len = torch.clamp(track.centerline_segment_len[indices], min=1e-6)
    t = ((wrapped - track.centerline_cumdist_px[indices]) / seg_len).unsqueeze(-1)
    point = start + vec * t
    tangent = torch.atan2(-vec[..., 1], vec[..., 0])
    return point, tangent


def lookahead_heading_errors_batch(
    *,
    raw_progress_m: torch.Tensor,
    heading_rad: torch.Tensor,
    lookahead_m: tuple[float, ...] | torch.Tensor,
    track: GpuTrackTensors,
) -> torch.Tensor:
    if isinstance(lookahead_m, torch.Tensor):
        lookahead = lookahead_m.to(device=raw_progress_m.device, dtype=raw_progress_m.dtype)
    else:
        lookahead = torch.tensor(
            tuple(float(value) for value in lookahead_m),
            device=raw_progress_m.device,
            dtype=raw_progress_m.dtype,
        )
    if lookahead.numel() == 0:
        return torch.empty((raw_progress_m.shape[0], 0), device=raw_progress_m.device, dtype=raw_progress_m.dtype)
    current_px = raw_progress_m / torch.clamp(track.meters_per_pixel, min=1e-6)
    flat_lookahead = lookahead.reshape(1, -1)
    target_px = current_px[:, None] + flat_lookahead / torch.clamp(track.meters_per_pixel, min=1e-6)
    _, tangent = sample_centerline_at_batch(track, target_px)
    return wrap_radians_batch(tangent - heading_rad[:, None]) / math.pi


def distance_to_next_braking_gate_batch(
    progress_m: torch.Tensor,
    *,
    gates_m: torch.Tensor,
    length_m: torch.Tensor,
    epsilon_m: float = 1e-3,
) -> torch.Tensor:
    lap_progress = torch.remainder(progress_m, length_m)
    gate_delta = gates_m[None, :] - lap_progress[:, None]
    distances = torch.where(
        gate_delta >= 0.0,
        gate_delta,
        length_m - lap_progress[:, None] + gates_m[None, :],
    )
    distances = torch.where(torch.abs(gate_delta) <= max(0.0, float(epsilon_m)), torch.zeros_like(distances), distances)
    return torch.min(distances, dim=1).values


def point_is_drivable_batch(x: torch.Tensor, y: torch.Tensor, track: GpuTrackTensors) -> torch.Tensor:
    xi = torch.round(x).to(torch.int64)
    yi = torch.round(y).to(torch.int64)
    height = int(track.drivable_mask.shape[0])
    width = int(track.drivable_mask.shape[1])
    in_bounds = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    safe_x = torch.clamp(xi, 0, width - 1)
    safe_y = torch.clamp(yi, 0, height - 1)
    return in_bounds & track.drivable_mask[safe_y, safe_x]


def segments_intersect_any_batch(
    movements: torch.Tensor,
    segments: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    if segments.numel() == 0:
        return torch.zeros(movements.shape[0], device=movements.device, dtype=torch.bool)
    x1 = movements[:, 0]
    y1 = movements[:, 1]
    x2 = movements[:, 2]
    y2 = movements[:, 3]
    dx12 = x2 - x1
    dy12 = y2 - y1
    hits = torch.zeros(movements.shape[0], device=movements.device, dtype=torch.bool)
    for start in range(0, int(segments.shape[0]), chunk_size):
        chunk = segments[start : start + chunk_size]
        x3 = chunk[:, 0]
        y3 = chunk[:, 1]
        x4 = chunk[:, 2]
        y4 = chunk[:, 3]
        dx34 = x4 - x3
        dy34 = y4 - y3
        denom = dy34[None, :] * dx12[:, None] - dx34[None, :] * dy12[:, None]
        valid = torch.abs(denom) >= 1e-9
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        s = (dx34[None, :] * (y1[:, None] - y3[None, :]) - dy34[None, :] * (x1[:, None] - x3[None, :])) / safe_denom
        t = (dx12[:, None] * (y1[:, None] - y3[None, :]) - dy12[:, None] * (x1[:, None] - x3[None, :])) / safe_denom
        intersects = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
        hits |= torch.any(intersects, dim=1)
    return hits


def segments_intersect_any_grid_batch(
    movements: torch.Tensor,
    track: GpuTrackTensors,
    *,
    query_span: int = 3,
) -> torch.Tensor:
    """Exact movement/boundary intersection using the track's boundary-cell grid."""

    if track.boundary_segments.numel() == 0:
        return torch.zeros(movements.shape[0], device=movements.device, dtype=torch.bool)
    if track.boundary_grid_indices.numel() == 0:
        return segments_intersect_any_batch(movements, track.boundary_segments)
    span = max(1, int(query_span))
    x1 = movements[:, 0]
    y1 = movements[:, 1]
    x2 = movements[:, 2]
    y2 = movements[:, 3]
    cell_size = torch.clamp(track.boundary_grid_cell_size_px, min=1.0)
    min_cell_x = torch.floor(torch.minimum(x1, x2) / cell_size).to(torch.int64)
    min_cell_y = torch.floor(torch.minimum(y1, y2) / cell_size).to(torch.int64)
    max_cell_x = torch.floor(torch.maximum(x1, x2) / cell_size).to(torch.int64)
    max_cell_y = torch.floor(torch.maximum(y1, y2) / cell_size).to(torch.int64)
    min_cell_x = torch.clamp(min_cell_x, 0, track.boundary_grid_width - 1)
    max_cell_x = torch.clamp(max_cell_x, 0, track.boundary_grid_width - 1)
    min_cell_y = torch.clamp(min_cell_y, 0, track.boundary_grid_height - 1)
    max_cell_y = torch.clamp(max_cell_y, 0, track.boundary_grid_height - 1)
    offsets = torch.arange(span, device=movements.device, dtype=torch.int64)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    flat_offset_x = offset_x.reshape(1, -1)
    flat_offset_y = offset_y.reshape(1, -1)
    cell_x = min_cell_x[:, None] + flat_offset_x
    cell_y = min_cell_y[:, None] + flat_offset_y
    valid_cell = (cell_x <= max_cell_x[:, None]) & (cell_y <= max_cell_y[:, None])
    cell_x = torch.clamp(cell_x, 0, track.boundary_grid_width - 1)
    cell_y = torch.clamp(cell_y, 0, track.boundary_grid_height - 1)
    cell_ids = cell_y * track.boundary_grid_width + cell_x
    candidate_ids = track.boundary_grid_indices[cell_ids]
    valid_candidates = (candidate_ids >= 0) & valid_cell[:, :, None]
    safe_ids = torch.clamp(candidate_ids, min=0)
    candidate_segments = track.boundary_segments[safe_ids].reshape(movements.shape[0], -1, 4)
    valid_flat = valid_candidates.reshape(movements.shape[0], -1)

    dx12 = x2 - x1
    dy12 = y2 - y1
    x3 = candidate_segments[:, :, 0]
    y3 = candidate_segments[:, :, 1]
    x4 = candidate_segments[:, :, 2]
    y4 = candidate_segments[:, :, 3]
    dx34 = x4 - x3
    dy34 = y4 - y3
    denom = dy34 * dx12[:, None] - dx34 * dy12[:, None]
    valid = valid_flat & (torch.abs(denom) >= 1e-9)
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    s = (dx34 * (y1[:, None] - y3) - dy34 * (x1[:, None] - x3)) / safe_denom
    t = (dx12[:, None] * (y1[:, None] - y3) - dy12[:, None] * (x1[:, None] - x3)) / safe_denom
    intersects = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    return torch.any(intersects, dim=1)


def sensor_angles_tensor(*, count: int, spread_deg: float, forward_bias: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    resolved_count = max(3, int(count))
    if resolved_count % 2 == 0:
        resolved_count += 1
    base = torch.linspace(-1.0, 1.0, resolved_count, device=device, dtype=dtype)
    biased = torch.sign(base) * torch.pow(torch.abs(base), float(forward_bias))
    return biased * (float(spread_deg) * math.pi / 180.0 / 2.0)


def build_sensor_rays_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    heading_rad: torch.Tensor,
    *,
    sensor_angles: torch.Tensor,
    max_distance_px: torch.Tensor,
) -> torch.Tensor:
    headings = heading_rad[:, None] + sensor_angles[None, :]
    x0 = x[:, None].expand_as(headings)
    y0 = y[:, None].expand_as(headings)
    x1 = x0 + torch.cos(headings) * max_distance_px
    y1 = y0 - torch.sin(headings) * max_distance_px
    return torch.stack((x0, y0, x1, y1), dim=2)


def nearest_intersection_distance_batch(
    rays: torch.Tensor,
    segments: torch.Tensor,
    *,
    max_distance_px: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return nearest ray/segment hit distance in pixels for rays shaped ``[N, R, 4]``."""

    if segments.numel() == 0:
        return max_distance_px.expand(rays.shape[:2]).clone()
    flat = rays.reshape((-1, 4))
    x1 = flat[:, 0]
    y1 = flat[:, 1]
    x2 = flat[:, 2]
    y2 = flat[:, 3]
    dx12 = x2 - x1
    dy12 = y2 - y1
    nearest = max_distance_px.expand(flat.shape[0]).clone()
    for start in range(0, int(segments.shape[0]), chunk_size):
        chunk = segments[start : start + chunk_size]
        x3 = chunk[:, 0]
        y3 = chunk[:, 1]
        x4 = chunk[:, 2]
        y4 = chunk[:, 3]
        dx34 = x4 - x3
        dy34 = y4 - y3
        denom = dy34[None, :] * dx12[:, None] - dx34[None, :] * dy12[:, None]
        valid = torch.abs(denom) >= 1e-9
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        s = (dx34[None, :] * (y1[:, None] - y3[None, :]) - dy34[None, :] * (x1[:, None] - x3[None, :])) / safe_denom
        t = (dx12[:, None] * (y1[:, None] - y3[None, :]) - dy12[:, None] * (x1[:, None] - x3[None, :])) / safe_denom
        intersects = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
        distances = torch.sqrt(torch.clamp((dx12[:, None] * s) ** 2 + (dy12[:, None] * s) ** 2, min=0.0))
        distances = torch.where(intersects, distances, torch.full_like(distances, torch.inf))
        nearest = torch.minimum(nearest, torch.min(distances, dim=1).values)
    return nearest.reshape(rays.shape[:2])


def ray_distances_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    heading_rad: torch.Tensor,
    track: GpuTrackTensors,
    *,
    sensor_angles: torch.Tensor,
    range_m: float,
    chunk_size: int = 256,
) -> torch.Tensor:
    max_px = torch.as_tensor(float(range_m), device=x.device, dtype=x.dtype) / torch.clamp(track.meters_per_pixel, min=1e-6)
    rays = build_sensor_rays_batch(
        x,
        y,
        heading_rad,
        sensor_angles=sensor_angles.to(device=x.device, dtype=x.dtype),
        max_distance_px=max_px,
    )
    distances_px = nearest_intersection_distance_batch(
        rays,
        track.boundary_segments,
        max_distance_px=max_px,
        chunk_size=chunk_size,
    )
    return distances_px * track.meters_per_pixel
