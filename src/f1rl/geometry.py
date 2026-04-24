"""Small geometry helpers for track, ray, and progress calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def closed_loop(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        raise ValueError("expected non-empty Nx2 point array")
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack((arr, arr[0]))
    return arr.astype(np.float32)


def polyline_to_segments(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        raise ValueError("polyline requires at least two points")
    return np.column_stack((points[:-1], points[1:])).astype(np.float32)


def polyline_lengths(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths))).astype(np.float32)


def resample_closed_polyline(points: np.ndarray, count: int) -> np.ndarray:
    closed = closed_loop(points)
    cumulative = polyline_lengths(closed)
    total = float(cumulative[-1])
    if total <= 1e-6:
        raise ValueError("cannot resample zero-length polyline")
    samples = np.linspace(0.0, total, count, endpoint=False, dtype=np.float32)
    return sample_polyline_at(closed, cumulative, samples)


def sample_polyline_at(points: np.ndarray, cumulative: np.ndarray, distances: np.ndarray) -> np.ndarray:
    total = float(cumulative[-1])
    wrapped = np.mod(distances, total)
    indices = np.searchsorted(cumulative, wrapped, side="right") - 1
    indices = np.clip(indices, 0, len(cumulative) - 2)
    seg_start = points[indices]
    seg_end = points[indices + 1]
    seg_len = np.maximum(cumulative[indices + 1] - cumulative[indices], 1e-6)
    t = ((wrapped - cumulative[indices]) / seg_len).reshape(-1, 1)
    return (seg_start + (seg_end - seg_start) * t).astype(np.float32)


def line_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> tuple[float, float] | None:
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return None
    s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
        return None
    return x1 + s * (x2 - x1), y1 + s * (y2 - y1)


def segment_intersects_any(segment: np.ndarray, candidates: np.ndarray) -> bool:
    x1, y1, x2, y2 = [float(v) for v in segment]
    for x3, y3, x4, y4 in candidates:
        if line_intersection(x1, y1, x2, y2, float(x3), float(y3), float(x4), float(y4)):
            return True
    return False


def nearest_intersection_distance(ray: np.ndarray, segments: np.ndarray, max_distance: float) -> float:
    x1, y1, x2, y2 = [float(v) for v in ray]
    best = float(max_distance)
    for x3, y3, x4, y4 in segments:
        point = line_intersection(x1, y1, x2, y2, float(x3), float(y3), float(x4), float(y4))
        if point is None:
            continue
        px, py = point
        distance = float(np.hypot(px - x1, py - y1))
        if distance < best:
            best = distance
    return best


def project_point_to_polyline(
    point: np.ndarray,
    points: np.ndarray,
    cumulative: np.ndarray,
    *,
    previous_progress: float | None = None,
    window: float | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Project a point onto a closed centerline.

    Returns progress in polyline units, unsigned distance, tangent heading radians,
    and the closest projected point. If previous_progress/window are given, only
    nearby centerline segments are considered to avoid jumps across close track
    sections.
    """

    total = float(cumulative[-1])
    point = np.asarray(point, dtype=np.float32)
    best_distance = float("inf")
    best_progress = 0.0
    best_heading = 0.0
    best_projection = points[0].astype(np.float32)
    prev_mod = None if previous_progress is None else float(previous_progress % total)

    for idx in range(points.shape[0] - 1):
        if prev_mod is not None and window is not None:
            seg_mid = float((cumulative[idx] + cumulative[idx + 1]) * 0.5)
            wrapped_delta = abs(((seg_mid - prev_mod + total * 0.5) % total) - total * 0.5)
            if wrapped_delta > window:
                continue
        start = points[idx]
        end = points[idx + 1]
        line = end - start
        norm = float(np.dot(line, line))
        if norm <= 1e-9:
            continue
        t = float(np.clip(np.dot(point - start, line) / norm, 0.0, 1.0))
        projection = start + line * t
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_progress = float(cumulative[idx] + np.sqrt(norm) * t)
            best_heading = float(np.arctan2(-(end[1] - start[1]), end[0] - start[0]))
            best_projection = projection.astype(np.float32)
    return best_progress % total, best_distance, best_heading, best_projection


def wrap_radians(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
