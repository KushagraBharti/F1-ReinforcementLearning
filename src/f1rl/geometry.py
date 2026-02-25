"""Geometry helper functions for track and sensor intersection math."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


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
    """Return line segment intersection point or None."""
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return None

    s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
        return None

    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


def segment_intersects_any(segment: np.ndarray, candidates: np.ndarray) -> bool:
    """Check if a segment intersects any candidate segment."""
    x1, y1, x2, y2 = segment
    for x3, y3, x4, y4 in candidates:
        if line_intersection(x1, y1, x2, y2, x3, y3, x4, y4) is not None:
            return True
    return False


def nearest_intersection_distance(ray: np.ndarray, segments: np.ndarray, max_distance: float) -> float:
    """Return nearest intersection distance for a ray and track segments."""
    x1, y1, x2, y2 = ray
    best = max_distance
    for x3, y3, x4, y4 in segments:
        point = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        if point is None:
            continue
        px, py = point
        distance = float(np.hypot(px - x1, py - y1))
        if distance < best:
            best = distance
    return best


def polyline_to_segments(points: np.ndarray) -> np.ndarray:
    """Convert closed polyline points array into segment vectors."""
    if points.shape[0] < 2:
        raise ValueError("polyline needs at least two points")
    segments = np.zeros((points.shape[0] - 1, 4), dtype=np.float32)
    segments[:, 0:2] = points[:-1]
    segments[:, 2:4] = points[1:]
    return segments


def closed_loop(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Ensure the contour is a closed loop by repeating first point."""
    arr = np.asarray(list(points), dtype=np.float32)
    if arr.shape[0] == 0:
        raise ValueError("empty contour")
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack((arr, arr[0]))
    return arr
