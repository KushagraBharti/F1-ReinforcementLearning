import numpy as np

from f1rl.geometry import line_intersection, nearest_polyline_distance


def test_line_intersection_crossing() -> None:
    point = line_intersection(0, 0, 2, 2, 0, 2, 2, 0)
    assert point is not None
    x, y = point
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 1.0) < 1e-6


def test_line_intersection_parallel() -> None:
    point = line_intersection(0, 0, 1, 0, 0, 1, 1, 1)
    assert point is None


def test_nearest_polyline_distance() -> None:
    polyline = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    distance = nearest_polyline_distance(np.array([1.0, 1.0], dtype=np.float32), polyline)
    assert abs(distance - 1.0) < 1e-6
