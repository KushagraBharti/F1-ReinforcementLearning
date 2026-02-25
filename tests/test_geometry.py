from f1rl.geometry import line_intersection


def test_line_intersection_crossing() -> None:
    point = line_intersection(0, 0, 2, 2, 0, 2, 2, 0)
    assert point is not None
    x, y = point
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 1.0) < 1e-6


def test_line_intersection_parallel() -> None:
    point = line_intersection(0, 0, 1, 0, 0, 1, 1, 1)
    assert point is None
