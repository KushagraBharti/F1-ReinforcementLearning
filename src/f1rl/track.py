"""Track extraction and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from f1rl.config import TrackConfig
from f1rl.constants import DEFAULT_START_POS, WINDOW_SIZE
from f1rl.geometry import closed_loop, polyline_to_segments


@dataclass(slots=True)
class TrackDefinition:
    outer: np.ndarray
    inner: np.ndarray
    goals: np.ndarray
    collision_segments: np.ndarray

    @property
    def num_goals(self) -> int:
        return int(self.goals.shape[0])

    def get_goal(self, index: int) -> np.ndarray:
        return self.goals[index % self.num_goals]


def _contour_to_points(contour: np.ndarray) -> np.ndarray:
    return contour[:, 0, :].astype(np.float32)


@lru_cache(maxsize=8)
def load_track_definition(
    contour_path: str,
    threshold: int,
    num_goals: int,
    render_scale: float,
) -> TrackDefinition:
    image = cv2.imread(contour_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load contour image: {contour_path}")

    resized = cv2.resize(image, WINDOW_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    contours_simple, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_full, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours_simple) < 2 or len(contours_full) < 2:
        raise RuntimeError("Expected at least two contours (inner + outer track boundaries).")

    # Keep the two largest contours. The larger one is the outer boundary.
    simple_sorted = sorted(contours_simple, key=cv2.contourArea, reverse=True)[:2]
    full_sorted = sorted(contours_full, key=cv2.contourArea, reverse=True)[:2]

    outer = closed_loop(_contour_to_points(simple_sorted[0]))
    inner = closed_loop(_contour_to_points(simple_sorted[1]))

    outer_dense = _contour_to_points(full_sorted[0])
    inner_dense = _contour_to_points(full_sorted[1])

    collision_segments = np.vstack((polyline_to_segments(outer), polyline_to_segments(inner)))
    goals = _build_goals_legacy_style(
        outer_dense=outer_dense,
        inner_dense=inner_dense,
        num_goals=num_goals,
    )

    return TrackDefinition(
        outer=outer,
        inner=inner,
        goals=goals,
        collision_segments=collision_segments,
    )


def _build_goals_legacy_style(outer_dense: np.ndarray, inner_dense: np.ndarray, num_goals: int) -> np.ndarray:
    # Preserve legacy goal ordering from the original repository to avoid behavior drift.
    samples = num_goals + 1
    outer_idx = np.linspace(0, outer_dense.shape[0] - 1, samples).astype(int)
    inner_idx = np.linspace(0, inner_dense.shape[0] - 1, samples).astype(int)

    temp = np.zeros((samples, 4), dtype=np.float32)
    for i in range(samples):
        ax, ay = outer_dense[outer_idx[i]]
        bx, by = inner_dense[inner_idx[samples - 1 - i]]
        temp[i] = np.array([ax, ay, bx, by], dtype=np.float32)

    temp = temp[:-1]
    # Legacy cyclic reindexing to align line numbering with historical checkpoints.
    split = (num_goals // 2) + 1
    order = list(range(split, num_goals)) + list(range(0, split))
    order.reverse()
    temp = temp[order]
    temp = temp[:-1]

    goals = np.zeros((num_goals, 4), dtype=np.float32)
    sx, sy = DEFAULT_START_POS
    goals[0] = np.array([sx, sy + 12.0, sx, sy - 13.0], dtype=np.float32)
    goals[1:] = temp[: num_goals - 1]
    return goals


def build_track(config: TrackConfig) -> TrackDefinition:
    return load_track_definition(
        contour_path=str(config.contour_image),
        threshold=config.threshold,
        num_goals=config.num_goals,
        render_scale=config.render_scale,
    )
