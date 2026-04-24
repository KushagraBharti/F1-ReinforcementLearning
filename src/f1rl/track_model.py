"""TrackSpec persistence and runtime track access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1rl.config import SimConfig, TrackBuildConfig
from f1rl.geometry import polyline_lengths, polyline_to_segments


@dataclass(slots=True)
class TrackSpec:
    name: str
    source_image_size: tuple[int, int]
    centerline: np.ndarray
    centerline_s: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    checkpoints: np.ndarray
    checkpoint_s: np.ndarray
    start_pose: np.ndarray
    finish_line: np.ndarray
    meters_per_pixel: float
    real_track_length_m: float
    drivable_mask: np.ndarray

    @property
    def length_m(self) -> float:
        return float(self.real_track_length_m)

    @property
    def length_px(self) -> float:
        return float(self.centerline_s[-1])

    @property
    def boundary_segments(self) -> np.ndarray:
        return np.vstack((polyline_to_segments(self.left_boundary), polyline_to_segments(self.right_boundary))).astype(
            np.float32
        )

    def point_is_drivable(self, x: float, y: float) -> bool:
        xi = int(round(x))
        yi = int(round(y))
        if yi < 0 or xi < 0 or yi >= self.drivable_mask.shape[0] or xi >= self.drivable_mask.shape[1]:
            return False
        return bool(self.drivable_mask[yi, xi])


def save_track_spec(spec: TrackSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(spec.name),
        source_image_size=np.asarray(spec.source_image_size, dtype=np.int32),
        centerline=spec.centerline.astype(np.float32),
        centerline_s=spec.centerline_s.astype(np.float32),
        left_boundary=spec.left_boundary.astype(np.float32),
        right_boundary=spec.right_boundary.astype(np.float32),
        checkpoints=spec.checkpoints.astype(np.float32),
        checkpoint_s=spec.checkpoint_s.astype(np.float32),
        start_pose=spec.start_pose.astype(np.float32),
        finish_line=spec.finish_line.astype(np.float32),
        meters_per_pixel=np.asarray(spec.meters_per_pixel, dtype=np.float32),
        real_track_length_m=np.asarray(spec.real_track_length_m, dtype=np.float32),
        drivable_mask=spec.drivable_mask.astype(np.bool_),
    )


def load_track_spec(path: Path | None = None) -> TrackSpec:
    resolved = path or SimConfig().track_path
    if not resolved.exists():
        from f1rl.track_build import build_track

        build_track(TrackBuildConfig(output_dir=resolved.parent))
    with np.load(resolved, allow_pickle=False) as data:
        source_size = tuple(int(v) for v in data["source_image_size"].tolist())
        return TrackSpec(
            name=str(data["name"].item()),
            source_image_size=(source_size[0], source_size[1]),
            centerline=data["centerline"].astype(np.float32),
            centerline_s=data["centerline_s"].astype(np.float32),
            left_boundary=data["left_boundary"].astype(np.float32),
            right_boundary=data["right_boundary"].astype(np.float32),
            checkpoints=data["checkpoints"].astype(np.float32),
            checkpoint_s=data["checkpoint_s"].astype(np.float32),
            start_pose=data["start_pose"].astype(np.float32),
            finish_line=data["finish_line"].astype(np.float32),
            meters_per_pixel=float(data["meters_per_pixel"]),
            real_track_length_m=float(data["real_track_length_m"]),
            drivable_mask=data["drivable_mask"].astype(bool),
        )


def compute_centerline_s(centerline: np.ndarray) -> np.ndarray:
    return polyline_lengths(centerline).astype(np.float32)
