"""Build the persisted Monza TrackSpec from the source images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from f1rl.config import TrackBuildConfig, dataclass_to_dict
from f1rl.geometry import closed_loop, polyline_lengths, resample_closed_polyline
from f1rl.track_model import TrackSpec, save_track_spec


def _contour_points(contour: np.ndarray) -> np.ndarray:
    return contour[:, 0, :].astype(np.float32)


def _find_track_contours(config: TrackBuildConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(config.contour_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load contour image: {config.contour_image}")
    if config.coordinate_scale != 1.0:
        image = cv2.resize(
            image,
            (
                int(image.shape[1] * config.coordinate_scale),
                int(image.shape[0] * config.coordinate_scale),
            ),
        )
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        raise RuntimeError("Expected at least two contours for Monza inner/outer boundaries.")
    outer_contour, inner_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [outer_contour], -1, 255, thickness=-1)
    cv2.drawContours(mask, [inner_contour], -1, 0, thickness=-1)
    return _contour_points(outer_contour), _contour_points(inner_contour), mask.astype(bool), (width, height)


def _legacy_ordered_centerline(
    outer_dense: np.ndarray,
    inner_dense: np.ndarray,
    *,
    start_pos: tuple[float, float],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = count + 1
    outer_idx = np.linspace(0, outer_dense.shape[0] - 1, samples).astype(int)
    inner_idx = np.linspace(0, inner_dense.shape[0] - 1, samples).astype(int)
    gates = np.zeros((samples, 4), dtype=np.float32)
    for i in range(samples):
        ax, ay = outer_dense[outer_idx[i]]
        bx, by = inner_dense[inner_idx[samples - 1 - i]]
        gates[i] = np.array([ax, ay, bx, by], dtype=np.float32)
    gates = gates[:-1]
    split = (count // 2) + 1
    order = list(range(split, count)) + list(range(0, split))
    order.reverse()
    gates = gates[order]
    gates[0] = np.array([start_pos[0], start_pos[1] + 18.0, start_pos[0], start_pos[1] - 18.0], dtype=np.float32)
    midpoints = np.column_stack(((gates[:, 0] + gates[:, 2]) * 0.5, (gates[:, 1] + gates[:, 3]) * 0.5)).astype(
        np.float32
    )
    centerline = closed_loop(midpoints)
    checkpoints = midpoints.astype(np.float32)
    return centerline, checkpoints, gates


def build_track(config: TrackBuildConfig | None = None) -> Path:
    config = config or TrackBuildConfig()
    outer_dense, inner_dense, drivable_mask, source_size = _find_track_contours(config)
    left_boundary = closed_loop(resample_closed_polyline(outer_dense, 900))
    right_boundary = closed_loop(resample_closed_polyline(inner_dense, 900))
    centerline, checkpoints, gates = _legacy_ordered_centerline(
        outer_dense,
        inner_dense,
        start_pos=config.start_pos,
        count=config.num_checkpoints,
    )
    centerline_s = polyline_lengths(centerline)
    meters_per_pixel = float(config.real_track_length_m / max(float(centerline_s[-1]), 1e-6))
    checkpoint_s = np.linspace(0.0, float(centerline_s[-1]), checkpoints.shape[0], endpoint=False, dtype=np.float32)
    heading = float(np.deg2rad(config.start_heading_deg))
    spec = TrackSpec(
        name=config.name,
        source_image_size=source_size,
        centerline=centerline,
        centerline_s=centerline_s,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        checkpoints=checkpoints,
        checkpoint_s=checkpoint_s,
        start_pose=np.asarray([config.start_pos[0], config.start_pos[1], heading], dtype=np.float32),
        finish_line=gates[0].astype(np.float32),
        meters_per_pixel=meters_per_pixel,
        real_track_length_m=float(config.real_track_length_m),
        drivable_mask=drivable_mask,
    )
    output_path = config.output_dir / "track_spec.npz"
    save_track_spec(spec, output_path)
    manifest = {
        "config": dataclass_to_dict(config),
        "output": str(output_path),
        "source_image_size": source_size,
        "centerline_length_px": float(centerline_s[-1]),
        "meters_per_pixel": meters_per_pixel,
        "checkpoints": int(checkpoints.shape[0]),
    }
    (config.output_dir / "track_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the persisted Monza TrackSpec.")
    parser.add_argument("--output-dir", type=Path, default=TrackBuildConfig().output_dir)
    parser.add_argument("--threshold", type=int, default=TrackBuildConfig().threshold)
    parser.add_argument("--checkpoints", type=int, default=TrackBuildConfig().num_checkpoints)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = build_track(
        TrackBuildConfig(
            output_dir=args.output_dir,
            threshold=args.threshold,
            num_checkpoints=args.checkpoints,
        )
    )
    print(f"track_built path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
