from pathlib import Path

import numpy as np

from f1rl.config import MONZA_LENGTH_METERS, TrackBuildConfig
from f1rl.track_build import build_track
from f1rl.track_model import load_track_spec


def test_track_build_persists_monza_spec(tmp_path: Path) -> None:
    output = build_track(TrackBuildConfig(output_dir=tmp_path, num_checkpoints=48))
    spec = load_track_spec(output)
    assert spec.name == "monza"
    assert spec.source_image_size == (1894, 956)
    assert spec.centerline.shape[1] == 2
    assert np.allclose(spec.centerline[0], spec.centerline[-1])
    assert spec.left_boundary.shape[0] > 100
    assert spec.right_boundary.shape[0] > 100
    assert spec.checkpoints.shape == (48, 2)
    assert spec.drivable_mask.any()
    assert abs(spec.real_track_length_m - MONZA_LENGTH_METERS) < 1e-3
    assert spec.meters_per_pixel > 0.0


def test_start_pose_is_drivable(tmp_path: Path) -> None:
    output = build_track(TrackBuildConfig(output_dir=tmp_path))
    spec = load_track_spec(output)
    assert spec.point_is_drivable(float(spec.start_pose[0]), float(spec.start_pose[1]))
