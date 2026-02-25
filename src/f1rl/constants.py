"""Project-wide constants and default paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
IMAGES_DIR = REPO_ROOT / "imgs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
RENDERS_DIR = ARTIFACTS_DIR / "renders"

TRACK_SOURCE_SIZE = (2463, 1244)
TRACK_RENDER_SCALE = 1.0 / 1.3
WINDOW_SIZE = tuple(int(v * TRACK_RENDER_SCALE) for v in TRACK_SOURCE_SIZE)
DEFAULT_START_POS = (1501.0, 870.0)
DEFAULT_START_HEADING_DEG = 92.0
