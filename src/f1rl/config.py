"""Configuration and project paths for the simplified simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
IMAGES_DIR = REPO_ROOT / "imgs"
ASSETS_DIR = REPO_ROOT / "assets"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MONZA_ASSET_DIR = ASSETS_DIR / "tracks" / "monza"

TRACK_SOURCE_SIZE = (2463, 1244)
TRACK_RENDER_SCALE = 1.0 / 1.3
TRACK_WINDOW_SIZE = (
    int(TRACK_SOURCE_SIZE[0] / 1.3),
    int(TRACK_SOURCE_SIZE[1] / 1.3),
)
MONZA_LENGTH_METERS = 5793.0
DEFAULT_START_POS = (1501.0, 870.0)


@dataclass(slots=True)
class TrackBuildConfig:
    name: str = "monza"
    contour_image: Path = IMAGES_DIR / "Monza_track_extra_wide_contour.png"
    track_image: Path = IMAGES_DIR / "Monza_track_extra_wide_2.png"
    background_image: Path = IMAGES_DIR / "Monza_background.png"
    output_dir: Path = MONZA_ASSET_DIR
    threshold: int = 225
    num_checkpoints: int = 120
    real_track_length_m: float = MONZA_LENGTH_METERS
    start_pos: tuple[float, float] = DEFAULT_START_POS
    start_heading_deg: float = 180.0
    coordinate_scale: float = TRACK_RENDER_SCALE


@dataclass(slots=True)
class SensorConfig:
    count: int = 7
    spread_deg: float = 120.0
    forward_bias: float = 1.6
    range_m: float = 1300.0


@dataclass(slots=True)
class CarParams:
    mass: float = 798.0
    wheelbase_m: float = 3.6
    max_steer_deg: float = 18.0
    steer_response: float = 6.0
    engine_accel_mps2: float = 24.5
    brake_accel_mps2: float = 38.0
    drag_coefficient: float = 0.0025
    rolling_resistance_mps2: float = 0.25
    grip_g: float = 4.1
    max_speed_mps: float = 110.0
    dt: float = 1.0 / 60.0


@dataclass(slots=True)
class RewardConfig:
    progress_scale: float = 0.08
    finish_bonus: float = 100.0
    collision_penalty: float = -25.0
    off_track_penalty: float = -25.0
    no_progress_penalty: float = -10.0
    smoothness_penalty: float = 0.0

    def component_keys(self) -> tuple[str, ...]:
        return (
            "progress",
            "finish",
            "collision",
            "off_track",
            "no_progress",
            "smoothness",
        )


@dataclass(slots=True)
class SimConfig:
    track_path: Path = MONZA_ASSET_DIR / "track_spec.npz"
    car_image: Path = IMAGES_DIR / "ferrari.png"
    max_steps: int = 3600
    no_progress_limit_steps: int = 180
    local_projection_window_m: float = 500.0
    lookahead_m: tuple[float, ...] = (40.0, 90.0, 160.0, 280.0)
    car: CarParams = field(default_factory=CarParams)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class RenderConfig:
    window_size: tuple[int, int] = TRACK_WINDOW_SIZE
    car_sprite_scale: float = 0.1
    car_length_m: float = 5.6
    car_width_m: float = 2.0
    min_car_length_px: int = 16
    min_car_width_px: int = 6
    draw_track_image: bool = True
    draw_boundaries: bool = False
    draw_centerline: bool = False
    draw_checkpoints: bool = False
    draw_rays: bool = True
    draw_ray_hits: bool = True
    fps: int = 60


@dataclass(slots=True)
class RunConfig:
    seed: int = 7
    artifacts_dir: Path = ARTIFACTS_DIR
    telemetry_enabled: bool = True


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    data = asdict(value)
    return _stringify_paths(data)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


DISCRETE_ACTIONS: tuple[tuple[str, float, float, float], ...] = (
    ("coast", 0.0, 0.0, 0.0),
    ("throttle", 1.0, 0.0, 0.0),
    ("brake", 0.0, 1.0, 0.0),
    ("left", 0.0, 0.0, -1.0),
    ("right", 0.0, 0.0, 1.0),
    ("throttle_left", 1.0, 0.0, -1.0),
    ("throttle_right", 1.0, 0.0, 1.0),
    ("brake_left", 0.0, 1.0, -1.0),
    ("brake_right", 0.0, 1.0, 1.0),
)


def action_to_controls(action_id: int) -> tuple[float, float, float]:
    _, throttle, brake, steer = DISCRETE_ACTIONS[int(action_id) % len(DISCRETE_ACTIONS)]
    return throttle, brake, steer
