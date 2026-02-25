"""Configuration dataclasses for environment, dynamics, and rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from f1rl.constants import DEFAULT_START_POS, IMAGES_DIR, WINDOW_SIZE


@dataclass(slots=True)
class DynamicsConfig:
    max_speed: float = 9.0
    acceleration_rate: float = 0.24
    braking_rate: float = 0.32
    friction: float = 0.02
    steering_rate_deg: float = 4.5
    sensor_count: int = 7
    sensor_spread_deg: float = 120.0
    sensor_range: float = 750.0
    dt: float = 1.0


@dataclass(slots=True)
class RewardConfig:
    progress_reward: float = 1.0
    reverse_penalty: float = -1.0
    collision_penalty: float = -2.0
    step_penalty: float = -0.001
    speed_weight: float = 0.02
    lap_bonus: float = 5.0
    idle_speed_threshold: float = 0.05
    idle_penalty: float = -0.002


@dataclass(slots=True)
class TrackConfig:
    contour_image: Path = IMAGES_DIR / "Monza_track_extra_wide_contour.png"
    track_image: Path = IMAGES_DIR / "Monza_track_extra_wide_2.png"
    background_image: Path = IMAGES_DIR / "Monza_background.png"
    car_image: Path = IMAGES_DIR / "ferrari.png"
    car_off_image: Path = IMAGES_DIR / "ferrari_off.png"
    # Car sprite points up in source image, while world heading 0 deg points right.
    # Offset keeps rendered nose aligned with physics/sensor forward direction.
    car_sprite_heading_offset_deg: float = -90.0
    num_goals: int = 120
    threshold: int = 225
    render_scale: float = 1.0 / 1.3


@dataclass(slots=True)
class RenderConfig:
    enabled: bool = False
    draw_track_lines: bool = False
    draw_goals: bool = False
    draw_sensors: bool = True
    draw_hud: bool = True
    fps: int = 60
    export_frames: bool = False
    frame_prefix: str = "rollout"
    window_size: tuple[int, int] = (WINDOW_SIZE[0], WINDOW_SIZE[1])


@dataclass(slots=True)
class EnvConfig:
    max_steps: int = 2000
    max_laps: int = 1
    terminate_on_collision: bool = True
    render_mode: str | None = None
    seed: int | None = None
    start_pos: tuple[float, float] = DEFAULT_START_POS
    # If None, heading is auto-aligned to the first checkpoint direction at reset.
    start_heading_deg: float | None = None
    headless: bool = False
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    render: RenderConfig = field(default_factory=lambda: RenderConfig())

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None = None) -> EnvConfig:
        raw = raw or {}
        base = cls()
        for key, value in raw.items():
            if key == "dynamics" and isinstance(value, dict):
                base.dynamics = DynamicsConfig(**{**asdict(base.dynamics), **value})
            elif key == "reward" and isinstance(value, dict):
                base.reward = RewardConfig(**{**asdict(base.reward), **value})
            elif key == "track" and isinstance(value, dict):
                track_values = {
                    **asdict(base.track),
                    **{
                        sub_key: Path(sub_val) if "image" in sub_key else sub_val
                        for sub_key, sub_val in value.items()
                    },
                }
                base.track = TrackConfig(**track_values)
            elif key == "render" and isinstance(value, dict):
                render_values = {**asdict(base.render), **value}
                if "window_size" in render_values:
                    render_values["window_size"] = tuple(render_values["window_size"])
                base.render = RenderConfig(**render_values)
            elif hasattr(base, key):
                setattr(base, key, value)
        return base


def to_dict(config: EnvConfig) -> dict[str, Any]:
    data = asdict(config)
    track = data.get("track", {})
    for key, value in track.items():
        if "image" in key and isinstance(value, Path):
            track[key] = str(value)
    return data
