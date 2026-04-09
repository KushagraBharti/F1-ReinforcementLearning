"""Configuration dataclasses for environment, dynamics, and rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from f1rl.constants import DEFAULT_START_POS, IMAGES_DIR, WINDOW_SIZE


@dataclass(slots=True)
class DynamicsConfig:
    max_speed: float = 11.0
    max_reverse_speed: float = 3.5
    engine_acceleration_rate: float = 0.32
    brake_deceleration_rate: float = 0.52
    reverse_acceleration_rate: float = 0.14
    coast_deceleration_rate: float = 0.04
    friction: float = 0.015
    drag_coefficient: float = 0.0025
    wheelbase: float = 28.0
    max_steer_angle_deg: float = 30.0
    steering_response: float = 0.4
    high_speed_steer_reduction: float = 0.6
    lateral_grip: float = 0.92
    sensor_count: int = 9
    sensor_spread_deg: float = 140.0
    sensor_forward_bias: float = 1.6
    sensor_range: float = 750.0
    dt: float = 1.0


@dataclass(slots=True)
class RewardConfig:
    progress_reward: float = 2.5
    reverse_penalty: float = -5.0
    collision_penalty: float = -8.0
    step_penalty: float = -0.002
    forward_speed_weight: float = 0.035
    negative_speed_penalty_weight: float = 0.08
    alignment_weight: float = 0.02
    centerline_penalty_weight: float = 0.02
    steering_change_penalty_weight: float = 0.003
    lap_bonus: float = 15.0
    stall_speed_threshold: float = 0.10
    stall_penalty: float = -0.012
    idle_penalty: float = -0.003


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
    terminate_on_no_progress: bool = True
    no_progress_limit_steps: int = 150
    reverse_speed_threshold: float = 0.35
    reverse_speed_limit_steps: int = 50
    reverse_progress_limit: int = 4
    render_mode: str | None = None
    seed: int | None = None
    start_pos: tuple[float, float] = DEFAULT_START_POS
    # If None, heading is auto-aligned to the first checkpoint direction at reset.
    start_heading_deg: float | None = None
    headless: bool = False
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None = None) -> EnvConfig:
        raw = raw or {}
        base = cls()
        for key, value in raw.items():
            if key == "dynamics" and isinstance(value, dict):
                dynamics_values = {**asdict(base.dynamics), **_upgrade_dynamics_dict(value)}
                base.dynamics = DynamicsConfig(**dynamics_values)
            elif key == "reward" and isinstance(value, dict):
                reward_values = {**asdict(base.reward), **_upgrade_reward_dict(value)}
                base.reward = RewardConfig(**reward_values)
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


def _upgrade_dynamics_dict(raw: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(raw)
    if "acceleration_rate" in upgraded:
        upgraded["engine_acceleration_rate"] = upgraded.pop("acceleration_rate")
    if "braking_rate" in upgraded:
        upgraded["brake_deceleration_rate"] = upgraded.pop("braking_rate")
    if "reverse_rate" in upgraded:
        upgraded["reverse_acceleration_rate"] = upgraded.pop("reverse_rate")
    if "steering_rate_deg" in upgraded:
        upgraded["max_steer_angle_deg"] = upgraded.pop("steering_rate_deg") * 6.0
    return upgraded


def _upgrade_reward_dict(raw: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(raw)
    if "speed_weight" in upgraded:
        upgraded["forward_speed_weight"] = upgraded.pop("speed_weight")
    if "idle_speed_threshold" in upgraded:
        upgraded["stall_speed_threshold"] = upgraded.pop("idle_speed_threshold")
    return upgraded
