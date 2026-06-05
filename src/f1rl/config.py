"""Configuration and project paths for the simplified simulator."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
IMAGES_DIR = REPO_ROOT / "imgs"
ASSETS_DIR = REPO_ROOT / "assets"
ARTIFACT_ROOT = Path(os.environ.get("F1RL_ARTIFACT_ROOT", REPO_ROOT / "artifacts")).expanduser().resolve()
ARTIFACTS_DIR = ARTIFACT_ROOT / "runs"
DATASETS_DIR = ARTIFACT_ROOT / "datasets"
LEARNED_DIR = ARTIFACT_ROOT / "learned"
CALIBRATION_ARTIFACTS_DIR = ARTIFACT_ROOT / "calibration"
FASTF1_CACHE_DIR = ARTIFACT_ROOT / "fastf1-cache"
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
    grip_g: float = 2.2
    aero_grip_per_mps2: float = 0.00023
    max_grip_g: float = 4.2
    max_drive_g: float = 2.45
    max_brake_g: float = 4.6
    steering_speed_sensitivity: float = 0.0008
    max_speed_mps: float = 110.0
    dt: float = 1.0 / 60.0


@dataclass(slots=True)
class RewardConfig:
    progress_scale: float = 0.08
    finish_bonus: float = 100.0
    collision_penalty: float = -60.0
    off_track_penalty: float = -60.0
    no_progress_penalty: float = -90.0
    lateral_deadzone_m: float = 4.0
    lateral_penalty_scale: float = 0.025
    track_limit_safe_ray_m: float = 10.0
    track_limit_penalty_scale: float = 0.03
    heading_deadzone_deg: float = 5.0
    heading_penalty_scale: float = 0.0
    speed_target_min_kph: float = 80.0
    speed_target_max_kph: float = 340.0
    speed_target_heading_scale: float = 2.5
    speed_target_deadzone_kph: float = 15.0
    speed_target_penalty_scale: float = 0.0
    overspeed_throttle_penalty_scale: float = 0.0
    overspeed_brake_reward_scale: float = 0.0
    steering_target_deadzone: float = 0.08
    steering_target_penalty_scale: float = 0.0
    smoothness_penalty: float = 0.0
    scaffold_scale: float = 1.0
    scaffold_brake_reward_scale: float = 0.0
    scaffold_no_throttle_penalty_scale: float = 0.0
    scaffold_turn_in_speed_penalty_scale: float = 0.0
    scaffold_apex_clean_reward_scale: float = 0.0
    scaffold_exit_alignment_reward_scale: float = 0.0
    scaffold_exit_speed_reward_scale: float = 0.0
    scaffold_release_reward_scale: float = 0.0
    scaffold_overbrake_penalty_scale: float = 0.0
    scaffold_release_min_speed_kph: float = 135.0
    scaffold_release_max_speed_kph: float = 210.0
    scaffold_brake_curve_penalty_scale: float = 0.0
    scaffold_brake_curve_start_speed_kph: float = 335.0
    scaffold_brake_curve_deadzone_kph: float = 8.0
    scaffold_corridor_center_penalty_scale: float = 0.0
    scaffold_corridor_center_deadzone_m: float = 2.0
    scaffold_segment_speed_penalty_scale: float = 0.0

    def component_keys(self) -> tuple[str, ...]:
        return (
            "progress",
            "finish",
            "collision",
            "off_track",
            "no_progress",
            "lateral",
            "track_limit",
            "heading",
            "speed_target",
            "overspeed_action",
            "steering_target",
            "smoothness",
            "scaffold_brake",
            "scaffold_no_throttle",
            "scaffold_turn_in_speed",
            "scaffold_apex_clean",
            "scaffold_exit_alignment",
            "scaffold_exit_speed",
            "scaffold_release",
            "scaffold_overbrake",
            "scaffold_brake_curve",
            "scaffold_corridor_center",
            "scaffold_segment_speed",
            "assist_overspeed_gate",
            "assist_throttle_brake_demand",
            "assist_no_brake_gate",
            "assist_overbrake_gate",
            "assist_forbidden_steering_gate",
            "assist_virtual_corridor",
            "assist_brake_zone_progress_suppression",
        )


@dataclass(slots=True)
class AssistConfig:
    enabled: bool = False
    overspeed_turn_in_terminate: bool = False
    overspeed_turn_in_margin_kph: float = 45.0
    overspeed_turn_in_penalty: float = -80.0
    throttle_brake_demand_penalty_scale: float = 0.0
    throttle_brake_demand_terminate: bool = False
    throttle_brake_demand_min_throttle: float = 0.75
    no_brake_penalty: float = 0.0
    no_brake_min_brake: float = 0.05
    no_brake_min_speed_kph: float = 0.0
    no_brake_terminate: bool = False
    overbrake_penalty: float = 0.0
    overbrake_terminate: bool = False
    overbrake_max_speed_kph: float = 135.0
    overbrake_min_brake: float = 0.50
    steering_gate_penalty: float = 0.0
    steering_gate_terminate: bool = False
    steering_gate_start_m: float = 0.0
    steering_gate_end_m: float = 0.0
    steering_gate_min_abs_steer: float = 0.0
    steering_gate_required_sign: float = 0.0
    steering_gate_min_speed_kph: float = 0.0
    forbidden_steering_gate_penalty: float = 0.0
    forbidden_steering_gate_terminate: bool = False
    forbidden_steering_gate_start_m: float = 0.0
    forbidden_steering_gate_end_m: float = 0.0
    forbidden_steering_gate_min_abs_steer: float = 0.0
    forbidden_steering_gate_sign: float = 0.0
    forbidden_steering_gate_min_speed_kph: float = 0.0
    virtual_corridor_m: float = 0.0
    virtual_corridor_penalty: float = -80.0
    virtual_corridor_terminate: bool = False
    brake_zone_progress_multiplier: float = 1.0


@dataclass(slots=True)
class SimConfig:
    track_path: Path = MONZA_ASSET_DIR / "track_spec.npz"
    car_image: Path = IMAGES_DIR / "ferrari.png"
    max_steps: int = 3600
    action_mode: str = "discrete"
    action_set: str = "legacy"
    continuous_action_scheme: str = "drive_brake"
    observation_profile: str = "base"
    no_progress_limit_steps: int = 180
    local_projection_window_m: float = 500.0
    checkpoint_lateral_limit_m: float = 24.0
    launch_guard_progress_m: float = 0.0
    launch_guard_min_speed_kph: float = 0.0
    launch_guard_throttle: float = 0.22
    lookahead_m: tuple[float, ...] = (40.0, 90.0, 160.0, 280.0)
    car: CarParams = field(default_factory=CarParams)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    assist: AssistConfig = field(default_factory=AssistConfig)


def build_reward_config(overrides: dict[str, float | None] | None = None) -> RewardConfig:
    reward = RewardConfig()
    if not overrides:
        return reward
    valid_keys = set(RewardConfig.__dataclass_fields__)
    unknown = set(overrides) - valid_keys
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown reward override(s): {unknown_text}")
    cleaned = {key: float(value) for key, value in overrides.items() if value is not None}
    return replace(reward, **cleaned)


def build_assist_config(overrides: dict[str, bool | float | None] | None = None) -> AssistConfig:
    assist = AssistConfig()
    if not overrides:
        return assist
    valid_keys = set(AssistConfig.__dataclass_fields__)
    unknown = set(overrides) - valid_keys
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown assist override(s): {unknown_text}")
    cleaned = {key: value for key, value in overrides.items() if value is not None}
    return replace(assist, **cleaned)


SCAFFOLD_REWARD_FIELDS = (
    "scaffold_scale",
    "scaffold_brake_reward_scale",
    "scaffold_no_throttle_penalty_scale",
    "scaffold_turn_in_speed_penalty_scale",
    "scaffold_apex_clean_reward_scale",
    "scaffold_exit_alignment_reward_scale",
    "scaffold_exit_speed_reward_scale",
    "scaffold_release_reward_scale",
    "scaffold_overbrake_penalty_scale",
    "scaffold_brake_curve_penalty_scale",
)


def scaffold_rewards_enabled(reward: RewardConfig) -> bool:
    return any(abs(float(getattr(reward, key))) > 1e-12 for key in SCAFFOLD_REWARD_FIELDS if key != "scaffold_scale")


def disable_scaffold_rewards(config: SimConfig) -> SimConfig:
    for key in SCAFFOLD_REWARD_FIELDS:
        setattr(config.reward, key, 0.0)
    return config


def training_assists_enabled(assist: AssistConfig) -> bool:
    return bool(
        assist.enabled
        or assist.overspeed_turn_in_terminate
        or assist.throttle_brake_demand_terminate
        or abs(assist.throttle_brake_demand_penalty_scale) > 1e-12
        or abs(assist.no_brake_penalty) > 1e-12
        or assist.no_brake_terminate
        or assist.no_brake_min_speed_kph > 0.0
        or abs(assist.overbrake_penalty) > 1e-12
        or assist.overbrake_terminate
        or abs(assist.steering_gate_penalty) > 1e-12
        or assist.steering_gate_terminate
        or assist.steering_gate_end_m > assist.steering_gate_start_m
        or abs(assist.forbidden_steering_gate_penalty) > 1e-12
        or assist.forbidden_steering_gate_terminate
        or assist.forbidden_steering_gate_end_m > assist.forbidden_steering_gate_start_m
        or assist.virtual_corridor_m > 0.0
        or assist.virtual_corridor_terminate
        or abs(assist.brake_zone_progress_multiplier - 1.0) > 1e-12
    )


def disable_training_assists(config: SimConfig) -> SimConfig:
    config.assist = AssistConfig()
    return config


def build_sim_config(
    *,
    max_steps: int = 3600,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    reward_overrides: dict[str, float | None] | None = None,
    assist_overrides: dict[str, bool | float | None] | None = None,
) -> SimConfig:
    if action_mode not in ACTION_MODES:
        valid = ", ".join(sorted(ACTION_MODES))
        raise ValueError(f"action_mode must be one of: {valid}.")
    actions_for_action_set(action_set)
    if continuous_action_scheme not in CONTINUOUS_ACTION_SCHEMES:
        valid = ", ".join(sorted(CONTINUOUS_ACTION_SCHEMES))
        raise ValueError(f"Unknown continuous action scheme {continuous_action_scheme!r}; expected one of: {valid}")
    if observation_profile not in OBSERVATION_PROFILES:
        valid = ", ".join(sorted(OBSERVATION_PROFILES))
        raise ValueError(f"Unknown observation profile {observation_profile!r}; expected one of: {valid}")
    return SimConfig(
        max_steps=max_steps,
        action_mode=action_mode,
        action_set=action_set,
        continuous_action_scheme=continuous_action_scheme,
        observation_profile=observation_profile,
        launch_guard_progress_m=launch_guard_progress_m,
        launch_guard_min_speed_kph=launch_guard_min_speed_kph,
        launch_guard_throttle=launch_guard_throttle,
        reward=build_reward_config(reward_overrides),
        assist=build_assist_config(assist_overrides),
    )


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


ActionSpec = tuple[str, float, float, float]
DriveSpec = tuple[str, float, float]
SteerSpec = tuple[str, float]
DEFAULT_ACTION_SET = "legacy"
ACTION_MODES = frozenset({"continuous", "discrete", "multidiscrete"})
CONTINUOUS_ACTION_SCHEMES = frozenset({"drive_brake", "exclusive_throttle_bias", "throttle_bias"})
OBSERVATION_PROFILES = frozenset({"base", "brake", "guidance", "racing", "racing_release", "racing_v2"})

LEGACY_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
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

EXPANDED_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    *LEGACY_DISCRETE_ACTIONS,
    ("half_throttle", 0.5, 0.0, 0.0),
    ("soft_left", 0.0, 0.0, -0.45),
    ("soft_right", 0.0, 0.0, 0.45),
    ("throttle_soft_left", 1.0, 0.0, -0.45),
    ("throttle_soft_right", 1.0, 0.0, 0.45),
    ("half_throttle_soft_left", 0.5, 0.0, -0.45),
    ("half_throttle_soft_right", 0.5, 0.0, 0.45),
    ("half_throttle_left", 0.5, 0.0, -1.0),
    ("half_throttle_right", 0.5, 0.0, 1.0),
    ("soft_brake", 0.0, 0.35, 0.0),
    ("soft_brake_left", 0.0, 0.35, -0.45),
    ("soft_brake_right", 0.0, 0.35, 0.45),
)

RACING_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("throttle_left", 1.0, 0.0, -1.0),
    ("throttle_soft_left", 1.0, 0.0, -0.45),
    ("throttle", 1.0, 0.0, 0.0),
    ("throttle_soft_right", 1.0, 0.0, 0.45),
    ("throttle_right", 1.0, 0.0, 1.0),
    ("half_throttle_left", 0.5, 0.0, -1.0),
    ("half_throttle_soft_left", 0.5, 0.0, -0.45),
    ("half_throttle", 0.5, 0.0, 0.0),
    ("half_throttle_soft_right", 0.5, 0.0, 0.45),
    ("half_throttle_right", 0.5, 0.0, 1.0),
    ("soft_brake_left", 0.0, 0.35, -1.0),
    ("soft_brake_soft_left", 0.0, 0.35, -0.45),
    ("soft_brake", 0.0, 0.35, 0.0),
    ("soft_brake_soft_right", 0.0, 0.35, 0.45),
    ("soft_brake_right", 0.0, 0.35, 1.0),
    ("brake_left", 0.0, 1.0, -1.0),
    ("brake_soft_left", 0.0, 1.0, -0.45),
    ("brake", 0.0, 1.0, 0.0),
    ("brake_soft_right", 0.0, 1.0, 0.45),
    ("brake_right", 0.0, 1.0, 1.0),
)

BRAKE_STRAIGHT_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("coast", 0.0, 0.0, 0.0),
    ("throttle", 1.0, 0.0, 0.0),
    ("brake", 0.0, 1.0, 0.0),
    ("left", 0.0, 0.0, -1.0),
    ("right", 0.0, 0.0, 1.0),
)

STRAIGHT_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("coast", 0.0, 0.0, 0.0),
    ("throttle", 1.0, 0.0, 0.0),
    ("brake", 0.0, 1.0, 0.0),
)

BRAKE_COAST_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("brake", 0.0, 1.0, 0.0),
    ("coast", 0.0, 0.0, 0.0),
)

BRAKE_RELEASE_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("brake", 0.0, 1.0, 0.0),
    ("trail_brake", 0.0, 0.35, 0.0),
    ("coast", 0.0, 0.0, 0.0),
)

RELEASE_BRAKE_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("coast", 0.0, 0.0, 0.0),
    ("trail_brake", 0.0, 0.35, 0.0),
    ("brake", 0.0, 1.0, 0.0),
)

TURNIN_POWER_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance", 0.22, 0.0, 0.0),
    ("maintenance_soft_left", 0.22, 0.0, -0.45),
    ("maintenance_soft_right", 0.22, 0.0, 0.45),
    ("half_throttle", 0.5, 0.0, 0.0),
    ("half_throttle_soft_left", 0.5, 0.0, -0.45),
    ("half_throttle_soft_right", 0.5, 0.0, 0.45),
    ("soft_brake", 0.0, 0.35, 0.0),
    ("soft_brake_soft_left", 0.0, 0.35, -0.45),
    ("soft_brake_soft_right", 0.0, 0.35, 0.45),
)

TURNIN_MICRO_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance", 0.22, 0.0, 0.0),
    ("maintenance_micro_left", 0.22, 0.0, -0.15),
    ("maintenance_micro_right", 0.22, 0.0, 0.15),
    ("half_throttle", 0.5, 0.0, 0.0),
    ("half_throttle_micro_left", 0.5, 0.0, -0.15),
    ("half_throttle_micro_right", 0.5, 0.0, 0.15),
    ("soft_brake", 0.0, 0.35, 0.0),
    ("soft_brake_micro_left", 0.0, 0.35, -0.15),
    ("soft_brake_micro_right", 0.0, 0.35, 0.15),
)

DELAYED_TURN_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance", 0.22, 0.0, 0.0),
    ("maintenance_left", 0.22, 0.0, -0.12),
    ("maintenance_tiny_left", 0.22, 0.0, -0.06),
    ("trail_left", 0.0, 0.12, -0.06),
    ("soft_left", 0.0, 0.25, -0.06),
    ("maintenance_tiny_right", 0.22, 0.0, 0.06),
    ("trail_right", 0.0, 0.12, 0.06),
    ("soft_brake", 0.0, 0.25, 0.0),
)

DELAYED_LEFT_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance", 0.22, 0.0, 0.0),
    ("maintenance_left", 0.22, 0.0, -0.12),
    ("maintenance_tiny_left", 0.22, 0.0, -0.06),
    ("trail_left", 0.0, 0.12, -0.06),
    ("soft_left", 0.0, 0.25, -0.06),
    ("soft_brake", 0.0, 0.25, 0.0),
)

STABILIZE_RIGHT_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance", 0.22, 0.0, 0.0),
    ("maintenance_tiny_right", 0.22, 0.0, 0.06),
    ("maintenance_micro_right", 0.22, 0.0, 0.12),
    ("trail_right", 0.0, 0.12, 0.06),
    ("soft_right", 0.0, 0.25, 0.06),
    ("soft_brake", 0.0, 0.25, 0.0),
)

EXIT_TINY_DISCRETE_ACTIONS: tuple[ActionSpec, ...] = (
    ("maintenance_tiny_left", 0.22, 0.0, -0.06),
    ("maintenance_tiny_right", 0.22, 0.0, 0.06),
    ("maintenance_micro_left", 0.22, 0.0, -0.12),
    ("maintenance_micro_right", 0.22, 0.0, 0.12),
    ("trail_brake_tiny_left", 0.0, 0.12, -0.06),
    ("trail_brake_tiny_right", 0.0, 0.12, 0.06),
    ("soft_brake_tiny_left", 0.0, 0.25, -0.06),
    ("soft_brake_tiny_right", 0.0, 0.25, 0.06),
    ("soft_brake", 0.0, 0.25, 0.0),
)

MULTIDISCRETE_DRIVE_LEVELS: tuple[DriveSpec, ...] = (
    ("brake", 0.0, 1.0),
    ("soft_brake", 0.0, 0.35),
    ("maintenance_throttle", 0.22, 0.0),
    ("half_throttle", 0.5, 0.0),
    ("throttle", 1.0, 0.0),
)

MULTIDISCRETE_STEER_LEVELS: tuple[SteerSpec, ...] = (
    ("left", -1.0),
    ("soft_left", -0.45),
    ("straight", 0.0),
    ("soft_right", 0.45),
    ("right", 1.0),
)

ACTION_SETS: dict[str, tuple[ActionSpec, ...]] = {
    "brake_coast": BRAKE_COAST_DISCRETE_ACTIONS,
    "brake_release": BRAKE_RELEASE_DISCRETE_ACTIONS,
    "brake_straight": BRAKE_STRAIGHT_DISCRETE_ACTIONS,
    "delayed_left": DELAYED_LEFT_DISCRETE_ACTIONS,
    "delayed_turn": DELAYED_TURN_DISCRETE_ACTIONS,
    "legacy": LEGACY_DISCRETE_ACTIONS,
    "release_brake": RELEASE_BRAKE_DISCRETE_ACTIONS,
    "expanded": EXPANDED_DISCRETE_ACTIONS,
    "exit_tiny": EXIT_TINY_DISCRETE_ACTIONS,
    "racing": RACING_DISCRETE_ACTIONS,
    "stabilize_right": STABILIZE_RIGHT_DISCRETE_ACTIONS,
    "straight": STRAIGHT_DISCRETE_ACTIONS,
    "turnin_micro": TURNIN_MICRO_DISCRETE_ACTIONS,
    "turnin_power": TURNIN_POWER_DISCRETE_ACTIONS,
}

DISCRETE_ACTIONS = LEGACY_DISCRETE_ACTIONS


def actions_for_action_set(action_set: str) -> tuple[ActionSpec, ...]:
    try:
        return ACTION_SETS[action_set]
    except KeyError as exc:
        valid = ", ".join(sorted(ACTION_SETS))
        raise ValueError(f"Unknown action set {action_set!r}; expected one of: {valid}") from exc


def action_to_controls(action_id: int, *, action_set: str = DEFAULT_ACTION_SET) -> tuple[float, float, float]:
    actions = actions_for_action_set(action_set)
    _, throttle, brake, steer = actions[int(action_id) % len(actions)]
    return throttle, brake, steer


def multidiscrete_action_nvec() -> tuple[int, int]:
    return (len(MULTIDISCRETE_DRIVE_LEVELS), len(MULTIDISCRETE_STEER_LEVELS))


def multidiscrete_action_to_controls(action: Any) -> tuple[float, float, float, int]:
    values = list(action)
    if len(values) < 2:
        raise ValueError("MultiDiscrete actions must contain drive and steering indices.")
    drive_index = int(values[0]) % len(MULTIDISCRETE_DRIVE_LEVELS)
    steer_index = int(values[1]) % len(MULTIDISCRETE_STEER_LEVELS)
    _, throttle, brake = MULTIDISCRETE_DRIVE_LEVELS[drive_index]
    _, steer = MULTIDISCRETE_STEER_LEVELS[steer_index]
    action_id = drive_index * len(MULTIDISCRETE_STEER_LEVELS) + steer_index
    return throttle, brake, steer, action_id
