"""Fast-F1 Monza telemetry reference helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import ASSETS_DIR, CarParams, PhysicsV2Params, SimConfig
from f1rl.geometry import sample_polyline_at, wrap_radians
from f1rl.physics import (
    CarState,
    apply_physics,
    gear_and_rpm_v2,
    grip_limit_g,
    mechanical_grip_scale_v2,
    tire_lateral_force_v2,
    weight_transfer_v2,
)
from f1rl.sim import MonzaSim

REFERENCE_CSV = ASSETS_DIR / "reference" / "monza_2024_Q_VER_telemetry.csv"
REFERENCE_SUMMARY = ASSETS_DIR / "reference" / "monza_2024_Q_VER_summary.json"
SPEED_BIN_EDGES_KPH = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 330.0, 360.0)


@dataclass(slots=True)
class CalibrationTargets:
    source: str
    lap_time_s: float
    distance_m: float
    min_speed_kph: float
    max_speed_kph: float
    mean_speed_kph: float
    p10_speed_kph: float
    p50_speed_kph: float
    p90_speed_kph: float
    curvature_abs_p90_rad_per_m: float
    curvature_abs_p95_rad_per_m: float
    radius_p05_m: float
    radius_p10_m: float
    lateral_g_p90: float
    lateral_g_p95: float


def _parse_timedelta_seconds(value: str) -> float:
    value = value.strip()
    if "days" in value:
        _, value = value.split("days", maxsplit=1)
    value = value.strip()
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _finite_quantile(values: np.ndarray, quantile: float, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.quantile(finite, quantile))


def _optional_quantile(values: np.ndarray, quantile: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, quantile))


def _optional_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def load_targets(path: Path = REFERENCE_SUMMARY) -> CalibrationTargets:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationTargets(
        source=str(data["source"]),
        lap_time_s=float(data["lap_time_s"]),
        distance_m=float(data["distance_m"]),
        min_speed_kph=float(data["min_speed_kph"]),
        max_speed_kph=float(data["max_speed_kph"]),
        mean_speed_kph=float(data["mean_speed_kph"]),
        p10_speed_kph=float(data["p10_speed_kph"]),
        p50_speed_kph=float(data["p50_speed_kph"]),
        p90_speed_kph=float(data["p90_speed_kph"]),
        curvature_abs_p90_rad_per_m=float(data["curvature_abs_p90_rad_per_m"]),
        curvature_abs_p95_rad_per_m=float(data["curvature_abs_p95_rad_per_m"]),
        radius_p05_m=float(data["radius_p05_m"]),
        radius_p10_m=float(data["radius_p10_m"]),
        lateral_g_p90=float(data["lateral_g_p90"]),
        lateral_g_p95=float(data["lateral_g_p95"]),
    )


def load_reference_telemetry(csv_path: Path = REFERENCE_CSV) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as file:
        rows.extend(csv.DictReader(file))
    if not rows:
        raise ValueError(f"reference telemetry is empty: {csv_path}")

    time_s = np.asarray([_parse_timedelta_seconds(row["Time"]) for row in rows], dtype=np.float64)
    distance_m = np.asarray([float(row["Distance"]) for row in rows], dtype=np.float64)
    x = np.asarray([float(row["X"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["Y"]) for row in rows], dtype=np.float64)
    speed_kph = np.asarray([float(row["Speed"]) for row in rows], dtype=np.float64)
    rpm = np.asarray([float(row["RPM"]) for row in rows], dtype=np.float64)
    gear = np.asarray([int(float(row["nGear"])) for row in rows], dtype=np.int16)
    throttle = np.asarray([float(row["Throttle"]) / 100.0 for row in rows], dtype=np.float64)
    brake = np.asarray([1.0 if _parse_bool(row["Brake"]) else 0.0 for row in rows], dtype=np.float64)
    drs = np.asarray([int(float(row["DRS"])) for row in rows], dtype=np.int16)

    keep = np.r_[True, np.diff(distance_m) > 1e-6]
    return {
        "time_s": time_s[keep],
        "distance_m": distance_m[keep],
        "x": x[keep],
        "y": y[keep],
        "speed_kph": speed_kph[keep],
        "rpm": rpm[keep],
        "gear": gear[keep],
        "throttle": throttle[keep],
        "brake": brake[keep],
        "drs": drs[keep],
    }


def _curvature_arrays(
    distance_m: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed_kph: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed_mps = speed_kph / 3.6
    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    smoothing_kernel = np.ones(9, dtype=np.float64) / 9.0
    smoothed_heading = np.convolve(heading, smoothing_kernel, mode="same")
    ds = np.gradient(distance_m)
    dtheta = np.gradient(smoothed_heading)
    curvature = np.divide(dtheta, ds, out=np.zeros_like(dtheta), where=np.abs(ds) > 1e-6)
    abs_curvature = np.abs(curvature)
    radius = np.divide(
        1.0,
        abs_curvature,
        out=np.full_like(abs_curvature, np.inf),
        where=abs_curvature > 1e-6,
    )
    lateral_g = speed_mps * speed_mps * abs_curvature / 9.81
    return abs_curvature, radius, lateral_g


def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges


def _interp(distance_m: np.ndarray, values: np.ndarray, target_distance_m: float) -> float:
    clamped = float(np.clip(target_distance_m, float(distance_m[0]), float(distance_m[-1])))
    return float(np.interp(clamped, distance_m, values))


def _speed_trace_samples(
    telemetry: dict[str, np.ndarray],
    *,
    sample_spacing_m: float,
) -> list[dict[str, float | int]]:
    distance_m = telemetry["distance_m"]
    speed_kph = telemetry["speed_kph"]
    throttle = telemetry["throttle"]
    brake = telemetry["brake"]
    rpm = telemetry["rpm"]
    gear = telemetry["gear"].astype(np.float64)
    max_distance = float(distance_m[-1])
    sample_distances = [
        float(distance)
        for distance in np.arange(0.0, max_distance, sample_spacing_m, dtype=np.float64)
    ]
    if not sample_distances or sample_distances[-1] < max_distance:
        sample_distances.append(max_distance)
    return [
        {
            "distance_m": float(distance),
            "speed_kph": _interp(distance_m, speed_kph, float(distance)),
            "throttle": _interp(distance_m, throttle, float(distance)),
            "brake": _interp(distance_m, brake, float(distance)),
            "gear": int(round(_interp(distance_m, gear, float(distance)))),
            "rpm": _interp(distance_m, rpm, float(distance)),
        }
        for distance in sample_distances
    ]


def _braking_zones(telemetry: dict[str, np.ndarray]) -> list[dict[str, float | int]]:
    distance_m = telemetry["distance_m"]
    time_s = telemetry["time_s"]
    speed_kph = telemetry["speed_kph"]
    brake = telemetry["brake"] > 0.5
    zones: list[dict[str, float | int]] = []
    for start, end in _contiguous_true_ranges(brake):
        start_distance = float(distance_m[start])
        end_distance = float(distance_m[end])
        length = end_distance - start_distance
        zone_speed = speed_kph[start : end + 1]
        if zone_speed.size == 0:
            continue
        local_min_offset = int(np.argmin(zone_speed))
        min_index = start + local_min_offset
        start_speed = float(speed_kph[start])
        min_speed = float(speed_kph[min_index])
        speed_drop = start_speed - min_speed
        if length < 10.0 and speed_drop < 8.0:
            continue
        min_time_delta = max(float(time_s[min_index] - time_s[start]), 1e-9)
        mean_decel = ((start_speed - min_speed) / 3.6) / min_time_delta
        zones.append(
            {
                "index": len(zones),
                "start_distance_m": start_distance,
                "end_distance_m": end_distance,
                "length_m": length,
                "start_speed_kph": start_speed,
                "end_speed_kph": float(speed_kph[end]),
                "min_speed_kph": min_speed,
                "min_speed_distance_m": float(distance_m[min_index]),
                "speed_drop_kph": speed_drop,
                "mean_decel_mps2": mean_decel,
            }
        )
    return zones


def _corner_speed_targets(
    telemetry: dict[str, np.ndarray],
    lateral_g: np.ndarray,
    abs_curvature: np.ndarray,
) -> list[dict[str, float | int | str]]:
    distance_m = telemetry["distance_m"]
    speed_kph = telemetry["speed_kph"]
    low_speed_threshold = min(240.0, float(np.quantile(speed_kph, 0.35)))
    low_speed_mask = speed_kph <= low_speed_threshold
    corners: list[dict[str, float | int | str]] = []
    for start, end in _contiguous_true_ranges(low_speed_mask):
        start_distance = float(distance_m[start])
        end_distance = float(distance_m[end])
        if end_distance - start_distance < 25.0:
            continue
        zone_speed = speed_kph[start : end + 1]
        min_index = start + int(np.argmin(zone_speed))
        if float(speed_kph[min_index]) > 235.0:
            continue
        min_distance = float(distance_m[min_index])
        entry_distance = max(float(distance_m[0]), min_distance - 100.0)
        exit_distance = min(float(distance_m[-1]), min_distance + 150.0)
        zone_lateral_g = lateral_g[start : end + 1]
        zone_curvature = abs_curvature[start : end + 1]
        corners.append(
            {
                "index": len(corners),
                "name": f"corner_{len(corners) + 1:02d}",
                "zone_start_distance_m": start_distance,
                "zone_end_distance_m": end_distance,
                "min_speed_distance_m": min_distance,
                "entry_speed_kph": _interp(distance_m, speed_kph, entry_distance),
                "min_speed_kph": float(speed_kph[min_index]),
                "exit_speed_kph": _interp(distance_m, speed_kph, exit_distance),
                "lateral_g_at_min_speed": float(lateral_g[min_index]),
                "lateral_g_p50": _finite_quantile(zone_lateral_g, 0.50),
                "lateral_g_p75": _finite_quantile(zone_lateral_g, 0.75),
                "lateral_g_p90": _finite_quantile(zone_lateral_g, 0.90),
                "lateral_g_max": float(np.max(zone_lateral_g[np.isfinite(zone_lateral_g)]))
                if bool(np.any(np.isfinite(zone_lateral_g)))
                else 0.0,
                "curvature_abs_at_min_speed_rad_per_m": float(abs_curvature[min_index]),
                "curvature_abs_p90_rad_per_m": _finite_quantile(zone_curvature, 0.90),
            }
        )
    return corners


def _sustained_corner_targets(
    telemetry: dict[str, np.ndarray],
    lateral_g: np.ndarray,
    abs_curvature: np.ndarray,
    radius: np.ndarray,
) -> list[dict[str, float | int | str]]:
    distance_m = telemetry["distance_m"]
    speed_kph = telemetry["speed_kph"]
    valid = (
        np.isfinite(radius)
        & (np.gradient(distance_m) > 0.0)
        & (distance_m > 50.0)
        & (distance_m < float(distance_m.max() - 50.0))
    )
    curvature_threshold = float(np.quantile(abs_curvature[valid], 0.70)) if bool(np.any(valid)) else 0.0
    sustained_mask = valid & (speed_kph > 180.0) & (abs_curvature > curvature_threshold)
    corners: list[dict[str, float | int | str]] = []
    for start, end in _contiguous_true_ranges(sustained_mask):
        start_distance = float(distance_m[start])
        end_distance = float(distance_m[end])
        length_m = end_distance - start_distance
        if length_m < 80.0:
            continue
        zone_speed = speed_kph[start : end + 1]
        zone_lateral_g = lateral_g[start : end + 1]
        zone_curvature = abs_curvature[start : end + 1]
        zone_radius = radius[start : end + 1]
        corners.append(
            {
                "index": len(corners),
                "name": f"sustained_corner_{len(corners) + 1:02d}",
                "start_distance_m": start_distance,
                "end_distance_m": end_distance,
                "length_m": length_m,
                "mean_speed_kph": float(np.mean(zone_speed)),
                "min_speed_kph": float(np.min(zone_speed)),
                "max_speed_kph": float(np.max(zone_speed)),
                "lateral_g_p50": _finite_quantile(zone_lateral_g, 0.50),
                "lateral_g_p75": _finite_quantile(zone_lateral_g, 0.75),
                "lateral_g_p90": _finite_quantile(zone_lateral_g, 0.90),
                "lateral_g_max": float(np.max(zone_lateral_g[np.isfinite(zone_lateral_g)]))
                if bool(np.any(np.isfinite(zone_lateral_g)))
                else 0.0,
                "curvature_abs_p90_rad_per_m": _finite_quantile(zone_curvature, 0.90),
                "radius_p10_m": _finite_quantile(zone_radius, 0.10),
            }
        )
    return corners


def _gear_rpm_summary(telemetry: dict[str, np.ndarray]) -> dict[str, Any]:
    distance_m = telemetry["distance_m"]
    speed_kph = telemetry["speed_kph"]
    rpm = telemetry["rpm"]
    gear = telemetry["gear"]
    gears: dict[str, dict[str, float | int]] = {}
    for gear_id in sorted({int(value) for value in gear}):
        mask = gear == gear_id
        if not bool(np.any(mask)):
            continue
        gears[str(gear_id)] = {
            "samples": int(np.count_nonzero(mask)),
            "min_speed_kph": float(np.min(speed_kph[mask])),
            "median_speed_kph": float(np.median(speed_kph[mask])),
            "max_speed_kph": float(np.max(speed_kph[mask])),
            "min_rpm": float(np.min(rpm[mask])),
            "mean_rpm": float(np.mean(rpm[mask])),
            "max_rpm": float(np.max(rpm[mask])),
        }
    shift_indices = np.flatnonzero(np.diff(gear) != 0) + 1
    shift_points = [
        {
            "distance_m": float(distance_m[index]),
            "speed_kph": float(speed_kph[index]),
            "rpm": float(rpm[index]),
            "from_gear": int(gear[index - 1]),
            "to_gear": int(gear[index]),
        }
        for index in shift_indices
    ]
    return {
        "min_gear": int(np.min(gear)),
        "max_gear": int(np.max(gear)),
        "gears": gears,
        "shift_points": shift_points,
    }


def _acceleration_summary(telemetry: dict[str, np.ndarray]) -> dict[str, float]:
    throttle = telemetry["throttle"]
    brake = telemetry["brake"]
    accel = _longitudinal_acceleration(telemetry)
    positive = accel[(accel > 0.0) & (throttle > 0.95) & (brake < 0.5)]
    braking = -accel[(accel < 0.0) & (brake > 0.5)]
    return {
        "positive_accel_p95_mps2": _finite_quantile(positive, 0.95),
        "positive_accel_max_mps2": float(np.max(positive)) if positive.size else 0.0,
        "braking_decel_p95_mps2": _finite_quantile(braking, 0.95),
        "braking_decel_max_mps2": float(np.max(braking)) if braking.size else 0.0,
    }


def _longitudinal_acceleration(telemetry: dict[str, np.ndarray]) -> np.ndarray:
    time_s = telemetry["time_s"]
    speed_mps = telemetry["speed_kph"] / 3.6
    dt = np.gradient(time_s)
    return np.divide(
        np.gradient(speed_mps),
        dt,
        out=np.zeros_like(speed_mps),
        where=np.abs(dt) > 1e-9,
    )


def _longitudinal_reference_bins(telemetry: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    speed_kph = telemetry["speed_kph"]
    throttle = telemetry["throttle"]
    brake = telemetry["brake"]
    accel = _longitudinal_acceleration(telemetry)
    rows: list[dict[str, Any]] = []
    for low_kph, high_kph in zip(SPEED_BIN_EDGES_KPH[:-1], SPEED_BIN_EDGES_KPH[1:], strict=True):
        speed_mask = (speed_kph >= low_kph) & (speed_kph < high_kph) & np.isfinite(accel)
        throttle_mask = speed_mask & (throttle > 0.95) & (brake < 0.5) & (accel > 0.0)
        braking_mask = speed_mask & (brake > 0.5) & (accel < 0.0)
        throttle_accel = accel[throttle_mask]
        braking_decel = -accel[braking_mask]
        rows.append(
            {
                "speed_min_kph": low_kph,
                "speed_max_kph": high_kph,
                "samples": int(np.count_nonzero(speed_mask)),
                "mean_speed_kph": _optional_mean(speed_kph[speed_mask]),
                "full_throttle_samples": int(np.count_nonzero(throttle_mask)),
                "full_throttle_accel_p50_mps2": _optional_quantile(throttle_accel, 0.50),
                "full_throttle_accel_p90_mps2": _optional_quantile(throttle_accel, 0.90),
                "full_throttle_accel_p95_mps2": _optional_quantile(throttle_accel, 0.95),
                "braking_samples": int(np.count_nonzero(braking_mask)),
                "braking_decel_p50_mps2": _optional_quantile(braking_decel, 0.50),
                "braking_decel_p90_mps2": _optional_quantile(braking_decel, 0.90),
                "braking_decel_p95_mps2": _optional_quantile(braking_decel, 0.95),
            }
        )
    return rows


def reference_trace_features(
    csv_path: Path = REFERENCE_CSV,
    *,
    sample_spacing_m: float = 250.0,
) -> dict[str, Any]:
    telemetry = load_reference_telemetry(csv_path)
    distance_m = telemetry["distance_m"]
    speed_kph = telemetry["speed_kph"]
    abs_curvature, radius, lateral_g = _curvature_arrays(
        distance_m,
        telemetry["x"],
        telemetry["y"],
        speed_kph,
    )
    mask = (
        np.isfinite(radius)
        & (np.gradient(distance_m) > 0.0)
        & (distance_m > 50.0)
        & (distance_m < float(distance_m.max() - 50.0))
    )
    return {
        "rows": int(len(distance_m)),
        "lap_time_s": float(telemetry["time_s"][-1]),
        "distance_m": float(distance_m[-1]),
        "speed_trace_samples": _speed_trace_samples(telemetry, sample_spacing_m=sample_spacing_m),
        "braking_zones": _braking_zones(telemetry),
        "corner_speed_targets": _corner_speed_targets(telemetry, lateral_g, abs_curvature),
        "sustained_corner_targets": _sustained_corner_targets(telemetry, lateral_g, abs_curvature, radius),
        "acceleration": _acceleration_summary(telemetry),
        "longitudinal_by_speed_bin": _longitudinal_reference_bins(telemetry),
        "gear_rpm": _gear_rpm_summary(telemetry),
        "curvature_lateral_g": {
            "curvature_abs_p90_rad_per_m": float(np.quantile(abs_curvature[mask], 0.90)),
            "curvature_abs_p95_rad_per_m": float(np.quantile(abs_curvature[mask], 0.95)),
            "radius_p05_m": float(np.quantile(radius[mask], 0.05)),
            "radius_p10_m": float(np.quantile(radius[mask], 0.10)),
            "lateral_g_p90": float(np.quantile(lateral_g[mask], 0.90)),
            "lateral_g_p95": float(np.quantile(lateral_g[mask], 0.95)),
        },
    }


def reference_summary_from_csv(
    csv_path: Path = REFERENCE_CSV,
    *,
    source: str = "Fast-F1 2024 Italian Grand Prix Qualifying VER fastest lap",
) -> dict[str, Any]:
    telemetry = load_reference_telemetry(csv_path)
    features = reference_trace_features(csv_path)
    turning = features["curvature_lateral_g"]
    speed_kph = telemetry["speed_kph"]
    return {
        "source": source,
        "rows": features["rows"],
        "lap_time_s": features["lap_time_s"],
        "distance_m": features["distance_m"],
        "min_speed_kph": float(np.min(speed_kph)),
        "max_speed_kph": float(np.max(speed_kph)),
        "mean_speed_kph": float(np.mean(speed_kph)),
        "p10_speed_kph": float(np.quantile(speed_kph, 0.10)),
        "p50_speed_kph": float(np.quantile(speed_kph, 0.50)),
        "p90_speed_kph": float(np.quantile(speed_kph, 0.90)),
        **turning,
        "trace_features": features,
    }


def theoretical_terminal_speed_kph(params: CarParams) -> float:
    accel = max(min(params.engine_accel_mps2, params.max_drive_g * 9.81) - params.rolling_resistance_mps2, 0.0)
    terminal_mps = math.sqrt(accel / max(params.drag_coefficient, 1e-9))
    terminal_mps = min(terminal_mps, params.max_speed_mps)
    return terminal_mps * 3.6


def straight_line_speed_after(params: CarParams, seconds: float) -> float:
    speed = 0.0
    steps = int(seconds / params.dt)
    for _ in range(steps):
        speed += min(params.engine_accel_mps2, params.max_drive_g * 9.81) * params.dt
        speed -= params.rolling_resistance_mps2 * params.dt
        speed -= params.drag_coefficient * speed * speed * params.dt
        speed = float(np.clip(speed, 0.0, params.max_speed_mps))
    return speed * 3.6


def straight_line_speed_after_model(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    seconds: float,
) -> float:
    state = CarState(x=0.0, y=0.0, heading_rad=0.0)
    steps = int(seconds / params.dt)
    for _ in range(steps):
        state, _ = apply_physics(
            state,
            throttle=1.0,
            brake=0.0,
            steer=0.0,
            params=params,
            meters_per_pixel=1.0,
            physics_model=physics_model,
            physics_v2=physics_v2,
        )
    return state.speed_mps * 3.6


def braking_distance(params: CarParams, *, from_kph: float, to_kph: float) -> float:
    speed = from_kph / 3.6
    target = to_kph / 3.6
    distance = 0.0
    while speed > target:
        distance += speed * params.dt
        speed -= min(params.brake_accel_mps2, params.max_brake_g * 9.81) * params.dt
        speed -= params.rolling_resistance_mps2 * params.dt
        speed -= params.drag_coefficient * speed * speed * params.dt
        speed = max(0.0, speed)
        if distance > 2000.0:
            break
    return distance


def braking_distance_model(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    from_kph: float,
    to_kph: float,
) -> float:
    state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=from_kph / 3.6)
    distance = 0.0
    target = to_kph / 3.6
    while state.speed_mps > target:
        previous_speed = state.speed_mps
        state, _ = apply_physics(
            state,
            throttle=0.0,
            brake=1.0,
            steer=0.0,
            params=params,
            meters_per_pixel=1.0,
            physics_model=physics_model,
            physics_v2=physics_v2,
        )
        distance += previous_speed * params.dt
        if distance > 2000.0:
            break
    return distance


def compute_turning_targets(csv_path: Path = REFERENCE_CSV) -> dict[str, float]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows.extend(reader)
    distance = np.asarray([float(row["Distance"]) for row in rows], dtype=np.float64)
    x = np.asarray([float(row["X"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["Y"]) for row in rows], dtype=np.float64)
    speed_mps = np.asarray([float(row["Speed"]) / 3.6 for row in rows], dtype=np.float64)

    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    smoothing_kernel = np.ones(9, dtype=np.float64) / 9.0
    smoothed_heading = np.convolve(heading, smoothing_kernel, mode="same")
    ds = np.gradient(distance)
    dtheta = np.gradient(smoothed_heading)
    curvature = np.divide(dtheta, ds, out=np.zeros_like(dtheta), where=np.abs(ds) > 1e-6)
    abs_curvature = np.abs(curvature)
    radius = np.divide(
        1.0,
        abs_curvature,
        out=np.full_like(abs_curvature, np.inf),
        where=abs_curvature > 1e-6,
    )
    lateral_g = speed_mps * speed_mps * abs_curvature / 9.81
    mask = (
        np.isfinite(radius)
        & (ds > 0.0)
        & (distance > 50.0)
        & (distance < float(distance.max() - 50.0))
    )
    return {
        "curvature_abs_p90_rad_per_m": float(np.quantile(abs_curvature[mask], 0.90)),
        "curvature_abs_p95_rad_per_m": float(np.quantile(abs_curvature[mask], 0.95)),
        "radius_p05_m": float(np.quantile(radius[mask], 0.05)),
        "radius_p10_m": float(np.quantile(radius[mask], 0.10)),
        "lateral_g_p90": float(np.quantile(lateral_g[mask], 0.90)),
        "lateral_g_p95": float(np.quantile(lateral_g[mask], 0.95)),
    }


def steering_limited_radius(params: CarParams) -> float:
    steer_rad = math.radians(params.max_steer_deg)
    return params.wheelbase_m / max(math.tan(steer_rad), 1e-9)


def cornering_capacity(params: CarParams, speed_kph: float) -> dict[str, float]:
    speed_mps = speed_kph / 3.6
    effective_steer = math.radians(params.max_steer_deg) / (1.0 + params.steering_speed_sensitivity * speed_mps * speed_mps)
    steering_curvature = math.tan(effective_steer) / max(params.wheelbase_m, 1e-9)
    grip_g = grip_limit_g(params, speed_mps)
    grip_curvature = grip_g * 9.81 / max(speed_mps * speed_mps, 1e-9)
    max_curvature = min(steering_curvature, grip_curvature)
    radius = 1.0 / max(max_curvature, 1e-9)
    lateral_g = speed_mps * speed_mps * max_curvature / 9.81
    return {
        "speed_kph": speed_kph,
        "max_curvature_rad_per_m": max_curvature,
        "min_radius_m": radius,
        "lateral_g_at_limit": lateral_g,
        "available_grip_g": grip_g,
    }


def cornering_capacity_v2(params: CarParams, v2: PhysicsV2Params, speed_kph: float) -> dict[str, float]:
    speed_mps = speed_kph / 3.6
    front_load, rear_load = weight_transfer_v2(
        params=params,
        v2=v2,
        longitudinal_accel_mps2=0.0,
        lateral_accel_mps2=0.0,
        speed_mps=speed_mps,
    )
    reference_front = params.mass * 9.81 * v2.front_weight_distribution
    reference_rear = params.mass * 9.81 - reference_front
    slip_peak = math.radians(v2.slip_angle_peak_deg)
    grip_scale = mechanical_grip_scale_v2(speed_mps, v2)
    front_force = abs(
        tire_lateral_force_v2(
            slip_peak,
            front_load,
            stiffness_n_per_rad=v2.front_cornering_stiffness_n_per_rad,
            peak_mu=v2.front_peak_mu * grip_scale,
            shape_c=v2.tire_shape_c,
            slip_angle_peak_rad=slip_peak,
            post_peak_falloff=v2.post_peak_falloff,
            load_sensitivity=v2.load_sensitivity,
            reference_load_n=reference_front,
            surface_mu=v2.surface_mu,
        )
    )
    rear_force = abs(
        tire_lateral_force_v2(
            slip_peak,
            rear_load,
            stiffness_n_per_rad=v2.rear_cornering_stiffness_n_per_rad,
            peak_mu=v2.rear_peak_mu * grip_scale,
            shape_c=v2.tire_shape_c,
            slip_angle_peak_rad=slip_peak,
            post_peak_falloff=v2.post_peak_falloff,
            load_sensitivity=v2.load_sensitivity,
            reference_load_n=reference_rear,
            surface_mu=v2.surface_mu,
        )
    )
    lateral_accel = (front_force + rear_force) / max(params.mass, 1e-9)
    max_curvature = lateral_accel / max(speed_mps * speed_mps, 1e-9)
    return {
        "speed_kph": speed_kph,
        "max_curvature_rad_per_m": max_curvature,
        "min_radius_m": 1.0 / max(max_curvature, 1e-9),
        "lateral_g_at_limit": lateral_accel / 9.81,
        "front_load_n": front_load,
        "rear_load_n": rear_load,
    }


def _accel_brake_envelope_model(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
) -> dict[str, Any]:
    speed_grid_kph = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 330.0)
    samples: list[dict[str, float]] = []
    for speed_kph in speed_grid_kph:
        speed_mps = speed_kph / 3.6
        throttle_state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=speed_mps)
        next_throttle, _ = apply_physics(
            throttle_state,
            throttle=1.0,
            brake=0.0,
            steer=0.0,
            params=params,
            meters_per_pixel=1.0,
            physics_model=physics_model,
            physics_v2=physics_v2,
        )
        brake_state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=speed_mps)
        next_brake, _ = apply_physics(
            brake_state,
            throttle=0.0,
            brake=1.0,
            steer=0.0,
            params=params,
            meters_per_pixel=1.0,
            physics_model=physics_model,
            physics_v2=physics_v2,
        )
        samples.append(
            {
                "speed_kph": speed_kph,
                "full_throttle_accel_mps2": (next_throttle.speed_mps - speed_mps) / params.dt,
                "full_brake_decel_mps2": (speed_mps - next_brake.speed_mps) / params.dt,
            }
        )
    accel_values = np.asarray([sample["full_throttle_accel_mps2"] for sample in samples], dtype=np.float64)
    decel_values = np.asarray([sample["full_brake_decel_mps2"] for sample in samples], dtype=np.float64)
    return {
        "samples": samples,
        "max_full_throttle_accel_mps2": float(np.max(accel_values)),
        "p95_full_throttle_accel_mps2": float(np.quantile(accel_values, 0.95)),
        "max_full_brake_decel_mps2": float(np.max(decel_values)),
        "p95_full_brake_decel_mps2": float(np.quantile(decel_values, 0.95)),
    }


def _gear_rpm_model_profile(v2: PhysicsV2Params) -> list[dict[str, float | int]]:
    profile: list[dict[str, float | int]] = []
    for speed_kph in (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 330.0, 350.0):
        gear, rpm = gear_and_rpm_v2(speed_kph / 3.6, 1, v2)
        profile.append({"speed_kph": speed_kph, "gear": gear, "rpm": rpm})
    return profile


def _gear_rpm_reference_comparison(
    reference_features: dict[str, Any],
    v2: PhysicsV2Params,
) -> dict[str, Any]:
    rows: list[dict[str, float | int | str]] = []
    for reference_gear, target in reference_features["gear_rpm"]["gears"].items():
        target_speed = float(target["median_speed_kph"])
        target_rpm = float(target["mean_rpm"])
        model_gear, model_rpm = gear_and_rpm_v2(target_speed / 3.6, 1, v2)
        rows.append(
            {
                "reference_gear": reference_gear,
                "reference_median_speed_kph": target_speed,
                "reference_mean_rpm": target_rpm,
                "model_gear": model_gear,
                "model_rpm": model_rpm,
                "gear_delta": model_gear - int(reference_gear),
                "rpm_error": model_rpm - target_rpm,
            }
        )
    if not rows:
        return {"rows": [], "gear_match_rate": 0.0, "mean_abs_rpm_error": 0.0}
    gear_matches = sum(1 for row in rows if int(row["gear_delta"]) == 0)
    mean_abs_rpm_error = sum(abs(float(row["rpm_error"])) for row in rows) / len(rows)
    return {
        "rows": rows,
        "gear_match_rate": gear_matches / len(rows),
        "mean_abs_rpm_error": mean_abs_rpm_error,
    }


def _longitudinal_accel_model_at_speed(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    speed_kph: float,
    throttle: float,
    brake: float,
) -> float:
    speed_mps = speed_kph / 3.6
    state = CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=speed_mps)
    next_state, _ = apply_physics(
        state,
        throttle=throttle,
        brake=brake,
        steer=0.0,
        params=params,
        meters_per_pixel=1.0,
        physics_model=physics_model,
        physics_v2=physics_v2,
    )
    return (next_state.speed_mps - speed_mps) / params.dt


def _mean_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(abs(value) for value in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(value) for value in values))


def _speed_trace_feasibility_comparison(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    reference_features: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    accel_errors: list[float] = []
    brake_errors: list[float] = []
    accel_deficits: list[float] = []
    accel_excesses: list[float] = []
    brake_deficits: list[float] = []
    brake_excesses: list[float] = []
    for reference_bin in reference_features["longitudinal_by_speed_bin"]:
        low_kph = float(reference_bin["speed_min_kph"])
        high_kph = float(reference_bin["speed_max_kph"])
        compare_speed = reference_bin["mean_speed_kph"]
        if compare_speed is None:
            compare_speed = (low_kph + high_kph) * 0.5
        compare_speed = float(compare_speed)
        model_accel = _longitudinal_accel_model_at_speed(
            params,
            physics_model=physics_model,
            physics_v2=physics_v2,
            speed_kph=compare_speed,
            throttle=1.0,
            brake=0.0,
        )
        model_brake_decel = -_longitudinal_accel_model_at_speed(
            params,
            physics_model=physics_model,
            physics_v2=physics_v2,
            speed_kph=compare_speed,
            throttle=0.0,
            brake=1.0,
        )
        reference_accel_p95 = reference_bin["full_throttle_accel_p95_mps2"]
        reference_brake_p95 = reference_bin["braking_decel_p95_mps2"]
        accel_error = None
        brake_error = None
        if reference_accel_p95 is not None:
            accel_error = model_accel - float(reference_accel_p95)
            accel_errors.append(accel_error)
            accel_deficits.append(max(-accel_error, 0.0))
            accel_excesses.append(max(accel_error, 0.0))
        if reference_brake_p95 is not None:
            brake_error = model_brake_decel - float(reference_brake_p95)
            brake_errors.append(brake_error)
            brake_deficits.append(max(-brake_error, 0.0))
            brake_excesses.append(max(brake_error, 0.0))
        rows.append(
            {
                "speed_min_kph": low_kph,
                "speed_max_kph": high_kph,
                "compare_speed_kph": compare_speed,
                "reference_full_throttle_samples": int(reference_bin["full_throttle_samples"]),
                "reference_full_throttle_accel_p95_mps2": reference_accel_p95,
                "model_full_throttle_accel_mps2": model_accel,
                "full_throttle_accel_p95_error_mps2": accel_error,
                "reference_braking_samples": int(reference_bin["braking_samples"]),
                "reference_braking_decel_p95_mps2": reference_brake_p95,
                "model_full_brake_decel_mps2": model_brake_decel,
                "full_brake_decel_p95_error_mps2": brake_error,
            }
        )
    return {
        "reference_lap_time_s": float(reference_features["lap_time_s"]),
        "reference_distance_m": float(reference_features["distance_m"]),
        "bins": rows,
        "mean_abs_full_throttle_accel_p95_error_mps2": _mean_abs(accel_errors),
        "max_abs_full_throttle_accel_p95_error_mps2": _max_abs(accel_errors),
        "max_full_throttle_accel_deficit_mps2": max(accel_deficits) if accel_deficits else 0.0,
        "max_full_throttle_accel_excess_mps2": max(accel_excesses) if accel_excesses else 0.0,
        "mean_abs_full_brake_decel_p95_error_mps2": _mean_abs(brake_errors),
        "max_abs_full_brake_decel_p95_error_mps2": _max_abs(brake_errors),
        "max_full_brake_decel_deficit_mps2": max(brake_deficits) if brake_deficits else 0.0,
        "max_full_brake_decel_excess_mps2": max(brake_excesses) if brake_excesses else 0.0,
    }


def _braking_zone_comparison(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    reference_features: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    for zone in reference_features["braking_zones"]:
        start_speed = float(zone["start_speed_kph"])
        min_speed = float(zone["min_speed_kph"])
        if start_speed <= min_speed + 1.0:
            continue
        reference_distance = max(0.0, float(zone["min_speed_distance_m"]) - float(zone["start_distance_m"]))
        model_distance = braking_distance_model(
            params,
            physics_model=physics_model,
            physics_v2=physics_v2,
            from_kph=start_speed,
            to_kph=min_speed,
        )
        rows.append(
            {
                "index": int(zone["index"]),
                "reference_start_speed_kph": start_speed,
                "reference_min_speed_kph": min_speed,
                "reference_distance_to_min_m": reference_distance,
                "model_full_brake_distance_to_min_m": model_distance,
                "distance_error_m": model_distance - reference_distance,
            }
        )
    if not rows:
        return {"rows": [], "mean_abs_distance_error_m": 0.0, "max_abs_distance_error_m": 0.0}
    abs_errors = [abs(float(row["distance_error_m"])) for row in rows]
    return {
        "rows": rows,
        "mean_abs_distance_error_m": sum(abs_errors) / len(abs_errors),
        "max_abs_distance_error_m": max(abs_errors),
    }


def _corner_capacity_comparison(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    reference_features: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, float | int | str]] = []
    for target in reference_features["corner_speed_targets"]:
        speed_kph = float(target["min_speed_kph"])
        if physics_model == "v2":
            capacity = cornering_capacity_v2(params, physics_v2 or PhysicsV2Params(), speed_kph)
        else:
            capacity = cornering_capacity(params, speed_kph)
        target_lateral_g = float(target["lateral_g_at_min_speed"])
        robust_target_lateral_g = float(target.get("lateral_g_p75", target_lateral_g))
        rows.append(
            {
                "name": str(target["name"]),
                "min_speed_kph": speed_kph,
                "reference_lateral_g": target_lateral_g,
                "reference_lateral_g_p75": robust_target_lateral_g,
                "reference_lateral_g_p90": float(target.get("lateral_g_p90", target_lateral_g)),
                "reference_lateral_g_max": float(target.get("lateral_g_max", target_lateral_g)),
                "model_lateral_g_at_limit": float(capacity["lateral_g_at_limit"]),
                "lateral_g_margin": float(capacity["lateral_g_at_limit"]) - target_lateral_g,
                "robust_lateral_g_margin": float(capacity["lateral_g_at_limit"]) - robust_target_lateral_g,
                "reference_curvature_rad_per_m": float(target["curvature_abs_at_min_speed_rad_per_m"]),
                "reference_curvature_p90_rad_per_m": float(
                    target.get("curvature_abs_p90_rad_per_m", target["curvature_abs_at_min_speed_rad_per_m"])
                ),
                "model_max_curvature_rad_per_m": float(capacity["max_curvature_rad_per_m"]),
            }
        )
    if not rows:
        return {
            "rows": [],
            "min_lateral_g_margin": 0.0,
            "mean_lateral_g_margin": 0.0,
            "min_robust_lateral_g_margin": 0.0,
            "mean_robust_lateral_g_margin": 0.0,
        }
    margins = [float(row["lateral_g_margin"]) for row in rows]
    robust_margins = [float(row["robust_lateral_g_margin"]) for row in rows]
    return {
        "rows": rows,
        "min_lateral_g_margin": min(margins),
        "mean_lateral_g_margin": sum(margins) / len(margins),
        "min_robust_lateral_g_margin": min(robust_margins),
        "mean_robust_lateral_g_margin": sum(robust_margins) / len(robust_margins),
    }


def _sim_centerline_point_at(sim: MonzaSim, progress_m: float) -> np.ndarray:
    progress_px = np.asarray([progress_m / sim.track.meters_per_pixel], dtype=np.float32)
    return sample_polyline_at(sim.track.centerline, sim.track.centerline_s, progress_px)[0]


def _sim_progress_from_reference_distance(sim: MonzaSim, reference_distance_max_m: float, distance_m: float) -> float:
    return float(distance_m / max(reference_distance_max_m, 1e-9) * sim.track.length_m)


def _reference_speed_at_sim_progress(
    *,
    sim: MonzaSim,
    telemetry: dict[str, np.ndarray],
    progress_m: float,
) -> float:
    reference_distance = progress_m / max(sim.track.length_m, 1e-9) * float(telemetry["distance_m"][-1])
    return _interp(telemetry["distance_m"], telemetry["speed_kph"], reference_distance)


def _controlled_section_run(
    *,
    sim: MonzaSim,
    telemetry: dict[str, np.ndarray],
    section_start_progress_m: float,
    section_end_progress_m: float,
    start_progress_m: float,
    stop_progress_m: float,
    start_speed_kph: float,
    speed_mode: str,
    fixed_speed_kph: float | None,
) -> dict[str, Any]:
    sim.reset(seed=11, options={"start_progress_m": start_progress_m, "start_speed_kph": start_speed_kph})
    max_steer_deg = (
        float(sim.config.physics_v2.max_steer_deg)
        if sim.config.physics_model == "v2"
        else float(sim.config.car.max_steer_deg)
    )
    max_steer_rad = max(math.radians(max_steer_deg), 1e-9)
    section_steps = []
    for _ in range(1400):
        progress_m = float(sim.state.monotonic_progress_m)
        speed_kph = float(sim.state.speed_mps * 3.6)
        lookahead_m = float(np.clip(15.0 + speed_kph * 0.05, 20.0, 110.0))
        target = _sim_centerline_point_at(sim, progress_m + lookahead_m)
        dx = float(target[0] - sim.state.x)
        dy = float(target[1] - sim.state.y)
        desired_heading = math.atan2(-dy, dx)
        heading_error = wrap_radians(desired_heading - sim.state.heading_rad)
        steer = float(np.clip(2.0 * heading_error / max_steer_rad, -1.0, 1.0))
        if speed_mode == "reference":
            target_speed_kph = _reference_speed_at_sim_progress(
                sim=sim,
                telemetry=telemetry,
                progress_m=progress_m,
            )
        elif fixed_speed_kph is not None:
            target_speed_kph = fixed_speed_kph
        else:
            raise ValueError("fixed_speed_kph is required for fixed speed diagnostics.")
        speed_error_kph = target_speed_kph - speed_kph
        throttle = float(np.clip(speed_error_kph / 32.0, 0.0, 1.0)) if speed_error_kph > 1.5 else 0.02
        brake = float(np.clip(-speed_error_kph / 28.0, 0.0, 1.0)) if speed_error_kph < -1.0 else 0.0
        result = sim.step_controls(
            throttle=throttle,
            brake=brake,
            steer=steer,
            action_id=-333,
            collect_observation=False,
            collect_rays=False,
            compute_reward=False,
        )
        if section_start_progress_m <= progress_m <= section_end_progress_m:
            section_steps.append(result.telemetry)
        if progress_m >= stop_progress_m or result.terminated or result.truncated:
            break
    if not section_steps:
        return {
            "speed_mode": speed_mode,
            "target_speed_kph": fixed_speed_kph,
            "steps": 0,
            "control_pass": False,
            "termination_reason": sim.termination_reason,
        }
    abs_lateral_error_m = np.asarray([abs(step.lateral_error_m) for step in section_steps], dtype=np.float64)
    abs_heading_error_deg = np.asarray([abs(step.heading_error_deg) for step in section_steps], dtype=np.float64)
    abs_lateral_g = np.asarray([abs(step.lateral_g) for step in section_steps], dtype=np.float64)
    abs_steering = np.asarray([abs(step.steering) for step in section_steps], dtype=np.float64)
    speed_kph = np.asarray([step.speed_kph for step in section_steps], dtype=np.float64)
    fixed_target_speed_kph = float(fixed_speed_kph) if fixed_speed_kph is not None else 0.0
    target_speed_kph = np.asarray(
        [
            _reference_speed_at_sim_progress(sim=sim, telemetry=telemetry, progress_m=step.monotonic_progress_m)
            if speed_mode == "reference"
            else fixed_target_speed_kph
            for step in section_steps
        ],
        dtype=np.float64,
    )
    curvature = np.asarray([abs(step.curvature_rad_per_m) for step in section_steps], dtype=np.float64)
    front_slip = np.asarray([abs(step.front_slip_angle_deg or 0.0) for step in section_steps], dtype=np.float64)
    rear_slip = np.asarray([abs(step.rear_slip_angle_deg or 0.0) for step in section_steps], dtype=np.float64)
    front_force = np.asarray([abs(step.front_lateral_force_n or 0.0) for step in section_steps], dtype=np.float64)
    rear_force = np.asarray([abs(step.rear_lateral_force_n or 0.0) for step in section_steps], dtype=np.float64)
    tire_saturation = np.asarray([step.tire_saturation or 0.0 for step in section_steps], dtype=np.float64)
    throttle = np.asarray([step.throttle for step in section_steps], dtype=np.float64)
    brake = np.asarray([step.brake for step in section_steps], dtype=np.float64)
    off_track_rate = float(np.mean([step.off_track for step in section_steps]))
    p95_lateral_error_m = _finite_quantile(abs_lateral_error_m, 0.95)
    return {
        "speed_mode": speed_mode,
        "target_speed_kph": fixed_speed_kph,
        "steps": len(section_steps),
        "mean_speed_kph": float(np.mean(speed_kph)),
        "mean_target_speed_kph": float(np.mean(target_speed_kph)),
        "mean_speed_error_kph": float(np.mean(speed_kph - target_speed_kph)),
        "p95_abs_lateral_error_m": p95_lateral_error_m,
        "max_abs_lateral_error_m": float(np.max(abs_lateral_error_m)),
        "p95_abs_heading_error_deg": _finite_quantile(abs_heading_error_deg, 0.95),
        "steering_saturation_rate": float(np.mean(abs_steering >= 0.98)),
        "mean_abs_steering": float(np.mean(abs_steering)),
        "curvature_abs_p90_rad_per_m": _finite_quantile(curvature, 0.90),
        "lateral_g_p90": _finite_quantile(abs_lateral_g, 0.90),
        "lateral_g_max": float(np.max(abs_lateral_g)),
        "front_slip_angle_p90_deg": _finite_quantile(front_slip, 0.90),
        "rear_slip_angle_p90_deg": _finite_quantile(rear_slip, 0.90),
        "front_lateral_force_p90_n": _finite_quantile(front_force, 0.90),
        "rear_lateral_force_p90_n": _finite_quantile(rear_force, 0.90),
        "tire_saturation_p90": _finite_quantile(tire_saturation, 0.90),
        "mean_throttle": float(np.mean(throttle)),
        "mean_brake": float(np.mean(brake)),
        "off_track_rate": off_track_rate,
        "terminated": bool(section_steps[-1].terminated),
        "termination_reason": section_steps[-1].termination_reason,
        "control_pass": bool(off_track_rate <= 0.0 and p95_lateral_error_m <= 4.0),
    }


def _sustained_corner_diagnostic(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    reference_features: dict[str, Any],
    telemetry: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reference_distance_max_m = float(telemetry["distance_m"][-1])
    sim = MonzaSim(
        SimConfig(
            car=params,
            physics_model=physics_model,
            physics_v2=physics_v2 or PhysicsV2Params(),
            max_steps=2000,
            checkpoint_lateral_limit_m=80.0,
            no_progress_limit_steps=2000,
        )
    )
    controlled_speeds_kph = (150.0, 180.0, 200.0, 220.0, 230.0, 240.0, 250.0)
    for target in reference_features.get("sustained_corner_targets", []):
        start_ref_m = float(target["start_distance_m"])
        end_ref_m = float(target["end_distance_m"])
        section_start_progress_m = _sim_progress_from_reference_distance(sim, reference_distance_max_m, start_ref_m)
        section_end_progress_m = _sim_progress_from_reference_distance(sim, reference_distance_max_m, end_ref_m)
        start_progress_m = _sim_progress_from_reference_distance(
            sim,
            reference_distance_max_m,
            max(start_ref_m - 120.0, 0.0),
        )
        stop_progress_m = _sim_progress_from_reference_distance(
            sim,
            reference_distance_max_m,
            min(end_ref_m + 60.0, reference_distance_max_m),
        )
        reference_speed_start_kph = _interp(telemetry["distance_m"], telemetry["speed_kph"], max(start_ref_m - 120.0, 0.0))
        reference_run = _controlled_section_run(
            sim=sim,
            telemetry=telemetry,
            section_start_progress_m=section_start_progress_m,
            section_end_progress_m=section_end_progress_m,
            start_progress_m=start_progress_m,
            stop_progress_m=stop_progress_m,
            start_speed_kph=reference_speed_start_kph,
            speed_mode="reference",
            fixed_speed_kph=None,
        )
        controlled_runs = [
            _controlled_section_run(
                sim=sim,
                telemetry=telemetry,
                section_start_progress_m=section_start_progress_m,
                section_end_progress_m=section_end_progress_m,
                start_progress_m=start_progress_m,
                stop_progress_m=stop_progress_m,
                start_speed_kph=speed_kph,
                speed_mode="fixed",
                fixed_speed_kph=speed_kph,
            )
            for speed_kph in controlled_speeds_kph
        ]
        rows.append(
            {
                "name": str(target["name"]),
                "reference_start_distance_m": start_ref_m,
                "reference_end_distance_m": end_ref_m,
                "sim_start_progress_m": section_start_progress_m,
                "sim_end_progress_m": section_end_progress_m,
                "reference_length_m": float(target["length_m"]),
                "reference_mean_speed_kph": float(target["mean_speed_kph"]),
                "reference_min_speed_kph": float(target["min_speed_kph"]),
                "reference_max_speed_kph": float(target["max_speed_kph"]),
                "reference_curvature_abs_p90_rad_per_m": float(target["curvature_abs_p90_rad_per_m"]),
                "reference_radius_p10_m": float(target["radius_p10_m"]),
                "reference_lateral_g_p75": float(target["lateral_g_p75"]),
                "reference_lateral_g_p90": float(target["lateral_g_p90"]),
                "reference_lateral_g_max": float(target["lateral_g_max"]),
                "reference_run": reference_run,
                "controlled_speed_runs": controlled_runs,
            }
        )
    reference_runs = [row["reference_run"] for row in rows]
    if not reference_runs:
        return {
            "rows": [],
            "controlled_speeds_kph": list(controlled_speeds_kph),
            "reference_control_pass_rate": 0.0,
            "min_reference_lateral_g_p75_margin": 0.0,
            "min_reference_lateral_g_p90_margin": 0.0,
            "max_reference_p95_abs_lateral_error_m": 0.0,
            "max_reference_abs_mean_speed_error_kph": 0.0,
        }
    return {
        "rows": rows,
        "controlled_speeds_kph": list(controlled_speeds_kph),
        "reference_control_pass_rate": sum(1 for row in reference_runs if bool(row.get("control_pass"))) / len(reference_runs),
        "min_reference_lateral_g_p75_margin": min(
            float(row["reference_run"]["lateral_g_p90"]) - float(row["reference_lateral_g_p75"])
            for row in rows
        ),
        "min_reference_lateral_g_p90_margin": min(
            float(row["reference_run"]["lateral_g_p90"]) - float(row["reference_lateral_g_p90"])
            for row in rows
        ),
        "max_reference_p95_abs_lateral_error_m": max(
            float(row.get("p95_abs_lateral_error_m", 0.0)) for row in reference_runs
        ),
        "max_reference_abs_mean_speed_error_kph": max(
            abs(float(row.get("mean_speed_error_kph", 0.0))) for row in reference_runs
        ),
    }


def _model_estimates(
    params: CarParams,
    *,
    physics_model: str,
    physics_v2: PhysicsV2Params | None,
    reference_features: dict[str, Any] | None = None,
    reference_telemetry: dict[str, np.ndarray] | None = None,
) -> dict:
    if physics_model == "v1":
        estimates: dict[str, Any] = {
            "terminal_speed_kph": theoretical_terminal_speed_kph(params),
            "speed_after_5s_full_throttle_kph": straight_line_speed_after(params, 5.0),
            "speed_after_8s_full_throttle_kph": straight_line_speed_after(params, 8.0),
            "braking_330_to_100_kph_m": braking_distance(params, from_kph=330.0, to_kph=100.0),
            "braking_330_to_150_kph_m": braking_distance(params, from_kph=330.0, to_kph=150.0),
            "steering_limited_radius_m": steering_limited_radius(params),
            "cornering_capacity": [
                cornering_capacity(params, speed)
                for speed in (100.0, 150.0, 200.0, 250.0, 300.0)
            ],
        }
        estimates["acceleration_envelope"] = _accel_brake_envelope_model(
            params,
            physics_model="v1",
            physics_v2=None,
        )
        if reference_features is not None:
            estimates["braking_zone_comparison"] = _braking_zone_comparison(
                params,
                physics_model="v1",
                physics_v2=None,
                reference_features=reference_features,
            )
            estimates["corner_capacity_comparison"] = _corner_capacity_comparison(
                params,
                physics_model="v1",
                physics_v2=None,
                reference_features=reference_features,
            )
            estimates["speed_trace_feasibility"] = _speed_trace_feasibility_comparison(
                params,
                physics_model="v1",
                physics_v2=None,
                reference_features=reference_features,
            )
            if reference_telemetry is not None:
                estimates["sustained_corner_diagnostic"] = _sustained_corner_diagnostic(
                    params,
                    physics_model="v1",
                    physics_v2=None,
                    reference_features=reference_features,
                    telemetry=reference_telemetry,
                )
        return estimates
    v2 = physics_v2 or PhysicsV2Params()
    estimates = {
        "terminal_speed_kph": straight_line_speed_after_model(
            params,
            physics_model="v2",
            physics_v2=v2,
            seconds=24.0,
        ),
        "speed_after_5s_full_throttle_kph": straight_line_speed_after_model(
            params,
            physics_model="v2",
            physics_v2=v2,
            seconds=5.0,
        ),
        "speed_after_8s_full_throttle_kph": straight_line_speed_after_model(
            params,
            physics_model="v2",
            physics_v2=v2,
            seconds=8.0,
        ),
        "braking_330_to_100_kph_m": braking_distance_model(
            params,
            physics_model="v2",
            physics_v2=v2,
            from_kph=330.0,
            to_kph=100.0,
        ),
        "braking_330_to_150_kph_m": braking_distance_model(
            params,
            physics_model="v2",
            physics_v2=v2,
            from_kph=330.0,
            to_kph=150.0,
        ),
        "steering_limited_radius_m": steering_limited_radius(params),
        "cornering_capacity": [
            cornering_capacity_v2(params, v2, speed)
            for speed in (100.0, 150.0, 200.0, 250.0, 300.0)
        ],
        "calibration_id": v2.calibration_id,
        "physics_version": v2.version,
    }
    estimates["acceleration_envelope"] = _accel_brake_envelope_model(
        params,
        physics_model="v2",
        physics_v2=v2,
    )
    estimates["gear_rpm_profile"] = _gear_rpm_model_profile(v2)
    if reference_features is not None:
        estimates["braking_zone_comparison"] = _braking_zone_comparison(
            params,
            physics_model="v2",
            physics_v2=v2,
            reference_features=reference_features,
        )
        estimates["corner_capacity_comparison"] = _corner_capacity_comparison(
            params,
            physics_model="v2",
            physics_v2=v2,
            reference_features=reference_features,
        )
        estimates["gear_rpm_reference_comparison"] = _gear_rpm_reference_comparison(reference_features, v2)
        estimates["speed_trace_feasibility"] = _speed_trace_feasibility_comparison(
            params,
            physics_model="v2",
            physics_v2=v2,
            reference_features=reference_features,
        )
        if reference_telemetry is not None:
            estimates["sustained_corner_diagnostic"] = _sustained_corner_diagnostic(
                params,
                physics_model="v2",
                physics_v2=v2,
                reference_features=reference_features,
                telemetry=reference_telemetry,
            )
    return estimates


def _calibration_errors(
    targets: CalibrationTargets,
    estimates: dict,
    *,
    reference_features: dict[str, Any] | None = None,
) -> dict[str, float]:
    corner_300 = estimates["cornering_capacity"][-1]
    errors = {
        "max_speed_error_kph": float(estimates["terminal_speed_kph"] - targets.max_speed_kph),
        "mean_speed_proxy_error_kph": float(estimates["speed_after_8s_full_throttle_kph"] - targets.mean_speed_kph),
        "lateral_g_p95_proxy_error": float(corner_300["lateral_g_at_limit"] - targets.lateral_g_p95),
        "radius_p05_proxy_error_m": float(corner_300["min_radius_m"] - targets.radius_p05_m),
        "braking_330_to_150_m": float(estimates["braking_330_to_150_kph_m"]),
        "braking_330_to_100_m": float(estimates["braking_330_to_100_kph_m"]),
    }
    if reference_features is not None:
        acceleration = reference_features["acceleration"]
        envelope = estimates["acceleration_envelope"]
        errors["acceleration_p95_error_mps2"] = float(
            envelope["p95_full_throttle_accel_mps2"] - acceleration["positive_accel_p95_mps2"]
        )
        errors["braking_decel_p95_error_mps2"] = float(
            envelope["p95_full_brake_decel_mps2"] - acceleration["braking_decel_p95_mps2"]
        )
    if "braking_zone_comparison" in estimates:
        braking = estimates["braking_zone_comparison"]
        errors["mean_abs_braking_zone_distance_error_m"] = float(braking["mean_abs_distance_error_m"])
        errors["max_abs_braking_zone_distance_error_m"] = float(braking["max_abs_distance_error_m"])
    if "corner_capacity_comparison" in estimates:
        cornering = estimates["corner_capacity_comparison"]
        errors["min_corner_lateral_g_margin"] = float(cornering["min_lateral_g_margin"])
        errors["mean_corner_lateral_g_margin"] = float(cornering["mean_lateral_g_margin"])
        errors["min_robust_corner_lateral_g_margin"] = float(cornering["min_robust_lateral_g_margin"])
        errors["mean_robust_corner_lateral_g_margin"] = float(cornering["mean_robust_lateral_g_margin"])
    if "gear_rpm_reference_comparison" in estimates:
        gear_rpm = estimates["gear_rpm_reference_comparison"]
        errors["gear_match_rate"] = float(gear_rpm["gear_match_rate"])
        errors["mean_abs_rpm_error"] = float(gear_rpm["mean_abs_rpm_error"])
    if "speed_trace_feasibility" in estimates:
        speed_trace = estimates["speed_trace_feasibility"]
        errors["speed_trace_accel_p95_mae_mps2"] = float(
            speed_trace["mean_abs_full_throttle_accel_p95_error_mps2"]
        )
        errors["speed_trace_accel_p95_max_abs_error_mps2"] = float(
            speed_trace["max_abs_full_throttle_accel_p95_error_mps2"]
        )
        errors["speed_trace_accel_p95_max_deficit_mps2"] = float(
            speed_trace["max_full_throttle_accel_deficit_mps2"]
        )
        errors["speed_trace_accel_p95_max_excess_mps2"] = float(
            speed_trace["max_full_throttle_accel_excess_mps2"]
        )
        errors["speed_trace_brake_p95_mae_mps2"] = float(
            speed_trace["mean_abs_full_brake_decel_p95_error_mps2"]
        )
        errors["speed_trace_brake_p95_max_abs_error_mps2"] = float(
            speed_trace["max_abs_full_brake_decel_p95_error_mps2"]
        )
        errors["speed_trace_brake_p95_max_deficit_mps2"] = float(
            speed_trace["max_full_brake_decel_deficit_mps2"]
        )
        errors["speed_trace_brake_p95_max_excess_mps2"] = float(
            speed_trace["max_full_brake_decel_excess_mps2"]
        )
    if "sustained_corner_diagnostic" in estimates:
        sustained = estimates["sustained_corner_diagnostic"]
        errors["sustained_corner_reference_control_pass_rate"] = float(
            sustained["reference_control_pass_rate"]
        )
        errors["min_sustained_corner_reference_lateral_g_p75_margin"] = float(
            sustained["min_reference_lateral_g_p75_margin"]
        )
        errors["min_sustained_corner_reference_lateral_g_p90_margin"] = float(
            sustained["min_reference_lateral_g_p90_margin"]
        )
        errors["max_sustained_corner_reference_p95_abs_lateral_error_m"] = float(
            sustained["max_reference_p95_abs_lateral_error_m"]
        )
        errors["max_sustained_corner_reference_abs_mean_speed_error_kph"] = float(
            sustained["max_reference_abs_mean_speed_error_kph"]
        )
    return errors


def calibration_report(params: CarParams | None = None, physics_v2: PhysicsV2Params | None = None) -> dict:
    params = params or CarParams()
    physics_v2 = physics_v2 or PhysicsV2Params()
    targets = load_targets()
    reference_telemetry = load_reference_telemetry()
    trace_features = reference_trace_features()
    v1_estimates = _model_estimates(
        params,
        physics_model="v1",
        physics_v2=None,
        reference_features=trace_features,
        reference_telemetry=reference_telemetry,
    )
    v2_estimates = _model_estimates(
        params,
        physics_model="v2",
        physics_v2=physics_v2,
        reference_features=trace_features,
        reference_telemetry=reference_telemetry,
    )
    return {
        "targets": asdict(targets),
        "reference_trace_features": trace_features,
        "sim_car_params": asdict(params),
        "sim_estimates": v1_estimates,
        "physics_models": {
            "v1": {
                "physics_model": "v1",
                "physics_version": "physics_v1.0.0",
                "estimates": v1_estimates,
                "error_terms": _calibration_errors(
                    targets,
                    v1_estimates,
                    reference_features=trace_features,
                ),
            },
            "v2": {
                "physics_model": "v2",
                "physics_version": physics_v2.version,
                "physics_calibration_id": physics_v2.calibration_id,
                "params": asdict(physics_v2),
                "estimates": v2_estimates,
                "error_terms": _calibration_errors(
                    targets,
                    v2_estimates,
                    reference_features=trace_features,
                ),
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Fast-F1 Monza calibration targets and simulator estimates.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = calibration_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        targets = report["targets"]
        estimates = report["sim_estimates"]
        v2 = report["physics_models"]["v2"]
        v2_estimates = v2["estimates"]
        v2_errors = v2["error_terms"]
        trace_features = report["reference_trace_features"]
        print(f"reference: {targets['source']}")
        print(
            f"target lap={targets['lap_time_s']:.3f}s "
            f"max={targets['max_speed_kph']:.1f}kph mean={targets['mean_speed_kph']:.1f}kph"
        )
        print(
            f"target turn p95={targets['lateral_g_p95']:.2f}g "
            f"radius p05={targets['radius_p05_m']:.1f}m "
            f"curv p95={targets['curvature_abs_p95_rad_per_m']:.4f}rad/m"
        )
        print(
            f"reference zones braking={len(trace_features['braking_zones'])} "
            f"corners={len(trace_features['corner_speed_targets'])} "
            f"gear_shifts={len(trace_features['gear_rpm']['shift_points'])}"
        )
        print(
            f"sim terminal={estimates['terminal_speed_kph']:.1f}kph "
            f"5s={estimates['speed_after_5s_full_throttle_kph']:.1f}kph "
            f"8s={estimates['speed_after_8s_full_throttle_kph']:.1f}kph"
        )
        print(
            f"sim braking 330->150={estimates['braking_330_to_150_kph_m']:.1f}m "
            f"330->100={estimates['braking_330_to_100_kph_m']:.1f}m"
        )
        print(f"sim steering-limited radius={estimates['steering_limited_radius_m']:.1f}m")
        for row in estimates["cornering_capacity"]:
            print(
                f"sim corner {row['speed_kph']:.0f}kph: "
                f"radius={row['min_radius_m']:.1f}m "
                f"limit={row['lateral_g_at_limit']:.2f}g"
            )
        print(
            f"v2 calibration={v2['physics_calibration_id']} "
            f"terminal={v2_estimates['terminal_speed_kph']:.1f}kph "
            f"max_speed_error={v2_errors['max_speed_error_kph']:.1f}kph "
            f"trace_accel_mae={v2_errors['speed_trace_accel_p95_mae_mps2']:.2f}m/s^2 "
            f"brake_zone_mae={v2_errors['mean_abs_braking_zone_distance_error_m']:.1f}m "
            f"robust_corner_min_margin={v2_errors['min_robust_corner_lateral_g_margin']:.2f}g "
            f"gear_match={v2_errors['gear_match_rate']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
