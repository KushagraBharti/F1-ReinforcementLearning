"""Fast-F1 Monza telemetry reference helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from f1rl.config import ASSETS_DIR, CarParams

REFERENCE_CSV = ASSETS_DIR / "reference" / "monza_2024_Q_VER_telemetry.csv"
REFERENCE_SUMMARY = ASSETS_DIR / "reference" / "monza_2024_Q_VER_summary.json"


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


def theoretical_terminal_speed_kph(params: CarParams) -> float:
    accel = max(params.engine_accel_mps2 - params.rolling_resistance_mps2, 0.0)
    terminal_mps = math.sqrt(accel / max(params.drag_coefficient, 1e-9))
    terminal_mps = min(terminal_mps, params.max_speed_mps)
    return terminal_mps * 3.6


def straight_line_speed_after(params: CarParams, seconds: float) -> float:
    speed = 0.0
    steps = int(seconds / params.dt)
    for _ in range(steps):
        speed += params.engine_accel_mps2 * params.dt
        speed -= params.rolling_resistance_mps2 * params.dt
        speed -= params.drag_coefficient * speed * speed * params.dt
        speed = float(np.clip(speed, 0.0, params.max_speed_mps))
    return speed * 3.6


def braking_distance(params: CarParams, *, from_kph: float, to_kph: float) -> float:
    speed = from_kph / 3.6
    target = to_kph / 3.6
    distance = 0.0
    while speed > target:
        distance += speed * params.dt
        speed -= params.brake_accel_mps2 * params.dt
        speed -= params.rolling_resistance_mps2 * params.dt
        speed -= params.drag_coefficient * speed * speed * params.dt
        speed = max(0.0, speed)
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
    steering_curvature = math.tan(math.radians(params.max_steer_deg)) / max(params.wheelbase_m, 1e-9)
    grip_curvature = params.grip_g * 9.81 / max(speed_mps * speed_mps, 1e-9)
    max_curvature = min(steering_curvature, grip_curvature)
    radius = 1.0 / max(max_curvature, 1e-9)
    lateral_g = speed_mps * speed_mps * max_curvature / 9.81
    return {
        "speed_kph": speed_kph,
        "max_curvature_rad_per_m": max_curvature,
        "min_radius_m": radius,
        "lateral_g_at_limit": lateral_g,
    }


def calibration_report(params: CarParams | None = None) -> dict:
    params = params or CarParams()
    targets = load_targets()
    return {
        "targets": asdict(targets),
        "sim_car_params": asdict(params),
        "sim_estimates": {
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
