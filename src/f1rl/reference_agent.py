"""Fast-F1 telemetry reference agent and ghost replay baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1rl.calibration import REFERENCE_CSV, load_targets
from f1rl.config import ARTIFACTS_DIR, SimConfig
from f1rl.geometry import sample_polyline_at, wrap_radians
from f1rl.sim import MonzaSim
from f1rl.telemetry import REWARD_COMPONENT_KEYS, StepTelemetry, TelemetryWriter


@dataclass(slots=True)
class ReferenceProfile:
    time_s: np.ndarray
    distance_m: np.ndarray
    speed_kph: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    source: str
    lap_time_s: float

    @property
    def distance_max_m(self) -> float:
        return float(self.distance_m[-1])

    def speed_at(self, distance_m: float) -> float:
        wrapped = float(distance_m % self.distance_max_m)
        return float(np.interp(wrapped, self.distance_m, self.speed_kph))

    def throttle_at(self, distance_m: float) -> float:
        wrapped = float(distance_m % self.distance_max_m)
        return float(np.interp(wrapped, self.distance_m, self.throttle))

    def brake_at(self, distance_m: float) -> float:
        wrapped = float(distance_m % self.distance_max_m)
        return float(np.interp(wrapped, self.distance_m, self.brake))


def parse_timedelta_seconds(value: str) -> float:
    value = value.strip()
    if "days" in value:
        _, value = value.split("days", maxsplit=1)
    value = value.strip()
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_reference_profile(path: Path = REFERENCE_CSV) -> ReferenceProfile:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        rows.extend(csv.DictReader(file))
    if not rows:
        raise ValueError(f"reference telemetry is empty: {path}")

    time_s = np.asarray([parse_timedelta_seconds(row["Time"]) for row in rows], dtype=np.float64)
    distance_m = np.asarray([float(row["Distance"]) for row in rows], dtype=np.float64)
    speed_kph = np.asarray([float(row["Speed"]) for row in rows], dtype=np.float64)
    throttle = np.asarray([float(row["Throttle"]) / 100.0 for row in rows], dtype=np.float64)
    brake = np.asarray([1.0 if _parse_bool(row["Brake"]) else 0.0 for row in rows], dtype=np.float64)

    keep = np.r_[True, np.diff(distance_m) > 1e-6]
    targets = load_targets()
    return ReferenceProfile(
        time_s=time_s[keep],
        distance_m=distance_m[keep],
        speed_kph=speed_kph[keep],
        throttle=throttle[keep],
        brake=brake[keep],
        source=targets.source,
        lap_time_s=targets.lap_time_s,
    )


def centerline_heading_at(sim: MonzaSim, progress_m: float) -> float:
    progress_px = (progress_m / sim.track.meters_per_pixel) % sim.track.length_px
    idx = np.searchsorted(sim.track.centerline_s, progress_px, side="right") - 1
    idx = int(np.clip(idx, 0, len(sim.track.centerline_s) - 2))
    start = sim.track.centerline[idx]
    end = sim.track.centerline[idx + 1]
    return float(np.arctan2(-(end[1] - start[1]), end[0] - start[0]))


def centerline_point_at(sim: MonzaSim, progress_m: float) -> np.ndarray:
    progress_px = np.asarray([progress_m / sim.track.meters_per_pixel], dtype=np.float32)
    return sample_polyline_at(sim.track.centerline, sim.track.centerline_s, progress_px)[0]


def centerline_curvature_at(sim: MonzaSim, progress_m: float, delta_m: float = 8.0) -> float:
    prev_heading = centerline_heading_at(sim, progress_m - delta_m)
    next_heading = centerline_heading_at(sim, progress_m + delta_m)
    return wrap_radians(next_heading - prev_heading) / max(2.0 * delta_m, 1e-6)


def _empty_reward_components() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def run_reference_ghost(*, seed: int, telemetry: bool = True, profile_path: Path = REFERENCE_CSV) -> Path | None:
    profile = load_reference_profile(profile_path)
    sim = MonzaSim(SimConfig(max_steps=max(len(profile.time_s) + 10, 3600)))
    sim.reset(seed=seed)
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="reference-ghost", seed=seed) if telemetry else None
    checkpoint_spacing = sim.track.length_m / max(len(sim.track.checkpoints), 1)
    steering_curvature = np.tan(np.deg2rad(sim.config.car.max_steer_deg)) / sim.config.car.wheelbase_m
    previous_progress = 0.0

    for idx, (time_s, ref_distance_m, speed_kph, throttle, brake) in enumerate(
        zip(profile.time_s, profile.distance_m, profile.speed_kph, profile.throttle, profile.brake, strict=True)
    ):
        progress_m = min(float(ref_distance_m / profile.distance_max_m * sim.track.length_m), sim.track.length_m)
        point = centerline_point_at(sim, progress_m)
        heading = centerline_heading_at(sim, progress_m)
        curvature = centerline_curvature_at(sim, progress_m)
        steering = float(np.clip(curvature / max(steering_curvature, 1e-9), -1.0, 1.0))
        progress_delta = max(0.0, progress_m - previous_progress)
        previous_progress = progress_m

        sim.state.x = float(point[0])
        sim.state.y = float(point[1])
        sim.state.heading_rad = heading
        sim.state.speed_mps = float(speed_kph / 3.6)
        sim.state.yaw_rate_rps = float(sim.state.speed_mps * curvature)
        sim.state.steering = float(steering * np.deg2rad(sim.config.car.max_steer_deg))
        sim.state.elapsed_steps = idx
        sim.state.raw_progress_m = progress_m
        sim.state.monotonic_progress_m = progress_m
        sim.state.checkpoint_index = int((progress_m // checkpoint_spacing) % len(sim.track.checkpoints))
        sim.state.lap_index = 1 if progress_m >= sim.track.length_m else 0
        sim.completed_lap = progress_m >= sim.track.length_m
        sim.termination_reason = "lap_complete" if idx == len(profile.time_s) - 1 else "active"

        components = _empty_reward_components()
        components["progress"] = progress_delta * sim.config.reward.progress_scale
        if idx == len(profile.time_s) - 1:
            components["finish"] = sim.config.reward.finish_bonus
        reward = float(sum(components.values()))
        telemetry_row = StepTelemetry(
            step_index=idx,
            sim_time_s=float(time_s),
            x=float(sim.state.x),
            y=float(sim.state.y),
            heading_deg=float(np.rad2deg(sim.state.heading_rad)),
            speed_mps=float(sim.state.speed_mps),
            speed_kph=float(speed_kph),
            yaw_rate_rps=float(sim.state.yaw_rate_rps),
            throttle=float(throttle),
            brake=float(brake),
            steering=float(steering),
            action_id=-100,
            raw_progress_m=float(progress_m),
            monotonic_progress_m=float(progress_m),
            progress_delta_m=float(progress_delta),
            lateral_error_m=0.0,
            heading_error_deg=0.0,
            checkpoint_index=int(sim.state.checkpoint_index),
            lap_index=int(sim.state.lap_index),
            ray_distances_m=[float(v) for v in sim.ray_distances_m()],
            collided=False,
            off_track=False,
            terminated=False,
            truncated=bool(idx == len(profile.time_s) - 1),
            termination_reason=sim.termination_reason,
            reward_total=reward,
            reward_components=components,
        )
        if writer:
            writer.write_step(telemetry_row)

    if not writer:
        return None
    summary = writer.close_episode(termination_reason="lap_complete", completed_lap=True)
    print_reference_summary(profile, summary.lap_time_s or 0.0, summary.avg_speed_kph, summary.max_speed_kph, writer.root)
    return writer.root


def pure_pursuit_controls(sim: MonzaSim, profile: ReferenceProfile) -> tuple[float, float, float]:
    progress_m = sim.state.monotonic_progress_m
    speed_kph = sim.state.speed_mps * 3.6
    lookahead_m = float(np.clip(35.0 + speed_kph * 0.33, 45.0, 165.0))
    target = centerline_point_at(sim, progress_m + lookahead_m)
    dx = float(target[0] - sim.state.x)
    dy = float(target[1] - sim.state.y)
    desired = float(np.arctan2(-dy, dx))
    heading_error = wrap_radians(desired - sim.state.heading_rad)
    steer = float(np.clip(heading_error / np.deg2rad(sim.config.car.max_steer_deg), -1.0, 1.0))

    target_now = profile.speed_at(progress_m / sim.track.length_m * profile.distance_max_m)
    target_ahead = profile.speed_at((progress_m + 120.0) / sim.track.length_m * profile.distance_max_m)
    target_speed = min(target_now, target_ahead + 14.0)
    error = target_speed - speed_kph
    if error > 4.0:
        throttle = float(np.clip(error / 35.0, 0.15, 1.0))
        brake = 0.0
    elif error < -2.0:
        throttle = 0.0
        brake = float(np.clip(-error / 70.0, 0.05, 1.0))
    else:
        throttle = 0.15
        brake = 0.0
    if abs(heading_error) > 0.8 and speed_kph > 90.0:
        throttle = min(throttle, 0.2)
        brake = max(brake, 0.35)
    return throttle, brake, steer


def run_reference_control(
    *, seed: int, steps: int, telemetry: bool = True, profile_path: Path = REFERENCE_CSV
) -> Path | None:
    profile = load_reference_profile(profile_path)
    sim = MonzaSim(SimConfig(max_steps=steps, no_progress_limit_steps=600))
    sim.reset(seed=seed)
    sim.state.speed_mps = profile.speed_at(0.0) / 3.6
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="reference-control", seed=seed) if telemetry else None

    for _ in range(steps):
        throttle, brake, steer = pure_pursuit_controls(sim, profile)
        result = sim.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=-101)
        if writer:
            writer.write_step(result.telemetry)
        if result.terminated or result.truncated:
            break

    if not writer:
        return None
    summary = writer.close_episode(termination_reason=sim.termination_reason, completed_lap=sim.completed_lap)
    print_reference_summary(profile, summary.lap_time_s or summary.elapsed_time_s, summary.avg_speed_kph, summary.max_speed_kph, writer.root)
    print(f"control_result completed={summary.completed_lap} reason={summary.termination_reason}")
    return writer.root


def print_reference_summary(
    profile: ReferenceProfile, sim_lap_s: float, sim_avg_kph: float, sim_max_kph: float, root: Path
) -> None:
    print(f"reference_source={profile.source}")
    print(f"target_lap={profile.lap_time_s:.3f}s target_avg={np.mean(profile.speed_kph):.1f}kph target_max={np.max(profile.speed_kph):.1f}kph")
    print(f"sim_lap={sim_lap_s:.3f}s sim_avg={sim_avg_kph:.1f}kph sim_max={sim_max_kph:.1f}kph")
    print(f"telemetry={root}")
    print(f"replay_command=uv run f1-replay {root / 'steps.jsonl'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fast-F1 Monza reference baseline.")
    parser.add_argument("--mode", choices=("ghost", "control"), default="ghost")
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", type=Path, default=REFERENCE_CSV)
    parser.add_argument("--no-telemetry", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "ghost":
        run_reference_ghost(seed=args.seed, telemetry=not args.no_telemetry, profile_path=args.profile)
    else:
        run_reference_control(
            seed=args.seed,
            steps=args.steps,
            telemetry=not args.no_telemetry,
            profile_path=args.profile,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
