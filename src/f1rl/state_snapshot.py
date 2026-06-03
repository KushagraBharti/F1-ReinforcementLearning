"""Serializable simulator state snapshots for curriculum starts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import radians
from typing import Any

import numpy as np

from f1rl.config import CarParams
from f1rl.geometry import wrap_radians
from f1rl.physics import CarState

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    id: str
    source: str
    source_file: str | None
    step_index: int
    sim_time_s: float
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    yaw_rate_rps: float
    steering_rad: float
    raw_progress_m: float
    monotonic_progress_m: float
    checkpoint_index: int
    next_checkpoint_index: int
    checkpoints_passed: int
    missed_checkpoint_count: int
    lap_index: int
    valid_lap: bool
    finish_crossed: bool
    completed_lap: bool
    segment_complete: bool
    last_throttle: float
    last_brake: float
    last_steer: float
    last_action_id: int
    curriculum_stage: str | None = None
    segment_target_progress_m: float | None = None
    schema_version: int = SNAPSHOT_SCHEMA_VERSION


def snapshot_to_dict(snapshot: StateSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def snapshot_from_mapping(value: StateSnapshot | Mapping[str, Any]) -> StateSnapshot:
    if isinstance(value, StateSnapshot):
        return value
    data = dict(value)
    heading_rad = float(data.get("heading_rad", radians(float(data.get("heading_deg", 0.0)))))
    speed_mps = float(data.get("speed_mps", float(data.get("speed_kph", 0.0)) / 3.6))
    raw_progress_m = float(data.get("raw_progress_m", data.get("monotonic_progress_m", 0.0)))
    monotonic_progress_m = float(data.get("monotonic_progress_m", raw_progress_m))
    step_index = int(data.get("step_index", 0))
    source = str(data.get("source", "unknown"))
    snapshot_id = str(data.get("id", f"{source}:{step_index}:{monotonic_progress_m:.3f}"))
    steering_rad = data.get("steering_rad")
    last_steer = float(data.get("last_steer", data.get("steering", 0.0)))
    if steering_rad is None:
        steering_rad = float(np.clip(last_steer, -1.0, 1.0)) * float(np.deg2rad(CarParams().max_steer_deg))
    return StateSnapshot(
        id=snapshot_id,
        source=source,
        source_file=data.get("source_file"),
        step_index=step_index,
        sim_time_s=float(data.get("sim_time_s", 0.0)),
        x=float(data["x"]),
        y=float(data["y"]),
        heading_rad=wrap_radians(heading_rad),
        speed_mps=max(speed_mps, 0.0),
        yaw_rate_rps=float(data.get("yaw_rate_rps", 0.0)),
        steering_rad=float(steering_rad),
        raw_progress_m=raw_progress_m,
        monotonic_progress_m=monotonic_progress_m,
        checkpoint_index=int(data.get("checkpoint_index", 0)),
        next_checkpoint_index=int(data.get("next_checkpoint_index", data.get("checkpoint_index", 0) + 1)),
        checkpoints_passed=int(data.get("checkpoints_passed", data.get("checkpoint_index", 0))),
        missed_checkpoint_count=int(data.get("missed_checkpoint_count", 0)),
        lap_index=int(data.get("lap_index", 0)),
        valid_lap=bool(data.get("valid_lap", True)),
        finish_crossed=bool(data.get("finish_crossed", False)),
        completed_lap=bool(data.get("completed_lap", data.get("termination_reason") == "lap_complete")),
        segment_complete=bool(data.get("segment_complete", False)),
        last_throttle=float(data.get("last_throttle", data.get("throttle", 0.0))),
        last_brake=float(data.get("last_brake", data.get("brake", 0.0))),
        last_steer=last_steer,
        last_action_id=int(data.get("last_action_id", data.get("action_id", 0))),
        curriculum_stage=data.get("curriculum_stage"),
        segment_target_progress_m=data.get("segment_target_progress_m"),
        schema_version=int(data.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),
    )


def snapshot_to_car_state(snapshot: StateSnapshot) -> CarState:
    return CarState(
        x=snapshot.x,
        y=snapshot.y,
        heading_rad=snapshot.heading_rad,
        speed_mps=snapshot.speed_mps,
        yaw_rate_rps=snapshot.yaw_rate_rps,
        steering=snapshot.steering_rad,
        lap_index=snapshot.lap_index,
        checkpoint_index=snapshot.checkpoint_index,
        raw_progress_m=snapshot.raw_progress_m,
        monotonic_progress_m=snapshot.monotonic_progress_m,
        elapsed_steps=0,
        alive=True,
    )


def snapshot_from_sim(sim: Any, *, source: str, source_file: str | None = None) -> StateSnapshot:
    progress_m = float(sim.state.monotonic_progress_m)
    return StateSnapshot(
        id=f"{source}:{int(sim.state.elapsed_steps):06d}:{progress_m:.3f}",
        source=source,
        source_file=source_file,
        step_index=int(sim.state.elapsed_steps),
        sim_time_s=float(sim.state.elapsed_steps * sim.config.car.dt),
        x=float(sim.state.x),
        y=float(sim.state.y),
        heading_rad=float(sim.state.heading_rad),
        speed_mps=float(sim.state.speed_mps),
        yaw_rate_rps=float(sim.state.yaw_rate_rps),
        steering_rad=float(sim.state.steering),
        raw_progress_m=float(sim.state.raw_progress_m),
        monotonic_progress_m=progress_m,
        checkpoint_index=int(sim.state.checkpoint_index),
        next_checkpoint_index=int(sim.next_checkpoint_index),
        checkpoints_passed=int(sim.checkpoints_passed),
        missed_checkpoint_count=int(sim.missed_checkpoint_count),
        lap_index=int(sim.state.lap_index),
        valid_lap=bool(sim.valid_lap),
        finish_crossed=bool(sim.finish_crossed),
        completed_lap=bool(sim.completed_lap),
        segment_complete=bool(sim.segment_complete),
        last_throttle=float(sim.last_throttle),
        last_brake=float(sim.last_brake),
        last_steer=float(sim.last_steer),
        last_action_id=int(sim.last_action_id),
        curriculum_stage=sim.curriculum_stage,
        segment_target_progress_m=sim.segment_target_progress_m,
    )


def snapshot_from_telemetry_row(row: Mapping[str, Any], *, source_file: str | None = None) -> StateSnapshot:
    data = dict(row)
    data["source"] = data.get("source", "telemetry")
    data["source_file"] = source_file
    return snapshot_from_mapping(data)
