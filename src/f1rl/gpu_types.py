# pyright: reportPrivateImportUsage=false
"""PyTorch tensor data structures for batched evolution simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from f1rl.config import CarParams
from f1rl.physics import CarState
from f1rl.state_snapshot import StateSnapshot
from f1rl.track_model import TrackSpec

FLOAT_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def torch_dtype_from_name(name: str) -> torch.dtype:
    try:
        return FLOAT_DTYPES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(FLOAT_DTYPES))
        raise ValueError(f"Unknown GPU dtype {name!r}; expected one of: {valid}") from exc


def normalize_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for GPU evolution backend, but torch.cuda.is_available() is false.")
    return resolved


@dataclass(frozen=True, slots=True)
class GpuCarParams:
    wheelbase_m: float
    max_steer_deg: float
    steer_response: float
    engine_accel_mps2: float
    brake_accel_mps2: float
    drag_coefficient: float
    rolling_resistance_mps2: float
    grip_g: float
    aero_grip_per_mps2: float
    max_grip_g: float
    max_drive_g: float
    max_brake_g: float
    steering_speed_sensitivity: float
    max_speed_mps: float
    dt: float


def gpu_car_params_from_cpu(params: CarParams) -> GpuCarParams:
    return GpuCarParams(
        wheelbase_m=float(params.wheelbase_m),
        max_steer_deg=float(params.max_steer_deg),
        steer_response=float(params.steer_response),
        engine_accel_mps2=float(params.engine_accel_mps2),
        brake_accel_mps2=float(params.brake_accel_mps2),
        drag_coefficient=float(params.drag_coefficient),
        rolling_resistance_mps2=float(params.rolling_resistance_mps2),
        grip_g=float(params.grip_g),
        aero_grip_per_mps2=float(params.aero_grip_per_mps2),
        max_grip_g=float(params.max_grip_g),
        max_drive_g=float(params.max_drive_g),
        max_brake_g=float(params.max_brake_g),
        steering_speed_sensitivity=float(params.steering_speed_sensitivity),
        max_speed_mps=float(params.max_speed_mps),
        dt=float(params.dt),
    )


@dataclass(slots=True)
class GpuCarBatch:
    x: torch.Tensor
    y: torch.Tensor
    heading_rad: torch.Tensor
    speed_mps: torch.Tensor
    yaw_rate_rps: torch.Tensor
    steering: torch.Tensor
    raw_progress_m: torch.Tensor
    monotonic_progress_m: torch.Tensor
    last_raw_progress_px: torch.Tensor
    checkpoint_index: torch.Tensor
    next_checkpoint_index: torch.Tensor
    checkpoints_passed: torch.Tensor
    missed_checkpoint_count: torch.Tensor
    lap_index: torch.Tensor
    elapsed_steps: torch.Tensor
    no_progress_steps: torch.Tensor
    alive: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    termination_reason_id: torch.Tensor
    valid_lap: torch.Tensor
    finish_crossed: torch.Tensor
    completed_lap: torch.Tensor
    segment_complete: torch.Tensor
    segment_release_observed: torch.Tensor
    last_throttle: torch.Tensor
    last_brake: torch.Tensor
    last_steer: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    def position_xy(self) -> torch.Tensor:
        return torch.stack((self.x, self.y), dim=1)


@dataclass(frozen=True, slots=True)
class GpuMovementBatch:
    x0: torch.Tensor
    y0: torch.Tensor
    x1: torch.Tensor
    y1: torch.Tensor

    def as_segments(self) -> torch.Tensor:
        return torch.stack((self.x0, self.y0, self.x1, self.y1), dim=1)


@dataclass(frozen=True, slots=True)
class GpuTrackTensors:
    centerline_xy: torch.Tensor
    centerline_next_xy: torch.Tensor
    centerline_segment_vec: torch.Tensor
    centerline_segment_len: torch.Tensor
    centerline_segment_len2: torch.Tensor
    centerline_cumdist_px: torch.Tensor
    centerline_segment_mid_px: torch.Tensor
    boundary_segments: torch.Tensor
    finish_line: torch.Tensor
    drivable_mask: torch.Tensor
    meters_per_pixel: torch.Tensor
    length_px: torch.Tensor
    length_m: torch.Tensor
    checkpoint_count: int
    checkpoint_spacing_m: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.centerline_xy.device

    @property
    def dtype(self) -> torch.dtype:
        return self.centerline_xy.dtype


def gpu_track_from_cpu(
    track: TrackSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> GpuTrackTensors:
    centerline = torch.as_tensor(track.centerline, device=device, dtype=dtype)
    centerline_s = torch.as_tensor(track.centerline_s, device=device, dtype=dtype)
    starts = centerline[:-1]
    ends = centerline[1:]
    vec = ends - starts
    len2 = torch.sum(vec * vec, dim=1).clamp_min(torch.finfo(dtype).eps)
    seg_len = torch.sqrt(len2)
    mid = (centerline_s[:-1] + centerline_s[1:]) * 0.5
    boundary_segments = torch.as_tensor(track.boundary_segments, device=device, dtype=dtype)
    finish_line = torch.as_tensor(track.finish_line.reshape(1, 4), device=device, dtype=dtype)
    drivable_mask = torch.as_tensor(track.drivable_mask, device=device, dtype=torch.bool)
    meters_per_pixel = torch.tensor(float(track.meters_per_pixel), device=device, dtype=dtype)
    length_m = torch.tensor(float(track.length_m), device=device, dtype=dtype)
    length_px = torch.tensor(float(track.length_px), device=device, dtype=dtype)
    checkpoint_count = int(len(track.checkpoints))
    checkpoint_spacing_m = length_m / max(checkpoint_count, 1)
    return GpuTrackTensors(
        centerline_xy=starts,
        centerline_next_xy=ends,
        centerline_segment_vec=vec,
        centerline_segment_len=seg_len,
        centerline_segment_len2=len2,
        centerline_cumdist_px=centerline_s[:-1],
        centerline_segment_mid_px=mid,
        boundary_segments=boundary_segments,
        finish_line=finish_line,
        drivable_mask=drivable_mask,
        meters_per_pixel=meters_per_pixel,
        length_px=length_px,
        length_m=length_m,
        checkpoint_count=checkpoint_count,
        checkpoint_spacing_m=checkpoint_spacing_m,
    )


def _float_tensor(values: list[float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def _int_tensor(values: list[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.int64)


def car_batch_from_states(
    states: list[CarState],
    *,
    last_throttle: list[float] | None = None,
    last_brake: list[float] | None = None,
    last_steer: list[float] | None = None,
    next_checkpoint_index: list[int] | None = None,
    checkpoints_passed: list[int] | None = None,
    missed_checkpoint_count: list[int] | None = None,
    valid_lap: list[bool] | None = None,
    meters_per_pixel: float = 1.0,
    device: torch.device,
    dtype: torch.dtype,
) -> GpuCarBatch:
    count = len(states)
    zeros_f = [0.0] * count
    zeros_i = [0] * count
    return GpuCarBatch(
        x=_float_tensor([state.x for state in states], device=device, dtype=dtype),
        y=_float_tensor([state.y for state in states], device=device, dtype=dtype),
        heading_rad=_float_tensor([state.heading_rad for state in states], device=device, dtype=dtype),
        speed_mps=_float_tensor([state.speed_mps for state in states], device=device, dtype=dtype),
        yaw_rate_rps=_float_tensor([state.yaw_rate_rps for state in states], device=device, dtype=dtype),
        steering=_float_tensor([state.steering for state in states], device=device, dtype=dtype),
        raw_progress_m=_float_tensor([state.raw_progress_m for state in states], device=device, dtype=dtype),
        monotonic_progress_m=_float_tensor(
            [state.monotonic_progress_m for state in states],
            device=device,
            dtype=dtype,
        ),
        last_raw_progress_px=_float_tensor(
            [state.raw_progress_m / max(meters_per_pixel, 1e-6) for state in states],
            device=device,
            dtype=dtype,
        ),
        checkpoint_index=_int_tensor([state.checkpoint_index for state in states], device=device),
        next_checkpoint_index=_int_tensor(next_checkpoint_index or [state.checkpoint_index + 1 for state in states], device=device),
        checkpoints_passed=_int_tensor(checkpoints_passed or [state.checkpoint_index for state in states], device=device),
        missed_checkpoint_count=_int_tensor(missed_checkpoint_count or zeros_i, device=device),
        lap_index=_int_tensor([state.lap_index for state in states], device=device),
        elapsed_steps=_int_tensor(zeros_i, device=device),
        no_progress_steps=_int_tensor(zeros_i, device=device),
        alive=torch.ones(count, device=device, dtype=torch.bool),
        terminated=torch.zeros(count, device=device, dtype=torch.bool),
        truncated=torch.zeros(count, device=device, dtype=torch.bool),
        termination_reason_id=_int_tensor(zeros_i, device=device),
        valid_lap=torch.tensor(valid_lap or [True] * count, device=device, dtype=torch.bool),
        finish_crossed=torch.zeros(count, device=device, dtype=torch.bool),
        completed_lap=torch.zeros(count, device=device, dtype=torch.bool),
        segment_complete=torch.zeros(count, device=device, dtype=torch.bool),
        segment_release_observed=torch.zeros(count, device=device, dtype=torch.bool),
        last_throttle=_float_tensor(last_throttle or zeros_f, device=device, dtype=dtype),
        last_brake=_float_tensor(last_brake or zeros_f, device=device, dtype=dtype),
        last_steer=_float_tensor(last_steer or zeros_f, device=device, dtype=dtype),
    )


def car_batch_from_snapshots(
    snapshots: list[StateSnapshot],
    *,
    meters_per_pixel: float,
    device: torch.device,
    dtype: torch.dtype,
) -> GpuCarBatch:
    states = [
        CarState(
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
        )
        for snapshot in snapshots
    ]
    return car_batch_from_states(
        states,
        last_throttle=[snapshot.last_throttle for snapshot in snapshots],
        last_brake=[snapshot.last_brake for snapshot in snapshots],
        last_steer=[snapshot.last_steer for snapshot in snapshots],
        next_checkpoint_index=[snapshot.next_checkpoint_index for snapshot in snapshots],
        checkpoints_passed=[snapshot.checkpoints_passed for snapshot in snapshots],
        missed_checkpoint_count=[snapshot.missed_checkpoint_count for snapshot in snapshots],
        valid_lap=[snapshot.valid_lap for snapshot in snapshots],
        meters_per_pixel=meters_per_pixel,
        device=device,
        dtype=dtype,
    )
