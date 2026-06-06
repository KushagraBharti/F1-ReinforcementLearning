# pyright: reportPrivateImportUsage=false
"""PyTorch tensor data structures for batched evolution simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from f1rl.config import CarParams, PhysicsV2Params
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
    physics_model: str
    mass: float
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
    v2_calibration_id: str
    v2_version: str
    v2_max_steer_deg: float
    v2_steer_response: float
    v2_steering_speed_sensitivity: float
    v2_front_weight_distribution: float
    v2_cg_height_m: float
    v2_track_width_m: float
    v2_front_axle_distance_m: float
    v2_rear_axle_distance_m: float
    v2_front_cornering_stiffness_n_per_rad: float
    v2_rear_cornering_stiffness_n_per_rad: float
    v2_front_peak_mu: float
    v2_rear_peak_mu: float
    v2_mechanical_grip_low_speed_scale: float
    v2_mechanical_grip_high_speed_scale: float
    v2_mechanical_grip_transition_mps: float
    v2_tire_shape_c: float
    v2_slip_angle_peak_deg: float
    v2_rear_slip_steer_coupling: float
    v2_post_peak_falloff: float
    v2_load_sensitivity: float
    v2_aero_downforce_n_per_mps2: float
    v2_aero_balance_front: float
    v2_engine_power_w: float
    v2_drivetrain_efficiency: float
    v2_power_min_speed_mps: float
    v2_max_drive_g: float
    v2_max_brake_g: float
    v2_brake_bias_front: float
    v2_brake_lock_threshold: float
    v2_brake_lock_min_speed_mps: float
    v2_brake_lock_steer_loss: float
    v2_drag_coefficient: float
    v2_rolling_resistance_mps2: float
    v2_max_speed_mps: float
    v2_gear_ratios: tuple[float, ...]
    v2_final_drive_ratio: float
    v2_wheel_radius_m: float
    v2_idle_rpm: float
    v2_shift_up_rpm: float
    v2_shift_down_rpm: float
    v2_max_rpm: float
    v2_torque_peak_rpm: float
    v2_torque_peak_nm: float
    v2_torque_low_rpm_factor: float
    v2_torque_high_rpm_factor: float
    v2_tire_scrub_drag: float
    v2_surface_mu: float


def gpu_car_params_from_cpu(
    params: CarParams,
    *,
    physics_model: str = "v1",
    physics_v2: PhysicsV2Params | None = None,
) -> GpuCarParams:
    v2 = physics_v2 or PhysicsV2Params()
    return GpuCarParams(
        physics_model=str(physics_model),
        mass=float(params.mass),
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
        v2_calibration_id=v2.calibration_id,
        v2_version=v2.version,
        v2_max_steer_deg=float(v2.max_steer_deg),
        v2_steer_response=float(v2.steer_response),
        v2_steering_speed_sensitivity=float(v2.steering_speed_sensitivity),
        v2_front_weight_distribution=float(v2.front_weight_distribution),
        v2_cg_height_m=float(v2.cg_height_m),
        v2_track_width_m=float(v2.track_width_m),
        v2_front_axle_distance_m=float(v2.front_axle_distance_m),
        v2_rear_axle_distance_m=float(v2.rear_axle_distance_m),
        v2_front_cornering_stiffness_n_per_rad=float(v2.front_cornering_stiffness_n_per_rad),
        v2_rear_cornering_stiffness_n_per_rad=float(v2.rear_cornering_stiffness_n_per_rad),
        v2_front_peak_mu=float(v2.front_peak_mu),
        v2_rear_peak_mu=float(v2.rear_peak_mu),
        v2_mechanical_grip_low_speed_scale=float(v2.mechanical_grip_low_speed_scale),
        v2_mechanical_grip_high_speed_scale=float(v2.mechanical_grip_high_speed_scale),
        v2_mechanical_grip_transition_mps=float(v2.mechanical_grip_transition_mps),
        v2_tire_shape_c=float(v2.tire_shape_c),
        v2_slip_angle_peak_deg=float(v2.slip_angle_peak_deg),
        v2_rear_slip_steer_coupling=float(v2.rear_slip_steer_coupling),
        v2_post_peak_falloff=float(v2.post_peak_falloff),
        v2_load_sensitivity=float(v2.load_sensitivity),
        v2_aero_downforce_n_per_mps2=float(v2.aero_downforce_n_per_mps2),
        v2_aero_balance_front=float(v2.aero_balance_front),
        v2_engine_power_w=float(v2.engine_power_w),
        v2_drivetrain_efficiency=float(v2.drivetrain_efficiency),
        v2_power_min_speed_mps=float(v2.power_min_speed_mps),
        v2_max_drive_g=float(v2.max_drive_g),
        v2_max_brake_g=float(v2.max_brake_g),
        v2_brake_bias_front=float(v2.brake_bias_front),
        v2_brake_lock_threshold=float(v2.brake_lock_threshold),
        v2_brake_lock_min_speed_mps=float(v2.brake_lock_min_speed_mps),
        v2_brake_lock_steer_loss=float(v2.brake_lock_steer_loss),
        v2_drag_coefficient=float(v2.drag_coefficient),
        v2_rolling_resistance_mps2=float(v2.rolling_resistance_mps2),
        v2_max_speed_mps=float(v2.max_speed_mps),
        v2_gear_ratios=tuple(float(item) for item in v2.gear_ratios),
        v2_final_drive_ratio=float(v2.final_drive_ratio),
        v2_wheel_radius_m=float(v2.wheel_radius_m),
        v2_idle_rpm=float(v2.idle_rpm),
        v2_shift_up_rpm=float(v2.shift_up_rpm),
        v2_shift_down_rpm=float(v2.shift_down_rpm),
        v2_max_rpm=float(v2.max_rpm),
        v2_torque_peak_rpm=float(v2.torque_peak_rpm),
        v2_torque_peak_nm=float(v2.torque_peak_nm),
        v2_torque_low_rpm_factor=float(v2.torque_low_rpm_factor),
        v2_torque_high_rpm_factor=float(v2.torque_high_rpm_factor),
        v2_tire_scrub_drag=float(v2.tire_scrub_drag),
        v2_surface_mu=float(v2.surface_mu),
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
    local_projection_indices: torch.Tensor
    boundary_segments: torch.Tensor
    boundary_grid_indices: torch.Tensor
    boundary_grid_counts: torch.Tensor
    boundary_grid_cell_size_px: torch.Tensor
    boundary_grid_width: int
    boundary_grid_height: int
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
    length_px = torch.tensor(float(track.length_px), device=device, dtype=dtype)
    segment_count = int(starts.shape[0])
    local_count = max(1, min(64, segment_count))
    wrapped_mid_delta = torch.abs(
        torch.remainder(mid[None, :] - mid[:, None] + length_px * 0.5, length_px) - length_px * 0.5
    )
    local_projection_indices = torch.argsort(wrapped_mid_delta, dim=1)[:, :local_count].to(dtype=torch.int64)
    boundary_segments = torch.as_tensor(track.boundary_segments, device=device, dtype=dtype)
    mask_height, mask_width = int(track.drivable_mask.shape[0]), int(track.drivable_mask.shape[1])
    grid_cell_size = 64.0
    grid_width = max(1, int((mask_width + grid_cell_size - 1.0) // grid_cell_size))
    grid_height = max(1, int((mask_height + grid_cell_size - 1.0) // grid_cell_size))
    grid_cells: list[list[int]] = [[] for _ in range(grid_width * grid_height)]
    for segment_index, segment in enumerate(track.boundary_segments):
        x0, y0, x1, y1 = (float(value) for value in segment)
        min_x = max(0.0, min(x0, x1))
        max_x = min(float(mask_width - 1), max(x0, x1))
        min_y = max(0.0, min(y0, y1))
        max_y = min(float(mask_height - 1), max(y0, y1))
        min_cell_x = max(0, min(grid_width - 1, int(min_x // grid_cell_size)))
        max_cell_x = max(0, min(grid_width - 1, int(max_x // grid_cell_size)))
        min_cell_y = max(0, min(grid_height - 1, int(min_y // grid_cell_size)))
        max_cell_y = max(0, min(grid_height - 1, int(max_y // grid_cell_size)))
        for cell_y in range(min_cell_y, max_cell_y + 1):
            row_offset = cell_y * grid_width
            for cell_x in range(min_cell_x, max_cell_x + 1):
                grid_cells[row_offset + cell_x].append(segment_index)
    max_segments_per_cell = max((len(indices) for indices in grid_cells), default=0)
    if max_segments_per_cell <= 0:
        max_segments_per_cell = 1
    grid_indices_cpu = torch.full((grid_width * grid_height, max_segments_per_cell), -1, dtype=torch.int64)
    grid_counts_cpu = torch.empty(grid_width * grid_height, dtype=torch.int64)
    for cell_index, segment_indices in enumerate(grid_cells):
        grid_counts_cpu[cell_index] = len(segment_indices)
        if segment_indices:
            grid_indices_cpu[cell_index, : len(segment_indices)] = torch.tensor(segment_indices, dtype=torch.int64)
    boundary_grid_indices = grid_indices_cpu.to(device=device)
    boundary_grid_counts = grid_counts_cpu.to(device=device)
    boundary_grid_cell_size_px = torch.tensor(grid_cell_size, device=device, dtype=dtype)
    finish_line = torch.as_tensor(track.finish_line.reshape(1, 4), device=device, dtype=dtype)
    drivable_mask = torch.as_tensor(track.drivable_mask, device=device, dtype=torch.bool)
    meters_per_pixel = torch.tensor(float(track.meters_per_pixel), device=device, dtype=dtype)
    length_m = torch.tensor(float(track.length_m), device=device, dtype=dtype)
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
        local_projection_indices=local_projection_indices,
        boundary_segments=boundary_segments,
        boundary_grid_indices=boundary_grid_indices,
        boundary_grid_counts=boundary_grid_counts,
        boundary_grid_cell_size_px=boundary_grid_cell_size_px,
        boundary_grid_width=grid_width,
        boundary_grid_height=grid_height,
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
