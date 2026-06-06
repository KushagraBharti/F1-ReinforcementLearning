# pyright: reportPrivateImportUsage=false
"""Batched Monza rollout loop for GPU-backed evolution search."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from f1rl.config import SimConfig
from f1rl.gpu_features import braking_gate_tensor, controller_controls_batch, search_features_batch
from f1rl.gpu_physics import apply_physics_batch
from f1rl.gpu_scoring import (
    TERMINATION_REASON_TO_ID,
    GpuScoreAccumulator,
    GpuScoringDiagnostics,
    create_score_accumulator,
    score_profiles_batch,
    termination_reason_strings,
    update_score_accumulator_static,
)
from f1rl.gpu_track import (
    point_is_drivable_batch,
    segments_intersect_any_batch,
    segments_intersect_any_grid_batch,
    track_errors_batch,
)
from f1rl.gpu_types import (
    GpuCarBatch,
    GpuCarParams,
    GpuTrackTensors,
    car_batch_from_snapshots,
    gpu_car_params_from_cpu,
)
from f1rl.state_snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class GpuControlProgram:
    kind: str
    action_names: tuple[str, ...] = ()
    controller_weights: torch.Tensor | None = None
    action_controls: torch.Tensor | None = None
    phase_action_ids: torch.Tensor | None = None
    phase_thresholds: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GpuRolloutResult:
    accumulator: GpuScoreAccumulator
    profile_scores: dict[str, torch.Tensor]
    rollout_seconds: float
    steps_executed: int
    sim_steps: int
    control_kind: str
    action_names: tuple[str, ...]
    final_action_id: torch.Tensor
    host_sync_count: int
    kernel_backend: str
    graph_error: str | None = None
    graph_replay_seconds: float | None = None
    graph_replay_count: int = 0


@dataclass(frozen=True, slots=True)
class _GpuRolloutTensors:
    accumulator: GpuScoreAccumulator
    profile_scores: dict[str, torch.Tensor]
    output_state: GpuCarBatch
    output_last_segment_idx: torch.Tensor
    final_action_id: torch.Tensor
    sim_steps_tensor: torch.Tensor
    steps_executed: int
    host_sync_count: int


@dataclass(slots=True)
class GpuCapturedRollout:
    graph: Any
    input_state: GpuCarBatch
    input_last_segment_idx: torch.Tensor
    input_segment_start_progress_m: torch.Tensor
    input_segment_target_progress_m: torch.Tensor | None
    control_program: GpuControlProgram
    tensors: _GpuRolloutTensors
    capture_seconds: float


@dataclass(slots=True)
class GpuCapturedChunkRollout:
    graph: Any
    input_state: GpuCarBatch
    input_last_segment_idx: torch.Tensor
    input_segment_start_progress_m: torch.Tensor
    input_segment_target_progress_m: torch.Tensor | None
    input_accumulator: GpuScoreAccumulator
    input_final_action_id: torch.Tensor
    input_sim_steps_tensor: torch.Tensor
    control_program: GpuControlProgram
    scoring_profiles: tuple[str, ...]
    frontier_focus_start_m: float
    frontier_focus_end_m: float
    tensors: _GpuRolloutTensors
    chunk_steps: int
    replay_count: int
    capture_seconds: float
    tail_graph: Any | None = None
    tail_tensors: _GpuRolloutTensors | None = None
    tail_steps: int = 0


def _reason_id(name: str) -> int:
    return TERMINATION_REASON_TO_ID[name]


def _mask_float(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


def _mask_int(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


def _mask_bool(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


def _scoring_diagnostics(
    *,
    control_diagnostics: dict[str, torch.Tensor],
    step_diagnostics: dict[str, torch.Tensor],
    telemetry_valid_lap: torch.Tensor,
) -> GpuScoringDiagnostics:
    return GpuScoringDiagnostics(
        lateral_error_m=step_diagnostics["lateral_error_m"],
        heading_error_deg=control_diagnostics["heading_error_deg"],
        future_brake_demand=control_diagnostics["future_brake_demand"],
        brake_gate_proximity=control_diagnostics["brake_gate_proximity"],
        target_speed_drop_norm=control_diagnostics["target_speed_drop_norm"],
        telemetry_valid_lap=telemetry_valid_lap,
        target_speed_kph=control_diagnostics["target_speed_kph"],
        near_target_speed_kph=control_diagnostics["near_target_speed_kph"],
        min_future_target_speed_kph=control_diagnostics["min_future_target_speed_kph"],
        target_speed_drop_kph=control_diagnostics["target_speed_drop_kph"],
        brake_demand=control_diagnostics["brake_demand"],
        braking_gate_distance_m=control_diagnostics["braking_gate_distance_m"],
        brake_gate_distance_norm=control_diagnostics["brake_gate_distance_norm"],
        lookahead_abs_max=control_diagnostics["lookahead_abs_max"],
    )


def _clone_car_batch(state: GpuCarBatch) -> GpuCarBatch:
    return GpuCarBatch(**{field.name: getattr(state, field.name).clone() for field in fields(GpuCarBatch)})


def _copy_car_batch_(target: GpuCarBatch, source: GpuCarBatch) -> None:
    for field in fields(GpuCarBatch):
        getattr(target, field.name).copy_(getattr(source, field.name))


def _clone_score_accumulator(accumulator: GpuScoreAccumulator) -> GpuScoreAccumulator:
    return GpuScoreAccumulator(
        **{field.name: getattr(accumulator, field.name).clone() for field in fields(GpuScoreAccumulator)}
    )


def _score_accumulator_view(accumulator: GpuScoreAccumulator) -> GpuScoreAccumulator:
    return GpuScoreAccumulator(**{field.name: getattr(accumulator, field.name) for field in fields(GpuScoreAccumulator)})


def _copy_score_accumulator_(target: GpuScoreAccumulator, source: GpuScoreAccumulator) -> None:
    for field in fields(GpuScoreAccumulator):
        getattr(target, field.name).copy_(getattr(source, field.name))


def _tensor_compatible(target: torch.Tensor | None, source: torch.Tensor | None) -> bool:
    if target is None or source is None:
        return target is None and source is None
    return target.shape == source.shape and target.dtype == source.dtype and target.device == source.device


def _control_program_compatible(target: GpuControlProgram, source: GpuControlProgram) -> bool:
    if target.kind != source.kind or target.action_names != source.action_names:
        return False
    return (
        _tensor_compatible(target.controller_weights, source.controller_weights)
        and _tensor_compatible(target.action_controls, source.action_controls)
        and _tensor_compatible(target.phase_action_ids, source.phase_action_ids)
        and _tensor_compatible(target.phase_thresholds, source.phase_thresholds)
    )


def _copy_optional_tensor_(target: torch.Tensor | None, source: torch.Tensor | None) -> None:
    if target is None or source is None:
        return
    target.copy_(source)


def _copy_control_program_inputs_(target: GpuControlProgram, source: GpuControlProgram) -> None:
    if not _control_program_compatible(target, source):
        raise ValueError("Captured CUDA Graph control program shape does not match the requested program.")
    _copy_optional_tensor_(target.controller_weights, source.controller_weights)
    _copy_optional_tensor_(target.action_controls, source.action_controls)
    _copy_optional_tensor_(target.phase_action_ids, source.phase_action_ids)
    _copy_optional_tensor_(target.phase_thresholds, source.phase_thresholds)


def _replace_state_with_mask(old: GpuCarBatch, new: GpuCarBatch, active: torch.Tensor) -> GpuCarBatch:
    return replace(
        old,
        x=_mask_float(active, new.x, old.x),
        y=_mask_float(active, new.y, old.y),
        heading_rad=_mask_float(active, new.heading_rad, old.heading_rad),
        speed_mps=_mask_float(active, new.speed_mps, old.speed_mps),
        yaw_rate_rps=_mask_float(active, new.yaw_rate_rps, old.yaw_rate_rps),
        steering=_mask_float(active, new.steering, old.steering),
        elapsed_steps=_mask_int(active, new.elapsed_steps, old.elapsed_steps),
    )


class GpuMonzaBatch:
    """Evaluate a population of controller genomes in one tensor rollout."""

    def __init__(
        self,
        *,
        track: GpuTrackTensors,
        sim_config: SimConfig,
        feature_names: tuple[str, ...],
        collision_check: bool = True,
        collision_chunk_size: int = 2048,
        collision_mode: str = "exact_all_segments",
        compile_rollout: bool = False,
        compile_mode: str = "reduce-overhead",
        active_check_interval: int = 16,
        disable_early_stop: bool = False,
        profile_ranges: bool = False,
    ) -> None:
        self.track = track
        self.sim_config = sim_config
        self.feature_names = feature_names
        self.params: GpuCarParams = gpu_car_params_from_cpu(
            sim_config.car,
            physics_model=sim_config.physics_model,
            physics_v2=sim_config.physics_v2,
        )
        self.collision_check = collision_check
        self.collision_chunk_size = max(1, int(collision_chunk_size))
        self.collision_mode = str(collision_mode)
        self.active_check_interval = max(1, int(active_check_interval))
        self.disable_early_stop = bool(disable_early_stop)
        self.profile_ranges = bool(profile_ranges)
        self.braking_gates_m = braking_gate_tensor(device=track.device, dtype=track.dtype)
        self.local_projection_window_px = torch.as_tensor(
            float(sim_config.local_projection_window_m),
            device=track.device,
            dtype=track.dtype,
        ) / torch.clamp(track.meters_per_pixel, min=1e-6)
        self.lookahead_m = torch.tensor(
            tuple(float(value) for value in sim_config.lookahead_m),
            device=track.device,
            dtype=track.dtype,
        )
        self._reason_id_tensors = {
            reason: torch.tensor(reason_id, device=track.device, dtype=torch.int64)
            for reason, reason_id in TERMINATION_REASON_TO_ID.items()
        }
        self.state: GpuCarBatch | None = None
        self.segment_start_progress_m: torch.Tensor | None = None
        self.segment_target_progress_m: torch.Tensor | None = None
        self._row_index: torch.Tensor | None = None
        self._last_segment_idx: torch.Tensor | None = None
        self._continuous_action_id: torch.Tensor | None = None
        self._inactive_action_id: torch.Tensor | None = None
        self._zero_float: torch.Tensor | None = None
        self._zero_int: torch.Tensor | None = None
        self._zero_bool: torch.Tensor | None = None
        self._one_bool: torch.Tensor | None = None
        self._warp_controller_feature_ids: torch.Tensor | None = None
        self.compile_requested = bool(compile_rollout)
        self.compile_enabled = False
        self.compile_error: str | None = None
        self._apply_physics_batch: Any = apply_physics_batch
        self._controller_controls_batch: Any = controller_controls_batch
        self._search_features_batch: Any = search_features_batch
        if self.compile_requested:
            try:
                self._apply_physics_batch = torch.compile(apply_physics_batch, mode=compile_mode)
                self._controller_controls_batch = torch.compile(controller_controls_batch, mode=compile_mode)
                self._search_features_batch = torch.compile(search_features_batch, mode=compile_mode)
                self.compile_enabled = True
            except Exception as exc:  # pragma: no cover - depends on local torch compiler support.
                self.compile_error = f"{type(exc).__name__}: {exc}"

    def _disable_compile(self, exc: Exception) -> None:
        self.compile_enabled = False
        self.compile_error = f"{type(exc).__name__}: {exc}"
        self._apply_physics_batch = apply_physics_batch
        self._controller_controls_batch = controller_controls_batch
        self._search_features_batch = search_features_batch

    def _apply_physics(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._apply_physics_batch(*args, **kwargs)
        except Exception as exc:
            if not self.compile_enabled:
                raise
            self._disable_compile(exc)
            return apply_physics_batch(*args, **kwargs)

    def _search_features(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._search_features_batch(*args, **kwargs)
        except Exception as exc:
            if not self.compile_enabled:
                raise
            self._disable_compile(exc)
            return search_features_batch(*args, **kwargs)

    def _controller_controls(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._controller_controls_batch(*args, **kwargs)
        except Exception as exc:
            if not self.compile_enabled:
                raise
            self._disable_compile(exc)
            return controller_controls_batch(*args, **kwargs)

    def reset(self, snapshots: list[StateSnapshot], *, target_progress_m: float, terminate_at_target: bool) -> None:
        next_state = car_batch_from_snapshots(
            snapshots,
            meters_per_pixel=float(self.track.meters_per_pixel.detach().cpu()),
            device=self.track.device,
            dtype=self.track.dtype,
        )
        if self.state is None or self.state.size != next_state.size:
            self.state = next_state
        else:
            _copy_car_batch_(self.state, next_state)
        next_segment_start = self.state.monotonic_progress_m.clone()
        if self.segment_start_progress_m is None or self.segment_start_progress_m.shape != next_segment_start.shape:
            self.segment_start_progress_m = next_segment_start
        else:
            self.segment_start_progress_m.copy_(next_segment_start)
        if self._row_index is None or self._row_index.shape[0] != self.state.size:
            self._row_index = torch.arange(self.state.size, device=self.track.device, dtype=torch.int64)
        next_segment_idx = torch.searchsorted(
            self.track.centerline_cumdist_px,
            torch.remainder(self.state.last_raw_progress_px, self.track.length_px),
            right=True,
        ) - 1
        next_segment_idx = torch.clamp(next_segment_idx, 0, self.track.centerline_xy.shape[0] - 1)
        if self._last_segment_idx is None or self._last_segment_idx.shape != next_segment_idx.shape:
            self._last_segment_idx = next_segment_idx
        else:
            self._last_segment_idx.copy_(next_segment_idx)
        if self._continuous_action_id is None or self._continuous_action_id.shape[0] != self.state.size:
            self._continuous_action_id = torch.full((self.state.size,), -2, device=self.track.device, dtype=torch.int64)
            self._inactive_action_id = torch.full((self.state.size,), -1, device=self.track.device, dtype=torch.int64)
            self._zero_float = torch.zeros(self.state.size, device=self.track.device, dtype=self.track.dtype)
            self._zero_int = torch.zeros(self.state.size, device=self.track.device, dtype=torch.int64)
            self._zero_bool = torch.zeros(self.state.size, device=self.track.device, dtype=torch.bool)
            self._one_bool = torch.ones(self.state.size, device=self.track.device, dtype=torch.bool)
        if terminate_at_target:
            length = torch.clamp(float(target_progress_m) - self.segment_start_progress_m, min=1.0)
            next_target = self.segment_start_progress_m + length
            if self.segment_target_progress_m is None or self.segment_target_progress_m.shape != next_target.shape:
                self.segment_target_progress_m = next_target
            else:
                self.segment_target_progress_m.copy_(next_target)
        else:
            self.segment_target_progress_m = None

    def _active(self) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout().")
        return self.state.alive & ~self.state.terminated & ~self.state.truncated

    def _set_reason(self, mask: torch.Tensor, reason: str) -> None:
        if self.state is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout().")
        self.state.termination_reason_id = torch.where(
            mask,
            self._reason_id_tensors[reason],
            self.state.termination_reason_id,
        )

    def _update_checkpoint_validity(
        self,
        *,
        old_progress_m: torch.Tensor,
        progress_delta_m: torch.Tensor,
        lateral_error_m: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        assert self.state is not None
        assert self._zero_int is not None
        assert self._zero_bool is not None
        spacing = self.track.checkpoint_spacing_m
        checkpoint_count = max(self.track.checkpoint_count, 1)
        skipped = active & (progress_delta_m > spacing * 1.75)
        skipped_count = torch.clamp(torch.floor(progress_delta_m / spacing).to(torch.int64) - 1, min=1)
        self.state.missed_checkpoint_count += torch.where(skipped, skipped_count, self._zero_int)
        self.state.valid_lap = torch.where(skipped, self._zero_bool, self.state.valid_lap)

        next_idx = self.state.next_checkpoint_index
        threshold = spacing * next_idx.to(dtype=self.track.dtype)
        due = active & (next_idx < checkpoint_count) & (self.state.monotonic_progress_m + 1e-6 >= threshold)
        expected = (old_progress_m <= threshold) & (threshold <= self.state.monotonic_progress_m + spacing * 0.75)
        lateral_ok = torch.abs(lateral_error_m) <= float(self.sim_config.checkpoint_lateral_limit_m)
        invalid = due & ~(expected & lateral_ok)
        self.state.missed_checkpoint_count += invalid.to(dtype=torch.int64)
        self.state.valid_lap = torch.where(invalid, self._zero_bool, self.state.valid_lap)
        self.state.checkpoints_passed = torch.where(due, next_idx, self.state.checkpoints_passed)
        self.state.next_checkpoint_index = torch.where(due, next_idx + 1, next_idx)
        checkpoint_index = torch.remainder(
            torch.floor(self.state.monotonic_progress_m / spacing).to(torch.int64),
            checkpoint_count,
        )
        self.state.checkpoint_index = checkpoint_index

    def _step(
        self,
        *,
        throttle: torch.Tensor,
        brake: torch.Tensor,
        steer: torch.Tensor,
        active: torch.Tensor,
        gates: Any,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.state is not None
        assert self.segment_start_progress_m is not None
        assert self._zero_int is not None
        assert self._zero_bool is not None
        assert self._one_bool is not None
        if (
            self.sim_config.launch_guard_progress_m > 0.0
            and self.sim_config.launch_guard_min_speed_kph > 0.0
        ):
            guard = (
                active
                & (self.state.monotonic_progress_m <= self.sim_config.launch_guard_progress_m)
                & (self.state.speed_mps * 3.6 < self.sim_config.launch_guard_min_speed_kph)
                & (throttle <= 1e-6)
                & (brake > 1e-6)
            )
            throttle = torch.where(guard, torch.full_like(throttle, self.sim_config.launch_guard_throttle), throttle)
            brake = torch.where(guard, torch.zeros_like(brake), brake)

        old_progress_m = self.state.monotonic_progress_m.clone()
        old_raw_px = self.state.last_raw_progress_px.clone()
        old_lap = self.state.lap_index.clone()
        old_state = self.state
        next_state, movement = self._apply_physics(
            old_state,
            throttle=throttle,
            brake=brake,
            steer=steer,
            params=self.params,
            meters_per_pixel=self.track.meters_per_pixel,
        )
        self.state = _replace_state_with_mask(old_state, next_state, active)
        self.state.last_throttle = torch.where(active, throttle, self.state.last_throttle)
        self.state.last_brake = torch.where(active, brake, self.state.last_brake)
        self.state.last_steer = torch.where(active, steer, self.state.last_steer)

        assert self._last_segment_idx is not None
        errors = track_errors_batch(
            self.state.position_xy(),
            self.state.heading_rad,
            self.track,
            previous_progress_px=old_raw_px,
            previous_segment_idx=self._last_segment_idx,
            window_px=self.local_projection_window_px,
            row_index=self._row_index,
        )
        self._last_segment_idx = torch.where(active, errors["segment_idx"], self._last_segment_idx)
        raw_px = errors["raw_progress_px"]
        raw_delta_px = raw_px - old_raw_px
        raw_delta_px = torch.where(raw_delta_px < -0.5 * self.track.length_px, raw_delta_px + self.track.length_px, raw_delta_px)
        raw_delta_px = torch.where(raw_delta_px > 0.5 * self.track.length_px, raw_delta_px - self.track.length_px, raw_delta_px)
        progress_delta_m = torch.clamp(raw_delta_px * self.track.meters_per_pixel, min=0.0)
        progress_delta_m = torch.where(
            progress_delta_m > self.sim_config.local_projection_window_m,
            torch.zeros_like(progress_delta_m),
            progress_delta_m,
        )
        self.state.last_raw_progress_px = torch.where(active, raw_px, self.state.last_raw_progress_px)
        self.state.raw_progress_m = torch.where(active, raw_px * self.track.meters_per_pixel, self.state.raw_progress_m)
        self.state.monotonic_progress_m = torch.where(
            active,
            old_progress_m + progress_delta_m,
            self.state.monotonic_progress_m,
        )
        self._update_checkpoint_validity(
            old_progress_m=old_progress_m,
            progress_delta_m=progress_delta_m,
            lateral_error_m=errors["lateral_error_m"],
            active=active,
        )

        movements = movement.as_segments()
        if self.collision_check:
            if self.collision_mode == "exact_grid":
                collided = segments_intersect_any_grid_batch(movements, self.track)
            else:
                collided = segments_intersect_any_batch(
                    movements,
                    self.track.boundary_segments,
                    chunk_size=self.collision_chunk_size,
                )
        else:
            collided = self._zero_bool
        collided = collided & active
        off_track = active & ~point_is_drivable_batch(self.state.x, self.state.y, self.track)
        no_progress = active & (progress_delta_m <= 1e-4)
        self.state.no_progress_steps = torch.where(
            no_progress,
            self.state.no_progress_steps + 1,
            torch.where(active, self._zero_int, self.state.no_progress_steps),
        )

        speed_kph_after = self.state.speed_mps * 3.6
        heading_error_abs_deg = torch.abs(errors["heading_error_rad"] * (180.0 / torch.pi))
        if bool(getattr(gates, "segment_require_release", False)):
            release_min = min(float(gates.segment_release_min_speed_kph), float(gates.segment_release_max_speed_kph))
            release_max = max(float(gates.segment_release_min_speed_kph), float(gates.segment_release_max_speed_kph))
            release_observed = (
                active
                & (speed_kph_after >= release_min)
                & (speed_kph_after <= release_max)
                & (brake <= float(gates.segment_release_max_brake))
                & (throttle <= float(gates.segment_release_max_throttle))
            )
            self.state.segment_release_observed |= release_observed

        if self.segment_target_progress_m is not None:
            not_already_terminated = active & ~self.state.terminated
            target_crossed = (old_progress_m < self.segment_target_progress_m) & (
                self.segment_target_progress_m <= self.state.monotonic_progress_m
            )
            target_reached = self.state.monotonic_progress_m >= self.segment_target_progress_m
            speed_gate_ok = self._one_bool
            if getattr(gates, "target_min_speed_kph", None) is not None:
                speed_gate_ok = speed_gate_ok & (speed_kph_after >= float(gates.target_min_speed_kph))
            if getattr(gates, "target_max_speed_kph", None) is not None:
                speed_gate_ok = speed_gate_ok & (speed_kph_after <= float(gates.target_max_speed_kph))
            lateral_gate_ok = self._one_bool
            if getattr(gates, "target_max_lateral_error_m", None) is not None:
                lateral_gate_ok = lateral_gate_ok & (
                    torch.abs(errors["lateral_error_m"]) <= float(gates.target_max_lateral_error_m)
                )
            heading_gate_ok = self._one_bool
            if getattr(gates, "target_max_heading_error_deg", None) is not None:
                heading_gate_ok = heading_gate_ok & (heading_error_abs_deg <= float(gates.target_max_heading_error_deg))
            yaw_rate_gate_ok = self._one_bool
            if getattr(gates, "target_max_abs_yaw_rate_rps", None) is not None:
                yaw_rate_gate_ok = yaw_rate_gate_ok & (
                    torch.abs(self.state.yaw_rate_rps) <= float(gates.target_max_abs_yaw_rate_rps)
                )
            steering_gate_ok = self._one_bool
            if getattr(gates, "target_max_abs_steering", None) is not None:
                steering_gate_ok = steering_gate_ok & (
                    torch.abs(self.state.steering) <= float(gates.target_max_abs_steering)
                )
            release_gate_ok = self._one_bool
            if bool(getattr(gates, "segment_require_release", False)):
                release_gate_ok = release_gate_ok & self.state.segment_release_observed
            complete = (
                not_already_terminated
                & target_reached
                & speed_gate_ok
                & lateral_gate_ok
                & heading_gate_ok
                & yaw_rate_gate_ok
                & steering_gate_ok
                & release_gate_ok
            )
            self.state.segment_complete |= complete
            self.state.truncated |= complete
            self._set_reason(complete, "segment_complete")
            release_failed = (
                not_already_terminated
                & target_crossed
                & speed_gate_ok
                & bool(getattr(gates, "segment_require_release", False))
                & ~self.state.segment_release_observed
                & ~complete
            )
            self.state.truncated |= release_failed
            self._set_reason(release_failed, "segment_release_gate_failed")
            gate_miss = (
                not_already_terminated
                & target_crossed
                & bool(getattr(gates, "segment_fail_on_speed_gate_miss", False))
                & ~complete
                & ~release_failed
            )
            min_speed_failed = self._zero_bool
            if getattr(gates, "target_min_speed_kph", None) is not None:
                min_speed_failed = gate_miss & (speed_kph_after < float(gates.target_min_speed_kph))
            max_speed_failed = self._zero_bool
            if getattr(gates, "target_max_speed_kph", None) is not None:
                max_speed_failed = gate_miss & (speed_kph_after > float(gates.target_max_speed_kph))
            lateral_failed = self._zero_bool
            if getattr(gates, "target_max_lateral_error_m", None) is not None:
                lateral_failed = gate_miss & ~lateral_gate_ok
            heading_failed = self._zero_bool
            if getattr(gates, "target_max_heading_error_deg", None) is not None:
                heading_failed = gate_miss & ~heading_gate_ok
            yaw_failed = self._zero_bool
            if getattr(gates, "target_max_abs_yaw_rate_rps", None) is not None:
                yaw_failed = gate_miss & ~yaw_rate_gate_ok
            steering_failed = self._zero_bool
            if getattr(gates, "target_max_abs_steering", None) is not None:
                steering_failed = gate_miss & ~steering_gate_ok
            for reason, mask in (
                ("segment_min_speed_gate_failed", min_speed_failed),
                ("segment_speed_gate_failed", max_speed_failed),
                ("segment_lateral_gate_failed", lateral_failed),
                ("segment_heading_gate_failed", heading_failed),
                ("segment_yaw_rate_gate_failed", yaw_failed),
                ("segment_steering_gate_failed", steering_failed),
            ):
                self.state.truncated |= mask
                self._set_reason(mask, reason)

        target_lap_progress_m = (old_lap.to(dtype=self.track.dtype) + 1.0) * self.track.length_m
        near_finish = old_progress_m >= target_lap_progress_m - self.track.checkpoint_spacing_m * 2.0
        physical_finish = segments_intersect_any_batch(
            movements,
            self.track.finish_line,
            chunk_size=1,
        )
        virtual_finish = (old_progress_m < target_lap_progress_m) & (
            target_lap_progress_m <= self.state.monotonic_progress_m
        )
        crossed_finish = active & near_finish & (physical_finish | virtual_finish)
        self.state.finish_crossed |= crossed_finish
        telemetry_valid_lap = (
            self.state.valid_lap
            & (self.state.missed_checkpoint_count == 0)
            & (self.state.checkpoints_passed >= max(self.track.checkpoint_count - 1, 0))
        )
        lap_complete = (
            active
            & (self.segment_target_progress_m is None)
            & crossed_finish
            & telemetry_valid_lap
            & (self.state.monotonic_progress_m >= target_lap_progress_m)
            & ~self.state.terminated
        )
        self.state.lap_index = torch.where(lap_complete, self.state.lap_index + 1, self.state.lap_index)
        self.state.completed_lap |= lap_complete
        self.state.truncated |= lap_complete
        self._set_reason(lap_complete, "lap_complete")

        self.state.terminated |= collided
        self._set_reason(collided, "collision")
        self.state.terminated |= off_track
        self._set_reason(off_track, "off_track")
        no_progress_terminated = active & (self.state.no_progress_steps >= self.sim_config.no_progress_limit_steps)
        self.state.terminated |= no_progress_terminated
        self._set_reason(no_progress_terminated, "no_progress")
        max_steps = active & (self.state.elapsed_steps >= self.sim_config.max_steps)
        self.state.truncated |= max_steps
        self._set_reason(max_steps, "max_steps")
        self.state.alive = torch.where(self.state.terminated, self._zero_bool, self.state.alive)

        errors["telemetry_valid_lap"] = telemetry_valid_lap
        errors["progress_delta_m"] = progress_delta_m
        return errors, collided, off_track, telemetry_valid_lap

    def _program_controls(
        self,
        program: GpuControlProgram,
        *,
        gates: Any,
        control_backend: str = "torch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        assert self.state is not None
        assert self.segment_start_progress_m is not None
        if program.kind == "controller":
            if program.controller_weights is None:
                raise ValueError("controller control programs require controller_weights")
            assert self._zero_float is not None
            if control_backend == "warp":
                from f1rl.gpu_fused_warp import (
                    controller_controls_warp_batch,
                    controller_feature_ids_tensor,
                    search_features_warp_batch,
                )

                assert self._last_segment_idx is not None
                if (
                    self._warp_controller_feature_ids is None
                    or int(self._warp_controller_feature_ids.numel()) != len(self.feature_names)
                    or self._warp_controller_feature_ids.device != self.track.device
                ):
                    self._warp_controller_feature_ids = controller_feature_ids_tensor(
                        self.feature_names,
                        device=self.track.device,
                    )
                features, diagnostics = search_features_warp_batch(
                    self.state,
                    self.track,
                    self.sim_config,
                    feature_names=self.feature_names,
                    segment_start_progress_m=self.segment_start_progress_m,
                    segment_target_progress_m=float(gates.target_progress_m),
                    braking_gates_m=self.braking_gates_m,
                    lookahead_m=self.lookahead_m,
                    local_projection_window_px=self.local_projection_window_px,
                    previous_segment_idx=self._last_segment_idx,
                    zero_feature=self._zero_float,
                    feature_ids=self._warp_controller_feature_ids,
                )
                throttle, brake, steer = controller_controls_warp_batch(program.controller_weights, features)
            else:
                features, diagnostics = self._search_features(
                    self.state,
                    self.track,
                    self.sim_config,
                    feature_names=self.feature_names,
                    segment_start_progress_m=self.segment_start_progress_m,
                    segment_target_progress_m=float(gates.target_progress_m),
                    braking_gates_m=self.braking_gates_m,
                    lookahead_m=self.lookahead_m,
                    local_projection_window_px=self.local_projection_window_px,
                    previous_segment_idx=self._last_segment_idx,
                    zero_feature=self._zero_float,
                    row_index=self._row_index,
                )
                throttle, brake, steer = self._controller_controls(program.controller_weights, features)
            self._last_segment_idx = diagnostics["segment_idx"]
            assert self._continuous_action_id is not None
            action_id = self._continuous_action_id
            return throttle, brake, steer, action_id, diagnostics
        if program.kind not in {"phase", "progress_phase"}:
            raise ValueError(f"Unsupported GPU control program kind {program.kind!r}")
        if program.action_controls is None or program.phase_action_ids is None or program.phase_thresholds is None:
            raise ValueError(f"{program.kind} control programs require action controls and phase tensors")
        if control_backend == "warp":
            from f1rl.gpu_fused_warp import phase_controls_warp_batch

            progress_delta_m = torch.clamp(self.state.monotonic_progress_m - self.segment_start_progress_m, min=0.0)
            throttle, brake, steer, action_id = phase_controls_warp_batch(
                action_controls=program.action_controls,
                phase_action_ids=program.phase_action_ids,
                phase_thresholds=program.phase_thresholds,
                elapsed_steps=self.state.elapsed_steps,
                progress_delta_m=progress_delta_m,
                use_progress=program.kind == "progress_phase",
            )
        else:
            if program.kind == "phase":
                phase_index = torch.sum(self.state.elapsed_steps[:, None] >= program.phase_thresholds, dim=1)
            else:
                progress_delta_m = torch.clamp(self.state.monotonic_progress_m - self.segment_start_progress_m, min=0.0)
                phase_index = torch.sum(progress_delta_m[:, None] >= program.phase_thresholds, dim=1)
            phase_index = torch.clamp(phase_index, max=program.phase_action_ids.shape[1] - 1).to(dtype=torch.int64)
            assert self._row_index is not None
            row_index = self._row_index
            action_id = program.phase_action_ids[row_index, phase_index]
            controls = program.action_controls[action_id]
            throttle, brake, steer = controls[:, 0], controls[:, 1], controls[:, 2]
        if control_backend == "warp":
            from f1rl.gpu_fused_warp import track_errors_warp_local_batch

            assert self._last_segment_idx is not None
            diagnostics = track_errors_warp_local_batch(
                self.state.position_xy(),
                self.state.heading_rad,
                self.track,
                previous_progress_px=self.state.last_raw_progress_px,
                previous_segment_idx=self._last_segment_idx,
                window_px=self.local_projection_window_px,
            )
        else:
            diagnostics = track_errors_batch(
                self.state.position_xy(),
                self.state.heading_rad,
                self.track,
                previous_progress_px=self.state.last_raw_progress_px,
                previous_segment_idx=self._last_segment_idx,
                window_px=self.local_projection_window_px,
                row_index=self._row_index,
            )
        self._last_segment_idx = diagnostics["segment_idx"]
        assert self._zero_float is not None
        zero = self._zero_float
        diagnostics.update(
            {
                "heading_error_deg": diagnostics["heading_error_rad"] * (180.0 / torch.pi),
                "speed_kph": self.state.speed_mps * 3.6,
                "target_speed_kph": zero,
                "near_target_speed_kph": zero,
                "min_future_target_speed_kph": zero,
                "target_speed_drop_kph": zero,
                "target_speed_drop_norm": zero,
                "brake_demand": zero,
                "future_brake_demand": zero,
                "brake_gate_proximity": zero,
                "braking_gate_distance_m": zero,
                "brake_gate_distance_norm": zero,
                "lookahead_abs_max": torch.abs(diagnostics["heading_error_rad"]),
            }
        )
        return throttle, brake, steer, action_id, diagnostics

    def rollout(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuRolloutResult:
        started = time.perf_counter()
        tensors = self._rollout_tensors(
            control_program=control_program,
            gates=gates,
            scoring_profiles=scoring_profiles,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
            allow_host_sync=True,
            profile_ranges=self.profile_ranges,
        )
        if self.track.device.type == "cuda":
            torch.cuda.synchronize(self.track.device)
        sim_steps = int(tensors.sim_steps_tensor.detach().cpu().item())
        if self.track.device.type == "cuda":
            torch.cuda.synchronize(self.track.device)
        elapsed = time.perf_counter() - started
        return GpuRolloutResult(
            accumulator=tensors.accumulator,
            profile_scores=tensors.profile_scores,
            rollout_seconds=elapsed,
            steps_executed=tensors.steps_executed,
            sim_steps=sim_steps,
            control_kind=control_program.kind,
            action_names=control_program.action_names,
            final_action_id=tensors.final_action_id,
            host_sync_count=tensors.host_sync_count,
            kernel_backend="pytorch_compile_helpers" if self.compile_requested else "pytorch_eager",
        )

    def rollout_cuda_graph(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuRolloutResult:
        if self.track.device.type != "cuda":
            fallback = self.rollout(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
            return replace(
                fallback,
                kernel_backend="pytorch_eager_graph_fallback",
                graph_error="CUDA Graph capture requires a CUDA device.",
            )
        if self.state is None or self._last_segment_idx is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout_cuda_graph().")
        try:
            initial_state = _clone_car_batch(self.state)
            initial_last_segment_idx = self._last_segment_idx.clone()
            assert self.segment_start_progress_m is not None
            initial_segment_start_progress_m = self.segment_start_progress_m.clone()
            initial_segment_target_progress_m = (
                self.segment_target_progress_m.clone() if self.segment_target_progress_m is not None else None
            )
            captured = self.capture_cuda_graph(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
            _copy_car_batch_(captured.input_state, initial_state)
            captured.input_last_segment_idx.copy_(initial_last_segment_idx)
            captured.input_segment_start_progress_m.copy_(initial_segment_start_progress_m)
            _copy_optional_tensor_(captured.input_segment_target_progress_m, initial_segment_target_progress_m)
            result = self.replay_cuda_graph(
                captured,
                snapshots=None,
                control_program=control_program,
                gates=gates,
            )
            return replace(result, rollout_seconds=captured.capture_seconds + result.rollout_seconds)
        except Exception as exc:  # pragma: no cover - CUDA graph support is platform/operator dependent.
            fallback = self.rollout(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
            return replace(
                fallback,
                kernel_backend="pytorch_eager_graph_fallback",
                graph_error=f"{type(exc).__name__}: {exc}",
            )

    def rollout_warp_open(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuRolloutResult:
        """Run a Warp-backed rollout for fused-backend parity work."""

        if self.track.device.type != "cuda" or self.track.dtype != torch.float32:
            raise RuntimeError("Warp rollout requires CUDA float32 track/state tensors.")
        if (
            control_program.kind == "controller"
            and not bool(gates.terminate_at_target_progress)
            and self.collision_mode == "exact_grid"
            and not self.profile_ranges
        ):
            return self._rollout_warp_persistent_controller_open(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
        started = time.perf_counter()
        tensors = self._rollout_tensors(
            control_program=control_program,
            gates=gates,
            scoring_profiles=scoring_profiles,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
            allow_host_sync=True,
            profile_ranges=self.profile_ranges,
            step_backend="warp_open",
        )
        torch.cuda.synchronize(self.track.device)
        sim_steps = int(tensors.sim_steps_tensor.detach().cpu().item())
        elapsed = time.perf_counter() - started
        return GpuRolloutResult(
            accumulator=tensors.accumulator,
            profile_scores=tensors.profile_scores,
            rollout_seconds=elapsed,
            steps_executed=tensors.steps_executed,
            sim_steps=sim_steps,
            control_kind=control_program.kind,
            action_names=control_program.action_names,
            final_action_id=tensors.final_action_id,
            host_sync_count=tensors.host_sync_count,
            kernel_backend="warp_open_step",
        )

    def _rollout_warp_persistent_controller_open(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuRolloutResult:
        """Run the explicit controller/no-target fused path with one persistent Warp kernel."""

        if self.state is None or self.segment_start_progress_m is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout_warp_open().")
        if control_program.controller_weights is None:
            raise ValueError("controller persistent Warp rollout requires controller_weights.")
        assert self._last_segment_idx is not None
        from f1rl.gpu_fused_warp import (
            controller_feature_ids_tensor,
            persistent_controller_open_rollout_warp_batch,
            score_profiles_warp_batch,
        )

        if (
            self._warp_controller_feature_ids is None
            or int(self._warp_controller_feature_ids.numel()) != len(self.feature_names)
            or self._warp_controller_feature_ids.device != self.track.device
        ):
            self._warp_controller_feature_ids = controller_feature_ids_tensor(
                self.feature_names,
                device=self.track.device,
            )
        accumulator = create_score_accumulator(self.state)
        assert self._inactive_action_id is not None
        final_action_id = self._inactive_action_id.clone()
        sim_steps_per_row = torch.empty_like(self.state.elapsed_steps)
        started = time.perf_counter()
        persistent_controller_open_rollout_warp_batch(
            self.state,
            accumulator,
            last_segment_idx=self._last_segment_idx,
            controller_weights=control_program.controller_weights,
            feature_ids=self._warp_controller_feature_ids,
            braking_gates_m=self.braking_gates_m,
            lookahead_m=self.lookahead_m,
            params=self.params,
            track=self.track,
            sim_config=self.sim_config,
            target_progress_m=float(gates.target_progress_m),
            final_action_id=final_action_id,
            sim_steps_per_row=sim_steps_per_row,
            collision_check=self.collision_check,
        )
        profile_scores = score_profiles_warp_batch(
            accumulator,
            profiles=scoring_profiles,
            target_progress_m=float(gates.target_progress_m),
            terminate_at_target_progress=False,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
        )
        torch.cuda.synchronize(self.track.device)
        sim_steps = int(sim_steps_per_row.sum().detach().cpu().item())
        elapsed = time.perf_counter() - started
        return GpuRolloutResult(
            accumulator=accumulator,
            profile_scores=profile_scores,
            rollout_seconds=elapsed,
            steps_executed=int(self.sim_config.max_steps),
            sim_steps=sim_steps,
            control_kind=control_program.kind,
            action_names=control_program.action_names,
            final_action_id=final_action_id,
            host_sync_count=0,
            kernel_backend="warp_persistent_controller_open",
        )

    def capture_cuda_graph(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuCapturedRollout:
        if self.track.device.type != "cuda":
            raise RuntimeError("CUDA Graph capture requires a CUDA device.")
        if (
            self.state is None
            or self._last_segment_idx is None
            or self.segment_start_progress_m is None
        ):
            raise RuntimeError("GpuMonzaBatch.reset() must be called before capture_cuda_graph().")
        graph_input_state = self.state
        graph_input_segment_idx = self._last_segment_idx
        graph_input_start_progress = self.segment_start_progress_m
        graph_input_target_progress = self.segment_target_progress_m

        def restore_initial_state() -> None:
            self.state = graph_input_state
            self._last_segment_idx = graph_input_segment_idx
            self.segment_start_progress_m = graph_input_start_progress
            self.segment_target_progress_m = graph_input_target_progress

        started = time.perf_counter()
        current_stream = torch.cuda.current_stream(self.track.device)
        warmup_stream = torch.cuda.Stream(device=self.track.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            self._rollout_tensors(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
                allow_host_sync=False,
                profile_ranges=False,
            )
        current_stream.wait_stream(warmup_stream)
        restore_initial_state()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tensors = self._rollout_tensors(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
                allow_host_sync=False,
                profile_ranges=False,
            )
        torch.cuda.synchronize(self.track.device)
        capture_seconds = time.perf_counter() - started
        return GpuCapturedRollout(
            graph=graph,
            input_state=graph_input_state,
            input_last_segment_idx=graph_input_segment_idx,
            input_segment_start_progress_m=graph_input_start_progress,
            input_segment_target_progress_m=graph_input_target_progress,
            control_program=control_program,
            tensors=tensors,
            capture_seconds=capture_seconds,
        )

    def replay_cuda_graph(
        self,
        captured: GpuCapturedRollout,
        *,
        snapshots: list[StateSnapshot] | None,
        control_program: GpuControlProgram,
        gates: Any,
    ) -> GpuRolloutResult:
        self.state = captured.input_state
        self._last_segment_idx = captured.input_last_segment_idx
        self.segment_start_progress_m = captured.input_segment_start_progress_m
        self.segment_target_progress_m = captured.input_segment_target_progress_m
        if snapshots is not None:
            self.reset(
                snapshots,
                target_progress_m=float(gates.target_progress_m),
                terminate_at_target=bool(gates.terminate_at_target_progress),
            )
        _copy_control_program_inputs_(captured.control_program, control_program)
        replay_started = time.perf_counter()
        captured.graph.replay()
        torch.cuda.synchronize(self.track.device)
        replay_elapsed = time.perf_counter() - replay_started
        sim_steps = int(captured.tensors.sim_steps_tensor.detach().cpu().item())
        return GpuRolloutResult(
            accumulator=captured.tensors.accumulator,
            profile_scores=captured.tensors.profile_scores,
            rollout_seconds=replay_elapsed,
            steps_executed=captured.tensors.steps_executed,
            sim_steps=sim_steps,
            control_kind=captured.control_program.kind,
            action_names=captured.control_program.action_names,
            final_action_id=captured.tensors.final_action_id,
            host_sync_count=0,
            kernel_backend="pytorch_cuda_graph",
            graph_replay_seconds=replay_elapsed,
            graph_replay_count=1,
        )

    def capture_cuda_graph_chunk(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        chunk_steps: int,
        replay_count: int,
        tail_steps: int = 0,
    ) -> GpuCapturedChunkRollout:
        if self.track.device.type != "cuda":
            raise RuntimeError("CUDA Graph chunk capture requires a CUDA device.")
        if (
            self.state is None
            or self._last_segment_idx is None
            or self.segment_start_progress_m is None
        ):
            raise RuntimeError("GpuMonzaBatch.reset() must be called before capture_cuda_graph_chunk().")
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        if replay_count <= 0:
            raise ValueError("replay_count must be positive")
        if tail_steps < 0:
            raise ValueError("tail_steps must be non-negative")

        graph_input_state = self.state
        graph_input_segment_idx = self._last_segment_idx
        graph_input_start_progress = self.segment_start_progress_m
        graph_input_target_progress = self.segment_target_progress_m
        graph_input_accumulator = create_score_accumulator(graph_input_state)
        warmup_accumulator = _clone_score_accumulator(graph_input_accumulator)
        capture_accumulator = _score_accumulator_view(graph_input_accumulator)
        graph_input_final_action_id = self._inactive_action_id.clone() if self._inactive_action_id is not None else None
        if graph_input_final_action_id is None:
            raise RuntimeError("GpuMonzaBatch.reset() must initialize action-id buffers before graph capture.")
        graph_input_sim_steps_tensor = torch.zeros((), device=self.track.device, dtype=torch.int64)

        def restore_initial_refs() -> None:
            self.state = graph_input_state
            self._last_segment_idx = graph_input_segment_idx
            self.segment_start_progress_m = graph_input_start_progress
            self.segment_target_progress_m = graph_input_target_progress

        started = time.perf_counter()
        current_stream = torch.cuda.current_stream(self.track.device)
        warmup_stream = torch.cuda.Stream(device=self.track.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            self._rollout_tensors(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
                allow_host_sync=False,
                profile_ranges=False,
                max_steps=chunk_steps,
                accumulator=warmup_accumulator,
                initial_final_action_id=graph_input_final_action_id,
                initial_sim_steps_tensor=graph_input_sim_steps_tensor,
            )
            if tail_steps > 0:
                self._rollout_tensors(
                    control_program=control_program,
                    gates=gates,
                    scoring_profiles=scoring_profiles,
                    frontier_focus_start_m=frontier_focus_start_m,
                    frontier_focus_end_m=frontier_focus_end_m,
                    allow_host_sync=False,
                    profile_ranges=False,
                    max_steps=tail_steps,
                    accumulator=warmup_accumulator,
                    initial_final_action_id=graph_input_final_action_id,
                    initial_sim_steps_tensor=graph_input_sim_steps_tensor,
                )
        current_stream.wait_stream(warmup_stream)
        restore_initial_refs()
        assert self._inactive_action_id is not None
        graph_input_final_action_id.copy_(self._inactive_action_id)
        graph_input_sim_steps_tensor.zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tensors = self._rollout_tensors(
                control_program=control_program,
                gates=gates,
                scoring_profiles=scoring_profiles,
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
                allow_host_sync=False,
                profile_ranges=False,
                max_steps=chunk_steps,
                accumulator=capture_accumulator,
                initial_final_action_id=graph_input_final_action_id,
                initial_sim_steps_tensor=graph_input_sim_steps_tensor,
            )
        tail_graph: Any | None = None
        tail_tensors: _GpuRolloutTensors | None = None
        if tail_steps > 0:
            restore_initial_refs()
            _copy_score_accumulator_(graph_input_accumulator, create_score_accumulator(graph_input_state))
            graph_input_final_action_id.copy_(self._inactive_action_id)
            graph_input_sim_steps_tensor.zero_()
            tail_capture_accumulator = _score_accumulator_view(graph_input_accumulator)
            tail_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(tail_graph):
                tail_tensors = self._rollout_tensors(
                    control_program=control_program,
                    gates=gates,
                    scoring_profiles=scoring_profiles,
                    frontier_focus_start_m=frontier_focus_start_m,
                    frontier_focus_end_m=frontier_focus_end_m,
                    allow_host_sync=False,
                    profile_ranges=False,
                    max_steps=tail_steps,
                    accumulator=tail_capture_accumulator,
                    initial_final_action_id=graph_input_final_action_id,
                    initial_sim_steps_tensor=graph_input_sim_steps_tensor,
                )
        torch.cuda.synchronize(self.track.device)
        capture_seconds = time.perf_counter() - started
        return GpuCapturedChunkRollout(
            graph=graph,
            input_state=graph_input_state,
            input_last_segment_idx=graph_input_segment_idx,
            input_segment_start_progress_m=graph_input_start_progress,
            input_segment_target_progress_m=graph_input_target_progress,
            input_accumulator=graph_input_accumulator,
            input_final_action_id=graph_input_final_action_id,
            input_sim_steps_tensor=graph_input_sim_steps_tensor,
            control_program=control_program,
            scoring_profiles=scoring_profiles,
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
            tensors=tensors,
            chunk_steps=chunk_steps,
            replay_count=replay_count,
            capture_seconds=capture_seconds,
            tail_graph=tail_graph,
            tail_tensors=tail_tensors,
            tail_steps=tail_steps,
        )

    def replay_cuda_graph_chunks(
        self,
        captured: GpuCapturedChunkRollout,
        *,
        snapshots: list[StateSnapshot] | None,
        control_program: GpuControlProgram,
        gates: Any,
    ) -> GpuRolloutResult:
        self.state = captured.input_state
        self._last_segment_idx = captured.input_last_segment_idx
        self.segment_start_progress_m = captured.input_segment_start_progress_m
        self.segment_target_progress_m = captured.input_segment_target_progress_m
        if snapshots is not None:
            self.reset(
                snapshots,
                target_progress_m=float(gates.target_progress_m),
                terminate_at_target=bool(gates.terminate_at_target_progress),
            )
        _copy_control_program_inputs_(captured.control_program, control_program)
        _copy_score_accumulator_(captured.input_accumulator, create_score_accumulator(captured.input_state))
        assert self._inactive_action_id is not None
        captured.input_final_action_id.copy_(self._inactive_action_id)
        captured.input_sim_steps_tensor.zero_()

        replay_started = time.perf_counter()
        for _ in range(captured.replay_count):
            captured.graph.replay()
            _copy_car_batch_(captured.input_state, captured.tensors.output_state)
            captured.input_last_segment_idx.copy_(captured.tensors.output_last_segment_idx)
            _copy_score_accumulator_(captured.input_accumulator, captured.tensors.accumulator)
            captured.input_final_action_id.copy_(captured.tensors.final_action_id)
            captured.input_sim_steps_tensor.copy_(captured.tensors.sim_steps_tensor)
        if captured.tail_graph is not None and captured.tail_tensors is not None:
            captured.tail_graph.replay()
            _copy_car_batch_(captured.input_state, captured.tail_tensors.output_state)
            captured.input_last_segment_idx.copy_(captured.tail_tensors.output_last_segment_idx)
            _copy_score_accumulator_(captured.input_accumulator, captured.tail_tensors.accumulator)
            captured.input_final_action_id.copy_(captured.tail_tensors.final_action_id)
            captured.input_sim_steps_tensor.copy_(captured.tail_tensors.sim_steps_tensor)
        torch.cuda.synchronize(self.track.device)
        replay_elapsed = time.perf_counter() - replay_started
        profile_scores = score_profiles_batch(
            captured.input_accumulator,
            profiles=captured.scoring_profiles,
            target_progress_m=float(gates.target_progress_m),
            terminate_at_target_progress=bool(gates.terminate_at_target_progress),
            frontier_focus_start_m=captured.frontier_focus_start_m,
            frontier_focus_end_m=captured.frontier_focus_end_m,
        )
        sim_steps = int(captured.input_sim_steps_tensor.detach().cpu().item())
        return GpuRolloutResult(
            accumulator=captured.input_accumulator,
            profile_scores=profile_scores,
            rollout_seconds=replay_elapsed,
            steps_executed=captured.chunk_steps * captured.replay_count + captured.tail_steps,
            sim_steps=sim_steps,
            control_kind=captured.control_program.kind,
            action_names=captured.control_program.action_names,
            final_action_id=captured.input_final_action_id,
            host_sync_count=0,
            kernel_backend="pytorch_cuda_graph_chunked",
            graph_replay_seconds=replay_elapsed,
            graph_replay_count=captured.replay_count + (1 if captured.tail_steps > 0 else 0),
        )

    def _rollout_tensors(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        allow_host_sync: bool,
        profile_ranges: bool,
        max_steps: int | None = None,
        accumulator: GpuScoreAccumulator | None = None,
        initial_final_action_id: torch.Tensor | None = None,
        initial_sim_steps_tensor: torch.Tensor | None = None,
        step_backend: str = "torch",
    ) -> _GpuRolloutTensors:
        if self.state is None or self.segment_start_progress_m is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout().")
        if step_backend not in {"torch", "warp_open"}:
            raise ValueError(f"Unknown GPU rollout step backend {step_backend!r}.")
        warp_open_step: Any | None = None
        score_accumulator_update: Any = update_score_accumulator_static
        if step_backend == "warp_open":
            from f1rl.gpu_fused_warp import (
                open_step_warp_batch,
                update_score_accumulator_warp_batch,
            )

            warp_open_step = open_step_warp_batch
            score_accumulator_update = update_score_accumulator_warp_batch
        acc = accumulator if accumulator is not None else create_score_accumulator(self.state)
        steps_executed = 0
        assert self._inactive_action_id is not None
        assert self._zero_float is not None
        final_action_id = initial_final_action_id if initial_final_action_id is not None else self._inactive_action_id
        zero_float = self._zero_float
        host_sync_count = 0
        steps_to_run = self.sim_config.max_steps if max_steps is None else max(0, int(max_steps))
        with torch.inference_mode():
            sim_steps_tensor = (
                initial_sim_steps_tensor
                if initial_sim_steps_tensor is not None
                else torch.zeros((), device=self.track.device, dtype=torch.int64)
            )
            for step_index in range(steps_to_run):
                active = self._active()
                if (
                    allow_host_sync
                    and not self.disable_early_stop
                    and step_index > 0
                    and step_index % self.active_check_interval == 0
                ):
                    host_sync_count += 1
                    if not bool(active.any().detach().cpu().item()):
                        break
                if profile_ranges:
                    with torch.profiler.record_function("gpu_rollout_controls"):
                        throttle, brake, steer, action_id, diagnostics = self._program_controls(
                            control_program,
                            gates=gates,
                            control_backend="warp" if step_backend == "warp_open" else "torch",
                        )
                        final_action_id = torch.where(active, action_id, final_action_id)
                    with torch.profiler.record_function("gpu_rollout_step"):
                        if warp_open_step is None:
                            step_diagnostics, collided, off_track, telemetry_valid_lap = self._step(
                                throttle=throttle,
                                brake=brake,
                                steer=steer,
                                active=active,
                                gates=gates,
                            )
                        else:
                            assert self._last_segment_idx is not None
                            step_diagnostics, collided, off_track, telemetry_valid_lap = warp_open_step(
                                self.state,
                                last_segment_idx=self._last_segment_idx,
                                active=active,
                                throttle=throttle,
                                brake=brake,
                                steer=steer,
                                params=self.params,
                                track=self.track,
                                sim_config=self.sim_config,
                                segment_target_progress_m=self.segment_target_progress_m,
                                gates=gates,
                                collision_check=self.collision_check,
                                collision_mode=self.collision_mode,
                            )
                        scoring_diagnostics = _scoring_diagnostics(
                            control_diagnostics=diagnostics,
                            step_diagnostics=step_diagnostics,
                            telemetry_valid_lap=telemetry_valid_lap,
                        )
                    with torch.profiler.record_function("gpu_rollout_scoring"):
                        score_accumulator_update(
                            acc,
                            state=self.state,
                            diagnostics=scoring_diagnostics,
                            throttle=throttle,
                            brake=brake,
                            steer=steer,
                            active=active,
                            collided=collided,
                            off_track=off_track,
                            config=self.sim_config,
                            zero=zero_float,
                        )
                else:
                    throttle, brake, steer, action_id, diagnostics = self._program_controls(
                        control_program,
                        gates=gates,
                        control_backend="warp" if step_backend == "warp_open" else "torch",
                    )
                    final_action_id = torch.where(active, action_id, final_action_id)
                    if warp_open_step is None:
                        step_diagnostics, collided, off_track, telemetry_valid_lap = self._step(
                            throttle=throttle,
                            brake=brake,
                            steer=steer,
                            active=active,
                            gates=gates,
                        )
                    else:
                        assert self._last_segment_idx is not None
                        step_diagnostics, collided, off_track, telemetry_valid_lap = warp_open_step(
                            self.state,
                            last_segment_idx=self._last_segment_idx,
                            active=active,
                            throttle=throttle,
                            brake=brake,
                            steer=steer,
                            params=self.params,
                            track=self.track,
                            sim_config=self.sim_config,
                            segment_target_progress_m=self.segment_target_progress_m,
                            gates=gates,
                            collision_check=self.collision_check,
                            collision_mode=self.collision_mode,
                        )
                    scoring_diagnostics = _scoring_diagnostics(
                        control_diagnostics=diagnostics,
                        step_diagnostics=step_diagnostics,
                        telemetry_valid_lap=telemetry_valid_lap,
                    )
                    score_accumulator_update(
                        acc,
                        state=self.state,
                        diagnostics=scoring_diagnostics,
                        throttle=throttle,
                        brake=brake,
                        steer=steer,
                        active=active,
                        collided=collided,
                        off_track=off_track,
                        config=self.sim_config,
                        zero=zero_float,
                    )
                steps_executed += 1
                sim_steps_tensor = sim_steps_tensor + active.to(dtype=torch.int64).sum()
        assert self.state is not None
        assert self._last_segment_idx is not None
        score_profiles_fn: Any = score_profiles_batch
        if step_backend == "warp_open":
            from f1rl.gpu_fused_warp import score_profiles_warp_batch

            score_profiles_fn = score_profiles_warp_batch
        if profile_ranges:
            with torch.profiler.record_function("gpu_rollout_profile_scores"):
                profile_scores = score_profiles_fn(
                    acc,
                    profiles=scoring_profiles,
                    target_progress_m=float(gates.target_progress_m),
                    terminate_at_target_progress=bool(gates.terminate_at_target_progress),
                    frontier_focus_start_m=frontier_focus_start_m,
                    frontier_focus_end_m=frontier_focus_end_m,
                )
        else:
            profile_scores = score_profiles_fn(
                acc,
                profiles=scoring_profiles,
                target_progress_m=float(gates.target_progress_m),
                terminate_at_target_progress=bool(gates.terminate_at_target_progress),
                frontier_focus_start_m=frontier_focus_start_m,
                frontier_focus_end_m=frontier_focus_end_m,
            )
        return _GpuRolloutTensors(
            accumulator=acc,
            profile_scores=profile_scores,
            output_state=self.state,
            output_last_segment_idx=self._last_segment_idx,
            final_action_id=final_action_id,
            sim_steps_tensor=sim_steps_tensor,
            steps_executed=steps_executed,
            host_sync_count=host_sync_count,
        )


def _tensor_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def gpu_rollout_rows(
    *,
    result: GpuRolloutResult,
    candidates: list[Any],
    snapshots: list[StateSnapshot],
    generation: int,
    seed: int,
    scoring_profiles: tuple[str, ...],
    target_progress_m: float,
    max_steps: int,
    genome_to_dict: Any,
    physics_model: str = "v1",
    physics_version: str | None = None,
    physics_calibration_id: str | None = None,
    candidate_index_offset: int = 0,
    row_detail: str = "full",
) -> list[dict[str, Any]]:
    acc = result.accumulator
    primary_profile = scoring_profiles[0]
    compact = row_detail == "compact"
    primary_scores = _tensor_list(result.profile_scores[primary_profile])
    profile_score_values = {profile: _tensor_list(scores) for profile, scores in result.profile_scores.items()}
    reasons = termination_reason_strings(acc.final_termination_reason_id)
    best_progress = _tensor_list(acc.best_progress_m)
    final_progress = _tensor_list(acc.final_progress_m)
    start_progress = _tensor_list(acc.start_progress_m)
    start_speed_kph = _tensor_list(acc.start_speed_kph)
    elapsed = _tensor_list(acc.final_sim_time_s)
    pace_kph = _tensor_list(
        (acc.best_progress_m - acc.start_progress_m) / torch.clamp(acc.final_sim_time_s, min=1e-6) * 3.6
    )
    avg_speed_300 = _tensor_list(
        torch.where(
            acc.speed_count_first_300_m > 0.0,
            acc.speed_sum_first_300_m / torch.clamp(acc.speed_count_first_300_m, min=1.0),
            acc.final_speed_kph,
        )
    )
    avg_speed_450 = _tensor_list(
        torch.where(
            acc.speed_count_first_450_m > 0.0,
            acc.speed_sum_first_450_m / torch.clamp(acc.speed_count_first_450_m, min=1.0),
            acc.final_speed_kph,
        )
    )
    avg_brake_300: list[Any] = []
    avg_brake_450: list[Any] = []
    if not compact:
        avg_brake_300 = _tensor_list(
            torch.where(
                acc.brake_count_first_300_m > 0.0,
                acc.brake_sum_first_300_m / torch.clamp(acc.brake_count_first_300_m, min=1.0),
                acc.final_brake,
            )
        )
        avg_brake_450 = _tensor_list(
            torch.where(
                acc.brake_count_first_450_m > 0.0,
                acc.brake_sum_first_450_m / torch.clamp(acc.brake_count_first_450_m, min=1.0),
                acc.final_brake,
            )
        )
    time_300_values = _tensor_list(acc.time_to_300_m)
    time_450_values = _tensor_list(acc.time_to_450_m)
    final_action_ids = [int(value) for value in _tensor_list(result.final_action_id)]
    if compact:
        compact_values = {
            "speed_kph": _tensor_list(acc.final_speed_kph),
            "lateral_error_m": _tensor_list(acc.final_lateral_error_m),
            "heading_error_deg": _tensor_list(acc.final_heading_error_deg),
            "yaw_rate_rps": _tensor_list(acc.final_yaw_rate_rps),
            "steering": _tensor_list(acc.final_steering),
            "step_index": _tensor_list(acc.final_step_index),
            "valid_lap": _tensor_list(acc.final_valid_lap),
            "finish_crossed": _tensor_list(acc.final_finish_crossed),
            "completed_lap": _tensor_list(acc.final_completed_lap),
            "segment_complete": _tensor_list(acc.final_segment_complete),
            "collided": _tensor_list(acc.final_collided),
            "off_track": _tensor_list(acc.final_off_track),
        }
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_index = candidate_index_offset + index
            row_seed = seed + generation * 1_000_000 + candidate_index
            final_action_id = final_action_ids[index]
            if final_action_id == -2:
                final_action_name = "continuous"
            elif 0 <= final_action_id < len(result.action_names):
                final_action_name = result.action_names[final_action_id]
            else:
                final_action_name = result.control_kind
            segment_complete = bool(compact_values["segment_complete"][index])
            target_reached = float(best_progress[index]) >= float(target_progress_m)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "generation": generation,
                    "seed": row_seed,
                    "score": float(primary_scores[index]),
                    "primary_scoring_profile": primary_profile,
                    "profile_scores": {
                        profile: float(values[index]) for profile, values in profile_score_values.items()
                    },
                    "snapshot_index": int(candidate.snapshot_index),
                    "start_snapshot_id": snapshots[int(candidate.snapshot_index)].id,
                    "start_progress_m": float(start_progress[index]),
                    "start_speed_kph": float(start_speed_kph[index]),
                    "genome": genome_to_dict(candidate.genome),
                    "lineage": dict(candidate.lineage),
                    "physics_model": physics_model,
                    "physics_version": physics_version,
                    "physics_calibration_id": physics_calibration_id,
                    "best_progress_m": float(best_progress[index]),
                    "final_progress_m": float(final_progress[index]),
                    "remaining_m": max(0.0, float(target_progress_m) - float(best_progress[index])),
                    "segment_complete": bool(segment_complete or target_reached),
                    "target_reached": bool(target_reached),
                    "sim_segment_complete": segment_complete,
                    "completed_lap": bool(compact_values["completed_lap"][index]),
                    "valid_lap": bool(compact_values["valid_lap"][index]),
                    "finish_crossed": bool(compact_values["finish_crossed"][index]),
                    "termination_reason": reasons[index],
                    "collided": bool(compact_values["collided"][index]),
                    "off_track": bool(compact_values["off_track"][index]),
                    "final_speed_kph": float(compact_values["speed_kph"][index]),
                    "final_lateral_error_m": float(compact_values["lateral_error_m"][index]),
                    "final_heading_error_deg": float(compact_values["heading_error_deg"][index]),
                    "final_yaw_rate_rps": float(compact_values["yaw_rate_rps"][index]),
                    "final_steering": float(compact_values["steering"][index]),
                    "final_action_id": final_action_id,
                    "final_action_name": final_action_name,
                    "max_steps": int(max_steps),
                    "steps": int(compact_values["step_index"][index]),
                    "elapsed_s": float(elapsed[index]),
                    "pace_mps": float(pace_kph[index]) / 3.6,
                    "pace_kph": float(pace_kph[index]),
                    "time_to_300_m": None if float(time_300_values[index]) < 0.0 else float(time_300_values[index]),
                    "time_to_450_m": None if float(time_450_values[index]) < 0.0 else float(time_450_values[index]),
                    "avg_speed_first_300_m": float(avg_speed_300[index]),
                    "avg_speed_first_450_m": float(avg_speed_450[index]),
                    "backend": "gpu",
                    "gpu_verified": False,
                    "gpu_score": float(primary_scores[index]),
                    "gpu_row_detail": "compact",
                    "gpu_rollout_seconds": result.rollout_seconds,
                }
            )
        return rows
    final_values = {
        "x": _tensor_list(acc.final_x),
        "y": _tensor_list(acc.final_y),
        "heading_deg": _tensor_list(acc.final_heading_deg),
        "speed_mps": _tensor_list(acc.final_speed_mps),
        "speed_kph": _tensor_list(acc.final_speed_kph),
        "raw_progress_m": _tensor_list(acc.final_raw_progress_m),
        "monotonic_progress_m": final_progress,
        "lateral_error_m": _tensor_list(acc.final_lateral_error_m),
        "heading_error_deg": _tensor_list(acc.final_heading_error_deg),
        "yaw_rate_rps": _tensor_list(acc.final_yaw_rate_rps),
        "curvature_rad_per_m": _tensor_list(acc.final_curvature_rad_per_m),
        "throttle": _tensor_list(acc.final_throttle),
        "brake": _tensor_list(acc.final_brake),
        "steering": _tensor_list(acc.final_steering),
        "step_index": _tensor_list(acc.final_step_index),
        "sim_time_s": elapsed,
        "missed_checkpoint_count": _tensor_list(acc.final_missed_checkpoint_count),
        "valid_lap": _tensor_list(acc.final_valid_lap),
        "finish_crossed": _tensor_list(acc.final_finish_crossed),
        "completed_lap": _tensor_list(acc.final_completed_lap),
        "segment_complete": _tensor_list(acc.final_segment_complete),
        "collided": _tensor_list(acc.final_collided),
        "off_track": _tensor_list(acc.final_off_track),
        "terminated": _tensor_list(acc.final_terminated),
        "truncated": _tensor_list(acc.final_truncated),
        "target_speed_kph": _tensor_list(acc.final_target_speed_kph),
        "near_target_speed_kph": _tensor_list(acc.final_near_target_speed_kph),
        "min_future_target_speed_kph": _tensor_list(acc.final_min_future_target_speed_kph),
        "target_speed_drop_kph": _tensor_list(acc.final_target_speed_drop_kph),
        "target_speed_drop_norm": _tensor_list(acc.final_target_speed_drop_norm),
        "brake_demand": _tensor_list(acc.final_brake_demand),
        "future_brake_demand": _tensor_list(acc.final_future_brake_demand),
        "brake_gate_proximity": _tensor_list(acc.final_brake_gate_proximity),
        "braking_gate_distance_m": _tensor_list(acc.final_braking_gate_distance_m),
        "brake_gate_distance_norm": _tensor_list(acc.final_brake_gate_distance_norm),
        "lookahead_abs_max": _tensor_list(acc.final_lookahead_abs_max),
    }
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_index = candidate_index_offset + index
        row_seed = seed + generation * 1_000_000 + candidate_index
        final_action_id = final_action_ids[index]
        if final_action_id == -2:
            final_action_name = "continuous"
        elif 0 <= final_action_id < len(result.action_names):
            final_action_name = result.action_names[final_action_id]
        else:
            final_action_name = result.control_kind
        final_row = {
            key: values[index]
            for key, values in final_values.items()
        }
        final_row.update(
            {
                "progress_delta_m": max(0.0, float(final_progress[index]) - float(start_progress[index])),
                "termination_reason": reasons[index],
                "action_id": final_action_id,
                "action_name": final_action_name,
                "physics_model": physics_model,
                "physics_version": physics_version,
                "physics_calibration_id": physics_calibration_id,
            }
        )
        target_reached = float(best_progress[index]) >= float(target_progress_m)
        row = {
            "candidate_index": candidate_index,
            "generation": generation,
            "seed": row_seed,
            "score": float(primary_scores[index]),
            "primary_scoring_profile": primary_profile,
            "profile_scores": {
                profile: float(values[index]) for profile, values in profile_score_values.items()
            },
            "snapshot_index": int(candidate.snapshot_index),
            "start_snapshot_id": snapshots[int(candidate.snapshot_index)].id,
            "start_progress_m": float(start_progress[index]),
            "start_speed_kph": float(start_speed_kph[index]),
            "genome": genome_to_dict(candidate.genome),
            "lineage": dict(candidate.lineage),
            "physics_model": physics_model,
            "physics_version": physics_version,
            "physics_calibration_id": physics_calibration_id,
            "best_progress_m": float(best_progress[index]),
            "final_progress_m": float(final_progress[index]),
            "remaining_m": max(0.0, float(target_progress_m) - float(best_progress[index])),
            "segment_complete": bool(final_row["segment_complete"] or target_reached),
            "target_reached": bool(target_reached),
            "sim_segment_complete": bool(final_row["segment_complete"]),
            "completed_lap": bool(final_row["completed_lap"]),
            "valid_lap": bool(final_row["valid_lap"]),
            "termination_reason": reasons[index],
            "collided": bool(final_row["collided"]),
            "off_track": bool(final_row["off_track"]),
            "final_speed_kph": final_row["speed_kph"],
            "final_lateral_error_m": final_row["lateral_error_m"],
            "final_heading_error_deg": final_row["heading_error_deg"],
            "final_yaw_rate_rps": final_row["yaw_rate_rps"],
            "final_steering": final_row["steering"],
            "final_row": final_row,
            "max_steps": int(max_steps),
            "steps": int(final_row["step_index"]),
            "elapsed_s": float(elapsed[index]),
            "pace_mps": float(pace_kph[index]) / 3.6,
            "pace_kph": float(pace_kph[index]),
            "time_to_300_m": None if float(time_300_values[index]) < 0.0 else float(time_300_values[index]),
            "time_to_450_m": None if float(time_450_values[index]) < 0.0 else float(time_450_values[index]),
            "avg_speed_first_300_m": float(avg_speed_300[index]),
            "avg_speed_first_450_m": float(avg_speed_450[index]),
            "avg_brake_first_300_m": float(avg_brake_300[index]),
            "avg_brake_first_450_m": float(avg_brake_450[index]),
            "backend": "gpu",
            "gpu_verified": False,
            "gpu_score": float(primary_scores[index]),
            "gpu_profile_scores": {
                profile: float(values[index]) for profile, values in profile_score_values.items()
            },
            "gpu_row_detail": "full",
            "gpu_rollout_seconds": result.rollout_seconds,
        }
        rows.append(row)
    return rows
