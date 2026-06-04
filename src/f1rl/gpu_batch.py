# pyright: reportPrivateImportUsage=false
"""Batched Monza rollout loop for GPU-backed evolution search."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from f1rl.config import SimConfig
from f1rl.gpu_features import braking_gate_tensor, controller_controls_batch, search_features_batch
from f1rl.gpu_physics import apply_physics_batch
from f1rl.gpu_scoring import (
    TERMINATION_REASON_TO_ID,
    GpuScoreAccumulator,
    create_score_accumulator,
    score_profiles_batch,
    termination_reason_strings,
    update_score_accumulator,
)
from f1rl.gpu_track import point_is_drivable_batch, segments_intersect_any_batch, track_errors_batch
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


def _reason_id(name: str) -> int:
    return TERMINATION_REASON_TO_ID[name]


def _mask_float(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


def _mask_int(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


def _mask_bool(mask: torch.Tensor, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, previous)


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
        compile_rollout: bool = False,
        compile_mode: str = "reduce-overhead",
        active_check_interval: int = 16,
    ) -> None:
        self.track = track
        self.sim_config = sim_config
        self.feature_names = feature_names
        self.params: GpuCarParams = gpu_car_params_from_cpu(sim_config.car)
        self.collision_check = collision_check
        self.collision_chunk_size = max(1, int(collision_chunk_size))
        self.active_check_interval = max(1, int(active_check_interval))
        self.braking_gates_m = braking_gate_tensor(device=track.device, dtype=track.dtype)
        self._reason_id_tensors = {
            reason: torch.tensor(reason_id, device=track.device, dtype=torch.int64)
            for reason, reason_id in TERMINATION_REASON_TO_ID.items()
        }
        self.state: GpuCarBatch | None = None
        self.segment_start_progress_m: torch.Tensor | None = None
        self.segment_target_progress_m: torch.Tensor | None = None
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
        self.state = car_batch_from_snapshots(
            snapshots,
            meters_per_pixel=float(self.track.meters_per_pixel.detach().cpu()),
            device=self.track.device,
            dtype=self.track.dtype,
        )
        self.segment_start_progress_m = self.state.monotonic_progress_m.clone()
        if terminate_at_target:
            length = torch.clamp(float(target_progress_m) - self.segment_start_progress_m, min=1.0)
            self.segment_target_progress_m = self.segment_start_progress_m + length
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
        spacing = self.track.checkpoint_spacing_m
        checkpoint_count = max(self.track.checkpoint_count, 1)
        skipped = active & (progress_delta_m > spacing * 1.75)
        skipped_count = torch.clamp(torch.floor(progress_delta_m / spacing).to(torch.int64) - 1, min=1)
        self.state.missed_checkpoint_count += torch.where(skipped, skipped_count, torch.zeros_like(skipped_count))
        self.state.valid_lap = torch.where(skipped, torch.zeros_like(self.state.valid_lap), self.state.valid_lap)

        next_idx = self.state.next_checkpoint_index
        threshold = spacing * next_idx.to(dtype=self.track.dtype)
        due = active & (next_idx < checkpoint_count) & (self.state.monotonic_progress_m + 1e-6 >= threshold)
        expected = (old_progress_m <= threshold) & (threshold <= self.state.monotonic_progress_m + spacing * 0.75)
        lateral_ok = torch.abs(lateral_error_m) <= float(self.sim_config.checkpoint_lateral_limit_m)
        invalid = due & ~(expected & lateral_ok)
        self.state.missed_checkpoint_count += invalid.to(dtype=torch.int64)
        self.state.valid_lap = torch.where(invalid, torch.zeros_like(self.state.valid_lap), self.state.valid_lap)
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

        errors = track_errors_batch(
            self.state.position_xy(),
            self.state.heading_rad,
            self.track,
            previous_progress_px=old_raw_px,
            window_px=float(self.sim_config.local_projection_window_m) / torch.clamp(self.track.meters_per_pixel, min=1e-6),
        )
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
            collided = segments_intersect_any_batch(
                movements,
                self.track.boundary_segments,
                chunk_size=self.collision_chunk_size,
            )
        else:
            collided = torch.zeros_like(active)
        collided &= active
        off_track = active & ~point_is_drivable_batch(self.state.x, self.state.y, self.track)
        no_progress = active & (progress_delta_m <= 1e-4)
        self.state.no_progress_steps = torch.where(
            no_progress,
            self.state.no_progress_steps + 1,
            torch.where(active, torch.zeros_like(self.state.no_progress_steps), self.state.no_progress_steps),
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
            speed_gate_ok = torch.ones_like(active)
            if getattr(gates, "target_min_speed_kph", None) is not None:
                speed_gate_ok &= speed_kph_after >= float(gates.target_min_speed_kph)
            if getattr(gates, "target_max_speed_kph", None) is not None:
                speed_gate_ok &= speed_kph_after <= float(gates.target_max_speed_kph)
            lateral_gate_ok = torch.ones_like(active)
            if getattr(gates, "target_max_lateral_error_m", None) is not None:
                lateral_gate_ok &= torch.abs(errors["lateral_error_m"]) <= float(gates.target_max_lateral_error_m)
            heading_gate_ok = torch.ones_like(active)
            if getattr(gates, "target_max_heading_error_deg", None) is not None:
                heading_gate_ok &= heading_error_abs_deg <= float(gates.target_max_heading_error_deg)
            yaw_rate_gate_ok = torch.ones_like(active)
            if getattr(gates, "target_max_abs_yaw_rate_rps", None) is not None:
                yaw_rate_gate_ok &= torch.abs(self.state.yaw_rate_rps) <= float(gates.target_max_abs_yaw_rate_rps)
            steering_gate_ok = torch.ones_like(active)
            if getattr(gates, "target_max_abs_steering", None) is not None:
                steering_gate_ok &= torch.abs(self.state.steering) <= float(gates.target_max_abs_steering)
            release_gate_ok = torch.ones_like(active)
            if bool(getattr(gates, "segment_require_release", False)):
                release_gate_ok &= self.state.segment_release_observed
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
            min_speed_failed = torch.zeros_like(active)
            if getattr(gates, "target_min_speed_kph", None) is not None:
                min_speed_failed = gate_miss & (speed_kph_after < float(gates.target_min_speed_kph))
            max_speed_failed = torch.zeros_like(active)
            if getattr(gates, "target_max_speed_kph", None) is not None:
                max_speed_failed = gate_miss & (speed_kph_after > float(gates.target_max_speed_kph))
            lateral_failed = torch.zeros_like(active)
            if getattr(gates, "target_max_lateral_error_m", None) is not None:
                lateral_failed = gate_miss & ~lateral_gate_ok
            heading_failed = torch.zeros_like(active)
            if getattr(gates, "target_max_heading_error_deg", None) is not None:
                heading_failed = gate_miss & ~heading_gate_ok
            yaw_failed = torch.zeros_like(active)
            if getattr(gates, "target_max_abs_yaw_rate_rps", None) is not None:
                yaw_failed = gate_miss & ~yaw_rate_gate_ok
            steering_failed = torch.zeros_like(active)
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
        self.state.alive = torch.where(self.state.terminated, torch.zeros_like(self.state.alive), self.state.alive)

        errors["telemetry_valid_lap"] = telemetry_valid_lap
        errors["progress_delta_m"] = progress_delta_m
        return errors, collided, off_track, telemetry_valid_lap

    def _program_controls(
        self,
        program: GpuControlProgram,
        *,
        gates: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        assert self.state is not None
        assert self.segment_start_progress_m is not None
        if program.kind == "controller":
            if program.controller_weights is None:
                raise ValueError("controller control programs require controller_weights")
            features, diagnostics = self._search_features(
                self.state,
                self.track,
                self.sim_config,
                feature_names=self.feature_names,
                segment_start_progress_m=self.segment_start_progress_m,
                segment_target_progress_m=float(gates.target_progress_m),
                braking_gates_m=self.braking_gates_m,
            )
            throttle, brake, steer = self._controller_controls(program.controller_weights, features)
            action_id = torch.full_like(self.state.elapsed_steps, -2)
            return throttle, brake, steer, action_id, diagnostics
        if program.kind not in {"phase", "progress_phase"}:
            raise ValueError(f"Unsupported GPU control program kind {program.kind!r}")
        if program.action_controls is None or program.phase_action_ids is None or program.phase_thresholds is None:
            raise ValueError(f"{program.kind} control programs require action controls and phase tensors")
        if program.kind == "phase":
            phase_index = torch.sum(self.state.elapsed_steps[:, None] >= program.phase_thresholds, dim=1)
        else:
            progress_delta_m = torch.clamp(self.state.monotonic_progress_m - self.segment_start_progress_m, min=0.0)
            phase_index = torch.sum(progress_delta_m[:, None] >= program.phase_thresholds, dim=1)
        phase_index = torch.clamp(phase_index, max=program.phase_action_ids.shape[1] - 1).to(dtype=torch.int64)
        row_index = torch.arange(program.phase_action_ids.shape[0], device=self.track.device, dtype=torch.int64)
        action_id = program.phase_action_ids[row_index, phase_index]
        controls = program.action_controls[action_id]
        diagnostics = track_errors_batch(
            self.state.position_xy(),
            self.state.heading_rad,
            self.track,
            previous_progress_px=self.state.last_raw_progress_px,
            window_px=float(self.sim_config.local_projection_window_m) / torch.clamp(self.track.meters_per_pixel, min=1e-6),
        )
        diagnostics.update(
            {
                "heading_error_deg": diagnostics["heading_error_rad"] * (180.0 / torch.pi),
                "speed_kph": self.state.speed_mps * 3.6,
                "target_speed_kph": torch.zeros_like(self.state.speed_mps),
                "near_target_speed_kph": torch.zeros_like(self.state.speed_mps),
                "min_future_target_speed_kph": torch.zeros_like(self.state.speed_mps),
                "target_speed_drop_kph": torch.zeros_like(self.state.speed_mps),
                "target_speed_drop_norm": torch.zeros_like(self.state.speed_mps),
                "brake_demand": torch.zeros_like(self.state.speed_mps),
                "future_brake_demand": torch.zeros_like(self.state.speed_mps),
                "brake_gate_proximity": torch.zeros_like(self.state.speed_mps),
                "braking_gate_distance_m": torch.zeros_like(self.state.speed_mps),
                "brake_gate_distance_norm": torch.zeros_like(self.state.speed_mps),
                "lookahead_abs_max": torch.abs(diagnostics["heading_error_rad"]),
            }
        )
        return controls[:, 0], controls[:, 1], controls[:, 2], action_id, diagnostics

    def rollout(
        self,
        *,
        control_program: GpuControlProgram,
        gates: Any,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
    ) -> GpuRolloutResult:
        if self.state is None or self.segment_start_progress_m is None:
            raise RuntimeError("GpuMonzaBatch.reset() must be called before rollout().")
        started = time.perf_counter()
        acc = create_score_accumulator(self.state)
        steps_executed = 0
        sim_steps = 0
        final_action_id = torch.full_like(self.state.elapsed_steps, -1)
        with torch.inference_mode():
            sim_steps_tensor = torch.zeros((), device=self.track.device, dtype=torch.int64)
            for step_index in range(self.sim_config.max_steps):
                active = self._active()
                if step_index % self.active_check_interval == 0 and not bool(active.any().detach().cpu().item()):
                    break
                throttle, brake, steer, action_id, diagnostics = self._program_controls(control_program, gates=gates)
                final_action_id = torch.where(active, action_id, final_action_id)
                step_diagnostics, collided, off_track, telemetry_valid_lap = self._step(
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    active=active,
                    gates=gates,
                )
                diagnostics.update(step_diagnostics)
                diagnostics["telemetry_valid_lap"] = telemetry_valid_lap
                update_score_accumulator(
                    acc,
                    state=self.state,
                    diagnostics=diagnostics,
                    throttle=throttle,
                    brake=brake,
                    steer=steer,
                    active=active,
                    collided=collided,
                    off_track=off_track,
                    config=self.sim_config,
                )
                steps_executed += 1
                sim_steps_tensor += active.to(dtype=torch.int64).sum()
        if self.track.device.type == "cuda":
            torch.cuda.synchronize(self.track.device)
        sim_steps = int(sim_steps_tensor.detach().cpu().item())
        profile_scores = score_profiles_batch(
            acc,
            profiles=scoring_profiles,
            target_progress_m=float(gates.target_progress_m),
            terminate_at_target_progress=bool(gates.terminate_at_target_progress),
            frontier_focus_start_m=frontier_focus_start_m,
            frontier_focus_end_m=frontier_focus_end_m,
        )
        if self.track.device.type == "cuda":
            torch.cuda.synchronize(self.track.device)
        elapsed = time.perf_counter() - started
        return GpuRolloutResult(
            accumulator=acc,
            profile_scores=profile_scores,
            rollout_seconds=elapsed,
            steps_executed=steps_executed,
            sim_steps=sim_steps,
            control_kind=control_program.kind,
            action_names=control_program.action_names,
            final_action_id=final_action_id,
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
    candidate_index_offset: int = 0,
) -> list[dict[str, Any]]:
    acc = result.accumulator
    primary_profile = scoring_profiles[0]
    primary_scores = _tensor_list(result.profile_scores[primary_profile])
    profile_score_values = {profile: _tensor_list(scores) for profile, scores in result.profile_scores.items()}
    reasons = termination_reason_strings(acc.final_termination_reason_id)
    best_progress = _tensor_list(acc.best_progress_m)
    final_progress = _tensor_list(acc.final_progress_m)
    start_progress = _tensor_list(acc.start_progress_m)
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
            "start_speed_kph": float(acc.start_speed_kph.detach().cpu().tolist()[index]),
            "genome": genome_to_dict(candidate.genome),
            "lineage": dict(candidate.lineage),
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
            "gpu_rollout_seconds": result.rollout_seconds,
        }
        rows.append(row)
    return rows
