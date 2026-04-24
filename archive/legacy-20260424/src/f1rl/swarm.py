"""Torch-native swarm diagnostics and telemetry export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1rl.artifacts import new_run_paths, resolve_checkpoint
from f1rl.controllers import ScriptedController
from f1rl.gpu_renderer import PygletSwarmRenderer
from f1rl.inference import (
    deep_merge_dicts,
    load_inference_policy,
    training_env_config_for_checkpoint,
)
from f1rl.stages import build_stage_env_overrides, logical_car_target
from f1rl.torch_runtime import TERMINATION_REASONS, TelemetryThresholds, TorchSimBatch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batched swarm of independent cars.")
    parser.add_argument("--policy", choices=["random", "scripted", "checkpoint"], default="checkpoint")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--cars", type=int, default=32)
    parser.add_argument("--swarm-stage", choices=["competence", "lap", "stability", "performance"], default="competence")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--spawn-jitter-px", type=float, default=0.0)
    parser.add_argument("--spawn-heading-jitter-deg", type=float, default=0.0)
    parser.add_argument("--render-scale", type=float, default=0.6)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--export-clips", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser.parse_args(argv)


def _base_env_config(args: argparse.Namespace, *, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = getattr(args, "steps", None) or build_stage_env_overrides(args.swarm_stage)["max_steps"]
    config = deep_merge_dicts(dict(base_config or {}), build_stage_env_overrides(args.swarm_stage))
    config["max_steps"] = int(steps)
    config["headless"] = bool(args.headless)
    config["render_mode"] = None
    config.setdefault("render", {})
    config["render"]["enabled"] = bool(args.render and not args.headless)
    config["render"]["export_frames"] = False
    config["render"]["draw_hud"] = True
    spawn_jitter_px = float(getattr(args, "spawn_jitter_px", 0.0))
    spawn_heading_jitter_deg = float(getattr(args, "spawn_heading_jitter_deg", 0.0))
    if spawn_jitter_px > 0.0:
        config["spawn_position_jitter_px"] = spawn_jitter_px
    if spawn_heading_jitter_deg > 0.0:
        config["spawn_heading_jitter_deg"] = spawn_heading_jitter_deg
    return config


def _info_rows(simulator: TorchSimBatch, telemetry: dict[str, torch.Tensor]) -> list[dict[str, float | int | bool | str]]:
    rows: list[dict[str, float | int | bool | str]] = []
    cpu = {key: value.detach().cpu().tolist() for key, value in telemetry.items()}
    for idx in range(simulator.num_cars):
        rows.append(
            {
                "x": float(cpu["x"][idx]),
                "y": float(cpu["y"][idx]),
                "heading_deg": float(cpu["heading_deg"][idx]),
                "speed": float(cpu["speed"][idx]),
                "speed_kph": float(cpu["speed_kph"][idx]),
                "lap_count": int(cpu["lap_count"][idx]),
                "next_goal_idx": int(cpu["next_goal_idx"][idx]),
                "progress_delta": int(cpu["progress_delta"][idx]),
                "collided": bool(cpu["termination_code"][idx] == TERMINATION_REASONS.index("collision")),
                "terminated_reason": TERMINATION_REASONS[int(cpu["termination_code"][idx])],
            }
        )
    return rows


def _scripted_actions(controllers: list[ScriptedController], observations: torch.Tensor, info_rows: list[dict[str, Any]]) -> torch.Tensor:
    actions = []
    for idx, controller in enumerate(controllers):
        obs = observations[idx].detach().cpu().numpy()
        action = controller.action(obs, info_rows[idx])
        actions.append(action)
    return torch.from_numpy(np.stack(actions).astype(np.float32))


def _build_live_rows(telemetry_cpu: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_cars = int(telemetry_cpu["x"].shape[0])
    for idx in range(num_cars):
        rows.append(
            {
                "car_index": idx,
                "x": float(telemetry_cpu["x"][idx].item()),
                "y": float(telemetry_cpu["y"][idx].item()),
                "heading_deg": float(telemetry_cpu["heading_deg"][idx].item()),
                "speed": float(telemetry_cpu["speed"][idx].item()),
                "speed_kph": float(telemetry_cpu["speed_kph"][idx].item()),
                "current_checkpoint_index": int(telemetry_cpu["current_checkpoint_index"][idx].item()),
                "alive": bool(telemetry_cpu["alive"][idx].item()),
                "moving": bool(telemetry_cpu["moving"][idx].item()),
                "termination_reason": TERMINATION_REASONS[int(telemetry_cpu["termination_code"][idx].item())],
            }
        )
    return rows


def run_swarm(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    run_paths = new_run_paths(prefix=f"swarm-{args.swarm_stage}")
    print(
        f"[swarm] starting policy={args.policy} cars={args.cars} stage={args.swarm_stage} "
        f"render={args.render and not args.headless} device={args.device}",
        flush=True,
    )
    policy = None
    policy_device = "cpu"
    env_config = _base_env_config(args)
    if args.policy == "checkpoint":
        policy = load_inference_policy(resolve_checkpoint(args.checkpoint), device=args.device)
        env_config = _base_env_config(args, base_config=training_env_config_for_checkpoint(policy.checkpoint_path))
        policy_device = policy.device

    simulator = TorchSimBatch(
        env_config=env_config,
        num_cars=args.cars,
        device=args.device,
        telemetry_thresholds=TelemetryThresholds(),
    )
    print(f"[swarm] simulator ready sim_device={simulator.device} observation_dim={simulator.observation_dim}", flush=True)
    observations = simulator.reset(seed=args.seed)
    print("[swarm] reset complete", flush=True)
    controllers = (
        [ScriptedController(sensor_count=simulator.sensor_count, goals=simulator.track.definition.goals.copy()) for _ in range(args.cars)]
        if args.policy == "scripted"
        else None
    )
    if controllers is not None:
        for controller in controllers:
            controller.reset()

    telemetry_rows: list[dict[str, Any]] = []
    car_telemetry_rows: list[dict[str, Any]] = []
    checkpoint_events: list[dict[str, float | int]] = []
    trace_history = [[] for _ in range(simulator.num_cars)]
    started = time.perf_counter()

    renderer_backend = "headless"
    renderer: PygletSwarmRenderer | None = None
    if args.render and not args.headless:
        print(
            f"[swarm] initializing live GPU renderer cars={args.cars} stage={args.swarm_stage} "
            f"scale={args.render_scale} device={args.device}",
            flush=True,
        )
        renderer = PygletSwarmRenderer(
            simulator.track.definition,
            display_scale=args.render_scale,
            caption="F1RL GPU Swarm",
            sprite_heading_offset_deg=simulator.config.track.car_sprite_heading_offset_deg,
            show_track_texture=True,
        )
        renderer_backend = renderer.backend
        print(f"[swarm] renderer ready backend={renderer_backend}", flush=True)

    try:
        for _ in range(int(env_config["max_steps"])):
            if args.policy == "random":
                actions = torch.rand((simulator.num_cars, simulator.action_dim), device=simulator.device) * 2.0 - 1.0
            elif args.policy == "scripted":
                assert controllers is not None
                info_rows = _info_rows(simulator, simulator.snapshot() | {"next_goal_idx": simulator.next_goal_idx, "progress_delta": torch.zeros_like(simulator.next_goal_idx), "speed": simulator.speeds, "speed_kph": simulator.speeds * 3.6, "heading_deg": simulator.headings_deg, "lap_count": simulator.lap_count, "termination_code": simulator.reason_codes, "x": simulator.positions[:, 0], "y": simulator.positions[:, 1]})
                actions = _scripted_actions(controllers, observations, info_rows).to(simulator.device)
            else:
                assert policy is not None
                actions = policy.compute_actions(observations, deterministic=args.deterministic)

            step = simulator.step(actions)
            observations = step.observations
            checkpoint_events.extend(step.checkpoint_events)
            telemetry = step.telemetry
            cpu = {key: value.detach().cpu() for key, value in telemetry.items()}

            for idx in range(simulator.num_cars):
                trace_history[idx].append((float(cpu["x"][idx].item()), float(cpu["y"][idx].item())))
                car_telemetry_rows.append(
                    {
                        "step": int(simulator.step_count[idx].item()),
                        "car_index": idx,
                        "x": float(cpu["x"][idx].item()),
                        "y": float(cpu["y"][idx].item()),
                        "heading_deg": float(cpu["heading_deg"][idx].item()),
                        "speed": float(cpu["speed"][idx].item()),
                        "speed_kph": float(cpu["speed_kph"][idx].item()),
                        "distance_travelled": float(cpu["distance_travelled"][idx].item()),
                        "lap_count": int(cpu["lap_count"][idx].item()),
                        "next_goal_idx": int(cpu["next_goal_idx"][idx].item()),
                        "current_checkpoint_index": int(cpu["current_checkpoint_index"][idx].item()),
                        "progress_delta": int(cpu["progress_delta"][idx].item()),
                        "positive_progress_total": int(cpu["positive_progress_total"][idx].item()),
                        "time_since_last_checkpoint": int(cpu["time_since_last_checkpoint"][idx].item()),
                        "throttle": float(cpu["throttle"][idx].item()),
                        "brake": float(cpu["brake"][idx].item()),
                        "steering": float(cpu["steering"][idx].item()),
                        "minimum_wall_distance_ratio": float(cpu["min_wall_distance_ratio"][idx].item()),
                        "heading_error_norm": float(cpu["heading_error_norm"][idx].item()),
                        "goal_distance_norm": float(cpu["goal_distance_norm"][idx].item()),
                        "lateral_offset_norm": float(cpu["lateral_offset_norm"][idx].item()),
                        "no_progress_steps": int(cpu["no_progress_steps"][idx].item()),
                        "reverse_speed_steps": int(cpu["reverse_speed_steps"][idx].item()),
                        "reverse_progress_events": int(cpu["reverse_progress_events"][idx].item()),
                        "heading_away_steps": int(cpu["heading_away_steps"][idx].item()),
                        "wall_hugging_steps": int(cpu["wall_hugging_steps"][idx].item()),
                        "moving": bool(cpu["moving"][idx].item()),
                        "alive": bool(cpu["alive"][idx].item()),
                        "termination_reason": TERMINATION_REASONS[int(cpu["termination_code"][idx].item())],
                    }
                )

            checkpoint_death_distribution: dict[str, int] = {}
            for idx in range(simulator.num_cars):
                if bool(cpu["terminated"][idx].item()) or bool(cpu["truncated"][idx].item()):
                    key = str(int(cpu["current_checkpoint_index"][idx].item()))
                    checkpoint_death_distribution[key] = checkpoint_death_distribution.get(key, 0) + 1

            aggregate_row = {
                "step": int(simulator.step_count.max().item()),
                "alive_cars": int(cpu["alive"].sum().item()),
                "moving_cars": int(cpu["moving"].sum().item()),
                "avg_current_speed": float(cpu["speed"].mean().item()),
                "avg_current_speed_kph": float(cpu["speed_kph"].mean().item()),
                "avg_distance_travelled": float(cpu["distance_travelled"].mean().item()),
                "max_distance_travelled": float(cpu["distance_travelled"].max().item()),
                "avg_checkpoints_reached": float(cpu["positive_progress_total"].float().mean().item()),
                "max_checkpoints_reached": int(cpu["positive_progress_total"].max().item()),
                "dominant_termination_reason": max(
                    (
                        (TERMINATION_REASONS[int(code)], int((cpu["termination_code"] == code).sum().item()))
                        for code in cpu["termination_code"].unique().tolist()
                    ),
                    key=lambda pair: pair[1],
                    default=("active", 0),
                )[0],
                "best_car_index": int(torch.argmax(cpu["distance_travelled"]).item()),
                "checkpoint_death_distribution": dict(checkpoint_death_distribution),
            }
            telemetry_rows.append(aggregate_row)

            top_k_indices = sorted(
                range(simulator.num_cars),
                key=lambda idx: float(simulator.distance_travelled[idx].item()),
                reverse=True,
            )[: max(1, args.top_k)]
            if renderer is not None:
                live_rows = _build_live_rows(cpu)
                focus_index = aggregate_row["best_car_index"]
                sensor_rays = [
                    tuple(float(value) for value in ray)
                    for ray in simulator.build_sensor_rays()[focus_index].detach().cpu().tolist()
                ]
                keep_open = renderer.draw(
                    live_rows,
                    aggregate_row,
                    top_k_indices,
                    trace_history=trace_history,
                    sensor_rays=sensor_rays,
                    extra_hud_lines=[
                        f"sim: {simulator.device}",
                        f"policy: {policy_device}",
                        f"moving threshold: {simulator.telemetry_thresholds.moving_speed_threshold_kph:.1f} kph",
                        f"focus car: {focus_index}",
                        f"focus speed: {float(live_rows[focus_index]['speed_kph']):.1f} kph",
                    ],
                )
                if not keep_open:
                    break

            if bool((step.terminated | step.truncated).all().item()):
                break
    finally:
        if renderer is not None:
            renderer.close()

    telemetry_json_path = run_paths.metrics / "telemetry.json"
    telemetry_csv_path = run_paths.metrics / "telemetry.csv"
    car_telemetry_json_path = run_paths.metrics / "car_telemetry.json"
    car_telemetry_csv_path = run_paths.metrics / "car_telemetry.csv"
    checkpoint_events_path = run_paths.metrics / "checkpoint_events.json"
    top_k_traces_path = run_paths.metrics / "top_k_traces.json"

    telemetry_json_path.write_text(json.dumps(telemetry_rows, indent=2), encoding="utf-8")
    car_telemetry_json_path.write_text(json.dumps(car_telemetry_rows, indent=2), encoding="utf-8")
    checkpoint_events_path.write_text(json.dumps(checkpoint_events, indent=2), encoding="utf-8")

    if telemetry_rows:
        with telemetry_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(telemetry_rows[0].keys()))
            writer.writeheader()
            writer.writerows(telemetry_rows)
    if car_telemetry_rows:
        with car_telemetry_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(car_telemetry_rows[0].keys()))
            writer.writeheader()
            writer.writerows(car_telemetry_rows)

    top_k_indices = sorted(
        range(simulator.num_cars),
        key=lambda idx: float(simulator.distance_travelled[idx].item()),
        reverse=True,
    )[: max(1, args.top_k)]
    top_k_traces_path.write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "cars": [
                    {
                        "index": idx,
                        "distance_travelled": float(simulator.distance_travelled[idx].item()),
                        "checkpoints_reached": int(simulator.positive_progress_total[idx].item()),
                        "trace": trace_history[idx],
                    }
                    for idx in top_k_indices
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    snapshot = simulator.snapshot()
    failure_histogram: dict[str, int] = {}
    for code in snapshot["termination_code"].detach().cpu().tolist():
        reason = TERMINATION_REASONS[int(code)]
        failure_histogram[reason] = failure_histogram.get(reason, 0) + 1

    summary = {
        "policy": args.policy,
        "backend": "torch_native" if args.policy == "checkpoint" else args.policy,
        "device": policy_device if args.policy == "checkpoint" else str(simulator.device),
        "cars": simulator.num_cars,
        "swarm_stage": args.swarm_stage,
        "logical_cars_target": logical_car_target(args.swarm_stage),
        "elapsed_s": time.perf_counter() - started,
        "aggregate": {
            "completion_rate": float((snapshot["termination_code"] == TERMINATION_REASONS.index("lap_complete")).float().mean().item()),
            "collision_rate": float((snapshot["termination_code"] == TERMINATION_REASONS.index("collision")).float().mean().item()),
            "avg_checkpoints_reached": float(snapshot["positive_progress_total"].float().mean().item()),
            "max_checkpoints_reached": int(snapshot["positive_progress_total"].max().item()),
            "avg_speed": float(snapshot["speed"].mean().item()),
            "avg_current_speed": float(snapshot["speed"].mean().item()),
            "avg_distance_travelled": float(snapshot["distance_travelled"].mean().item()),
            "max_distance_travelled": float(snapshot["distance_travelled"].max().item()),
            "moving_car_fraction": float(
                (
                    torch.abs(snapshot["speed_kph"]) >= simulator.telemetry_thresholds.moving_speed_threshold_kph
                ).float().mean().item()
            ),
            "best_reward": float(simulator.total_reward.max().item()),
            "samples_per_second": float(simulator.step_count.sum().item() / max(time.perf_counter() - started, 1e-6)),
        },
        "failure_histogram": failure_histogram,
        "failure_heatmap": telemetry_rows[-1]["checkpoint_death_distribution"] if telemetry_rows else {},
        "telemetry_json_path": str(telemetry_json_path),
        "telemetry_csv_path": str(telemetry_csv_path),
        "car_telemetry_json_path": str(car_telemetry_json_path),
        "car_telemetry_csv_path": str(car_telemetry_csv_path),
        "checkpoint_events_path": str(checkpoint_events_path),
        "top_k_traces_path": str(top_k_traces_path),
        "runtime": simulator.runtime_metadata(policy_device=policy_device, renderer_backend=renderer_backend),
    }
    summary_path = run_paths.root / "swarm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path, summary = run_swarm(args)
    print(
        f"swarm_complete policy={args.policy} backend={summary['backend']} cars={summary['cars']} "
        f"stage={args.swarm_stage} summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
