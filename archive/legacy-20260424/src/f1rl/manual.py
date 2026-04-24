"""Manual and scripted single-car gameplay entrypoint."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from f1rl.config import EnvConfig, to_dict
from f1rl.controllers import ScriptedController
from f1rl.gpu_renderer import PygletSwarmRenderer
from f1rl.torch_runtime import TERMINATION_REASONS, TorchSimBatch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual F1 driving mode")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--controller", choices=["human", "scripted"], default=None)
    parser.add_argument("--draw-goals", action="store_true")
    parser.add_argument("--draw-track-lines", action="store_true")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--render-scale", type=float, default=0.6)
    return parser.parse_args(argv)


def build_env_config(args: argparse.Namespace) -> dict:
    config = EnvConfig()
    config.max_steps = args.max_steps
    config.render.enabled = False
    config.render_mode = None
    config.headless = args.headless
    return to_dict(config)


def run_manual(args: argparse.Namespace) -> int:
    simulator = TorchSimBatch(build_env_config(args), num_cars=1, device=args.device)
    controller_mode = args.controller or ("scripted" if args.headless else "human")
    if controller_mode == "human" and args.headless:
        controller_mode = "scripted"
    scripted = ScriptedController(sensor_count=simulator.sensor_count, goals=simulator.track.definition.goals.copy()) if controller_mode == "scripted" else None
    renderer = None
    if not args.headless:
        print(
            f"[manual] initializing live GPU renderer controller={controller_mode} "
            f"scale={args.render_scale} device={args.device}",
            flush=True,
        )
        renderer = PygletSwarmRenderer(
            simulator.track.definition,
            display_scale=args.render_scale,
            caption="F1RL Manual",
            sprite_heading_offset_deg=simulator.config.track.car_sprite_heading_offset_deg,
            show_track_texture=True,
        )
        print(f"[manual] renderer ready backend={renderer.backend}", flush=True)
    trace_history: list[tuple[float, float]] = []
    total_steps = 0
    total_reward = 0.0
    try:
        for episode in range(args.episodes):
            observations = simulator.reset(seed=args.seed + episode)
            trace_history.clear()
            if scripted is not None:
                scripted.reset()
            while True:
                if renderer is not None and not renderer.poll_events():
                    raise KeyboardInterrupt
                if scripted is not None:
                    info = {
                        "x": float(simulator.positions[0, 0].item()),
                        "y": float(simulator.positions[0, 1].item()),
                        "heading_deg": float(simulator.headings_deg[0].item()),
                        "next_goal_idx": int(simulator.next_goal_idx[0].item()),
                    }
                    action_np = scripted.action(observations[0].detach().cpu().numpy(), info)
                else:
                    assert renderer is not None
                    action_np = np.asarray(renderer.keyboard_action(), dtype=np.float32)
                step = simulator.step(np.asarray(action_np, dtype=np.float32)[None, :])
                observations = step.observations
                total_steps += 1
                total_reward += float(step.rewards[0].item())
                trace_history.append((float(simulator.positions[0, 0].item()), float(simulator.positions[0, 1].item())))
                if renderer is not None:
                    telemetry = {key: value.detach().cpu() for key, value in step.telemetry.items()}
                    sensor_rays = [
                        tuple(float(value) for value in ray)
                        for ray in simulator.build_sensor_rays()[0].detach().cpu().tolist()
                    ]
                    aggregate = {
                        "alive_cars": int(telemetry["alive"].sum().item()),
                        "moving_cars": int(telemetry["moving"].sum().item()),
                        "max_checkpoints_reached": int(telemetry["positive_progress_total"].max().item()),
                        "dominant_termination_reason": TERMINATION_REASONS[int(telemetry["termination_code"][0].item())],
                    }
                    renderer.draw(
                        [
                            {
                                "car_index": 0,
                                "x": float(telemetry["x"][0].item()),
                                "y": float(telemetry["y"][0].item()),
                                "heading_deg": float(telemetry["heading_deg"][0].item()),
                                "speed_kph": float(telemetry["speed_kph"][0].item()),
                                "current_checkpoint_index": int(telemetry["current_checkpoint_index"][0].item()),
                                "alive": bool(telemetry["alive"][0].item()),
                                "moving": bool(telemetry["moving"][0].item()),
                            }
                        ],
                        aggregate,
                        [0],
                        trace_history=[trace_history],
                        sensor_rays=sensor_rays,
                        extra_hud_lines=[
                            f"controller: {controller_mode}",
                            f"sim: {simulator.device}",
                            f"speed: {float(telemetry['speed_kph'][0].item()):.1f} kph",
                            f"checkpoint: {int(telemetry['current_checkpoint_index'][0].item())}",
                            f"throttle/brake/steer: {float(telemetry['throttle'][0].item()):.2f} / {float(telemetry['brake'][0].item()):.2f} / {float(telemetry['steering'][0].item()):.2f}",
                            f"wall min: {float(telemetry['min_wall_distance_ratio'][0].item()):.2f}",
                            "keys: arrows / WASD",
                        ],
                        poll_events_first=False,
                    )
                    time.sleep(max(0.0, 1.0 / max(args.fps, 1)))
                if bool((step.terminated | step.truncated)[0].item()):
                    break
        print(
            f"manual_run_complete steps={total_steps} laps={int(simulator.lap_count[0].item())} "
            f"reward={total_reward:.3f} reason={TERMINATION_REASONS[int(simulator.reason_codes[0].item())]}"
        )
        return 0
    finally:
        if renderer is not None:
            renderer.close()


def main(argv: list[str] | None = None) -> int:
    return run_manual(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
