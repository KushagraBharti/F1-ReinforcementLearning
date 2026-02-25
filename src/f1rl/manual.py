"""Manual gameplay entrypoint."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pygame

from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv


def keyboard_action(keys: pygame.key.ScancodeWrapper) -> np.ndarray:
    throttle = 0.0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = 1.0
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        throttle = -1.0

    steering = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steering = -1.0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steering = 1.0

    return np.array([steering, throttle], dtype=np.float32)


def autopilot_action(step: int) -> np.ndarray:
    # Simple periodic steering to keep the vehicle moving for smoke checks.
    steering = float(np.sin(step / 40.0))
    throttle = 0.8
    return np.array([steering, throttle], dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual F1 driving mode")
    parser.add_argument("--max-steps", type=int, default=1200, help="Steps per episode before reset.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--fps", type=int, default=60, help="Render frames per second.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for env reset.")
    parser.add_argument("--headless", action="store_true", help="Disable display window.")
    parser.add_argument(
        "--autodrive",
        action="store_true",
        help="Use deterministic automated controls (recommended for smoke tests).",
    )
    parser.add_argument("--draw-goals", action="store_true", help="Render goal checkpoint lines.")
    parser.add_argument("--draw-track-lines", action="store_true", help="Render contour line overlays.")
    parser.add_argument("--export-frames", action="store_true", help="Save rendered frames under artifacts.")
    return parser.parse_args(argv)


def build_env_config(args: argparse.Namespace) -> dict:
    config = EnvConfig()
    config.max_steps = args.max_steps
    config.render_mode = "human" if not args.headless else "rgb_array" if args.export_frames else None
    config.headless = args.headless
    config.render.enabled = (not args.headless) or args.export_frames
    config.render.fps = args.fps
    config.render.draw_goals = args.draw_goals
    config.render.draw_track_lines = args.draw_track_lines
    config.render.export_frames = args.export_frames
    config.render.frame_prefix = "manual"
    return to_dict(config)


def run_manual(args: argparse.Namespace) -> int:
    env = F1RaceEnv(build_env_config(args))
    try:
        running = True
        total_steps = 0
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            _ = obs, info
            if env.config.render.enabled:
                env.render()

            while running:
                if env.config.render.enabled and env.renderer is not None:
                    events = env.renderer.poll_events()
                else:
                    events = []

                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                if not running:
                    break

                if args.autodrive:
                    action = autopilot_action(total_steps)
                else:
                    if args.headless:
                        action = autopilot_action(total_steps)
                    else:
                        action = keyboard_action(pygame.key.get_pressed())

                _, _, terminated, truncated, _ = env.step(action)
                total_steps += 1
                if env.config.render.enabled:
                    env.render()

                if terminated or truncated:
                    break

        print(
            f"manual_run_complete steps={total_steps} laps={env.car.lap_count} "
            f"reward={env.car.total_reward:.3f}"
        )
        return 0
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_manual(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
