"""Pygame rendering implementation for the F1 environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from f1rl.config import EnvConfig
from f1rl.constants import RENDERS_DIR
from f1rl.dynamics import CarState, build_sensor_rays
from f1rl.track import TrackDefinition

Color = tuple[int, int, int]

BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
GRAY: Color = (60, 60, 60)
GREEN: Color = (38, 218, 48)
RED: Color = (224, 40, 40)
YELLOW: Color = (244, 246, 59)
LIGHT_BLUE: Color = (92, 162, 255)


@dataclass(slots=True)
class RenderState:
    last_frame_path: Path | None = None
    frame_index: int = 0


class PygameRenderer:
    def __init__(self, config: EnvConfig, track: TrackDefinition) -> None:
        self.config = config
        self.track = track
        self.surface: pygame.Surface | None = None
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.car_image_on: pygame.Surface | None = None
        self.car_image_off: pygame.Surface | None = None
        self.track_image: pygame.Surface | None = None
        self.background_image: pygame.Surface | None = None
        self.render_state = RenderState()

    def _ensure_initialized(self) -> None:
        if self.surface is not None:
            return

        if self.config.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("F1 Reinforcement Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        window_size = self.config.render.window_size
        if self.config.render_mode == "human" and not self.config.headless:
            self.screen = pygame.display.set_mode(window_size)
            self.surface = self.screen
        else:
            self.surface = pygame.Surface(window_size)

        self.car_image_on = self._load_image(self.config.track.car_image, scale=0.1)
        self.car_image_off = self._load_image(self.config.track.car_off_image, scale=0.1)
        self.track_image = self._load_image(self.config.track.track_image, scale=self.config.track.render_scale)
        self.background_image = self._load_image(
            self.config.track.background_image, scale=self.config.track.render_scale
        )

    @staticmethod
    def _load_image(path: Path, scale: float) -> pygame.Surface | None:
        if not path.exists():
            return None
        image = pygame.image.load(str(path))
        if pygame.display.get_surface() is not None:
            image = image.convert_alpha()
        width = max(1, round(image.get_width() * scale))
        height = max(1, round(image.get_height() * scale))
        return pygame.transform.scale(image, (width, height))

    def render(
        self,
        car_state: CarState,
        total_reward: float,
        lap_reward: float,
        next_goal_idx: int,
        extra_hud: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        self._ensure_initialized()
        assert self.surface is not None

        surface = self.surface
        if self.background_image is not None:
            surface.blit(self.background_image, (0, 0))
        else:
            surface.fill(GRAY)

        if self.track_image is not None:
            surface.blit(self.track_image, (0, 0))

        if self.config.render.draw_track_lines:
            pygame.draw.lines(surface, YELLOW, False, self.track.outer.tolist(), 3)
            pygame.draw.lines(surface, YELLOW, False, self.track.inner.tolist(), 3)

        if self.config.render.draw_goals:
            goal = self.track.get_goal(next_goal_idx)
            start = (float(goal[0]), float(goal[1]))
            end = (float(goal[2]), float(goal[3]))
            pygame.draw.line(surface, GREEN, start, end, 2)

        if self.config.render.draw_sensors:
            rays = build_sensor_rays(car_state, self.config.dynamics)
            for ray in rays:
                start = (float(ray[0]), float(ray[1]))
                end = (float(ray[2]), float(ray[3]))
                pygame.draw.line(surface, LIGHT_BLUE, start, end, 1)

        self._draw_car(surface, car_state)
        if self.config.render.draw_hud:
            self._draw_hud(
                surface=surface,
                car_state=car_state,
                total_reward=total_reward,
                lap_reward=lap_reward,
                next_goal_idx=next_goal_idx,
                extra_hud=extra_hud or {},
            )

        if self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()

        frame = None
        if self.config.render_mode == "rgb_array" or self.config.render.export_frames:
            frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

        if self.config.render.export_frames and frame is not None:
            RENDERS_DIR.mkdir(parents=True, exist_ok=True)
            frame_path = RENDERS_DIR / f"{self.config.render.frame_prefix}_{self.render_state.frame_index:06d}.png"
            pygame.image.save(surface, str(frame_path))
            self.render_state.last_frame_path = frame_path
            self.render_state.frame_index += 1

        if self.clock is not None:
            self.clock.tick(self.config.render.fps)

        return frame

    def _draw_car(self, surface: pygame.Surface, car_state: CarState) -> None:
        image = self.car_image_on if car_state.alive else self.car_image_off
        if image is None:
            pygame.draw.circle(surface, RED, (int(car_state.x), int(car_state.y)), 8)
            return
        rotated = pygame.transform.rotate(image, car_state.heading_deg)
        rect = rotated.get_rect(center=(car_state.x, car_state.y))
        surface.blit(rotated, rect.topleft)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        car_state: CarState,
        total_reward: float,
        lap_reward: float,
        next_goal_idx: int,
        extra_hud: dict[str, Any],
    ) -> None:
        if self.font is None:
            return
        lines = [
            f"speed: {car_state.speed:.2f}",
            f"lap: {car_state.lap_count}",
            f"next_goal: {next_goal_idx}",
            f"reward: {total_reward:.2f}",
            f"step_reward: {lap_reward:.3f}",
            f"steps: {car_state.step_count}",
        ]
        for key, value in extra_hud.items():
            lines.append(f"{key}: {value}")

        for idx, line in enumerate(lines):
            text = self.font.render(line, True, BLACK, WHITE)
            surface.blit(text, (10, 10 + idx * 22))

    def poll_events(self) -> list[pygame.event.Event]:
        self._ensure_initialized()
        return list(pygame.event.get())

    def close(self) -> None:
        if self.surface is None:
            return
        pygame.quit()
        self.surface = None
        self.screen = None
