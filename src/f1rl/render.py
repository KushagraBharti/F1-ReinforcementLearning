"""Pygame full-track renderer for manual mode and replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from f1rl.config import IMAGES_DIR, RenderConfig, SimConfig
from f1rl.sim import MonzaSim
from f1rl.track_model import TrackSpec

F1_YELLOW = (244, 246, 59)
F1_GREEN = (38, 218, 48)
GRAY = (80, 80, 80)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GHOST_BLUE = (45, 170, 255)


@dataclass(slots=True)
class RenderGhost:
    x: float
    y: float
    heading_rad: float
    speed_kph: float
    label: str = "REF"
    color: tuple[int, int, int] = GHOST_BLUE


class PygameRenderer:
    def __init__(
        self,
        track: TrackSpec,
        config: SimConfig,
        *,
        render_config: RenderConfig | None = None,
    ) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.track = track
        self.config = config
        self.render_config = render_config or RenderConfig()
        self.size = self.render_config.window_size
        self.scale = min(
            self.size[0] / track.source_image_size[0],
            self.size[1] / track.source_image_size[1],
        )
        self.offset = (0, 0)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("F1RL Monza")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.track_surface = self._load_to_window(IMAGES_DIR / "Monza_track_extra_wide_2.png")
        self.car_surface = self._load_car(config.car_image)
        self.car_off_surface = self._load_car(IMAGES_DIR / "ferrari_off.png")

    def _load_to_window(self, path: Path):
        if not path.exists():
            return None
        image = self.pygame.image.load(str(path)).convert_alpha()
        return self.pygame.transform.scale(image, self.size)

    def _load_car(self, path: Path):
        if not path.exists():
            return None
        image = self.pygame.image.load(str(path)).convert_alpha()
        width = round(image.get_width() * self.render_config.car_sprite_scale)
        height = round(image.get_height() * self.render_config.car_sprite_scale)
        # Full-track view makes a true-scale F1 car about 3x1 px on this asset.
        # The reference repo uses a 0.1 sprite scale; keep that feel with a minimum.
        width = max(
            self.render_config.min_car_width_px,
            width,
        )
        height = max(
            self.render_config.min_car_length_px,
            height,
        )
        return self.pygame.transform.scale(image, (width, height))

    def _p(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale + self.offset[0]), int(y * self.scale + self.offset[1])

    def _draw_polyline(self, points: np.ndarray, color: tuple[int, int, int], width: int = 2) -> None:
        if points.shape[0] < 2:
            return
        self.pygame.draw.lines(self.screen, color, False, [self._p(float(x), float(y)) for x, y in points], width)

    def _draw_car_sprite(
        self,
        *,
        x: float,
        y: float,
        heading_rad: float,
        alive: bool,
        ghost: RenderGhost | None = None,
    ) -> None:
        car_pos = self._p(x, y)
        if ghost is not None:
            self.pygame.draw.circle(self.screen, ghost.color, car_pos, 9, 2)
            self.pygame.draw.circle(self.screen, ghost.color, car_pos, 2)
            nose = self._p(
                x + np.cos(heading_rad) * 18.0,
                y - np.sin(heading_rad) * 18.0,
            )
            self.pygame.draw.line(self.screen, ghost.color, car_pos, nose, 2)
            label = self.font.render(ghost.label, True, ghost.color)
            self.screen.blit(label, (car_pos[0] + 10, car_pos[1] - 9))
            return
        car_surface = self.car_surface if alive else self.car_off_surface
        if car_surface is None:
            self.pygame.draw.circle(self.screen, (220, 30, 30), car_pos, 6)
        else:
            angle = np.rad2deg(heading_rad) - 90.0
            rotated = self.pygame.transform.rotate(car_surface, angle)
            rect = rotated.get_rect(center=car_pos)
            self.screen.blit(rotated, rect)

    def poll(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                return False
        return True

    def keyboard_action(self) -> int:
        keys = self.pygame.key.get_pressed()
        throttle = keys[self.pygame.K_w] or keys[self.pygame.K_UP]
        brake = keys[self.pygame.K_s] or keys[self.pygame.K_DOWN]
        left = keys[self.pygame.K_a] or keys[self.pygame.K_LEFT]
        right = keys[self.pygame.K_d] or keys[self.pygame.K_RIGHT]
        if throttle and left:
            return 5
        if throttle and right:
            return 6
        if brake and left:
            return 7
        if brake and right:
            return 8
        if throttle:
            return 1
        if brake:
            return 2
        if left:
            return 3
        if right:
            return 4
        return 0

    def reset_pressed(self) -> bool:
        keys = self.pygame.key.get_pressed()
        return bool(keys[self.pygame.K_r])

    def render(
        self,
        sim: MonzaSim,
        *,
        human: bool = True,
        extra_lines: list[str] | None = None,
        ghosts: list[RenderGhost] | None = None,
    ) -> np.ndarray:
        self.screen.fill(GRAY)
        if self.render_config.draw_track_image and self.track_surface is not None:
            self.screen.blit(self.track_surface, (0, 0))
        if self.render_config.draw_boundaries:
            self._draw_polyline(sim.track.left_boundary, F1_YELLOW, 4)
            self._draw_polyline(sim.track.right_boundary, F1_YELLOW, 4)
        if self.render_config.draw_centerline:
            self._draw_polyline(sim.track.centerline, (40, 135, 200), 2)
        if self.render_config.draw_checkpoints:
            for idx, point in enumerate(sim.track.checkpoints):
                if idx % 8 == 0 or idx == sim.state.checkpoint_index:
                    color = F1_GREEN if idx == sim.state.checkpoint_index else GRAY
                    self.pygame.draw.circle(self.screen, color, self._p(float(point[0]), float(point[1])), 3)
        ray_distances_px = sim.ray_distances_m() / sim.track.meters_per_pixel
        ray_angles = sim.state.heading_rad + sim.sensor_angles
        for distance_px, angle in zip(ray_distances_px, ray_angles, strict=True):
            end_x = sim.state.x + np.cos(float(angle)) * float(distance_px)
            end_y = sim.state.y - np.sin(float(angle)) * float(distance_px)
            if self.render_config.draw_rays:
                self.pygame.draw.line(
                    self.screen,
                    GRAY,
                    self._p(sim.state.x, sim.state.y),
                    self._p(end_x, end_y),
                    2,
                )
            if self.render_config.draw_ray_hits:
                self.pygame.draw.circle(self.screen, RED, self._p(end_x, end_y), 2)
        for ghost in ghosts or []:
            self._draw_car_sprite(
                x=ghost.x,
                y=ghost.y,
                heading_rad=ghost.heading_rad,
                alive=True,
                ghost=ghost,
            )
        self._draw_car_sprite(
            x=sim.state.x,
            y=sim.state.y,
            heading_rad=sim.state.heading_rad,
            alive=sim.state.alive,
        )
        lines = [
            f"speed {sim.state.speed_mps * 3.6:6.1f} kph",
            f"progress {sim.state.monotonic_progress_m:7.1f} m",
            f"checkpoint {sim.state.checkpoint_index:03d}",
            f"lap {sim.state.lap_index}",
            f"reason {sim.termination_reason}",
            "WASD/arrows drive  R reset  Esc quit",
        ]
        if extra_lines:
            lines.extend(extra_lines)
        for idx, text in enumerate(lines):
            surface = self.font.render(text, True, (0, 0, 0))
            back = self.pygame.Surface(surface.get_size())
            back.fill(WHITE)
            x = 18
            y = 18 + idx * 19
            self.screen.blit(back, (x, y))
            self.screen.blit(surface, (x, y))
        if human:
            self.pygame.display.flip()
            self.clock.tick(self.render_config.fps)
        return np.transpose(self.pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        self.pygame.quit()
