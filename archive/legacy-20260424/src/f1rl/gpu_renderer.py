"""GPU-backed live renderer for swarm diagnostics and manual play."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from f1rl.constants import IMAGES_DIR, WINDOW_SIZE
from f1rl.track import TrackDefinition

try:  # pragma: no cover - optional dependency for live rendering.
    import pyglet
    from pyglet import shapes
    from pyglet.window import key
except ImportError:  # pragma: no cover
    pyglet = None
    shapes = None
    key = None


def renderer_available() -> bool:
    return pyglet is not None and shapes is not None and key is not None


class PygletSwarmRenderer:
    def __init__(
        self,
        track: TrackDefinition,
        *,
        display_scale: float = 0.6,
        caption: str = "F1RL GPU Render",
        sprite_heading_offset_deg: float = -90.0,
        show_track_texture: bool = True,
    ) -> None:
        if not renderer_available():
            raise RuntimeError("pyglet is required for the GPU render path. Install the viz extra.")
        assert pyglet is not None
        assert shapes is not None
        assert key is not None

        self.track = track
        self.display_scale = max(0.35, min(1.0, float(display_scale)))
        self.sprite_heading_offset_deg = float(sprite_heading_offset_deg)
        self.show_track_texture = bool(show_track_texture)
        width = max(1, int(WINDOW_SIZE[0] * self.display_scale))
        height = max(1, int(WINDOW_SIZE[1] * self.display_scale))
        self.window = pyglet.window.Window(width=width, height=height, caption=caption, resizable=False, vsync=False)
        self.key_handler = key.KeyStateHandler()
        self.window.push_handlers(self.key_handler)
        self.background_sprite = self._try_sprite(IMAGES_DIR / "Monza_background.png", width, height)
        self.track_sprite = self._try_sprite(IMAGES_DIR / "Monza_track_extra_wide_2.png", width, height) if self.show_track_texture else None
        self.car_images = self._load_car_images(
            [
                IMAGES_DIR / "ferrari.png",
                IMAGES_DIR / "mercedes.png",
                IMAGES_DIR / "redbull.png",
            ]
        )
        dead_images = self._load_car_images([IMAGES_DIR / "ferrari_off.png"])
        self.dead_car_image = dead_images[0] if dead_images else None
        # Keep live GPU sprites aligned with the previously working pygame renderer envelope.
        # The car source sprites are 69x186 px and already match F1-like proportions well.
        self.base_sprite_scale = 0.10
        self.top_sprite_scale = 0.115
        self.last_frame_time = time.perf_counter()
        self.cached_fps = 0.0
        self.backend = "pyglet-opengl"

    def _try_sprite(self, path: Path, width: int, height: int) -> Any | None:
        if pyglet is None or not path.exists():
            return None
        image = pyglet.image.load(str(path))
        sprite = pyglet.sprite.Sprite(image)
        sprite.scale_x = width / max(image.width, 1)
        sprite.scale_y = height / max(image.height, 1)
        return sprite

    def _load_car_images(self, paths: list[Path]) -> list[Any]:
        images: list[Any] = []
        if pyglet is None:
            return images
        for path in paths:
            if not path.exists():
                continue
            image = pyglet.image.load(str(path))
            image.anchor_x = image.width // 2
            image.anchor_y = image.height // 2
            images.append(image)
        return images

    def _point(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return (x * self.display_scale, self.window.height - (y * self.display_scale))

    def poll_events(self) -> bool:
        self.window.switch_to()
        self.window.dispatch_events()
        return not self.window.has_exit

    def is_pressed(self, code: int) -> bool:
        return bool(self.key_handler[code])

    def keyboard_action(self) -> tuple[float, float]:
        assert key is not None
        throttle = 0.0
        if self.is_pressed(key.UP) or self.is_pressed(key.W):
            throttle = 1.0
        elif self.is_pressed(key.DOWN) or self.is_pressed(key.S):
            throttle = -1.0

        steering = 0.0
        if self.is_pressed(key.LEFT) or self.is_pressed(key.A):
            steering = -1.0
        elif self.is_pressed(key.RIGHT) or self.is_pressed(key.D):
            steering = 1.0
        return steering, throttle

    def _draw_checkpoint_lines(
        self,
        batch: Any,
    ) -> None:
        checkpoint_stride = max(1, self.track.goals.shape[0] // 24)
        for idx in range(0, self.track.goals.shape[0], checkpoint_stride):
            goal = self.track.goals[idx]
            x1, y1 = self._point(float(goal[0]), float(goal[1]))
            x2, y2 = self._point(float(goal[2]), float(goal[3]))
            color = (255, 220, 60) if idx == 0 else (110, 110, 110)
            shapes.Line(x1, y1, x2, y2, thickness=2 if idx == 0 else 1, color=color, batch=batch)
            if pyglet is not None:
                pyglet.text.Label(
                    str(idx),
                    x=(x1 + x2) * 0.5,
                    y=(y1 + y2) * 0.5,
                    color=(255, 255, 255, 200),
                    font_size=9,
                    anchor_x="center",
                    anchor_y="center",
                ).draw()

    def _draw_traces(
        self,
        batch: Any,
        trace_history: list[list[tuple[float, float]]],
        top_k_indices: list[int],
    ) -> None:
        for idx in top_k_indices:
            history = trace_history[idx][-240:]
            if len(history) < 2:
                continue
            for start, end in zip(history[:-1], history[1:], strict=False):
                x1, y1 = self._point(*start)
                x2, y2 = self._point(*end)
                shapes.Line(x1, y1, x2, y2, thickness=4, color=(0, 255, 255), batch=batch)

    def draw(
        self,
        telemetry_rows: list[dict[str, Any]],
        aggregate: dict[str, Any],
        top_k_indices: list[int],
        *,
        trace_history: list[list[tuple[float, float]]] | None = None,
        sensor_rays: list[tuple[float, float, float, float]] | None = None,
        extra_hud_lines: list[str] | None = None,
        focus_index: int | None = None,
        poll_events_first: bool = True,
    ) -> bool:
        if poll_events_first and not self.poll_events():
            return False
        now = time.perf_counter()
        delta = max(now - self.last_frame_time, 1e-6)
        self.cached_fps = 1.0 / delta
        self.last_frame_time = now

        self.window.clear()
        if self.background_sprite is not None:
            self.background_sprite.draw()
        elif shapes is not None:
            shapes.Rectangle(0, 0, self.window.width, self.window.height, color=(42, 42, 42)).draw()
        if self.track_sprite is not None:
            self.track_sprite.draw()

        batch = pyglet.graphics.Batch()
        self._draw_checkpoint_lines(batch)
        if trace_history is not None:
            self._draw_traces(batch, trace_history, top_k_indices)
        if sensor_rays is not None:
            for x1, y1, x2, y2 in sensor_rays:
                sx1, sy1 = self._point(x1, y1)
                sx2, sy2 = self._point(x2, y2)
                shapes.Line(sx1, sy1, sx2, sy2, thickness=1, color=(120, 220, 255), batch=batch)

        sprite_specs: list[tuple[Any, float, float, float, float]] = []
        for row in telemetry_rows:
            x, y = self._point(float(row["x"]), float(row["y"]))
            idx = int(row["car_index"])
            alive = bool(row["alive"])
            moving = bool(row.get("moving", False))
            color = (255, 220, 60) if idx in top_k_indices else (40, 255, 90) if alive and moving else (70, 180, 255) if alive else (255, 70, 70)
            outline_radius = 10 if idx in top_k_indices else 8
            inner_radius = 7 if idx in top_k_indices else 5
            shapes.Circle(x, y, radius=outline_radius, color=(0, 0, 0), batch=batch)
            shapes.Circle(x, y, radius=inner_radius, color=color, batch=batch)
            shapes.Circle(x, y, radius=2, color=(255, 255, 255), batch=batch)

            heading_deg = float(row.get("heading_deg", 0.0))
            heading_rad = math.radians(heading_deg)
            heading_length = 18 if idx in top_k_indices else 14
            hx = x + math.cos(heading_rad) * heading_length
            hy = y + math.sin(heading_rad) * heading_length
            shapes.Line(x, y, hx, hy, thickness=3 if idx in top_k_indices else 2, color=(255, 255, 255), batch=batch)
            speed_kph = float(row.get("speed_kph", 0.0))
            velocity_length = min(56.0, 12.0 + speed_kph * 0.45)
            vx = x + math.cos(heading_rad) * velocity_length
            vy = y + math.sin(heading_rad) * velocity_length
            shapes.Line(x, y, vx, vy, thickness=4 if idx in top_k_indices else 3, color=(255, 80, 80), batch=batch)
            if self.car_images or self.dead_car_image is not None:
                image = self.car_images[idx % len(self.car_images)] if alive and self.car_images else self.dead_car_image
                if image is None:
                    continue
                scale = self.top_sprite_scale if idx in top_k_indices else self.base_sprite_scale
                draw_heading = heading_deg + self.sprite_heading_offset_deg
                sprite_specs.append((image, x, y, -draw_heading, scale))
        batch.draw()

        if pyglet is not None:
            for image, x, y, rotation, scale in sprite_specs:
                sprite = pyglet.sprite.Sprite(image, x=x, y=y)
                sprite.scale = scale
                sprite.rotation = rotation
                sprite.draw()

        for row in telemetry_rows:
            idx = int(row["car_index"])
            if idx not in top_k_indices and len(telemetry_rows) > 12:
                continue
            x, y = self._point(float(row["x"]), float(row["y"]))
            pyglet.text.Label(
                f"{idx}",
                x=x,
                y=y + 12,
                color=(255, 255, 255, 255),
                font_size=10,
                anchor_x="center",
                anchor_y="bottom",
            ).draw()

        if focus_index is not None and trace_history is not None and 0 <= focus_index < len(telemetry_rows):
            self._draw_zoom_panel(telemetry_rows, trace_history, focus_index)

        labels = [
            f"renderer: {self.backend}",
            f"fps: {self.cached_fps:.1f}",
            f"alive: {aggregate['alive_cars']}",
            f"moving: {aggregate['moving_cars']}",
            f"best checkpoints: {aggregate['max_checkpoints_reached']}",
            f"dominant reason: {aggregate['dominant_termination_reason']}",
        ]
        if extra_hud_lines:
            labels.extend(extra_hud_lines)

        for idx, text in enumerate(labels):
            pyglet.text.Label(
                text,
                x=12,
                y=self.window.height - 20 - idx * 18,
                color=(255, 255, 255, 255),
                font_size=11,
            ).draw()

        self.window.flip()
        return True

    def _draw_zoom_panel(
        self,
        telemetry_rows: list[dict[str, Any]],
        trace_history: list[list[tuple[float, float]]],
        focus_index: int,
    ) -> None:
        assert pyglet is not None
        assert shapes is not None
        focus = telemetry_rows[focus_index]
        panel_w = 280
        panel_h = 220
        margin = 16
        panel_x = self.window.width - panel_w - margin
        panel_y = margin
        panel = shapes.BorderedRectangle(
            panel_x,
            panel_y,
            panel_w,
            panel_h,
            border=2,
            color=(24, 24, 24),
            border_color=(230, 230, 230),
        )
        panel.opacity = 210
        panel.draw()

        focus_x = float(focus["x"])
        focus_y = float(focus["y"])
        zoom_scale = 18.0

        def panel_point(world_x: float, world_y: float) -> tuple[float, float]:
            dx = world_x - focus_x
            dy = world_y - focus_y
            return (
                panel_x + panel_w * 0.5 + dx * zoom_scale,
                panel_y + panel_h * 0.5 - dy * zoom_scale,
            )

        checkpoint_stride = max(1, self.track.goals.shape[0] // 40)
        for idx in range(0, self.track.goals.shape[0], checkpoint_stride):
            goal = self.track.goals[idx]
            midpoint_x = float((goal[0] + goal[2]) * 0.5)
            midpoint_y = float((goal[1] + goal[3]) * 0.5)
            if abs(midpoint_x - focus_x) > 35.0 or abs(midpoint_y - focus_y) > 35.0:
                continue
            x1, y1 = panel_point(float(goal[0]), float(goal[1]))
            x2, y2 = panel_point(float(goal[2]), float(goal[3]))
            color = (255, 220, 60) if idx == int(focus.get("current_checkpoint_index", 0)) else (150, 150, 150)
            shapes.Line(x1, y1, x2, y2, thickness=2, color=color).draw()

        history = trace_history[focus_index][-120:]
        if len(history) >= 2:
            for start, end in zip(history[:-1], history[1:], strict=False):
                x1, y1 = panel_point(*start)
                x2, y2 = panel_point(*end)
                shapes.Line(x1, y1, x2, y2, thickness=3, color=(0, 255, 255)).draw()

        cx, cy = panel_point(focus_x, focus_y)
        shapes.Circle(cx, cy, radius=12, color=(0, 0, 0)).draw()
        shapes.Circle(cx, cy, radius=8, color=(255, 60, 60)).draw()
        heading_deg = float(focus.get("heading_deg", 0.0))
        heading_rad = math.radians(heading_deg)
        hx = cx + math.cos(heading_rad) * 24
        hy = cy + math.sin(heading_rad) * 24
        shapes.Line(cx, cy, hx, hy, thickness=3, color=(255, 255, 255)).draw()

        if self.car_images:
            image = self.car_images[focus_index % len(self.car_images)]
            sprite = pyglet.sprite.Sprite(image, x=cx, y=cy)
            sprite.scale = 0.18
            sprite.rotation = -(heading_deg + self.sprite_heading_offset_deg)
            sprite.draw()

        label_lines = [
            "focus zoom",
            f"car: {focus_index}",
            f"speed: {float(focus.get('speed_kph', 0.0)):.1f} kph",
            f"checkpoint: {int(focus.get('current_checkpoint_index', 0))}",
        ]
        for idx, text in enumerate(label_lines):
            pyglet.text.Label(
                text,
                x=panel_x + 10,
                y=panel_y + panel_h - 18 - idx * 18,
                color=(255, 255, 255, 255),
                font_size=10,
            ).draw()

    def close(self) -> None:
        self.window.close()
