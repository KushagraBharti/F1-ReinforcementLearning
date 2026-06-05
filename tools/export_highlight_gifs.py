"""Export compact GIF previews from full-generation highlight telemetry."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from f1rl.config import RenderConfig, SimConfig  # noqa: E402
from f1rl.render import PygameRenderer, RenderGhost  # noqa: E402
from f1rl.replay import (  # noqa: E402
    ReplayTrace,
    _apply_row,
    _load_trace,
    _manifest_metadata,
    _row_at_time,
    _trace_best_progress,
    _trace_candidate_label,
)
from f1rl.sim import MonzaSim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REEL_ROOT = REPO_ROOT / "artifacts" / "highlights" / "full-generation-reel-20260605"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _ranked_generation_paths(generation_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    metadata_by_path = _manifest_metadata(generation_dir)
    paths = [path for path in metadata_by_path if path.exists() and path.suffix == ".gz"]
    unique: dict[Path, dict[str, Any]] = {}
    for path in paths:
        unique[path.resolve()] = metadata_by_path[path]
    return sorted(
        unique.items(),
        key=lambda item: (
            float(item[1].get("score", float("-inf")) or float("-inf")),
            float(item[1].get("best_progress_m", float("-inf")) or float("-inf")),
        ),
        reverse=True,
    )


def _metadata_elapsed_s(metadata: dict[str, Any]) -> float:
    raw_elapsed = metadata.get("elapsed_s")
    if raw_elapsed is not None:
        try:
            return float(raw_elapsed)
        except (TypeError, ValueError):
            pass
    raw_steps = metadata.get("steps")
    if raw_steps is not None:
        try:
            return float(raw_steps) / 60.0
        except (TypeError, ValueError):
            pass
    return float("inf")


def _band(
    rows: list[tuple[Path, dict[str, Any]]],
    *,
    center_index: int,
    count: int,
    prefer_short: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    if not rows or count <= 0:
        return []
    if len(rows) <= count:
        return rows
    if not prefer_short:
        half = count // 2
        start = center_index - half
        start = max(0, min(start, len(rows) - count))
        return rows[start : start + count]

    window = min(len(rows), max(count * 4, count))
    half = window // 2
    start = center_index - half
    start = max(0, min(start, len(rows) - window))
    candidates = list(enumerate(rows[start : start + window], start))
    candidates.sort(
        key=lambda item: (
            _metadata_elapsed_s(item[1][1]),
            abs(item[0] - center_index),
        )
    )
    selected = candidates[:count]
    selected.sort(key=lambda item: item[0])
    return [item for _index, item in selected]


def _select_generation_paths(
    generation_dir: Path,
    *,
    limit: int,
    selection: str,
    stratum_size: int,
) -> list[tuple[Path, dict[str, Any]]]:
    rows = _ranked_generation_paths(generation_dir)
    if selection == "best":
        return rows[:limit]
    if selection != "stratified":
        raise ValueError(f"unknown selection strategy: {selection}")

    selected: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()

    def add(items: list[tuple[Path, dict[str, Any]]]) -> None:
        for path, metadata in items:
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            selected.append((path, metadata))

    add(rows[:stratum_size])
    for percentile in (0.75, 0.50, 0.25, 0.0):
        center = round((1.0 - percentile) * max(len(rows) - 1, 0))
        add(_band(rows, center_index=center, count=stratum_size, prefer_short=True))

    if len(selected) < limit:
        add(rows)
    return selected[:limit]


def _ghost_from_trace(trace: ReplayTrace, replay_time_s: float, index: int) -> RenderGhost:
    row = _row_at_time(trace, replay_time_s)
    return RenderGhost(
        x=float(row["x"]),
        y=float(row["y"]),
        heading_rad=float(np.deg2rad(row["heading_deg"])),
        speed_kph=float(row.get("speed_kph", 0.0)),
        label=_trace_candidate_label(trace, index),
        color=(
            (45, 170, 255),
            (255, 80, 80),
            (80, 210, 120),
            (255, 170, 40),
            (190, 95, 255),
            (40, 210, 220),
            (255, 110, 190),
            (120, 180, 255),
        )[index % 8],
    )


def _trace_elapsed_s(trace: ReplayTrace) -> float:
    raw_elapsed = trace.metadata.get("elapsed_s")
    if raw_elapsed is not None:
        try:
            return float(raw_elapsed)
        except (TypeError, ValueError):
            pass
    return float(trace.duration_s)


def _trace_actual_best_progress(trace: ReplayTrace) -> float:
    if trace.steps:
        return max(float(row.get("monotonic_progress_m", 0.0) or 0.0) for row in trace.steps)
    return _trace_best_progress(trace)


def _trace_finished(trace: ReplayTrace) -> bool:
    if not trace.steps:
        return False
    final = trace.steps[-1]
    return bool(
        final.get("valid_lap", False)
        or final.get("completed_lap", False)
        or final.get("finish_crossed", False)
    )


def _anchor_trace_index(traces: list[ReplayTrace]) -> int:
    valid_indices = [
        index
        for index, trace in enumerate(traces)
        if bool(trace.metadata.get("valid_lap", False))
        or bool(trace.metadata.get("completed_lap", False))
        or bool(trace.metadata.get("finish_crossed", False))
        or _trace_finished(trace)
    ]
    if valid_indices:
        return min(
            valid_indices,
            key=lambda index: (
                _trace_elapsed_s(traces[index]),
                -_trace_actual_best_progress(traces[index]),
            ),
        )
    return max(
        range(len(traces)),
        key=lambda index: (_trace_actual_best_progress(traces[index]), -_trace_elapsed_s(traces[index])),
    )


def _annotate(frame: np.ndarray, lines: list[str]) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    x = 18
    y = image.height - 18 - (len(lines) * 17)
    for line in lines:
        bbox = draw.textbbox((x, y), line)
        draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(255, 255, 255))
        draw.text((x, y), line, fill=(0, 0, 0))
        y += 17
    return image


def _render_reel_gif(
    *,
    run_dir: Path,
    output_path: Path,
    trace_limit: int,
    fps: int,
    speed: float,
    min_frames_per_generation: int,
    max_frames_per_generation: int,
    clip_buffer_s: float,
    width: int,
    label: str,
    selection: str = "best",
    stratum_size: int = 30,
    draw_rays: bool = False,
    draw_labels: bool = False,
) -> None:
    generation_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("gen-"))
    if not generation_dirs:
        raise FileNotFoundError(f"no gen-* directories found under {run_dir}")

    height = int(round(width * 454 / 900))
    sim = MonzaSim(SimConfig())
    renderer = PygameRenderer(
        sim.track,
        sim.config,
        render_config=RenderConfig(
            window_size=(width, height),
            fps=fps,
            draw_rays=draw_rays,
            draw_ray_hits=draw_rays,
        ),
    )
    frames: list[Image.Image] = []
    try:
        for generation_index, generation_dir in enumerate(generation_dirs, 1):
            group_paths = _select_generation_paths(
                generation_dir,
                limit=trace_limit,
                selection=selection,
                stratum_size=stratum_size,
            )
            if not group_paths:
                continue
            print(
                f"loading {label} {generation_dir.name}: {len(group_paths)} traces "
                f"selection={selection}",
                flush=True,
            )
            traces: list[ReplayTrace] = []
            for load_index, (path, metadata) in enumerate(group_paths, 1):
                traces.append(_load_trace(path, metadata=metadata))
                if load_index % 10 == 0 or load_index == len(group_paths):
                    print(
                        f"loaded {label} {generation_dir.name}: "
                        f"{load_index}/{len(group_paths)} traces",
                        flush=True,
                    )
            anchor_index = _anchor_trace_index(traces)
            if anchor_index:
                anchor = traces.pop(anchor_index)
                traces.insert(0, anchor)
            max_duration_s = max(trace.duration_s for trace in traces)
            anchor_duration_s = float(traces[0].duration_s)
            duration_s = min(max_duration_s, anchor_duration_s + max(0.0, clip_buffer_s))
            frame_count = int(np.ceil((duration_s / speed) * fps))
            frame_count = int(np.clip(frame_count, min_frames_per_generation, max_frames_per_generation))
            best_progress_m = max(_trace_best_progress(trace) for trace in traces)
            generation = traces[0].metadata.get("generation", generation_dir.name.removeprefix("gen-"))
            print(
                f"rendering {label} {generation_dir.name}: "
                f"anchor={anchor_duration_s:.1f}s clip={duration_s:.1f}s "
                f"speed={speed:.1f}x frames={frame_count}",
                flush=True,
            )
            for frame_index in range(frame_count):
                alpha = frame_index / max(frame_count - 1, 1)
                replay_time_s = duration_s * alpha
                primary_row = _row_at_time(traces[0], replay_time_s)
                _apply_row(sim, primary_row)
                ghosts = [
                    _ghost_from_trace(trace, replay_time_s, index)
                    for index, trace in enumerate(traces[1:], 1)
                ]
                if not draw_labels:
                    ghosts = [
                        RenderGhost(
                            x=ghost.x,
                            y=ghost.y,
                            heading_rad=ghost.heading_rad,
                            speed_kph=ghost.speed_kph,
                            label="",
                            color=ghost.color,
                        )
                        for ghost in ghosts
                    ]
                frame = renderer.render(
                    sim,
                    human=False,
                    ghosts=ghosts,
                    extra_lines=[
                        f"{label} generation {generation_index}/{len(generation_dirs)} raw {generation}",
                        f"cars {len(traces)}  best {best_progress_m:7.1f}m",
                        f"clip {replay_time_s:6.1f}s / {duration_s:6.1f}s  anchor + {clip_buffer_s:.1f}s",
                    ],
                )
                frames.append(
                    _annotate(
                        frame,
                        [
                            f"{label} gen {generation_index}/{len(generation_dirs)}",
                            f"raw {generation}  cars {len(traces)}",
                        ],
                    ).convert("P", palette=Image.Palette.ADAPTIVE)
                )
            del traces
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("no frames rendered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    print(f"wrote {output_path} frames={len(frames)}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reel-root", type=Path, default=DEFAULT_REEL_ROOT)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--clip-buffer-s", type=float, default=4.0)
    parser.add_argument("--min-frames-per-generation", type=int, default=24)
    parser.add_argument("--max-frames-per-generation", type=int, default=420)
    parser.add_argument("--cpu-trace-limit", type=int, default=150)
    parser.add_argument("--gpu-trace-limit", type=int, default=150)
    parser.add_argument("--gpu-selection", choices=("best", "stratified"), default="stratified")
    parser.add_argument("--gpu-stratum-size", type=int, default=30)
    parser.add_argument("--draw-rays", action="store_true")
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--only", choices=("all", "cpu", "gpu"), default="all")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    reel_root = _resolve_path(args.reel_root)
    if args.only in {"all", "cpu"}:
        _render_reel_gif(
            run_dir=reel_root / "cpu-es-150x60",
            output_path=reel_root / "cpu-es-150x60-all6-all150.gif",
            trace_limit=int(args.cpu_trace_limit),
            fps=int(args.fps),
            speed=float(args.speed),
            min_frames_per_generation=int(args.min_frames_per_generation),
            max_frames_per_generation=int(args.max_frames_per_generation),
            clip_buffer_s=float(args.clip_buffer_s),
            width=int(args.width),
            label="CPU ES 150x60",
            selection="best",
            stratum_size=30,
            draw_rays=bool(args.draw_rays),
            draw_labels=bool(args.draw_labels),
        )
    if args.only in {"all", "gpu"}:
        suffix = (
            f"stratified{int(args.gpu_trace_limit)}"
            if args.gpu_selection == "stratified"
            else f"top{int(args.gpu_trace_limit)}"
        )
        _render_reel_gif(
            run_dir=reel_root / "gpu-es-2000x150",
            output_path=reel_root / f"gpu-es-2000x150-all6-{suffix}.gif",
            trace_limit=int(args.gpu_trace_limit),
            fps=int(args.fps),
            speed=float(args.speed),
            min_frames_per_generation=int(args.min_frames_per_generation),
            max_frames_per_generation=int(args.max_frames_per_generation),
            clip_buffer_s=float(args.clip_buffer_s),
            width=int(args.width),
            label="GPU ES 2000x150",
            selection=str(args.gpu_selection),
            stratum_size=int(args.gpu_stratum_size),
            draw_rays=bool(args.draw_rays),
            draw_labels=bool(args.draw_labels),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
