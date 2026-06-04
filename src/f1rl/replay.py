"""Replay telemetry JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from f1rl.config import SimConfig
from f1rl.render import GHOST_BLUE, PygameRenderer, RenderGhost
from f1rl.sim import MonzaSim
from f1rl.telemetry import load_steps

GHOST_COLORS = (
    GHOST_BLUE,
    (255, 80, 80),
    (80, 210, 120),
    (255, 170, 40),
    (190, 95, 255),
    (40, 210, 220),
    (255, 110, 190),
    (120, 180, 255),
)
ReplayGroupResult = Literal["complete", "skip", "quit"]
_MIN_REPLAY_SPEED = 0.05
_MAX_REPLAY_SPEED = 32.0
_REPLAY_SPEED_STEP = 1.25
_SKIP_REST_SENTINEL = 1_000_000


@dataclass(slots=True)
class ReplayTrace:
    path: Path
    steps: list[dict[str, Any]]
    times: np.ndarray
    duration_s: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class ReplayControls:
    speed: float
    queued_generation_skips: int = 0

    def __post_init__(self) -> None:
        self.speed = _clamp_replay_speed(self.speed)

    def speed_up(self) -> None:
        self.speed = _clamp_replay_speed(self.speed * _REPLAY_SPEED_STEP)

    def speed_down(self) -> None:
        self.speed = _clamp_replay_speed(self.speed / _REPLAY_SPEED_STEP)

    def reset_speed(self) -> None:
        self.speed = 1.0

    def queue_generation_skip(self, count: int = 1) -> None:
        self.queued_generation_skips = max(0, self.queued_generation_skips + count)

    def skip_remaining_generations(self) -> None:
        self.queued_generation_skips = _SKIP_REST_SENTINEL

    def take_generation_skip(self) -> bool:
        if self.queued_generation_skips <= 0:
            return False
        self.queued_generation_skips -= 1
        return True


def _clamp_replay_speed(speed: float) -> float:
    return float(np.clip(speed, _MIN_REPLAY_SPEED, _MAX_REPLAY_SPEED))


def _apply_row(sim: MonzaSim, row: dict[str, Any]) -> None:
    sim.state.x = float(row["x"])
    sim.state.y = float(row["y"])
    sim.state.heading_rad = float(np.deg2rad(row["heading_deg"]))
    sim.state.speed_mps = float(row["speed_mps"])
    sim.state.yaw_rate_rps = float(row["yaw_rate_rps"])
    sim.state.checkpoint_index = int(row["checkpoint_index"])
    sim.state.lap_index = int(row["lap_index"])
    sim.state.raw_progress_m = float(row["raw_progress_m"])
    sim.state.monotonic_progress_m = float(row["monotonic_progress_m"])
    sim.state.alive = not bool(row.get("terminated", False))
    sim.termination_reason = str(row["termination_reason"])


def _lerp_angle_deg(start: float, end: float, alpha: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return start + delta * alpha


def _interpolate_row(steps: list[dict[str, Any]], times: np.ndarray, replay_time_s: float) -> dict[str, Any]:
    if not steps:
        raise ValueError("cannot interpolate empty replay")
    first_time = float(times[0])
    target_time = first_time + replay_time_s
    if target_time <= first_time:
        return dict(steps[0])
    if target_time >= float(times[-1]):
        return dict(steps[-1])

    idx = int(np.searchsorted(times, target_time, side="right") - 1)
    idx = int(np.clip(idx, 0, len(steps) - 2))
    current = steps[idx]
    nxt = steps[idx + 1]
    span = max(float(times[idx + 1] - times[idx]), 1e-9)
    alpha = float(np.clip((target_time - float(times[idx])) / span, 0.0, 1.0))
    row = dict(current)

    numeric_keys = (
        "sim_time_s",
        "x",
        "y",
        "speed_mps",
        "speed_kph",
        "yaw_rate_rps",
        "throttle",
        "brake",
        "steering",
        "raw_progress_m",
        "monotonic_progress_m",
        "progress_delta_m",
        "lateral_error_m",
        "heading_error_deg",
        "reward_total",
    )
    for key in numeric_keys:
        if key in current and key in nxt:
            row[key] = float(current[key]) + (float(nxt[key]) - float(current[key])) * alpha
    row["heading_deg"] = _lerp_angle_deg(float(current["heading_deg"]), float(nxt["heading_deg"]), alpha)
    row["checkpoint_index"] = int(current["checkpoint_index"])
    row["lap_index"] = int(current["lap_index"])
    row["terminated"] = False
    row["truncated"] = False
    row["termination_reason"] = "active"
    return row


def _manifest_metadata(directory: Path) -> dict[Path, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata: dict[Path, dict[str, Any]] = {}
    for row in payload.get("traces", []):
        path = Path(str(row.get("path", "")))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        metadata[path.resolve()] = dict(row)
        alternate_path = directory / path.name
        metadata[alternate_path.resolve()] = dict(row)
    return metadata


def _metadata_for_inputs(paths: Sequence[Path]) -> dict[Path, dict[str, Any]]:
    metadata: dict[Path, dict[str, Any]] = {}
    manifest_dirs: set[Path] = set()
    for path in paths:
        if path.is_dir():
            manifest_dirs.add(path)
        elif (path.parent / "manifest.json").exists():
            manifest_dirs.add(path.parent)
    for directory in manifest_dirs:
        metadata.update(_manifest_metadata(directory))
    return metadata


def _manifest_scores(directory: Path, *, sort_by: str) -> dict[Path, float]:
    key = "score" if sort_by == "score" else "best_progress_m"
    scores: dict[Path, float] = {}
    for path, row in _manifest_metadata(directory).items():
        scores[path] = float(row.get(key, float("-inf")) or float("-inf"))
    return scores


def _resolve_replay_paths(paths: Sequence[Path], *, limit: int | None = None, sort_by: str = "name") -> list[Path]:
    resolved: list[Path] = []
    scores: dict[Path, float] = {}
    for path in paths:
        if path.is_dir():
            if sort_by in {"best-progress", "score"}:
                scores.update(_manifest_scores(path, sort_by=sort_by))
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            resolved.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    if sort_by in {"best-progress", "score"}:
        unique.sort(key=lambda path: scores.get(path.resolve(), float("-inf")), reverse=True)
    if not unique:
        raise FileNotFoundError("no replay telemetry JSONL files found")
    missing = [path for path in unique if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"replay telemetry file not found: {missing_text}")
    if limit is not None and limit > 0:
        unique = unique[:limit]
    return unique


def _load_trace(path: Path, *, metadata: dict[str, Any] | None = None) -> ReplayTrace:
    steps = load_steps(path)
    if not steps:
        raise ValueError(f"empty replay telemetry: {path}")
    times = np.asarray([float(row["sim_time_s"]) for row in steps], dtype=np.float64)
    duration_s = float(times[-1] - times[0]) if len(times) else 0.0
    return ReplayTrace(path=path, steps=steps, times=times, duration_s=duration_s, metadata=metadata or {})


def _row_at_time(trace: ReplayTrace, replay_time_s: float) -> dict[str, Any]:
    return _interpolate_row(trace.steps, trace.times, min(replay_time_s, trace.duration_s))


def _row_at_index(trace: ReplayTrace, index: int) -> dict[str, Any]:
    return dict(trace.steps[min(index, len(trace.steps) - 1)])


def _ghost_from_row(row: dict[str, Any], *, index: int, label: str | None = None) -> RenderGhost:
    return RenderGhost(
        x=float(row["x"]),
        y=float(row["y"]),
        heading_rad=float(np.deg2rad(row["heading_deg"])),
        speed_kph=float(row.get("speed_kph", 0.0)),
        label=label or f"C{index:02d}",
        color=GHOST_COLORS[index % len(GHOST_COLORS)],
    )


def _group_traces_by_generation(
    traces: Sequence[ReplayTrace],
    *,
    generation_limit: int | None,
) -> list[tuple[int | None, list[ReplayTrace]]]:
    grouped: dict[int | None, list[ReplayTrace]] = defaultdict(list)
    for trace in traces:
        raw_generation = trace.metadata.get("generation")
        generation = int(raw_generation) if raw_generation is not None else None
        grouped[generation].append(trace)

    groups: list[tuple[int | None, list[ReplayTrace]]] = []
    for generation in sorted(grouped, key=lambda value: (value is None, value if value is not None else 0)):
        generation_traces = grouped[generation]
        if generation_limit is not None and generation_limit > 0:
            generation_traces = generation_traces[:generation_limit]
        groups.append((generation, generation_traces))
    return groups


def _group_paths_by_generation(
    trace_paths: Sequence[Path],
    *,
    metadata_by_path: dict[Path, dict[str, Any]],
    generation_limit: int | None,
) -> list[tuple[int | None, list[tuple[Path, dict[str, Any]]]]]:
    grouped: dict[int | None, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in trace_paths:
        metadata = metadata_by_path.get(path.resolve(), {})
        raw_generation = metadata.get("generation")
        generation = int(raw_generation) if raw_generation is not None else None
        grouped[generation].append((path, metadata))

    groups: list[tuple[int | None, list[tuple[Path, dict[str, Any]]]]] = []
    for generation in sorted(grouped, key=lambda value: (value is None, value if value is not None else 0)):
        generation_paths = grouped[generation]
        if generation_limit is not None and generation_limit > 0:
            generation_paths = generation_paths[:generation_limit]
        groups.append((generation, generation_paths))
    return groups


def _trace_best_progress(trace: ReplayTrace) -> float:
    if "best_progress_m" in trace.metadata:
        return float(trace.metadata["best_progress_m"] or 0.0)
    return max(float(row.get("monotonic_progress_m", 0.0)) for row in trace.steps)


def _trace_candidate_label(trace: ReplayTrace, index: int) -> str:
    candidate_index = trace.metadata.get("candidate_index")
    if candidate_index is None:
        return f"C{index:02d}"
    return f"C{int(candidate_index):03d}"


def _pygame_keys(pygame_module: Any, *names: str) -> tuple[int, ...]:
    keys: list[int] = []
    for name in names:
        value = getattr(pygame_module, name, None)
        if value is not None:
            keys.append(int(value))
    return tuple(keys)


def _consume_named_keydown(renderer: PygameRenderer, *names: str) -> bool:
    return renderer.consume_keydown(*_pygame_keys(renderer.pygame, *names))


def _poll_replay_controls(
    *,
    renderer: PygameRenderer,
    controls: ReplayControls,
    allow_generation_skip: bool,
) -> bool:
    if not renderer.poll():
        return False

    if _consume_named_keydown(renderer, "K_RIGHTBRACKET", "K_EQUALS", "K_PLUS", "K_KP_PLUS"):
        controls.speed_up()
    if _consume_named_keydown(renderer, "K_LEFTBRACKET", "K_MINUS", "K_UNDERSCORE", "K_KP_MINUS"):
        controls.speed_down()
    if _consume_named_keydown(renderer, "K_0", "K_KP0", "K_BACKSPACE"):
        controls.reset_speed()

    if allow_generation_skip:
        if _consume_named_keydown(renderer, "K_n", "K_SPACE"):
            controls.queue_generation_skip()
        if _consume_named_keydown(renderer, "K_PAGEDOWN"):
            controls.queue_generation_skip(5)
        if _consume_named_keydown(renderer, "K_END"):
            controls.skip_remaining_generations()
    return True


def _control_overlay_lines(
    controls: ReplayControls,
    *,
    realtime: bool,
    allow_generation_skip: bool,
) -> list[str]:
    lines = []
    if realtime:
        lines.append(f"speed x{controls.speed:.2f}  [/] or +/- adjust  0 reset")
    if allow_generation_skip:
        skip_text = "N/Space skip gen  PgDn +5  End skip rest"
        if controls.queued_generation_skips >= _SKIP_REST_SENTINEL:
            skip_text += "  queued rest"
        elif controls.queued_generation_skips > 0:
            skip_text += f"  queued {controls.queued_generation_skips}"
        lines.append(skip_text)
    return lines


def _render_trace_group(
    *,
    renderer: PygameRenderer,
    sim: MonzaSim,
    traces: Sequence[ReplayTrace],
    controls: ReplayControls,
    realtime: bool,
    base_lines: Sequence[str],
    allow_skip: bool = False,
) -> ReplayGroupResult:
    primary = traces[0]
    duration_s = max(trace.duration_s for trace in traces)
    first_time_s = float(primary.times[0]) if len(primary.times) else 0.0
    max_steps = max(len(trace.steps) for trace in traces)
    if realtime:
        last_wall_s = time.perf_counter()
        replay_time_s = 0.0
        while True:
            if not _poll_replay_controls(
                renderer=renderer,
                controls=controls,
                allow_generation_skip=allow_skip,
            ):
                return "quit"
            if allow_skip and controls.take_generation_skip():
                return "skip"
            now_s = time.perf_counter()
            replay_time_s = min(replay_time_s + (now_s - last_wall_s) * controls.speed, duration_s)
            last_wall_s = now_s
            row = _row_at_time(primary, replay_time_s)
            _apply_row(sim, row)
            ghosts = [
                _ghost_from_row(_row_at_time(trace, replay_time_s), index=idx, label=_trace_candidate_label(trace, idx))
                for idx, trace in enumerate(traces[1:], 1)
            ]
            renderer.render(
                sim,
                human=True,
                ghosts=ghosts,
                extra_lines=[
                    *base_lines,
                    f"replay {replay_time_s:6.2f}s",
                    f"cars {len(traces)}",
                    *_control_overlay_lines(
                        controls,
                        realtime=realtime,
                        allow_generation_skip=allow_skip,
                    ),
                ],
            )
            if replay_time_s >= duration_s:
                return "complete"
    else:
        for step_index in range(max_steps):
            if not _poll_replay_controls(
                renderer=renderer,
                controls=controls,
                allow_generation_skip=allow_skip,
            ):
                return "quit"
            if allow_skip and controls.take_generation_skip():
                return "skip"
            row = _row_at_index(primary, step_index)
            _apply_row(sim, row)
            replay_time_s = float(row["sim_time_s"]) - first_time_s
            ghosts = [
                _ghost_from_row(_row_at_index(trace, step_index), index=idx, label=_trace_candidate_label(trace, idx))
                for idx, trace in enumerate(traces[1:], 1)
            ]
            renderer.render(
                sim,
                human=True,
                ghosts=ghosts,
                extra_lines=[
                    *base_lines,
                    f"replay {replay_time_s:6.2f}s",
                    "untimed",
                    f"cars {len(traces)}",
                    *_control_overlay_lines(
                        controls,
                        realtime=realtime,
                        allow_generation_skip=allow_skip,
                    ),
                ],
            )
    return "complete"


def _pause_between_groups(
    *,
    renderer: PygameRenderer,
    sim: MonzaSim,
    controls: ReplayControls,
    pause_s: float,
    lines: Sequence[str],
) -> ReplayGroupResult:
    if pause_s <= 0.0:
        return "complete"
    start_wall_s = time.perf_counter()
    while time.perf_counter() - start_wall_s < pause_s:
        if not _poll_replay_controls(
            renderer=renderer,
            controls=controls,
            allow_generation_skip=True,
        ):
            return "quit"
        if controls.queued_generation_skips > 0:
            return "skip"
        renderer.render(
            sim,
            human=True,
            extra_lines=[
                *lines,
                *_control_overlay_lines(
                    controls,
                    realtime=True,
                    allow_generation_skip=True,
                ),
            ],
        )
    return "complete"


def _load_trace_group_interactive(
    *,
    renderer: PygameRenderer,
    sim: MonzaSim,
    controls: ReplayControls,
    group_paths: Sequence[tuple[Path, dict[str, Any]]],
    base_lines: Sequence[str],
) -> tuple[ReplayGroupResult, list[ReplayTrace]]:
    traces: list[ReplayTrace] = []
    total = len(group_paths)
    for load_index, (path, metadata) in enumerate(group_paths, 1):
        if controls.take_generation_skip():
            return "skip", traces
        if not _poll_replay_controls(
            renderer=renderer,
            controls=controls,
            allow_generation_skip=True,
        ):
            return "quit", traces
        if controls.take_generation_skip():
            return "skip", traces
        renderer.render(
            sim,
            human=True,
            extra_lines=[
                *base_lines,
                f"loading trace {load_index}/{total}",
                path.name,
                *_control_overlay_lines(
                    controls,
                    realtime=True,
                    allow_generation_skip=True,
                ),
            ],
        )
        traces.append(_load_trace(path, metadata=metadata))

    return "complete", traces


def run_replay_paths(
    paths: Sequence[Path],
    *,
    headless: bool,
    speed: float = 1.0,
    realtime: bool = True,
    limit: int | None = None,
    sort_by: str = "name",
    by_generation: bool = False,
    generation_limit: int | None = None,
    generation_pause_s: float = 0.75,
) -> int:
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    controls = ReplayControls(speed=speed)
    trace_paths = _resolve_replay_paths(paths, limit=limit, sort_by=sort_by)
    metadata_by_path = _metadata_for_inputs(paths)
    if by_generation and not headless:
        sim = MonzaSim(SimConfig())
        renderer = PygameRenderer(sim.track, sim.config)
        try:
            groups = _group_paths_by_generation(
                trace_paths,
                metadata_by_path=metadata_by_path,
                generation_limit=generation_limit,
            )
            group_count = len(groups)
            for display_index, (generation, group_paths) in enumerate(groups, 1):
                if not group_paths:
                    continue
                if controls.take_generation_skip():
                    continue
                generation_text = "-" if generation is None else str(generation)
                if not _poll_replay_controls(
                    renderer=renderer,
                    controls=controls,
                    allow_generation_skip=True,
                ):
                    break
                renderer.render(
                    sim,
                    human=True,
                    extra_lines=[
                        f"loading generation {display_index}/{group_count} (raw {generation_text})",
                        f"loading traces {len(group_paths)}",
                        *_control_overlay_lines(
                            controls,
                            realtime=True,
                            allow_generation_skip=True,
                        ),
                    ],
                )
                load_result, traces = _load_trace_group_interactive(
                    renderer=renderer,
                    sim=sim,
                    controls=controls,
                    group_paths=group_paths,
                    base_lines=[
                        f"loading generation {display_index}/{group_count} (raw {generation_text})",
                        f"generation traces {len(group_paths)}",
                    ],
                )
                if load_result == "quit":
                    break
                if load_result == "skip":
                    continue
                if not traces:
                    continue
                best_progress = max(_trace_best_progress(trace) for trace in traces)
                keep_running = _render_trace_group(
                    renderer=renderer,
                    sim=sim,
                    traces=traces,
                    controls=controls,
                    realtime=realtime,
                    allow_skip=True,
                    base_lines=[
                        f"generation {display_index}/{group_count} (raw {generation_text})",
                        f"generation cars {len(traces)}",
                        f"generation best {best_progress:7.1f} m",
                    ],
                )
                if keep_running == "quit":
                    break
                if keep_running == "skip":
                    continue
                if display_index < group_count:
                    pause_result = _pause_between_groups(
                        renderer=renderer,
                        sim=sim,
                        controls=controls,
                        pause_s=generation_pause_s,
                        lines=[
                            f"finished generation {display_index}/{group_count}",
                            f"next generation {display_index + 1}/{group_count}",
                        ],
                    )
                    if pause_result == "quit":
                        break
        finally:
            renderer.close()
        return 0

    traces = [_load_trace(path, metadata=metadata_by_path.get(path.resolve(), {})) for path in trace_paths]
    if headless:
        if by_generation:
            groups = _group_traces_by_generation(traces, generation_limit=generation_limit)
            print(f"replay_loaded groups={len(groups)} traces={sum(len(group) for _, group in groups)}")
            for display_index, (generation, group) in enumerate(groups, 1):
                max_duration = max(trace.duration_s for trace in group)
                best_progress = max(_trace_best_progress(trace) for trace in group)
                print(
                    "generation_group "
                    f"index={display_index} generation={generation} traces={len(group)} "
                    f"duration={max_duration:.3f}s best_progress_m={best_progress:.3f}"
                )
        else:
            max_duration = max(trace.duration_s for trace in traces)
            print(f"replay_loaded traces={len(traces)} duration={max_duration:.3f}s")
            for idx, trace in enumerate(traces):
                print(f"trace index={idx} steps={len(trace.steps)} duration={trace.duration_s:.3f}s path={trace.path}")
        return 0
    sim = MonzaSim(SimConfig())
    renderer = PygameRenderer(sim.track, sim.config)
    try:
        _render_trace_group(
            renderer=renderer,
            sim=sim,
            traces=traces,
            controls=controls,
            realtime=realtime,
            base_lines=[],
        )
    finally:
        renderer.close()
    return 0


def run_replay(path: Path, *, headless: bool, speed: float = 1.0, realtime: bool = True) -> int:
    return run_replay_paths([path], headless=headless, speed=speed, realtime=realtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved telemetry JSONL files.")
    parser.add_argument("paths", type=Path, nargs="+", help="Telemetry JSONL file(s) or directory containing JSONL files.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier; default is real time.")
    parser.add_argument("--no-timing", action="store_true", help="Advance one telemetry row per render frame.")
    parser.add_argument("--limit", type=int, help="Maximum number of telemetry files to load from the provided paths.")
    parser.add_argument("--by-generation", action="store_true", help="Play manifest traces one generation at a time.")
    parser.add_argument(
        "--generation-limit",
        type=int,
        help="Maximum traces to load per generation in --by-generation replay.",
    )
    parser.add_argument(
        "--generation-pause-s",
        type=float,
        default=0.75,
        help="Pause between generation groups in seconds.",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "best-progress", "score"],
        default="name",
        help="Directory replay ordering; best-progress uses selected_telemetry/manifest.json when present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_replay_paths(
        args.paths,
        headless=args.headless,
        speed=args.speed,
        realtime=not args.no_timing,
        limit=args.limit,
        sort_by=args.sort,
        by_generation=args.by_generation,
        generation_limit=args.generation_limit,
        generation_pause_s=args.generation_pause_s,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
