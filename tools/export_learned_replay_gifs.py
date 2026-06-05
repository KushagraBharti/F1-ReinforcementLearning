"""Export learned-policy replay GIFs from the same pygame view used by replay CLIs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from f1rl.config import SimConfig  # noqa: E402
from f1rl.render import PygameRenderer  # noqa: E402
from f1rl.replay import (  # noqa: E402
    ReplayControls,
    ReplayTrace,
    _apply_row,
    _control_overlay_lines,
    _ghost_from_row,
    _group_paths_by_generation,
    _load_trace,
    _metadata_for_inputs,
    _resolve_replay_paths,
    _row_at_time,
    _trace_best_progress,
    _trace_candidate_label,
)
from f1rl.sim import MonzaSim  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_ROOT = (
    REPO_ROOT / "artifacts" / "learned" / "v1-sac-lpv1-sac79p750-broad16-stable-v1"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "highlights" / "learned-policy-replays-20260605"


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _frame_count(duration_s: float, *, fps: int, speed: float) -> int:
    if duration_s <= 0.0:
        return 1
    return int(np.floor((duration_s / speed) * fps)) + 1


def _append_frame(frames: list[Image.Image], frame: np.ndarray) -> None:
    frames.append(Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE))


def _save_gif(frames: Sequence[Image.Image], output_path: Path, *, fps: int) -> None:
    if not frames:
        raise RuntimeError(f"no frames rendered for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )


def _render_trace_group_exact(
    *,
    renderer: PygameRenderer,
    sim: MonzaSim,
    traces: Sequence[ReplayTrace],
    controls: ReplayControls,
    fps: int,
    speed: float,
    base_lines: Sequence[str],
    allow_skip_overlay: bool,
    output_path: Path,
) -> None:
    duration_s = max(trace.duration_s for trace in traces)
    frame_total = _frame_count(duration_s, fps=fps, speed=speed)
    frames: list[Image.Image] = []
    print(
        f"rendering {output_path.name}: traces={len(traces)} duration={duration_s:.3f}s "
        f"fps={fps} speed={speed:.2f} frames={frame_total}",
        flush=True,
    )
    for frame_index in range(frame_total):
        replay_time_s = min((frame_index / fps) * speed, duration_s)
        row = _row_at_time(traces[0], replay_time_s)
        _apply_row(sim, row)
        ghosts = [
            _ghost_from_row(
                _row_at_time(trace, replay_time_s),
                index=idx,
                label=_trace_candidate_label(trace, idx),
            )
            for idx, trace in enumerate(traces[1:], 1)
        ]
        frame = renderer.render(
            sim,
            human=False,
            ghosts=ghosts,
            extra_lines=[
                *base_lines,
                f"replay {replay_time_s:6.2f}s",
                f"cars {len(traces)}",
                *_control_overlay_lines(
                    controls,
                    realtime=True,
                    allow_generation_skip=allow_skip_overlay,
                ),
            ],
        )
        _append_frame(frames, frame)
        if frame_index and (frame_index % max(fps * 5, 1) == 0):
            print(f"{output_path.name}: frame {frame_index}/{frame_total}", flush=True)
    _save_gif(frames, output_path, fps=fps)
    print(f"wrote {output_path} frames={len(frames)}", flush=True)


def _load_traces_from_paths(paths: Sequence[Path]) -> list[ReplayTrace]:
    metadata_by_path = _metadata_for_inputs(paths)
    trace_paths = _resolve_replay_paths(paths)
    return [_load_trace(path, metadata=metadata_by_path.get(path.resolve(), {})) for path in trace_paths]


def export_promotion_eval(*, policy_root: Path, output_dir: Path, fps: int, speed: float) -> None:
    telemetry_dir = policy_root / "promotion_cpu_eval" / "selected_telemetry"
    traces = _load_traces_from_paths([telemetry_dir])
    sim = MonzaSim(SimConfig())
    renderer = PygameRenderer(sim.track, sim.config)
    try:
        _render_trace_group_exact(
            renderer=renderer,
            sim=sim,
            traces=traces,
            controls=ReplayControls(speed=speed),
            fps=fps,
            speed=speed,
            base_lines=[],
            allow_skip_overlay=False,
            output_path=output_dir / "promotion_cpu_eval-selected_telemetry-exact.gif",
        )
    finally:
        renderer.close()


def export_policy_swarm(*, policy_root: Path, output_dir: Path, fps: int, speed: float) -> None:
    swarm_dir = policy_root / "policy_swarm_1000"
    trace_paths = _resolve_replay_paths([swarm_dir])
    metadata_by_path = _metadata_for_inputs([swarm_dir])
    share_deterministic_trace = _can_share_policy_swarm_trace(swarm_dir)
    groups = _group_paths_by_generation(
        trace_paths,
        metadata_by_path=metadata_by_path,
        generation_limit=None,
    )
    sim = MonzaSim(SimConfig())
    renderer = PygameRenderer(sim.track, sim.config)
    controls = ReplayControls(speed=speed)
    try:
        for display_index, (generation, group_paths) in enumerate(groups, 1):
            generation_text = "-" if generation is None else str(generation)
            print(
                f"loading policy swarm group {display_index}/{len(groups)} raw={generation_text} "
                f"traces={len(group_paths)}",
                flush=True,
            )
            if share_deterministic_trace and group_paths:
                template_path, template_metadata = group_paths[0]
                template = _load_trace(template_path, metadata=template_metadata)
                traces = [
                    ReplayTrace(
                        path=path,
                        steps=template.steps,
                        times=template.times,
                        duration_s=template.duration_s,
                        metadata=metadata,
                    )
                    for path, metadata in group_paths
                ]
                print(
                    "using shared deterministic trace for policy swarm "
                    f"template={template_path.name} clones={len(traces)}",
                    flush=True,
                )
            else:
                traces = [_load_trace(path, metadata=metadata) for path, metadata in group_paths]
            best_progress = max(_trace_best_progress(trace) for trace in traces)
            suffix = f"checkpoint_{display_index:04d}_raw_{generation_text}"
            _render_trace_group_exact(
                renderer=renderer,
                sim=sim,
                traces=traces,
                controls=controls,
                fps=fps,
                speed=speed,
                base_lines=[
                    f"generation {display_index}/{len(groups)} (raw {generation_text})",
                    f"generation cars {len(traces)}",
                    f"generation best {best_progress:7.1f} m",
                ],
                allow_skip_overlay=True,
                output_path=output_dir / f"policy_swarm_1000-{suffix}-exact.gif",
            )
    finally:
        renderer.close()


def _can_share_policy_swarm_trace(swarm_dir: Path) -> bool:
    manifest_path = swarm_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(payload.get("deterministic", False)):
        return False
    start_noise = payload.get("start_noise", {})
    noise_keys = ("position_noise_m", "heading_noise_deg", "speed_noise_kph")
    return all(float(start_noise.get(key, 0.0) or 0.0) == 0.0 for key in noise_keys)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--only", choices=("all", "promotion", "swarm"), default="all")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    policy_root = _resolve_repo_path(args.policy_root)
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in {"all", "promotion"}:
        export_promotion_eval(
            policy_root=policy_root,
            output_dir=output_dir,
            fps=int(args.fps),
            speed=float(args.speed),
        )
    if args.only in {"all", "swarm"}:
        export_policy_swarm(
            policy_root=policy_root,
            output_dir=output_dir,
            fps=int(args.fps),
            speed=float(args.speed),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
