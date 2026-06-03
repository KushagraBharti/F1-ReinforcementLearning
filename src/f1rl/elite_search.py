"""Short segment search for elite section exits and state-library seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import ARTIFACTS_DIR, SimConfig
from f1rl.policy_io import (
    close_vecnormalize,
    load_sb3_ppo,
    load_vecnormalize_stats,
    normalize_observation,
    resolve_ppo_eval_config,
)
from f1rl.scripted import ScriptedController
from f1rl.sim import MonzaSim
from f1rl.state_library import load_state_library, write_state_library
from f1rl.state_snapshot import StateSnapshot, snapshot_from_sim, snapshot_to_dict


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _filter_snapshots(
    snapshots: list[StateSnapshot],
    *,
    start_min_progress_m: float | None,
    start_max_progress_m: float | None,
) -> list[StateSnapshot]:
    if start_min_progress_m is None or start_max_progress_m is None:
        return snapshots
    return [
        snapshot
        for snapshot in snapshots
        if start_min_progress_m <= snapshot.monotonic_progress_m <= start_max_progress_m
    ]


def _score_attempt(rows: list[dict[str, Any]], *, start_progress_m: float) -> float:
    if not rows:
        return float("-inf")
    final = rows[-1]
    best_progress_m = max(float(row["monotonic_progress_m"]) for row in rows)
    progress_delta_m = max(0.0, best_progress_m - start_progress_m)
    score = progress_delta_m
    if final.get("segment_complete"):
        score += 100_000.0
    if final.get("valid_lap"):
        score += 500.0
    if not final.get("collided") and not final.get("off_track"):
        score += 1_000.0
    score += min(float(final.get("speed_kph", 0.0)), 260.0) * 2.0
    score -= abs(float(final.get("heading_error_deg", 0.0))) * 5.0
    score -= abs(float(final.get("lateral_error_m", 0.0))) * 10.0
    score -= int(final.get("missed_checkpoint_count", 0)) * 1000.0
    return float(score)


def _scripted_controls(sim: MonzaSim, controller: ScriptedController, rng: np.random.Generator, noise: float) -> tuple[float, float, float]:
    throttle, brake, steer = controller.controls(sim)
    if noise > 0.0:
        throttle = float(np.clip(throttle + rng.normal(0.0, noise), 0.0, 1.0))
        brake = float(np.clip(brake + rng.normal(0.0, noise), 0.0, 1.0))
        steer = float(np.clip(steer + rng.normal(0.0, noise), -1.0, 1.0))
    return throttle, brake, steer


def run_elite_search(
    *,
    state_library: Path,
    output_dir: Path,
    attempts: int,
    max_steps: int,
    seed: int,
    top_k: int,
    segment_length_m: float,
    start_min_progress_m: float | None = None,
    start_max_progress_m: float | None = None,
    policy: str = "scripted",
    action_noise: float = 0.0,
    position_noise_m: float = 0.0,
    heading_noise_deg: float = 0.0,
    speed_noise_kph: float = 0.0,
    checkpoint: str | None = None,
    device: str = "auto",
    metadata_mode: str = "auto",
) -> Path:
    snapshots = _filter_snapshots(
        load_state_library(state_library),
        start_min_progress_m=start_min_progress_m,
        start_max_progress_m=start_max_progress_m,
    )
    if not snapshots:
        raise ValueError("No state-library snapshots match the requested elite-search range.")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_telemetry"
    rng = np.random.default_rng(seed)
    controller = ScriptedController()
    ppo_config = None
    model = None
    vec_normalize = None
    if policy.startswith("ppo"):
        if checkpoint is None:
            raise ValueError("--checkpoint is required for PPO elite-search policies.")
        ppo_config = resolve_ppo_eval_config(
            checkpoint,
            max_steps=max_steps,
            fallback_config=SimConfig(max_steps=max_steps),
            metadata_mode=metadata_mode,
        )
        vec_normalize = load_vecnormalize_stats(ppo_config)
        model = load_sb3_ppo(ppo_config.checkpoint_path, device=device)
    attempts_summary: list[dict[str, Any]] = []
    attempt_rows: list[tuple[float, list[dict[str, Any]], StateSnapshot, dict[str, Any]]] = []
    try:
        for attempt_index in range(attempts):
            snapshot = snapshots[int(rng.integers(0, len(snapshots)))]
            sim_config = ppo_config.sim_config if ppo_config is not None else SimConfig(max_steps=max_steps)
            sim = MonzaSim(sim_config)
            obs, _ = sim.reset(
                seed=seed + attempt_index,
                options={
                    "state_snapshot": snapshot_to_dict(snapshot),
                    "segment_length_m": segment_length_m,
                    "curriculum_stage": "elite-search",
                    "position_noise_m": position_noise_m,
                    "heading_noise_deg": heading_noise_deg,
                    "speed_noise_kph": speed_noise_kph,
                },
            )
            obs = normalize_observation(obs, vec_normalize)
            rows: list[dict[str, Any]] = []
            for _ in range(max_steps):
                if policy == "random":
                    result = sim.step(int(rng.integers(0, sim.action_dim)))
                elif policy.startswith("ppo"):
                    if model is None:
                        raise RuntimeError("PPO model was not loaded.")
                    action, _ = model.predict(obs, deterministic=policy == "ppo-deterministic")
                    if sim.config.action_mode == "continuous":
                        result = sim.step_continuous(np.asarray(action, dtype=np.float32))
                    elif sim.config.action_mode == "multidiscrete":
                        result = sim.step_multidiscrete(np.asarray(action, dtype=np.int64))
                    else:
                        result = sim.step(int(np.asarray(action).reshape(-1)[0]))
                    obs = normalize_observation(result.observation, vec_normalize)
                else:
                    throttle, brake, steer = _scripted_controls(sim, controller, rng, action_noise)
                    result = sim.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=-10)
                rows.append(asdict(result.telemetry))
                if result.terminated or result.truncated:
                    break
            score = _score_attempt(rows, start_progress_m=snapshot.monotonic_progress_m)
            elite_snapshot = snapshot_from_sim(sim, source="elite_search", source_file=str(state_library))
            summary = {
                "attempt": attempt_index,
                "seed": seed + attempt_index,
                "policy": policy,
                "score": score,
                "start_snapshot_id": snapshot.id,
                "start_progress_m": snapshot.monotonic_progress_m,
                "final_progress_m": rows[-1]["monotonic_progress_m"] if rows else snapshot.monotonic_progress_m,
                "segment_complete": rows[-1].get("segment_complete", False) if rows else False,
                "termination_reason": rows[-1].get("termination_reason", "empty") if rows else "empty",
                "valid_lap": rows[-1].get("valid_lap", False) if rows else False,
                "collided": rows[-1].get("collided", False) if rows else False,
                "off_track": rows[-1].get("off_track", False) if rows else False,
                "steps": len(rows),
            }
            attempts_summary.append(summary)
            attempt_rows.append((score, rows, elite_snapshot, summary))
    finally:
        close_vecnormalize(vec_normalize)
    ranked = sorted(attempt_rows, key=lambda item: item[0], reverse=True)
    elite_snapshots = [item[2] for item in ranked[:top_k]]
    for rank, (_, rows, _, summary) in enumerate(ranked[:top_k]):
        telemetry_path = selected_dir / f"elite-rank-{rank:03d}-attempt-{summary['attempt']:03d}-steps.jsonl"
        _write_jsonl(telemetry_path, rows)
        summary["selected_telemetry"] = str(telemetry_path)
    _write_jsonl(output_dir / "attempts.jsonl", attempts_summary)
    elite_library_path = output_dir / "elite_state_library.json"
    write_state_library(
        elite_library_path,
        elite_snapshots,
        source="elite_search",
        metadata={
            "state_library": str(state_library),
            "attempts": attempts,
            "max_steps": max_steps,
            "seed": seed,
            "top_k": top_k,
            "segment_length_m": segment_length_m,
            "start_min_progress_m": start_min_progress_m,
            "start_max_progress_m": start_max_progress_m,
            "policy": policy,
            "action_noise": action_noise,
            "checkpoint": checkpoint,
            "metadata_mode": metadata_mode,
        },
    )
    summary = {
        "run_id": output_dir.name,
        "state_library": str(state_library),
        "elite_state_library": str(elite_library_path),
        "attempts": attempts_summary,
        "top_attempts": [item[3] for item in ranked[:top_k]],
    }
    (output_dir / "elite_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"elite-search-{timestamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short segment attempts and save elite section-exit states.")
    parser.add_argument("--state-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--segment-length-m", type=float, default=700.0)
    parser.add_argument("--start-min-progress-m", type=float)
    parser.add_argument("--start-max-progress-m", type=float)
    parser.add_argument("--policy", choices=["scripted", "random", "ppo-deterministic", "ppo-stochastic"], default="scripted")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metadata-mode", choices=["auto", "require", "ignore"], default="auto")
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--position-noise-m", type=float, default=0.0)
    parser.add_argument("--heading-noise-deg", type=float, default=0.0)
    parser.add_argument("--speed-noise-kph", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or default_output_dir()
    run_root = run_elite_search(
        state_library=args.state_library,
        output_dir=output_dir,
        attempts=args.attempts,
        max_steps=args.max_steps,
        seed=args.seed,
        top_k=args.top_k,
        segment_length_m=args.segment_length_m,
        start_min_progress_m=args.start_min_progress_m,
        start_max_progress_m=args.start_max_progress_m,
        policy=args.policy,
        action_noise=args.action_noise,
        position_noise_m=args.position_noise_m,
        heading_noise_deg=args.heading_noise_deg,
        speed_noise_kph=args.speed_noise_kph,
        checkpoint=args.checkpoint,
        device=args.device,
        metadata_mode=args.metadata_mode,
    )
    print(f"elite_search_complete run={run_root} elite_library={run_root / 'elite_state_library.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
