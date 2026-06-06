# pyright: reportPrivateImportUsage=false
"""Deterministic CPU evaluation for learned policy checkpoints."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1rl.config import LEARNED_DIR, SimConfig, dataclass_to_dict
from f1rl.learned_policy import load_policy_checkpoint
from f1rl.sim import MonzaSim


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonl_open(path: Path, mode: str = "wt") -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _telemetry_path(output_dir: Path, episode: int, compression: str) -> Path:
    suffix = ".jsonl.gz" if compression == "gzip" else ".jsonl"
    return output_dir / "selected_telemetry" / f"policy-eval-episode-{episode:03d}-steps{suffix}"


def _policy_action(
    actor: Any,
    normalizer: Any,
    obs: np.ndarray,
    *,
    device: torch.device,
    deterministic: bool,
) -> tuple[float, float, float]:
    obs_tensor = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
    obs_tensor = normalizer.normalize_tensor(obs_tensor)
    with torch.no_grad():
        if deterministic:
            action = actor.deterministic_action(obs_tensor)
        else:
            action, _, _ = actor.sample_action(obs_tensor)
    values = action.detach().cpu().numpy().reshape(-1)
    return float(values[0]), float(values[1]), float(values[2])


def evaluate_policy(
    *,
    policy: Path,
    output_dir: Path,
    episodes: int,
    deterministic: bool,
    normal_start: bool,
    observation_profile: str,
    physics_model: str,
    write_telemetry: str,
    max_steps: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if physics_model not in {"v1", "v2"}:
        raise ValueError("physics_model must be one of: v1, v2")
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    actor, normalizer, metadata = load_policy_checkpoint(policy, device=torch_device)
    sim_config = SimConfig(
        max_steps=max_steps,
        action_mode="continuous",
        action_set="racing",
        observation_profile=observation_profile,
        physics_model=physics_model,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    episodes_summary: list[dict[str, Any]] = []

    for episode in range(episodes):
        sim = MonzaSim(sim_config)
        options: dict[str, Any] = {}
        if normal_start:
            options["start_speed_kph"] = 80.0
            options["start_progress_m"] = 0.0
        obs, _ = sim.reset(seed=seed + episode, options=options)
        telemetry_rows: list[dict[str, Any]] = []
        path = _telemetry_path(output_dir, episode, write_telemetry) if write_telemetry != "none" else None
        writer = _jsonl_open(path, "wt") if path is not None else None
        try:
            for _ in range(max_steps):
                throttle, brake, steer = _policy_action(
                    actor,
                    normalizer,
                    obs,
                    device=torch_device,
                    deterministic=deterministic,
                )
                result = sim.step_controls(throttle=throttle, brake=brake, steer=steer, action_id=-20)
                obs = result.observation
                row = asdict(result.telemetry)
                if writer is not None:
                    writer.write(json.dumps(row, default=_json_default) + "\n")
                telemetry_rows.append(row)
                if result.terminated or result.truncated:
                    break
        finally:
            if writer is not None:
                writer.close()
        final = telemetry_rows[-1] if telemetry_rows else {}
        best_progress = max([0.0, *[float(row["monotonic_progress_m"]) for row in telemetry_rows]])
        valid = bool(final.get("termination_reason") == "lap_complete" and final.get("valid_lap", False))
        lap_time_s = float(final.get("sim_time_s", 0.0) or 0.0) if valid else None
        episode_summary = {
            "episode": episode,
            "seed": seed + episode,
            "steps": len(telemetry_rows),
            "termination_reason": final.get("termination_reason", "empty"),
            "valid_lap": valid,
            "completed_lap": bool(final.get("termination_reason") == "lap_complete"),
            "lap_time_s": lap_time_s,
            "best_progress_m": best_progress,
            "final_progress_m": float(final.get("monotonic_progress_m", 0.0) or 0.0),
            "final_speed_kph": final.get("speed_kph"),
            "telemetry_path": str(path) if path is not None else None,
        }
        episodes_summary.append(episode_summary)
        if path is not None:
            trace_rows.append(
                {
                    "rank": episode,
                    "generation": episode,
                    "candidate_index": episode,
                    "seed": seed + episode,
                    "path": str(path),
                    "best_progress_m": best_progress,
                    "termination_reason": episode_summary["termination_reason"],
                    "valid_lap": valid,
                    "lap_time_s": lap_time_s,
                }
            )

    valid_times = [float(row["lap_time_s"]) for row in episodes_summary if row["lap_time_s"] is not None]
    summary = {
        "kind": "f1rl_policy_eval_summary",
        "policy": str(policy),
        "policy_metadata": metadata,
        "output_dir": str(output_dir),
        "physics_model": physics_model,
        "physics_version": sim_config.physics_version,
        "physics_calibration_id": sim_config.physics_calibration_id,
        "sim_config": dataclass_to_dict(sim_config),
        "deterministic": deterministic,
        "normal_start": normal_start,
        "episodes": episodes,
        "valid_lap_count": len(valid_times),
        "valid_lap_rate": len(valid_times) / max(episodes, 1),
        "fastest_valid_lap_s": min(valid_times) if valid_times else None,
        "average_valid_lap_s": float(np.mean(valid_times)) if valid_times else None,
        "terminal_reason_counts": {
            reason: sum(1 for row in episodes_summary if row["termination_reason"] == reason)
            for reason in sorted({str(row["termination_reason"]) for row in episodes_summary})
        },
        "episodes_summary": episodes_summary,
        "replay_command": f'uv run --no-sync python -m f1rl.replay "{output_dir / "selected_telemetry"}"',
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    manifest = {
        "kind": "f1rl_policy_eval_manifest",
        "backend": "cpu_policy_eval",
        "physics_model": physics_model,
        "physics_version": sim_config.physics_version,
        "physics_calibration_id": sim_config.physics_calibration_id,
        "trace_count": len(trace_rows),
        "traces": trace_rows,
    }
    (output_dir / "selected_telemetry" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        "policy_eval_complete "
        f"valid_laps={summary['valid_lap_count']}/{episodes} "
        f"fastest_valid_lap_s={summary['fastest_valid_lap_s']}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned policy with CPU MonzaSim.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", default=str(LEARNED_DIR / "policy-eval"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--normal-start", action="store_true", default=True)
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--physics-model", default="v1")
    parser.add_argument("--write-telemetry", choices=("none", "gzip"), default="gzip")
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_policy(
        policy=Path(args.policy),
        output_dir=Path(args.output_dir),
        episodes=args.episodes,
        deterministic=args.deterministic,
        normal_start=args.normal_start,
        observation_profile=args.observation_profile,
        physics_model=args.physics_model,
        write_telemetry=args.write_telemetry,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
