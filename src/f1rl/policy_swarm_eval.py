# pyright: reportPrivateImportUsage=false
"""Evaluate many rollout traces for learned policy checkpoints."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
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


def _find_checkpoints(policy_dir: Path, checkpoints: str) -> list[Path]:
    if policy_dir.is_file():
        return [policy_dir]
    if checkpoints == "best":
        return [policy_dir / "best_policy.pt"]
    paths = sorted((policy_dir / "checkpoints").glob("*.pt"))
    if checkpoints == "latest":
        return paths[-1:] if paths else [policy_dir / "best_policy.pt"]
    candidates = [policy_dir / "best_policy.pt", *paths, policy_dir / "final_policy.pt"]
    seen: set[Path] = set()
    selected: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            selected.append(path)
    return selected


def _action(actor: Any, normalizer: Any, obs: np.ndarray, *, device: torch.device, deterministic: bool) -> np.ndarray:
    obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
    obs_t = normalizer.normalize_tensor(obs_t)
    with torch.no_grad():
        if deterministic:
            action = actor.deterministic_action(obs_t)
        else:
            action, _, _ = actor.sample_action(obs_t)
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def run_policy_swarm_eval(
    *,
    policy_dir: Path,
    output_dir: Path,
    swarm_size: int,
    checkpoints: str,
    observation_profile: str,
    physics_model: str,
    deterministic: bool,
    max_steps: int,
    seed: int,
    device: str,
    full_telemetry_limit: int,
    start_position_noise_m: float,
    start_heading_noise_deg: float,
    start_speed_noise_kph: float,
) -> Path:
    if physics_model not in {"v1", "v2"}:
        raise ValueError("physics_model must be one of: v1, v2")
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = _find_checkpoints(policy_dir, checkpoints)
    manifest_traces: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    sim_config = SimConfig(
        max_steps=max_steps,
        action_mode="continuous",
        action_set="racing",
        observation_profile=observation_profile,
        physics_model=physics_model,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        actor, normalizer, metadata = load_policy_checkpoint(checkpoint_path, device=torch_device)
        summaries: list[dict[str, Any]] = []
        trace_dir = output_dir / f"checkpoint_{checkpoint_index:04d}"
        reuse_deterministic_trace = (
            deterministic
            and start_position_noise_m == 0.0
            and start_heading_noise_deg == 0.0
            and start_speed_noise_kph == 0.0
        )
        template_rows: list[dict[str, Any]] | None = None
        template_path: Path | None = None
        for car_index in range(swarm_size):
            write_full = car_index < full_telemetry_limit
            path = trace_dir / f"policy-swarm-checkpoint-{checkpoint_index:04d}-car-{car_index:05d}-steps.jsonl.gz"
            if reuse_deterministic_trace and template_rows is not None:
                telemetry_rows = template_rows
                if write_full:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        path.unlink()
                    if template_path is not None:
                        try:
                            os.link(template_path, path)
                        except OSError:
                            shutil.copyfile(template_path, path)
            else:
                sim = MonzaSim(sim_config)
                reset_options = {
                    "start_speed_kph": 80.0,
                    "start_progress_m": 0.0,
                    "position_noise_m": start_position_noise_m,
                    "heading_noise_deg": start_heading_noise_deg,
                    "speed_noise_kph": start_speed_noise_kph,
                }
                obs, _ = sim.reset(seed=seed + checkpoint_index * 100_000 + car_index, options=reset_options)
                telemetry_rows = []
                writer = _jsonl_open(path, "wt") if write_full else None
                try:
                    for _ in range(max_steps):
                        action = _action(actor, normalizer, obs, device=torch_device, deterministic=deterministic)
                        result = sim.step_controls(
                            throttle=float(action[0]),
                            brake=float(action[1]),
                            steer=float(action[2]),
                            action_id=-40,
                        )
                        obs = result.observation
                        row = asdict(result.telemetry)
                        telemetry_rows.append(row)
                        if writer is not None:
                            writer.write(json.dumps(row, default=_json_default) + "\n")
                        if result.terminated or result.truncated:
                            break
                finally:
                    if writer is not None:
                        writer.close()
                if reuse_deterministic_trace:
                    template_rows = list(telemetry_rows)
                    template_path = path if write_full else None
            final = telemetry_rows[-1] if telemetry_rows else {}
            best_progress = max([0.0, *[float(row["monotonic_progress_m"]) for row in telemetry_rows]])
            valid = bool(final.get("termination_reason") == "lap_complete" and final.get("valid_lap", False))
            summary = {
                "checkpoint_index": checkpoint_index,
                "checkpoint": str(checkpoint_path),
                "car_index": car_index,
                "seed": seed + checkpoint_index * 100_000 + car_index,
                "steps": len(telemetry_rows),
                "best_progress_m": best_progress,
                "termination_reason": final.get("termination_reason", "empty"),
                "valid_lap": valid,
                "lap_time_s": float(final.get("sim_time_s", 0.0) or 0.0) if valid else None,
                "path": str(path) if write_full else None,
            }
            summaries.append(summary)
            if write_full:
                manifest_traces.append(
                    {
                        "rank": car_index,
                        "generation": checkpoint_index,
                        "candidate_index": car_index,
                        "seed": summary["seed"],
                        "path": str(path),
                        "best_progress_m": best_progress,
                        "termination_reason": summary["termination_reason"],
                        "valid_lap": valid,
                        "lap_time_s": summary["lap_time_s"],
                    }
                )
        valid_times = [float(row["lap_time_s"]) for row in summaries if row["lap_time_s"] is not None]
        checkpoint_summaries.append(
            {
                "checkpoint_index": checkpoint_index,
                "checkpoint": str(checkpoint_path),
                "policy_metadata": metadata,
                "physics_model": physics_model,
                "physics_version": sim_config.physics_version,
                "physics_calibration_id": sim_config.physics_calibration_id,
                "swarm_size": swarm_size,
                "full_telemetry_count": min(swarm_size, full_telemetry_limit),
                "valid_lap_count": len(valid_times),
                "fastest_valid_lap_s": min(valid_times) if valid_times else None,
                "best_progress_m": max(float(row["best_progress_m"]) for row in summaries),
                "cars": summaries,
            }
        )
    manifest = {
        "kind": "f1rl_policy_swarm_manifest",
        "backend": "cpu_policy_swarm_eval",
        "physics_model": physics_model,
        "physics_version": sim_config.physics_version,
        "physics_calibration_id": sim_config.physics_calibration_id,
        "sim_config": dataclass_to_dict(sim_config),
        "deterministic": deterministic,
        "start_noise": {
            "position_noise_m": start_position_noise_m,
            "heading_noise_deg": start_heading_noise_deg,
            "speed_noise_kph": start_speed_noise_kph,
        },
        "trace_count": len(manifest_traces),
        "traces": manifest_traces,
        "checkpoint_summaries": checkpoint_summaries,
        "replay_command": f'uv run --no-sync python -m f1rl.policy_swarm_replay "{output_dir}" --by-checkpoint --speed 1',
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(f"policy_swarm_eval_complete manifest={output_dir / 'manifest.json'} traces={len(manifest_traces)}")
    return output_dir / "manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped policy checkpoint swarm evaluation.")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--output-dir", default=str(LEARNED_DIR / "v1-sac-swarms"))
    parser.add_argument("--swarm-size", type=int, default=1000)
    parser.add_argument("--checkpoints", default="all", choices=("all", "best", "latest"))
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--physics-model", default="v1")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-telemetry-limit", type=int, default=1000)
    parser.add_argument("--start-position-noise-m", type=float, default=0.0)
    parser.add_argument("--start-heading-noise-deg", type=float, default=0.0)
    parser.add_argument("--start-speed-noise-kph", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_policy_swarm_eval(
        policy_dir=Path(args.policy_dir),
        output_dir=Path(args.output_dir),
        swarm_size=args.swarm_size,
        checkpoints=args.checkpoints,
        observation_profile=args.observation_profile,
        physics_model=args.physics_model,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
        full_telemetry_limit=args.full_telemetry_limit,
        start_position_noise_m=args.start_position_noise_m,
        start_heading_noise_deg=args.start_heading_noise_deg,
        start_speed_noise_kph=args.start_speed_noise_kph,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
