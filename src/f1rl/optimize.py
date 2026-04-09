"""Run a small benchmark-driven candidate loop and keep only better checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from f1rl.artifacts import new_run_paths
from f1rl.constants import CHAMPIONS_DIR

TRAIN_CHECKPOINT_RE = re.compile(r"checkpoint=(?P<checkpoint>.+?) use_gpu=")
BENCHMARK_SUMMARY_RE = re.compile(r"summary=(?P<summary>.+?) promoted=(?P<promoted>True|False)")

CANDIDATE_LIBRARY: list[dict[str, Any]] = [
    {
        "name": "control_focus",
        "env_overrides": {
            "dynamics": {
                "steering_response": 0.48,
                "high_speed_steer_reduction": 0.7,
                "lateral_grip": 0.95,
            },
            "reward": {
                "progress_reward": 3.0,
                "alignment_weight": 0.03,
                "centerline_penalty_weight": 0.025,
            },
            "no_progress_limit_steps": 120,
        },
    },
    {
        "name": "sensor_focus",
        "env_overrides": {
            "dynamics": {
                "sensor_count": 11,
                "sensor_spread_deg": 150.0,
                "sensor_forward_bias": 2.0,
            },
            "reward": {
                "alignment_weight": 0.025,
                "steering_change_penalty_weight": 0.004,
            },
        },
    },
    {
        "name": "grip_speed_balance",
        "env_overrides": {
            "dynamics": {
                "max_speed": 11.5,
                "lateral_grip": 0.97,
                "wheelbase": 30.0,
            },
            "reward": {
                "forward_speed_weight": 0.04,
                "centerline_penalty_weight": 0.018,
            },
        },
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark candidate configs against current champion.")
    parser.add_argument("--baseline-checkpoint", default="latest")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--profile", choices=["quick", "standard"], default="quick")
    parser.add_argument("--no-clips", action="store_true")
    return parser.parse_args(argv)


def _run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def _extract_checkpoint(stdout: str) -> str:
    match = TRAIN_CHECKPOINT_RE.search(stdout)
    if match is None:
        raise RuntimeError(f"Unable to parse checkpoint from output:\n{stdout}")
    return match.group("checkpoint")


def _extract_benchmark(stdout: str) -> tuple[str, bool]:
    match = BENCHMARK_SUMMARY_RE.search(stdout)
    if match is None:
        raise RuntimeError(f"Unable to parse benchmark summary from output:\n{stdout}")
    return match.group("summary"), match.group("promoted") == "True"


def _ensure_baseline_champion(args: argparse.Namespace) -> dict[str, Any]:
    champion_path = CHAMPIONS_DIR / "current.json"
    if champion_path.exists():
        return json.loads(champion_path.read_text(encoding="utf-8"))

    benchmark_stdout = _run_command(
        [
            sys.executable,
            "-m",
            "f1rl.benchmark",
            "--checkpoint",
            args.baseline_checkpoint,
            "--profile",
            args.profile,
            "--promote-if-best",
            *([] if not args.no_clips else ["--no-clips"]),
        ]
    )
    summary_path, _ = _extract_benchmark(benchmark_stdout)
    return json.loads(Path(summary_path).read_text(encoding="utf-8"))


def run_optimize(args: argparse.Namespace) -> int:
    run_paths = new_run_paths(prefix="optimize")
    baseline = _ensure_baseline_champion(args)
    manifest: dict[str, Any] = {
        "baseline": baseline,
        "candidates": [],
    }

    for candidate in CANDIDATE_LIBRARY[: max(args.max_candidates, 0)]:
        env_json = json.dumps(candidate["env_overrides"])
        train_stdout = _run_command(
            [
                sys.executable,
                "-m",
                "f1rl.train",
                "--mode",
                "benchmark",
                "--iterations",
                str(args.iterations),
                "--device",
                args.device,
                "--env-config-json",
                env_json,
                "--run-tag",
                candidate["name"],
                *(["--require-gpu"] if args.device == "gpu" else []),
            ]
        )
        checkpoint_path = _extract_checkpoint(train_stdout)
        benchmark_stdout = _run_command(
            [
                sys.executable,
                "-m",
                "f1rl.benchmark",
                "--checkpoint",
                checkpoint_path,
                "--profile",
                args.profile,
                "--promote-if-best",
                *([] if not args.no_clips else ["--no-clips"]),
            ]
        )
        summary_path, promoted = _extract_benchmark(benchmark_stdout)
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        manifest["candidates"].append(
            {
                "name": candidate["name"],
                "env_overrides": candidate["env_overrides"],
                "checkpoint": checkpoint_path,
                "summary_path": summary_path,
                "promoted": promoted,
                "comparison_reason": summary.get("comparison_reason"),
            }
        )

    manifest_path = run_paths.root / "optimize_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    kept = sum(1 for candidate in manifest["candidates"] if candidate["promoted"])
    rejected = len(manifest["candidates"]) - kept
    print(f"optimize_complete kept={kept} rejected={rejected} manifest={manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_optimize(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
