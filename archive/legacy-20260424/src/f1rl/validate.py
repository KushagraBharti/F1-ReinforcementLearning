"""Run end-to-end project validations for the torch-native runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from gymnasium.utils.env_checker import check_env

from f1rl.artifacts import resolve_checkpoint
from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv
from f1rl.hardware import query_nvidia_smi, query_nvidia_smi_compute_apps
from f1rl.torch_runtime import TERMINATION_REASONS, TorchSimBatch


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[validate] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return completed


def run_import_smoke() -> None:
    from f1rl.benchmark import run_benchmark as _run_benchmark
    from f1rl.manual import run_manual as _run_manual
    from f1rl.swarm import run_swarm as _run_swarm
    from f1rl.torch_agent import load_torch_policy as _load_torch_policy
    from f1rl.train import run_training as _run_training

    _ = (_run_benchmark, _run_manual, _run_swarm, _load_torch_policy, _run_training)


def run_env_checker() -> None:
    config = EnvConfig()
    config.max_steps = 32
    config.render.enabled = False
    config.render_mode = None
    env = F1RaceEnv(to_dict(config))
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def run_sim_parity() -> None:
    config = to_dict(EnvConfig())
    cpu_sim = TorchSimBatch(config, num_cars=4, device="cpu")
    cpu_obs = cpu_sim.reset(seed=7)
    assert cpu_obs.shape[-1] == cpu_sim.observation_dim
    action_batches = [
        torch.tensor([[0.2, 0.8], [-0.3, 0.2], [0.0, -0.5], [0.4, 0.1]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0], [0.1, 0.6], [-0.1, 0.0], [0.3, -0.4]], dtype=torch.float32),
        torch.tensor([[-0.2, 0.5], [0.0, 0.0], [0.2, 0.8], [-0.5, -0.2]], dtype=torch.float32),
    ]
    if torch.cuda.is_available():
        gpu_sim = TorchSimBatch(config, num_cars=4, device="gpu")
        gpu_obs = gpu_sim.reset(seed=7)
        assert torch.allclose(cpu_obs.cpu(), gpu_obs.cpu(), atol=1e-3, rtol=1e-3)
        for actions in action_batches:
            cpu_step = cpu_sim.step(actions)
            gpu_step = gpu_sim.step(actions.to(gpu_sim.device))
            assert torch.allclose(cpu_step.observations.cpu(), gpu_step.observations.cpu(), atol=1e-3, rtol=1e-3)
            assert torch.allclose(cpu_step.rewards.cpu(), gpu_step.rewards.cpu(), atol=1e-3, rtol=1e-3)
            assert torch.equal(cpu_step.telemetry["termination_code"].cpu(), gpu_step.telemetry["termination_code"].cpu())
            assert torch.equal(cpu_step.telemetry["current_checkpoint_index"].cpu(), gpu_step.telemetry["current_checkpoint_index"].cpu())


def run_telemetry_gates() -> None:
    config = to_dict(EnvConfig())
    simulator = TorchSimBatch(config, num_cars=3, device="cpu")
    simulator.reset(seed=9)
    step = None
    moving_seen = False
    for _ in range(12):
        step = simulator.step(torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 1.0]], dtype=torch.float32))
        moving_seen = moving_seen or bool(step.telemetry["moving"].any().item())
    assert step is not None
    expected_keys = {
        "speed_kph",
        "distance_travelled",
        "current_checkpoint_index",
        "time_since_last_checkpoint",
        "termination_code",
        "throttle",
        "brake",
        "steering",
        "min_wall_distance_ratio",
        "moving",
    }
    assert expected_keys.issubset(step.telemetry.keys())
    assert moving_seen, "moving-car classification never triggered"
    metadata = simulator.runtime_metadata(policy_device="cpu", renderer_backend="headless")
    thresholds = metadata["telemetry_thresholds"]
    assert thresholds["moving_speed_threshold_kph"] > 0.0
    assert thresholds["moving_progress_threshold"] > 0.0
    snapshot = simulator.snapshot()
    reasons = {
        TERMINATION_REASONS[int(code)]
        for code in snapshot["termination_code"].detach().cpu().tolist()
    }
    assert reasons


def run_cuda_visibility_gate() -> None:
    if not torch.cuda.is_available():
        return
    _ = torch.zeros((4096, 4096), device="cuda")
    current_pid = str(os.getpid())
    compute_rows = query_nvidia_smi_compute_apps()
    if not any(row["pid"] == current_pid for row in compute_rows):
        raise RuntimeError(
            "CUDA runtime is available but the current Python process was not visible in nvidia-smi compute apps."
        )


def _extract_path(stdout: str, key: str) -> Path:
    marker = f"{key}="
    if marker not in stdout:
        raise RuntimeError(f"Unable to find '{key}=...' in command output:\n{stdout}")
    tail = stdout.split(marker, 1)[1].strip()
    if key == "summary" and " promoted=" in tail:
        tail = tail.split(" promoted=", 1)[0].strip()
    return Path(tail)


def run_runtime_metadata_gates(train_output: str, eval_output: str, benchmark_output: str, swarm_output: str) -> None:
    checkpoint_dir = _extract_path(train_output, "checkpoint")
    checkpoint_metadata = checkpoint_dir / "torch_metadata.json"
    if not checkpoint_metadata.exists():
        raise RuntimeError(f"Missing torch checkpoint metadata: {checkpoint_metadata}")
    training_metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
    if torch.cuda.is_available():
        assert training_metadata["training_metadata"]["sim_device"].startswith("cuda")
        assert training_metadata["training_metadata"]["policy_device"].startswith("cuda")

    eval_summary = json.loads(_extract_path(eval_output, "summary").read_text(encoding="utf-8"))
    benchmark_summary = json.loads(_extract_path(benchmark_output, "summary").read_text(encoding="utf-8"))
    swarm_summary = json.loads(_extract_path(swarm_output, "summary").read_text(encoding="utf-8"))
    for summary in (eval_summary, benchmark_summary, swarm_summary):
        runtime = summary["runtime"]
        assert "sim_device" in runtime
        assert "policy_device" in runtime
        assert "renderer_backend" in runtime
        assert "telemetry_thresholds" in runtime
        if torch.cuda.is_available():
            assert str(runtime["sim_device"]).startswith("cuda")
            if summary.get("backend") == "torch_native":
                assert str(runtime["policy_device"]).startswith("cuda")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1 RL validation suite.")
    parser.add_argument("--skip-train", action="store_true", help="Skip train/eval checks.")
    parser.add_argument("--skip-hardware", action="store_true", help="Skip nvidia-smi hardware validation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_import_smoke()
    run_env_checker()
    run_sim_parity()
    run_telemetry_gates()

    python = sys.executable
    if not args.skip_hardware and torch.cuda.is_available():
        print(query_nvidia_smi())
        run_cuda_visibility_gate()
    run_command([python, "-m", "f1rl.manual", "--headless", "--controller", "scripted", "--max-steps", "80"])
    swarm_result = run_command([python, "-m", "f1rl.swarm", "--policy", "random", "--headless", "--cars", "8", "--steps", "40", "--device", "auto"])

    if not args.skip_train:
        train_result = run_command([python, "-m", "f1rl.train", "--mode", "smoke", "--iterations", "1", "--device", "auto"])
        checkpoint_dir = resolve_checkpoint("latest")
        eval_result = run_command([python, "-m", "f1rl.eval", "--checkpoint", str(checkpoint_dir), "--headless", "--steps", "120", "--device", "auto"])
        benchmark_result = run_command([python, "-m", "f1rl.benchmark", "--checkpoint", str(checkpoint_dir), "--profile", "quick", "--no-clips"])
        run_runtime_metadata_gates(
            train_result.stdout,
            eval_result.stdout,
            benchmark_result.stdout,
            swarm_result.stdout,
        )

    print("[validate] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
