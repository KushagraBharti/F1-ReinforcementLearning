"""Hardware detection helpers for local runtime configuration."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class TorchRuntimeInfo:
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: int | None
    device_count: int
    device_name: str | None


def get_torch_runtime_info() -> TorchRuntimeInfo:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else None
    return TorchRuntimeInfo(
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device_count=device_count,
        device_name=device_name,
    )


def detect_use_gpu(device: str = "auto") -> bool:
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported device mode: {device}")
    runtime = get_torch_runtime_info()
    if device == "cpu":
        return False
    if device == "gpu":
        return runtime.cuda_available
    return runtime.cuda_available


def require_gpu_available(device: str = "auto") -> None:
    if device == "cpu":
        return
    runtime = get_torch_runtime_info()
    if not runtime.cuda_available:
        raise RuntimeError(
            "GPU was requested but CUDA-enabled Torch is not available in the project environment."
        )


def query_nvidia_smi() -> str:
    completed = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    return completed.stdout.strip()


def query_nvidia_smi_compute_apps() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi compute-app query failed")
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, process_name, used_gpu_memory = next(csv.reader([stripped], skipinitialspace=True))
        rows.append(
            {
                "pid": pid.strip(),
                "process_name": process_name.strip(),
                "used_gpu_memory_mib": used_gpu_memory.strip(),
            }
        )
    return rows
