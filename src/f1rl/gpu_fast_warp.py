# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""Optional NVIDIA Warp integration checks for the future fused GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WarpStatus:
    installed: bool
    version: str | None = None
    cuda_available: bool | None = None
    cuda_device_count: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "error": self.error,
        }


def _import_warp() -> Any:
    import warp as wp

    return wp


def warp_status() -> WarpStatus:
    try:
        wp = _import_warp()
    except Exception as exc:
        return WarpStatus(installed=False, error=f"{type(exc).__name__}: {exc}")
    try:
        wp.init()
        cuda_device_count = int(wp.get_cuda_device_count())
        return WarpStatus(
            installed=True,
            version=str(getattr(wp, "__version__", "unknown")),
            cuda_available=cuda_device_count > 0,
            cuda_device_count=cuda_device_count,
        )
    except Exception as exc:
        return WarpStatus(
            installed=True,
            version=str(getattr(wp, "__version__", "unknown")),
            cuda_available=False,
            cuda_device_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def require_warp() -> Any:
    status = warp_status()
    if not status.installed:
        raise RuntimeError(
            "GPU fused engine requires optional dependency 'warp-lang'. "
            "Install with `uv sync --extra gpu-fast` or `uv run --extra gpu-fast ...`."
        )
    if status.error is not None:
        raise RuntimeError(f"GPU fused engine cannot initialize Warp: {status.error}")
    return _import_warp()


def run_warp_torch_interop_smoke(*, device: str = "cuda") -> dict[str, Any]:
    """Run a tiny Warp kernel over a torch tensor without copying through CPU."""

    wp = require_warp()
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Warp/Torch smoke requested but torch.cuda.is_available() is false.")
    torch_device = torch.device(device)
    values = torch.arange(8, device=torch_device, dtype=torch.float32)
    output = torch.empty_like(values)

    @wp.kernel
    def _scale_kernel(
        src: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
        dst: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    ) -> None:
        tid = wp.tid()
        dst[tid] = src[tid] * 2.0 + 1.0

    wp_device = "cuda" if torch_device.type == "cuda" else "cpu"
    wp.launch(
        _scale_kernel,
        dim=values.numel(),
        inputs=[wp.from_torch(values), wp.from_torch(output)],
        device=wp_device,
    )
    wp.synchronize_device(wp_device)
    expected = values * 2.0 + 1.0
    max_abs_error = float(torch.max(torch.abs(output - expected)).detach().cpu().item())
    return {
        "device": str(torch_device),
        "warp_device": wp_device,
        "numel": int(values.numel()),
        "max_abs_error": max_abs_error,
        "passed": max_abs_error <= 1e-6,
    }
