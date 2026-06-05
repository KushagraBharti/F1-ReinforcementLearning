"""Local CPU/GPU runtime checks."""

from __future__ import annotations

import argparse
import json
import sys


def torch_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch is not installed.")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def runtime_info() -> dict:
    info = {"torch_installed": False, "cuda_available": False, "device_name": None, "torch_version": None, "cuda": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch_installed"] = True
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def warp_runtime_info() -> dict:
    from f1rl.gpu_fast_warp import warp_status

    return warp_status().as_dict()


def compute_policy(requested: str = "auto") -> dict:
    policy_device = torch_device(requested)
    return {
        "policy_device": policy_device,
        "neural_training": policy_device,
        "neural_inference": policy_device,
        "env_stepping": "cpu",
        "physics": "cpu",
        "geometry": "cpu",
        "rendering": "cpu",
        "keyboard_input": "cpu",
        "telemetry": "cpu",
        "track_preprocessing": "cpu",
        "vector_env_workers": "cpu",
        "evolution_search_gpu_backend": (
            "cuda"
            if policy_device == "cuda"
            else "cpu"
        ),
        "rule": (
            "Default manual/replay/SB3 simulator stepping stays CPU-bound. "
            "Use CUDA for PyTorch model forward/backward/inference and only for the explicit "
            "evolution-search GPU backend selected with --backend gpu."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyTorch/CUDA runtime visibility.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-warp", action="store_true")
    parser.add_argument("--warp-smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    info = runtime_info()
    if args.require_gpu and not info["cuda_available"]:
        raise RuntimeError(f"GPU required but unavailable: {info}")
    warp_info = warp_runtime_info()
    if args.require_warp and not warp_info["installed"]:
        raise RuntimeError(f"Warp required but unavailable: {warp_info}")
    if args.warp_smoke:
        from f1rl.gpu_fast_warp import run_warp_torch_interop_smoke

        smoke_device = "cuda" if args.device in {"auto", "cuda"} and info["cuda_available"] else "cpu"
        warp_info["torch_interop_smoke"] = run_warp_torch_interop_smoke(device=smoke_device)
    policy = compute_policy(args.device)
    info["compute_policy"] = policy
    info["warp"] = warp_info
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(
            "hardware "
            f"torch={info['torch_version']} cuda_available={info['cuda_available']} "
            f"cuda={info['cuda']} device={info['device_name']}"
        )
        print(
            "compute_policy "
            f"policy={policy['policy_device']} env={policy['env_stepping']} "
            f"physics={policy['physics']} rendering={policy['rendering']} telemetry={policy['telemetry']}"
        )
        print(
            "warp "
            f"installed={warp_info['installed']} version={warp_info['version']} "
            f"cuda_available={warp_info['cuda_available']} error={warp_info['error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
