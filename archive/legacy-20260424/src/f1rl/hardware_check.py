"""Hardware validation command for local GPU-enabled workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from f1rl.hardware import get_torch_runtime_info, query_nvidia_smi, require_gpu_available


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local GPU runtime for F1 RL training.")
    parser.add_argument("--skip-train-smoke", action="store_true", help="Skip PPO smoke-train validation.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = get_torch_runtime_info()
    require_gpu_available("gpu")
    smi_output = query_nvidia_smi()

    payload: dict[str, object] = {
        "torch_version": runtime.torch_version,
        "cuda_available": runtime.cuda_available,
        "cuda_version": runtime.cuda_version,
        "cudnn_version": runtime.cudnn_version,
        "device_count": runtime.device_count,
        "device_name": runtime.device_name,
        "train_smoke_ran": False,
    }

    if not args.skip_train_smoke:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "f1rl.train",
                "--mode",
                "smoke",
                "--iterations",
                "1",
                "--device",
                "gpu",
                "--require-gpu",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["train_smoke_ran"] = True
        payload["train_smoke_returncode"] = completed.returncode
        if completed.returncode != 0:
            payload["train_smoke_stdout"] = completed.stdout[-4000:]
            payload["train_smoke_stderr"] = completed.stderr[-4000:]
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(smi_output)
                print(json.dumps(payload, indent=2))
            return completed.returncode or 1

    if args.json:
        payload["nvidia_smi"] = smi_output
        print(json.dumps(payload, indent=2))
    else:
        print(smi_output)
        print(
            "hardware_check_complete "
            f"cuda_available={runtime.cuda_available} "
            f"cuda_version={runtime.cuda_version} "
            f"device_name={runtime.device_name} "
            f"train_smoke_ran={payload['train_smoke_ran']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
