import subprocess
import sys


def run_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"cmd failed: {' '.join(cmd)}\nstdout={result.stdout}\nstderr={result.stderr}"


def test_manual_headless_smoke() -> None:
    run_module("f1rl.manual", "--headless", "--controller", "scripted", "--max-steps", "60")


def test_rollout_random_headless_smoke() -> None:
    run_module("f1rl.rollout", "--policy", "random", "--headless", "--steps", "100")
