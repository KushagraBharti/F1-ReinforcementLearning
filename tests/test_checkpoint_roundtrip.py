import subprocess
import sys

import pytest


@pytest.mark.timeout(300)
def test_checkpoint_roundtrip_smoke() -> None:
    pytest.importorskip("ray")

    train_cmd = [sys.executable, "-m", "f1rl.train", "--mode", "smoke", "--iterations", "1"]
    train = subprocess.run(train_cmd, check=False, capture_output=True, text=True)
    assert train.returncode == 0, f"train failed:\nstdout={train.stdout}\nstderr={train.stderr}"

    eval_cmd = [sys.executable, "-m", "f1rl.eval", "--checkpoint", "latest", "--headless", "--steps", "80"]
    eval_result = subprocess.run(eval_cmd, check=False, capture_output=True, text=True)
    assert (
        eval_result.returncode == 0
    ), f"eval failed:\nstdout={eval_result.stdout}\nstderr={eval_result.stderr}"
