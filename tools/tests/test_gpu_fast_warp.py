import importlib.util

import pytest

from f1rl.gpu_fast_warp import require_warp, warp_status
from f1rl.hardware import warp_runtime_info


def test_warp_status_reports_optional_dependency_state() -> None:
    status = warp_status()
    info = status.as_dict()

    assert set(info) == {"installed", "version", "cuda_available", "cuda_device_count", "error"}
    assert info["installed"] == (importlib.util.find_spec("warp") is not None)
    assert warp_runtime_info() == info


def test_require_warp_fails_cleanly_when_optional_dependency_missing() -> None:
    if importlib.util.find_spec("warp") is not None:
        pytest.skip("warp is installed in this environment")

    with pytest.raises(RuntimeError, match="warp-lang"):
        require_warp()
