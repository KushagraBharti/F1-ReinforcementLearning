from f1rl.replay import run_replay
from f1rl.scripted import run_scripted


def test_scripted_and_replay_smoke() -> None:
    run_scripted(steps=8, seed=11, telemetry=True)
    # Use the newest scripted telemetry from artifacts.
    from f1rl.config import ARTIFACTS_DIR

    steps_path = sorted(ARTIFACTS_DIR.glob("scripted-*/steps.jsonl"))[-1]
    assert run_replay(steps_path, headless=True) == 0
