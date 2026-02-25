"""Run end-to-end project validations."""

from __future__ import annotations

import argparse
import subprocess
import sys

from gymnasium.utils.env_checker import check_env

from f1rl.config import EnvConfig, to_dict
from f1rl.env import F1RaceEnv


def run_command(cmd: list[str]) -> None:
    print(f"[validate] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def run_import_smoke() -> None:
    from f1rl import F1RaceEnv as _F1RaceEnv
    from f1rl.manual import run_manual as _run_manual
    from f1rl.rollout import run_rollout as _run_rollout
    from f1rl.train import run_training as _run_training

    _ = _F1RaceEnv, _run_manual, _run_rollout, _run_training


def run_env_checker() -> None:
    config = EnvConfig()
    config.max_steps = 64
    config.render.enabled = False
    config.render_mode = None
    env = F1RaceEnv(to_dict(config))
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1 RL validation suite.")
    parser.add_argument("--skip-train", action="store_true", help="Skip RLlib train/eval checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_import_smoke()
    run_env_checker()

    python = sys.executable
    run_command([python, "-m", "f1rl.manual", "--headless", "--autodrive", "--max-steps", "80"])
    run_command([python, "-m", "f1rl.rollout", "--policy", "random", "--headless", "--steps", "120"])

    if not args.skip_train:
        run_command([python, "-m", "f1rl.train", "--mode", "smoke", "--iterations", "1"])
        run_command([python, "-m", "f1rl.eval", "--checkpoint", "latest", "--headless", "--steps", "120"])

    print("[validate] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
