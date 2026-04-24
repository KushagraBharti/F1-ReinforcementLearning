"""Stable-Baselines3 PPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from f1rl.config import ARTIFACTS_DIR, SimConfig, dataclass_to_dict
from f1rl.env import MonzaEnv
from f1rl.hardware import torch_device


def _make_env(max_steps: int, seed: int):
    def factory() -> MonzaEnv:
        env = MonzaEnv(SimConfig(max_steps=max_steps))
        env.reset(seed=seed)
        return env

    return factory


def run_training(
    *,
    timesteps: int,
    seed: int,
    n_envs: int,
    max_steps: int,
    device: str,
    checkpoint_every: int,
) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise RuntimeError("Training requires stable-baselines3. Run `uv sync --active --all-extras --all-packages`.") from exc

    resolved_device = torch_device(device)
    run_id = f"train-{time.strftime('%Y%m%d-%H%M%S')}"
    run_root = ARTIFACTS_DIR / run_id
    checkpoint_dir = run_root / "checkpoints"
    log_dir = run_root / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(_make_env(max_steps, seed), n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        device=resolved_device,
        tensorboard_log=str(log_dir),
        n_steps=64,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        gamma=0.995,
    )
    callback = CheckpointCallback(
        save_freq=max(checkpoint_every, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_monza",
    )
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "device": resolved_device,
        "sim_config": dataclass_to_dict(SimConfig(max_steps=max_steps)),
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.learn(total_timesteps=timesteps, callback=callback, tb_log_name="ppo")
    final_path = checkpoint_dir / "final_model.zip"
    model.save(final_path)
    env.close()
    print(f"train_complete run={run_root} checkpoint={final_path} device={resolved_device}")
    return final_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the simplified Monza simulator.")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
