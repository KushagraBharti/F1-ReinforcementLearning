"""Stable-Baselines3 PPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import ARTIFACTS_DIR, build_sim_config, dataclass_to_dict
from f1rl.curriculum import DEFAULT_SEGMENT_STAGES, CurriculumConfig, CurriculumSampler
from f1rl.env import MonzaEnv
from f1rl.hardware import compute_policy, torch_device
from f1rl.sim import MonzaSim


def _action_to_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def _make_env(
    max_steps: int,
    seed: int,
    curriculum: CurriculumConfig | None = None,
    reward_overrides: dict[str, float | None] | None = None,
):
    def factory() -> MonzaEnv:
        env = MonzaEnv(build_sim_config(max_steps=max_steps, reward_overrides=reward_overrides), curriculum=curriculum)
        env.reset(seed=seed)
        return env

    return factory


def _build_curriculum_config(
    *,
    mode: str,
    stage_count: int | None,
    promotion_resets: int,
    normal_start_probability: float,
) -> CurriculumConfig:
    if mode != "segments":
        return CurriculumConfig(mode=mode)
    stages = DEFAULT_SEGMENT_STAGES
    if stage_count is not None:
        if stage_count < 1:
            raise ValueError("--curriculum-stage-count must be at least 1.")
        stages = DEFAULT_SEGMENT_STAGES[: min(stage_count, len(DEFAULT_SEGMENT_STAGES))]
    return CurriculumConfig(
        mode=mode,
        stages=stages,
        promotion_resets=promotion_resets,
        normal_start_probability=normal_start_probability,
    )


def _reward_overrides_from_args(args: argparse.Namespace) -> dict[str, float | None]:
    return {
        "progress_scale": args.reward_progress_scale,
        "finish_bonus": args.reward_finish_bonus,
        "collision_penalty": args.reward_collision_penalty,
        "off_track_penalty": args.reward_off_track_penalty,
        "no_progress_penalty": args.reward_no_progress_penalty,
        "lateral_deadzone_m": args.reward_lateral_deadzone_m,
        "lateral_penalty_scale": args.reward_lateral_penalty_scale,
        "track_limit_safe_ray_m": args.reward_track_limit_safe_ray_m,
        "track_limit_penalty_scale": args.reward_track_limit_penalty_scale,
        "smoothness_penalty": args.reward_smoothness_penalty,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _episode_metrics(*, steps: list[dict[str, Any]], policy: str, episode: int, seed: int) -> dict[str, Any]:
    if not steps:
        return {
            "policy": policy,
            "episode": episode,
            "seed": seed,
            "steps": 0,
            "completed_lap": False,
            "valid_lap": False,
            "segment_complete": False,
            "termination_reason": "empty",
            "total_reward": 0.0,
            "final_progress_m": 0.0,
            "best_progress_m": 0.0,
        }
    reward_totals: dict[str, float] = {}
    for step in steps:
        for key, value in step["reward_components"].items():
            reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
    final = steps[-1]
    best_progress_m = float(max(step["monotonic_progress_m"] for step in steps))
    start_progress_m = float(steps[0]["monotonic_progress_m"])
    segment_target_progress_m = final.get("segment_target_progress_m")
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "steps": len(steps),
        "completed_lap": bool(final["termination_reason"] == "lap_complete"),
        "valid_lap": bool(final.get("valid_lap", False)),
        "segment_complete": bool(final.get("segment_complete", False)),
        "termination_reason": final["termination_reason"],
        "elapsed_time_s": float(final["sim_time_s"]),
        "total_reward": float(sum(step["reward_total"] for step in steps)),
        "reward_totals": reward_totals,
        "final_progress_m": float(final["monotonic_progress_m"]),
        "best_progress_m": best_progress_m,
        "segment_progress_delta_m": max(0.0, best_progress_m - start_progress_m),
        "segment_target_progress_m": segment_target_progress_m,
        "segment_remaining_m": (
            max(0.0, float(segment_target_progress_m) - best_progress_m)
            if segment_target_progress_m is not None
            else None
        ),
        "checkpoints_passed": int(final.get("checkpoints_passed", final["checkpoint_index"])),
        "missed_checkpoint_count": int(final.get("missed_checkpoint_count", 0)),
        "avg_speed_kph": float(sum(step["speed_kph"] for step in steps) / len(steps)),
        "max_speed_kph": float(max(step["speed_kph"] for step in steps)),
        "crashed": bool(final["collided"]),
        "off_track": bool(final["off_track"]),
        "curriculum_stage": final.get("curriculum_stage"),
    }


def run_model_rollouts(
    model: Any,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    telemetry_dir: Path | None = None,
    telemetry_mode: str = "none",
    policy_name: str = "ppo",
    curriculum: CurriculumConfig | None = None,
    reward_overrides: dict[str, float | None] | None = None,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    curriculum_sampler: CurriculumSampler | None = None
    if curriculum is not None and curriculum.enabled:
        curriculum_sampler = CurriculumSampler(curriculum)
    for episode in range(episodes):
        sim = MonzaSim(build_sim_config(max_steps=max_steps, reward_overrides=reward_overrides))
        reset_options = curriculum_sampler.sample_options(seed + episode) if curriculum_sampler is not None else None
        obs, _ = sim.reset(seed=seed + episode, options=reset_options)
        rows: list[dict[str, Any]] = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            result = sim.step(_action_to_int(action))
            obs = result.observation
            row = asdict(result.telemetry)
            rows.append(row)
            if result.terminated or result.truncated:
                break
        metrics.append(_episode_metrics(steps=rows, policy=policy_name, episode=episode, seed=seed + episode))
        should_write = telemetry_mode == "all" or (telemetry_mode == "selected" and episode == 0)
        if should_write and telemetry_dir is not None:
            _write_jsonl(telemetry_dir / f"{policy_name}-episode-{episode:03d}-steps.jsonl", rows)
    return metrics


class TrainingEvalCallback:
    def __init__(
        self,
        *,
        run_root: Path,
        eval_every: int,
        eval_episodes: int,
        max_steps: int,
        seed: int,
        telemetry: str,
        telemetry_every: int,
        save_best: bool,
        curriculum: CurriculumConfig,
        reward_overrides: dict[str, float | None] | None = None,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is required for callbacks.") from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: TrainingEvalCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

        self.callback = _Callback(self)
        self.run_root = run_root
        self.eval_every = max(eval_every, 1)
        self.eval_episodes = max(eval_episodes, 1)
        self.max_steps = max_steps
        self.seed = seed
        self.telemetry = telemetry
        self.telemetry_every = max(telemetry_every, 1)
        self.save_best = save_best
        self.curriculum = curriculum
        self.reward_overrides = reward_overrides
        self.last_eval = 0
        self.best_score = float("-inf")
        self.eval_root = run_root / "eval"
        self.eval_root.mkdir(parents=True, exist_ok=True)
        self.eval_metrics_path = self.eval_root / "eval_metrics.jsonl"
        self.selected_telemetry_dir = self.eval_root / "selected_telemetry"

    def on_step(self, callback: Any) -> bool:
        if callback.num_timesteps - self.last_eval < self.eval_every:
            return True
        self.last_eval = int(callback.num_timesteps)
        self.evaluate(callback.model, timesteps=int(callback.num_timesteps), phase="train")
        return True

    def evaluate(self, model: Any, *, timesteps: int, phase: str) -> dict[str, Any]:
        telemetry_mode = "none"
        if self.telemetry == "all" or (self.telemetry == "selected" and timesteps % self.telemetry_every == 0):
            telemetry_mode = self.telemetry
        metrics = run_model_rollouts(
            model,
            episodes=self.eval_episodes,
            max_steps=self.max_steps,
            seed=self.seed + timesteps,
            telemetry_dir=self.selected_telemetry_dir,
            telemetry_mode=telemetry_mode,
            policy_name="ppo_full_lap",
            reward_overrides=self.reward_overrides,
        )
        mean_reward = sum(row["total_reward"] for row in metrics) / len(metrics)
        mean_progress = sum(row["best_progress_m"] for row in metrics) / len(metrics)
        completion_rate = sum(1 for row in metrics if row["completed_lap"]) / len(metrics)
        segment_metrics: list[dict[str, Any]] = []
        if self.curriculum.enabled:
            segment_metrics = run_model_rollouts(
                model,
                episodes=self.eval_episodes,
                max_steps=self.max_steps,
                seed=self.seed + timesteps + 50000,
                telemetry_dir=self.selected_telemetry_dir,
                telemetry_mode=telemetry_mode,
                policy_name="ppo_curriculum_segment",
                curriculum=self.curriculum,
                reward_overrides=self.reward_overrides,
            )
        segment_source = segment_metrics or metrics
        segment_rate = sum(1 for row in segment_source if row["segment_complete"]) / len(segment_source)
        mean_segment_progress = sum(row["best_progress_m"] for row in segment_source) / len(segment_source)
        mean_segment_delta = sum(row["segment_progress_delta_m"] for row in segment_source) / len(segment_source)
        summary = {
            "timesteps": int(timesteps),
            "phase": phase,
            "scratch_initial_policy": phase == "initial_scratch",
            "mean_reward": mean_reward,
            "mean_best_progress_m": mean_progress,
            "completion_rate": completion_rate,
            "segment_completion_rate": segment_rate,
            "mean_segment_best_progress_m": mean_segment_progress,
            "mean_segment_progress_delta_m": mean_segment_delta,
            "episodes": metrics,
            "full_lap_episodes": metrics,
            "curriculum_segment_episodes": segment_metrics,
        }
        with self.eval_metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary) + "\n")
        score = mean_progress + mean_segment_delta + completion_rate * 10000.0 + segment_rate * 1000.0
        if self.save_best and score > self.best_score:
            self.best_score = score
            model.save(self.run_root / "best_model.zip")
            (self.eval_root / "best_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def run_training(
    *,
    timesteps: int,
    seed: int,
    n_envs: int,
    max_steps: int,
    device: str,
    checkpoint_every: int,
    require_gpu: bool = False,
    eval_every: int = 0,
    eval_episodes: int = 1,
    telemetry: str = "none",
    telemetry_every: int = 1,
    run_name: str | None = None,
    curriculum: str = "none",
    curriculum_stage_count: int | None = None,
    curriculum_promotion_resets: int = 300,
    vec_env: str = "dummy",
    save_best: bool = True,
    benchmark_throughput: bool = False,
    resume_checkpoint: Path | None = None,
    reward_overrides: dict[str, float | None] | None = None,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    ent_coef: float = 0.02,
) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError("Training requires stable-baselines3. Run `uv sync --active --all-extras --all-packages`.") from exc

    resolved_device = torch_device(device)
    if require_gpu and resolved_device != "cuda":
        raise RuntimeError(f"GPU required but resolved device is {resolved_device!r}.")
    policy = compute_policy(device)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{run_name}-{timestamp}" if run_name else f"train-{timestamp}"
    run_root = ARTIFACTS_DIR / run_id
    checkpoint_dir = run_root / "checkpoints"
    log_dir = run_root / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curriculum_config = _build_curriculum_config(
        mode=curriculum,
        stage_count=curriculum_stage_count,
        promotion_resets=curriculum_promotion_resets,
        normal_start_probability=curriculum_normal_start_probability,
    )
    vec_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    env = make_vec_env(
        _make_env(max_steps, seed, curriculum_config, reward_overrides),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_cls,
    )
    env = VecMonitor(env)
    ppo_hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "ent_coef": ent_coef,
    }
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint}")
        model = PPO.load(
            resume_checkpoint,
            env=env,
            device=resolved_device,
            tensorboard_log=str(log_dir),
            print_system_info=False,
            **ppo_hyperparams,
        )
        scratch_initialization = False
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=resolved_device,
            tensorboard_log=str(log_dir),
            **ppo_hyperparams,
        )
        scratch_initialization = True
    initial_checkpoint_path = checkpoint_dir / "initial_model.zip"
    initial_root_path = run_root / "initial_model.zip"
    model.save(initial_checkpoint_path)
    model.save(initial_root_path)
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=max(checkpoint_every // max(n_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_monza",
        )
    ]
    eval_callback: TrainingEvalCallback | None = None
    if eval_every > 0:
        eval_callback = TrainingEvalCallback(
            run_root=run_root,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 1000,
            telemetry=telemetry,
            telemetry_every=telemetry_every,
            save_best=save_best,
            curriculum=curriculum_config,
            reward_overrides=reward_overrides,
        )
        callbacks.append(
            eval_callback.callback
        )
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "max_steps": max_steps,
        "device": resolved_device,
        "require_gpu": require_gpu,
        "vec_env": vec_env,
        "telemetry": telemetry,
        "telemetry_every": telemetry_every,
        "curriculum": curriculum_config.mode,
        "curriculum_stage_count": len(curriculum_config.stages) if curriculum_config.enabled else 0,
        "curriculum_promotion_resets": curriculum_config.promotion_resets,
        "curriculum_config": CurriculumSampler(curriculum_config).to_dict(),
        "compute_policy": policy,
        "benchmark_throughput": benchmark_throughput,
        "scratch_initialization": scratch_initialization,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "initial_checkpoint": str(initial_checkpoint_path),
        "ppo_hyperparams": ppo_hyperparams,
        "sim_config": dataclass_to_dict(build_sim_config(max_steps=max_steps, reward_overrides=reward_overrides)),
    }
    metadata_path = run_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if eval_callback is not None:
        initial_phase = "initial_scratch" if scratch_initialization else "initial_resume"
        eval_callback.evaluate(model, timesteps=0, phase=initial_phase)
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks), tb_log_name="ppo")
    elapsed = time.perf_counter() - started
    final_checkpoint_path = checkpoint_dir / "final_model.zip"
    final_root_path = run_root / "final_model.zip"
    model.save(final_checkpoint_path)
    model.save(final_root_path)
    env.close()
    metadata.update(
        {
            "wall_clock_s": elapsed,
            "training_fps": float(timesteps / max(elapsed, 1e-9)),
            "env_steps_per_second": float((timesteps * max(n_envs, 1)) / max(elapsed, 1e-9)),
            "final_checkpoint": str(final_checkpoint_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "train_complete "
        f"run={run_root} checkpoint={final_checkpoint_path} device={resolved_device} "
        f"vec_env={vec_env} fps={metadata['training_fps']:.1f}"
    )
    return final_checkpoint_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the simplified Monza simulator.")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.set_defaults(save_best=True)
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--telemetry", choices=["none", "selected", "all"], default="none")
    parser.add_argument("--telemetry-every", type=int, default=1)
    parser.add_argument("--run-name")
    parser.add_argument("--curriculum", choices=["none", "segments"], default="none")
    parser.add_argument("--curriculum-stage-count", type=int)
    parser.add_argument("--curriculum-promotion-resets", type=int, default=300)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--benchmark-throughput", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.02)
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
        require_gpu=args.require_gpu,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
        run_name=args.run_name,
        curriculum=args.curriculum,
        curriculum_stage_count=args.curriculum_stage_count,
        curriculum_promotion_resets=args.curriculum_promotion_resets,
        vec_env=args.vec_env,
        save_best=args.save_best,
        benchmark_throughput=args.benchmark_throughput,
        resume_checkpoint=args.resume_checkpoint,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
