"""Checkpoint evaluation and telemetry export."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from f1rl.config import (
    ACTION_MODES,
    ACTION_SETS,
    ARTIFACTS_DIR,
    CONTINUOUS_ACTION_SCHEMES,
    OBSERVATION_PROFILES,
    build_sim_config,
    disable_scaffold_rewards,
    disable_training_assists,
)
from f1rl.env import MonzaEnv
from f1rl.policy_io import (
    close_vecnormalize,
    load_sb3_ppo,
    load_vecnormalize_stats,
    normalize_observation,
    resolve_ppo_eval_config,
    validate_model_spaces,
)
from f1rl.telemetry import TelemetryWriter


def run_eval(
    *,
    checkpoint: str,
    steps: int,
    seed: int,
    device: str,
    action_mode: str = "discrete",
    action_set: str = "legacy",
    continuous_action_scheme: str = "drive_brake",
    observation_profile: str = "base",
    launch_guard_progress_m: float = 0.0,
    launch_guard_min_speed_kph: float = 0.0,
    launch_guard_throttle: float = 0.22,
    metadata_mode: str = "auto",
    disable_scaffold: bool = False,
    disable_assists: bool = False,
    ppo_deterministic: bool = True,
) -> int:
    fallback_config = build_sim_config(
        max_steps=steps,
        action_mode=action_mode,
        action_set=action_set,
        continuous_action_scheme=continuous_action_scheme,
        observation_profile=observation_profile,
        launch_guard_progress_m=launch_guard_progress_m,
        launch_guard_min_speed_kph=launch_guard_min_speed_kph,
        launch_guard_throttle=launch_guard_throttle,
    )
    ppo_config = resolve_ppo_eval_config(
        checkpoint,
        max_steps=steps,
        fallback_config=fallback_config,
        metadata_mode=metadata_mode,
    )
    if disable_scaffold:
        disable_scaffold_rewards(ppo_config.sim_config)
    if disable_assists:
        disable_training_assists(ppo_config.sim_config)
    env = MonzaEnv(ppo_config.sim_config)
    vec_normalize = load_vecnormalize_stats(ppo_config)
    model = load_sb3_ppo(ppo_config.checkpoint_path, env=None, device=device)
    validate_model_spaces(model, observation_space=env.observation_space, action_space=env.action_space)
    obs, _ = env.reset(seed=seed)
    obs = normalize_observation(obs, vec_normalize)
    writer = TelemetryWriter(ARTIFACTS_DIR, mode="eval", seed=seed, lap_length_m=env.sim.track.length_m)
    try:
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=ppo_deterministic)
            if ppo_config.sim_config.action_mode == "continuous":
                resolved_action = np.asarray(action, dtype=np.float32)
            elif ppo_config.sim_config.action_mode == "multidiscrete":
                resolved_action = np.asarray(action, dtype=np.int64)
            else:
                resolved_action = int(action)
            obs, _, terminated, truncated, _ = env.step(resolved_action)
            obs = normalize_observation(obs, vec_normalize)
            if env.last_telemetry is not None:
                writer.write_step(env.last_telemetry)
            if terminated or truncated:
                break
    finally:
        env.close()
        close_vecnormalize(vec_normalize)
    summary = writer.close_episode(
        termination_reason=env.sim.termination_reason,
        completed_lap=env.sim.completed_lap,
    )
    print(
        "eval_complete "
        f"run={writer.root} reason={summary.termination_reason} "
        f"checkpoint={ppo_config.checkpoint_path} config_source={ppo_config.config_source} "
        f"vecnormalize={ppo_config.vecnormalize_path} deterministic={ppo_deterministic}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint.")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--action-mode", choices=sorted(ACTION_MODES), default="discrete")
    parser.add_argument("--action-set", choices=sorted(ACTION_SETS), default="legacy")
    parser.add_argument("--continuous-action-scheme", choices=sorted(CONTINUOUS_ACTION_SCHEMES), default="drive_brake")
    parser.add_argument("--observation-profile", choices=sorted(OBSERVATION_PROFILES), default="base")
    parser.add_argument("--launch-guard-progress-m", type=float, default=0.0)
    parser.add_argument("--launch-guard-min-speed-kph", type=float, default=0.0)
    parser.add_argument("--launch-guard-throttle", type=float, default=0.22)
    parser.add_argument("--metadata-mode", choices=["auto", "require", "ignore"], default="auto")
    parser.add_argument("--disable-scaffold-rewards", action="store_true")
    parser.add_argument("--disable-training-assists", action="store_true")
    parser.set_defaults(ppo_deterministic=True)
    parser.add_argument("--ppo-deterministic", dest="ppo_deterministic", action="store_true")
    parser.add_argument("--ppo-stochastic", dest="ppo_deterministic", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_eval(
        checkpoint=args.checkpoint,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        action_mode=args.action_mode,
        action_set=args.action_set,
        continuous_action_scheme=args.continuous_action_scheme,
        observation_profile=args.observation_profile,
        launch_guard_progress_m=args.launch_guard_progress_m,
        launch_guard_min_speed_kph=args.launch_guard_min_speed_kph,
        launch_guard_throttle=args.launch_guard_throttle,
        metadata_mode=args.metadata_mode,
        disable_scaffold=args.disable_scaffold_rewards,
        disable_assists=args.disable_training_assists,
        ppo_deterministic=args.ppo_deterministic,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
