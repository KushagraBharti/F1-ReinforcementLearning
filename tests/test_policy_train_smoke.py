import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from f1rl import policy_io, train
from f1rl.config import SimConfig
from f1rl.curriculum import CurriculumConfig, CurriculumStage
from f1rl.env import MonzaEnv
from f1rl.train import _full_lap_selection_score, _segment_eval_curriculum_config


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_smoke_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    checkpoint = train.run_training(
        timesteps=64,
        seed=5,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=32,
        eval_episodes=1,
        telemetry="selected",
        run_name="smoke",
        reward_overrides={"lateral_penalty_scale": 0.0, "track_limit_penalty_scale": 0.0},
    )
    root = checkpoint.parents[1]
    assert checkpoint.exists()
    assert (root / "initial_model.zip").exists()
    assert (root / "checkpoints" / "initial_model.zip").exists()
    assert (root / "final_model.zip").exists()
    assert (root / "best_model.zip").exists()
    assert (root / "eval" / "eval_metrics.jsonl").exists()
    assert any((root / "eval" / "selected_telemetry").glob("*.jsonl"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["device"] == "cpu"
    assert metadata["vec_env"] == "dummy"
    assert metadata["scratch_initialization"] is True
    assert metadata["initial_checkpoint"].endswith("initial_model.zip")
    assert metadata["sim_config"]["action_mode"] == "discrete"
    assert metadata["sim_config"]["action_set"] == "legacy"
    assert metadata["sim_config"]["observation_profile"] == "base"
    assert metadata["sim_config"]["reward"]["lateral_penalty_scale"] == 0.0
    assert metadata["sim_config"]["reward"]["track_limit_penalty_scale"] == 0.0
    assert metadata["normalize_reward"] is False
    assert metadata["training_fps"] > 0.0
    eval_rows = [
        json.loads(line)
        for line in (root / "eval" / "eval_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert eval_rows[0]["timesteps"] == 0
    assert eval_rows[0]["phase"] == "initial_scratch"
    assert eval_rows[0]["scratch_initial_policy"] is True
    assert eval_rows[0]["selection_priority"] == "valid full-lap completion, then normal-start best progress"


def test_full_lap_selection_score_keeps_segment_success_as_tiebreaker() -> None:
    stronger_full_lap = {
        "completion_rate": 0.0,
        "mean_best_progress_m": 1000.0,
        "segment_completion_rate": 0.0,
        "mean_segment_progress_delta_m": 0.0,
        "full_lap_episodes": [],
    }
    weaker_full_lap_with_segment_win = {
        "completion_rate": 0.0,
        "mean_best_progress_m": 900.0,
        "segment_completion_rate": 1.0,
        "mean_segment_progress_delta_m": 500.0,
        "full_lap_episodes": [],
    }
    assert _full_lap_selection_score(stronger_full_lap) > _full_lap_selection_score(
        weaker_full_lap_with_segment_win
    )


def test_segment_eval_curriculum_removes_normal_start_mix() -> None:
    stage = CurriculumStage("unit", 25.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    config = CurriculumConfig(
        mode="segments",
        stages=(stage,),
        promotion_resets=10,
        normal_start_probability=0.75,
        focus_start_progress_m=500.0,
        focus_window_m=100.0,
        start_mode="normal",
    )
    segment_eval = _segment_eval_curriculum_config(config)
    assert segment_eval.normal_start_probability == 0.0
    assert segment_eval.stages == config.stages
    assert segment_eval.focus_start_progress_m == 500.0
    assert segment_eval.start_mode == "normal"


def test_latest_checkpoint_finds_named_runs(tmp_path: Path) -> None:
    older = tmp_path / "named-run-a" / "checkpoints" / "older.zip"
    newer = tmp_path / "named-run-b" / "checkpoints" / "newer.zip"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    os.utime(older, (1.0, 1.0))
    os.utime(newer, (2.0, 2.0))
    assert policy_io.latest_checkpoint(tmp_path) == newer


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_smoke_can_save_reward_normalization_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    checkpoint = train.run_training(
        timesteps=64,
        seed=8,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="norm-smoke",
        normalize_reward=True,
    )
    root = checkpoint.parents[1]
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["normalize_reward"] is True
    assert metadata["final_vecnormalize"] is not None
    assert (root / "vecnormalize.pkl").exists()


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_vecnormalize_reward_fallback_allows_observation_expansion(tmp_path: Path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    old_env = DummyVecEnv([lambda: MonzaEnv(SimConfig(max_steps=20, observation_profile="racing"))])
    old_stats = VecNormalize(old_env, norm_obs=False, norm_reward=True)
    old_stats.ret_rms.mean = 3.0
    old_stats.ret_rms.var = 4.0
    old_stats.ret_rms.count = 5.0
    stats_path = tmp_path / "vecnormalize.pkl"
    old_stats.save(str(stats_path))
    old_stats.close()

    new_env = DummyVecEnv([lambda: MonzaEnv(SimConfig(max_steps=20, observation_profile="racing_v2"))])
    loaded, mode = train._load_vecnormalize_with_reward_fallback(
        VecNormalize,
        stats_path,
        new_env,
        normalize_reward=True,
        normalize_reward_gamma=0.997,
        normalize_reward_clip=7.0,
    )
    try:
        assert mode.startswith("reward_stats_only_observation_shape_changed")
        assert loaded.observation_space.shape == new_env.observation_space.shape
        assert loaded.norm_obs is False
        assert loaded.norm_reward is True
        assert loaded.gamma == pytest.approx(0.997)
        assert loaded.clip_reward == pytest.approx(7.0)
        assert loaded.ret_rms.mean == pytest.approx(3.0)
        assert loaded.ret_rms.var == pytest.approx(4.0)
        assert loaded.ret_rms.count == pytest.approx(5.0)
    finally:
        loaded.close()


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_resume_checkpoint_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    base_checkpoint = train.run_training(
        timesteps=64,
        seed=6,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="base",
    )
    resumed_checkpoint = train.run_training(
        timesteps=64,
        seed=7,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="resumed",
        resume_checkpoint=base_checkpoint,
    )
    root = resumed_checkpoint.parents[1]
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["scratch_initialization"] is False
    assert metadata["resume_checkpoint"] == str(base_checkpoint)


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="stable-baselines3 not installed")
def test_ppo_transfer_initialization_can_expand_observation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)
    base_checkpoint = train.run_training(
        timesteps=64,
        seed=9,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="base-transfer",
        observation_profile="base",
    )
    transfer_checkpoint = train.run_training(
        timesteps=64,
        seed=10,
        n_envs=1,
        max_steps=50,
        device="cpu",
        checkpoint_every=64,
        eval_every=0,
        run_name="racing-transfer",
        observation_profile="racing",
        initialize_from_checkpoint=base_checkpoint,
    )
    root = transfer_checkpoint.parents[1]
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["scratch_initialization"] is False
    assert metadata["transfer_initialization"] is True
    assert metadata["resume_checkpoint"] is None
    assert metadata["initialize_from_checkpoint"] == str(base_checkpoint)
    assert metadata["sim_config"]["observation_profile"] == "racing"
    assert any(row["mode"] == "expanded_input" for row in metadata["transfer_weight_report"])


class _FakePolicy:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._state.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._state = {key: value.clone() for key, value in state.items()}


class _FakeActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n


class _FakeModel:
    def __init__(self, state: dict[str, torch.Tensor], action_count: int) -> None:
        self.policy = _FakePolicy(state)
        self.action_space = _FakeActionSpace(action_count)


def test_transfer_initialization_can_expand_discrete_action_head() -> None:
    source_weight = torch.stack([torch.full((3,), float(index)) for index in range(9)])
    source_bias = torch.arange(9, dtype=torch.float32)
    target = _FakeModel(
        {
            "action_net.weight": torch.zeros((20, 3), dtype=torch.float32),
            "action_net.bias": torch.zeros(20, dtype=torch.float32),
        },
        action_count=20,
    )
    source = _FakeModel(
        {
            "action_net.weight": source_weight,
            "action_net.bias": source_bias,
        },
        action_count=9,
    )

    report = train._copy_compatible_policy_weights(target, source, target_action_set="racing")
    target_state = target.policy.state_dict()

    assert torch.equal(target_state["action_net.weight"][2], source_weight[1])
    assert target_state["action_net.bias"][2].item() == pytest.approx(source_bias[1].item())
    assert torch.equal(target_state["action_net.weight"][15], source_weight[7])
    assert target_state["action_net.bias"][15].item() == pytest.approx(source_bias[7].item())
    assert target_state["action_net.bias"][12].item() < source_bias[2].item()
    assert [row["mode"] for row in report].count("expanded_discrete_action_head") == 2
    bias_report = next(row for row in report if row["key"] == "action_net.bias")
    soft_brake_row = next(row for row in bias_report["rows"] if row["target_action"] == "soft_brake")
    assert soft_brake_row["mode"] == "nearest_with_bias_penalty"
