import json
from pathlib import Path

import pytest
from gymnasium import spaces

from f1rl.config import AssistConfig, RewardConfig, SimConfig, dataclass_to_dict
from f1rl.policy_io import resolve_ppo_eval_config, validate_model_spaces


class _FakeModel:
    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space


def test_resolve_ppo_eval_config_loads_run_metadata_from_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "metadata-run"
    checkpoint = run_root / "checkpoints" / "ppo_monza_100_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    (run_root / "vecnormalize.pkl").write_bytes(b"fake-stats")
    sim_config = SimConfig(
        max_steps=123,
        action_mode="continuous",
        action_set="expanded",
        continuous_action_scheme="throttle_bias",
        observation_profile="brake",
        launch_guard_progress_m=42.0,
        launch_guard_min_speed_kph=11.0,
        launch_guard_throttle=0.33,
        reward=RewardConfig(lateral_penalty_scale=0.123),
        assist=AssistConfig(enabled=True, overspeed_turn_in_terminate=True),
    )
    (run_root / "run_metadata.json").write_text(
        json.dumps({"run_id": run_root.name, "sim_config": dataclass_to_dict(sim_config)}),
        encoding="utf-8",
    )

    resolved = resolve_ppo_eval_config(checkpoint, max_steps=777, metadata_mode="require", root=tmp_path)

    assert resolved.metadata_loaded is True
    assert resolved.config_source == "run_metadata"
    assert resolved.checkpoint_path == checkpoint
    assert resolved.metadata_path == run_root / "run_metadata.json"
    assert resolved.vecnormalize_path == run_root / "vecnormalize.pkl"
    assert resolved.sim_config.max_steps == 777
    assert resolved.sim_config.action_mode == "continuous"
    assert resolved.sim_config.action_set == "expanded"
    assert resolved.sim_config.continuous_action_scheme == "throttle_bias"
    assert resolved.sim_config.observation_profile == "brake"
    assert resolved.sim_config.launch_guard_progress_m == 42.0
    assert resolved.sim_config.reward.lateral_penalty_scale == 0.123
    assert resolved.sim_config.assist.enabled is True
    assert resolved.sim_config.assist.overspeed_turn_in_terminate is True


def test_resolve_ppo_eval_config_preserves_v2_physics_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "v2-metadata-run"
    checkpoint = run_root / "checkpoints" / "ppo_monza_100_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    sim_config = SimConfig(
        max_steps=123,
        action_mode="continuous",
        observation_profile="base",
        physics_model="v2",
    )
    (run_root / "run_metadata.json").write_text(
        json.dumps({"run_id": run_root.name, "sim_config": dataclass_to_dict(sim_config)}),
        encoding="utf-8",
    )

    resolved = resolve_ppo_eval_config(checkpoint, max_steps=777, metadata_mode="require", root=tmp_path)

    assert resolved.sim_config.max_steps == 777
    assert resolved.sim_config.physics_model == "v2"
    assert resolved.sim_config.physics_version == sim_config.physics_version
    assert resolved.sim_config.physics_calibration_id == sim_config.physics_calibration_id
    assert resolved.sim_config.physics_calibration_id == resolved.sim_config.physics_v2.calibration_id


def test_resolve_ppo_eval_config_accepts_artifact_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "artifact-run"
    best_model = run_root / "best_model.zip"
    run_root.mkdir()
    best_model.write_bytes(b"fake")
    (run_root / "run_metadata.json").write_text(
        json.dumps({"run_id": run_root.name, "sim_config": dataclass_to_dict(SimConfig(action_set="racing"))}),
        encoding="utf-8",
    )

    resolved = resolve_ppo_eval_config(run_root, max_steps=10, metadata_mode="auto", root=tmp_path)

    assert resolved.checkpoint_path == best_model
    assert resolved.sim_config.action_set == "racing"


def test_resolve_ppo_eval_config_can_ignore_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "ignore-run"
    checkpoint = run_root / "best_model.zip"
    run_root.mkdir()
    checkpoint.write_bytes(b"fake")
    (run_root / "run_metadata.json").write_text(
        json.dumps({"run_id": run_root.name, "sim_config": dataclass_to_dict(SimConfig(action_mode="continuous"))}),
        encoding="utf-8",
    )
    fallback = SimConfig(action_mode="discrete", action_set="legacy")

    resolved = resolve_ppo_eval_config(checkpoint, max_steps=9, fallback_config=fallback, metadata_mode="ignore")

    assert resolved.metadata_loaded is False
    assert resolved.config_source == "explicit_args"
    assert resolved.sim_config.action_mode == "discrete"
    assert resolved.sim_config.action_set == "legacy"


def test_validate_model_spaces_raises_on_observation_shape_mismatch() -> None:
    model = _FakeModel(spaces.Box(-1.0, 1.0, shape=(18,)), spaces.Discrete(9))

    with pytest.raises(ValueError, match="observation shape"):
        validate_model_spaces(
            model,
            observation_space=spaces.Box(-1.0, 1.0, shape=(21,)),
            action_space=spaces.Discrete(9),
        )


def test_validate_model_spaces_raises_on_action_space_mismatch() -> None:
    model = _FakeModel(spaces.Box(-1.0, 1.0, shape=(18,)), spaces.Discrete(9))

    with pytest.raises(ValueError, match="action count"):
        validate_model_spaces(
            model,
            observation_space=spaces.Box(-1.0, 1.0, shape=(18,)),
            action_space=spaces.Discrete(21),
        )


def test_validate_model_spaces_accepts_matching_multidiscrete() -> None:
    model = _FakeModel(spaces.Box(-1.0, 1.0, shape=(23,)), spaces.MultiDiscrete([5, 5]))

    validate_model_spaces(
        model,
        observation_space=spaces.Box(-1.0, 1.0, shape=(23,)),
        action_space=spaces.MultiDiscrete([5, 5]),
    )
