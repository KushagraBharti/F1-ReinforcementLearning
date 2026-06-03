from f1rl.config import SimConfig
from f1rl.curriculum import CurriculumConfig, CurriculumSampler, CurriculumStage
from f1rl.env import MonzaEnv
from f1rl.sim import MonzaSim
from f1rl.state_library import write_state_library
from f1rl.state_snapshot import snapshot_from_mapping, snapshot_from_sim
from f1rl.train import _build_curriculum_config, _curriculum_stage_metrics


def _snapshot_at(progress_m: float):
    return snapshot_from_mapping(
        {
            "id": f"snapshot-{progress_m}",
            "source": "unit",
            "step_index": int(progress_m),
            "sim_time_s": progress_m / 100.0,
            "x": 100.0 + progress_m,
            "y": 200.0,
            "heading_rad": 0.0,
            "speed_mps": 40.0,
            "yaw_rate_rps": 0.0,
            "steering_rad": 0.0,
            "raw_progress_m": progress_m,
            "monotonic_progress_m": progress_m,
            "checkpoint_index": 1,
            "next_checkpoint_index": 2,
            "checkpoints_passed": 1,
            "missed_checkpoint_count": 0,
            "lap_index": 0,
            "valid_lap": True,
            "finish_crossed": False,
            "segment_complete": False,
            "last_throttle": 0.5,
            "last_brake": 0.0,
            "last_steer": 0.0,
            "last_action_id": 1,
        }
    )


def test_curriculum_sampler_promotes_by_reset_count() -> None:
    stages = (
        CurriculumStage("easy", 100.0, 10.0, 20.0, 0.0, 0.0, 0.0),
        CurriculumStage("hard", 200.0, 30.0, 40.0, 0.0, 0.0, 0.0),
    )
    sampler = CurriculumSampler(
        CurriculumConfig(mode="segments", stages=stages, promotion_resets=1),
        checkpoint_count=12,
    )
    first = sampler.sample_options(seed=1)
    second = sampler.sample_options(seed=1)
    assert first["curriculum_stage"] == "easy"
    assert second["curriculum_stage"] == "hard"
    assert 0 <= first["start_checkpoint"] < 12
    assert first["segment_length_m"] == 100.0


def test_training_curriculum_config_can_limit_stages() -> None:
    config = _build_curriculum_config(
        mode="segments",
        stage_count=1,
        stage_start_index=0,
        promotion_resets=100_000,
        normal_start_probability=0.0,
    )
    sampler = CurriculumSampler(config, checkpoint_count=12)
    first = sampler.sample_options(seed=1)
    later = None
    for index in range(20):
        later = sampler.sample_options(seed=index)
    assert len(config.stages) == 1
    assert config.promotion_resets == 100_000
    assert first["curriculum_stage"] == "A0-short-control"
    assert later is not None
    assert later["curriculum_stage"] == "A0-short-control"


def test_training_curriculum_config_can_start_from_later_stage() -> None:
    config = _build_curriculum_config(
        mode="segments",
        stage_count=2,
        stage_start_index=3,
        promotion_resets=1,
        normal_start_probability=0.0,
    )
    assert [stage.name for stage in config.stages] == ["C-long", "D-random-checkpoint"]


def test_training_curriculum_config_can_build_prefix_start_stages() -> None:
    config = _build_curriculum_config(
        mode="segments",
        preset="prefix-start",
        stage_count=2,
        stage_start_index=0,
        promotion_resets=1,
        normal_start_probability=0.75,
    )
    sampler = CurriculumSampler(config, checkpoint_count=12)
    first = sampler.sample_options(seed=1)
    second = sampler.sample_options(seed=2)
    assert config.start_mode == "normal"
    assert [stage.name for stage in config.stages] == ["P0-launch-240m", "P1-start-500m"]
    assert first["curriculum_stage"] == "P0-launch-240m"
    assert second["curriculum_stage"] == "P1-start-500m"
    assert "start_checkpoint" not in first
    assert "start_progress_m" not in first
    assert first["segment_length_m"] == 240.0


def test_curriculum_sampler_can_mix_normal_starts() -> None:
    stage = CurriculumStage("unit", 25.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    sampler = CurriculumSampler(
        CurriculumConfig(mode="segments", stages=(stage,), promotion_resets=100, normal_start_probability=1.0),
        checkpoint_count=12,
    )
    options = sampler.sample_options(seed=1)
    assert options == {"curriculum_stage": "normal-start-mix"}


def test_curriculum_sampler_can_focus_progress_window() -> None:
    stage = CurriculumStage("focus", 300.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    sampler = CurriculumSampler(
        CurriculumConfig(
            mode="segments",
            stages=(stage,),
            promotion_resets=100,
            focus_start_progress_m=750.0,
            focus_window_m=100.0,
        ),
        checkpoint_count=12,
    )
    options = sampler.sample_options(seed=2)
    assert "start_checkpoint" not in options
    assert 700.0 <= options["start_progress_m"] <= 800.0
    assert options["start_speed_kph"] == 100.0
    assert options["segment_length_m"] == 300.0
    assert options["curriculum_stage"] == "focus"


def test_training_curriculum_config_can_build_focus_stage() -> None:
    config = _build_curriculum_config(
        mode="segments",
        stage_count=None,
        stage_start_index=0,
        promotion_resets=10,
        normal_start_probability=0.25,
        focus_start_progress_m=720.0,
        focus_window_m=80.0,
        focus_segment_length_m=900.0,
        focus_min_speed_kph=240.0,
        focus_max_speed_kph=320.0,
    )
    assert len(config.stages) == 1
    assert config.stages[0].name == "focus-window"
    assert config.stages[0].segment_length_m == 900.0
    assert config.stages[0].min_speed_kph == 240.0
    assert config.stages[0].max_speed_kph == 320.0
    assert config.focus_start_progress_m == 720.0
    assert config.focus_window_m == 80.0
    assert config.normal_start_probability == 0.25


def test_training_curriculum_config_can_build_state_library_stage(tmp_path) -> None:
    sim = MonzaSim(SimConfig(max_steps=20))
    sim.reset(seed=1)
    snapshot = snapshot_from_sim(sim, source="unit")
    library_path = write_state_library(tmp_path / "state_library.json", [snapshot], source="unit")
    config = _build_curriculum_config(
        mode="segments",
        stage_count=None,
        stage_start_index=0,
        promotion_resets=10,
        normal_start_probability=0.0,
        state_library_path=library_path,
        state_library_segment_length_m=321.0,
    )
    sampler = CurriculumSampler(config, checkpoint_count=12)
    options = sampler.sample_options(seed=5)
    assert config.start_mode == "state_library"
    assert config.state_library_path == library_path
    assert options["curriculum_stage"] == "state-library"
    assert options["segment_length_m"] == 321.0
    assert "state_snapshot" in options
    assert "start_speed_kph" not in options


def test_chicane_skill_curriculum_samples_progress_range_without_library() -> None:
    config = _build_curriculum_config(
        mode="segments",
        preset="chicane-skill",
        stage_count=1,
        stage_start_index=0,
        promotion_resets=10,
        normal_start_probability=0.0,
        chicane="rettifilo",
    )
    sampler = CurriculumSampler(config, checkpoint_count=12)
    options = sampler.sample_options(seed=5)
    assert config.start_mode == "progress_range"
    assert options["curriculum_stage"] == "rettifilo-approach-brake"
    assert 450.0 <= options["start_progress_m"] <= 560.0
    assert options["segment_length_m"] == 720.0 - options["start_progress_m"]
    assert options["segment_target_max_speed_kph"] == 190.0


def test_chicane_skill_curriculum_filters_state_library_by_stage(tmp_path) -> None:
    library_path = write_state_library(
        tmp_path / "state_library.json",
        [_snapshot_at(500.0), _snapshot_at(2000.0)],
        source="unit",
    )
    config = _build_curriculum_config(
        mode="segments",
        preset="chicane-skill",
        stage_count=1,
        stage_start_index=0,
        promotion_resets=10,
        normal_start_probability=0.0,
        state_library_path=library_path,
        chicane="rettifilo",
    )
    sampler = CurriculumSampler(config, checkpoint_count=12)
    options = sampler.sample_options(seed=5)
    assert options["curriculum_stage"] == "rettifilo-approach-brake"
    assert options["state_snapshot"]["monotonic_progress_m"] == 500.0
    assert options["segment_length_m"] == 220.0
    assert options["segment_target_max_speed_kph"] == 190.0


def test_curriculum_stage_metrics_report_actual_section_success() -> None:
    metrics = [
        {
            "curriculum_stage": "rettifilo-approach-brake",
            "segment_complete": True,
            "segment_progress_delta_m": 220.0,
            "best_progress_m": 720.0,
            "termination_reason": "segment_complete",
        },
        {
            "curriculum_stage": "rettifilo-approach-brake",
            "segment_complete": False,
            "segment_progress_delta_m": 80.0,
            "best_progress_m": 580.0,
            "termination_reason": "off_track",
        },
    ]

    summary = _curriculum_stage_metrics(metrics)

    assert summary["rettifilo-approach-brake"]["episodes"] == 2
    assert summary["rettifilo-approach-brake"]["segment_completion_rate"] == 0.5
    assert summary["rettifilo-approach-brake"]["mean_segment_progress_delta_m"] == 150.0
    assert summary["rettifilo-approach-brake"]["termination_reasons"] == {
        "off_track": 1,
        "segment_complete": 1,
    }


def test_env_curriculum_reset_sets_segment_metadata() -> None:
    stage = CurriculumStage("unit", 25.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    env = MonzaEnv(
        SimConfig(max_steps=20),
        curriculum=CurriculumConfig(mode="segments", stages=(stage,), promotion_resets=100),
    )
    try:
        _, info = env.reset(seed=10)
        assert info["curriculum_stage"] == "unit"
        assert info["segment_target_progress_m"] is not None
        assert info["segment_target_max_speed_kph"] is None
        assert info["segment_complete"] is False
    finally:
        env.close()


def test_env_curriculum_can_reset_from_state_library(tmp_path) -> None:
    sim = MonzaSim(SimConfig(max_steps=20))
    sim.reset(seed=1)
    for _ in range(5):
        sim.step(1)
    snapshot = snapshot_from_sim(sim, source="unit")
    library_path = write_state_library(tmp_path / "state_library.json", [snapshot], source="unit")
    stage = CurriculumStage("state-library", 25.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    env = MonzaEnv(
        SimConfig(max_steps=20),
        curriculum=CurriculumConfig(
            mode="segments",
            stages=(stage,),
            promotion_resets=100,
            start_mode="state_library",
            state_library_path=library_path,
        ),
    )
    try:
        _, info = env.reset(seed=10)
        assert info["curriculum_stage"] == "state-library"
        assert info["segment_target_progress_m"] == snapshot.monotonic_progress_m + 25.0
        assert info["raw_progress_m"] == snapshot.raw_progress_m
    finally:
        env.close()


def test_sim_segment_completion_terminates_segment() -> None:
    sim = MonzaSim(SimConfig(max_steps=80))
    sim.reset(
        seed=2,
        options={
            "start_checkpoint": 5,
            "start_speed_kph": 80.0,
            "segment_length_m": 2.0,
            "curriculum_stage": "unit",
        },
    )
    result = None
    for _ in range(80):
        result = sim.step(1)
        if result.terminated or result.truncated:
            break
    assert result is not None
    assert result.telemetry.curriculum_stage == "unit"
    assert result.telemetry.segment_complete is True
    assert result.telemetry.termination_reason == "segment_complete"


def test_sim_segment_completion_respects_target_max_speed() -> None:
    blocked = MonzaSim(SimConfig(max_steps=20))
    blocked.reset(
        seed=2,
        options={
            "start_progress_m": 100.0,
            "start_speed_kph": 220.0,
            "segment_length_m": 1.0,
            "segment_target_max_speed_kph": 80.0,
            "curriculum_stage": "speed-gated",
        },
    )
    blocked_result = None
    for _ in range(5):
        blocked_result = blocked.step_controls(throttle=0.0, brake=0.0, steer=0.0)
        if blocked_result.truncated:
            break
    assert blocked_result is not None
    assert blocked.state.monotonic_progress_m >= blocked.segment_target_progress_m
    assert blocked.state.speed_mps * 3.6 > 80.0
    assert blocked_result.telemetry.segment_complete is False
    assert blocked_result.telemetry.termination_reason != "segment_complete"

    allowed = MonzaSim(SimConfig(max_steps=20))
    allowed.reset(
        seed=2,
        options={
            "start_progress_m": 100.0,
            "start_speed_kph": 220.0,
            "segment_length_m": 1.0,
            "segment_target_max_speed_kph": 260.0,
            "curriculum_stage": "speed-gated",
        },
    )
    allowed_result = None
    for _ in range(5):
        allowed_result = allowed.step_controls(throttle=0.0, brake=0.0, steer=0.0)
        if allowed_result.truncated:
            break
    assert allowed_result is not None
    assert allowed_result.telemetry.segment_complete is True
    assert allowed_result.telemetry.termination_reason == "segment_complete"
