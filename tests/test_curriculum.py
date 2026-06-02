from f1rl.config import SimConfig
from f1rl.curriculum import CurriculumConfig, CurriculumSampler, CurriculumStage
from f1rl.env import MonzaEnv
from f1rl.sim import MonzaSim


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
        assert info["segment_complete"] is False
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
