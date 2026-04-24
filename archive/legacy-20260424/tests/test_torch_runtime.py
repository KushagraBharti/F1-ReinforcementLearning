import pytest
import torch

from f1rl.config import EnvConfig, to_dict
from f1rl.torch_runtime import TelemetryThresholds, TorchSimBatch


def _base_config() -> dict:
    config = EnvConfig()
    config.max_steps = 64
    config.render.enabled = False
    config.render_mode = None
    config.headless = True
    return to_dict(config)


def test_torch_runtime_step_emits_expected_telemetry_fields() -> None:
    simulator = TorchSimBatch(_base_config(), num_cars=2, device="cpu")
    observations = simulator.reset(seed=3)
    assert observations.shape == (2, simulator.observation_dim)

    step = simulator.step(torch.tensor([[0.1, 1.0], [-0.2, 0.4]], dtype=torch.float32))
    expected = {
        "x",
        "y",
        "heading_deg",
        "speed",
        "speed_kph",
        "distance_travelled",
        "lap_count",
        "next_goal_idx",
        "current_checkpoint_index",
        "progress_delta",
        "positive_progress_total",
        "time_since_last_checkpoint",
        "throttle",
        "brake",
        "steering",
        "min_wall_distance_ratio",
        "heading_error_norm",
        "goal_distance_norm",
        "lateral_offset_norm",
        "no_progress_steps",
        "reverse_speed_steps",
        "reverse_progress_events",
        "heading_away_steps",
        "wall_hugging_steps",
        "alive",
        "moving",
        "terminated",
        "truncated",
        "termination_code",
    }
    assert expected.issubset(step.telemetry.keys())
    assert step.observations.shape == (2, simulator.observation_dim)
    assert step.rewards.shape == (2,)


def test_moving_thresholds_are_explicit_and_persisted() -> None:
    thresholds = TelemetryThresholds(moving_speed_threshold_kph=0.5, moving_progress_threshold=0.01)
    simulator = TorchSimBatch(_base_config(), num_cars=1, device="cpu", telemetry_thresholds=thresholds)
    simulator.reset(seed=5)
    moving_seen = False
    for _ in range(8):
        step = simulator.step(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
        moving_seen = moving_seen or bool(step.telemetry["moving"][0].item())
    assert moving_seen is True

    runtime = simulator.runtime_metadata(policy_device="cpu", renderer_backend="headless")
    assert runtime["telemetry_thresholds"]["moving_speed_threshold_kph"] == 0.5
    assert runtime["telemetry_thresholds"]["moving_progress_threshold"] == 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity_for_fixed_actions() -> None:
    config = _base_config()
    cpu_sim = TorchSimBatch(config, num_cars=4, device="cpu")
    gpu_sim = TorchSimBatch(config, num_cars=4, device="gpu")
    cpu_obs = cpu_sim.reset(seed=7)
    gpu_obs = gpu_sim.reset(seed=7)
    assert torch.allclose(cpu_obs.cpu(), gpu_obs.cpu(), atol=1e-3, rtol=1e-3)

    actions = [
        torch.tensor([[0.2, 0.8], [-0.3, 0.2], [0.0, -0.5], [0.4, 0.1]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0], [0.1, 0.6], [-0.1, 0.0], [0.3, -0.4]], dtype=torch.float32),
        torch.tensor([[-0.2, 0.5], [0.0, 0.0], [0.2, 0.8], [-0.5, -0.2]], dtype=torch.float32),
    ]
    for batch in actions:
        cpu_step = cpu_sim.step(batch)
        gpu_step = gpu_sim.step(batch.to(gpu_sim.device))
        assert torch.allclose(cpu_step.observations.cpu(), gpu_step.observations.cpu(), atol=1e-3, rtol=1e-3)
        assert torch.allclose(cpu_step.rewards.cpu(), gpu_step.rewards.cpu(), atol=1e-3, rtol=1e-3)
        assert torch.equal(cpu_step.telemetry["termination_code"].cpu(), gpu_step.telemetry["termination_code"].cpu())
        assert torch.equal(cpu_step.telemetry["current_checkpoint_index"].cpu(), gpu_step.telemetry["current_checkpoint_index"].cpu())
