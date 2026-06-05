import numpy as np
import torch

from f1rl.config import SimConfig
from f1rl.gpu_observation import observation_batch
from f1rl.gpu_track import ray_distances_batch, sensor_angles_tensor
from f1rl.gpu_types import car_batch_from_snapshots, gpu_track_from_cpu
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import snapshot_from_sim
from f1rl.track_model import load_track_spec


def _batch_from_sim(sim: MonzaSim) -> tuple[torch.Tensor, torch.Tensor]:
    snapshot = snapshot_from_sim(sim, source="unit")
    track = gpu_track_from_cpu(load_track_spec(sim.config.track_path), device=torch.device("cpu"), dtype=torch.float64)
    batch = car_batch_from_snapshots(
        [snapshot],
        meters_per_pixel=sim.track.meters_per_pixel,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    angles = sensor_angles_tensor(
        count=sim.config.sensors.count,
        spread_deg=sim.config.sensors.spread_deg,
        forward_bias=sim.config.sensors.forward_bias,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return (
        ray_distances_batch(
            batch.x,
            batch.y,
            batch.heading_rad,
            track,
            sensor_angles=angles,
            range_m=sim.config.sensors.range_m,
        ),
        observation_batch(batch, track, sim.config, sensor_angles=angles),
    )


def test_gpu_ray_distances_match_cpu() -> None:
    sim = MonzaSim(SimConfig(action_mode="continuous", observation_profile="base"))
    for progress_m, speed_kph in ((0.0, 80.0), (520.0, 230.0), (2500.0, 160.0), (5260.0, 220.0)):
        sim.reset(seed=3, options={"start_progress_m": progress_m, "start_speed_kph": speed_kph})
        gpu_rays, _ = _batch_from_sim(sim)

        assert np.allclose(gpu_rays[0].numpy(), sim.ray_distances_m(), atol=1e-3)


def test_gpu_observations_match_cpu_profiles() -> None:
    for profile in ("base", "brake", "guidance", "racing", "racing_release", "racing_v2", "learned_policy_v1"):
        sim = MonzaSim(SimConfig(action_mode="continuous", observation_profile=profile))
        sim.reset(seed=4, options={"start_progress_m": 2140.0, "start_speed_kph": 155.0})
        _gpu_rays, gpu_obs = _batch_from_sim(sim)

        assert gpu_obs.shape == (1, sim.observation_dim)
        assert np.allclose(gpu_obs[0].numpy(), sim.observation(), atol=2e-3), profile
