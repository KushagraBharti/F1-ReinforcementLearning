import numpy as np
import torch

from f1rl.evolution_search import (
    CONTROLLER_FEATURE_NAMES,
    CONTROLLER_OUTPUT_COUNT,
    Genome,
    _controller_controls,
)
from f1rl.gpu_features import controller_controls_batch, search_features_batch
from f1rl.gpu_types import car_batch_from_snapshots, gpu_track_from_cpu
from f1rl.sim import MonzaSim
from f1rl.state_snapshot import snapshot_from_sim
from f1rl.track_model import load_track_spec


def _gpu_features_at(progress_m: float, speed_kph: float) -> tuple[dict[str, float], dict[str, float]]:
    sim = MonzaSim()
    sim.reset(seed=4, options={"start_progress_m": progress_m, "start_speed_kph": speed_kph})
    snapshot = snapshot_from_sim(sim, source="unit")
    track = gpu_track_from_cpu(load_track_spec(sim.config.track_path), device=torch.device("cpu"), dtype=torch.float64)
    batch = car_batch_from_snapshots([snapshot], meters_per_pixel=sim.track.meters_per_pixel, device=torch.device("cpu"), dtype=torch.float64)
    features, diagnostics = search_features_batch(
        batch,
        track,
        sim.config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        segment_start_progress_m=torch.tensor([snapshot.monotonic_progress_m], dtype=torch.float64),
        segment_target_progress_m=1500.0,
    )
    gpu = {name: float(features[0, index]) for index, name in enumerate(CONTROLLER_FEATURE_NAMES)}
    gpu.update({key: float(value[0]) for key, value in diagnostics.items() if value.dtype != torch.bool})
    cpu = sim.search_features(
        segment_start_progress_m=snapshot.monotonic_progress_m,
        segment_target_progress_m=1500.0,
    )
    return cpu, gpu


def test_gpu_controller_controls_match_numpy_reference() -> None:
    rng = np.random.default_rng(5)
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = rng.normal(0.0, 2.0, (64, CONTROLLER_OUTPUT_COUNT, feature_count))
    features = rng.normal(0.0, 1.0, (64, feature_count))
    torch_controls = controller_controls_batch(
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(features, dtype=torch.float64),
    )

    for index in range(64):
        genome = Genome(kind="controller", controller_weights=tuple(weights[index].reshape(-1)))
        feature_map = {
            name: float(features[index, feature_index])
            for feature_index, name in enumerate(CONTROLLER_FEATURE_NAMES)
        }
        expected = _controller_controls(genome, feature_map)
        actual = tuple(float(item[index]) for item in torch_controls)
        assert np.allclose(actual, expected, atol=1e-10)


def test_gpu_search_features_match_cpu_at_representative_points() -> None:
    for progress_m, speed_kph in (
        (0.0, 80.0),
        (520.0, 260.0),
        (2140.0, 180.0),
        (4740.0, 260.0),
        (5260.0, 280.0),
    ):
        cpu, gpu = _gpu_features_at(progress_m, speed_kph)
        for key in CONTROLLER_FEATURE_NAMES:
            assert np.isclose(gpu[key], cpu[key], atol=2e-3), (progress_m, key, cpu[key], gpu[key])
        for key in (
            "lateral_error_m",
            "signed_lateral_error_m",
            "heading_error_deg",
            "target_speed_kph",
            "near_target_speed_kph",
            "min_future_target_speed_kph",
            "target_speed_drop_kph",
            "braking_gate_distance_m",
            "speed_kph",
        ):
            assert np.isclose(gpu[key], cpu[key], atol=0.2), (progress_m, key, cpu[key], gpu[key])
