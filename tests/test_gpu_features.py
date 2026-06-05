# pyright: reportPrivateImportUsage=false

import numpy as np
import pytest
import torch

from f1rl.evolution_search import (
    CONTROLLER_FEATURE_NAMES,
    CONTROLLER_OUTPUT_COUNT,
    Genome,
    _controller_controls,
)
from f1rl.gpu_fast_warp import warp_status
from f1rl.gpu_features import controller_controls_batch, search_features_batch
from f1rl.gpu_fused_warp import (
    controller_controls_warp_batch,
    phase_controls_warp_batch,
    search_features_warp_batch,
)
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


def test_controller_controls_use_brake_preferred_near_tie() -> None:
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = torch.zeros((1, CONTROLLER_OUTPUT_COUNT, feature_count), dtype=torch.float64)
    features = torch.zeros((1, feature_count), dtype=torch.float64)
    features[0, 0] = 1.0
    weights[0, 0, 0] = 8.0e-4
    genome = Genome(kind="controller", controller_weights=tuple(float(value) for value in weights.reshape(-1)))
    feature_map = {name: float(features[0, index]) for index, name in enumerate(CONTROLLER_FEATURE_NAMES)}

    cpu_throttle, cpu_brake, cpu_steer = _controller_controls(genome, feature_map)
    torch_throttle, torch_brake, torch_steer = controller_controls_batch(weights, features)

    assert cpu_brake > cpu_throttle
    assert float(torch_brake[0]) > float(torch_throttle[0])
    assert np.allclose((float(torch_throttle[0]), float(torch_brake[0]), float(torch_steer[0])), (cpu_throttle, cpu_brake, cpu_steer))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp controller parity requires CUDA and Warp CUDA support",
)
def test_warp_controller_controls_match_torch_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(909)
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = torch.randn(
        (96, CONTROLLER_OUTPUT_COUNT, feature_count),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ) * 2.0
    features = torch.randn((96, feature_count), generator=generator, device="cuda", dtype=torch.float32)

    expected = controller_controls_batch(weights, features)
    actual = controller_controls_warp_batch(weights, features)
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp controller parity requires CUDA and Warp CUDA support",
)
def test_warp_controller_controls_match_torch_reference_near_tie() -> None:
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = torch.zeros((1, CONTROLLER_OUTPUT_COUNT, feature_count), device="cuda", dtype=torch.float32)
    features = torch.zeros((1, feature_count), device="cuda", dtype=torch.float32)
    features[0, 0] = 1.0
    weights[0, 0, 0] = 8.0e-4

    expected = controller_controls_batch(weights, features)
    actual = controller_controls_warp_batch(weights, features)
    torch.cuda.synchronize()

    assert float(expected[1][0]) > float(expected[0][0])
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp controller parity requires CUDA and Warp CUDA support",
)
def test_warp_controller_controls_match_torch_reference_at_saturation() -> None:
    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = torch.zeros((6, CONTROLLER_OUTPUT_COUNT, feature_count), device="cuda", dtype=torch.float32)
    features = torch.ones((6, feature_count), device="cuda", dtype=torch.float32)
    weights[:, 0, 0] = torch.tensor([-80.0, -42.0, -1.0, 1.0, 42.0, 80.0], device="cuda")
    weights[:, 1, 1] = torch.tensor([80.0, 42.0, 1.0, -1.0, -42.0, -80.0], device="cuda")
    weights[:, 2, 2] = torch.tensor([-12.0, -4.0, -0.5, 0.5, 4.0, 12.0], device="cuda")

    expected = controller_controls_batch(weights, features)
    actual = controller_controls_warp_batch(weights, features)
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp phase-control parity requires CUDA and Warp CUDA support",
)
def test_warp_phase_controls_match_torch_reference() -> None:
    action_controls = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, -0.5],
            [0.2, 0.0, 0.75],
            [0.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    phase_action_ids = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 0, 1],
            [3, 1, 0],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    phase_thresholds = torch.tensor(
        [
            [2.0, 5.0, torch.inf],
            [1.0, 2.0, 4.0],
            [0.0, 3.0, 6.0],
            [4.0, torch.inf, torch.inf],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    elapsed_steps = torch.tensor([0, 2, 3, 6], device="cuda", dtype=torch.int64)
    progress_delta_m = torch.zeros(4, device="cuda", dtype=torch.float32)

    phase_index = torch.sum(elapsed_steps[:, None] >= phase_thresholds, dim=1)
    phase_index = torch.clamp(phase_index, max=phase_action_ids.shape[1] - 1).to(dtype=torch.int64)
    row_index = torch.arange(phase_action_ids.shape[0], device="cuda")
    expected_action_id = phase_action_ids[row_index, phase_index]
    expected_controls = action_controls[expected_action_id]

    throttle, brake, steer, action_id = phase_controls_warp_batch(
        action_controls=action_controls,
        phase_action_ids=phase_action_ids,
        phase_thresholds=phase_thresholds,
        elapsed_steps=elapsed_steps,
        progress_delta_m=progress_delta_m,
        use_progress=False,
    )
    torch.cuda.synchronize()

    assert torch.equal(action_id, expected_action_id)
    assert torch.allclose(throttle, expected_controls[:, 0])
    assert torch.allclose(brake, expected_controls[:, 1])
    assert torch.allclose(steer, expected_controls[:, 2])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp progress-phase-control parity requires CUDA and Warp CUDA support",
)
def test_warp_progress_phase_controls_match_torch_reference() -> None:
    action_controls = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.6, -0.25],
            [0.1, 0.0, 0.5],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    phase_action_ids = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 2],
            [0, 2, 1],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    phase_thresholds = torch.tensor(
        [
            [4.0, 12.0, torch.inf],
            [2.0, 3.0, 8.0],
            [0.0, 9.0, torch.inf],
            [5.0, 10.0, 15.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    elapsed_steps = torch.tensor([99, 99, 99, 99], device="cuda", dtype=torch.int64)
    progress_delta_m = torch.tensor([0.0, 3.5, 9.0, 20.0], device="cuda", dtype=torch.float32)

    phase_index = torch.sum(progress_delta_m[:, None] >= phase_thresholds, dim=1)
    phase_index = torch.clamp(phase_index, max=phase_action_ids.shape[1] - 1).to(dtype=torch.int64)
    row_index = torch.arange(phase_action_ids.shape[0], device="cuda")
    expected_action_id = phase_action_ids[row_index, phase_index]
    expected_controls = action_controls[expected_action_id]

    throttle, brake, steer, action_id = phase_controls_warp_batch(
        action_controls=action_controls,
        phase_action_ids=phase_action_ids,
        phase_thresholds=phase_thresholds,
        elapsed_steps=elapsed_steps,
        progress_delta_m=progress_delta_m,
        use_progress=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(action_id, expected_action_id)
    assert torch.allclose(throttle, expected_controls[:, 0])
    assert torch.allclose(brake, expected_controls[:, 1])
    assert torch.allclose(steer, expected_controls[:, 2])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not warp_status().cuda_available,
    reason="Warp search-feature parity requires CUDA and Warp CUDA support",
)
def test_warp_search_features_match_torch_reference() -> None:
    sim = MonzaSim()
    snapshots = []
    for seed, progress_m, speed_kph in (
        (14, 0.0, 80.0),
        (15, 520.0, 260.0),
        (16, 2140.0, 180.0),
        (17, 4740.0, 260.0),
        (18, 5260.0, 280.0),
    ):
        sim.reset(seed=seed, options={"start_progress_m": progress_m, "start_speed_kph": speed_kph})
        snapshots.append(snapshot_from_sim(sim, source="warp_feature_unit"))
    track = gpu_track_from_cpu(load_track_spec(sim.config.track_path), device=torch.device("cuda"), dtype=torch.float32)
    state = car_batch_from_snapshots(
        snapshots,
        meters_per_pixel=sim.track.meters_per_pixel,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    previous_segment_idx = torch.searchsorted(
        track.centerline_cumdist_px,
        torch.remainder(state.last_raw_progress_px, track.length_px),
        right=True,
    ) - 1
    previous_segment_idx = torch.clamp(previous_segment_idx, 0, track.centerline_xy.shape[0] - 1)
    segment_start_progress_m = state.monotonic_progress_m.clone()
    lookahead_m = torch.tensor(tuple(float(value) for value in sim.config.lookahead_m), device="cuda", dtype=torch.float32)
    braking_gates_m = torch.tensor([520.0, 2140.0, 4740.0], device="cuda", dtype=torch.float32)
    window_px = torch.as_tensor(
        float(sim.config.local_projection_window_m),
        device="cuda",
        dtype=torch.float32,
    ) / torch.clamp(track.meters_per_pixel, min=1e-6)
    zero = torch.zeros(state.size, device="cuda", dtype=torch.float32)
    row_index = torch.arange(state.size, device="cuda", dtype=torch.int64)

    expected_features, expected_diagnostics = search_features_batch(
        state,
        track,
        sim.config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        segment_start_progress_m=segment_start_progress_m,
        segment_target_progress_m=1500.0,
        braking_gates_m=braking_gates_m,
        lookahead_m=lookahead_m,
        local_projection_window_px=window_px,
        previous_segment_idx=previous_segment_idx,
        zero_feature=zero,
        row_index=row_index,
    )
    actual_features, actual_diagnostics = search_features_warp_batch(
        state,
        track,
        sim.config,
        feature_names=CONTROLLER_FEATURE_NAMES,
        segment_start_progress_m=segment_start_progress_m,
        segment_target_progress_m=1500.0,
        braking_gates_m=braking_gates_m,
        lookahead_m=lookahead_m,
        local_projection_window_px=window_px,
        previous_segment_idx=previous_segment_idx,
        zero_feature=zero,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual_features, expected_features, rtol=2e-4, atol=2e-3)
    assert torch.equal(actual_diagnostics["segment_idx"], expected_diagnostics["segment_idx"])
    for key, expected_value in expected_diagnostics.items():
        if key == "segment_idx":
            continue
        assert torch.allclose(actual_diagnostics[key], expected_value, rtol=2e-4, atol=2e-3), key


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
