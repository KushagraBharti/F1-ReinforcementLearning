from dataclasses import replace

import numpy as np
import pytest
import torch

from f1rl.config import CarParams
from f1rl.gpu_physics import apply_physics_batch
from f1rl.gpu_types import car_batch_from_states, gpu_car_params_from_cpu
from f1rl.physics import CarState, apply_physics


def _batch(states: list[CarState], *, device: str = "cpu", dtype: torch.dtype = torch.float64):
    return car_batch_from_states(
        states,
        meters_per_pixel=1.0,
        device=torch.device(device),
        dtype=dtype,
    )


def _compare_state(cpu: CarState, batch, index: int, *, atol: float) -> None:
    assert np.isclose(float(batch.x[index].detach().cpu()), cpu.x, atol=atol)
    assert np.isclose(float(batch.y[index].detach().cpu()), cpu.y, atol=atol)
    assert np.isclose(float(batch.heading_rad[index].detach().cpu()), cpu.heading_rad, atol=atol)
    assert np.isclose(float(batch.speed_mps[index].detach().cpu()), cpu.speed_mps, atol=atol)
    assert np.isclose(float(batch.yaw_rate_rps[index].detach().cpu()), cpu.yaw_rate_rps, atol=atol)
    assert np.isclose(float(batch.steering[index].detach().cpu()), cpu.steering, atol=atol)


def test_gpu_physics_one_step_matches_cpu() -> None:
    params = CarParams()
    states = [
        CarState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=20.0),
        CarState(x=4.0, y=-2.0, heading_rad=0.2, speed_mps=35.0),
        CarState(x=8.0, y=3.0, heading_rad=-0.4, speed_mps=55.0),
    ]
    controls = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.4, 0.0, 1.0)]
    batch = _batch(states)
    next_batch, movement = apply_physics_batch(
        batch,
        throttle=torch.tensor([item[0] for item in controls], dtype=torch.float64),
        brake=torch.tensor([item[1] for item in controls], dtype=torch.float64),
        steer=torch.tensor([item[2] for item in controls], dtype=torch.float64),
        params=gpu_car_params_from_cpu(params),
        meters_per_pixel=torch.tensor(1.0, dtype=torch.float64),
    )

    for index, (state, control) in enumerate(zip(states, controls, strict=True)):
        cpu_state, cpu_movement = apply_physics(
            state,
            throttle=control[0],
            brake=control[1],
            steer=control[2],
            params=params,
            meters_per_pixel=1.0,
        )
        _compare_state(cpu_state, next_batch, index, atol=1e-8)
        assert np.allclose(movement.as_segments()[index].detach().cpu().numpy(), cpu_movement, atol=1e-6)


def test_gpu_physics_multi_step_matches_cpu_loop() -> None:
    params = CarParams()
    base = [CarState(x=float(i), y=float(i * 2), heading_rad=0.1 * i, speed_mps=10.0 + i) for i in range(32)]
    cpu_states = list(base)
    gpu_state = _batch(base)
    rng = np.random.default_rng(3)

    for _ in range(300):
        throttle = rng.uniform(0.0, 1.0, len(base))
        brake = rng.uniform(0.0, 0.4, len(base))
        steer = rng.uniform(-1.0, 1.0, len(base))
        gpu_state, _ = apply_physics_batch(
            gpu_state,
            throttle=torch.tensor(throttle, dtype=torch.float64),
            brake=torch.tensor(brake, dtype=torch.float64),
            steer=torch.tensor(steer, dtype=torch.float64),
            params=gpu_car_params_from_cpu(params),
            meters_per_pixel=torch.tensor(0.1, dtype=torch.float64),
        )
        cpu_states = [
            apply_physics(
                state,
                throttle=float(throttle[index]),
                brake=float(brake[index]),
                steer=float(steer[index]),
                params=params,
                meters_per_pixel=0.1,
            )[0]
            for index, state in enumerate(cpu_states)
        ]

    for index, state in enumerate(cpu_states):
        _compare_state(state, gpu_state, index, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_physics_cuda_matches_cpu_with_float32_tolerance() -> None:
    params = CarParams()
    state = CarState(x=2.0, y=4.0, heading_rad=0.3, speed_mps=42.0)
    cpu_state, _ = apply_physics(
        state,
        throttle=0.7,
        brake=0.0,
        steer=-0.6,
        params=params,
        meters_per_pixel=0.1,
    )
    batch = _batch([replace(state)], device="cuda", dtype=torch.float32)
    gpu_state, _ = apply_physics_batch(
        batch,
        throttle=torch.tensor([0.7], device="cuda"),
        brake=torch.tensor([0.0], device="cuda"),
        steer=torch.tensor([-0.6], device="cuda"),
        params=gpu_car_params_from_cpu(params),
        meters_per_pixel=torch.tensor(0.1, device="cuda"),
    )
    _compare_state(cpu_state, gpu_state, 0, atol=1e-3)
