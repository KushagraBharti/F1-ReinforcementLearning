import json
from pathlib import Path

from f1rl.gpu_ppo import GpuPPOConfig, train_gpu_ppo


def test_gpu_ppo_cpu_device_smoke_writes_policy_and_cpu_eval(tmp_path: Path) -> None:
    run_dir = train_gpu_ppo(
        GpuPPOConfig(
            output_dir=tmp_path / "gpu-ppo",
            device="cpu",
            dtype="float32",
            seed=71,
            timesteps=8,
            n_envs=2,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            hidden_size=32,
            max_steps=8,
            physics_model="v2",
            observation_profile="base",
            start_speed_kph=60.0,
            target_progress_m=6.0,
            terminate_at_target=True,
            cpu_eval_episodes=1,
            cpu_eval_max_steps=8,
        )
    )

    training_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    cpu_eval = json.loads((run_dir / "cpu_eval_summary.json").read_text(encoding="utf-8"))

    assert training_summary["backend"] == "gpu_ppo"
    assert training_summary["device"] == "cpu"
    assert training_summary["physics_model"] == "v2"
    assert training_summary["physics_version"].startswith("physics_v2.")
    assert training_summary["timesteps_collected"] == 8
    assert Path(training_summary["policy_path"]).exists()
    assert cpu_eval["backend"] == "cpu_replay"
    assert cpu_eval["physics_model"] == "v2"
    assert cpu_eval["episodes"]
    assert (run_dir / "cpu_eval_episode_000.jsonl").exists()
