import json
from pathlib import Path

import numpy as np

from f1rl.bc_train import train_bc
from f1rl.config import SimConfig
from f1rl.es_dataset import export_dataset
from f1rl.evolution_postcheck import postcheck_evolution_run
from f1rl.evolution_search import EvolutionGates, EvolutionSearchConfig, run_evolution_search
from f1rl.learned_policy import (
    PolicyNormalizer,
    SACActor,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from f1rl.policy_eval import evaluate_policy
from f1rl.policy_swarm_eval import run_policy_swarm_eval
from f1rl.sac_train import train_sac
from f1rl.sim import MonzaSim
from f1rl.telemetry import load_steps


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v2_evolution_metadata_reaches_generation_checkpoint_manifest_and_dataset(tmp_path: Path) -> None:
    output_dir = run_evolution_search(
        output_dir=tmp_path / "v2-evolution",
        config=EvolutionSearchConfig(
            physics_model="v2",
            action_set="straight",
            observation_profile="base",
            max_steps=6,
            population=3,
            generations=1,
            elite_count=1,
            random_immigrants=0,
            min_phases=1,
            max_phases=2,
            min_phase_steps=2,
            max_phase_steps=4,
            seed=211,
            top_k=1,
            workers=1,
            genome_type="phase",
            scoring_profiles=("max_progress", "clean_exit"),
            checkpoint_every_generations=1,
            telemetry_compression="gzip",
        ),
        gates=EvolutionGates(target_progress_m=504.0, terminate_at_target_progress=False),
        start_progress_m=500.0,
        start_speed_kph=60.0,
    )

    summary = json.loads((output_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    generation = _jsonl_rows(output_dir / "generation_summary.jsonl")[0]
    checkpoint = json.loads((output_dir / "population_checkpoint.json").read_text(encoding="utf-8"))
    bridge = json.loads((output_dir / "ppo_bridge.json").read_text(encoding="utf-8"))
    attempts = _jsonl_rows(output_dir / "attempts.jsonl")
    selected_manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))
    first_steps = load_steps(Path(selected_manifest["traces"][0]["path"]))

    assert summary["physics_model"] == "v2"
    assert summary["physics_version"].startswith("physics_v2.")
    assert summary["physics_calibration_id"]
    assert generation["physics_model"] == "v2"
    assert generation["physics_version"] == summary["physics_version"]
    assert checkpoint["config"]["physics_model"] == "v2"
    assert bridge["physics_model"] == "v2"
    assert bridge["physics_version"] == summary["physics_version"]
    assert bridge["physics_calibration_id"] == summary["physics_calibration_id"]
    assert "--physics-model v2" in bridge["recommended_commands"]["train_curriculum"]
    assert "--physics-model v2" in bridge["recommended_commands"]["honest_normal_start_benchmark"]
    assert all(row["physics_model"] == "v2" for row in attempts)
    assert selected_manifest["physics_model"] == "v2"
    assert first_steps[0]["physics_model"] == "v2"
    assert first_steps[0]["physics_calibration_id"] == summary["physics_calibration_id"]

    postcheck_summary_path = postcheck_evolution_run(output_dir, top_k=1, telemetry_compression="gzip")
    postcheck_summary = json.loads(postcheck_summary_path.read_text(encoding="utf-8"))
    postcheck_manifest = json.loads((output_dir / "selected_telemetry" / "manifest.json").read_text(encoding="utf-8"))
    postchecked_rows = _jsonl_rows(output_dir / "postchecked_attempts.jsonl")
    postcheck_steps = load_steps(Path(postcheck_manifest["traces"][0]["path"]))

    assert postcheck_summary["physics_model"] == "v2"
    assert postcheck_summary["physics_version"] == summary["physics_version"]
    assert postcheck_summary["physics_calibration_id"] == summary["physics_calibration_id"]
    assert postcheck_manifest["physics_model"] == "v2"
    assert postcheck_manifest["physics_version"] == summary["physics_version"]
    assert postcheck_manifest["physics_calibration_id"] == summary["physics_calibration_id"]
    assert postcheck_manifest["traces"][0]["physics_model"] == "v2"
    assert postchecked_rows[0]["physics_model"] == "v2"
    assert postchecked_rows[0]["physics_calibration_id"] == summary["physics_calibration_id"]
    assert postcheck_steps[0]["physics_model"] == "v2"

    dataset_manifest_path = export_dataset(
        run_dir=output_dir,
        selected_telemetry=output_dir / "selected_telemetry",
        output_dir=tmp_path / "v2-dataset",
        observation_profile="base",
        physics_model="v2",
        max_candidates=1,
        max_per_generation=1,
        balanced_buckets=False,
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    source_rows = _jsonl_rows(dataset_manifest_path.parent / "source_candidates.jsonl")

    assert dataset_manifest["physics_model"] == "v2"
    assert dataset_manifest["physics_version"] == summary["physics_version"]
    assert dataset_manifest["physics_calibration_id"] == summary["physics_calibration_id"]
    assert dataset_manifest["sim_config"]["physics_model"] == "v2"
    assert source_rows
    assert source_rows[0]["physics_model"] == "v2"
    assert source_rows[0]["physics_calibration_id"] == summary["physics_calibration_id"]

    bc_policy = train_bc(
        dataset=dataset_manifest_path.parent,
        output_dir=tmp_path / "v2-bc",
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=3e-4,
        hidden_size=8,
        seed=13,
        smoke=True,
        overfit_source_id=None,
        resume=None,
        late_progress_threshold_m=4800.0,
        late_progress_weight=1.0,
        fastest_valid_weight=1.0,
        brake_action_weight=1.0,
        control_mode="dominance",
    )
    _, _, bc_metadata = load_policy_checkpoint(bc_policy)

    assert bc_metadata["physics_model"] == "v2"
    assert bc_metadata["physics_version"] == summary["physics_version"]
    assert bc_metadata["physics_calibration_id"] == summary["physics_calibration_id"]
    assert bc_metadata["config"]["physics_model"] == "v2"

    sac_policy = train_sac(
        dataset=dataset_manifest_path.parent,
        bc_checkpoint=bc_policy,
        output_dir=tmp_path / "v2-sac",
        device="cpu",
        timesteps=1,
        n_envs=1,
        max_steps=3,
        batch_size=2,
        hidden_size=8,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        updates_per_step=1.0,
        bc_loss_weight=0.1,
        rollout_deterministic=True,
        rollout_noise_std=0.0,
        initial_alpha=0.1,
        target_entropy=-3.0,
        freeze_alpha=True,
        control_mode="dominance",
        rollout_start_position_noise_m=0.0,
        rollout_start_heading_noise_deg=0.0,
        rollout_start_speed_noise_kph=0.0,
        es_fastest_weight=1.0,
        es_valid_time_power=1.0,
        es_late_progress_weight=1.0,
        es_frontier_weight=1.0,
        reward_time_penalty_per_step=0.0,
        reward_speed_scale=0.0,
        reward_valid_finish_bonus=0.0,
        eval_every=1,
        swarm_every=0,
        swarm_size=0,
        physics_model="dataset",
        seed=19,
    )
    _, _, sac_metadata = load_policy_checkpoint(sac_policy)

    assert sac_metadata["physics_model"] == "v2"
    assert sac_metadata["physics_version"] == summary["physics_version"]
    assert sac_metadata["physics_calibration_id"] == summary["physics_calibration_id"]
    assert sac_metadata["config"]["physics_model"] == "v2"

    v1_transfer_manifest_path = export_dataset(
        run_dir=output_dir,
        selected_telemetry=output_dir / "selected_telemetry",
        output_dir=tmp_path / "v1-transfer-dataset",
        observation_profile="base",
        physics_model="v1",
        max_candidates=1,
        max_per_generation=1,
        balanced_buckets=False,
    )
    v1_transfer_manifest = json.loads(v1_transfer_manifest_path.read_text(encoding="utf-8"))
    v1_transfer_source_rows = _jsonl_rows(v1_transfer_manifest_path.parent / "source_candidates.jsonl")

    assert v1_transfer_manifest["physics_model"] == "v1"
    assert v1_transfer_manifest["physics_version"] == "physics_v1.0.0"
    assert v1_transfer_manifest["physics_calibration_id"] is None
    assert v1_transfer_manifest["sim_config"]["physics_model"] == "v1"
    assert v1_transfer_source_rows[0]["physics_model"] == "v1"
    assert v1_transfer_source_rows[0]["physics_calibration_id"] is None


def _write_dummy_policy(path: Path, *, observation_profile: str, physics_model: str) -> Path:
    sim = MonzaSim(
        SimConfig(
            max_steps=3,
            action_mode="continuous",
            action_set="racing",
            observation_profile=observation_profile,
            physics_model=physics_model,
        )
    )
    obs, _ = sim.reset(seed=1)
    actor = SACActor(int(obs.shape[0]), hidden_sizes=(), control_mode="dominance")
    for parameter in actor.parameters():
        parameter.data.zero_()
    normalizer = PolicyNormalizer.identity(int(obs.shape[0]))
    save_policy_checkpoint(
        path,
        actor=actor,
        normalizer=normalizer,
        metadata={
            "stage": "unit",
            "observation_profile": observation_profile,
            "observation_dim": int(obs.shape[0]),
            "physics_model": physics_model,
            "normalizer_mean": np.zeros(int(obs.shape[0]), dtype=np.float32).tolist(),
        },
    )
    return path


def test_v2_policy_eval_and_swarm_manifests_record_physics_metadata(tmp_path: Path) -> None:
    policy = _write_dummy_policy(tmp_path / "policy.pt", observation_profile="base", physics_model="v2")

    summary = evaluate_policy(
        policy=policy,
        output_dir=tmp_path / "policy-eval",
        episodes=1,
        deterministic=True,
        normal_start=True,
        observation_profile="base",
        physics_model="v2",
        write_telemetry="gzip",
        max_steps=3,
        seed=17,
        device="cpu",
    )
    eval_manifest_path = tmp_path / "policy-eval" / "selected_telemetry" / "manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    eval_steps = load_steps(Path(eval_manifest["traces"][0]["path"]))

    assert summary["physics_model"] == "v2"
    assert summary["physics_version"].startswith("physics_v2.")
    assert summary["physics_calibration_id"]
    assert eval_manifest["physics_model"] == "v2"
    assert eval_manifest["physics_calibration_id"] == summary["physics_calibration_id"]
    assert eval_steps[0]["physics_model"] == "v2"

    swarm_manifest_path = run_policy_swarm_eval(
        policy_dir=policy,
        output_dir=tmp_path / "policy-swarm",
        swarm_size=2,
        checkpoints="all",
        observation_profile="base",
        physics_model="v2",
        deterministic=True,
        max_steps=3,
        seed=23,
        device="cpu",
        full_telemetry_limit=1,
        start_position_noise_m=0.0,
        start_heading_noise_deg=0.0,
        start_speed_noise_kph=0.0,
    )
    swarm_manifest = json.loads(swarm_manifest_path.read_text(encoding="utf-8"))
    swarm_steps = load_steps(Path(swarm_manifest["traces"][0]["path"]))

    assert swarm_manifest["physics_model"] == "v2"
    assert swarm_manifest["physics_version"] == summary["physics_version"]
    assert swarm_manifest["physics_calibration_id"] == summary["physics_calibration_id"]
    assert swarm_manifest["checkpoint_summaries"][0]["physics_model"] == "v2"
    assert swarm_steps[0]["physics_model"] == "v2"
