# pyright: reportPrivateImportUsage=false
"""Initialize a neural learned-policy checkpoint from a CPU-replayable ES controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1rl.config import LEARNED_DIR, LEARNED_POLICY_V1_FEATURES, SimConfig
from f1rl.es_dataset import _load_source_rows, observation_feature_schema
from f1rl.evolution_search import (
    CONTROLLER_FEATURE_NAMES,
    CONTROLLER_OUTPUT_COUNT,
    genome_from_mapping,
)
from f1rl.learned_policy import PolicyNormalizer, SACActor, save_policy_checkpoint


def initialize_controller_policy(
    *,
    run_dir: Path,
    output_path: Path,
    source_candidate_id: int,
    observation_profile: str,
    max_per_generation: int,
    source_json: Path | None = None,
) -> Path:
    if observation_profile != "learned_policy_v1":
        raise ValueError("Controller policy initialization requires observation_profile='learned_policy_v1'.")
    if source_json is not None:
        source_path = source_json
        if not source_path.is_absolute() and not source_path.exists():
            source_path = run_dir / source_path
        row = json.loads(source_path.read_text(encoding="utf-8"))
        selected_source_candidate_id = source_candidate_id
    else:
        source_rows = _load_source_rows(run_dir, max_per_generation=max_per_generation)
        if source_candidate_id < 0 or source_candidate_id >= len(source_rows):
            raise ValueError(f"source_candidate_id={source_candidate_id} is outside 0..{len(source_rows) - 1}")
        row = source_rows[source_candidate_id]
        selected_source_candidate_id = source_candidate_id
    genome = genome_from_mapping(row["genome"])
    if genome.kind != "controller":
        raise ValueError(f"source_candidate_id={source_candidate_id} is genome kind {genome.kind!r}, not 'controller'.")

    feature_count = len(CONTROLLER_FEATURE_NAMES)
    weights = np.asarray(genome.controller_weights, dtype=np.float32).reshape((CONTROLLER_OUTPUT_COUNT, feature_count))
    sim_config = SimConfig(
        observation_profile=observation_profile,
        action_mode="continuous",
        action_set="racing",
    )
    from f1rl.sim import MonzaSim

    sim = MonzaSim(sim_config)
    actor = SACActor(sim.observation_dim, hidden_sizes=(), control_mode="dominance")
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.zero_()
        actor.log_std_head.weight.zero_()
        actor.log_std_head.bias.fill_(-5.0)
        learned_offset = sim.observation_dim - len(LEARNED_POLICY_V1_FEATURES)
        feature_index = {name: index for index, name in enumerate(LEARNED_POLICY_V1_FEATURES)}
        for controller_index, name in enumerate(CONTROLLER_FEATURE_NAMES):
            if name not in feature_index:
                continue
            column = learned_offset + feature_index[name]
            actor.mean_head.weight[0, column] = float(weights[0, controller_index]) * 0.5
            actor.mean_head.weight[1, column] = float(weights[1, controller_index]) * 0.5
            actor.mean_head.weight[2, column] = float(weights[2, controller_index])

    normalizer = PolicyNormalizer.identity(sim.observation_dim)
    metadata: dict[str, Any] = {
        "stage": "controller_distill",
        "source": "cpu_replayable_es_controller",
        "source_run": str(run_dir),
        "source_candidate_id": selected_source_candidate_id,
        "source_json": str(source_json) if source_json is not None else None,
        "source_generation": int(row.get("generation", 0)),
        "source_candidate_index": int(row.get("candidate_index", -1)),
        "source_seed": int(row.get("seed", 0)),
        "source_terminal_reason": row.get("termination_reason"),
        "source_lap_time_s": row.get("lap_time_s") or row.get("elapsed_s"),
        "observation_profile": observation_profile,
        "observation_dim": sim.observation_dim,
        "observation_feature_schema": observation_feature_schema(observation_profile),
        "control_mode": actor.control_mode,
        "actor_form": "linear_controller_distillation",
    }
    save_policy_checkpoint(output_path, actor=actor, normalizer=normalizer, metadata=metadata)
    (output_path.parent / "controller_policy_init.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"controller_policy_init_complete checkpoint={output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a neural policy checkpoint from an ES controller genome.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", default=str(LEARNED_DIR / "controller-distill" / "policy.pt"))
    parser.add_argument("--source-candidate-id", type=int, default=0)
    parser.add_argument(
        "--source-json",
        help="Initialize from exactly one source JSON row, such as best_so_far.json; relative paths resolve from the run dir.",
    )
    parser.add_argument("--observation-profile", default="learned_policy_v1")
    parser.add_argument("--max-per-generation", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    initialize_controller_policy(
        run_dir=Path(args.run_dir),
        output_path=Path(args.output_path),
        source_candidate_id=args.source_candidate_id,
        observation_profile=args.observation_profile,
        max_per_generation=args.max_per_generation,
        source_json=Path(args.source_json) if args.source_json else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
