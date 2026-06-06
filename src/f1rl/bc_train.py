# pyright: reportPrivateImportUsage=false
"""Behavior cloning trainer for ES transition datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1rl.config import LEARNED_DIR
from f1rl.es_dataset import load_dataset
from f1rl.learned_policy import (
    PolicyNormalizer,
    SACActor,
    bc_action_loss,
    load_policy_checkpoint,
    save_policy_checkpoint,
)


def _load_source_weights(dataset_root: Path) -> dict[int, float]:
    path = dataset_root / "source_candidates.jsonl"
    if not path.exists():
        return {}
    bucket_weight = {
        "valid_lap": 3.0,
        "near_valid_5500m": 1.8,
        "near_valid_5000m": 1.4,
        "late_frontier": 1.0,
        "mid_frontier": 0.7,
        "early_failure": 0.25,
    }
    weights: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        weights[int(row["source_candidate_id"])] = bucket_weight.get(str(row.get("source_bucket")), 1.0)
    return weights


def _fastest_valid_source_ids(dataset_root: Path) -> set[int]:
    path = dataset_root / "source_candidates.jsonl"
    if not path.exists():
        return set()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    valid_rows = [
        row
        for row in rows
        if str(row.get("termination_reason")) == "lap_complete" and row.get("lap_time_s") is not None
    ]
    if not valid_rows:
        return set()
    fastest = min(float(row["lap_time_s"]) for row in valid_rows)
    return {
        int(row["source_candidate_id"])
        for row in valid_rows
        if float(row["lap_time_s"]) <= fastest + 1e-6
    }


def _batch_indices(
    rng: np.random.Generator,
    *,
    count: int,
    batch_size: int,
    probabilities: np.ndarray | None,
) -> np.ndarray:
    return rng.choice(count, size=batch_size, replace=count < batch_size, p=probabilities)


def train_bc(
    *,
    dataset: Path,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    seed: int,
    smoke: bool,
    overfit_source_id: int | None,
    resume: Path | None,
    late_progress_threshold_m: float,
    late_progress_weight: float,
    fastest_valid_weight: float,
    brake_action_weight: float,
    control_mode: str,
) -> Path:
    data = load_dataset(dataset)
    arrays = data.arrays
    obs = arrays["obs"].astype(np.float32)
    actions = arrays["action"].astype(np.float32)
    source_ids = arrays["source_candidate_id"].astype(np.int64)
    progress_m = arrays["progress_m"].astype(np.float32)
    if overfit_source_id is not None:
        mask = source_ids == overfit_source_id
        if not np.any(mask):
            raise ValueError(f"No transitions for source_candidate_id={overfit_source_id}")
        obs = obs[mask]
        actions = actions[mask]
        source_ids = source_ids[mask]
        progress_m = progress_m[mask]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizer = PolicyNormalizer.from_observations(obs)
    actor = SACActor(obs.shape[1], hidden_sizes=(hidden_size, hidden_size), control_mode=control_mode).to(torch_device)
    if resume is not None:
        actor, normalizer, _ = load_policy_checkpoint(resume, device=torch_device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    validation_sources = set(rng.choice(np.unique(source_ids), size=max(1, len(np.unique(source_ids)) // 5), replace=False))
    train_mask = np.asarray([sid not in validation_sources for sid in source_ids], dtype=bool)
    if not np.any(train_mask) or smoke or overfit_source_id is not None:
        train_mask = np.ones_like(source_ids, dtype=bool)
    val_mask = ~train_mask
    if not np.any(val_mask):
        val_mask = train_mask
    source_weights = _load_source_weights(data.root)
    fastest_sources = _fastest_valid_source_ids(data.root)
    sample_weights = np.asarray([source_weights.get(int(sid), 1.0) for sid in source_ids[train_mask]], dtype=np.float64)
    if fastest_sources and fastest_valid_weight != 1.0:
        fastest_mask = np.asarray([int(sid) in fastest_sources for sid in source_ids[train_mask]], dtype=bool)
        sample_weights[fastest_mask] *= max(0.0, fastest_valid_weight)
    if late_progress_weight != 1.0:
        late_mask = progress_m[train_mask] >= late_progress_threshold_m
        sample_weights[late_mask] *= max(0.0, late_progress_weight)
    probabilities = sample_weights / sample_weights.sum() if sample_weights.sum() > 0.0 else None

    train_obs = torch.as_tensor(normalizer.normalize_np(obs[train_mask]), dtype=torch.float32, device=torch_device)
    train_actions = torch.as_tensor(actions[train_mask], dtype=torch.float32, device=torch_device)
    val_obs = torch.as_tensor(normalizer.normalize_np(obs[val_mask]), dtype=torch.float32, device=torch_device)
    val_actions = torch.as_tensor(actions[val_mask], dtype=torch.float32, device=torch_device)
    action_weights = torch.as_tensor(
        [1.0, max(0.0, brake_action_weight), 1.0],
        dtype=torch.float32,
        device=torch_device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "best_policy.pt"
    best_val = float("inf")
    max_batches = 8 if smoke else max(1, int(np.ceil(train_obs.shape[0] / batch_size)))
    config = {
        "dataset": str(data.root),
        "dataset_manifest": data.manifest,
        "observation_profile": data.manifest.get("observation_profile"),
        "observation_dim": int(obs.shape[1]),
        "observation_feature_schema": data.manifest.get("observation_feature_schema"),
        "physics_model": data.manifest.get("physics_model", "v1"),
        "physics_version": data.manifest.get("physics_version"),
        "physics_calibration_id": data.manifest.get("physics_calibration_id"),
        "device": str(torch_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_size": hidden_size,
        "smoke": smoke,
        "overfit_source_id": overfit_source_id,
        "late_progress_threshold_m": late_progress_threshold_m,
        "late_progress_weight": late_progress_weight,
        "fastest_valid_weight": fastest_valid_weight,
        "fastest_valid_source_ids": sorted(fastest_sources),
        "brake_action_weight": brake_action_weight,
        "control_mode": actor.control_mode,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(epochs):
            actor.train()
            losses: list[float] = []
            per_action: list[np.ndarray] = []
            for _ in range(max_batches):
                idx_np = _batch_indices(
                    rng,
                    count=train_obs.shape[0],
                    batch_size=batch_size,
                    probabilities=probabilities,
                )
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=torch_device)
                pred = actor.deterministic_action(train_obs[idx])
                target = train_actions[idx]
                loss = bc_action_loss(pred, target, action_weights=action_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                per_action.append(torch.mean(torch.abs(pred.detach() - target), dim=0).cpu().numpy())
            actor.eval()
            with torch.no_grad():
                val_pred = actor.deterministic_action(val_obs)
                val_loss = float(bc_action_loss(val_pred, val_actions, action_weights=action_weights).detach().cpu())
                val_abs = torch.mean(torch.abs(val_pred - val_actions), dim=0).detach().cpu().numpy()
            mean_action_error = np.mean(np.asarray(per_action), axis=0)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "train_abs_error_throttle": float(mean_action_error[0]),
                "train_abs_error_brake": float(mean_action_error[1]),
                "train_abs_error_steer": float(mean_action_error[2]),
                "val_abs_error_throttle": float(val_abs[0]),
                "val_abs_error_brake": float(val_abs[1]),
                "val_abs_error_steer": float(val_abs[2]),
                "simultaneous_throttle_brake_rate": float(np.mean((actions[:, 0] > 0.05) & (actions[:, 1] > 0.05))),
            }
            metrics_file.write(json.dumps(row) + "\n")
            checkpoint_path = output_dir / "checkpoints" / f"bc_epoch_{epoch:04d}.pt"
            metadata = {
                "stage": "bc",
                "epoch": epoch,
                "metrics": row,
                "config": config,
                "physics_model": config["physics_model"],
                "physics_version": config["physics_version"],
                "physics_calibration_id": config["physics_calibration_id"],
            }
            save_policy_checkpoint(checkpoint_path, actor=actor, normalizer=normalizer, metadata=metadata)
            if val_loss <= best_val:
                best_val = val_loss
                save_policy_checkpoint(best_path, actor=actor, normalizer=normalizer, metadata=metadata)
    final_path = output_dir / "final_policy.pt"
    save_policy_checkpoint(
        final_path,
        actor=actor,
        normalizer=normalizer,
        metadata={
            "stage": "bc",
            "config": config,
            "physics_model": config["physics_model"],
            "physics_version": config["physics_version"],
            "physics_calibration_id": config["physics_calibration_id"],
        },
    )
    print(f"bc_train_complete best_policy={best_path} best_val_loss={best_val:.6f}")
    return best_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior cloning from an ES transition dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=str(LEARNED_DIR / "v1-bc"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit-source-id", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--late-progress-threshold-m", type=float, default=4800.0)
    parser.add_argument("--late-progress-weight", type=float, default=1.0)
    parser.add_argument("--fastest-valid-weight", type=float, default=1.0)
    parser.add_argument("--brake-action-weight", type=float, default=1.0)
    parser.add_argument("--control-mode", choices=("independent", "dominance"), default="independent")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_bc(
        dataset=Path(args.dataset),
        output_dir=Path(args.output_dir),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed,
        smoke=args.smoke,
        overfit_source_id=args.overfit_source_id,
        resume=Path(args.resume) if args.resume else None,
        late_progress_threshold_m=args.late_progress_threshold_m,
        late_progress_weight=args.late_progress_weight,
        fastest_valid_weight=args.fastest_valid_weight,
        brake_action_weight=args.brake_action_weight,
        control_mode=args.control_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
