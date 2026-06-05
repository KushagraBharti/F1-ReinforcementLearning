# F1 Reinforcement Learning

Top-down 2D Monza simulator, replay system, PPO training harness, evolutionary search platform, and learned-policy pipeline.

The project started as a Gymnasium/SB3 PPO driving experiment. PPO infrastructure works, but the strongest result now comes from verified GPU evolutionary search data distilled and fine-tuned into a learned SAC policy, with CPU `MonzaSim` as the promotion oracle.

<p align="center">
  <img src="./pygame-window-gen49-fastest-89s-all-150-cars-slow.gif" alt="89 second evolved Monza lap replay" width="900" />
</p>

<p align="center">
  <img src="./pygame-window-fastest-81s-lap-2000x150.gif" alt="CPU-verified 81.2 second Monza lap replay" width="900" />
</p>

## Current Result

Promoted learned-policy lap:

- Policy archive: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-learned-20260605.tar.zst`
- Original policy path inside archive: `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt`
- Training path: verified ES source dataset -> BC checkpoint -> project-native PyTorch SAC checkpoint
- Dataset archive: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-datasets-20260605.tar.zst`
- BC checkpoint path inside archive: `artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt`
- SAC checkpoint path inside archive: `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt`
- CPU oracle eval path inside archive: `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\eval_summary.json`
- CPU `MonzaSim` result: `3/3` valid normal-start laps, fastest `79.750s`
- Local replay telemetry: `artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry`
- Local 1000-car policy swarm: `artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000`

Best evolved source lap used by the learned-policy path:

- Source archive: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-runs-20260605.tar.zst`
- Source run path inside archive: `artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910`
- Fastest CPU-replayed valid source lap: `79.750s`
- Source candidate: generation `0`, candidate `334`

Previous broad evolved lap:

- Full source archive: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\gpu-es-2000x150-full-12000-traces-20260605.tar.zst`
- Original source run path inside run archive: `artifacts\runs\gpu-speed-speedprofiles-2000x150-25k-20260605`
- Search scale: `2000` population x `150` generations = `300,000` candidates
- Backend: GPU fused evolutionary search
- Max steps: `25000`
- Target termination: disabled
- Fastest selected valid lap: `81.233s`
- CPU-verified selected candidate: generation `143`, candidate `1864`
- CPU verification result: `lap_complete`
- CPU/GPU reason mismatches: `0`
- CPU/GPU valid-lap mismatches: `0`
- Local curated replay telemetry: `artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150`

Earlier milestones were a CPU-evolution result around `89s` and the broad GPU ES `81.233s` result.

## What Exists

- `src/f1rl/sim.py`: shared CPU simulator and oracle.
- `src/f1rl/env.py`: Gymnasium environment for SB3 PPO.
- `src/f1rl/train.py`: CPU/Gym/SB3 PPO training entrypoint.
- `src/f1rl/gpu_ppo.py`: experimental custom GPU PPO path.
- `src/f1rl/evolution_search.py`: CPU/GPU evolutionary search.
- `src/f1rl/evolution_postcheck.py`: deferred CPU replay/rerank verification for GPU search.
- `src/f1rl/evolution_ladder.py`: repeatable search ladder runner.
- `src/f1rl/es_dataset.py`: CPU-replayed ES transition dataset export and QA.
- `src/f1rl/bc_train.py`: behavior cloning for learned policy actors.
- `src/f1rl/sac_train.py`: project-native PyTorch SAC fine-tuning.
- `src/f1rl/policy_eval.py`: deterministic CPU learned-policy oracle eval.
- `src/f1rl/policy_swarm_eval.py`: checkpoint swarm telemetry export.
- `src/f1rl/replay.py`: pygame/headless replay for telemetry.
- `src/f1rl/telemetry.py`: telemetry schema, summaries, and loading.

Legacy diagnostics still exist, but are not the main path:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`

## Active Direction

The current practical path is:

1. Use GPU evolutionary search to discover fast valid laps.
2. CPU-verify/rerank selected winners.
3. Preserve replayable selected telemetry.
4. Export a broad verified transition dataset from ES traces.
5. Train a learned neural policy from that data with behavior cloning.
6. Fine-tune with custom PyTorch SAC.
7. Optionally inject the learned actor back into ES as a smart candidate source.

PPO is still real and useful as infrastructure, but blind PPO is not the current lead path.

Detailed learned-policy goal docs:

- `docs/CurrentPhysicsLearnedPolicyPlan.md`
- `docs/PhysicsV2LearnedPolicyPlan.md`

## Setup

```powershell
cd "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning"
uv sync --active --all-extras --all-packages
```

Hardware check:

```powershell
uv run --no-sync f1-hardware-check --json --warp-smoke
```

## Core Commands

CPU simulator / manual driving:

```powershell
uv run --no-sync python -m f1rl.manual
```

Replay selected telemetry:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry"
```

Headless replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry" --headless --limit 1
```

Curated CPU ES replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\full-generation-reel-20260605\cpu-es-150x60" --by-generation --generation-limit 150 --sort score --speed 2
```

Curated GPU ES replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150" --by-generation --generation-limit 150 --sort score --speed 2
```

The reproduction commands below expect the archived `artifacts\runs`, `artifacts\datasets`, and `artifacts\learned` folders to be restored from `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605` first. The replay commands above work from the current local highlight set.

Learned-policy dataset export:

```powershell
uv run --no-sync python -m f1rl.es_dataset export --run-dir artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910 --output-dir artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 --observation-profile learned_policy_v1 --physics-model v1 --max-candidates 16 --max-per-generation 16 --balanced-buckets
```

BC -> SAC -> CPU oracle:

```powershell
uv run --no-sync python -m f1rl.bc_train --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 --output-dir artifacts\learned\v1-bc-lpv1-sac79p750-broad16 --device cuda --resume artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt --control-mode dominance
uv run --no-sync python -m f1rl.sac_train --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 --bc-checkpoint artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1 --device cuda --control-mode dominance
uv run --no-sync python -m f1rl.policy_eval --policy artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval --episodes 3 --max-steps 25000 --observation-profile learned_policy_v1 --write-telemetry gzip
```

Policy swarm replay:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000" --by-checkpoint --speed 1
```

Evolution search help:

```powershell
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_postcheck --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

CPU evolution smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

GPU fused production shape:

```powershell
uv run --no-sync python -m f1rl.evolution_search --backend gpu --gpu-engine fused --gpu-run-mode production --gpu-device cuda --gpu-dtype float32 --gpu-fast-geometry local_window --gpu-collision-mode exact_grid --gpu-cpu-replay-top-k 0 --gpu-telemetry-mode none --start-progress-m 0 --start-speed-kph 80 --target-progress-m 5793 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 25000 --population 1000 --generations 75 --elite-count 96 --random-immigrants 96 --top-k 24 --workers 1 --genome-type controller --scoring-profiles fast_valid_lap,time_attack,lap_pace,fast_frontier,frontier_fast,farthest_distance,clean_distance,max_progress --progress-every-generation --output-dir artifacts\runs\example-gpu-es
```

Deferred CPU postcheck/rerank:

```powershell
uv run --no-sync python -m f1rl.evolution_postcheck artifacts\runs\example-gpu-es --top-k 8 --candidate-pool-size 96 --cpu-rerank --workers 8 --telemetry-compression gzip
```

SB3 PPO smoke:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 16 --seed 91 --n-envs 1 --max-steps 8 --device cpu --checkpoint-every 0 --eval-every 0 --telemetry none --run-name validation-sb3-smoke --action-mode continuous --observation-profile base --n-steps 8 --batch-size 8 --n-epochs 1
```

GPU PPO smoke:

```powershell
uv run --no-sync python -m f1rl.gpu_ppo --device cuda --require-gpu --dtype float32 --output-dir artifacts\runs\gpu-ppo-smoke --timesteps 16 --n-envs 2 --n-steps 4 --batch-size 4 --n-epochs 1 --hidden-size 32 --max-steps 8 --observation-profile base --start-speed-kph 60 --target-progress-m 6 --terminate-at-target --cpu-eval-episodes 1 --cpu-eval-max-steps 8
```

## Validation

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

Use focused checks first when changing a small area, then broaden to the full set above before commit.

## Repository Layout

- `src/f1rl/`: simulator, learning, search, replay, telemetry, and tooling code.
- `tests/`: unit, parity, backend, replay, and CLI smoke tests.
- `tools/`: focused diagnostic scripts.
- `assets/`, `imgs/`: visual assets and track images.
- `archive/`: historical plans, old docs, old media, and legacy snapshots.
- `artifacts/`: ignored local artifact root.
- `artifacts/highlights/`: current local curated replay telemetry and GIFs.
- `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\`: compressed cold storage for bulk RL1 runs, datasets, checkpoints, and full GPU ES telemetry.

After the RL1 cleanup pass, the local repo intentionally keeps only curated highlight artifacts. New runs should still write under `artifacts/runs/`, datasets under `artifacts/datasets/`, learned checkpoints under `artifacts/learned/`, calibration output under `artifacts/calibration/`, and FastF1 cache data under `artifacts/fastf1-cache/`; compress and offload bulk outputs when they are no longer actively being debugged.

Current local highlight storage:

| Set | Local path | Files | Size |
|---|---|---:|---:|
| CPU ES telemetry | `artifacts\highlights\full-generation-reel-20260605\cpu-es-150x60` | 907 | `0.43 GB` |
| GPU ES telemetry | `artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150` | 907 | `0.65 GB` |
| RL telemetry | `artifacts\highlights\learned-policy-replays-20260605\telemetry` | 1005 | `1.55 GB` |
| All local highlights | `artifacts\highlights` | 2830 | `2.70 GB` |

Important D archives:

- Full GPU ES `12k` traces: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\gpu-es-2000x150-full-12000-traces-20260605.tar.zst`
- Current reduced local highlights: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst`
- Bulk runs: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-runs-20260605.tar.zst`
- Learned checkpoints/evals: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-learned-20260605.tar.zst`
- Datasets: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-datasets-20260605.tar.zst`

Root markdown is intentionally minimal:

- `README.md`: public overview and commands.
- `Documentation.md`: concise live status.
- `AGENTS.md`: agent operating rules.

Older markdown plans and writeups are archived under `archive/repo-cleanup-20260605/`.

## Current Caveats

- The `79.750s` headline result is a learned SAC checkpoint verified by CPU `MonzaSim`, not a PPO result.
- The `79.750s` ES source remains search data; learned-policy promotion uses the SAC checkpoint and CPU eval summary above.
- CPU PPO has not completed a valid normal-start lap.
- GPU PPO is implemented and smoke-tested, not learning-proven.
- Raw GPU search winners are not trusted until CPU postcheck/rerank passes.
- Large run artifacts can be tens of GB; keep only compact selected telemetry in the repo.
