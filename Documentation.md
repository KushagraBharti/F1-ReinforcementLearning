# Documentation

Last updated: 2026-06-05.

This is the concise live status file. Historical prompts, long plans, and transcript-derived notes are archived under `archive/`.

## Scoreboard

Best current learned-policy result:

- SAC policy path inside archive: `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt`
- Dataset manifest path inside archive: `artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\dataset_manifest.json`
- BC checkpoint path inside archive: `artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt`
- SAC promotion eval path inside archive: `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\eval_summary.json`
- CPU oracle result: `3/3` valid normal-start laps, fastest `79.750s`, average `79.750s`
- Local replay telemetry: `artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry`
- Local policy swarm: `artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000`

Best evolved source result:

- Run path inside archive: `artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910`
- Backend: GPU fused evolutionary search with learned actor injection
- CPU-replayed source lap: `79.750s`
- Source candidate: generation `0`, candidate `334`

Previous broad evolved result:

- Run: `artifacts\runs\gpu-speed-speedprofiles-2000x150-25k-20260605`
- Scale: `2000 x 150 = 300,000` candidates
- Backend: GPU fused evolutionary search
- Max steps: `25000`
- Target termination: disabled
- Fastest selected valid lap: `81.233s`
- CPU-verified selected candidate: generation `143`, candidate `1864`
- CPU verification reason: `lap_complete`
- Reason mismatches: `0`
- Valid-lap mismatches: `0`
- Max final-progress delta in selected CPU verification: about `0.0306m`
- Archived selected telemetry path: `artifacts\runs\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top`
- Local curated replay telemetry: `artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150`

Previous milestone:

- CPU evolutionary search reached about `89s`.
- The README keeps both the 89s GIF and the newer 81.2s GIF.

PPO status:

- CPU/Gym/SB3 PPO exists and works mechanically.
- GPU PPO exists as a separate experimental path.
- No PPO policy is currently the headline result.
- The old blind PPO loop should not be resumed as the primary path.

## Current Architecture

Core runtime:

`track assets -> track geometry -> simulator -> telemetry -> replay/eval -> learning/search`

Main modules:

- `src/f1rl/sim.py`: CPU simulator and promotion oracle.
- `src/f1rl/env.py`: Gymnasium wrapper.
- `src/f1rl/train.py`: SB3 PPO training path.
- `src/f1rl/gpu_ppo.py`: custom GPU PPO experiments.
- `src/f1rl/evolution_search.py`: CPU/GPU evolutionary search.
- `src/f1rl/evolution_postcheck.py`: CPU postcheck/rerank for GPU search.
- `src/f1rl/evolution_backend.py`: shared CPU/GPU evolution backend routing.
- `src/f1rl/gpu_batch.py`: batched GPU simulation.
- `src/f1rl/gpu_fused_warp.py`: Warp fused GPU kernels.
- `src/f1rl/es_dataset.py`: CPU-replayed ES transition dataset export and report.
- `src/f1rl/learned_policy.py`: shared learned policy actor/checkpoint code.
- `src/f1rl/bc_train.py`: behavior cloning trainer.
- `src/f1rl/sac_train.py`: project-native PyTorch SAC trainer.
- `src/f1rl/policy_eval.py`: deterministic CPU learned-policy eval.
- `src/f1rl/policy_swarm_eval.py`: grouped checkpoint swarm export.
- `src/f1rl/replay.py`: visual/headless replay.
- `src/f1rl/telemetry.py`: telemetry schema and summaries.

Legacy diagnostics:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`

They remain available, but they are not the main search path.

## Current Method

The practical loop is:

1. Run GPU evolutionary search in speed-first production mode.
2. Keep search-time telemetry compact.
3. CPU-rerank/postcheck selected candidates.
4. Generate replayable selected telemetry only for verified winners.
5. Inspect telemetry and replay.
6. Export fixed-profile transition datasets under `artifacts/datasets/`.
7. Train BC and SAC on CUDA where possible.
8. Promote only CPU-verified learned-policy checkpoints.
9. Use actor injection only as source discovery; ES/search laps are not learned-policy success.

Raw GPU winners are proposal data. CPU-verified selected winners are trusted promotion data.

## GPU ES Contract

The production GPU search path uses:

- `--backend gpu`
- `--gpu-engine fused`
- `--gpu-run-mode production`
- `--gpu-device cuda`
- `--gpu-dtype float32`
- `--gpu-fast-geometry local_window`
- `--gpu-collision-mode exact_grid`
- `--gpu-cpu-replay-top-k 0` for raw speed-first search
- deferred `f1rl.evolution_postcheck` for CPU verification/rerank

Deferred postcheck supports:

- candidate pool selection;
- CPU rerank;
- duplicate replay-key caching;
- parallel compact CPU pool replay;
- incremental pool progress files;
- full telemetry only for final selected winners.

Recent verified postcheck examples:

- `artifacts\runs\gpu-speed-deferred-rerank-1000x75-25k-20260605`
  - candidate pool: `96`
  - unique CPU replays: `16`
  - cache hits / duplicate skips: `80`
  - selected winners: `8`
  - final selected parity: passed
  - reason mismatches: `0`
  - valid-lap mismatches: `0`

- `artifacts\runs\gpu-speed-speedprofiles-2000x150-25k-20260605`
  - selected summary-top CPU verification: passed
  - selected winner: generation `143`, candidate `1864`
  - selected replay telemetry written under `selected_telemetry_summary_top`

## Learned-Policy Path

The current v1 learned-policy path now uses:

1. `learned_policy_v1`: fixed-order observation profile recorded in manifests and checkpoints.
2. `f1rl.es_dataset`: CPU replay export with source metadata, dedupe, manifest, and report.
3. `f1rl.bc_train`: BC actor initialization from verified ES controls.
4. `f1rl.sac_train`: project-native SAC with ES replay-buffer prefill and CPU eval cadence.
5. `f1rl.policy_eval`: CPU `MonzaSim` promotion oracle.
6. `f1rl.policy_swarm_eval`: replayable 1000-car checkpoint groups.
7. Actor injection back into ES after learned-policy eval works.

For iteration, single-source datasets can be used to preserve a promising line. For broader attempts, export more valid, near-valid, and diverse ES candidates with source metadata and dedupe.

Detailed goal docs:

- `docs/CurrentPhysicsLearnedPolicyPlan.md`: current physics v1 ES data -> transition dataset -> BC -> SAC -> learned policy.
- `docs/PhysicsV2LearnedPolicyPlan.md`: FastF1-calibrated physics v2 -> GPU ES v2 -> v2 dataset -> SAC learned policy v2.

## Important Commands

Replay the current best selected telemetry:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry"
```

Headless replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry" --headless --limit 1
```

Replay the promoted policy swarm:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000" --by-checkpoint --speed 1
```

Replay curated CPU ES:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\full-generation-reel-20260605\cpu-es-150x60" --by-generation --generation-limit 150 --sort score --speed 2
```

Replay curated GPU ES:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150" --by-generation --generation-limit 150 --sort score --speed 2
```

CPU evolution smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

Deferred CPU postcheck example:

```powershell
uv run --no-sync python -m f1rl.evolution_postcheck artifacts\runs\example-gpu-es --top-k 8 --candidate-pool-size 96 --cpu-rerank --workers 8 --telemetry-compression gzip
```

## Validation

Standard full checks:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

Before launching large search, also run a small CLI smoke and a replay-load check for any selected telemetry that will be used in docs or demos.

## Storage And Artifacts

Keep root clean:

- Do not place large run artifacts in repo root.
- Put active experiment outputs under `artifacts/runs/`.
- Put transition datasets under `artifacts/datasets/`.
- Put learned-policy checkpoints and evals under `artifacts/learned/`.
- Put calibration output and FastF1 cache data under `artifacts/calibration/` and `artifacts/fastf1-cache/`.
- Keep only small demo media in root when directly referenced by README.
- Archive old root documents under `archive/`.

Current local artifact policy after RL1 cleanup:

- Keep curated replay telemetry and GIFs local under `artifacts\highlights`.
- Keep bulk runs, datasets, checkpoints, and old artifacts compressed on `D:`.
- Keep the full `12k` GPU ES replay set on `D:` only; local GPU ES is the reduced `6 x 150 = 900` trace stratified set.

Current local highlight split:

| Set | Files | Size |
|---|---:|---:|
| CPU ES telemetry | 907 | `0.43 GB` |
| GPU ES telemetry, reduced local stratified set | 907 | `0.65 GB` |
| RL telemetry | 1005 | `1.55 GB` |
| All local highlights, including GIFs | 2830 | `2.70 GB` |

Important archive directory:

`D:\f1-rl-artifacts\archives\rl1-postgoal-20260605`

Important verified archives:

- `gpu-es-2000x150-full-12000-traces-20260605.tar.zst`: full GPU ES replay set, `13.04 GB`.
- `artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst`: current local highlight mirror, `1.10 GB`.
- `artifacts-runs-20260605.tar.zst`: bulk run artifacts, `8.06 GB`.
- `artifacts-learned-20260605.tar.zst`: learned checkpoints/evals, `4.90 GB`.
- `artifacts-datasets-20260605.tar.zst`: transition datasets, `0.14 GB`.

## Root Documentation Policy

Root markdown is intentionally limited to:

- `README.md`
- `Documentation.md`
- `AGENTS.md`

Archived root markdown from the previous planning phase lives under:

`archive/repo-cleanup-20260605/docs/`

Archived root media that is no longer referenced by README lives under:

`archive/repo-cleanup-20260605/media/`

Do not restore long transcript-style status logs to `Documentation.md`.
