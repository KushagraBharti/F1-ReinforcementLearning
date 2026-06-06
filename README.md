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

Physics V2 is the active result. V1 remains the default simulator unless `--physics-model v2` is explicitly selected.

FastF1-calibrated V2 learned-policy lap:

- FastF1 threshold: `79.327s` from 2024 Italian GP Qualifying NOR lap 11.
- Local threshold summary: `artifacts\highlights\v2-fastf1-final-20260606\calibration\summary.json`
- Training path: CPU-reranked V2 GPU ES data -> independent-control BC -> project-native PyTorch SAC workflow with initial BC promotion preserved.
- CPU `MonzaSim` promotion result: `1/1` valid normal-start lap, `77.6833s`.
- Local promoted replay: `artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz`
- Local learned-policy swarm: `artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_swarm_1000`
- Local GIF: `artifacts\highlights\v2-fastf1-final-20260606\gifs\learned-policy-promotion-4x.gif`

Best V2 evolved source lap:

- Original source run in archive: `artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606`
- Staging: `1000x5 -> 1000x10 -> 1000x25 -> 1000x50`
- Max steps: `15000`
- Backend: GPU fused evolutionary search, CPU postcheck/rerank oracle
- Best CPU-verified selected lap: `77.6833s`
- Source candidate: generation `46`, candidate `724`
- Local replay: `artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz`
- Local GIF: `artifacts\highlights\v2-fastf1-final-20260606\gifs\gpu-es-cpu-rerank-best-4x.gif`

V2 bulk storage:

- Local final highlights: `artifacts\highlights\v2-fastf1-final-20260606`
- D-drive bulk archive: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`

Earlier V1 milestones were a CPU-evolution result around `89s`, a broad GPU ES `81.233s` result, and a promoted V1 learned-policy lap at `79.750s`; those artifacts are archived under `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605`.

## Physics V2 Handoff

Physics V2 is an explicit opt-in simulator model and is the current promoted learned-policy benchmark. V1 remains the default runtime path for backward-compatible commands unless `--physics-model v2` is explicitly selected.

Current V2 gate status:

- CPU V2 and PyTorch GPU V2 share the same initial parameter contract.
- Calibration id: `monza_2022_2024_fastf1_multilap_v2_manual_balance_fix`
- FastF1 reference report command: `uv run --no-sync python -m f1rl.calibration --json`
- FastF1 multi-lap fetch command: `uv run --no-sync python -m f1rl.fastf1_calibration fetch-multi --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\fastf1-multilap-20260605 --max-laps-per-driver 1`
- FastF1 helper command: `uv run --no-sync python -m f1rl.fastf1_calibration compare --physics-model v2 --multi-summary artifacts\calibration\fastf1-multilap-20260605\section_distribution_summary.json --output artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json`
- FastF1 calibration data now includes the checked-in 2024 Italian GP Qualifying VER lap plus `60` selected clean dry Q/FP2/FP3 laps from Monza `2024`, `2023`, and `2022`; raw car data, raw position data, processed telemetry, per-lap summaries, and section distributions are under `artifacts\calibration\fastf1-multilap-20260605`.
- OpenF1 cross-check command: `uv run --no-sync python -m f1rl.openf1_crosscheck --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\openf1-monza-crosscheck-20260605 --request-delay-s 1.0`
- OpenF1 artifact `artifacts\calibration\openf1-monza-crosscheck-20260605\manifest.json` is an independent sanity check only: `42` selected Monza 2024/2023 laps, `3` explicit 2022 skips, lap p50 `80.8025s`, max-speed p90 `349.0 kph`, speed-trap p90 `345.0 kph`.
- Current V2 report includes speed trace samples, braking zones, corner speed targets, acceleration/braking envelopes, curvature/lateral-g, gear/RPM plausibility, sustained-corner controlled-speed diagnostics, and a multi-lap sustained-section comparison. Current metrics from `artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json`: terminal-speed error `+3.0 kph`, acceleration trace p95 MAE about `1.15 m/s^2`, braking-zone distance MAE about `15.5m`, robust corner lateral-g margin min about `-0.45g`, robust margin mean about `+0.21g`, sustained-corner pass rate `0.333`, max sustained-corner p95 lateral error about `17.95m`, minimum sustained p75/p90 margins `-2.173/-2.811`, gear match rate `1.0`, mean absolute RPM error under `9`.
- Multi-lap sustained-section clusters are distance-based; current p50 ranges are `2424.9..2590.4m`, `2780.8..2885.5m`, `3952.9..4035.8m`, and `5052.7..5317.4m`.
- Scripted, benchmark, SB3 PPO smoke, and GPU PPO smoke entrypoints accept explicit `--physics-model v2` and write V2 metadata.
- Tiny V2 fused GPU parity smoke under the manual-approved balance retune wrote `artifacts\runs\v2-manual-balance-fix-gpu-parity-smoke-20260606` with v2.0.10 metadata, `gpu_parity_status=passed`, max CPU/GPU progress delta about `0.00020m`, and `0` reason/valid-lap mismatches.
- Tiny V2 persistent-controller parity smoke wrote `artifacts\runs\v2-recalibration-persistent-controller-parity-smoke-20260605` with `gpu_kernel_backend=warp_persistent_controller_open`, `gpu_parity_status=passed`, `0` CPU replay reason mismatches, and `0` valid-lap mismatches.
- The `127.183s` scripted lap is reclassified as a conservative V2 scripted smoke/debug baseline only, not `scripted_threshold`; it is archived with the other superseded V2 bulk artifacts.
- Any V2 `1000x5` or `1000x10` ES artifacts created before this recalibration are exploratory/pre-recalibration only and must not be used for target, dataset, threshold, or promotion decisions.
- Post-retune validation for the current manual-approved gate passed: manual headless V2 ghost section smoke at `artifacts\runs\manual-headless-20260606-012833-seed7-300975100`, flying-start smoke at `artifacts\runs\manual-headless-20260606-012839-seed7-378636300`, `ruff check .`, `pyright src/f1rl`, full `pytest -q`, `f1-hardware-check --json --warp-smoke`, focused V2 pytest, and V2 fused GPU parity smoke.
- V2 manual handoff checklist: `artifacts\runs\qc-20260606-012846\manual_qc_checklist.md`.
- Manual gate failed on v2.0.3 through v2.0.9 for handling/longitudinal feel. The current v2.0.10 keeps the manually approved high-speed cornering balance, slightly eases low-speed grip from v2.0.9, raises acceleration, and softens braking per manual feedback. The user approved the manual check on 2026-06-06.
- V2 `scripted_threshold` is established from the fastest selected FastF1 Monza calibration lap, not from the slower simulator reference controller: `79.327s` from 2024 Monza Qualifying NOR lap 11, McLaren, SOFT, track status `1`. The local summary copy is `artifacts\highlights\v2-fastf1-final-20260606\calibration\summary.json`; the original calibration tree is archived under `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`. The staged GPU ES target is CPU-verified `<=86.327s` (`scripted_threshold + 7s`).
- CPU V2 reference-control baseline `artifacts\runs\reference-control-20260606-022941-seed7-987680600` replay-loaded as valid (`116.2167s`, zero collisions/off-track), but it is not the threshold; the original run is now in the D-drive archive.
- V2 GPU ES reached the CPU-verified target on 2026-06-06. Run `artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606` was staged from `1000x5` through `1000x50` with `max_steps=15000`; selected CPU-rerank parity passed with `0` selected reason/valid-lap mismatches. The best CPU-verified selected V2 lap is `77.6833s` from generation `46`, candidate `724`; local replay copy: `artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz`. The broader raw GPU pool still failed parity (`13` reason mismatches, `12` valid-lap mismatches in the final `512` pool), so raw GPU winners remain proposal data only.
- V2 dataset export from CPU-replayed V2 trajectories wrote `artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606`: `16` source candidates, `50128` transitions, `8` valid laps, fastest source lap `77.65s`, physics `physics_v2.0.10-fastf1-manual-balance-fix`.
- V2 learned-policy target is met by the saved SAC workflow checkpoint `artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\best_policy.pt`, now archived on `D:\`. CPU V2 promotion eval wrote `1/1` valid lap at `77.6833s`, under the `79.327s` FastF1 threshold; local replay copy: `artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz`.
- Final V2 highlights are local under `artifacts\highlights\v2-fastf1-final-20260606`: `1006` replay traces total (`5` CPU-reranked GPU ES traces, `1` promoted learned-policy trace, `1000` deterministic best-policy swarm entries), plus exact-pygame `4x` GIFs at `artifacts\highlights\v2-fastf1-final-20260606\gifs\gpu-es-cpu-rerank-best-4x.gif` and `artifacts\highlights\v2-fastf1-final-20260606\gifs\learned-policy-promotion-4x.gif`.
- Manual mode now swaps left/right steering keys in the renderer to match the observed on-screen car response, and the HUD is right-aligned in free screen space rather than covering the left-side driving line.

Manual V2 test command:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
```

Focused sustained-corner checks:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_02 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_03 --start-section-lead-in-m 120
```

These starts align the FastF1 ghost to the same reference section instead of starting the ghost at lap time zero.

Manual section telemetry can be summarized with QC:

```powershell
uv run --no-sync python -m f1rl.qc --telemetry artifacts\runs\manual-headless-20260606-012833-seed7-300975100 --output-dir artifacts\runs --max-telemetry-files 1
```

Latest QC report before threshold recording: `artifacts\runs\qc-20260606-012846`; its automated `manual_gate` block still keeps `scripted_threshold` unset because it predates the FastF1 threshold correction. The user manually approved v2.0.10 on 2026-06-06. V2 highlight curation, exact-pygame GIF export, bulk artifact offload, final validation, commit, and push are complete in `1c09889`.

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

Detailed achievement reports:

- `RL1-Achieved.md`
- `PhysicsV2-Achieved.md`

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
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion" --speed 4
```

Headless replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion" --headless --limit 1
```

Curated V2 GPU ES replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank" --sort score --speed 4
```

V2 exact-pygame GIF export:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz" --export-gif "artifacts\highlights\v2-fastf1-final-20260606\gifs\learned-policy-promotion-4x.gif" --speed 4 --gif-fps 12
```

The reproduction commands below expect archived bulk folders to be restored first. For V2, restore from `D:\f1-rl-artifacts\archives\physics-v2-20260606`; for legacy V1/RL1, restore from `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605`. The replay commands above work from the current local V2 highlight set.

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
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_swarm_1000" --by-checkpoint --speed 4
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
- `artifacts/highlights/v2-fastf1-final-20260606/`: current local V2 final replay telemetry, calibration summaries, manifest, and GIFs.
- `D:\f1-rl-artifacts\archives\physics-v2-20260606\`: compressed cold storage for V2 bulk runs, datasets, checkpoints, calibration trees, and superseded highlights.
- `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\`: compressed cold storage for bulk RL1 runs, datasets, checkpoints, and full GPU ES telemetry.

After the V2 cleanup pass, the local repo intentionally keeps only curated V2 final highlight artifacts. New runs should still write under `artifacts/runs/`, datasets under `artifacts/datasets/`, learned checkpoints under `artifacts/learned/`, calibration output under `artifacts/calibration/`, and FastF1 cache data under `artifacts/fastf1-cache/`; compress and offload bulk outputs when they are no longer actively being debugged.

Current local highlight storage:

| Set | Local path | Files | Size |
|---|---|---:|---:|
| V2 final ES/learned telemetry, calibration summaries, and GIFs | `artifacts\highlights\v2-fastf1-final-20260606` | 1017 | `1.97 GB` |

Important D archives:

- Full GPU ES `12k` traces: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\gpu-es-2000x150-full-12000-traces-20260605.tar.zst`
- Current reduced local highlights: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst`
- Bulk runs: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-runs-20260605.tar.zst`
- Learned checkpoints/evals: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-learned-20260605.tar.zst`
- Datasets: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-datasets-20260605.tar.zst`
- V2 bulk archive excluding final local highlights: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`

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
