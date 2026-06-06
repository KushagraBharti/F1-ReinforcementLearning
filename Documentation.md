# Documentation

Last updated: 2026-06-06.

This is the technical companion to the high-level `README.md`. It keeps live status, commands, validation notes, storage layout, and exact artifact paths. Historical prompts, long plans, and transcript-derived notes are archived under `archive/`.

## Scoreboard

Best current learned-policy result:

- Physics model: explicit `v2`; V1 remains the default unless selected.
- FastF1 threshold: `79.327s` from 2024 Italian GP Qualifying NOR lap 11.
- CPU oracle result: `1/1` valid normal-start lap, `77.6833s`.
- Local replay telemetry: `artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz`
- Local policy swarm: `artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_swarm_1000`
- Local GIF: `artifacts\highlights\physics2\gpu-rl\gifs\learned-policy-promotion-4x.gif`

Best current evolved source result:

- Run path inside archive: `artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606`
- Backend: GPU fused evolutionary search with CPU postcheck/rerank
- Search staging: `1000x5 -> 1000x10 -> 1000x25 -> 1000x50`
- Max steps: `15000`
- CPU-verified selected lap: `77.6833s`
- Source candidate: generation `46`, candidate `724`
- Local replay telemetry: `artifacts\highlights\physics2\gpu-es\telemetry\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz`
- Local GIF: `artifacts\highlights\physics2\gpu-es\gifs\gpu-es-cpu-rerank-best-4x.gif`

Storage:

- Current local highlights root: `artifacts\highlights`
- Physics 1 CPU ES: `artifacts\highlights\physics1\cpu-es`
- Physics 1 GPU ES: `artifacts\highlights\physics1\gpu-es`
- Physics 1 GPU RL: `artifacts\highlights\physics1\gpu-rl`
- Physics 2 GPU ES: `artifacts\highlights\physics2\gpu-es`
- Physics 2 GPU RL: `artifacts\highlights\physics2\gpu-rl`
- V2 bulk archive: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`
- Full V2 achievement report: `archive/docs/PhysicsV2-Achieved.md`

Previous V1 milestones:

- CPU evolutionary search reached about `89s`.
- Broad V1 GPU ES reached `81.233s`.
- Promoted V1 learned policy reached `79.750s`.
- Legacy V1/RL1 artifacts are archived under `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605`.

PPO status:

- CPU/Gym/SB3 PPO exists and works mechanically.
- GPU PPO exists as a separate experimental path.
- No PPO policy is currently the headline result.
- The old blind PPO loop should not be resumed as the primary path.

Physics V2 handoff status:

- `physics_model=v1` remains the default simulator behavior.
- `physics_model=v2` is now explicit in `SimConfig`, telemetry, scripted runs, benchmark runs, SB3/GPU PPO smoke metadata, evolution summaries, selected telemetry manifests, postcheck summaries, dataset manifests, policy evals, and policy swarm manifests.
- CPU V2 includes a FastF1 Monza-calibrated contract id: `monza_2022_2024_fastf1_multilap_v2_manual_balance_fix`.
- FastF1 calibration tools include offline `summarize`, `summarize-multi`, and `compare` commands plus live `fetch`/`fetch-multi` paths. The current multi-lap dataset is `artifacts\calibration\fastf1-multilap-20260605`, with manifest `manifest.json` and section distributions `section_distribution_summary.json`.
- The multi-lap FastF1 calibration dataset contains `60` selected clean dry laps from Monza `2024`, `2023`, and `2022` Q/FP2/FP3 for VER/NOR/PIA/LEC/SAI/HAM/RUS where available. It keeps raw `get_car_data()` and `get_pos_data()` CSVs plus processed `get_telemetry().add_distance()` CSVs for each selected lap. There were `5` rejected/skipped entries and `0` session errors.
- OpenF1 is now an independent sanity-check path, not the primary calibration source. Current artifact: `artifacts\calibration\openf1-monza-crosscheck-20260605\manifest.json` with `42` selected Monza 2024/2023 Q/FP2/FP3 laps and `3` explicit 2022 skips because OpenF1 returned 404 for those sessions. Summary: lap p50 `80.8025s`, max-speed p90 `349.0 kph`, speed-trap p90 `345.0 kph`.
- The single checked-in reference lap remains 2024 Italian GP Qualifying VER fastest lap: `79.662s`, `5745.669m`, max speed `348.0 kph`, mean speed `259.914 kph`, p10/p50/p90 speed `142.0 / 274.0 / 336.19 kph`, gear range `2..8`; it is no longer the only calibration target.
- Current V2 report path is `artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json`. Current metrics: terminal-speed error `+3.0 kph`, acceleration trace p95 MAE about `1.15 m/s^2`, braking-zone distance MAE about `15.5m`, robust corner lateral-g margin min about `-0.45g`, robust margin mean about `+0.21g`, sustained-corner reference control pass rate `0.333`, max sustained-corner p95 lateral error about `17.95m`, minimum sustained p75/p90 margins `-2.173/-2.811`, gear match rate `1.0`, mean absolute RPM error under `9`.
- Multi-lap sustained-section clusters are now distance-based. Current p50 start/end and speed/lateral-g distributions: section 01 `2424.9..2590.4m`, speed p50/p90 `215.3/222.9 kph`, lateral-g p90 p50/p90 `5.15/5.85`; section 02 `2780.8..2885.5m`, `198.3/205.1 kph`, `4.74/5.29g`; section 03 `3952.9..4035.8m`, `216.0/249.8 kph`, `3.73/4.51g`; section 04 `5052.7..5317.4m`, `228.9/235.7 kph`, `5.35/6.15g`.
- The focused sustained-corner diagnostic now starts before each section and reports reference-speed and controlled-speed runs at `150/180/200/220/230/240/250 kph`, including lateral error, heading error, steering saturation, actual/reference curvature, lateral-g, slip angles, front/rear lateral force, throttle/brake, tire saturation, and track-limit/off-track state.
- PyTorch GPU V2 and Warp fused V2 parity tests pass under the recalibrated contract.
- Tiny V2 fused GPU parity smoke under the manual-approved balance retune wrote `artifacts\runs\v2-manual-balance-fix-gpu-parity-smoke-20260606` with v2.0.10 metadata, `gpu_parity_status=passed`, max CPU/GPU progress delta about `0.00020m`, and `0` reason/valid-lap mismatches.
- Tiny V2 persistent-controller parity smoke wrote `artifacts\runs\v2-recalibration-persistent-controller-parity-smoke-20260605` with `gpu_kernel_backend=warp_persistent_controller_open`, `gpu_parity_status=passed`, `0` CPU replay reason mismatches, and `0` valid-lap mismatches.
- `tools/tests/test_v2_metadata_contracts.py` now exercises V2 metadata across CPU evolution summaries/checkpoints/bridge/selected telemetry, CPU postcheck summaries/manifests/rows, ES dataset export, BC/SAC policy checkpoints, policy eval manifests, policy swarm manifests, loaded telemetry rows, and the V1-labeled transfer export guard. `tools/tests/test_policy_io.py` also covers V2 PPO eval config reconstruction from `run_metadata.json`.
- The `127.183s` scripted lap is a conservative V2 scripted smoke/debug baseline only, not `scripted_threshold`; it is archived with the other superseded V2 bulk artifacts.
- Any V2 `1000x5` or `1000x10` ES artifacts created before this recalibration are exploratory/pre-recalibration only and must not be used for target, dataset, threshold, or promotion decisions.
- Post-retune validation passed on 2026-06-06 for this gate: manual headless V2 ghost section smoke wrote `artifacts\runs\manual-headless-20260606-012833-seed7-300975100`, flying-start smoke wrote `artifacts\runs\manual-headless-20260606-012839-seed7-378636300`, `ruff check .`, `pyright src/f1rl`, full `pytest -q`, `f1-hardware-check --json --warp-smoke`, focused V2 pytest, and V2 fused GPU parity smoke passed.
- V2 manual handoff checklist: `artifacts\runs\qc-20260606-012846\manual_qc_checklist.md`.
- Manual V2 handoff is approved for v2.0.10 as of 2026-06-06. v2.0.3 through v2.0.9 failed or were superseded for handling/longitudinal feel. v2.0.10 keeps the manually approved high-speed cornering balance, slightly eases low-speed grip from v2.0.9, raises acceleration, and softens braking. Manual left/right key mapping is intentionally swapped in the renderer so the keyboard matches the on-screen car response, and the HUD is right-aligned in free screen space instead of covering the left-side driving line.
- V2 `scripted_threshold` is now established from the fastest selected FastF1 Monza calibration lap, not the old scripted debug lap or the slower simulator reference controller: `79.327s` from 2024 Monza Qualifying NOR lap 11, McLaren, SOFT, track status `1`. Local summary copy: `artifacts\highlights\physics2\calibration\summary.json`; original calibration tree: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`. The V2 ES target is CPU-verified `<=86.327s` (`scripted_threshold + 7s`).
- CPU V2 reference-control baseline `artifacts\runs\reference-control-20260606-022941-seed7-987680600` replay-loaded as valid (`116.2167s`, zero collisions/off-track), but it is not the threshold; the original run is in the D-drive archive.
- V2 staged GPU ES target is met. Run `artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606` was started as `1000x5`, resumed through `1000x50`, and kept `max_steps=15000`. Final CPU postcheck/rerank used `candidate_pool_size=512`; selected parity passed with `0` selected reason mismatches and `0` selected valid-lap mismatches. The trusted selected winner is generation `46`, candidate `724`, CPU-verified `77.6833s`; local replay copy: `artifacts\highlights\physics2\gpu-es\telemetry\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz`.
- Raw GPU proposals are still not promotion data: the final broad pool postcheck failed parity with `13` reason mismatches and `12` valid-lap mismatches. CPU postcheck/rerank remains mandatory.
- V2 dataset `artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606` was exported by CPU replay from V2 source candidates: `16` candidates, `50128` transitions, `8` valid laps, fastest source lap `77.65s`, mean valid lap `85.925s`, V2 metadata `physics_v2.0.10-fastf1-manual-balance-fix`.
- V2 learned-policy target is met. BC with `control-mode independent` over source `3` produced a checkpoint that CPU-evaluated at `77.6833s`. SAC now preserves and evaluates the initial BC policy before updates; conservative SAC output promoted at `77.6833s` under CPU V2, below the FastF1 threshold `79.327s`. Original checkpoints/evals are in the D-drive archive; local replay copy: `artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz`.
- Final V2 highlights are local under `artifacts\highlights\physics2`: `1006` replay traces total (`5` CPU-reranked GPU ES traces, `1` promoted learned-policy trace, `1000` deterministic best-policy swarm entries), plus exact-pygame `4x` GIFs at `artifacts\highlights\physics2\gpu-es\gifs\gpu-es-cpu-rerank-best-4x.gif` and `artifacts\highlights\physics2\gpu-rl\gifs\learned-policy-promotion-4x.gif`.

Manual V2 handoff command:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
```

Focused sustained-corner manual commands:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_02 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_03 --start-section-lead-in-m 120
```

`--start-section` aligns the FastF1 ghost to the matching reference time/distance. Manual reset reuses the same section start. Mid-lap telemetry summaries now report run-local `distance_traveled_m`; lap sectors already passed before the section start are left blank instead of producing bogus sector speeds. Latest section-start headless smoke: `artifacts\runs\manual-headless-20260606-012833-seed7-300975100`.

Manual sustained-corner telemetry can be summarized with QC:

```powershell
uv run --no-sync python -m f1rl.qc --telemetry artifacts\runs\manual-headless-20260606-012833-seed7-300975100 --output-dir artifacts\runs --max-telemetry-files 1
```

Latest QC path before threshold recording: `artifacts\runs\qc-20260606-012846`. Its automated `manual_gate` block reports `scripted_threshold_status=unset` because it was generated before the FastF1 threshold correction. The user manually approved v2.0.10 on 2026-06-06. The QC section-smoke diagnostic is metadata/checklist evidence only because the headless manual smoke runs straight; use the local copied report `artifacts\highlights\physics2\calibration\fastf1-v2-manual-balance-fix-20260606.json` for handling-balance metrics. V2 highlight curation, exact-pygame GIF export, bulk offload, final validation, commit, and push are complete in `1c09889`.

FastF1 threshold and CPU controller baseline after manual approval:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode control --physics-model v2 --steps 9000 --seed 7
uv run --no-sync f1-replay artifacts\runs\reference-control-20260606-022941-seed7-987680600\steps.jsonl --headless
```

The threshold source is FastF1 `79.327s` from `artifacts\highlights\physics2\calibration\summary.json` locally, with the original source preserved in the D-drive archive. The CPU controller baseline wrote `artifacts\runs\reference-control-20260606-022941-seed7-987680600`, replay-loaded headlessly, and records `lap_time_s=116.2167`; it is baseline evidence only. Earlier post-manual attempts are superseded failed diagnostics: `artifacts\runs\reference-control-20260606-014926-seed7-381937500` failed `off_track` at `14.0s`, and `artifacts\runs\scripted-20260606-014937-seed7-461099600` failed `collision` at `104.0s`; those original run artifacts are archived on `D:\`.

Final validation passed on 2026-06-06:

```powershell
uv run --no-sync pytest -q tools/tests/test_physics.py tools/tests/test_gpu_physics.py tools/tests/test_calibration.py tools/tests/test_telemetry.py
uv run --no-sync python -m f1rl.manual --headless --physics-model v2 --max-steps 3 --seed 1
uv run --no-sync python -m f1rl.calibration --json
uv run --no-sync python -m f1rl.fastf1_calibration summarize assets\reference\monza_2024_Q_VER_telemetry.csv --output artifacts\calibration\fastf1-summary-smoke.json
uv run --no-sync python -m f1rl.fastf1_calibration fetch-multi --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\fastf1-multilap-20260605 --max-laps-per-driver 1
uv run --no-sync python -m f1rl.fastf1_calibration summarize-multi --output-dir artifacts\calibration\fastf1-multilap-20260605
uv run --no-sync python -m f1rl.fastf1_calibration compare --physics-model v2 --multi-summary artifacts\calibration\fastf1-multilap-20260605\section_distribution_summary.json --output artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json
uv run --no-sync python -m f1rl.openf1_crosscheck --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\openf1-monza-crosscheck-20260605 --request-delay-s 1.0
uv run --no-sync python -m f1rl.scripted --physics-model v2 --steps 3 --no-telemetry
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 1 --max-steps 3 --physics-model v2 --telemetry none
uv run --no-sync python -m f1rl.train --physics-model v2 --timesteps 8 --seed 9 --n-envs 1 --max-steps 8 --device cpu --checkpoint-every 8 --eval-every 0 --telemetry none --run-name v2-train-cli-smoke --n-steps 4 --batch-size 4 --n-epochs 1
uv run --no-sync python -m f1rl.gpu_ppo --output-dir artifacts\runs\v2-gpu-ppo-cli-smoke --device cpu --physics-model v2 --timesteps 8 --n-envs 2 --n-steps 4 --batch-size 4 --n-epochs 1 --hidden-size 32 --max-steps 8 --observation-profile base --start-speed-kph 60 --target-progress-m 6 --terminate-at-target --cpu-eval-episodes 1 --cpu-eval-max-steps 8
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-evolution-smoke-gear-rpm-cpu --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-evolution-smoke-gear-rpm-gpu-eager --backend gpu --gpu-engine eager --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-evolution-smoke-gear-rpm-gpu-fused --backend gpu --gpu-engine fused --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --gpu-collision-mode exact_grid --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-evolution-smoke-gear-rpm-gpu-fused-controller --backend gpu --gpu-engine fused --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --gpu-collision-mode exact_grid --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 1 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type controller --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.replay artifacts\runs\v2-evolution-smoke-gear-rpm-gpu-fused\selected_telemetry --headless --limit 1
uv run --no-sync python -m f1rl.replay artifacts\runs\v2-evolution-smoke-gear-rpm-gpu-fused-controller\selected_telemetry --headless --limit 1
uv run --no-sync pytest -q tools/tests/test_physics.py tools/tests/test_gpu_physics.py tools/tests/test_calibration.py tools/tests/test_gpu_evolution_backend.py
uv run --no-sync pytest -q tools/tests/test_calibration.py tools/tests/test_physics.py tools/tests/test_gpu_physics.py
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-recalibration-gpu-parity-smoke-20260605 --backend gpu --gpu-engine fused --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --gpu-collision-mode exact_grid --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-manual-balance-fix-gpu-parity-smoke-20260606 --backend gpu --gpu-engine fused --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --gpu-collision-mode exact_grid --physics-model v2 --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\v2-recalibration-persistent-controller-parity-smoke-20260605 --backend gpu --gpu-engine fused --gpu-run-mode parity --gpu-device cuda --gpu-dtype float32 --gpu-static-batch-size 4 --gpu-cpu-replay-top-k 2 --gpu-collision-mode exact_grid --physics-model v2 --start-progress-m 500 --start-speed-kph 60 --target-progress-m 506 --no-target-termination --action-set straight --observation-profile base --max-steps 8 --population 4 --generations 1 --elite-count 1 --random-immigrants 0 --top-k 2 --workers 1 --genome-type controller --scoring-profiles max_progress,clean_exit --progress-every-generation --telemetry-compression gzip
uv run --no-sync python -m f1rl.replay artifacts\runs\v2-recalibration-gpu-parity-smoke-20260605\selected_telemetry --headless --limit 1
uv run --no-sync python -m f1rl.replay artifacts\runs\v2-recalibration-persistent-controller-parity-smoke-20260605\selected_telemetry --headless --limit 1
uv run --no-sync python -m f1rl.replay artifacts\highlights\physics2\gpu-es\telemetry --headless --limit 1
uv run --no-sync pytest -q tools/tests/test_v2_metadata_contracts.py
uv run --no-sync pytest -q tools/tests/test_scripted_replay.py tools/tests/test_gpu_ppo.py tools/tests/test_benchmark.py tools/tests/test_policy_train_smoke.py
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

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

- `archive/docs/CurrentPhysicsLearnedPolicyPlan.md`: current physics v1 ES data -> transition dataset -> BC -> SAC -> learned policy.
- `archive/docs/PhysicsV2LearnedPolicyPlan.md`: FastF1-calibrated physics v2 -> GPU ES v2 -> v2 dataset -> SAC learned policy v2.

Detailed achievement reports:

- `archive/docs/RL1-Achieved.md`: V1 learned-policy achievement.
- `archive/docs/PhysicsV2-Achieved.md`: post-RL1 V2 physics, GPU ES, and learned-policy achievement.

## Important Commands

Replay the current best selected telemetry:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion" --speed 4
```

Headless replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion" --headless --limit 1
```

Replay the promoted policy swarm:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_swarm_1000" --by-checkpoint --speed 4
```

Replay curated V2 GPU ES selected traces:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-es\telemetry" --sort score --speed 4
```

Export an exact-pygame 4x GIF:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz" --export-gif "artifacts\highlights\physics2\gpu-rl\gifs\learned-policy-promotion-4x.gif" --speed 4 --gif-fps 12
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
- Keep small demo media under `assets/`.
- Archive long-form plans and achievement reports under `archive/docs/`.

Current local artifact policy after final highlight restore:

- Keep curated replay telemetry, manifests, and GIFs local under the final five-bucket layout in `artifacts\highlights`.
- Keep bulk V2 runs, datasets, checkpoints, calibration trees, superseded highlights, and old artifacts compressed on `D:`.
- Keep the full 12,000-trace Physics 1 GPU ES replay set on `D:` unless explicitly restored; the local Physics 1 GPU ES bucket is the reduced stratified replay set.

Current local highlight split:

| Set | Files | Size |
|---|---:|---:|
| Physics 1 CPU ES | 908 | `0.45 GiB` |
| Physics 1 GPU ES, reduced stratified local set | 909 | `0.68 GiB` |
| Physics 1 GPU RL | 1011 | `1.57 GiB` |
| Physics 2 GPU ES | 7 | `0.01 GiB` |
| Physics 2 GPU RL | 1005 | `1.96 GiB` |
| Physics 2 calibration summaries | 4 | `<0.01 GiB` |

Important archive directory:

`D:\f1-rl-artifacts\archives\physics-v2-20260606`

Important verified archives:

- `gpu-es-2000x150-full-12000-traces-20260605.tar.zst`: full GPU ES replay set, `13.04 GB`.
- `artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst`: current local highlight mirror, `1.10 GB`.
- `artifacts-runs-20260605.tar.zst`: bulk run artifacts, `8.06 GB`.
- `artifacts-learned-20260605.tar.zst`: learned checkpoints/evals, `4.90 GB`.
- `artifacts-datasets-20260605.tar.zst`: transition datasets, `0.14 GB`.
- `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`: V2 bulk archive excluding the current local final highlight tree.

## Root Documentation Policy

Root markdown is intentionally limited to:

- `README.md`
- `Documentation.md`

Archived long-form docs and plans live under:

`archive/docs/`

Archived root media that is no longer referenced by README lives under:

`assets/gifs/`

Do not restore long transcript-style status logs to `Documentation.md`.
