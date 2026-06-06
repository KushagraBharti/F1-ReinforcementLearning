# Physics V2 Achieved: FastF1-Calibrated Physics, GPU ES, And Learned Policy

Last updated: 2026-06-06.

This document is the detailed Physics V2 achievement report for the work completed after `RL1-Achieved.md`.

RL1 proved that the project could turn CPU-verified evolutionary-search telemetry into a real learned policy under the original `v1` physics. Physics V2 extended that result into a new benchmark category:

```text
FastF1 Monza references
  -> calibrated opt-in physics_v2 CPU oracle
  -> GPU physics_v2 parity
  -> staged GPU ES under physics_v2
  -> CPU-reranked V2 ES winners
  -> V2 transition dataset
  -> independent-control BC
  -> project-native PyTorch SAC
  -> CPU V2 learned-policy promotion
  -> replayable final highlights, GIFs, and archived bulk artifacts
```

The headline result is a saved learned SAC policy that completes a valid normal-start CPU `MonzaSim` lap under explicit `physics_model=v2` in `77.6833s`, against a FastF1-derived V2 threshold of `79.327s`.

This is a simulator result under a calibrated top-down 2D model. It is not a claim that the project is physically equivalent to real Formula 1 or that the learned policy is faster than a real driver in the real world.

## Executive Summary

Physics V2 is complete for the goal in `docs/PhysicsV2LearnedPolicyPlan.md`.

The final V2 physics contract is:

```text
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
```

The final FastF1 benchmark threshold is:

```text
scripted_threshold_s: 79.327
source: 2024 Italian GP Qualifying, NOR lap 11
driver: NOR
team: McLaren
compound: SOFT
```

The final learned-policy promotion result is:

```text
policy: artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\best_policy.pt
policy stage: sac
physics_model: v2
normal_start: true
deterministic: true
episodes: 1
valid_lap_count: 1
valid_lap_rate: 1.0
fastest_valid_lap_s: 77.68333333333334
average_valid_lap_s: 77.68333333333334
steps: 4661
termination_reason: lap_complete
best_progress_m: 5793.000137343199
final_progress_m: 5793.000137343199
final_speed_kph: 270.83024012629465
```

The final trusted V2 GPU ES source result is:

```text
run: artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606
backend: GPU fused evolution search
verification: CPU postcheck/rerank
max_steps: 15000
staging: 1000x5 -> 1000x10 -> 1000x25 -> 1000x50
best selected generation: 46
best selected candidate: 724
best selected seed: 46000743
best CPU-verified lap: 77.68333333333334
selected parity mismatches: 0 reason, 0 valid-lap
```

The final local replay/highlight root is:

```text
artifacts\highlights\v2-fastf1-final-20260606
```

The final local highlights contain:

```text
total_replay_traces: 1006
gpu_es_selected_cpu_rerank traces: 5
learned_policy_promotion traces: 1
learned_policy_swarm_1000 traces: 1000
files: 1017
size: 2117641804 bytes, about 1.97 GiB
```

The final bulk archive is:

```text
D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst
```

Its compressed size is:

```text
1565771809 bytes, about 1.46 GiB
```

The final git commit is:

```text
1c09889 Complete Physics V2 learned policy pipeline
```

It was pushed to:

```text
origin/main
```

The push advanced main as:

```text
2f2c5ac..1c09889  main -> main
```

## Relationship To RL1

`RL1-Achieved.md` is the prior historical report.

RL1 established the first learned-policy success:

- physics model: `v1`;
- promotion oracle: CPU `MonzaSim`;
- policy stage: SAC;
- fastest promoted lap: `79.750s`;
- learned-policy path: verified ES data -> BC -> SAC;
- key implementation: project-native PyTorch learning pipeline instead of relying on blind PPO progress.

Physics V2 starts after that point.

The V2 work did not replace or silently mutate RL1. It created an explicit second benchmark category:

- `v1 ES`;
- `v1 learned policy`;
- `v2 ES`;
- `v2 learned policy`.

The project now treats those categories separately. A V1 lap and a V2 lap are not interchangeable because the physics contract is different.

## Done Criteria

The V2 goal was not just "make a new physics file" or "train any policy." The goal required the whole pipeline to close:

1. Preserve V1 as the default simulator behavior.
2. Make V2 opt-in through explicit metadata.
3. Implement V2 in CPU and GPU paths from the same contract.
4. Calibrate V2 against FastF1 Monza telemetry.
5. Add independent OpenF1 sanity-check evidence.
6. Add CPU/GPU parity tests and smokes.
7. Add manual-mode handoff and human feel approval.
8. Establish the V2 threshold from FastF1 after manual approval.
9. Run staged GPU ES under V2.
10. CPU-rerank GPU ES winners under CPU V2.
11. Reach `scripted_threshold + 7s` with CPU-verified V2 ES.
12. Export V2-only datasets from CPU-replayed V2 trajectories.
13. Train BC and SAC under V2.
14. Promote only CPU V2 normal-start learned-policy evals.
15. Reach the FastF1 threshold with a learned V2 policy.
16. Create replayable V2 highlights and exact-pygame GIFs.
17. Archive/offload non-highlight bulk artifacts.
18. Update docs.
19. Run validation.
20. Commit and push.

All of those are complete as of commit `1c09889`.

## Proof Artifacts

### Final Highlight Manifest

Root manifest:

```text
artifacts\highlights\v2-fastf1-final-20260606\manifest.json
```

Important fields:

```text
kind: f1rl_v2_final_highlight_manifest
created_at: 2026-06-06T09:08:51.383714+00:00
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
total_replay_traces: 1006
```

### FastF1 Threshold Copy

Local threshold summary:

```text
artifacts\highlights\v2-fastf1-final-20260606\calibration\summary.json
```

Fields:

```text
lap_time_s: 79.327
distance_m: 5753.878867527226
max_speed_kph: 347.0
mean_speed_kph: 262.7248992137443
```

The manifest records the source as:

```text
artifacts\calibration\fastf1-multilap-20260605\monza_2024_Q\NOR_lap011\summary.json
```

That original bulk path has been archived to `D:\`.

### Final Calibration Report Copy

Local copied V2 calibration report:

```text
artifacts\highlights\v2-fastf1-final-20260606\calibration\fastf1-v2-manual-balance-fix-20260606.json
```

Key final V2 metadata inside `physics_models.v2`:

```text
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
```

Key error terms:

```text
max_speed_error_kph: 3.0
speed_trace_accel_p95_mae_mps2: 1.1486445889475458
mean_abs_braking_zone_distance_error_m: 15.45139760529571
max_abs_braking_zone_distance_error_m: 35.77155334652457
min_robust_corner_lateral_g_margin: -0.4516393807640777
mean_robust_corner_lateral_g_margin: 0.21234589368020992
gear_match_rate: 1.0
mean_abs_rpm_error: 8.863300498605765
sustained_corner_reference_control_pass_rate: 0.3333333333333333
max_sustained_corner_reference_p95_abs_lateral_error_m: 17.94986390116031
```

### GPU ES Highlight Manifest

Local manifest:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\manifest.json
```

Top selected trace:

```text
rank: 0
selection_reason: cpu_rerank_top_score
generation: 46
candidate_index: 724
seed: 46000743
termination_reason: lap_complete
best_progress_m: 5793.000137343198
final_progress_m: 5793.000137343198
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
```

Local top selected telemetry:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz
```

### Learned Policy Promotion Manifest

Local manifest:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\manifest.json
```

Important fields:

```text
kind: f1rl_v2_final_policy_promotion_highlight
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
scripted_threshold_s: 79.327
promotion_lap_s: 77.68333333333334
trace_count: 1
valid_lap: true
```

Local promoted telemetry:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz
```

### Learned Policy Swarm Manifest

Local swarm manifest:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_swarm_1000\manifest.json
```

Important fields:

```text
kind: f1rl_policy_swarm_manifest
backend: cpu_monzasim
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
deterministic: true
trace_count: 1000
```

The final manifest notes that the deterministic best-policy swarm uses hardlinked replay traces when start noise is zero.

### GIF Exports

GPU ES GIF:

```text
artifacts\highlights\v2-fastf1-final-20260606\gifs\gpu-es-cpu-rerank-best-4x.gif
```

Properties:

```text
speed: 4.0
fps: 12
frames: 234
bytes: 808380
source_trace: postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz
```

Learned policy GIF:

```text
artifacts\highlights\v2-fastf1-final-20260606\gifs\learned-policy-promotion-4x.gif
```

Properties:

```text
speed: 4.0
fps: 12
frames: 234
bytes: 807023
source_trace: policy-eval-episode-000-steps.jsonl.gz
```

Both GIFs were exported through the pygame replay renderer path, not through an approximate custom renderer.

## Final Replay Commands

Replay the learned policy promotion:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion" --speed 4
```

Replay the learned-policy 1000-car swarm:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_swarm_1000" --by-checkpoint --speed 4
```

Replay the curated CPU-reranked V2 GPU ES traces:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank" --sort score --speed 4
```

Headless learned-policy replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion" --headless --limit 1
```

Headless V2 ES replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank" --headless --limit 1
```

Export the learned-policy GIF again:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz" --export-gif "artifacts\highlights\v2-fastf1-final-20260606\gifs\learned-policy-promotion-4x.gif" --speed 4 --gif-fps 12
```

Export the GPU ES GIF again:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz" --export-gif "artifacts\highlights\v2-fastf1-final-20260606\gifs\gpu-es-cpu-rerank-best-4x.gif" --speed 4 --gif-fps 12
```

## Storage State

Local `artifacts` is intentionally small after final cleanup.

Current local `artifacts` contains:

```text
artifacts\highlights
```

Final local V2 highlight root:

```text
artifacts\highlights\v2-fastf1-final-20260606
```

Local highlight size:

```text
1017 files
2117641804 bytes
about 1.97 GiB
```

Bulk archive:

```text
D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst
```

Bulk archive compressed size:

```text
1565771809 bytes
about 1.46 GiB
```

The local highlights are replayable without restoring the D-drive archive.

The D-drive archive preserves the original large run folders, calibration trees, datasets, checkpoints, evals, and non-highlight bulk artifacts that were cleaned from local storage.

## Final Validation

The worker reported the following final validation as passing before commit and push:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

Reported validation details:

```text
ruff: passed
pyright: 0 errors, 0 warnings, 0 informations
pytest: passed, with one expected skip and existing SB3 VecMonitor warnings
CUDA: available
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Torch: 2.10.0+cu128
Warp: 1.14.0
Warp torch interop smoke: passed
max_abs_error: 0.0
```

The final documentation cleanup can be validated with smaller doc/replay checks because it does not change code:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion" --headless --limit 1
uv run --no-sync python -m f1rl.replay "artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank" --headless --limit 1
```

## Phase 0: Where V2 Started

Before Physics V2, the project had:

- a working top-down Monza simulator;
- CPU PPO infrastructure;
- experimental GPU PPO infrastructure;
- CPU evolutionary search;
- GPU evolutionary search;
- CPU postcheck/rerank for GPU search;
- replay and telemetry;
- V1 learned-policy training through BC and SAC;
- a promoted V1 learned-policy result at `79.750s`.

The V1 learned-policy success was useful, but it was still operating under the original simplified physics. The next goal was to create a stronger physics contract before pushing learning further.

The V2 work started with these principles:

- do not silently corrupt V1;
- do not compare V1 and V2 as the same benchmark;
- do not trust GPU winners until CPU replay confirms them;
- do not use raw ES traces as learned-policy success;
- keep telemetry replayable;
- keep storage controlled;
- treat CPU `MonzaSim` as the promotion oracle;
- use FastF1 as the calibration anchor.

## Phase 1: Explicit Physics Versioning

The first important architectural move was making physics explicit.

V1 stayed the default.

V2 became opt-in.

This matters because the old project had many entrypoints:

- manual mode;
- scripted mode;
- benchmark mode;
- CPU PPO;
- GPU PPO;
- CPU ES;
- GPU ES;
- postcheck;
- dataset export;
- BC;
- SAC;
- policy eval;
- policy swarm;
- replay.

If V2 had been implemented as a silent global mutation, old artifacts and new artifacts would have become impossible to compare. Instead, the work threaded explicit metadata through each relevant object and artifact.

The recurring metadata fields are:

```text
physics_model
physics_version
physics_calibration_id
```

For V2, those fields resolve to:

```text
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
physics_calibration_id: monza_2022_2024_fastf1_multilap_v2_manual_balance_fix
```

This metadata appears in:

- `SimConfig`;
- step telemetry rows;
- episode summaries;
- scripted run outputs;
- benchmark outputs;
- SB3 PPO smoke metadata;
- GPU PPO smoke metadata;
- evolution summaries;
- selected telemetry manifests;
- postcheck manifests;
- postcheck summaries;
- dataset manifests;
- dataset source rows;
- BC checkpoints;
- SAC checkpoints;
- policy eval summaries;
- policy eval manifests;
- policy swarm manifests;
- loaded telemetry rows.

The tests now enforce much of this through `tests/test_v2_metadata_contracts.py`.

## Phase 2: CPU Physics V2

CPU V2 lives mainly in:

```text
src\f1rl\config.py
src\f1rl\physics.py
src\f1rl\sim.py
```

The CPU path added a new `PhysicsV2Params` contract and made `SimConfig` carry:

```text
physics_model
physics_version
physics_calibration_id
physics_v2
```

V2 CPU physics added explicit tire and vehicle-state concepts that V1 did not model in the same way.

New diagnostic state includes:

- gear;
- rpm;
- front slip angle;
- rear slip angle;
- front normal load;
- rear normal load;
- front lateral force;
- rear lateral force;
- tire saturation;
- surface friction;
- wheel-lock tendency.

The main V2 CPU concepts are:

- speed-sensitive steering authority;
- front/rear lateral tire force;
- load transfer;
- aero downforce;
- front aero balance;
- mechanical grip scaling;
- post-peak tire falloff;
- tire scrub drag;
- power-limited acceleration;
- brake-lock steering loss;
- automatic gear and RPM diagnostics.

The implementation split V1 and V2 paths rather than blending them:

- `apply_physics_v1`;
- `apply_physics_v2`;
- dispatch through `apply_physics`.

That split keeps V1 preservation testable.

## Phase 3: GPU Physics V2

V2 also had to exist in GPU paths because GPU ES is the lead search method.

GPU V2 touched:

```text
src\f1rl\gpu_types.py
src\f1rl\gpu_batch.py
src\f1rl\gpu_physics.py
src\f1rl\gpu_fused_warp.py
src\f1rl\evolution_backend.py
```

`GpuCarParams` gained the V2 parameter fields. `gpu_car_params_from_cpu` now accepts the selected physics model and transfers the V2 contract into GPU code.

The PyTorch GPU path implements a V2 batch physics step.

The Warp fused path implements the same V2 concepts in kernels used by production GPU ES.

The persistent-controller Warp branch also supports V2, so controller-genome and fused replay paths remain available under the same physics label.

GPU V2 exists for speed, but CPU V2 remains the oracle.

That distinction is important:

- GPU proposals can be fast;
- GPU proposals can have numerical or collision-boundary mismatches;
- CPU replay decides promotion.

The final broad GPU pool still had parity mismatches:

```text
reason mismatches: 13
valid-lap mismatches: 12
pool size: 512
```

That does not invalidate the selected winner because selected CPU-reranked parity passed with:

```text
selected reason mismatches: 0
selected valid-lap mismatches: 0
```

It does reinforce the project rule: raw GPU winners are proposal data, not trusted results.

## Phase 4: FastF1 Calibration

V2 calibration started with a checked-in 2024 Monza FastF1 reference CSV:

```text
assets\reference\monza_2024_Q_VER_telemetry.csv
```

That source represents:

```text
Fast-F1 2024 Italian Grand Prix Qualifying VER fastest lap
lap_time_s: 79.662
distance_m: 5745.669358933476
min_speed_kph: 75.0
max_speed_kph: 348.0
mean_speed_kph: 259.91445380074884
p10_speed_kph: 142.0
p50_speed_kph: 274.22291253333333
p90_speed_kph: 336.23074976
lateral_g_p90: 3.2784794132687525
lateral_g_p95: 4.0949627805901985
```

Single-lap calibration was not enough. The project expanded the FastF1 data path to a multi-lap Monza reference set.

The multi-lap fetch command shape was:

```powershell
uv run --no-sync python -m f1rl.fastf1_calibration fetch-multi --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\fastf1-multilap-20260605 --max-laps-per-driver 1
```

The multi-lap dataset included:

- years: `2024`, `2023`, `2022`;
- sessions: `Q`, `FP2`, `FP3`;
- drivers: `VER`, `NOR`, `PIA`, `LEC`, `SAI`, `HAM`, `RUS`;
- selected clean dry laps: `60`;
- rejected/skipped entries: `5`;
- session errors: `0`.

For each selected lap, the tooling preserved:

- raw car data from `get_car_data()`;
- raw position data from `get_pos_data()`;
- processed telemetry from `get_telemetry().add_distance()`;
- per-lap summary JSON;
- multi-lap section distribution summary.

The calibration tooling lives in:

```text
src\f1rl\fastf1_calibration.py
src\f1rl\calibration.py
```

The commands include:

- `fetch`;
- `fetch-multi`;
- `summarize`;
- `summarize-multi`;
- `compare`.

The final comparison command shape was:

```powershell
uv run --no-sync python -m f1rl.fastf1_calibration compare --physics-model v2 --multi-summary artifacts\calibration\fastf1-multilap-20260605\section_distribution_summary.json --output artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json
```

## Phase 5: OpenF1 Cross-Check

OpenF1 was added as an independent sanity-check path, not as the primary calibration target.

The implementation lives in:

```text
src\f1rl\openf1_crosscheck.py
tests\test_openf1_crosscheck.py
```

The command shape was:

```powershell
uv run --no-sync python -m f1rl.openf1_crosscheck --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\openf1-monza-crosscheck-20260605 --request-delay-s 1.0
```

The cross-check recorded:

```text
selected laps: 42
explicit 2022 skips: 3
lap p50: 80.8025s
max-speed p90: 349.0 kph
speed-trap p90: 345.0 kph
```

OpenF1 was useful because it gave an outside check on lap-time and speed distributions. FastF1 stayed the main calibration source because the project already used FastF1 telemetry and position data more deeply.

## Phase 6: Manual Retuning

Manual retuning became the decisive gate.

The initial V2 calibration fixed some numbers but did not feel right in the simulator. The project therefore added a manual handoff loop before starting large ES or RL.

The user tested manual mode with a FastF1 ghost and section starts.

Manual mode gained:

- `--physics-model v2`;
- `--ghost-reference`;
- `--flying-start`;
- `--start-section sustained_corner_01`;
- `--start-section sustained_corner_02`;
- `--start-section sustained_corner_03`;
- `--start-section-lead-in-m`;
- `--start-progress-m`;
- `--start-speed-kph`;
- section-aligned ghost timing;
- reset behavior that reuses the same section start;
- run-local distance summaries for mid-lap starts.

Manual command:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
```

Focused sustained-corner commands:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_02 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_03 --start-section-lead-in-m 120
```

### v2.0.2: Understeer

The early FastF1-trace tune was too understeery.

Observed behavior:

- car washed wide in sustained corners;
- front end did not bite enough;
- reference-speed arcs were hard to hold;
- manual gate failed;
- ES/RL stayed blocked.

The investigation focused on:

- front/rear cornering stiffness;
- front/rear peak friction;
- aero front balance;
- steering speed sensitivity;
- max steer;
- steer response;
- load sensitivity;
- tire scrub drag;
- slip-angle peak;
- post-peak falloff.

### v2.0.3: Too Easy

The next major understeer fix overshot.

Manual feedback said the user could beat the "perfect" FastF1 ghost too easily. That was a clear failure because a keyboard/manual driver in the simplified sim should not casually match a `79s` FastF1-style reference lap.

The lesson was that fixing understeer by giving the car large front grip and forgiveness can make the model arcade-like.

The retune had to keep enough front bite while taking back excessive forgiveness.

### v2.0.4 Through v2.0.9: Conservative Retunes

The intermediate retunes tightened the car through:

- more steering-speed penalty;
- lower front peak grip;
- lower front stiffness from the too-easy tune;
- more tire scrub;
- more post-peak falloff;
- more load sensitivity;
- less rear-slip steering help;
- stricter low-speed mechanical grip;
- braking and acceleration adjustments.

The tuning loop repeatedly checked:

- calibration report metrics;
- sustained-corner reference control;
- controlled-speed section diagnostics;
- focused tests;
- GPU parity smoke;
- manual feel feedback.

Manual feedback evolved from:

- severe understeer;
- too easy;
- still too easy;
- go much harder;
- high-speed nearly right;
- low-speed too easy;
- acceleration/braking mismatch;
- braking nearly right, acceleration still off;
- final balance accepted.

### v2.0.10: Manual Balance Fix

The final manual-approved tune is:

```text
physics_v2.0.10-fastf1-manual-balance-fix
```

The final parameters include:

| Parameter | Value |
|---|---:|
| `max_steer_deg` | `20.0` |
| `steer_response` | `8.0` |
| `steering_speed_sensitivity` | `0.000445` |
| `front_weight_distribution` | `0.46` |
| `front_cornering_stiffness_n_per_rad` | `197000.0` |
| `rear_cornering_stiffness_n_per_rad` | `161000.0` |
| `front_peak_mu` | `3.28` |
| `rear_peak_mu` | `3.12` |
| `mechanical_grip_low_speed_scale` | `0.1925` |
| `mechanical_grip_high_speed_scale` | `0.70` |
| `mechanical_grip_transition_mps` | `80.0` |
| `slip_angle_peak_deg` | `9.5` |
| `rear_slip_steer_coupling` | `0.15` |
| `post_peak_falloff` | `0.270` |
| `load_sensitivity` | `0.065` |
| `aero_downforce_n_per_mps2` | `6.35` |
| `aero_balance_front` | `0.49` |
| `engine_power_w` | `792000.0` |
| `max_drive_g` | `2.42` |
| `max_brake_g` | `3.015` |
| `brake_bias_front` | `0.57` |
| `brake_lock_threshold` | `0.90` |
| `brake_lock_steer_loss` | `0.45` |
| `drag_coefficient` | `0.00082` |
| `rolling_resistance_mps2` | `0.23` |
| `max_speed_mps` | `97.5` |
| `tire_scrub_drag` | `1.20` |
| `surface_mu` | `1.0` |

Manual mode also received two practical usability fixes:

- left/right steering keys were swapped in the renderer to match observed on-screen car response;
- the HUD was moved to the right side so it did not cover the left-side driving line.

The user approved v2.0.10 on 2026-06-06.

## Phase 7: Threshold Correction

The project initially had a `127.183s` scripted debug lap under earlier V2 physics. That was invalidated as a threshold.

Why `127.183s` was invalid:

- it was a conservative debug/smoke lap;
- it was generated under superseded physics;
- it was far slower than real FastF1 reference laps;
- it did not represent the calibrated target pace;
- using it would make the learning goal too easy.

After manual approval, the threshold was set from the fastest selected FastF1 Monza calibration lap:

```text
79.327s
2024 Italian GP Qualifying
NOR lap 11
McLaren
SOFT
```

There was also a CPU V2 reference-control baseline:

```text
artifacts\runs\reference-control-20260606-022941-seed7-987680600
lap_time_s: 116.2167
status: valid
collisions: 0
off_track: 0
```

That baseline is useful evidence that the simulator reference controller can complete a valid lap. It is not the threshold because it is an internal controller, not the FastF1 benchmark.

The final target definitions were therefore:

```text
V2 ES target: CPU-verified <= 86.327s
V2 learned-policy target: CPU-verified <= 79.327s
```

The V2 ES winner and V2 learned policy both reached `77.6833s`, so both targets were met.

## Phase 8: Staged GPU ES

The V2 ES run was staged instead of jumping directly to a giant run.

The run root was:

```text
artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606
```

Staging:

```text
1000x5
1000x10
1000x25
1000x50
```

The run used:

```text
physics_model: v2
max_steps: 15000
backend: gpu
gpu_engine: fused
telemetry_compression: gzip
```

The `15000` step horizon mattered because lower horizons risked cutting off valid full-lap behavior under the V2 timing and start conditions.

The final CPU postcheck/rerank used:

```text
candidate_pool_size: 512
pool_postcheck_count: 23
cpu_rerank_clean_pool_count: 5
cpu_rerank_rejected_mismatch_count: 18
selected_reason_mismatches: 0
selected_valid_lap_mismatches: 0
```

The best trusted selected V2 ES lap was:

```text
generation: 46
candidate: 724
seed: 46000743
lap_time_s: 77.68333333333334
termination_reason: lap_complete
```

The selected trace is local:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\gpu_es_selected_cpu_rerank\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz
```

The important conceptual result is not just the `77.6833s` number. It is that the result came through the trusted path:

```text
GPU proposal -> CPU V2 replay -> CPU V2 rerank -> selected telemetry -> replay/GIF/highlight
```

Raw GPU proposals remain untrusted until that replay path passes.

## Phase 9: V2 Dataset Export

The V2 dataset came from CPU-replayed V2 source candidates, not mixed V1/V2 data.

Original dataset path:

```text
artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606
```

Dataset export command shape:

```powershell
uv run --no-sync python -m f1rl.es_dataset export --run-dir artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606 --selected-telemetry artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606\selected_telemetry\manifest.json --output-dir artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606 --observation-profile racing_v2 --physics-model v2 --max-candidates 16 --max-per-generation 16 --balanced-buckets
```

Dataset stats:

```text
dataset_id: v2-es-policy-dataset-fastf1-1000x50-20260606
postcheck_status: cpu_replayed_export
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
total_source_candidates: 16
total_transitions: 50128
valid_lap_count: 8
fastest_source_lap: 77.65
mean_valid_lap_time: 85.925
storage_size_bytes: 13535745
candidate_selection_counts_by_bucket:
  early_failure: 6
  valid_lap: 8
  mid_frontier: 2
terminal_reason_distribution:
  lap_complete: 8
  off_track: 8
```

The dataset used:

```text
observation_profile: racing_v2
observation_dim: 35
action_schema: throttle brake steer
normalization_strategy: dataset_mean_std
```

The dataset also exposed a key learning issue: simultaneous throttle/brake behavior mattered.

The initial dominance-control assumption was not good enough for the V2 ES source line. The learned policy needed to represent throttle and brake independently.

The dataset-level simultaneous throttle/brake rate was about `0.2713653`. The final source-3 independent BC metadata still recorded simultaneous throttle/brake behavior at about `0.1358661`, so the control-mode change is part of the achieved result rather than a cosmetic setting.

## Phase 10: Behavior Cloning

BC was trained on the V2 dataset with independent control.

Final BC source:

```text
artifacts\learned\v2-bc-fastf1-source3-independent-20260606\best_policy.pt
```

Command shape:

```powershell
uv run --no-sync python -m f1rl.bc_train --dataset artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606 --output-dir artifacts\learned\v2-bc-fastf1-source3-independent-20260606 --device cuda --epochs 120 --batch-size 1024 --control-mode independent --overfit-source-id 3 --seed 23
```

BC metadata:

```text
stage: bc
epoch: 115
physics_model: v2
physics_version: physics_v2.0.10-fastf1-manual-balance-fix
control_mode: independent
device: cuda
hidden_size: 256
epochs: 120
batch_size: 1024
lr: 0.0003
overfit_source_id: 3
```

BC metrics recorded in the promoted policy metadata:

```text
train_loss: 4.181449839961715e-05
val_loss: 3.854606984532438e-05
train_abs_error_throttle: 0.004085834138095379
train_abs_error_brake: 0.004997430834919214
train_abs_error_steer: 0.007003082428127527
val_abs_error_throttle: 0.004134438931941986
val_abs_error_brake: 0.004944815766066313
val_abs_error_steer: 0.0063115740194916725
simultaneous_throttle_brake_rate: 0.13586606567933032
```

The important BC change was not only "train longer." The important change was using an action representation compatible with the data. V2 source telemetry used meaningful independent throttle/brake behavior, so dominance control lost information.

BC CPU V2 eval command shape:

```powershell
uv run --no-sync python -m f1rl.policy_eval --policy artifacts\learned\v2-bc-fastf1-source3-independent-20260606\best_policy.pt --output-dir artifacts\learned\v2-bc-fastf1-source3-independent-20260606\cpu_eval --episodes 1 --deterministic --normal-start --observation-profile racing_v2 --physics-model v2 --write-telemetry gzip --max-steps 15000 --device cuda --seed 23
```

That eval reached the same `77.68333333333334s` lap under CPU V2.

## Phase 11: SAC Learned Policy

The final learned policy is a SAC checkpoint.

Original path:

```text
artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\best_policy.pt
```

Important SAC config:

```text
dataset: artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606
bc_checkpoint: artifacts\learned\v2-bc-fastf1-source3-independent-20260606\best_policy.pt
physics_model: v2
observation_profile: racing_v2
device: cuda
timesteps: 512
n_envs: 8
max_steps: 15000
batch_size: 1024
hidden_size: 256
lr: 0.0003
gamma: 0.99
tau: 0.005
bc_loss_weight: 10.0
rollout_deterministic: true
rollout_noise_std: 0.02
initial_alpha: 0.05
target_entropy: -3.0
freeze_alpha: true
control_mode: independent
rollout_backend: cpu_monzasim
```

SAC was changed to evaluate and preserve the initial BC policy at step `0`.

That mattered because the BC policy already closed the lap well. A fine-tuning loop should not destroy a good initialized policy before proving improvement. The final promoted SAC workflow checkpoint preserved the valid `77.6833s` CPU lap.

Final SAC command shape:

```powershell
uv run --no-sync python -m f1rl.sac_train --dataset artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606 --bc-checkpoint artifacts\learned\v2-bc-fastf1-source3-independent-20260606\best_policy.pt --output-dir artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606 --device cuda --timesteps 512 --n-envs 8 --max-steps 15000 --batch-size 1024 --control-mode independent --physics-model v2 --eval-every 512 --swarm-every 0 --swarm-size 0 --bc-loss-weight 10.0 --deterministic-rollout --rollout-noise-std 0.02 --freeze-alpha --initial-alpha 0.05 --seed 23
```

The step `0` eval was valid at `77.68333333333334s`. The later step `512` eval did not produce a valid lap, so preserving the initial BC policy as the best policy was the correct behavior for this workflow.

The final CPU promotion eval wrote:

```text
artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\promotion_cpu_eval
```

Local replay copy:

```text
artifacts\highlights\v2-fastf1-final-20260606\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz
```

Promotion result:

```text
valid_lap_count: 1
valid_lap_rate: 1.0
fastest_valid_lap_s: 77.68333333333334
average_valid_lap_s: 77.68333333333334
terminal_reason_counts:
  lap_complete: 1
```

Promotion eval command shape:

```powershell
uv run --no-sync python -m f1rl.policy_eval --policy artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\best_policy.pt --output-dir artifacts\learned\v2-sac-fastf1-independent-conservative-initialeval-20260606\promotion_cpu_eval --episodes 1 --deterministic --normal-start --observation-profile racing_v2 --physics-model v2 --write-telemetry gzip --max-steps 15000 --device cuda --seed 23
```

Margin to threshold:

```text
threshold_s: 79.327
promotion_lap_s: 77.68333333333334
margin_s: -1.643666666666661
```

This is the final V2 learned-policy achievement.

## Phase 12: Replay, GIF, And Highlight Curation

The final local highlights were curated into:

```text
artifacts\highlights\v2-fastf1-final-20260606
```

The three replay sets are:

| Set | Traces | Purpose |
|---|---:|---|
| `gpu_es_selected_cpu_rerank` | `5` | CPU-reranked V2 GPU ES selected candidates |
| `learned_policy_promotion` | `1` | promoted learned-policy CPU eval lap |
| `learned_policy_swarm_1000` | `1000` | deterministic best-policy swarm replay |

This creates three final visual/replay surfaces:

1. V2 GPU ES source lap.
2. V2 learned-policy promoted lap.
3. V2 learned-policy 1000-car swarm.

The two GIFs are intentionally the single best ES source lap and the single promoted learned-policy lap. They are not approximations; they come from replaying the actual telemetry through the pygame renderer path.

## Phase 13: Archiving And Cleanup

After V2 completion, local storage was cleaned.

The goal was:

- keep final highlights local and replayable;
- archive bulk runs/datasets/checkpoints/calibration trees to external `D:`;
- avoid leaving the repo with huge active artifact piles.

Final local:

```text
artifacts\highlights\v2-fastf1-final-20260606
```

Final archive:

```text
D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst
```

The archive name is explicit: it excludes the final local V2 highlights.

That means the local repo and D drive complement each other:

- local: curated replay/highlight set;
- D drive: original bulk artifact history.

## File-By-File Implementation Notes

The final commit touched `48` files.

### Root Docs

| File | Role |
|---|---|
| `README.md` | Current project entrypoint, scoreboard, V2 replay commands, storage layout |
| `Documentation.md` | Concise live status, current architecture, V2 handoff/result state |
| `docs\PhysicsV2LearnedPolicyPlan.md` | Goal plan and completion criteria for the V2 pipeline |

### Project Config

| File | Role |
|---|---|
| `pyproject.toml` | Adds calibration/OpenF1/FastF1-related package wiring and CLI/tooling support |

### Physics And Config

| File | Role |
|---|---|
| `src\f1rl\config.py` | Adds `PhysicsV2Params`, V2 metadata, V2 sim config support |
| `src\f1rl\physics.py` | Adds CPU V2 tire/load/aero/engine/brake/gear physics and dispatch |
| `src\f1rl\sim.py` | Carries physics metadata through the CPU simulator and telemetry |

### Calibration

| File | Role |
|---|---|
| `src\f1rl\calibration.py` | Adds FastF1 target extraction, V2 estimates, error terms, sustained-corner diagnostics, multi-lap comparisons |
| `src\f1rl\fastf1_calibration.py` | New FastF1 CLI for fetch, fetch-multi, summarize, summarize-multi, and compare |
| `src\f1rl\openf1_crosscheck.py` | New OpenF1 sanity-check CLI with raw JSON saving and summaries |
| `src\f1rl\qc.py` | Adds V2 manual/QC diagnostics and manual gate reporting |

### GPU Physics And Search

| File | Role |
|---|---|
| `src\f1rl\gpu_types.py` | Adds V2 fields to GPU parameter transfer |
| `src\f1rl\gpu_physics.py` | Adds PyTorch GPU V2 batch stepping |
| `src\f1rl\gpu_fused_warp.py` | Adds Warp fused V2 and persistent-controller V2 support |
| `src\f1rl\gpu_batch.py` | Carries V2 selection into batched GPU simulation |
| `src\f1rl\evolution_backend.py` | Keeps backend routing compatible with V2 |
| `src\f1rl\evolution_search.py` | Adds `--physics-model`, V2 metadata in summaries/manifests, compressed telemetry handling |
| `src\f1rl\evolution_postcheck.py` | Adds V2 CPU postcheck/rerank support and V2 metadata |

### Dataset And Learning

| File | Role |
|---|---|
| `src\f1rl\es_dataset.py` | Adds V2 dataset export, V1/V2 guardrails, metadata in manifests/source rows |
| `src\f1rl\bc_train.py` | Stores V2 metadata and supports independent control for V2 BC |
| `src\f1rl\sac_train.py` | Adds V2 metadata, dataset/physics compatibility checks, initial BC evaluation/preservation |
| `src\f1rl\policy_eval.py` | Adds V2 CPU eval support and V2 metadata in summaries/manifests |
| `src\f1rl\policy_swarm_eval.py` | Adds V2 policy swarm export and V2 metadata |
| `src\f1rl\policy_io.py` | Updates checkpoint loading/config reconstruction for V2 metadata |

### Runtime Entrypoints

| File | Role |
|---|---|
| `src\f1rl\manual.py` | Adds V2 manual mode, FastF1 ghost section starts, headless smoke metadata |
| `src\f1rl\reference_agent.py` | Adds V2 reference controller and ghost alignment helpers |
| `src\f1rl\scripted.py` | Adds V2 scripted run support and V2 metadata |
| `src\f1rl\benchmark.py` | Adds V2 benchmark support |
| `src\f1rl\train.py` | Adds V2 SB3 PPO smoke metadata/support |
| `src\f1rl\gpu_ppo.py` | Adds V2 GPU PPO smoke metadata/support |

### Replay, Rendering, Telemetry

| File | Role |
|---|---|
| `src\f1rl\render.py` | Moves HUD, fixes manual steering key mapping, supports exact GIF/replay rendering behavior |
| `src\f1rl\replay.py` | Adds replay/GIF handling used for final V2 highlights |
| `src\f1rl\telemetry.py` | Adds V2 telemetry fields, summary handling, replay compatibility |

### Tests

| File | Role |
|---|---|
| `tests\test_physics.py` | CPU V2 physics behavior tests |
| `tests\test_gpu_physics.py` | GPU V2 parity tests |
| `tests\test_calibration.py` | FastF1/V2 calibration tests |
| `tests\test_gpu_evolution_backend.py` | GPU evolution backend V2/parity tests |
| `tests\test_v2_metadata_contracts.py` | New broad V2 metadata contract tests |
| `tests\test_manual.py` | New manual/headless/ghost tests |
| `tests\test_openf1_crosscheck.py` | New OpenF1 cross-check tests with mocked request path |
| `tests\test_render.py` | New render/HUD/GIF-related tests |
| `tests\test_benchmark.py` | V2 benchmark metadata smoke |
| `tests\test_gpu_ppo.py` | V2 GPU PPO smoke metadata |
| `tests\test_policy_io.py` | V2 policy config reconstruction |
| `tests\test_policy_train_smoke.py` | V2 policy training smoke coverage |
| `tests\test_qc.py` | V2 QC/manual gate coverage |
| `tests\test_reference_agent.py` | V2 reference/ghost alignment coverage |
| `tests\test_scripted_replay.py` | V2 scripted/replay metadata coverage |

## Architecture After V2

The runtime path remains:

```text
track assets -> track geometry -> simulator -> telemetry -> replay/eval -> learning/search
```

The current strongest V2 path is:

```text
FastF1 calibration
  -> explicit physics_model=v2
  -> GPU ES proposals
  -> CPU V2 postcheck/rerank
  -> V2 selected telemetry
  -> V2 dataset export
  -> V2 BC
  -> V2 SAC
  -> CPU V2 policy eval
  -> replay/GIF/highlight
```

The old blind PPO micro-rung loop is still not the lead path.

PPO remains in the repo because:

- it is useful infrastructure;
- it proves Gym/SB3 compatibility;
- it can be used for controlled experiments;
- it is not the current headline result.

## Important Guardrails

### V1 Is Preserved

V1 remains the default. V2 is explicit.

Do not silently switch old commands or old artifacts to V2.

### CPU Is The Oracle

GPU search is a proposal generator.

CPU `MonzaSim` replay/eval is the promotion oracle.

### FastF1 Threshold Is The V2 Target

The final V2 threshold is `79.327s` from FastF1 multi-lap data.

Do not resurrect:

- the `127.183s` debug lap as a threshold;
- the `116.2167s` CPU reference-control lap as the threshold.

Those are useful diagnostics, not targets.

### V2 Data Must Stay V2

Do not train V2 policies from V1 ES telemetry unless the work is explicitly labeled as transfer learning and guarded separately.

The achieved V2 result uses V2 dataset metadata and V2 CPU replay.

### Replay Must Remain Real

The final GIFs and replay commands use the actual pygame replay renderer path.

Do not replace them with approximate custom plots when the goal is to see exactly what the pygame window would show.

### Storage Must Stay Controlled

Bulk artifacts are archived to `D:`.

Local artifacts should remain curated highlights only unless new active work explicitly needs a local run.

## What The Final Result Does And Does Not Mean

It does mean:

- V2 exists as an explicit opt-in physics model.
- V2 has CPU and GPU implementations.
- V2 has FastF1 and OpenF1 calibration evidence.
- V2 passed manual handoff.
- V2 GPU ES can find CPU-verified laps under the FastF1 threshold.
- V2 ES data can train a learned SAC policy.
- The final learned policy completes a CPU V2 normal-start lap in `77.6833s`.
- The final replays and GIFs are local and viewable.
- Bulk artifacts are preserved externally.
- The final V2 pipeline is committed and pushed.

It does not mean:

- the top-down model is a full real F1 simulator;
- the learned policy is physically faster than real Norris/Verstappen at Monza;
- raw GPU candidates can be trusted without CPU replay;
- the V2 physics should be mutated casually after threshold establishment;
- PPO became the lead method.

## Reproduction Notes

The local repo after cleanup does not contain all original bulk folders. If a command refers to:

```text
artifacts\runs
artifacts\datasets
artifacts\learned
artifacts\calibration
```

outside the final highlights tree, restore the bulk archive first:

```text
D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst
```

The final replay commands do not require restoration because the curated telemetry is local.

## Final Status

Physics V2 is complete.

V2 GPU ES target:

```text
target: <= 86.327s
achieved: 77.6833s
status: met
verification: CPU V2 postcheck/rerank
```

V2 learned-policy target:

```text
target: <= 79.327s
achieved: 77.6833s
status: met
verification: CPU V2 normal-start policy eval
```

Final commit:

```text
1c09889 Complete Physics V2 learned policy pipeline
```

Final push:

```text
main -> origin/main
```

The project now has both historical achieved reports:

```text
RL1-Achieved.md
PhysicsV2-Achieved.md
```

`RL1-Achieved.md` documents the first V1 learned-policy success.

`PhysicsV2-Achieved.md` documents the post-RL1 calibrated V2 physics, V2 GPU ES, and V2 learned-policy success.
