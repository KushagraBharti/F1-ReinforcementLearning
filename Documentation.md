# Documentation

## Current Status

Date: 2026-06-04.

The project has pivoted from PPO micro-rung tuning to reusable elitist evolutionary search. The current active goal is evolved-controller search first: get a valid normal-start evolved Monza lap under `80.0s`, then use useful search behavior for PPO transfer later.

Current scoreboard:

- Valid normal-start PPO lap: no.
- Lap time: none.
- Best honest normal-start PPO progress: about `970.775m`.
- Historical best valid evolved lap: `148.3s`.
- Latest completed evolved run: `C:\f1rl-artifacts\evolution-speed-100x30-fastlap-v1-20260604`.
- Latest evolved run result: `39` valid laps, fastest valid lap `161.95s`, final generation average distance `2733.5m`, final generation average pace `117.2 kph`, final generation top-decile pace `168.4 kph`.
- Active target: valid normal-start evolved lap `<=80.0s`.

Why the pivot happened:

- The previous PPO/curriculum loop produced local Rettifilo subskills.
- It did not improve honest normal-start behavior.
- It drifted into narrow gates, one-off action sets, and slow hand-written schedule searches.
- The project now needs population search: spawn candidates, score them, keep elites, mutate, repeat.

## Active Direction

Use `goal.md` and `workflow.md`.

Immediate method:

1. Analyze the latest full-run artifacts.
2. Make aggressive scoring/selection/storage changes aimed at speed and population breadth.
3. Validate with focused checks and small searches.
4. Re-read the latest full run against the changes.
5. Start archiving the analyzed old run to D: after the second review confirms it is safe.
6. Verify archive, delete the old C: original, and run the next large comparison.
7. Repeat until the evolved valid lap is `<=80.0s`.

Do not resume the old PPO micro-engineering loop.

## Storage Contract

Large evolution runs split artifacts by purpose:

- Hot control artifacts stay on `C:\f1rl-artifacts\<run-name>`: summaries, attempts, generation summaries, checkpoints, manifests, top genomes, elite libraries, bridges, and next-command files.
- Bulky all-candidate step telemetry goes to `D:\f1-rl-artifacts\cold-telemetry\<run-name>` as lossless `.jsonl.gz`.
- Old analyzed large-run archives go to `D:\f1-rl-artifacts\archives`.
- The C: manifest points to D: cold trace paths, so replay and state-library extraction can still start from the C: run folder.
- Do not delete a C: original until the D: archive exists and `tar -tzf` can list it.

## Active Implementation

Evolution search module:

- `src/f1rl/evolution_search.py`

CLI:

- `python -m f1rl.evolution_search`
- `f1-evolution-search`
- `python -m f1rl.evolution_ladder`
- `f1-evolution-ladder`

Outputs:

- `evolution_summary.json`
- `attempts.jsonl`
- `generation_summary.jsonl`
- `best_so_far.json`
- `population_checkpoint.json`
- `top_genomes/*.json`
- `selected_telemetry/*.jsonl`
- `selected_telemetry/manifest.json`
- optional cold all-candidate `*.jsonl.gz` traces on D:
- `elite_state_library.json`
- `ppo_bridge.json`
- `next_commands.md`

Core behavior:

- phase-based genomes;
- progress-phase genomes;
- observation-driven controller genomes;
- population initialization;
- elite retention;
- mutation;
- crossover;
- random immigrants;
- chunked optional multiprocessing;
- strict target gates;
- scoring profiles;
- multi-profile elite selection;
- streaming artifacts;
- checkpoint/resume.

Evolution ladder module:

- `src/f1rl/evolution_ladder.py`

It runs smoke/small/medium/large search ladders that start with normal-start probes, search Rettifilo rungs, and return to full-lap probes at major milestones.

Legacy diagnostics:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`

They remain available for tiny investigations, but they are not the active main path.

## GPU Backend Update - 2026-06-04

Implemented an explicit PyTorch/CUDA backend while keeping CPU `MonzaSim` as the truth/reference path.

New runtime pieces:

- GPU batched tensors/physics/track/features/scoring/evolution backend:
  - `src/f1rl/gpu_types.py`
  - `src/f1rl/gpu_physics.py`
  - `src/f1rl/gpu_track.py`
  - `src/f1rl/gpu_features.py`
  - `src/f1rl/gpu_scoring.py`
  - `src/f1rl/gpu_batch.py`
  - `src/f1rl/gpu_observation.py`
  - `src/f1rl/evolution_backend.py`
- Separate opt-in GPU-native PPO trainer:
  - `src/f1rl/gpu_ppo.py`
  - CLI: `python -m f1rl.gpu_ppo` / `f1-gpu-ppo`

Evolution search flags:

```powershell
--backend cpu|gpu
--gpu-device cpu|cuda
--gpu-dtype float32|float64
--gpu-batch-size auto|N
--gpu-verify-top-k N
--gpu-telemetry-mode selected|top|all-cpu-replay|none
--gpu-parity-check
--gpu-fallback-to-cpu
--gpu-compile
--gpu-engine eager|graph|fused
--gpu-profile none|torch|nsight
--gpu-profile-output PATH
--gpu-chunk-steps N
--gpu-static-batch-size auto|N
--gpu-disable-early-stop
--gpu-fast-geometry local_window|grid
--gpu-collision-mode exact_all_segments|exact_grid|mask_only_debug
--gpu-cpu-replay-top-k N
--gpu-run-mode parity|production|postcheck
```

Backend contract:

- `--backend cpu` remains default and unchanged.
- `--backend gpu` supports controller, phase, and progress-phase genomes.
- GPU rollout covers batched physics, search features, discrete/control selection, raycasts/observations, boundary collision, checkpoint/lap validity, termination, and scoring.
- Selected GPU candidates are CPU-replayed through `MonzaSim`; verified rows keep `gpu_score`, `cpu_verified_score`, and `backend_parity_error`.
- GPU selected telemetry remains normal replay-compatible JSONL generated by CPU replay.
- `--telemetry-selection all` on GPU requires explicit `--gpu-telemetry-mode all-cpu-replay`.
- Manual driving, replay rendering, JSONL/gzip writing, track preprocessing, and the SB3/Gymnasium PPO path remain CPU-reference compatible.

GPU speed architecture status:

- CPU `MonzaSim` is still the oracle and `--backend cpu` is unchanged.
- `--backend gpu --gpu-engine eager` is the PyTorch eager reference backend.
- `--backend gpu --gpu-engine graph` now attempts real `torch.cuda.CUDAGraph` capture/replay for fixed-shape rollout chunks and caches captured graphs on the long-lived GPU backend across matching generations. Short runs capture one rollout graph. Longer runs where `max_steps > --gpu-chunk-steps` capture a bounded chunk graph and replay it repeatedly, carrying static state and scoring accumulators between chunks. If `max_steps` is not divisible by `--gpu-chunk-steps`, graph mode captures an exact smaller tail graph instead of executing extra masked timesteps. Successful graph runs report `gpu_kernel_backend=pytorch_cuda_graph` or `pytorch_cuda_graph_chunked`; CPU or unsupported capture paths report `pytorch_eager_graph_fallback` with `gpu_graph_errors`.
- `--backend gpu --gpu-engine fused` is promoted for the explicit CUDA float32/local-window/exact-grid path: `--gpu-device cuda --gpu-dtype float32 --gpu-fast-geometry local_window --gpu-collision-mode exact_grid`. It supports both open-distance `--no-target-termination` runs and target-terminated segment runs. That path uses Warp CUDA kernels for controller feature assembly/lookahead/target-steer math, controller-output math, controller diagnostic local projection, phase/progress-phase control selection, phase/progress-phase diagnostic local projection, physics, cached local-window projection, drivable-mask lookup, exact-grid collision, finish-line checks, lap/open-distance bookkeeping, segment target gate bookkeeping, per-step score-accumulator updates, and final profile scoring. The serious controller/no-target search case now routes to a persistent device-side full-rollout kernel and reports `gpu_kernel_backend=warp_persistent_controller_open`; broader fused cases still report `gpu_kernel_backend=warp_open_step`. Unsupported fused configurations still fail loudly instead of silently falling back.
- `--gpu-run-mode parity|production|postcheck` separates correctness checks from throughput runs. `parity` keeps conservative CPU replay and detailed artifacts. `production` defaults to compact search artifacts by setting `--gpu-telemetry-mode none`, and now CPU-reranks selected leaders with an effective replay count of `max(16, --gpu-verify-top-k, --top-k * 2)` unless `--gpu-cpu-replay-top-k` is explicitly provided. An explicit `--gpu-cpu-replay-top-k 0` remains available for raw throughput/profiling, but those GPU-only winners are not trusted until deferred CPU postcheck passes. `postcheck` CPU-replays selected saved candidates after a production search.
- GPU track tensors now precompute local projection candidate indices and a boundary-cell grid.
- Controller lookahead feature extraction in the eager/graph reference path samples all configured lookaheads in one vectorized centerline call and reuses cached lookahead-distance, local-projection-window, action-id, row-index, and last-centerline-segment tensors per GPU batch. The fused controller path now writes the controller feature matrix and diagnostics from a Warp feature-assembly kernel instead of calling the PyTorch lookahead, braking-gate, target-steer, and feature-stacking helpers.
- GPU feature extraction, scoring, and step termination now reuse batch-shaped zero/one tensors instead of repeatedly allocating same-shaped masks for missing feature slots, accumulator updates, and gate updates.
- GPU rollout scoring now uses an explicit `GpuScoringDiagnostics` payload instead of merging control/step diagnostics dictionaries inside every simulated step. The old mapping-based scorer remains as a compatibility wrapper, and rollout passes the cached batch zero tensor into the static scorer.
- GPU rollout `torch.profiler.record_function` ranges are disabled in normal `--gpu-profile none` runs and enabled only for explicit `--gpu-profile torch|nsight` diagnostics.
- GPU rollout active checks no longer perform the redundant step-0 host synchronization; early-stop checks now happen only at actual chunk boundaries.
- `--gpu-collision-mode exact_grid` uses grid broadphase plus exact segment narrowphase for simulator-step movement collision. `exact_all_segments` remains the default reference mode.
- `--gpu-static-batch-size N` now pads the last GPU batch to a fixed tensor shape and disables padded lanes before rollout, so artifacts and step metrics count real candidates only.
- CUDA Graph generation summaries report `gpu_graph_capture_count`, `gpu_graph_replay_count`, `gpu_graph_fallback_count`, `gpu_graph_cache_hit_count`, `gpu_graph_cache_miss_count`, `gpu_graph_cache_size`, `gpu_graph_steps_per_capture`, `gpu_graph_requested_replay_count`, `gpu_graph_capture_seconds`, and `gpu_graph_replay_seconds`.
- GPU generation summaries now include replay aggregate fields: `gpu_cpu_replay_count`, `gpu_cpu_replay_seconds`, `gpu_parity_status`, top-replay progress deltas, reason mismatches, valid-lap mismatches, rank overlap, and score correlation.
- GPU generation summaries and console progress now include stage timings for evaluation, control-program upload, GPU rollout wall time, GPU result materialization, CPU replay, attempts JSONL writing, reproduction/selection/mutation, selected telemetry/postprocess writing, manifest/archive-style work, and fraction of backend time spent in GPU rollout, CPU replay, and result materialization.
- `--gpu-profile torch` writes Chrome trace, table, summary, and bottleneck-report artifacts. `--gpu-profile nsight` writes an Nsight Systems command hint artifact.

GPU PPO contract:

- `f1-gpu-ppo` is separate from `f1-train`; it does not replace SB3.
- Rollout collection, policy forward/backward, rewards, and buffers stay in PyTorch on the selected device.
- Final policies are saved as `policy.pt`.
- CPU deterministic replay/eval writes `cpu_eval_summary.json` and replayable `cpu_eval_episode_*.jsonl`.
- This is an opt-in custom PPO path for GPU-native experiments; honest promotion still requires normal-start CPU replay/eval.

Validation completed:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\validation-cpu-evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1
uv run --no-sync python -m f1rl.evolution_ladder --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base
uv run --no-sync python -m f1rl.train --timesteps 16 --seed 91 --n-envs 1 --max-steps 8 --device cpu --checkpoint-every 0 --eval-every 0 --telemetry none --run-name validation-sb3-smoke --action-mode continuous --observation-profile base --n-steps 8 --batch-size 8 --n-epochs 1
uv run --no-sync python -m f1rl.gpu_ppo --device cuda --require-gpu --dtype float32 --output-dir artifacts\gpu-ppo-cuda-smoke --timesteps 16 --n-envs 2 --n-steps 4 --batch-size 4 --n-epochs 1 --hidden-size 32 --max-steps 8 --observation-profile base --start-speed-kph 60 --target-progress-m 6 --terminate-at-target --cpu-eval-episodes 1 --cpu-eval-max-steps 8
uv run --no-sync python -m f1rl.replay artifacts\validation-gpu-100x10\selected_telemetry --headless --limit 2
```

Key validation artifacts:

- `artifacts\validation-gpu-controller-smoke`
- `artifacts\validation-gpu-phase-smoke`
- `artifacts\validation-gpu-progress-phase-smoke`
- `artifacts\validation-compare-cpu-20x4`
- `artifacts\validation-compare-gpu-20x4`
- `artifacts\validation-cpu-100x5`
- `artifacts\validation-gpu-100x5`
- `artifacts\validation-gpu-100x10`
- `artifacts\gpu-compile-cuda-smoke`
- `artifacts\gpu-ppo-cuda-smoke`
- `artifacts\gpu-ppo-compile-cuda-smoke-2`

Results:

- Full Ruff, Pyright, and pytest passed.
- CUDA available: `NVIDIA GeForce RTX 4060 Laptop GPU`, PyTorch `2.10.0+cu128`, CUDA `12.8`.
- CPU evolution smoke, ladder smoke, and SB3 PPO smoke passed.
- GPU controller/phase/progress-phase CUDA smokes passed with selected CPU replay.
- GPU raycast and observation parity tests passed.
- 20x4 all-candidate CPU/GPU parity comparison matched generation leaders exactly after CPU verification; score correlation was `0.9999998`.
- 100x5 GPU qualification completed `500` attempts; selected verified rows had max progress delta about `0.00111m`.
- 100x10 GPU qualification completed `1000` attempts; selected verified rows had max progress delta about `0.00111m`.
- Headless replay loaded selected GPU 100x10 telemetry.
- GPU PPO CUDA smoke saved `policy.pt`, wrote CPU eval telemetry, and CPU replay reached about `2.24m` in the tiny 8-step smoke.
- `--gpu-compile` works for the evolution backend on CUDA in `artifacts\gpu-compile-cuda-smoke`.
- `--compile-policy` for GPU PPO is implemented and falls back to eager locally because TorchInductor reported missing Triton; the fallback is recorded in `training_summary.json`.

Additional GPU speed validation artifacts:

- `artifacts\gpu-speed-cpu-evolution-smoke`: standard CPU evolution smoke passed.
- `artifacts\gpu-speed-exact-grid-smoke`: CUDA eager exact-grid smoke passed with static batch padding, cached segment projection, normal-mode profile ranges disabled, zero rollout host syncs, and CPU replay; `gpu_parity_status=passed`, `gpu_cpu_replay_count=3`, max top replay progress delta about `0.00032m`, reason mismatches `0`, `gpu_host_sync_count=0`.
- `artifacts\gpu-speed-graph-smoke-small`: earlier explicit graph/compile-helper smoke passed with cached segment projection, normal-mode profile ranges disabled, and zero rollout host syncs; this has been superseded by the CUDA Graph capture artifacts below.
- `artifacts\gpu-speed-profile-smoke`: torch profiler smoke wrote trace/table/summary with explicit launch metrics and enabled profile ranges; the tiny `6x1`, `24`-step CUDA run recorded `19684` `cudaLaunchKernel` launches after the feature/scoring/termination mask-cache cleanup, `gpu_host_sync_count=0`, `gpu_parity_status=passed`, and top events `cudaLaunchKernel`, `gpu_rollout_step`, and runtime module loading. This confirms the remaining eager path is launch/step overhead dominated.
- `artifacts\gpu-speed-compare-cpu-20x2` and `artifacts\gpu-speed-compare-gpu-20x2`: same-seed tiny CPU/GPU comparison completed `40` attempts each. CPU was faster at this small batch (`25.14` candidates/s final generation vs GPU `10.24` candidates/s), while GPU replay parity passed with max top replay progress delta about `0.00062m`, reason mismatches `0`, and `gpu_host_sync_count=0`.
- `artifacts\gpu-speed-static-diagnostics-cpu-smoke`: standard CPU evolution smoke passed after the static GPU scorer refactor.
- `artifacts\gpu-speed-static-diagnostics-cuda-smoke`: CUDA eager exact-grid smoke passed after the static scorer refactor; `gpu_parity_status=passed`, top replay max progress delta about `0.00032m`, reason mismatches `0`, rank overlap `1.0`, score correlation about `0.99999`, normal-mode profile ranges disabled, and `gpu_host_sync_count=0`.
- `artifacts\gpu-speed-static-diagnostics-profile-smoke`: torch profiler smoke after the static scorer refactor recorded `19660` `cudaLaunchKernel` launches on the same tiny `6x1`, `24`-step shape, down from `19684`; CPU replay parity still passed.
- `artifacts\gpu-speed-static-diagnostics-compare-cpu-20x2` and `artifacts\gpu-speed-static-diagnostics-compare-gpu-20x2`: same-seed CPU/GPU comparison completed `40` attempts each. CPU and GPU generation leaders matched, GPU replay parity passed in both generations, rank overlap was `1.0`, score correlation stayed above `0.99994`, and selected GPU telemetry replay loaded headlessly.
- `artifacts\gpu-static-diagnostics-sb3-smoke-20260604-115928`: tiny SB3 PPO smoke completed on the unchanged CPU Gym/SB3 path and saved `final_model.zip`.
- `artifacts\gpu-speed-cuda-graph-capture-smoke`: `--gpu-engine graph` captured one CUDA Graph rollout chunk on CUDA, reported `gpu_kernel_backend=pytorch_cuda_graph`, `gpu_graph_capture_count=1`, `gpu_graph_fallback_count=0`, `gpu_compile_requested=false`, `gpu_host_sync_count=0`, replay parity passed, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-cuda-graph-exact-grid-smoke`: same graph capture smoke with `--gpu-collision-mode exact_grid`; capture succeeded with no fallback, replay parity passed, rank overlap `1.0`, score correlation `1.0`, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-post-graph-eager-cuda-smoke`: post-refactor CUDA eager exact-grid smoke still reported `gpu_kernel_backend=pytorch_eager`, replay parity passed, `gpu_host_sync_count=0`, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-cuda-graph-honest-timing-smoke`: graph timing now includes warmup/capture/replay in `gpu_rollout_seconds` and reports replay-only time separately as `gpu_graph_replay_seconds`; the tiny exact-grid smoke recorded about `0.741s` total rollout versus `0.014s` graph replay, with parity passing.
- `artifacts\gpu-speed-cuda-graph-100x5-qualification` and `artifacts\gpu-speed-eager-100x5-post-graph-qualification`: matching bounded `100x5`, `120`-step CUDA runs completed `500` attempts each with CPU replay parity passing and selected telemetry replay loading. Graph captured every generation with no fallback, but eager was faster overall after honest capture/warmup timing (`~29.7-43.1` GPU candidates/s by generation for eager versus `~18.2-24.4` for graph). This confirms correctness and shows persistent graph reuse is needed before graph mode is a speed win.
- `artifacts\gpu-speed-final-cpu-smoke`: standard CPU evolution smoke passed after the persistent graph-cache change.
- `artifacts\evolution-ladder-20260604-122728`: ladder smoke passed after the persistent graph-cache change.
- `artifacts\gpu-speed-final-eager-cuda-smoke`: CUDA eager exact-grid smoke passed with CPU replay parity, no graph captures/fallbacks, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-final-graph-cache-smoke`: CUDA Graph exact-grid smoke passed with generation 0 capture/miss and generation 1 cache hit/no recapture. Generation 1 reported `gpu_graph_cache_hit_count=1`, `gpu_graph_capture_count=0`, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, and about `462` candidates/s on the tiny fixed-shape replay-only generation.
- `artifacts\gpu-speed-final-compare-cpu` and `artifacts\gpu-speed-final-compare-graph`: same-seed CPU versus CUDA Graph comparison completed `12` attempts each. GPU used `--gpu-parity-check`, CPU replay verified all candidates, graph cache hit on generation 1, selected telemetry replay loaded, and a row comparison found `0` mismatches across generation, candidate index, best/final progress, termination reason, and valid-lap fields.
- `artifacts\gpu-speed-final-profile-smoke`: torch profiler smoke passed and wrote trace/table/summary artifacts.
- `artifacts\gpu-final-ppo-smoke-20260604-122821`: tiny SB3 PPO smoke completed on the unchanged CPU Gym/SB3 path and saved `final_model.zip`.
- `artifacts\gpu-speed-chunked-graph-smoke-v2`: forced chunked CUDA Graph smoke with `--gpu-chunk-steps 4` and `max_steps=12` passed. It reported `gpu_kernel_backend=pytorch_cuda_graph_chunked`, `gpu_graph_steps_per_capture=[4]`, `gpu_graph_requested_replay_count=3`, no fallback, CPU replay parity passed, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-post-chunk-full-graph-smoke`: post-chunk full-rollout CUDA Graph smoke still passed with `gpu_kernel_backend=pytorch_cuda_graph`, one replay per generation, cache hit on generation 1, no fallback, CPU replay parity passed, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-chunked-tail-graph-smoke`: forced non-divisible chunked CUDA Graph smoke with `--gpu-chunk-steps 4` and `max_steps=10` passed. It reported `gpu_graph_steps_per_capture=[2,4]`, `gpu_graph_requested_replay_count=3`, `gpu_steps_executed=10`, no fallback, CPU replay parity passed, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-bench-smoke-cpu-32x2-512`, `artifacts\gpu-speed-bench-smoke-eager-32x2-512`, and `artifacts\gpu-speed-bench-smoke-graph-32x2-512`: same-seed performance smoke for `population=32`, `generations=2`, `max_steps=512`, no-target termination. Outcome metrics matched across backends. CPU total generation times were about `2.39s` and `6.45s`; eager GPU was slower at about `18.02s` and `13.73s`; chunked graph was `10.84s` with first-generation capture overhead and `3.38s` on the cached second generation. GPU rollout throughput in generation 1 was about `2.80` candidates/s for eager versus `52.05` candidates/s for cached chunked graph, with CPU replay parity passing and selected graph telemetry replay loading headlessly.
- `artifacts\gpu-speed-bottleneck-profile-smoke`: CUDA eager torch-profiler smoke wrote trace/table/summary/bottleneck report artifacts. The report identified `cudaLaunchKernel` as the top event with `5116` launches, followed by runtime module loading and `aten::bmm`; CPU replay parity passed and selected telemetry replay loaded headlessly.
- Focused fused-gate validation: `uv run --no-sync ruff check src\f1rl\gpu_fast_warp.py src\f1rl\hardware.py src\f1rl\evolution_search.py tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py`, `uv run --no-sync pyright src\f1rl\gpu_fast_warp.py src\f1rl\hardware.py src\f1rl\evolution_search.py`, and `uv run --no-sync pytest -q tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly` passed. Full `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `git diff --check` passed. `uv run --no-sync f1-hardware-check --json` originally reported CUDA available and Warp absent with `ModuleNotFoundError: No module named 'warp'`; direct `uv pip install warp-lang --verbose` then installed `warp-lang==1.14.0`. `uv run --no-sync f1-hardware-check --json --warp-smoke` now passes with `torch_interop_smoke.passed=true` and zero max error. `f1-evolution-search --backend gpu --gpu-engine fused ...` fails cleanly with the explicit "not promoted yet" message after confirming Warp is available.
- Focused Warp fused physics validation: `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py src\f1rl\gpu_fast_warp.py tests\test_gpu_physics.py`, `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py src\f1rl\gpu_fast_warp.py`, and `uv run --no-sync pytest -q tests\test_gpu_physics.py tests\test_gpu_fast_warp.py` passed. The new CUDA tests compare one-step fused Warp physics and an 80-step randomized fused Warp loop against the PyTorch GPU physics reference.
- Focused Warp fused geometry validation: `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py tests\test_gpu_track.py`, `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py`, and `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_local_projection_matches_torch_cached_projection tests\test_gpu_track.py::test_warp_local_projection_matches_torch_randomized_near_centerline` passed. The new CUDA tests compare deterministic and randomized cached local-window projection outputs against `track_errors_batch`.
- Focused Warp fused drivable-mask validation: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_drivable_mask_matches_torch_reference tests\test_gpu_track.py::test_warp_drivable_mask_matches_torch_randomized_points` passed. The new CUDA tests compare deterministic and randomized mask lookups against `point_is_drivable_batch`.
- Focused Warp fused grid-collision validation: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_grid_segment_intersection_matches_torch_reference tests\test_gpu_track.py::test_warp_grid_segment_intersection_matches_torch_randomized_step_movements` passed. The new CUDA tests compare deterministic and randomized movement/boundary intersection outputs against `segments_intersect_any_grid_batch`; the randomized movements also verify the PyTorch grid result against exact all-segment intersection.
- Focused Warp fused controller-output validation: `uv run --no-sync pytest -q tests\test_gpu_features.py::test_warp_controller_controls_match_torch_reference tests\test_gpu_features.py::test_warp_controller_controls_match_torch_reference_at_saturation` passed. The new CUDA tests compare randomized and saturation-heavy controller output against `controller_controls_batch`.
- Focused Warp score-profile validation: `uv run --no-sync pytest -q tests\test_gpu_scoring.py::test_warp_score_profiles_match_torch_reference tests\test_gpu_scoring.py::test_warp_core_score_profiles_reject_unsupported_profiles` passed. The CUDA test compares every current `SCORING_PROFILES` entry against `score_profiles_batch`; unsupported profiles fail explicitly rather than being approximated.
- Post-Warp full validation: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `git diff --check` passed.
- Post-projection/drivable full validation: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `git diff --check` passed. `f1-evolution-search --backend gpu --gpu-engine fused ...` still fails cleanly with the updated not-promoted message after confirming standalone Warp physics/local-projection/drivable-mask parity coverage.
- Post-collision full validation: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `git diff --check` passed. `f1-evolution-search --backend gpu --gpu-engine fused ...` still fails cleanly with the updated not-promoted message after confirming standalone Warp physics/local-projection/drivable-mask/grid-collision parity coverage.
- Post-controller full validation: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `git diff --check` passed. `f1-evolution-search --backend gpu --gpu-engine fused ...` still fails cleanly with the updated not-promoted message after confirming standalone Warp physics/local-projection/drivable-mask/grid-collision/controller-output parity coverage.
- `artifacts\gpu-speed-warp-physics-cpu-smoke`: post-Warp CPU evolution CLI smoke passed.
- `artifacts\gpu-speed-warp-physics-graph-smoke`: post-Warp CUDA Graph CLI smoke passed with generation 0 graph capture, generation 1 graph cache hit, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-projection-cpu-smoke`: post-fused-projection CPU evolution CLI smoke passed.
- `artifacts\gpu-speed-warp-projection-graph-smoke`: post-fused-projection CUDA Graph CLI smoke passed with generation 0 graph capture, generation 1 graph cache hit, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-collision-cpu-smoke`: post-fused-collision CPU evolution CLI smoke passed.
- `artifacts\gpu-speed-warp-collision-graph-smoke`: post-fused-collision CUDA Graph CLI smoke passed with generation 0 graph capture, generation 1 graph cache hit, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-controller-cpu-smoke`: post-fused-controller CPU evolution CLI smoke passed.
- `artifacts\gpu-speed-warp-controller-graph-smoke`: post-fused-controller CUDA Graph CLI smoke passed with generation 0 graph capture, generation 1 graph cache hit, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-gate-cpu-smoke`: post-fused-gate CPU evolution CLI smoke passed.
- `artifacts\evolution-ladder-20260604-131117`: post-fused-gate ladder smoke passed.
- `artifacts\gpu-speed-warp-gate-graph-smoke`: post-fused-gate CUDA Graph CLI smoke passed with generation 0 graph capture, generation 1 graph cache hit, `gpu_graph_fallback_count=0`, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, and selected telemetry replay loading headlessly.
- `artifacts\gpu-warp-gate-sb3-smoke-20260604-131121`: post-fused-gate tiny SB3 PPO CPU smoke passed and saved `final_model.zip`.
- Final gpu-focus validation after the Warp score-profile kernel: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `uv run --no-sync f1-hardware-check --json --warp-smoke` passed. Warp `1.14.0` initialized on the RTX 4060 and the Torch/Warp interop smoke reported zero max error.
- `artifacts\gpu-focus-final-cpu-smoke`: final required CPU evolution CLI smoke passed.
- `artifacts\gpu-focus-final-graph-smoke`: final CUDA Graph exact-grid smoke passed with generation 0 graph capture, generation 1 graph cache hit, no graph fallback, `gpu_parity_status=passed`, `gpu_cpu_replay_count=2` per generation, max top replay progress delta about `0.00011m`, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- Earlier fused-engine negative validation exited before rollout with the explicit not-promoted message after confirming Warp was available. That gate has since been narrowed to unsupported fused configurations; the supported no-target CUDA float32 exact-grid path now runs publicly behind `--gpu-engine fused`.
- `artifacts\gpu-focus-final-sb3-smoke-20260604-134431`: final tiny SB3 PPO CPU smoke passed and saved `final_model.zip`.
- `artifacts\gpu-focus-final-compare-cpu` and `artifacts\gpu-focus-final-compare-graph`: final same-seed CPU versus CUDA Graph comparison completed `16` attempts each. A direct row comparison found `0` mismatches for generation, candidate index, best/final progress, elapsed time, score, termination reason, and valid-lap fields at validation tolerance. The GPU run CPU-replayed all `8` candidates per generation, had no graph fallback, hit the graph cache in generation 1, and selected telemetry replay loaded headlessly.
- `artifacts\evolution-ladder-20260604-134536`: final ladder smoke passed after the score-profile kernel.
- Warp score-profile coverage was expanded from the initial core subset to every current evolution scoring profile: `apex`, `brake_zone`, `clean_distance`, `clean_exit`, `early_pace`, `exit_speed`, `farthest_distance`, `fast_frontier`, `fast_valid_lap`, `frontier`, `frontier_fast`, `frontier_novelty`, `frontier_recovery`, `full_lap_validity`, `lap_pace`, `max_progress`, `risk_seeking`, and `time_attack`. Focused validation passed: `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py src\f1rl\evolution_search.py tests\test_gpu_scoring.py Documentation.md`, `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py src\f1rl\evolution_search.py tests\test_gpu_scoring.py`, and `uv run --no-sync pytest -q tests\test_gpu_scoring.py::test_warp_score_profiles_match_torch_reference tests\test_gpu_scoring.py::test_warp_core_score_profiles_reject_unsupported_profiles`.
- Broader post-score-profile Warp validation passed: `uv run --no-sync pytest -q tests\test_gpu_scoring.py tests\test_gpu_features.py tests\test_gpu_track.py tests\test_gpu_physics.py tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly`.
- `artifacts\gpu-speed-score-profiles-fused-negative`: earlier fused CLI gating failed before rollout with the updated not-promoted message, naming `current-score-profile` parity coverage before the no-target fused-open rollout was promoted.
- Added a standalone Warp step-bookkeeping kernel for the serious no-target search path and then extended it to segment target gates. It consumes post-physics/projection/collision/drivable/finish tensors, mutates `GpuCarBatch` progress, checkpoint validity, no-progress counters, segment target/release-gate flags, finish/lap flags, collision/off-track/max-step termination, and returns `progress_delta_m`, collision/off-track masks, and telemetry-valid-lap flags. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_open_step_bookkeeping_matches_torch_reference`.
- Broader post-bookkeeping Warp validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py tests\test_gpu_physics.py tests\test_gpu_features.py tests\test_gpu_scoring.py tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly`, plus `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py tests\test_gpu_track.py` and `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py tests\test_gpu_track.py`.
- Added a standalone chained Warp step helper. It applies Warp physics, active-row physics commit, local projection, exact-grid collision, drivable-mask lookup, finish-line intersection, and target/open-distance bookkeeping, then returns the same diagnostics tuple shape as the eager `_step` path. Focused validation passed against `GpuMonzaBatch._step`: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_open_step_bookkeeping_matches_torch_reference tests\test_gpu_track.py::test_warp_open_step_matches_gpu_batch_step_reference`.
- Broader post-open-step Warp validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py tests\test_gpu_physics.py tests\test_gpu_features.py tests\test_gpu_scoring.py tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly`, plus `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py tests\test_gpu_track.py` and `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py tests\test_gpu_track.py`.
- Added `GpuMonzaBatch.rollout_warp_open()` for no-target CUDA float32 batches. It keeps existing control-program generation and static score-accumulator updates, but swaps the per-step simulator update to the chained Warp open-step helper and returns `kernel_backend=warp_open_step`. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_open_step_bookkeeping_matches_torch_reference tests\test_gpu_track.py::test_warp_open_step_matches_gpu_batch_step_reference tests\test_gpu_track.py::test_warp_open_rollout_matches_gpu_batch_rollout_reference`.
- Broader post-rollout-open Warp validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py tests\test_gpu_physics.py tests\test_gpu_features.py tests\test_gpu_scoring.py tests\test_gpu_fast_warp.py tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly`, plus focused Ruff/Pyright on `src\f1rl\gpu_batch.py`, `src\f1rl\gpu_fused_warp.py`, and `tests\test_gpu_track.py`.
- Public fused-open dispatch validation passed: `uv run --no-sync ruff check src\f1rl\evolution_search.py src\f1rl\evolution_backend.py tests\test_gpu_evolution_backend.py`, `uv run --no-sync pyright src\f1rl\evolution_search.py src\f1rl\evolution_backend.py tests\test_gpu_evolution_backend.py`, and `uv run --no-sync pytest -q tests\test_gpu_evolution_backend.py::test_gpu_unimplemented_speed_engines_fail_loudly tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts`.
- `artifacts\gpu-speed-fused-open-smoke`: public `--backend gpu --gpu-engine fused` no-target smoke passed with `gpu_kernel_backend=warp_open_step`, CPU replay parity passing in both generations, zero top replay reason mismatches, max top replay progress delta about `0.00004m`, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-fused-open-compare-eager` and `artifacts\gpu-speed-fused-open-compare-fused`: same-seed eager versus fused no-target comparison completed `16` attempts per backend. A direct row comparison found `0` mismatches for generation, candidate index, best/final progress, termination reason, and valid-lap fields at validation tolerance. The fused run CPU-replayed all `8` candidates per generation, reported `gpu_kernel_backend=warp_open_step`, had zero replay reason mismatches, and selected telemetry replay loaded headlessly.
- Public fused target-gate validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_target_step_matches_gpu_batch_target_reference tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_target_backend_writes_verified_artifacts`, with focused Ruff and Pyright also passing on the changed modules.
- `artifacts\gpu-speed-fused-target-smoke`: public `--backend gpu --gpu-engine fused` target-terminated segment smoke passed. All `16` attempts ended with `segment_complete`, selected CPU replay parity passed with zero top replay reason mismatches, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-fused-target-compare-eager` and `artifacts\gpu-speed-fused-target-compare-fused`: same-seed eager versus fused target comparison completed `16` attempts per backend. A direct row comparison found `0` mismatches for generation, candidate index, best/final progress, elapsed time, score, termination reason, valid-lap, and `sim_segment_complete` fields at validation tolerance. The fused run CPU-replayed all `8` candidates per generation, reported `gpu_kernel_backend=warp_open_step`, had zero replay reason mismatches, and selected telemetry replay loaded headlessly.
- Fused rollouts now use the Warp score-profile kernel for final profile scoring instead of returning to the PyTorch scorer. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_scoring.py tests\test_gpu_track.py::test_warp_open_rollout_matches_gpu_batch_rollout_reference tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_target_backend_writes_verified_artifacts`, with focused Ruff and Pyright also passing.
- `artifacts\gpu-speed-fused-warp-score-target-smoke`: public fused target smoke after the Warp scoring reroute passed with `gpu_kernel_backend=warp_open_step`, CPU replay parity passing in both generations, zero top replay reason mismatches, all attempts ending with `segment_complete`, and selected telemetry replay loading headlessly.
- Fused rollouts now use a Warp per-step score-accumulator update kernel instead of returning to the PyTorch static accumulator inside the fused hot loop. A direct CUDA parity test compares every `GpuScoreAccumulator` field against `update_score_accumulator_static`, including inactive rows. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_scoring.py tests\test_gpu_track.py::test_warp_open_rollout_matches_gpu_batch_rollout_reference tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_target_backend_writes_verified_artifacts`, with focused Ruff and Pyright also passing.
- Fused controller-genome rollouts now route controller outputs through the Warp controller kernel when `--gpu-engine fused` is selected. Eager, graph, CPU, and SB3 paths keep their existing control generation. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_features.py::test_warp_controller_controls_match_torch_reference tests\test_gpu_scoring.py::test_warp_score_accumulator_update_matches_torch_reference tests\test_gpu_track.py::test_warp_open_rollout_matches_gpu_batch_rollout_reference tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_controller_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_target_backend_writes_verified_artifacts`, with focused Ruff and Pyright also passing.
- Fused controller-genome rollouts now route controller feature extraction through `search_features_warp_batch`, which uses the Warp local projection helper before applying the existing target-speed, brake-demand, lookahead, target-steer, and feature-stacking math. Direct CUDA parity compares the feature matrix and diagnostics against `search_features_batch`, and the public fused controller artifact test CPU-replays selected candidates.
- `artifacts\gpu-speed-warp-controller-features-fused-smoke`: public fused controller smoke after the Warp feature-projection reroute passed with `gpu_kernel_backend=warp_open_step`, CPU replay parity passing, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-controller-features-eager-compare` versus `artifacts\gpu-speed-warp-controller-features-fused-smoke`: same-seed eager/fused controller comparison completed `16` attempts per backend. Direct row comparison found zero key mismatches, zero termination-reason mismatches, zero final-action mismatches, zero lateral-error drift, zero heading-error drift, max progress drift `0.000061m`, and max score drift about `0.038`.
- Fused phase and progress-phase rollouts now route discrete action selection through a Warp phase-control kernel when `--gpu-engine fused` is selected. Direct CUDA parity tests cover both elapsed-step thresholds and progress-distance thresholds, and the fused progress-phase artifact test CPU-replays selected candidates. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_features.py::test_warp_phase_controls_match_torch_reference tests\test_gpu_features.py::test_warp_progress_phase_controls_match_torch_reference tests\test_gpu_evolution_backend.py::test_gpu_fused_open_distance_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_progress_phase_backend_writes_verified_artifacts tests\test_gpu_evolution_backend.py::test_gpu_fused_target_backend_writes_verified_artifacts`, with focused Ruff and Pyright also passing.
- `artifacts\gpu-speed-fused-warp-acc-target-smoke`: public fused target smoke after the Warp accumulator update passed with `gpu_kernel_backend=warp_open_step`, CPU replay parity passing in both generations, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-post-acc-compare-cpu`, `artifacts\gpu-speed-post-acc-compare-eager`, and `artifacts\gpu-speed-post-acc-compare-fused`: same-seed CPU/eager/fused target comparison completed `16` attempts per backend after the Warp accumulator/controller changes. Eager and fused rows matched exactly. CPU versus GPU shared rows had zero termination-reason mismatches, max best-progress drift about `0.00018m`, and max score drift about `0.038`; later-generation candidate identities can diverge from CPU because tiny float32 score differences affect selection. The fused run CPU-replayed selected top candidates with zero replay reason mismatches and selected telemetry loaded headlessly.
- `artifacts\gpu-speed-warp-phase-control-progress-smoke`: public fused progress-phase smoke passed with `gpu_kernel_backend=warp_open_step`, `gpu_parity_status=passed` in both generations, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-phase-control-progress-eager-compare` versus `artifacts\gpu-speed-warp-phase-control-progress-smoke`: same-seed eager/fused progress-phase comparison completed `16` attempts per backend. Direct row comparison found zero key mismatches, zero termination-reason mismatches, zero final-action mismatches, zero progress drift, and zero score drift.
- Final post-Warp phase-control validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `git diff --check`. Pytest only reported the existing SB3 `VecMonitor` warnings; `git diff --check` only reported line-ending conversion notices.
- Fused phase and progress-phase rollouts now also use the Warp local projection helper for pre-step scoring diagnostics instead of the PyTorch `track_errors_batch` path. Rollout parity tests compare the changed phase/progress-phase fused route against the PyTorch rollout reference.
- `artifacts\gpu-speed-warp-diagnostic-progress-smoke`: public fused progress-phase smoke after the Warp diagnostic-projection reroute passed with `gpu_kernel_backend=warp_open_step`, `gpu_parity_status=passed` in both generations, `gpu_cpu_replay_count=2` per generation, zero top replay reason mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-warp-diagnostic-progress-eager-compare` versus `artifacts\gpu-speed-warp-diagnostic-progress-smoke`: same-seed eager/fused progress-phase comparison completed `16` attempts per backend. Direct row comparison found zero key mismatches, zero termination-reason mismatches, zero final-action mismatches, zero progress drift, zero score drift, zero lateral-error drift, and zero heading-error drift.
- `artifacts\gpu-speed-post-acc-bench-eager-32x2-512` and `artifacts\gpu-speed-post-acc-bench-fused-32x2-512`: same-seed no-target performance smoke after the Warp accumulator/controller changes completed `64` attempts per backend at `population=32`, `generations=2`, `max_steps=512`. Eager used `gpu_kernel_backend=pytorch_eager` and reported about `3.38` then `3.97` GPU candidates/s. Fused used `gpu_kernel_backend=warp_open_step` and reported about `7.03` then `6.92` GPU candidates/s. Eager/fused attempt rows had zero key mismatches, zero termination-reason mismatches, and max best-progress drift `0.000244m`; fused selected telemetry replay loaded headlessly.
- `artifacts\gpu-fused-post-acc-sb3-smoke-20260604-145027`: tiny SB3 PPO CPU smoke passed on the unchanged Gym/SB3 path and saved `final_model.zip` after the fused accumulator/controller changes.
- Final post-accumulator/controller validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `git diff --check`. Pytest only reported the existing SB3 `VecMonitor` warnings; `git diff --check` only reported line-ending conversion notices.
- `artifacts\gpu-fused-post-acc-cpu-evolution-smoke`: standard CPU evolution CLI smoke passed after the fused accumulator/controller changes.
- `artifacts\evolution-ladder-20260604-145242`: ladder smoke passed after the fused accumulator/controller changes.
- Final post-fused-open validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `git diff --check`. Pytest only reported the existing SB3 `VecMonitor` warnings; `git diff --check` only reported line-ending conversion notices.
- Final post-fused-target validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `git diff --check`. Pytest only reported the existing SB3 `VecMonitor` warnings; `git diff --check` only reported line-ending conversion notices.
- Final post-Warp-scoring validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `git diff --check`. Pytest only reported the existing SB3 `VecMonitor` warnings; `git diff --check` only reported line-ending conversion notices.
- Fixed the fused controller feature wrapper so it no longer references missing PyTorch helper imports and now launches the Warp feature-assembly kernel for lookahead, braking-gate distance, target-steer, diagnostics, and feature-matrix writes. Focused validation passed: `uv run --no-sync ruff check src\f1rl\gpu_fused_warp.py src\f1rl\gpu_batch.py tests\test_gpu_features.py tests\test_gpu_evolution_backend.py`, `uv run --no-sync pyright src\f1rl\gpu_fused_warp.py src\f1rl\gpu_batch.py`, and `uv run --no-sync pytest -q tests\test_gpu_features.py tests\test_gpu_evolution_backend.py::test_gpu_fused_controller_backend_writes_verified_artifacts`.
- `artifacts\gpu-speed-warp-feature-controller-eager-compare` versus `artifacts\gpu-speed-warp-feature-controller-fused-compare`: same-seed eager/fused controller comparison completed `32` attempts per backend at `population=16`, `generations=2`, `max_steps=128`. Direct row comparison found zero key/termination mismatches, max final-progress drift about `0.0014m`, max final-speed drift about `0.0074 kph`, max lateral drift about `0.00125m`, and max heading drift about `0.0038deg`. The fused run CPU-replayed the top `4` attempts with zero reason mismatches and max top replay progress delta about `0.00028m`; generation 1 after kernel compilation reported about `12.7` candidates/s on this tiny smoke.
- Added the first persistent Warp full-rollout kernel for the normal-start open-distance controller search path. It keeps the CPU simulator as the oracle but runs controller feature assembly, control output, physics, projection, exact-grid collision, bookkeeping, score accumulation, and final scoring inside one device-side rollout loop for `controller + --no-target-termination + CUDA float32 + local_window + exact_grid`; this route reports `gpu_kernel_backend=warp_persistent_controller_open`. Focused validation passed: `uv run --no-sync pytest -q tests\test_gpu_track.py::test_warp_persistent_controller_rollout_matches_gpu_batch_rollout_reference`.
- `artifacts\gpu-speed-persistent-controller-fused-150x2-18k`: strict fused parity run completed `population=150`, `generations=2`, `max_steps=18000`, no target termination, and selected CPU replay. GPU rollout itself was much faster than the old Python-step fused path (`gpu_rollout_seconds` about `1.40s` and `1.60s`), but CPU replay dominated the end-to-end run (`gpu_cpu_replay_seconds` about `15.5s` and `34.2s`). This run exposed a remaining correctness blocker: one selected candidate had a CPU/eager termination reason of `collision` while the persistent/Warp path reported `off_track` near the same boundary. A focused harness showed candidate `44` now matches eager/persistent tightly after the yaw-rate-limit physics fix, while candidate `119` is a shared Warp-vs-PyTorch/CPU boundary drift rather than a persistent-only bug.
- Added full-run production timing and compact search mode. `--gpu-run-mode production` keeps all-candidate replay and full telemetry packaging off the hot path, but the default correctness-oriented production path now CPU-reranks a bounded selected-leader set so long-horizon GPU drift cannot silently poison elite selection. Pure GPU throughput mode still exists by explicitly setting `--gpu-cpu-replay-top-k 0`; `parity` remains the conservative debug mode and `postcheck` replays selected winners after production.
- `artifacts\gpu-speed-production-fused-150x2-18k`: production fused run completed `population=150`, `generations=2`, `max_steps=18000`, no target termination, with `gpu_telemetry_mode=none`, `gpu_cpu_replay_top_k=0`, `manifest.trace_count=0`, and `gpu_parity_status=not_checked`. Generation 1 reported `gpu_backend_total_seconds=1.4005`, `gpu_rollout_seconds=1.3664`, `gpu_candidates_per_second=109.78`, `gpu_backend_candidates_per_second=107.10`, `gpu_rollout_time_fraction=0.976`, `gpu_cpu_replay_seconds=0.0`, `gpu_result_materialization_seconds=0.0207`, and `timing_attempts_jsonl_write_seconds=0.0314`. This proves CPU replay and selected-telemetry packaging can be removed from the hot path for this shape, but it is not a correctness proof.
- Final post-Warp-feature validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and `uv run --no-sync f1-hardware-check --json --warp-smoke`. Pytest only reported the existing SB3 `VecMonitor` warnings.
- Fixed candidate `119` parity by isolating the TT/WW/TW/WT split to a controller near-tie branch: tiny Warp/PyTorch float32 state drift pushed raw throttle/brake across opposite sides of the dominance branch, which later moved the collision endpoint onto a neighboring drivable-mask pixel. CPU `_controller_controls`, PyTorch `controller_controls_batch`, and Warp controller kernels now use the same brake-preferred near-tie epsilon; termination reason priority remains matched to CPU/eager semantics.
- `artifacts\gpu-speed-persistent-controller-fused-150x2-18k-post-tie-fix`: strict fused parity rerun completed `population=150`, `generations=2`, `max_steps=18000`, no target termination, and selected CPU replay with `gpu_kernel_backend=warp_persistent_controller_open`. Both generations passed parity with zero CPU/GPU reason mismatches and zero valid-lap mismatches. Selected replay max final-progress drift was `0.4551m` in generation 0 and `0.3070m` in generation 1; score correlation stayed above `0.999997`. Headless replay loaded selected traces from this run.
- Added compact GPU production attempts and a deferred postcheck CLI:
  - `--gpu-attempts-mode auto|full|compact`, where `auto` uses compact rows for GPU production/no-telemetry runs and full rows for parity/debug paths.
  - `python -m f1rl.evolution_postcheck` / `f1-evolution-postcheck` CPU-replays selected saved attempts from a completed run, writes `postchecked_attempts.jsonl`, updates `selected_telemetry/manifest.json`, and writes `postcheck_summary.json`.
- `artifacts\gpu-speed-production-compact-150x2-18k`: compact production proof completed `population=150`, `generations=2`, `max_steps=18000`, no target termination, persistent fused controller kernel, no inline CPU replay, and no search-time selected telemetry. Generation 1 reported `gpu_backend_total_seconds=1.3907`, `gpu_rollout_seconds=1.3775`, `gpu_rollout_time_fraction=0.9906`, `gpu_candidates_per_second=108.89`, `gpu_backend_candidates_per_second=107.86`, `gpu_result_materialization_seconds=0.0062`, `timing_attempts_jsonl_write_seconds=0.0146`, and `gpu_cpu_replay_seconds=0.0`. Deferred postcheck on top/profile/farthest selected winners passed with `postcheck_count=10`, zero reason mismatches, zero valid-lap mismatches, max final-progress delta `0.3070m`, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-production-compact-1000x5-18k`: raw GPU-only production completed with `population=1000`, `generations=5`, `max_steps=18000`, compact attempts, no inline CPU replay, and no search-time selected telemetry. The speed path behaved as intended: generations 1-4 reported `gpu_rollout_time_fraction` about `0.98`, `gpu_backend_candidates_per_second` about `617-672`, `gpu_result_materialization_seconds` about `0.011-0.014`, and `gpu_cpu_replay_seconds=0.0`. Deferred postcheck failed with `postcheck_count=8`, `parity_status=failed`, `reason_mismatches=5`, `valid_lap_mismatches=0`, max final-progress delta about `1929.7m`, and max score delta about `135395.2`. Focused TT/WT/TW/WW diagnostics found no compact-artifact reconstruction bug and no persistent-only reason-priority bug. The root issue is long-horizon controller sensitivity from small CPU/eager float32 and eager/Warp action differences: `3:578` first materially diverged CPU-vs-eager at step `601`, `4:277` at CPU-vs-eager step `67` and TT-vs-WW step `173`, and `4:793` at CPU-vs-eager step `630` and TT-vs-WW step `1022`. This raw GPU-only artifact remains a speed proof, not a trusted correctness proof.
- `artifacts\gpu-speed-production-rerank16-150x2-18k`: strict `150x2x18k` production rerun with default selected-leader CPU rerank passed deferred postcheck. Postcheck reported `postcheck_count=10`, `parity_status=passed`, zero reason mismatches, zero valid-lap mismatches, and selected telemetry replay loaded headlessly.
- `artifacts\gpu-speed-production-rerank48-1000x5-18k`: corrected `1000x5x18k` production rerun with effective CPU rerank top-k `48` passed deferred CPU postcheck. The postcheck reported `postcheck_count=8`, `parity_status=passed`, `reason_mismatches=0`, `valid_lap_mismatches=0`, max best/final progress deltas `0.0m`, max score delta `0.0`, and selected telemetry replay loaded headlessly. GPU rollout stayed fast at about `1.53-1.86s` per generation, while bounded CPU rerank/postcheck dominated the correctness-oriented end-to-end run; this is the intentional guard that keeps CPU `MonzaSim` as the promotion oracle.
- Added an additive deferred CPU pool rerank mode for speed-first production/postcheck use. Inline rerank remains the correctness-first production mode. Deferred rerank uses `--candidate-pool-size`, `--cpu-rerank`, and `--workers` in `f1rl.evolution_postcheck`; it writes incremental `postcheck_pool_progress.jsonl` and `postchecked_pool_attempts.jsonl`, dedupes compact pool replays by genome/snapshot/sim/scoring hash, runs compact CPU pool replays in parallel, and writes full selected telemetry only for final CPU-selected winners. Pool parity is diagnostic and may fail because the pool intentionally includes inflated GPU candidates; final selected `parity_status` is the promotion contract.
- `artifacts\gpu-speed-deferred-rerank-smoke-32x2-512`: deferred-rerank smoke passed with `candidate_pool_size=16`, `pool_cpu_replay_requested_count=16`, `pool_cpu_replay_unique_count=13`, `pool_cpu_replay_cache_hits=3`, `workers=2`, final `postcheck_count=4`, `parity_status=passed`, zero reason mismatches, zero valid-lap mismatches, and selected telemetry replay loading headlessly.
- `artifacts\gpu-speed-deferred-rerank-1000x5-18k`: speed-first `1000x5x18k` production run kept inline CPU replay disabled and deferred CPU verification to postcheck. Compared with `artifacts\gpu-speed-production-rerank48-1000x5-18k`, inline rerank spent about `1022.42s` in search-time CPU replay and ended near `2.09` candidates/s, while the deferred run spent `0.0s` in search-time CPU replay, summed `7.33s` GPU rollout / `7.70s` GPU backend time, and ended near `643.36` end-to-end candidates/s for the last generation. Deferred postcheck used `candidate_pool_size=48`, replayed `11` unique rows, skipped/cached `37` duplicate rows, used `8` workers, and took `100.57s`.
- The deferred `1000x5` postcheck now separates pool diagnostics from final selected pass/fail. The pool correctly reports `pool_parity_status=failed` and `pool_score_parity_status=failed` because it includes rejected GPU-inflated rows such as candidate `4:845` with `636.77m` progress drift. Final selected telemetry is filtered to CPU-clean rows before CPU rerank promotion: final `postcheck_count=7`, `parity_status=passed`, `reason_mismatches=0`, `valid_lap_mismatches=0`, max final-progress delta `0.3695m`, and selected telemetry replay loaded headlessly with `uv run --no-sync python -m f1rl.replay artifacts\gpu-speed-deferred-rerank-1000x5-18k\selected_telemetry --headless --limit 2`. Final `score_parity_status` remains a separate diagnostic because CPU score, not GPU score, drives deferred final ranking.
- Post-deferred-rerank validation passed after the selected-winner filter change: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, focused rerank/postcheck tests, deferred selected-telemetry replay loading, and the standard CPU evolution smoke at `artifacts\validation-post-deferred-rerank-cpu-smoke`. Full pytest only reported the existing SB3 `VecMonitor` warnings.
- Post-tie-fix validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync f1-hardware-check --json --warp-smoke`, and `uv run --no-sync pytest -q tests\test_gpu_features.py tests\test_gpu_physics.py tests\test_gpu_track.py tests\test_gpu_scoring.py tests\test_gpu_evolution_backend.py tests\test_gpu_fast_warp.py` with one skip.
- Final compact/postcheck validation passed: `uv run --no-sync ruff check .`, `uv run --no-sync pyright src\f1rl`, `uv run --no-sync pytest -q`, and focused postcheck/GPU tests. Full pytest only reported the existing SB3 `VecMonitor` warnings.
- `artifacts\evolution-ladder-20260604-113652`: latest ladder smoke passed.
- Headless replay loaded a selected trace from `artifacts\gpu-speed-exact-grid-smoke\selected_telemetry`.

Known constraints:

- GPU PPO is functional but intentionally separate and experimental; it is not a replacement for SB3 promotion/eval.
- The bounded `100x5`/`100x10` qualifications used `max_steps=120` and straight action set for backend validation, not full-lap search.
- All-candidate CPU replay on GPU is correctness-oriented and can dominate wall time. Default production now uses compact artifacts plus bounded selected-leader CPU rerank; explicit `--gpu-cpu-replay-top-k 0` is raw throughput/profiling mode and cannot be used to trust winners without deferred postcheck. Use inline rerank when correctness should be guarded during search; use deferred pool rerank when GPU throughput matters and final CPU verification/reranking is acceptable after the run.
- The fast GPU architecture now has a coherent production/postcheck split for the persistent controller/no-target path: production keeps all-candidate CPU replay and heavy telemetry off the hot path, bounded CPU rerank protects elite selection, and deferred postcheck CPU-replays saved winners and writes replay-compatible selected telemetry. Broader dtype/device support and persistent kernels for the remaining genome/target modes remain open.
- Reason mismatches are not acceptable. Candidate `119` is fixed in the `post-tie-fix` artifact, the compact `150x2` production/postcheck proof is clean, and the previous `1000x5` scale-parity blocker is closed by `artifacts\gpu-speed-production-rerank48-1000x5-18k`. Larger pure-throughput scaling is a separate benchmark and was not started here.

## Archived Context

The previous context-heavy docs were archived under:

- `archive/plans/evolution-doc-cleanup-20260603/`
- `archive/plans/evolution-pivot-20260603/`
- `archive/legacy-20260424/`

Do not read those by default. They are historical reference only.

## Latest Validation

After the Yosh-scale evolution implementation:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv run --no-sync pyright src/f1rl
```

Result:

- Ruff passed.
- Pytest passed.
- Pyright passed.

Focused validations run:

```powershell
uv run --no-sync pytest tests/test_evolution_search.py -q
uv run --no-sync pytest tests/test_evolution_search.py tests/test_action_search.py tests/test_elite_search.py -q
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

Result:

- Evolution search tests passed.
- Legacy search tests still passed.
- Phase, progress-phase, and controller CLI smokes completed.
- `--workers 8` CLI smoke completed.
- CLI stop/resume completed.
- Ladder smoke completed at `artifacts\evolution-ladder-smoke-20260603`.

## Next Step

Run real search at increasing scale:

1. `smoke`: validate commands and artifacts.
2. `small`: population around `100`, inspect behavior.
3. `medium`: population around `1000`, compare profiles/genomes.
4. `large`: long-running search only after streaming, resume, and telemetry look sane.

Suggested command shapes are in `EvolutionSearchPlan.md`.

After search finds useful elites:

1. Inspect selected telemetry.
2. Save/use the elite state library.
3. Train PPO curriculum from those states.
4. Test linked transfer.
5. Test honest normal-start eval.

Do not claim final success until PPO completes a valid normal-start lap near `<=80.0s` with scaffolds and assists disabled.

## Generation Replay Update - 2026-06-04

Added generation-by-generation replay for evolution swarm artifacts.

Use:

```powershell
uv run --no-sync f1-replay 'artifacts\evolution-swarm-100x5-visible-next\selected_telemetry' --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Behavior:

- replays one manifest generation at a time;
- keeps candidates sorted by best progress within each generation;
- shows the primary/best candidate plus the rest of that generation as ghost cars;
- displays generation index, raw generation number, car count, and generation best distance.
- press `N` or `Space` during generation replay to skip the current generation.

Validation:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Result:

- Ruff passed.
- Pyright passed.
- Pytest passed.

## Evolution Streaming Telemetry Update - 2026-06-04

`--telemetry-selection all` now preserves all candidate telemetry without returning full step traces through multiprocessing. Workers stream replay-compatible JSONL files directly into `selected_telemetry/` using per-candidate temporary files, then atomically rename them when complete. The parent process keeps only compact scoring rows in memory and writes the manifest after the run.

This prevents the `MemoryError` seen when long-horizon all-candidate telemetry tried to pickle large trace payloads through `concurrent.futures.process`.

Recommended for large debug runs:

- write `--output-dir` outside OneDrive, e.g. `C:\f1rl-artifacts\...`;
- keep `--telemetry-selection all` when full debugging/replay is needed;
- use `leaders` for high-throughput search when full replay for every candidate is not required.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py::test_evolution_search_can_save_all_candidate_telemetry_for_swarm_replay
```

Result:

- Ruff passed.
- Multi-worker all-candidate telemetry test passed.

Run:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-0to1220-100x10-stream-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1220 --action-set racing --observation-profile racing_v2 --max-steps 10000 --population 100 --generations 10 --elite-count 16 --random-immigrants 20 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles risk_seeking,max_progress,clean_exit --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation
```

Result:

- completed without the previous multiprocessing telemetry `MemoryError`;
- wrote 1000 candidate traces;
- best distance reached `1130.4m`;
- no `1220m` segment completion yet.

## Open-Distance Evolution Update - 2026-06-04

Added `--no-target-termination` for evolutionary search. In this mode, `--target-progress-m` is only a milestone/completion-rate threshold for scoring and reporting. It does not install `segment_length_m`, so candidates keep driving until normal simulator termination: lap complete, collision, off-track, no-progress, or `--max-steps`.

This fixes the capped-search behavior where a run could reach the target and stop before showing how much farther the controller could brute-force.

Additional search pressure changes:

- open-distance scoring now rewards progress beyond the milestone instead of capping base progress at `target_progress_m`;
- `frontier` and `risk_seeking` scoring profiles are more distance/speed-biased;
- controller mutation is more aggressive and resets more weights;
- parent selection is more rank-biased;
- generation logs now print `leader_progress_m`, `generation_best_distance_m`, `generation_average_distance_m`, `global_best_distance_m`, `completion_rate`, and `milestone_rate`.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --headless
```

Result:

- Ruff passed.
- Pyright passed.
- Focused evolution tests passed.
- Replay loaded 10 generation groups and 1000 traces.

Run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 10000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time: `104.03s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2409.611m`;
- best candidate: generation `9`, candidate `72`;
- best termination: `collision`, not simulator segment completion (`sim_segment_complete=false`);
- final speed: `300.68 kph`;
- final lateral error: `17.34m`;
- final heading error: `-4.20deg`;
- milestone counts across all attempts: `88 >=1000m`, `54 >=1220m`, `14 >=1500m`, `14 >=2000m`, `1 >=2400m`.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## General 100x30 Compute Probe - 2026-06-04

Ran a fresh normal-start open-distance evolutionary search from scratch to test whether more compute solves the frontier without segment initialization.

Command:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x30-general-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 30 --elite-count 12 --random-immigrants 24 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 30 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,frontier,risk_seeking,max_progress,early_pace,clean_distance,clean_exit,exit_speed,frontier_recovery,frontier_novelty --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2000 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 80 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time including 3000-trace packaging: `999.76s`;
- attempts: `3000`;
- saved telemetry traces: `3000`;
- best distance: `5223.047m`;
- best candidate: generation `16`, candidate `73`;
- source: `offspring` from `cleanest_distance`;
- parent: generation `15`, candidate `1`;
- mutation: `controller_masked_gaussian`;
- termination: `collision`;
- final speed: `214.33 kph`;
- no valid lap completion yet (`0 >=5793m`);
- counts: `414 >=1500m`, `385 >=2000m`, `275 >=2200m`, `245 >=2400m`, `151 >=2600m`, `72 >=3000m`, `66 >=4000m`, `54 >=5000m`;
- best reached `parabolica_finish`, about `37m` before its braking gate at `5260m`;
- best final behavior: full throttle, no brake, lateral error drifted to `20.59m`, heading error about `-33.8deg`, then collision.

Diagnosis:

- More compute did solve multiple previous frontiers. The run broke past Rettifilo/Roggia/Lesmo and reached the final sector.
- It did not finish the lap. The new blocker is Parabolica entry braking/positioning.
- Plateau mode preserved the best but was disruptive after activation; average distance dropped immediately after plateau-triggered generations. For late full-lap refinement, plateau extra mutation should probably be reduced or only applied to a smaller branch.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x30-general-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## Frontier Recovery / Roggia Segment Update - 2026-06-04

Implemented frontier-specific tools after the normal-start search plateaued around Roggia exit:

- added `frontier_recovery` scoring to prefer alive, controlled exits through a configurable focus band;
- added `frontier_novelty` scoring to keep useful variation near the frontier;
- added a `frontier_distance` parent bucket for rare far candidates;
- added plateau mode, which detects flat best distance with improving average distance, reduces elite-copy pressure, and applies extra mutations to frontier parents;
- added ladder rungs for `roggia-exit-lesmo-1900-2800` and `roggia-lesmo-transfer-2000-3030`;
- exposed frontier and plateau knobs through `f1-evolution-search` and `f1-evolution-ladder`.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py src/f1rl/evolution_ladder.py tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py src/f1rl/evolution_ladder.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-frontier-cli-smoke --start-progress-m 2200 --start-speed-kph 180 --target-progress-m 2600 --action-set straight --observation-profile base --max-steps 32 --population 8 --generations 2 --elite-count 2 --random-immigrants 2 --top-k 2 --workers 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,clean_distance --frontier-focus-start-m 2200 --frontier-focus-end-m 2600 --frontier-parent-min-progress-m 2200 --progress-every-generation --telemetry-selection leaders
uv run --no-sync python -m f1rl.evolution_ladder --output-dir artifacts\evolution-ladder-frontier-smoke --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base --scoring-profiles max_progress,frontier_recovery,frontier_novelty --progress-every-generation
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Result:

- focused Ruff passed;
- focused Pyright passed;
- focused evolution tests passed;
- frontier evolution CLI smoke passed;
- ladder smoke with Roggia rungs passed;
- full Ruff passed;
- full Pyright passed;
- full pytest passed.

Focused segment run:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-roggia-1900-2800-frontier-100x10-20260604 --start-progress-m 1900 --start-speed-kph 315 --target-progress-m 2800 --target-min-speed-kph 120 --target-max-speed-kph 245 --target-max-lateral-error-m 16 --target-max-heading-error-deg 34 --action-set racing --observation-profile racing_v2 --max-steps 1200 --population 100 --generations 10 --elite-count 12 --random-immigrants 20 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 24 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,clean_distance,exit_speed,max_progress,risk_seeking --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 2000,2200,2400,2600,2800 --survival-floor-pass-rate 0.25 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2200 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 60 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
```

Focused segment result:

- elapsed time: `69.98s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2800.629m`;
- best selected candidate: generation `8`, candidate `85`;
- selected candidate source: `offspring` from `fastest_pace`;
- selected candidate ended with `segment_complete`;
- final speed: `216.92 kph`;
- completion rate improved to `17%`;
- replay loaded `10` groups and `1000` traces.

Normal-start transfer probe:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-frontier-plateau-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 24 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2000 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 80 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
```

Normal-start transfer result:

- elapsed time: `185.25s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2157.566m`;
- best selected candidate: generation `5`, candidate `15`;
- selected candidate source: `offspring` from `frontier_distance`;
- selected candidate ended with `collision`;
- final speed: `137.18 kph`;
- `61` attempts reached `>=1500m`;
- `45` attempts reached `>=2000m`;
- `0` attempts reached `>=2200m`;
- replay loaded `10` groups and `1000` traces.

Diagnosis:

- The focused segment search solved a local Roggia-exit skill.
- The normal-start probe did not transfer that skill and regressed from the previous `2457.633m` best.
- The frontier-recovery profiles are useful for isolated segment discovery but too local/recovery-biased when used as the leading normal-start full-lap objective.
- Next full-lap search should keep `frontier_recovery`/`frontier_novelty` as secondary diversity profiles, while restoring `frontier_fast`/`farthest_distance` as the primary full-lap profiles.

Follow-up refinement:

- adaptive immigrant pressure now uses the broader quality pass rate as well as the active survival floor pass rate;
- this keeps more slots assigned to offspring/smart near-elite mutation after the population stops dying early, even when the active parent floor has moved outward.

Second run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-adaptive-quality-v2-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Second run result:

- elapsed time: `132.72s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2457.633m`;
- best candidate: generation `6`, candidate `50`;
- best candidate source: `offspring` from `survival_gate`;
- best parent: generation `5`, candidate `83`;
- best mutation: `controller_masked_gaussian`;
- best termination: `no_progress`;
- average distance improved from `71.08m` in generation 0 to `840.55m` in generation 9;
- total counts: `589 >=100m`, `473 >=450m`, `78 >=1000m`, `41 >=1220m`, `39 >=1500m`, `35 >=2000m`, `7 >=2400m`;
- all-candidate replay loaded `10` groups and `1000` traces.

Second run replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-adaptive-quality-v2-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## Adaptive Evolution Selection Update - 2026-06-04

Implemented the next aggressive evolution-search pass:

- performance-based dynamic survival floor for parent selection;
- tiered parent buckets: survival gate, farthest distance, fastest pace, cleanest distance;
- adaptive immigrant budget so later generations spend more slots on offspring when survival improves;
- smart immigrants split from pure random immigrants, mostly near current elites with some global archive support;
- conservative pace-sensitive scoring profiles: `frontier_fast`, `early_pace`, `clean_distance`, `farthest_distance`;
- lineage logging for parent, crossover partner, mutation type/sigma/reset count, genome distance, immigrant type, and source bucket;
- generation summaries now include survival floor/pass rates, lineage source counts, progress threshold counts, average pace, and next-population reproduction counts.

Research basis:

- performance-based exploit/explore mirrors Population Based Training style adaptation;
- keeping multiple useful buckets follows quality-diversity / MAP-Elites style pressure;
- smart archive/current elite reuse follows the hard-exploration lesson of returning to promising states before exploring.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-adaptive-cli-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 2 --top-k 2 --workers 1 --genome-type controller --scoring-profiles frontier_fast,early_pace,clean_distance,farthest_distance --progress-every-generation --telemetry-selection leaders
uv run --no-sync python -m f1rl.evolution_ladder --output-dir artifacts\evolution-ladder-adaptive-smoke --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base --progress-every-generation
```

Result:

- focused Ruff passed;
- focused evolution tests passed;
- focused Pyright passed;
- full Ruff passed;
- full Pyright passed;
- full pytest passed;
- evolution-search adaptive CLI smoke passed;
- evolution-ladder smoke passed.

Run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-adaptive-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time: `157.20s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2448.570m`;
- best candidate: generation `7`, candidate `80`;
- best candidate source: `smart_immigrant` from `smart_current_elite`;
- best parent: generation `6`, candidate `55`;
- best mutation: `controller_weight_reset`;
- best termination: `collision`;
- final speed: `169.45 kph`;
- average distance improved from `71.08m` in generation 0 to `766.60m` in generation 9;
- final generation counts: `63 >=450m`, `18 >=1000m`, `15 >=1220m`, `11 >=1500m`, `10 >=2000m`, `1 >=2400m`;
- all-candidate replay loaded `10` groups and `1000` traces.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-adaptive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.
