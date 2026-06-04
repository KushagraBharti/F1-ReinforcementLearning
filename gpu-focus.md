# GPU-Focused Batched Simulation Plan

Date: 2026-06-04.

This plan describes how to add a CUDA/PyTorch batched simulation backend to F1RL without rewriting or weakening the existing project architecture.

The core decision is:

> Do not port the app to CUDA. Add a second batched GPU backend for high-throughput evolutionary evaluation, while keeping the current CPU `MonzaSim` as the correctness, replay, manual, scripted, telemetry, PPO, and test reference.

## Current Context

Verified current project shape:

- The repo is a custom top-down Monza simulator and learning stack.
- The explicit runtime path remains:

`track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium/SB3 PPO -> eval/replay`

- The current discovery path is evolution-first:

`shared simulator -> evolutionary controller search -> elite telemetry/state libraries -> PPO transfer later`

Verified latest search result:

- Preserved evolution/search artifact scale: `27650` candidate evaluations across `76` run artifacts and `340` generations.
- Latest large run: `150x60`, `9000` controllers, `60` generations.
- Latest run valid laps: `926`.
- Latest fastest valid evolved rolling-start lap: `89.983333s`.
- Fast-F1 reference target: `79.662s`.

Verified hardware check:

- PyTorch installed: yes.
- CUDA available: yes.
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Torch version: `2.10.0+cu128`.
- CUDA runtime: `12.8`.
- Current compute policy says neural training/inference use CUDA, but env stepping, physics, geometry, rendering, telemetry, and track preprocessing remain CPU.

Current CPU bottleneck:

- Evolution evaluates independent candidates through CPU `MonzaSim`.
- The latest full run works behaviorally, but candidate throughput is still mostly limited by CPU stepping, geometry, scoring, and telemetry.
- The project wants much more brute-force search: more candidates, more generations, more valid laps, and faster lap-time optimization.

Why GPU now:

- Evolution is naturally batchable.
- Each candidate is independent.
- Each generation has many candidates executing the same physics/control/scoring loop.
- The latest search can use batches shaped like `[population]` or `[population, variants_per_parent]`.
- PPO/SB3 is not the first GPU target because Gym/VecEnv/SB3 APIs are NumPy/CPU-oriented and can accidentally burn time copying tensors between CPU and GPU every step.

## Non-Negotiables

1. `MonzaSim` stays the truth/reference backend.
2. Existing CPU behavior must keep working.
3. Manual driving stays CPU.
4. Pygame rendering and replay stay CPU.
5. JSONL/gzip telemetry writing stays CPU.
6. Track image preprocessing stays CPU.
7. PPO/SB3 compatibility stays intact.
8. Existing artifacts and replay commands stay readable.
9. Evolution CLI output schemas stay stable.
10. GPU search results must be replayable through CPU `MonzaSim`.
11. GPU backend must be optional behind a backend flag.
12. No final result is trusted until a selected top candidate replays correctly through CPU `MonzaSim`.

Correctness target:

- Behavioral parity, not bitwise identity.
- CUDA floating-point order will differ from NumPy CPU floating-point order.
- Accept tolerances for position, speed, heading, progress, and scores.
- Require qualitative and rank-level stability before relying on GPU search for serious runs.

## The Clean Architecture

Current CPU backend:

```text
EvolutionSearch
  -> _run_candidate()
  -> MonzaSim.reset()
  -> MonzaSim.search_features()
  -> _controller_controls()
  -> MonzaSim.step_controls()
  -> telemetry row list
  -> _score_profile()
  -> attempts/generation summaries/replay artifacts
```

New GPU backend:

```text
EvolutionSearch
  -> EvolutionEvaluationBackend
       -> CpuEvolutionBackend
       -> GpuEvolutionBackend
  -> GpuMonzaBatch.reset_batch()
  -> GpuMonzaBatch.step_batch()
  -> Gpu controller features
  -> Gpu controller forward
  -> Gpu progress/termination/scoring accumulators
  -> compact result rows
  -> CPU replay verification for selected/top candidates
  -> same attempts/generation summaries/replay artifacts
```

Key idea:

- GPU backend evaluates many candidates quickly.
- CPU backend remains authoritative for replay, telemetry, validation, and final promotion.
- The GPU path is a high-throughput scorer/filter first.
- The CPU path verifies and materializes selected candidates.

## Proposed New Modules

Add these files only when implementation starts:

- `src/f1rl/gpu_types.py`
- `src/f1rl/gpu_track.py`
- `src/f1rl/gpu_physics.py`
- `src/f1rl/gpu_features.py`
- `src/f1rl/gpu_scoring.py`
- `src/f1rl/gpu_batch.py`
- `src/f1rl/evolution_backend.py`

Possible test files:

- `tests/test_gpu_physics.py`
- `tests/test_gpu_track.py`
- `tests/test_gpu_features.py`
- `tests/test_gpu_scoring.py`
- `tests/test_gpu_evolution_backend.py`

Do not modify first:

- `src/f1rl/manual.py`
- `src/f1rl/replay.py`
- `src/f1rl/render.py`
- `src/f1rl/train.py`
- `src/f1rl/env.py`
- `src/f1rl/sim.py`

Touch `evolution_search.py` only at the integration layer:

- config fields;
- CLI flags;
- backend selection;
- CPU replay verification;
- artifact routing.

## Backend Interface

Create a small backend abstraction instead of scattering `if backend == "gpu"` across evolution search.

Suggested interface:

```python
class EvolutionEvaluationBackend(Protocol):
    name: str

    def evaluate_population(
        self,
        *,
        candidates: list[Candidate],
        snapshots: list[StateSnapshot],
        sim_config: SimConfig,
        gates: EvolutionGates,
        generation: int,
        seed: int,
        scoring_profiles: tuple[str, ...],
        frontier_focus_start_m: float,
        frontier_focus_end_m: float,
        capture_step_telemetry: bool,
        stream_telemetry_dir: Path | None,
        telemetry_compression: str,
    ) -> list[dict[str, Any]]:
        ...
```

CPU backend:

- Wraps the current `_evaluate_population()`.
- Behavior should be unchanged.

GPU backend:

- Converts candidates/snapshots/config to tensors.
- Evaluates rollout batch on CUDA.
- Returns compact result rows with the same top-level keys expected by evolution search.
- Optionally calls CPU replay verification for selected candidates.

Why this matters:

- Keeps risk contained.
- Makes `--backend cpu|gpu` real.
- Allows tests to run both backends against the same candidate set.
- Prevents GPU code from contaminating manual/PPO/replay paths.

## GPU State Layout

The core GPU backend should own tensors on one device.

Primary shape:

- `[N]`, where `N = population`.

Later optional shape:

- `[parents, variants_per_parent]`, then flatten to `[N]` for scoring.

State tensors:

- `x_px`: `[N]`.
- `y_px`: `[N]`.
- `heading_rad`: `[N]`.
- `speed_mps`: `[N]`.
- `yaw_rate_rps`: `[N]`.
- `steering`: `[N]`.
- `raw_progress_m`: `[N]`.
- `monotonic_progress_m`: `[N]`.
- `last_raw_progress_px`: `[N]`.
- `checkpoint_index`: `[N]`, int64.
- `lap_index`: `[N]`, int64.
- `missed_checkpoint_count`: `[N]`, int64.
- `elapsed_steps`: `[N]`, int64.
- `no_progress_steps`: `[N]`, int64.
- `alive`: `[N]`, bool.
- `terminated`: `[N]`, bool.
- `truncated`: `[N]`, bool.
- `termination_reason_id`: `[N]`, int64.

Last-control tensors:

- `last_throttle`: `[N]`.
- `last_brake`: `[N]`.
- `last_steer`: `[N]`.

Controller tensors:

- `controller_weights`: `[N, 3, feature_count]`.
- `feature_vector`: `[N, feature_count]`.
- `throttle`: `[N]`.
- `brake`: `[N]`.
- `steer`: `[N]`.

Scoring accumulator tensors:

- `best_progress_m`: `[N]`.
- `best_speed_kph`: `[N]`.
- `best_lateral_error_m`: `[N]`.
- `best_heading_error_deg`: `[N]`.
- `final_progress_m`: `[N]`.
- `final_speed_kph`: `[N]`.
- `final_lateral_error_m`: `[N]`.
- `final_heading_error_deg`: `[N]`.
- `final_yaw_rate_rps`: `[N]`.
- `final_steering`: `[N]`.
- `finish_crossed`: `[N]`.
- `valid_lap`: `[N]`.
- `completed_lap`: `[N]`.
- `collided`: `[N]`.
- `off_track`: `[N]`.
- `segment_complete`: `[N]`.
- `target_reached`: `[N]`.
- `elapsed_s`: `[N]`.
- `pace_kph`: `[N]`.
- `time_to_300_m`: `[N]`, sentinel for missing.
- `time_to_450_m`: `[N]`, sentinel for missing.
- `speed_sum_first_300_m`: `[N]`.
- `speed_count_first_300_m`: `[N]`.
- `speed_sum_first_450_m`: `[N]`.
- `speed_count_first_450_m`: `[N]`.
- `brake_sum_first_450_m`: `[N]`.
- `brake_count_first_450_m`: `[N]`.

Profile score tensors:

- `profile_scores`: `[N, P]`, where `P = number of scoring profiles`.
- `primary_score`: `[N]`.

Data types:

- Start with `torch.float64` parity mode for tests.
- Use `torch.float32` performance mode for large runs after parity is proven.
- Int tensors should be `torch.int64`.
- Masks should be `torch.bool`.

Device policy:

- `device="cuda"` for production GPU search.
- `device="cpu"` for GPU-backend unit tests that should run without CUDA.
- Tests that require real CUDA should be skipped if CUDA is unavailable.

## Track Data On GPU

Track tensors:

- `centerline_xy`: `[S, 2]`.
- `centerline_next_xy`: `[S, 2]`.
- `centerline_segment_vec`: `[S, 2]`.
- `centerline_segment_len2`: `[S]`.
- `centerline_cumdist_px`: `[S]`.
- `boundary_segments`: `[B, 4]`.
- `checkpoint_gates`: `[C, 4]`, if needed for lap validity.
- `drivable_mask`: `[H, W]`, bool or uint8.
- `meters_per_pixel`: scalar.
- `length_px`: scalar.
- `length_m`: scalar.

Hard geometry parts:

- centerline projection;
- signed lateral error;
- heading error;
- lookahead heading errors;
- checkpoint validity;
- off-track mask lookup;
- collision against boundary segments;
- raycasts for PPO observations.

Recommended geometry order:

1. Centerline projection.
2. Lateral and heading error.
3. Lookahead heading errors.
4. Drivable mask lookup.
5. Finish/checkpoint/lap validation.
6. Boundary collision.
7. Raycasts only later.

Why raycasts are not Phase 1:

- Current evolution controller search uses `search_features()`, not PPO raw ray observations.
- The active controller features are speed, target speed, brake demand, lookahead, lateral/heading error, curvature, target steer, last controls, and progress.
- Raycasts matter more for PPO observations and manual/replay visualization.
- Skipping raycasts in the first GPU scorer reduces geometry complexity without breaking the current evolution target.

## Phase 0: Baseline And Safety Audit

Goal:

- Establish CPU reference behavior and performance before adding GPU code.

Actions:

1. Freeze a small set of deterministic candidate genomes:
   - one straight-ish controller;
   - one braking-heavy controller;
   - one high-steer controller;
   - one known valid/fast archived genome if easy to load;
   - one phase genome;
   - one progress-phase genome.
2. Save or generate fixed start snapshots:
   - normal start at `0m`, `80kph`;
   - Rettifilo around `520m`;
   - mid-lap around `2500m`;
   - late lap around `5000m`.
3. Run CPU backend and save compact reference outputs:
   - final x/y/heading/speed;
   - best progress;
   - raw progress;
   - lateral error;
   - heading error;
   - termination reason;
   - valid-lap state;
   - primary/profile scores.
4. Add a small benchmark command that records:
   - candidates/sec;
   - sim steps/sec;
   - GPU memory if available;
   - CPU time in scoring;
   - CPU time in telemetry writing.

Acceptance:

- Reference fixture files exist under `tests/fixtures/gpu_reference/` or are generated deterministically inside tests.
- Current CPU backend behavior stays unchanged.
- Current test suite still passes.

Suggested commands:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json
```

## Phase 1: Batched PyTorch Physics

Goal:

- Port `apply_physics()` to a batched tensor function.

Current CPU reference:

- `src/f1rl/physics.py::apply_physics`.

New function:

```python
def apply_physics_batch(
    state: GpuCarBatch,
    *,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    steer: torch.Tensor,
    params: GpuCarParams,
    meters_per_pixel: torch.Tensor,
) -> tuple[GpuCarBatch, GpuMovementBatch]:
    ...
```

Implementation details:

- Use `torch.clamp`, `torch.tan`, `torch.cos`, `torch.sin`, `torch.sqrt`.
- Avoid Python loops over candidates.
- Use mask operations for stopped cars.
- Keep formulas visibly equivalent to CPU `apply_physics()`.
- Keep heading wrap formula equivalent:

```text
(heading + yaw_rate * dt + pi) % (2*pi) - pi
```

Physics parity tests:

1. Single car, one step, straight throttle.
2. Single car, one step, full brake.
3. Single car, one step, max steer.
4. Batch of 32 cars, same action as CPU loop.
5. Batch of 32 cars, different actions.
6. Multi-step fixed action sequence, e.g. `300` steps.

Tolerance targets:

- `speed_mps`: within `1e-5` in float64 parity mode.
- `heading_rad`: within `1e-5` in float64 parity mode.
- `x/y`: within `1e-4px` to `1e-3px` after short sequences.
- Relax tolerances for float32 performance mode.

Acceptance:

- Batched physics on CPU device matches CPU `apply_physics()` within tolerance.
- Batched physics on CUDA matches CPU reference within performance-mode tolerance.
- No evolution integration yet.

## Phase 2: Batched Controller Forward

Goal:

- Port `_controller_controls()` to PyTorch batch operations.

Current CPU reference:

- `src/f1rl/evolution_search.py::_controller_controls`.

New function:

```python
def controller_controls_batch(
    controller_weights: torch.Tensor,  # [N, 3, F]
    features: torch.Tensor,            # [N, F]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

Logic:

- logits = batch matrix multiply.
- throttle raw = sigmoid(logit 0).
- brake raw = sigmoid(logit 1).
- exclusive throttle/brake suppression matches CPU:
  - if brake raw > throttle raw, reduce throttle;
  - else reduce brake.
- steer = tanh(logit 2).
- clamp outputs.

Controller tests:

1. Random features, random weights, CPU NumPy vs torch CPU.
2. Random features, random weights, CPU NumPy vs torch CUDA.
3. Edge logits near large positive/negative values.
4. Brake dominates throttle.
5. Throttle dominates brake.

Acceptance:

- Controls match CPU reference within tolerance.
- Controller forward supports `[N]` batch size from `1` to at least `8192`.

## Phase 3: Batched Search Features

Goal:

- Port the subset of `MonzaSim.search_features()` required by controller evolution.

Current CPU reference:

- `src/f1rl/sim.py::MonzaSim.search_features`.

GPU features to compute:

- `bias`.
- `speed_norm`.
- `target_speed_norm`.
- `speed_error_norm`.
- `brake_demand`.
- `future_brake_demand`.
- `target_speed_drop_norm`.
- `brake_gate_proximity`.
- `brake_gate_distance_norm`.
- `lookahead_abs_max`.
- `signed_lateral_error_norm`.
- `heading_error_norm`.
- `yaw_rate_norm`.
- `curvature_norm`.
- `target_steer`.
- `last_throttle`.
- `last_brake`.
- `last_steer`.
- `segment_progress_ratio`.
- `lookahead_0`.
- `lookahead_1`.
- `lookahead_2`.
- `lookahead_3`.

Additional compact diagnostic values:

- `lateral_error_m`.
- `signed_lateral_error_m`.
- `heading_error_deg`.
- `target_speed_kph`.
- `near_target_speed_kph`.
- `min_future_target_speed_kph`.
- `target_speed_drop_kph`.
- `braking_gate_distance_m`.
- `speed_kph`.

Centerline projection plan:

- Start with all-segment projection over the persisted centerline.
- For each car:
  - vector from segment start to car position;
  - project onto each segment;
  - clamp projection ratio to `[0, 1]`;
  - compute closest projected point;
  - choose minimum squared distance.
- This is `O(N*S)` where `S` is about `120`, which is cheap for early batches.
- Later optimize using last-known segment windows if needed.

Lateral sign:

- Use cross product between segment direction and car offset.
- Match CPU convention through tests.

Heading error:

- Segment heading from centerline tangent.
- Wrap heading difference to `[-pi, pi]`.

Lookahead:

- Convert current projected progress to lookahead progress.
- Find target centerline points at `40m`, `90m`, `160m`, `280m`.
- Compute heading errors.

Braking gates:

- Port `distance_to_next_braking_gate()` logic to tensor form.
- Use Monza section brake gates as constant tensors.
- Compute distance to next gate with wraparound.

Target speed:

- Match current `MonzaSim._target_speed_kph()` logic using lookahead heading errors.
- Match reward config min/max/scale constants.

Target steer:

- Match current `MonzaSim._target_steer()` sufficiently for controller parity.
- This is important because fastest evolved controllers use `target_steer`.

Feature parity tests:

1. Fixed state at normal start.
2. Fixed state near Rettifilo brake gate.
3. Fixed state near Roggia.
4. Fixed state near Ascari.
5. Fixed state near Parabolica.
6. Batch of random points near track centerline.

Tolerance targets:

- Feature values within `1e-4` to `1e-3` for float64 parity.
- Target speed within `0.1kph`.
- Heading error within `0.1deg`.
- Lateral error within `0.1m`.

Acceptance:

- GPU features match CPU `search_features()` on representative states.
- Biggest mismatches are documented, not ignored.

## Phase 4: Batched Termination And Validity

Goal:

- Implement enough termination logic for evolution scoring to be meaningful and replay-verifiable.

Termination fields:

- `off_track`.
- `collision`.
- `no_progress`.
- `lap_complete`.
- `segment_complete`.
- `segment_gate_failed`.
- `max_steps`.

Recommended order:

1. Off-track mask lookup.
2. No-progress counter.
3. Segment target reached.
4. Finish crossing/lap-complete approximation.
5. Checkpoint validity.
6. Boundary segment collision.

Off-track lookup:

- Convert x/y to integer pixel coordinates.
- Bounds check.
- Lookup `drivable_mask[y, x]`.
- Off-track if out-of-bounds or mask false.

No-progress:

- If progress delta <= `1e-4`, increment `no_progress_steps`.
- Else reset to zero.
- Terminate at CPU config threshold.

Progress delta:

- Compute raw progress in px from projection.
- Handle wraparound:
  - if raw delta < `-0.5 * length_px`, add length;
  - if raw delta > `0.5 * length_px`, subtract length.
- Clamp negative deltas to zero.
- If progress delta exceeds local projection window, set to zero.

Checkpoint validity:

- MVP can start with finish and progress validity.
- Full parity should track checkpoint crossings and missed checkpoints.
- For serious full-lap claims, checkpoint semantics must match CPU.

Collision:

- CPU uses movement segment vs `1800` boundary segments.
- GPU collision can be expensive but batchable.
- MVP can rely on off-track mask for search filtering, but CPU replay verification must catch any mismatch.
- Full parity should implement segment intersection in batches:
  - movement `[N, 4]`;
  - boundary `[B, 4]`;
  - intersection `[N, B]`;
  - any over boundary dimension.
- Chunk boundary checks if memory is too high.

Acceptance:

- For controlled tests, CPU and GPU agree on:
  - off-track;
  - no-progress;
  - target reached;
  - lap complete;
  - valid lap;
  - checkpoint validity;
  - collision for curated movement segments.
- Top GPU candidates replay through CPU without large score/rank surprises.

## Phase 5: Batched Rollout Loop

Goal:

- Evaluate an entire generation on GPU.

Core loop:

```python
with torch.no_grad():
    batch.reset(...)
    for step in range(max_steps):
        active = batch.alive & ~batch.terminated & ~batch.truncated
        if not active.any():
            break
        features = batch.search_features()
        throttle, brake, steer = controller_controls_batch(weights, features)
        batch.step(throttle, brake, steer, active=active)
        scorer.update(batch, features, controls, active)
```

Important performance rules:

- No `.item()` inside the per-step loop.
- No `.cpu()` inside the per-step loop.
- No JSON writing inside the per-step loop.
- No Python loop over candidates.
- Only synchronize CUDA for measurement boundaries.
- Materialize compact result rows after the rollout.

State update masks:

- For inactive candidates, keep state unchanged.
- For newly terminated candidates, freeze final state.
- Continue active candidates until all done or max steps.

Telemetry strategy:

- Do not write full JSONL for every GPU candidate during the hot scoring loop.
- Record compact accumulators for all candidates.
- Replay selected top candidates through CPU `MonzaSim` to produce authoritative full telemetry.
- Preserve existing CPU all-candidate telemetry mode for debug runs.
- Add optional GPU debug trace capture later, but keep it off for large runs by default.

Why:

- Full all-candidate step telemetry for `150x60` at `18000` steps can dominate runtime and storage.
- The GPU win comes from keeping rollout and scoring on-device.
- Full telemetry is still preserved for selected/top candidates through CPU replay.
- If all-candidate full telemetry is required, use CPU backend or accept a slower GPU+CPU replay-all mode.

## Phase 6: GPU Scoring Accumulators

Goal:

- Stop requiring full row lists for scoring during GPU search.

Current CPU scoring:

- `_score_profile(rows, ...)` operates on list-of-dict step rows.

GPU scoring should use accumulated tensors:

- best progress;
- final row fields;
- elapsed time;
- early pace metrics;
- gate crossing times;
- average speed/brake before gates;
- valid-lap flags;
- termination flags;
- final lateral/heading/yaw/steering;
- demand-region control summaries if needed.

MVP profiles to port first:

- `max_progress`.
- `farthest_distance`.
- `frontier_fast`.
- `early_pace`.
- `fast_frontier`.
- `lap_pace`.
- `fast_valid_lap`.
- `time_attack`.

Profiles to port after MVP:

- `clean_distance`.
- `clean_exit`.
- `exit_speed`.
- `risk_seeking`.
- `frontier`.
- `frontier_recovery`.
- `frontier_novelty`.
- `full_lap_validity`.
- `brake_zone`.
- `apex`.

Scoring parity tests:

1. Feed synthetic accumulated metrics matching a CPU trace.
2. Compare GPU profile scores against CPU `_score_profile()` on replayed rows.
3. Compare ranking over 100 fixed candidates:
   - Spearman rank correlation;
   - top-10 overlap;
   - profile winner overlap.

Minimum acceptance:

- For MVP profiles, CPU/GPU score differences are within tolerance or explained by documented approximation.
- Top candidates selected by GPU replay well through CPU.
- GPU does not promote obvious invalid candidates as valid.

## Phase 7: Evolution Search Integration

Goal:

- Add GPU backend to the existing `f1-evolution-search` CLI.

New flags:

```powershell
--backend cpu|gpu
--gpu-device cuda|cuda:0|cpu
--gpu-dtype float32|float64
--gpu-batch-size auto|N
--gpu-verify-top-k N
--gpu-verify-elite-multiplier N
--gpu-telemetry-mode selected|top|all-cpu-replay|none
--gpu-parity-check
--gpu-fallback-to-cpu
```

Defaults:

- `--backend cpu`.
- GPU must be explicitly requested at first.
- Later, `--workers 0 --backend gpu` can become the recommended large-run path.

Verification flow:

1. GPU scores all candidates.
2. Select a verification set:
   - top K by primary profile;
   - top K per scoring profile;
   - candidate rows needed for elite selection;
   - random sample for rank sanity.
3. Replay verification set through CPU `MonzaSim`.
4. Replace verified rows with CPU-authoritative rows.
5. Use verified rows for:
   - saved selected telemetry;
   - top genomes;
   - best-so-far;
   - elite promotion;
   - final claims.

Parent selection question:

- For early GPU backend, it is acceptable to use GPU compact rows for broad parent buckets and CPU-verified rows for elites/top candidates.
- For final strict mode, parent selection should use CPU-verified top rows for elite preservation and GPU rows for lower-tier diversity.

Output schema:

- Keep `attempts.jsonl` compatible.
- Add backend fields:
  - `backend`: `gpu`.
  - `gpu_verified`: true/false.
  - `gpu_score`: value.
  - `cpu_verified_score`: value if replayed.
  - `cpu_replay_telemetry`: path if replayed.
  - `backend_parity_error`: optional summary.

Generation summary additions:

- `backend`.
- `gpu_candidates_per_second`.
- `gpu_steps_per_second`.
- `cpu_verification_count`.
- `cpu_verification_seconds`.
- `gpu_memory_allocated_gb`.
- `gpu_memory_reserved_gb`.
- `gpu_rank_cpu_top_overlap`.
- `gpu_cpu_score_correlation`.

Acceptance:

- `--backend cpu` unchanged.
- `--backend gpu` writes compatible `attempts.jsonl`, `generation_summary.jsonl`, `evolution_summary.json`.
- Replay commands still work for selected telemetry.
- CPU verification prevents false promotion.

## Phase 8: Telemetry And Replay Contract

Keep current functionality:

- CPU backend can still write all-candidate full telemetry.
- Replay still reads existing manifests and `.jsonl.gz`.
- Selected top candidates still get full replayable telemetry.

New GPU modes:

1. `selected`
   - Fast default.
   - GPU evaluates all.
   - CPU replays selected/top candidates only.

2. `top`
   - CPU replays top K per generation/profile.
   - Good for debugging serious runs.

3. `all-cpu-replay`
   - GPU ranks/evaluates all.
   - CPU replays all candidates after ranking.
   - Preserves all-candidate full telemetry, but will be slow.
   - Use for `20x4`, `100x5`, and maybe `100x10`, not huge runs.

4. `none`
   - GPU scoring only.
   - No full telemetry.
   - Only for perf smoke, not serious learning claims.

5. Future `gpu-compact-trace`
   - GPU stores compact per-step tensors for selected fields.
   - Later write JSONL/gzip from tensors.
   - Not Phase 1.

Important:

- Do not delete the existing all-candidate telemetry path.
- Do not pretend `selected` mode gives the same debugging value as `all`.
- For large GPU runs, use selected/top telemetry plus repeatable CPU replay verification.

## Phase 9: Performance Targets

Current reference numbers:

- Early CPU audit: about `0.82 candidates/sec` for `64x2` with workers `2`.
- Later optimized CPU search varied by run/generation, often around `1-5 candidates/sec` for long full-lap runs with telemetry.
- Latest final generation logged about `4.131 candidates/sec`.

Phase targets:

1. Physics-only batch:
   - At least `100x` more raw state updates/sec than a Python loop over candidates.

2. Controller+physics+features without full geometry:
   - At least `50x` CPU candidate throughput on short `max_steps` smokes.

3. Full GPU evolution MVP with centerline projection and mask:
   - At least `10x` faster than CPU backend on comparable `100x10` scoring-only run.

4. Serious run target:
   - Make `150x60` feel like a normal iteration, not a multi-hour blocker.
   - Target hundreds of candidates/sec for short-horizon runs.
   - Target tens to hundreds of candidates/sec for long `18000`-step full-lap search, depending geometry and verification settings.

Do not overclaim:

- If all-candidate CPU replay is enabled, telemetry can dominate runtime again.
- GPU throughput should be measured separately for:
  - GPU rollout/scoring time;
  - CPU verification time;
  - telemetry write time;
  - archive/compression time.

## Phase 10: PPO GPU Backend Later

Do not start here.

Why:

- SB3 expects Gym/VecEnv-style CPU/NumPy interaction.
- A GPU sim can lose its benefit if every step copies obs/actions between CPU and GPU.
- Current PPO bottleneck is not only environment stepping; it is discovering the right behavior.
- Evolution can use GPU batching immediately with much less API friction.

Future PPO options:

1. Keep SB3 CPU env and use evolved state libraries.
   - Lowest risk.
   - Best near-term path for PPO transfer.

2. Custom PyTorch rollout collector.
   - Keep sim state, policy forward, rewards, and rollout buffers on GPU.
   - More work, but avoids CPU/GPU sync.

3. TorchRL-style batched environment.
   - Consider after GPU sim parity is proven.
   - Useful if the project moves away from SB3 for GPU-native RL.

4. Warp/custom kernels.
   - Consider only if PyTorch batching hits a wall on geometry/collision.

## What Stays CPU-Bound

Keep these on CPU unless there is a separate proven need:

- `MonzaSim` reference backend.
- `MonzaEnv` Gymnasium/SB3 path.
- Manual driving.
- Pygame rendering.
- Replay visualization.
- JSONL/gzip telemetry writing.
- Track build from images.
- Artifact packaging/archive.
- Final selected telemetry generation.
- Scripted baseline.
- Reference ghost baseline.
- Metadata-faithful PPO eval.

Reason:

- These are correctness/debug/user-facing paths.
- They are not the main bottleneck for brute-force controller search once GPU evaluation exists.

## What Moves GPU-Bound

Move these first:

- Batched physics step.
- Batched controller feature extraction.
- Batched controller forward pass.
- Batched progress/lateral/heading calculations.
- Batched target speed/brake-demand features.
- Batched termination masks.
- Batched scoring accumulators.
- Batched profile scoring.
- Batched candidate ranking tensors.

Move later:

- Batched checkpoint/lap validation.
- Batched boundary collision.
- Batched raycasts.
- Batched PPO observations.

## Parity Contract

Physics parity:

- Same starting state.
- Same controls.
- Same car parameters.
- Same meters per pixel.
- Compare final state after fixed sequences.

Geometry parity:

- Same x/y/heading.
- Compare raw progress, monotonic progress, lateral error, heading error, lookahead errors.

Termination parity:

- Compare off-track, collision, no-progress, segment-complete, lap-complete, valid-lap flags.

Scoring parity:

- Compare profile scores for identical traces/accumulators.
- Compare rank order on candidate batches.

Replay parity:

- Top GPU candidates replay through CPU and remain top candidates.
- CPU replay termination reason should not wildly contradict GPU termination.

Suggested tolerances:

- Position after short fixed action sequence: `<0.05px` in float64, `<0.5px` in float32.
- Speed: `<0.05kph` in float64, `<0.5kph` in float32.
- Heading: `<0.05deg` in float64, `<0.5deg` in float32.
- Progress: `<0.1m` in float64, `<1.0m` in float32.
- Score rank top-10 overlap: start with `>=70%`, target `>=90%` for serious use.
- CPU replay of GPU top K: no invalid false positives among promoted elites.

## Test Plan

Unit tests:

- `test_gpu_physics_one_step_matches_cpu`.
- `test_gpu_physics_multi_step_matches_cpu`.
- `test_gpu_controller_controls_match_numpy`.
- `test_gpu_centerline_projection_matches_cpu`.
- `test_gpu_search_features_match_cpu`.
- `test_gpu_offtrack_mask_matches_cpu`.
- `test_gpu_checkpoint_progress_matches_cpu`.
- `test_gpu_scoring_profiles_match_cpu`.
- `test_gpu_backend_result_schema_matches_cpu_backend`.

Integration tests:

- `--backend gpu` tiny search with `population=8`, `generations=2`.
- `--backend gpu --gpu-device cpu` test for CI without CUDA.
- `--backend gpu --gpu-verify-top-k 4`.
- `--backend gpu --gpu-telemetry-mode selected`.
- CPU replay selected top candidates from GPU run.

Performance tests:

- Physics-only batch benchmark.
- Feature-only batch benchmark.
- Full rollout benchmark at:
  - `20x4`;
  - `100x5`;
  - `100x10`;
  - short `150x5`.

Regression tests:

- Existing CPU evolution tests pass.
- Existing replay tests pass.
- Existing benchmark tests pass.
- Existing PPO/env tests pass.

Validation commands:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json
```

GPU-specific smoke example:

```powershell
uv run --no-sync f1-evolution-search --backend gpu --gpu-device cuda --gpu-dtype float32 --gpu-verify-top-k 4 --gpu-telemetry-mode selected --output-dir artifacts\evolution-gpu-smoke --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 600 --population 16 --generations 2 --elite-count 4 --random-immigrants 4 --top-k 4 --genome-type controller --scoring-profiles fast_frontier,farthest_distance,early_pace,max_progress --progress-every-generation
```

CPU comparison smoke:

```powershell
uv run --no-sync f1-evolution-search --backend cpu --output-dir artifacts\evolution-cpu-parity-smoke --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 600 --population 16 --generations 2 --elite-count 4 --random-immigrants 4 --top-k 4 --workers 1 --genome-type controller --scoring-profiles fast_frontier,farthest_distance,early_pace,max_progress --progress-every-generation
```

## Implementation Order

### Step 1: Add GPU Data Types

Add:

- `GpuCarParams`.
- `GpuTrackTensors`.
- `GpuBatchState`.
- `GpuScoreAccumulator`.
- conversion helpers from existing `CarParams`, `TrackSpec`, `StateSnapshot`, and `Genome`.

No behavior change.

### Step 2: Add Batched Physics

Implement:

- `apply_physics_batch()`.
- CPU-vs-GPU physics tests.

No evolution integration.

### Step 3: Add Batched Controller Forward

Implement:

- `controller_controls_batch()`.
- `genomes_to_controller_weight_tensor()`.
- tests vs `_controller_controls()`.

No evolution integration.

### Step 4: Add Track Projection And Search Features

Implement:

- `project_to_centerline_batch()`.
- `track_errors_batch()`.
- `lookahead_heading_errors_batch()`.
- `distance_to_next_braking_gate_batch()`.
- `search_features_batch()`.

Tests compare to `MonzaSim.search_features()`.

### Step 5: Add Batched Rollout MVP

Implement:

- `GpuMonzaBatch`.
- reset from snapshots.
- step loop.
- off-track via mask.
- no-progress.
- target reached.
- compact result rows.

No full collision/checkpoint parity yet unless it falls out cleanly.

### Step 6: Add GPU Scoring MVP

Implement:

- accumulator updates.
- MVP scoring profiles.
- result row generation.

Run against fixed candidate fixtures.

### Step 7: Add Backend Integration

Implement:

- backend interface.
- CPU backend wrapper.
- GPU backend.
- CLI flags.
- schema-compatible results.
- CPU replay verification of top candidates.

### Step 8: Add Collision/Checkpoint/Lap Validity Parity

Implement:

- checkpoint tracking.
- finish crossing.
- missed checkpoint semantics.
- segment collision batch or mask-equivalent fallback with CPU verification.

This is the step that turns GPU backend from useful filter into serious search backend.

### Step 9: Add Performance Instrumentation

Log:

- GPU rollout time.
- GPU score time.
- CPU verification time.
- telemetry write time.
- candidates/sec.
- sim steps/sec.
- GPU memory allocated/reserved.
- top-k CPU/GPU rank overlap.

### Step 10: Run Scale Ladder

Run in order:

1. `8x2` GPU smoke.
2. `20x4` GPU vs CPU parity.
3. `100x5` GPU search.
4. `100x10` GPU search.
5. `150x10` GPU search.
6. `150x60` GPU comparison only after parity and replay verification are clean.

Do not jump straight to `150x60`.

## Expected File Changes

Low-risk first commit:

- `src/f1rl/gpu_types.py`.
- `src/f1rl/gpu_physics.py`.
- `tests/test_gpu_physics.py`.

Second commit:

- `src/f1rl/gpu_features.py`.
- `src/f1rl/gpu_track.py`.
- `tests/test_gpu_features.py`.

Third commit:

- `src/f1rl/gpu_scoring.py`.
- `tests/test_gpu_scoring.py`.

Fourth commit:

- `src/f1rl/gpu_batch.py`.
- `src/f1rl/evolution_backend.py`.
- changes to `src/f1rl/evolution_search.py`.
- tests for backend CLI/schema.

Why split:

- Physics parity is easier to review than full backend integration.
- Geometry bugs are likely and should be isolated.
- Evolution integration should happen after the tensor pieces are trusted.

## GPU Geometry Detail

### Centerline Projection

Naive all-segment version:

```python
# points: [N, 2]
# seg_start: [S, 2]
# seg_vec: [S, 2]
rel = points[:, None, :] - seg_start[None, :, :]
t = (rel * seg_vec[None, :, :]).sum(-1) / seg_len2[None, :]
t = t.clamp(0.0, 1.0)
proj = seg_start[None, :, :] + t[..., None] * seg_vec[None, :, :]
dist2 = ((points[:, None, :] - proj) ** 2).sum(-1)
segment_idx = dist2.argmin(dim=1)
```

This is simple and likely fine for `S ~= 120`.

Later optimization:

- Keep `last_segment_idx`.
- Search only within a local window.
- Fallback to global search when projected progress jumps too much.

### Boundary Collision

Full batch:

- movement segments `[N, 4]`.
- boundary segments `[B, 4]`, `B=1800`.
- intersection matrix `[N, B]`.

Memory:

- `N=9000`, `B=1800` gives `16.2M` booleans, acceptable if chunked but not free.
- For per-step loop, chunking is safer.

MVP:

- Use drivable mask termination for GPU search.
- CPU replay verification catches promoted collision mismatches.

Serious backend:

- Add boundary segment collision or equivalent edge crossing check.

### Raycasts

Do not implement in the first evolution backend.

Reasons:

- Evolution controller does not need raw rays.
- Raycasts are expensive.
- PPO is not the first GPU target.

Future:

- Batch rays `[N, R]` against boundary segments.
- Use chunking.
- Or precompute distance fields if approximate rays are acceptable for PPO experiments.

## Artifact Contract

GPU run must still write:

- `attempts.jsonl`.
- `generation_summary.jsonl`.
- `evolution_summary.json`.
- `best_so_far.json`.
- `population_checkpoint.json`.
- `top_genomes/*.json`.
- `selected_telemetry/manifest.json`.
- `elite_state_library.json`.
- `ppo_bridge.json`.
- `next_commands.md`.

Additional GPU metadata:

- backend settings.
- device name.
- torch version.
- CUDA version.
- dtype.
- batch size.
- verification mode.
- verification top K.
- GPU/CPU score agreement stats.

Telemetry:

- CPU replay selected candidates into normal JSONL/gzip format.
- If `--telemetry-selection all` is requested with GPU backend:
  - either warn that all-candidate full telemetry requires CPU replay;
  - or run replay-all mode explicitly;
  - never silently skip telemetry.

## Storage Strategy

Keep current hot/cold contract:

- Hot control artifacts on C:.
- Cold full traces on D: when available.
- Archives compressed after analysis.

GPU-specific storage:

- Do not store giant GPU tensors by default.
- Optionally store compact `.pt` debug snapshots for tiny runs only:
  - start states;
  - final states;
  - score tensors;
  - parity sample.

Do not put `.pt` dumps in normal large runs unless explicitly requested.

## Performance Pitfalls To Avoid

CPU/GPU sync traps:

- `.item()` inside loop.
- `.cpu()` inside loop.
- printing tensor values inside loop.
- Python `if tensor.any()` without care can sync. For MVP it may be acceptable once per step, but measure it.
- CPU-side ranking every step.
- JSON writing every step.

PyTorch compile traps:

- `torch.compile` can help after shapes are stable.
- It can also hide graph breaks or compile overhead.
- Do not start with `torch.compile`.
- First make eager PyTorch correct.
- Then benchmark compiled rollout kernels.

Small batch trap:

- GPU may be slower than CPU for tiny `8x2` tests.
- That is fine.
- Use tiny tests for correctness, not speed.

Telemetry trap:

- If CPU replay-all is enabled, the run may still be slow.
- Measure GPU rollout separately from telemetry.

Geometry trap:

- Physics is easy.
- Geometry is where correctness can drift.
- Treat projection/checkpoints/collision as the real migration risk.

## External Tools And Dependencies

Use now:

- PyTorch tensors.
- CUDA through existing `torch` dependency.
- Existing `uv`/optional `train` dependency group.

Use later if needed:

- `torch.func.vmap` for vectorizing functional per-candidate code if direct tensor batching becomes awkward.
- `torch.compile` after eager parity for stable-shape rollout functions.
- TorchRL only if the project moves toward GPU-native PPO/rollout collection.
- NVIDIA Warp only if PyTorch geometry/collision becomes the wall.

Do not add now:

- NVIDIA Warp.
- CuPy.
- custom CUDA/C++ extension.
- TorchRL as a required dependency.
- JAX.

Why PyTorch first:

- Already in the project.
- Already installed with CUDA.
- Works on the current machine.
- Lets the same code support CPU-device tests and CUDA-device performance.
- Keeps dependency risk low.

External references checked:

- PyTorch `torch.func.vmap`: https://docs.pytorch.org/docs/stable/generated/torch.func.vmap.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/2.9/generated/torch.compile.html
- PyTorch compiler guide: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html
- TorchRL vectorized environments: https://docs.pytorch.org/rl/stable/reference/envs_vectorized.html
- NVIDIA Warp: https://nvidia.github.io/warp/

## Acceptance Criteria For MVP

MVP means:

- CPU `MonzaSim` unchanged.
- `--backend cpu` unchanged.
- `--backend gpu` exists.
- GPU backend can evaluate controller genomes for an evolution smoke run.
- GPU physics matches CPU physics within tolerance.
- GPU controller forward matches CPU controller forward within tolerance.
- GPU search features match CPU `search_features()` on representative states.
- GPU compact results are schema-compatible.
- Top GPU candidates replay through CPU.
- Selected telemetry is produced through CPU replay.
- Full tests pass.

MVP does not need:

- GPU PPO.
- GPU raycasts.
- perfect full boundary collision parity.
- all-candidate GPU step telemetry.
- `torch.compile`.
- Warp.

## Acceptance Criteria For Serious Search

Serious GPU backend means:

- `100x10` GPU run produces reasonable behavior and CPU-verified top candidates.
- CPU/GPU top-ranked candidates overlap enough to trust selection.
- Checkpoint/lap validity semantics match CPU for promoted candidates.
- No false valid-lap claims after CPU replay.
- Performance beats CPU backend clearly after excluding telemetry.
- Artifacts remain compatible with analysis/replay tools.
- The backend can be used in the normal workflow before a `150x60` or larger run.

## Recommended First Implementation Prompt

Use this for the next agent:

```text
Implement Phase 1 of gpu-focus.md only. Keep MonzaSim untouched. Add PyTorch batched physics helpers that mirror f1rl.physics.apply_physics, plus CPU-vs-GPU parity tests. Do not integrate with evolution_search yet. Run ruff, pyright, pytest, and f1-hardware-check. The output should be a tiny, reviewed foundation for the GPU backend, not a broad rewrite.
```

After Phase 1 passes:

```text
Implement Phase 2 and Phase 3 of gpu-focus.md: batched controller forward and search feature extraction. Keep CPU MonzaSim as reference. Add parity tests against _controller_controls and MonzaSim.search_features for representative states across Monza. Do not wire the evolution CLI until feature parity is proven.
```

After Phase 2/3 pass:

```text
Implement the first GPU evolution backend behind --backend gpu, with CPU replay verification for selected top candidates. Keep --backend cpu unchanged. Preserve artifact schemas. Start with selected telemetry only, not all-candidate GPU telemetry. Validate with tiny GPU and CPU comparison runs.
```

## Bottom Line

This is the right move, but only if it is done as a second backend.

The CPU simulator is the truth.

The GPU backend is the brute-force engine.

Evolution gets GPU batching first because it naturally evaluates many independent controllers and can keep scoring on-device. PPO remains on the current SB3/Gymnasium path until the GPU backend is proven and the project is ready for a custom GPU-native rollout system.

The first win is not a shiny CUDA rewrite. The first win is a trustworthy `GpuMonzaBatch` that can evaluate whole generations much faster while still replaying selected winners through the existing CPU simulator.
