# GPU Speed Architecture Plan

Date: 2026-06-04

## Purpose

This file is the plan for turning the current GPU backend from a correctness/parity-first CUDA tensor evaluator into a real high-throughput GPU simulation engine.

The immediate problem is not that the simulator is on the wrong device. The immediate problem is that the current GPU evolution path still behaves like a Python-driven simulator loop:

- sim state is represented as CUDA tensors;
- each step still goes through Python control flow;
- each step launches many small PyTorch CUDA ops;
- geometry helpers allocate large temporary tensors;
- the rollout checks GPU state from the CPU during the loop;
- the GPU is under-occupied because each step has too much launch overhead and too little fused work.

That architecture can be correct and useful for parity testing, but it will not produce Yosh-scale brute force. The target architecture must move from "PyTorch tensors on GPU" to "GPU-native rollout execution with static memory, fused kernels, and CPU synchronization only at generation boundaries or coarse chunks."

The non-negotiable correctness rule remains:

> CPU `MonzaSim` stays the oracle. The fast GPU backend may become the search engine, but it does not get to redefine physics, track semantics, reward/scoring semantics, lap validity, checkpoint validity, or termination behavior.

## Bottom Line

The best architecture is a layered backend stack:

```text
CPU MonzaSim oracle
  -> current CPU evolution backend
  -> current GPU eager parity backend
  -> optimized PyTorch chunk/CUDA-Graph backend
  -> Warp/custom-kernel fused rollout backend
  -> optional custom CUDA extension only if Warp/PyTorch are not enough
```

The current `GpuMonzaBatch` should be treated as the `gpu-eager-reference` backend: useful for parity, debugging, and schema compatibility, but not the final speed backend.

The speed backend should be built around:

- fixed-shape generation batches;
- persistent device-side state buffers;
- no per-step host sync;
- no per-step Python branching on GPU values;
- no full centerline projection against every segment at every step;
- spatial acceleration for centerline projection and collision;
- fused control + physics + geometry + scoring updates;
- CPU replay verification only for selected top candidates after GPU ranking;
- profiler-driven acceptance gates.

## Research Takeaways

Sources consulted:

- PyTorch CUDA semantics and CUDA Graphs: https://docs.pytorch.org/docs/main/notes/cuda.html
- NVIDIA CUDA Graph best practices for PyTorch: https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/2.9/generated/torch.compile.html
- PyTorch profiler recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NVIDIA Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA Warp basics: https://nvidia.github.io/warp/basics.html
- NVIDIA Warp interoperability: https://nvidia.github.io/warp/modules/interoperability.html
- Triton documentation: https://triton-lang.org/main/index.html
- OpenAI Triton overview: https://openai.com/index/triton/

Key points from those sources, applied to this project:

1. CUDA Graphs are designed for repeated fixed GPU workloads. They reduce CPU/kernel-launch overhead by replaying a captured graph instead of dispatching every operation through Python/C++/driver setup each time.
2. CUDA Graphs require static shapes, stable memory addresses, and graph-safe control flow. CPU work inside a captured section is not replayed correctly, and dynamic shapes/control flow break capture.
3. `torch.compile(mode="reduce-overhead")` can use CUDA graphs to reduce Python overhead for CUDA-only graphs, but it is not a magic fix when the function mutates inputs, changes shapes, hits graph breaks, or relies on CPU syncs.
4. NVIDIA specifically calls out simulation timesteps as a workload where CUDA Graphs can help because the same operation pattern repeats many times.
5. Warp is built for Python-authored GPU simulation kernels. Warp kernels are Python-declared but compile to native C++/CUDA. They can interoperate with PyTorch tensors without copying via `warp.from_torch()` / `warp.to_torch()`.
6. Triton is excellent for custom high-throughput tensor kernels, but its main design center is dense/tiled compute. It can help for specific kernels, but track simulation with branching geometry is usually a better first fit for Warp or custom CUDA.
7. Profiling is required before claiming speed. `nvidia-smi` utilization alone is not enough. Use PyTorch profiler and Nsight Systems to see launch counts, CPU gaps between kernels, memory allocation, synchronization, and GPU occupancy.

## Current Observed Failure Mode

The aborted GPU `150x60`, `18k max_steps`, normal-start run showed:

- generation 0 throughput: `0.417 candidates/sec`;
- later generation throughput varied around `1.0-3.9 candidates/sec`;
- latest completed generation before abort: generation `16`;
- best distance at abort: `2585.444m`;
- average distance at abort: `1252.964m`;
- average pace at abort: `212.601 kph`;
- top-decile pace at abort: `272.489 kph`;
- valid laps at abort: `0`;
- GPU memory allocated was tiny compared to total GPU memory;
- GPU utilization was modest.

That means the workload was not saturating the GPU. It was mostly launch/control/geometry overhead, not raw GPU compute capacity.

The current code shape confirms this:

- `GpuMonzaBatch.rollout()` loops in Python over `range(self.sim_config.max_steps)`.
- The loop calls `_program_controls()`, `_step()`, and `update_score_accumulator()` each step.
- `_step()` calls PyTorch helper functions that each launch many small CUDA kernels.
- Active-state checking calls `active.any().detach().cpu().item()` every `active_check_interval`.
- `track_errors_batch()` performs projection by constructing `[N, centerline_segments, 2]` temporary tensors, then masking to a local window after most of the full projection work has already happened.
- Collision checks iterate over boundary segment chunks in Python.
- Final `sim_steps_tensor.detach().cpu().item()` is acceptable at the end, but not inside the hot loop.

The current backend is therefore a correct direction but not a final fast architecture.

## Definitions

`CPU oracle`

The existing `MonzaSim`, CPU geometry, CPU physics, CPU replay, and CPU scoring/replay verification path. This remains the source of truth.

`GPU eager reference`

The current PyTorch tensor backend. It is useful because it mirrors CPU semantics and is easier to test than custom kernels. It should remain available as a stepping stone and regression reference.

`GPU graph backend`

A PyTorch backend that keeps fixed static tensors and executes chunked rollout work using `torch.compile` and/or `torch.cuda.CUDAGraph`. This reduces Python/kernel-launch overhead while staying close to the current tensor code.

`GPU fused backend`

A Warp or custom CUDA backend that implements simulation rollout in a small number of fused kernels. This is the real speed target.

`CPU replay verification`

After GPU ranking, replay selected top genomes through CPU `MonzaSim` and compare final state, termination reason, lap validity, score, and replay behavior before promoting the candidate.

## Non-Negotiables

1. Do not change CPU `MonzaSim` to make GPU parity easier.
2. Do not weaken lap validity, checkpoint validity, collision/off-track, no-progress, finish-line, braking-gate, or target-gate semantics.
3. Do not remove CPU replay verification for serious GPU search runs.
4. Do not count a GPU-only candidate as valid unless selected top candidates replay through CPU.
5. Do not make PPO, replay, manual, or scripted workflows depend on the fast GPU backend.
6. Do not hide approximations. If an approximation is used for candidate pre-filtering, label it as approximate and re-score selected candidates with the exact CPU oracle.
7. Do not drop telemetry/debug value silently. Fast search may avoid all-candidate full telemetry in the hot path, but top candidates and selected diagnostics must remain replayable.
8. Do not trust speed without profiler evidence.
9. Do not trust correctness without parity tests and CPU replay proof.

## Architecture Overview

```text
Evolution CLI
  |
  |-- backend=cpu
  |     -> CPU MonzaSim evaluator
  |
  |-- backend=gpu-eager-reference
  |     -> current GpuMonzaBatch PyTorch eager evaluator
  |     -> parity/debug backend
  |
  |-- backend=gpu-graph
  |     -> static PyTorch buffers
  |     -> compiled/captured rollout chunks
  |     -> no per-step host sync
  |     -> still uses PyTorch ops
  |
  |-- backend=gpu-fused
        -> Warp/custom kernel rollout
        -> spatially accelerated track geometry
        -> device-side scoring
        -> compact result rows
        -> CPU replay top K
```

The active implementation should not delete the current GPU path. Rename or document it conceptually as the eager reference backend, then add the speed backend beside it.

Suggested CLI evolution:

```powershell
uv run --no-sync f1-evolution-search --backend cpu ...
uv run --no-sync f1-evolution-search --backend gpu --gpu-engine eager ...
uv run --no-sync f1-evolution-search --backend gpu --gpu-engine graph ...
uv run --no-sync f1-evolution-search --backend gpu --gpu-engine fused ...
```

Where:

- `gpu-engine=eager` means current parity backend.
- `gpu-engine=graph` means PyTorch chunk/CUDA Graph backend.
- `gpu-engine=fused` means Warp/custom-kernel backend.

If the CLI should stay smaller, `--backend gpu` can keep eager as default while `--gpu-engine graph|fused` selects faster modes.

## Why Current GPU Is Slow

### 1. Python controls the timestep loop

`for step_index in range(max_steps)` runs on the host. Even if tensors are on CUDA, the host is deciding every step.

At `18,000` steps and `60` generations, that is too much Python orchestration.

### 2. Many small kernels are launched per step

Each PyTorch operation on CUDA can launch one or more kernels. A single simulator step can trigger many kernels for:

- controller feature extraction;
- controller forward pass;
- physics integration;
- centerline projection;
- progress update;
- collision checks;
- drivable mask lookup;
- checkpoint update;
- finish-line checks;
- score accumulator updates;
- reason/termination updates.

Even if each kernel is fast, launch overhead and gaps between kernels dominate.

### 3. There is CPU synchronization inside the hot loop

This line pattern forces the CPU to wait for GPU results:

```python
bool(active.any().detach().cpu().item())
```

Even every 16 steps, this interrupts asynchronous CUDA execution and prevents clean graph capture.

### 4. Projection does too much work

The current batched projection computes across all centerline segments, then uses the local window to mask. That means the expensive `[N, S, 2]` tensor is still built.

The correct speed architecture must use a local projection candidate set before doing expensive math.

### 5. Boundary collision checks are broad

Chunked all-segment collision is still far more work than necessary. A movement segment only needs to be tested against nearby boundary segments.

### 6. The batch size is too small for the amount of per-step overhead

Population `150` is not enough to amortize a Python-launched sequence of many small kernels across an RTX 4060. The workload needs either much larger batches or fewer/fused launches.

### 7. The state representation is too object/dict heavy for the hot path

Dataclasses and dicts are fine at boundaries. The hot rollout path should be plain static buffers, not Python objects that get reassembled every step.

## Performance North Star

The speed backend should optimize these metrics:

- candidates/sec;
- simulator steps/sec;
- active simulator steps/sec;
- GPU occupancy/utilization during rollout;
- kernel launches per generation;
- host-to-device and device-to-host transfers per generation;
- peak allocated/reserved memory;
- generation wall time;
- replay verification wall time;
- selected telemetry packaging time.

Target direction:

- Current GPU eager: around `0.4-4 candidates/sec` in the aborted long run.
- Initial graph/chunk backend should beat CPU and eager GPU on the same constraints.
- Fused backend should be an order-of-magnitude improvement over eager GPU for large enough batches.

Do not promise a fixed number until profiling. The acceptance gate should be relative:

- `gpu-graph` must be materially faster than `gpu-eager-reference` on the same seed/config.
- `gpu-fused` must be materially faster than `gpu-graph`.
- Serious GPU search must show high GPU utilization during rollout and low CPU idle/dispatch overhead in profiler traces.

## Correctness North Star

Fast is useless if it changes the learning problem.

Correctness acceptance must include:

- CPU vs GPU final state parity;
- CPU vs GPU termination reason parity;
- CPU vs GPU checkpoint/lap validity parity;
- CPU vs GPU scoring profile parity;
- CPU vs GPU candidate ranking sanity;
- CPU replay verification of top GPU candidates;
- long-horizon stability;
- randomized and adversarial edge cases.

The fast backend can have tiny floating-point differences, but it cannot produce a materially different driving world.

Recommended tolerances:

- position error after short rollouts: <= `0.05m`;
- position error after long rollouts: <= `0.5m`, unless CPU/GPU trajectories diverge after a near-boundary event that is explicitly explained and covered by replay verification;
- speed error: <= `0.5 kph`;
- heading error: <= `0.5 deg`;
- progress error: <= `0.5m`;
- final score difference: profile-specific, but rank correlation must remain stable for top candidates;
- termination reason: exact for controlled tests, and explained/replayed for random near-boundary cases;
- lap validity/checkpoint validity: exact.

## Recommended Implementation Path

## Phase 0: Instrument The Current Backend

Goal: prove where time goes before changing architecture.

Add a repeatable profiler command:

```powershell
uv run --no-sync python -m f1rl.evolution_search `
  --backend gpu `
  --gpu-engine eager `
  --output-dir artifacts\gpu-prof-eager `
  --start-progress-m 0 `
  --start-speed-kph 80 `
  --target-progress-m 5793 `
  --no-target-termination `
  --action-set racing `
  --observation-profile racing_v2 `
  --max-steps 18000 `
  --population 150 `
  --generations 2 `
  --elite-count 24 `
  --random-immigrants 16 `
  --top-k 8 `
  --workers 1 `
  --genome-type controller `
  --scoring-profiles max_progress,fast_frontier,frontier_fast,early_pace,farthest_distance,clean_distance,risk_seeking `
  --telemetry-selection top `
  --telemetry-compression gzip `
  --progress-every-generation
```

Add optional flags:

- `--gpu-profile torch`
- `--gpu-profile-nsight-hints`
- `--gpu-profile-output PATH`

Profiler outputs should answer:

- how many CUDA kernels launch per simulator step;
- how many CUDA kernels launch per generation;
- where host synchronization occurs;
- how much time is spent in projection;
- how much time is spent in collision;
- how much time is spent in scoring;
- whether memory allocation occurs during rollout;
- whether GPU is idle between kernels;
- whether launch gaps dominate execution.

Implementation notes:

- Use `torch.profiler` for Python/PyTorch-level events.
- Use Nsight Systems for timeline-level CPU/GPU launch gaps.
- Keep profiling runs small: `population=150`, `generations=1-2`, same `max_steps=18000`.
- Add NVTX ranges around:
  - controller feature extraction;
  - controller forward;
  - physics;
  - projection;
  - collision;
  - scoring;
  - termination;
  - final row creation.

Acceptance:

- A profiler artifact exists.
- The report identifies top 3 time sinks.
- Any later speed claim is compared against this baseline.

## Phase 1: Make The Eager Backend A Clean Reference

Goal: keep the current backend valuable as a correctness reference and remove obvious pathological overhead without risky semantic changes.

Changes:

1. Rename/document current backend internally as `gpu-eager-reference`.
2. Remove all per-step CPU synchronization from the normal rollout path.
3. Replace step-level `active.any().cpu().item()` with one of:
   - no early stop in speed mode;
   - chunk-level active check every `256` or `512` steps;
   - device-side active count stored and read only at chunk boundaries.
4. Preallocate all temporary tensors used in rollout where possible.
5. Avoid per-step construction of Python dicts in the hot path.
6. Avoid `.clone()` unless state preservation is necessary.
7. Cache stable tensors:
   - `torch.arange(N)`;
   - profile ids;
   - reason ids;
   - checkpoint thresholds;
   - braking gates;
   - lookahead distances;
   - action ids.
8. Keep `torch.inference_mode()`.
9. Keep final host transfers only after generation results are ready.

Expected result:

- modest improvement;
- better graph-capture readiness;
- easier profiling;
- no semantic drift.

Acceptance:

- Existing GPU parity tests still pass.
- Full randomized parity battery still passes.
- `gpu-eager-reference` is no slower than current eager.
- No CPU sync appears inside the per-step hot path except explicitly allowed chunk boundaries.

## Phase 2: Build A PyTorch Chunked Graph Backend

Goal: reduce Python/kernel launch overhead without jumping immediately to custom kernels.

Design:

```text
GpuGraphRolloutEngine
  static state tensors [N]
  static genome tensors [N, W]
  static track tensors
  static accumulator tensors [N]
  static output tensors [N]
  rollout_chunk(K steps)
    repeated fixed operation pattern
    no CPU sync inside chunk
    alive mask stays on device
  host loop over chunks
    replay graph
    optional active check every chunk
```

Use chunk sizes:

- start with `K=64` for correctness/debug;
- then test `K=128`, `256`, `512`, `1024`;
- do not use a full `18,000`-step graph initially because graph capture/instantiation can become huge and brittle.

Core idea:

Instead of:

```python
for step in range(18000):
    launch many kernels
    maybe sync
```

Use:

```python
for chunk in range(ceil(18000 / K)):
    graph.replay()  # K steps worth of static GPU work
    optionally sync once to check active count
```

Implementation requirements:

- static batch size;
- static tensor shapes;
- stable tensor memory addresses;
- no allocation during captured chunk;
- no CPU work inside captured chunk;
- no Python branching based on tensor values inside captured chunk;
- dynamic behavior represented through tensor masks;
- dead cars remain in the batch but their state stops updating;
- generation-level and chunk-level code can be Python;
- per-step code cannot require Python decisions.

`torch.compile` plan:

- Compile the chunk function, not only tiny helper functions.
- Try `torch.compile(..., fullgraph=True, dynamic=False, mode="reduce-overhead")`.
- Use `TORCH_LOGS=graph_breaks,guards,perf_hints` during development.
- If fullgraph fails, use graph-break reports to remove CPU logic and allocations.
- If `reduce-overhead` is unstable or not useful, fall back to manual `torch.cuda.CUDAGraph` capture.

Manual CUDA Graph plan:

- Allocate long-lived static input/output tensors.
- Warm up on a side stream.
- Capture one chunk.
- Copy new generation inputs into static buffers.
- Replay chunk repeatedly.
- Read compact result tensors at generation end.

Important limitation:

CUDA Graph replay uses the same memory addresses and fixed op graph. That is good for fixed population search. It is not good for dynamic shape changes inside the hot path.

Acceptance:

- Same results as eager GPU within parity tolerance.
- Materially fewer CPU launch gaps in profiler.
- Throughput beats eager GPU on:
  - `150x2`, `18k`;
  - `512x2`, `18k`;
  - `1024x2`, `18k`.
- No top-candidate CPU replay regression.

## Phase 3: Fix Geometry Before Expecting Big Speedups

The biggest algorithmic problem is geometry.

Physics is cheap. Controller math is cheap. Scoring is mostly cheap. Projection and collision can dominate.

### 3.1 Centerline Projection

Current approach:

```text
for every car:
  compare point to every centerline segment
  then mask to local window
```

That is backwards for speed.

Correct approach:

```text
for every car:
  maintain previous segment/progress index
  choose small candidate segment window first
  project only against those local segments
  update segment/progress index
```

Implementation options:

Option A: progress-indexed local window

- Maintain `centerline_segment_idx` per car.
- Precompute fixed local windows of segment indices:
  - `local_projection_indices[segment_idx, window_size]`.
- For each car, gather only `window_size` segments.
- Run projection against that local window.
- If projection fails or progress jump is suspicious, fall back to a wider window on GPU.
- If still suspicious, mark for CPU replay/verification.

Option B: uniform spatial grid

- Divide the track map into cells.
- Precompute `cell -> centerline segment ids`.
- For each car position, query the nearby cells.
- Project against only those segments.
- More robust when cars are off-line or spun around.

Recommended:

- Implement Option A first because the simulator already uses monotonic progress and a local projection window.
- Add Option B later for recovery/off-track edge cases.

Correctness requirements:

- Projection must match CPU local projection behavior.
- Tie-breaking must be deterministic.
- Wrap-around at finish line must be tested.
- Boundary cases at braking gates and finish line must be tested.

### 3.2 Collision And Off-Track

Current approach:

- point drivable mask lookup;
- movement segment against boundary segments in chunks;
- finish line segment intersection.

Better approach:

1. Keep drivable mask lookup as the first cheap filter.
2. Add a boundary segment uniform grid:
   - precompute `cell -> boundary segment ids`;
   - for each movement segment, compute touched cells;
   - exact segment intersection only against those segment ids.
3. Keep finish-line exact intersection separate because it is tiny.
4. Optionally add a signed distance field or distance transform as a fast broadphase, but do not use it as the final correctness check unless parity proves it.

Do not replace exact collision semantics with an approximate mask-only collision for serious runs.

Mask-only can be a diagnostic speed mode, not the default correctness mode.

### 3.3 Checkpoints

Checkpoint update should be simple device-side integer arithmetic:

- fixed spacing;
- current checkpoint index per car;
- monotonic progress per car;
- missed checkpoint count per car;
- valid lap bool per car.

Avoid Python loops or per-checkpoint scans.

### 3.4 Braking Gates And Lookahead

Precompute on GPU:

- braking gate distances;
- target speed table along progress;
- curvature table along progress;
- lookahead heading table or sampled centerline headings.

Prefer progress-indexed table lookups over repeated geometric computations.

For example:

```text
progress_bin = floor(progress_m / bin_width_m)
target_speed = target_speed_table[progress_bin]
future_target_speed = min(target_speed_table[progress_bin : progress_bin + horizon_bins])
curvature = curvature_table[progress_bin]
lookahead_heading_error = heading_table[progress_bin + offsets] - heading
```

This is much cheaper than repeatedly projecting and sampling arbitrary geometry.

Correctness note:

Tables are acceptable only if their resolution is high enough and parity tests prove they match CPU features within tolerance. If table lookup changes behavior around braking gates, add tolerance logic like the braking-gate epsilon fix.

## Phase 4: Build The Fused GPU Backend

This is the real speed target.

The best practical implementation path for this repo is NVIDIA Warp first, not raw CUDA first.

Why Warp first:

- It is designed for simulation/robotics-style GPU workloads.
- Kernels are written in Python syntax but compile to native C++/CUDA.
- It supports CUDA devices.
- It interoperates with PyTorch tensors without copying.
- It avoids a large Visual Studio/CUDA C++ build-system detour on Windows.
- It is easier to iterate on than a custom CUDA extension.

Why not Triton first:

- Triton is excellent for dense/tiled tensor kernels.
- This simulation has branch-heavy geometry, state machines, masks, lap validity, and collision checks.
- Triton may still be useful for specific kernels, but Warp is a better first custom-kernel target.

Why not raw CUDA first:

- It can be fastest.
- It has the highest build/debug complexity.
- Windows CUDA extension setup can become a project of its own.
- Use it only if Warp cannot meet speed targets.

### Fused Backend Layout

New modules:

- `src/f1rl/gpu_fast_types.py`
- `src/f1rl/gpu_fast_track.py`
- `src/f1rl/gpu_fast_buffers.py`
- `src/f1rl/gpu_fast_warp.py`
- `src/f1rl/gpu_fast_backend.py`
- `tests/test_gpu_fast_parity.py`
- `tests/test_gpu_fast_performance.py`

Keep existing modules:

- `gpu_batch.py` remains eager reference.
- `gpu_track.py` remains eager reference/helper.
- `gpu_scoring.py` remains reference for vectorized score logic.
- `evolution_backend.py` selects engine.

### Data Layout

Use structure-of-arrays, not array-of-structs.

State buffers:

```text
x_px[N]
y_px[N]
heading_rad[N]
speed_mps[N]
yaw_rate_rps[N]
steering[N]
raw_progress_m[N]
monotonic_progress_m[N]
last_raw_progress_px[N]
lap_index[N]
checkpoint_index[N]
checkpoints_passed[N]
missed_checkpoint_count[N]
no_progress_steps[N]
elapsed_steps[N]
alive[N]
terminated[N]
truncated[N]
completed_lap[N]
valid_lap[N]
finish_crossed[N]
segment_complete[N]
termination_reason_id[N]
last_throttle[N]
last_brake[N]
last_steer[N]
```

Genome/control buffers:

```text
genome_kind[N]
controller_weights[N, W]
phase_schedule_offsets[N]
phase_schedule_lengths[N]
progress_phase_offsets[N]
progress_phase_lengths[N]
```

Scoring buffers:

```text
best_progress_m[N]
final_progress_m[N]
best_lateral_error_m[N]
best_heading_error_deg[N]
score_profile_values[N, P]
pace_sums[N]
speed_sums[N]
brake_sums[N]
gate_times[N, G]
valid_lap_time_s[N]
valid_lap_count[N]
```

Track buffers:

```text
centerline_x[S]
centerline_y[S]
centerline_dx[S]
centerline_dy[S]
centerline_len[S]
centerline_len2[S]
centerline_cumdist_px[S]
centerline_tangent_rad[S]
local_projection_indices[S, W]
boundary_x1[B]
boundary_y1[B]
boundary_x2[B]
boundary_y2[B]
boundary_cell_offsets[C + 1]
boundary_cell_indices[K]
drivable_mask[H, W]
target_speed_table[T]
curvature_table[T]
braking_gate_m[G]
checkpoint_threshold_m[Ck]
finish_line[1 or 2 segments]
```

Everything needed for rollout should already be on device before rollout starts.

### Kernel Strategy

There are two realistic fused-kernel designs.

### Strategy A: Multi-Kernel Per Step

Launch a small fixed sequence per step:

1. `control_kernel`
2. `physics_kernel`
3. `projection_collision_kernel`
4. `termination_scoring_kernel`

Pros:

- easier to debug;
- easier to compare to existing code;
- easier to incrementally implement;
- easier to profile.

Cons:

- still has per-step kernel launches;
- needs CUDA Graph capture or it may remain launch-bound;
- less ideal for `18k` steps.

Use Strategy A for the first fused backend milestone.

### Strategy B: Persistent Rollout Kernel

Launch one kernel or a very small number of kernels for the whole rollout chunk/full rollout.

Each candidate or candidate block loops over timesteps device-side:

```text
kernel rollout_kernel(candidate_id):
  load state
  for step in 0..max_steps:
    if not alive:
      break
    compute controller controls
    integrate physics
    project to track
    update progress/checkpoints/lap
    update collision/off-track/no-progress
    update scoring accumulators
  write compact final result
```

Pros:

- eliminates almost all per-step launch overhead;
- host only launches once per generation or chunk;
- naturally matches evolution search;
- best candidate for massive speedup.

Cons:

- more complex;
- branch divergence between candidates;
- one thread per car may underuse GPU if population is small;
- geometry projection/collision may need block-level parallelism;
- harder to debug than multi-kernel step mode.

Recommended persistent layout:

- one CUDA/Warp block per candidate;
- threads within block cooperate on geometry projection and collision;
- candidate state lives in registers/shared memory where possible;
- reductions select nearest centerline segment and collision hits;
- final compact result writes to global memory.

For population `150`, one block per candidate may still be too few blocks. Solve this by evaluating more candidates per generation on GPU:

- `population=512`;
- `population=1024`;
- `population=2048`;
- or `population x variants_per_parent`.

The GPU speed backend should not be judged only on `150` candidates. A GPU wants bigger batches.

### Recommended Fused Path

1. Build Strategy A with Warp kernels and CUDA Graph capture.
2. Pass parity.
3. Profile.
4. If launch overhead still dominates, build Strategy B persistent rollout kernel.
5. Use Strategy B for large brute-force runs.
6. Keep Strategy A as the debug kernel path.

## Phase 5: Full Speed Search Data Flow

Fast generation evaluation should look like this:

```text
CPU:
  build/mutate population genomes
  copy compact genomes/start states to static GPU buffers

GPU:
  reset state buffers
  rollout all candidates
  update score accumulators
  compute profile scores
  produce compact result rows
  top-k/select candidate ids per profile

CPU:
  copy compact result rows
  write attempts/generation summaries
  reproduce next generation
  CPU replay top selected candidates
  write selected telemetry
```

Do not write per-step JSONL from GPU during the hot search rollout.

For replay/debug:

- always CPU-replay top K;
- optionally CPU-replay profile leaders;
- optionally CPU-replay novelty/frontier samples;
- optionally store GPU sparse trace samples for all candidates, but not full JSONL per step in fast mode.

If all-candidate full telemetry is requested:

- treat it as debug mode;
- accept that it is slower;
- stream compressed output;
- do not compare its runtime to fast search runtime.

## Phase 6: PPO Implications

Do not start by forcing SB3 PPO to use the fast GPU simulator. SB3/Gym/VecEnv can pull observations/actions through CPU/NumPy every step, which destroys the benefit.

Correct PPO path:

1. Keep CPU PPO as existing baseline.
2. Use GPU ES to discover better elite states/laps.
3. Save elite replay/state libraries.
4. For real GPU PPO later, build a custom rollout collector:
   - actor network on CUDA;
   - batched sim state on CUDA;
   - observations on CUDA;
   - actions on CUDA;
   - rewards/dones on CUDA;
   - rollout buffers on CUDA;
   - only summaries/checkpoints copied to CPU.
5. Do not call this "real GPU PPO" if the environment step synchronizes through CPU each timestep.

The fast simulation backend should be designed so it can later serve:

- evolution controller rollouts;
- actor-network rollouts;
- off-policy dataset generation;
- SAC/TD3-style replay buffer generation.

But the first performance win should remain GPU ES.

## Specific Code-Level Changes To Plan

### CLI / Config

Add:

```text
--gpu-engine eager|graph|fused
--gpu-profile none|torch|nsight
--gpu-chunk-steps 256
--gpu-static-batch-size auto
--gpu-disable-early-stop
--gpu-fast-geometry local_window|grid
--gpu-collision-mode exact_grid|exact_all_segments|mask_only_debug
--gpu-cpu-replay-top-k 24
```

Default:

- `--gpu-engine eager` until graph/fused parity is proven.
- Once proven, `--gpu-engine graph` can become default.
- `--gpu-engine fused` becomes default only after it passes the full battery and replay verification.

### Backend Selection

Current:

```text
backend=cpu|gpu
```

Recommended:

```text
backend=cpu
backend=gpu, gpu_engine=eager
backend=gpu, gpu_engine=graph
backend=gpu, gpu_engine=fused
```

### Output Schema

Keep existing output keys:

- `attempts.jsonl`
- `generation_summary.jsonl`
- `best_so_far.json`
- `evolution_summary.json`
- `selected_telemetry/manifest.json`
- `elite_state_library.json`
- `ppo_bridge.json`
- `next_commands.md`

Add fields:

```json
{
  "backend": "gpu",
  "gpu_engine": "graph",
  "gpu_kernel_backend": "pytorch_cuda_graph",
  "gpu_rollout_seconds": 0.0,
  "gpu_candidates_per_second": 0.0,
  "gpu_active_steps_per_second": 0.0,
  "gpu_kernel_launches_per_generation": 0,
  "gpu_host_sync_count": 0,
  "gpu_cpu_replay_count": 0,
  "gpu_cpu_replay_seconds": 0.0,
  "gpu_parity_status": "not_checked|passed|failed",
  "gpu_cpu_top_replay_max_progress_delta_m": 0.0,
  "gpu_cpu_top_replay_reason_mismatches": 0
}
```

## Correctness Test Battery

### Unit Tests

Track:

- centerline projection short cases;
- wrapped progress at finish line;
- braking gate epsilon behavior;
- target speed table lookup;
- curvature table lookup;
- checkpoint threshold updates;
- drivable mask lookup;
- boundary grid query;
- exact segment intersection.

Physics:

- straight throttle;
- braking;
- steering at speed;
- steering at near-zero speed;
- throttle/brake conflict;
- launch guard;
- yaw rate limits;
- steering smoothing.

Controller:

- controller feature vector parity;
- controller output bounds;
- phase genome parity;
- progress-phase genome parity;
- controller genome parity.

Scoring:

- all scoring profiles parity;
- valid lap bonus;
- pace pressure;
- slow valid finisher penalty;
- frontier profiles;
- risk profiles;
- late braking/setup features.

### Randomized Parity Battery

Required:

- `1000` random start states;
- multiple fixed action tapes;
- controller genomes;
- phase genomes;
- progress-phase genomes;
- short rollouts: `16-128` steps;
- medium rollouts: `512-2048` steps;
- long rollouts: `5000-18000` steps;
- randomized seeds;
- randomized speed/progress/lateral/heading/yaw states;
- collision/off-track edge cases;
- checkpoint edge cases;
- braking gate boundary cases;
- finish-line/lap-complete edge cases;
- no-progress edge cases.

Compare:

- final x/y;
- final heading;
- final speed;
- final yaw rate;
- final steering;
- monotonic progress;
- raw progress;
- checkpoint index;
- missed checkpoint count;
- valid lap;
- completed lap;
- finish crossed;
- termination/truncation;
- termination reason;
- score per profile;
- rank order for top candidates.

### CPU Replay Verification

Every serious GPU run must:

- CPU replay top K overall;
- CPU replay each profile leader;
- CPU replay fastest valid candidate if any;
- CPU replay farthest candidate;
- CPU replay cleanest late-frontier candidate;
- write replay deltas to summary;
- refuse promotion if deltas exceed tolerance or reason/lap validity mismatch.

### Regression Tests

Keep tests for:

- `backend=cpu`;
- `backend=gpu --gpu-engine eager`;
- `backend=gpu --gpu-engine graph`;
- `backend=gpu --gpu-engine fused`;
- `gpu_engine` fallback behavior;
- output schema compatibility;
- replay command compatibility.

## Performance Test Battery

Use identical seeds/configs.

Smoke:

```text
population=32
generations=2
max_steps=512
```

Small:

```text
population=150
generations=2
max_steps=18000
```

Medium:

```text
population=512
generations=2
max_steps=18000
```

Large speed:

```text
population=2048
generations=2
max_steps=18000
```

Full comparison:

```text
population=150
generations=60
max_steps=18000
no target termination
same scoring profiles as CPU 150x60
```

Metrics:

- wall time;
- candidates/sec;
- active simulator steps/sec;
- GPU utilization;
- CPU utilization;
- kernel launch count;
- host sync count;
- peak memory;
- valid lap count;
- fastest valid lap;
- average distance;
- top-decile pace.

Acceptance:

- The fast backend must not be judged only by candidate outcome quality. Outcome quality can vary because evolution is stochastic.
- The speed comparison must use identical initial population/seeds where possible.
- Correctness must be proven separately from speed.
- A speed backend that is fast but fails CPU replay verification is rejected.

## Why Population Should Increase On GPU

A GPU can be slower than CPU at small batch sizes because:

- the CPU has low launch overhead for scalar logic;
- the GPU pays launch overhead;
- RTX 4060 has many lanes that need enough parallel work;
- population `150` may not provide enough work per kernel;
- many candidates terminate early, reducing active work.

For real GPU ES, use larger populations:

- `512xN`;
- `1024xN`;
- `2048xN`;
- or generate many variants per parent per generation.

This is not only for speed. It also helps learning:

- more frontier variants;
- more fast valid laps;
- more late-lap samples;
- more useful selection pressure;
- less over-reliance on elite cloning.

Recommended GPU search shape after fast backend is proven:

```text
population=1024
generations=30-60
max_steps=18000
telemetry-selection=top
cpu-replay-top-k=32-64
```

Then scale based on storage/runtime:

```text
population=2048-4096
generations=60+
```

## Storage And Telemetry Plan

Fast mode:

- write `attempts.jsonl`;
- write `generation_summary.jsonl`;
- write `best_so_far.json`;
- write compact final rows;
- CPU-replay selected top candidates;
- write selected telemetry compressed.

Debug mode:

- optionally write all-candidate telemetry;
- compressed JSONL only;
- expect much slower runtime;
- do not use it for speed benchmarking.

Cold storage:

- large old runs can be compressed and moved to external drive;
- active run summaries stay on C:;
- selected replay telemetry stays where replay can find it;
- manifests should record absolute paths if traces are split across drives.

## Recommended Dependency Decision

Do not add a heavy dependency for Phase 1 or Phase 2.

For Phase 4, add Warp if profiling shows PyTorch graph/chunk backend is still not enough.

Recommended optional dependency group:

```toml
[project.optional-dependencies]
gpu-fast = [
  "warp-lang>=1.10.0",
]
```

Before adding:

- verify `warp-lang` installs cleanly on Windows with current Python/PyTorch/CUDA setup;
- run a tiny Warp kernel smoke;
- verify `warp.from_torch()` with CUDA tensors;
- verify no copies in the hot path;
- verify imports do not break users without `gpu-fast`.

Triton:

- do not add as first fast backend;
- consider it only for isolated kernels if Warp is insufficient;
- check Windows compatibility and PyTorch/Triton packaging before depending on it.

Custom CUDA extension:

- reserve as final option;
- only after Warp proves insufficient;
- require build documentation for Windows;
- require CI/smoke strategy.

## The "Absolute Correction" Contract

The fast backend must implement the same semantic contract as CPU:

### Physics

Must match:

- throttle/brake acceleration;
- drag;
- steering smoothing;
- steering clamp;
- yaw rate;
- heading update;
- position update;
- speed clamp;
- launch guard.

### Geometry

Must match:

- centerline projection;
- signed lateral error;
- heading error;
- raw progress;
- monotonic progress;
- local projection window behavior;
- wrap-around behavior;
- drivable/off-track detection;
- boundary collision;
- finish-line crossing.

### Race Validity

Must match:

- checkpoint index;
- checkpoint pass thresholds;
- missed checkpoint count;
- valid lap;
- completed lap;
- finish crossed;
- lap index.

### Termination

Must match:

- collision;
- off-track;
- no-progress;
- max-steps;
- segment complete;
- segment gate failures;
- lap complete.

### Observations And Features

Must match:

- base/racing/racing_v2 features where GPU supports them;
- controller search features;
- target speed;
- near/future target speed;
- brake demand;
- future brake demand;
- target speed drop;
- braking gate distance/proximity;
- lookahead heading errors;
- curvature/yaw features.

### Scoring

Must match:

- `max_progress`;
- `fast_frontier`;
- `frontier_fast`;
- `early_pace`;
- `farthest_distance`;
- `clean_distance`;
- `risk_seeking`;
- `full_lap_validity`;
- any future speed/valid-lap profiles.

## Practical Implementation Order

### Step 1: Baseline Profiling

Run profiler on current GPU eager backend.

Deliverables:

- profiler report;
- top bottleneck list;
- baseline metrics table;
- launch/sync count estimate.

### Step 2: Eager Cleanup

Remove avoidable syncs/allocations.

Deliverables:

- no per-step CPU sync in normal mode;
- no avoidable per-step dict construction;
- cached stable tensors;
- parity tests pass.

### Step 3: Local Projection Acceleration

Replace full centerline segment projection with progress-indexed local projection.

Deliverables:

- `local_projection_indices`;
- segment index state;
- fallback wide window;
- projection parity tests;
- speed comparison.

### Step 4: Boundary Collision Grid

Replace all-boundary chunk checks with grid broadphase + exact narrowphase.

Deliverables:

- boundary grid precompute;
- movement-cell traversal;
- exact intersection against local segments;
- collision parity tests;
- speed comparison.

### Step 5: Chunked PyTorch Graph Backend

Build `gpu-engine=graph`.

Deliverables:

- static buffers;
- chunk function;
- compile/graph capture;
- chunk-level active check only;
- graph-break reports clean or documented;
- parity tests;
- performance comparison.

### Step 6: Warp Fused Step Backend

Build `gpu-engine=fused` with multi-kernel per-step or graph-captured fused step.

Deliverables:

- optional `gpu-fast` dependency;
- Warp smoke;
- fused control/physics/projection/scoring kernels;
- CPU replay verification;
- parity tests;
- speed comparison.

### Step 7: Persistent Rollout Kernel

Build one-block-per-candidate persistent rollout if launch overhead still dominates.

Deliverables:

- device-side loop;
- final compact result buffers;
- geometry block reductions;
- no host sync until generation end;
- top-K CPU replay;
- speed comparison at `512`, `1024`, `2048+` population.

### Step 8: Larger GPU Search Run

Only after correctness and speed are proven.

Run:

```text
150x60, 18k, no target termination
```

Then run:

```text
512x60 or 1024x60, 18k, no target termination
```

Compare:

- fastest valid lap;
- valid lap count;
- average valid lap time;
- top-decile pace;
- average distance;
- top-decile distance;
- runtime;
- storage.

## Risks And Mitigations

### Risk: Fast GPU Changes Physics

Mitigation:

- CPU oracle never changes.
- Parity battery before serious use.
- CPU replay top candidates.
- Keep eager GPU reference.

### Risk: CUDA Graph Capture Is Brittle

Mitigation:

- capture chunks, not whole full runs;
- static buffers;
- remove CPU work;
- use graph-break logs;
- fallback to eager for debugging.

### Risk: Warp Dependency Adds Complexity

Mitigation:

- optional dependency group;
- small smoke first;
- do not require Warp for CPU workflows;
- keep PyTorch graph backend as fallback.

### Risk: Geometry Acceleration Introduces Edge Bugs

Mitigation:

- local projection fallback;
- adversarial near-boundary tests;
- finish-line tests;
- braking-gate epsilon tests;
- collision grid exact narrowphase;
- CPU replay verification.

### Risk: Fast Mode Loses Debuggability

Mitigation:

- selected CPU telemetry;
- profile leaders replayed;
- deterministic seeds;
- compact final rows;
- optional debug all-telemetry mode;
- manifest paths.

### Risk: Small Populations Still Underuse GPU

Mitigation:

- benchmark `150`, `512`, `1024`, `2048`;
- generate variants per parent;
- do not judge GPU speed only at `150`;
- keep CPU backend for small diagnostics.

## Resolved Scale-Parity Blocker

The first raw `1000x5`, `18k max_steps`, no-target-termination production proof showed that the speed path was real, but raw GPU-only selected winners were not yet trustworthy.

Completed production run:

```text
artifacts/gpu-speed-production-compact-1000x5-18k
```

Important speed result:

- the search itself ran on the persistent fused GPU backend;
- no inline CPU replay ran during production search;
- steady-state throughput was about `616-672 candidates/sec`;
- GPU rollout was about `1.46-1.59s` per `1000` candidates;
- compact materialization and attempt writing were small compared with rollout time.

Important correctness result:

- deferred CPU postcheck replayed only selected candidates, not the full `1000x5` population;
- postcheck failed;
- `postcheck_count=8`;
- `parity_status=failed`;
- `reason_mismatches=5`;
- `valid_lap_mismatches=0`;
- `max_best_progress_delta_m=1929.7`;
- `max_final_progress_delta_m=1929.7`;
- `max_score_delta=135395`.

This means the raw production GPU backend is speed-first only. A GPU-only candidate is not real progress until CPU `MonzaSim` postcheck confirms it.

The failure is concentrated enough to debug:

- candidates `244` and `479` matched CPU postcheck cleanly;
- candidate `793` had the same termination reason but a huge progress/score mismatch;
- candidates related to the `578` lineage repeatedly mismatched termination/progress;
- several bad rows are repeated elite/offspring descendants, so one root divergence can poison later production selection.

Do not treat this as a reason to abandon the GPU backend. Treat it as the exact bug the postcheck system was built to catch.

Resolution:

- Raw GPU fast path: `artifacts/gpu-speed-production-compact-1000x5-18k` remains a speed proof, not a promotion proof.
- Inline rerank correctness-first path: `artifacts/gpu-speed-production-rerank48-1000x5-18k` passed deferred CPU postcheck with `postcheck_count=8`, `parity_status=passed`, `reason_mismatches=0`, `valid_lap_mismatches=0`, and `0.0m` max best/final progress drift. It kept winners trustworthy, but search-time CPU replay dominated runtime.
- Deferred pool rerank speed-first verification path: `artifacts/gpu-speed-deferred-rerank-1000x5-18k` kept inline CPU replay disabled during GPU search, then CPU-replayed a deduped selected pool after the run. The search summed about `7.33s` GPU rollout time / `7.70s` GPU backend time and ended near `643` candidates/sec for the last generation. Deferred postcheck requested `48` pool rows, replayed `11` unique rows, skipped/cached `37` duplicate rows, used `8` workers, and took `100.57s`.
- The deferred pool can fail parity as diagnostics because it intentionally contains inflated GPU candidates. In the final deferred `1000x5` artifact, `pool_parity_status=failed` and `pool_score_parity_status=failed` because rows such as candidate `4:845` had `636.77m` progress drift. Final selected winners are filtered to CPU-clean rows before promotion: `postcheck_count=7`, final `parity_status=passed`, `reason_mismatches=0`, `valid_lap_mismatches=0`, max final-progress delta `0.3695m`, and selected telemetry loaded through `python -m f1rl.replay artifacts\gpu-speed-deferred-rerank-1000x5-18k\selected_telemetry --headless --limit 2`.
- Final promotion contract: GPU proposes at speed; CPU verifies/reranks winners. Inline rerank is correctness-first production mode. Deferred pool rerank is optimized speed-first verification mode. Raw GPU-only runs are profiling/search proposal artifacts and are not trusted without CPU postcheck.

## Completed Scale-Parity Fix Loop

The `1000x5` postcheck failure was closed before any larger scale runs. The required loop was:

1. Load the failed rows from `artifacts/gpu-speed-production-compact-1000x5-18k/postchecked_attempts.jsonl`.
2. Build a focused failing-genome harness for at least candidates `793`, `277`, `578`, `0`, `120`, and `122`.
3. Run each failing genome through:
   - CPU `MonzaSim` oracle;
   - GPU eager reference;
   - GPU persistent Warp fused backend.
4. Compare state/action/feature/termination at step granularity.
5. Find the first divergence step, not just the final crash.
6. Determine whether the first divergence is:
   - compact artifact reconstruction;
   - seed/genome/snapshot mismatch;
   - controller feature mismatch;
   - throttle/brake/steer dominance mismatch;
   - target-speed/braking-gate feature mismatch;
   - physics/grip/clamping mismatch;
   - projection/local-window mismatch;
   - drivable mask or collision semantics mismatch;
   - termination priority mismatch;
   - score/profile accumulation mismatch.
7. Patch the root cause in the shared backend semantics, not only in the reporting layer.
8. Add a focused regression test for the failing candidate or a reduced reproducer.
9. Re-run the focused failing-genome harness.
10. Re-run strict parity at `150x2x18k`.
11. Re-run production `1000x5` only after focused parity is clean.
12. Re-run deferred postcheck on the new `1000x5`.

The blocker is closed by the corrected `1000x5` postcheck artifacts above:

- `parity_status=passed`;
- `reason_mismatches=0`;
- `valid_lap_mismatches=0`;
- max progress deltas within the documented tolerance;
- selected telemetry loads through replay.

If the first divergence is long-horizon float32 drift rather than a single local bug, do not hand-wave it. Add a robust strategy, such as tighter CPU/GPU math parity, periodic projection stabilization, exact-grid consistency fixes, or CPU-verified re-ranking of GPU-selected elites. Document the tradeoff clearly.

## What Not To Do

- Do not keep trying to fix speed by only compiling tiny helper functions.
- Do not call the current eager backend "done" because it uses CUDA tensors.
- Do not use all-candidate full telemetry during speed benchmarks.
- Do not insert CPU `.item()` checks in the per-step loop.
- Do not allocate new tensors inside captured/compiled hot paths.
- Do not project every car against every centerline segment every step.
- Do not intersect every movement against every boundary segment every step.
- Do not remove CPU verification to make GPU look faster.
- Do not move PPO first. Evolution search is the correct first speed target.
- Do not hard-code a Parabolica/Roggia-only geometry shortcut. The solution must remain full-track general.
- Do not start larger scale runs while `1000x5` postcheck is failing.
- Do not mark the GPU speed goal complete while selected production candidates fail CPU postcheck.
- Do not tune rewards or selection against GPU-only winners from a failing postcheck run.
- Do not bury a parity failure by lowering postcheck coverage or weakening tolerances.

## Success Criteria

The GPU speed work is successful only when all of this is true:

1. CPU backend still passes all tests.
2. GPU eager reference still passes all tests.
3. GPU graph/fused backend passes randomized parity.
4. Top GPU candidates replay correctly through CPU.
5. Output artifacts remain compatible with replay/analysis.
6. `150x60`, `18k`, no target termination completes faster than CPU and eager GPU.
7. Larger GPU populations show clear scaling.
8. Profiler shows reduced launch overhead and higher GPU utilization.
9. No final learning claims are made from unverified GPU-only rollouts.
10. Documentation records backend, speed, parity, and replay-verification status for every serious run.
11. `1000x5`, `18k`, production-mode compact search passes deferred CPU postcheck on selected candidates.
12. Scale runs use GPU for search and CPU only for explicit deferred postcheck, never inline hot-path replay.

## Immediate Next Goal Prompt

Use this file as the operating plan for GPU speed. Do not rewrite the simulator. Keep CPU `MonzaSim` as the oracle. Treat the current PyTorch GPU backend as a correctness reference, then build a true high-throughput GPU evolution backend in stages: profile first, remove hot-loop syncs/allocations, add local projection and collision broadphase, add chunked CUDA Graph/`torch.compile` execution, then add a Warp fused rollout backend if PyTorch graphing is still not fast enough.

Every speed change must preserve physics, geometry, checkpoint/lap validity, termination, observations/features, scoring profiles, artifact schemas, and CPU replay compatibility. Run parity tests, randomized batteries, edge-case tests, GPU PPO smoke, CPU/GPU evolution smoke, and performance benchmarks. Do not promote or trust GPU search results until selected top candidates replay correctly through CPU `MonzaSim`. Keep going until the GPU backend is both correct and actually faster under the same `150x60`, `18k max_steps`, no-target-termination constraints, then scale population upward to prove GPU throughput.

Finalization status: the `1000x5` scale-parity blocker is closed by the inline-rerank and deferred-pool-rerank artifacts recorded above. No larger run is required for this goal-agent finalization; future scale runs can build on the verified production/postcheck split.
