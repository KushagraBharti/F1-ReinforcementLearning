# Physics V2 Learned Policy Goal

This is a goal prompt for implementing the next-generation physics and learned-policy route.

The target is to build a more realistic, FastF1-calibrated simulator version without corrupting current `v1` results, then rerun search and learning under that new physics contract.

Do not mutate current physics silently. Do not train v2 policies from v1 data. Do not call GPU v2 ready until CPU/GPU parity is proven. Continue until physics versioning, CPU v2, GPU v2, FastF1 calibration, v2 GPU ES, v2 dataset export, SAC training, replay, documentation, and validation are implemented end to end.

## Objective

Build the full v2 pipeline:

```text
FastF1 Monza reference telemetry
  -> calibrated physics_v2 CPU oracle
  -> GPU physics_v2 parity
  -> new GPU ES under physics_v2
  -> CPU-verified v2 winners
  -> v2 transition dataset
  -> behavior-cloned v2 actor
  -> SAC fine-tuned v2 learned policy
  -> v2 CPU-verified normal-start policy laps
  -> 1000-car policy checkpoint swarm replay
```

The v2 project is a separate benchmark category:

- `v1 ES`
- `v1 learned policy`
- `v2 ES`
- `v2 learned policy`

Do not mix success claims across categories.

## Final Success Criteria

Mark this goal complete only when all of these are true:

1. Simulator versioning exists and every run/dataset/policy artifact records `physics_model`.
2. `physics_model=v1` preserves existing current-physics behavior and tests.
3. `physics_model=v2` exists in CPU `MonzaSim` and is selected explicitly.
4. Physics v2 is calibrated against FastF1 Monza telemetry targets.
5. CPU v2 has focused unit tests for each new physics component.
6. GPU v2 matches CPU v2 behaviorally across randomized parity batteries before any large ES run.
7. GPU v2 ES can run at meaningful scale using the same speed-first/search-first architecture.
8. V2 ES winners are CPU postchecked/reranked under CPU v2.
9. A v2 transition dataset is exported only from CPU-verified v2 trajectories.
10. A learned v2 policy is trained with BC plus SAC.
11. V2 policy evaluation uses CPU v2 as the promotion oracle.
12. A replay mode exists for v2 policy checkpoint swarms so the user can visually inspect many cars improving over checkpoints.
13. Documentation clearly separates v1 and v2 results.
14. Full validation passes:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

## Non-Negotiables

- Do not silently change current v1 physics.
- Do not train v2 policies from v1 ES telemetry.
- Do not compare v1 and v2 lap times as if they are the same benchmark.
- Do not scale GPU v2 ES before CPU/GPU v2 parity passes.
- Do not trust raw GPU v2 winners without CPU v2 postcheck/rerank.
- Do not drop replay compatibility.
- Do not remove CPU PPO, CPU ES, GPU ES, GPU PPO, telemetry, replay, or postcheck paths.
- Do not add heavy dependencies without a clear reason and a small smoke proving they work.
- Do not declare realism based only on feeling. Use FastF1 calibration reports and measurable error bands.

## Why Physics V2

Current physics is intentionally simple and has already been useful:

- bicycle steering model;
- steering response/inertia;
- speed-sensitive steering reduction;
- drag;
- rolling resistance;
- acceleration and brake caps;
- grip-limited combined lateral/longitudinal capacity;
- aero grip increasing with speed;
- max speed clamp;
- track projection;
- collision;
- drivable mask;
- checkpoints;
- lap validity.

That was enough for GPU ES to discover an `81.233s` CPU-verified evolved lap. The next step is not to throw it away. The next step is to create an explicitly versioned v2 that is more realistic and calibrated to real Monza timing/telemetry.

Physics v2 should make the learned behavior less like "game-line exploitation" and more like a real racing control problem:

- braking points should matter;
- tire slip should matter;
- load transfer should matter;
- throttle application should matter;
- curbs should matter;
- gear/torque behavior should matter;
- speed traces should resemble FastF1 telemetry;
- PPO, ES, SAC, replay, and eval should all see the same versioned physics.

## Research Grounding

FastF1 is the calibration source because it exposes real F1 timing, car telemetry, position, tyre/weather/session data, and examples for fastest-lap telemetry.

Useful FastF1 channels and APIs:

- `session = fastf1.get_session(...)`
- `session.load(...)`
- `session.laps.pick_fastest()`
- `lap.get_car_data()`
- `lap.get_pos_data()`
- `lap.get_telemetry()`
- `Telemetry.add_distance()`
- `Speed`
- `Throttle`
- `Brake`
- `nGear`
- `RPM`
- `DRS`
- `X`, `Y`, `Z`
- `Time`
- `Distance`

FastF1 documentation notes that `get_telemetry()` merges car and position data and may include interpolation, while `get_car_data()` and `get_pos_data()` are recommended when only those channels are needed. FastF1 also documents `add_distance()` and warns that integrated distance can accumulate error over long slices, so calibration should work lap-by-lap and sanity-check distance alignment.

Relevant sources:

- [FastF1 introduction](https://docs.fastf1.dev/)
- [FastF1 telemetry API reference](https://docs.fastf1.dev/api_reference/telemetry.html)
- [FastF1 core timing and telemetry data](https://docs.fastf1.dev/core.html)
- [FastF1 examples](https://docs.fastf1.dev/examples/index.html)

Vehicle dynamics grounding:

- Pacejka's Magic Formula is a standard tire-force modeling family. It models tire force as a nonlinear function of slip and load, and can be parameterized by coefficients such as `B`, `C`, `D`, and `E`.
- MathWorks' Magic Formula tire-road interaction documentation describes longitudinal tire force, vertical load, wheel slip, Magic Formula coefficients, and load-dependent parameterization.
- Pacejka's `Tire and Vehicle Dynamics` is a standard reference for tire modeling and vehicle dynamics.

Relevant sources:

- [MathWorks Tire-Road Interaction Magic Formula](https://www.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html)
- [Pacejka, Tire and Vehicle Dynamics](https://books.google.com/books?id=926T_pblHqQC)

SAC grounding:

- SAC is off-policy and continuous-action, which fits throttle/brake/steer learning from ES replay data plus online rollouts.
- Use SAC only for this plan.

Relevant sources:

- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/v2.3.0/modules/sac.html)

GPU grounding:

- The repo already uses PyTorch and NVIDIA Warp for GPU simulation.
- Warp is designed for GPU-accelerated simulation, robotics, and machine learning, and is appropriate for fused kernels once the CPU v2 oracle is stable.

Relevant source:

- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)

## Library And Setup Decision

Use custom PyTorch SAC for the production learned-policy path.

Do not install a separate SAC framework as the main route. Stable-Baselines3 SAC may be used only as a reference or small smoke baseline, because the production path should keep rollout collection and replay-buffer training aligned with the repo's GPU batch simulator.

Physics v2 does need calibration dependencies that are not currently part of the base dependency set.

Add an optional calibration extra before implementing the FastF1 fetch/calibration tools:

```toml
[project.optional-dependencies]
calibration = [
  "fastf1>=3.5.0",
  "scipy>=1.14.0",
]
```

Use:

- `fastf1` to fetch real Monza timing, car telemetry, and position data;
- `scipy` for interpolation, curve fitting, optimization, and calibration loss minimization;
- existing `numpy` for array processing;
- existing PyTorch/Warp for GPU v2 parity and production search.

Required setup before v2 work:

```powershell
uv sync --active --all-extras --all-packages
uv run --no-sync f1-hardware-check --json --warp-smoke
uv run --no-sync python -m f1rl.calibration --json
```

Do not make FastF1 network access a requirement for normal unit tests. Fetch once, cache processed references under `assets/reference/` or `artifacts\calibration`, and keep tests based on checked-in or local cached summaries.

## Existing Calibration Surface

The repo already has calibration scaffolding:

- `src/f1rl/calibration.py`
- `tests/test_calibration.py`
- `assets/reference/monza_2024_Q_VER_telemetry.csv`
- `assets/reference/monza_2024_Q_VER_summary.json`

Current tests expect the reference target to be in a realistic Monza band:

- lap time between `79.0s` and `81.0s`;
- max speed between `340kph` and `355kph`;
- mean speed between `250kph` and `270kph`.

Physics v2 should build from this instead of inventing an unrelated calibration pipeline.

## FastF1 Calibration Plan

### Calibration Data Sources

Start with the existing reference:

- `assets/reference/monza_2024_Q_VER_telemetry.csv`
- `assets/reference/monza_2024_Q_VER_summary.json`

Then add an official FastF1 fetch/update tool:

```powershell
uv run --no-sync python -m f1rl.fastf1_calibration fetch `
  --year 2024 `
  --event Monza `
  --session Q `
  --driver VER `
  --output-dir assets\reference\fastf1
```

Also support:

- fastest lap by session;
- driver-specific fastest lap;
- multiple years;
- dry-only filtering;
- quick-lap filtering;
- optional comparison across drivers.

The tool should cache raw FastF1 data and save processed summaries so tests do not require network access.

### Calibration Channels

Use the following channels when available:

- distance;
- time;
- speed;
- throttle;
- brake;
- gear;
- RPM;
- DRS;
- X/Y/Z position;
- session timing;
- lap time;
- sector times;
- track status;
- weather/tyre context if available.

Derived channels:

- distance-indexed speed trace;
- acceleration trace;
- braking-zone start/end;
- braking deceleration profile;
- corner entry/mid/exit speeds;
- approximate path curvature from X/Y;
- approximate lateral acceleration from speed and curvature;
- throttle application profile;
- gear shift points;
- DRS active regions;
- microsector timing if available or distance-bin timing if not.

### Calibration Targets

At minimum, v2 must report and track:

- lap time;
- max speed;
- mean speed;
- p10/p50/p90 speed;
- sector times;
- straight-line acceleration time;
- terminal speed;
- main braking distances;
- braking deceleration envelope;
- min speed per major corner;
- exit speed per major corner;
- distance-indexed speed error;
- throttle/brake zone overlap with reference;
- gear/RPM plausibility;
- path curvature distribution;
- lateral-g distribution;
- DRS/drag behavior if modeled.

Recommended Monza sections:

- main straight;
- Rettifilo braking and exit;
- Curva Grande;
- Roggia;
- Lesmo 1;
- Lesmo 2;
- Serraglio;
- Ascari;
- back straight;
- Parabolica/Alboreto.

### Calibration Loss

Create a calibration objective that reports separate terms:

- lap-time error;
- max-speed error;
- mean-speed error;
- distance-speed RMSE;
- braking-point error;
- braking deceleration error;
- corner min-speed error;
- exit-speed error;
- throttle/brake classification error;
- gear shift/torque plausibility;
- path/curvature plausibility.

Do not reduce all realism to one score. Report the separate terms and a weighted aggregate.

### Data Quality Rules

FastF1 data can include interpolation and distance integration error. Therefore:

- align by lap and distance, not by raw session time only;
- use single-lap slices for distance integration;
- sanity-check total lap distance against Monza length;
- keep raw and processed data;
- record FastF1 version;
- record session, driver, year, event, and track status;
- avoid calibrating from wet laps or laps with traffic/track-status issues unless deliberately labeled.

## Physics Versioning Plan

Add an explicit physics model id everywhere:

```text
physics_model: v1 | v2
physics_version: semantic or hash string
physics_calibration_id: optional v2 calibration id
```

Places that must carry the id:

- `SimConfig`;
- training configs;
- evolution configs;
- checkpoint files;
- dataset manifests;
- policy metadata;
- replay manifests;
- telemetry summaries;
- postcheck summaries;
- benchmark outputs.

Expected CLI examples:

```powershell
uv run --no-sync python -m f1rl.evolution_search --physics-model v1 ...
uv run --no-sync python -m f1rl.evolution_search --physics-model v2 ...
uv run --no-sync python -m f1rl.train --physics-model v1 ...
uv run --no-sync python -m f1rl.gpu_ppo --physics-model v2 ...
uv run --no-sync python -m f1rl.sac_train --physics-model v2 ...
```

Default should remain `v1` until v2 is fully validated and deliberately selected.

Do not let old artifacts become ambiguous. If an artifact does not declare a physics model, treat it as legacy `v1` only when safe and documented.

## Physics V2 Components

### 1. Tire Slip Angle And Tire Force Curve

Purpose:

- Make lateral grip depend on how the tires are being asked to move, not just a global grip cap.
- Make corner entry, trail braking, throttle application, and oversteer/understeer behavior matter.

Minimum v2 model:

- track vehicle body velocity in local coordinates;
- compute front/rear slip angles;
- compute lateral tire forces from a saturating curve;
- combine lateral and longitudinal force limits;
- model grip falloff after the peak so too much steering can slow or destabilize the car;
- keep formulas GPU-friendly.

Start with a "Pacejka-lite" or brush-style tire curve before a full complex Magic Formula:

```text
alpha_front = atan2(v_y + a * yaw_rate, max(abs(v_x), eps)) - steer_angle
alpha_rear  = atan2(v_y - b * yaw_rate, max(abs(v_x), eps))
F_y_front   = tire_curve(alpha_front, F_z_front, surface_mu)
F_y_rear    = tire_curve(alpha_rear, F_z_rear, surface_mu)
```

The tire curve should expose parameters like:

- cornering stiffness;
- peak friction coefficient;
- slip angle at peak;
- post-peak falloff;
- load sensitivity;
- surface multiplier.

FastF1 calibration use:

- match corner min speeds;
- match lateral-g distribution;
- match braking/turn-in behavior;
- match speed drop through chicanes and high-speed turns.

Tests:

- zero slip produces near-zero lateral force;
- increasing slip increases force until peak;
- beyond peak, force saturates or falls by configured amount;
- load changes peak force;
- surface multiplier changes peak force;
- CPU/GPU curves match within tolerance.

### 2. Weight Transfer

Purpose:

- Make braking, acceleration, and cornering change available grip at each tire/axle.
- Make hard braking into corners different from steady-state turning.

Minimum v2 model:

- longitudinal load transfer from acceleration/braking;
- lateral load transfer from cornering;
- front/rear static weight distribution;
- center of gravity height;
- track width;
- wheelbase.

Core idea:

```text
delta_fz_long = mass * accel_long * cg_height / wheelbase
delta_fz_lat  = mass * accel_lat  * cg_height / track_width
```

FastF1 calibration use:

- braking distance and deceleration shape;
- turn-in stability;
- throttle-on exit behavior;
- corner exit speed.

Tests:

- braking moves load forward;
- acceleration moves load rearward;
- left/right cornering shifts lateral load;
- total vertical load remains approximately mass * g;
- grip changes consistently with load.

### 3. Gear And Torque Curve

Purpose:

- Make acceleration and throttle behavior more realistic.
- Prevent a single constant drive acceleration from dominating the entire speed range.
- Let FastF1 gear/RPM telemetry constrain acceleration behavior.

Minimum v2 model:

- gear ratios;
- final drive;
- wheel radius;
- engine torque curve or power curve;
- automatic shift logic;
- RPM calculation;
- optional shift delay;
- traction-limited drive force.

Simplified path:

- Start with automatic gears.
- Actor still outputs throttle/brake/steer only.
- Gear is environment state, not policy action.
- Add manual gear output only after automatic gear behavior is stable and tested.

FastF1 calibration use:

- gear at distance;
- RPM envelope;
- shift points;
- straight acceleration;
- top speed;
- throttle traces.

Tests:

- RPM increases with speed within a gear;
- gear shifts at configured RPM;
- torque changes with RPM;
- top speed is plausible;
- CPU/GPU gear updates match.

### 4. Brake Bias And Brake Lock Tendency

Purpose:

- Make braking behavior more realistic, especially high-speed braking zones.
- Penalize unrealistic full-brake turning if it would overload the tires.

Minimum v2 model:

- front/rear brake bias;
- max brake torque/force;
- tire longitudinal slip approximation;
- lock tendency when demanded brake force exceeds available tire force;
- reduced steering authority or lateral grip under severe lock.

FastF1 calibration use:

- braking zone start/end;
- deceleration envelope;
- minimum speed at chicanes;
- brake release timing.

Tests:

- stronger brake input increases decel until tire limit;
- excessive brake can trigger lock indicator;
- brake bias changes front/rear force split;
- braking while turning reduces available lateral force through combined grip.

### 5. Curbs And Surface Modifiers

Purpose:

- Make line choice and track limits more meaningful.
- Let curbs be faster or riskier depending on section, rather than just binary drivable/off-track.

Minimum v2 model:

- surface classification map:
  - racing surface;
  - curb;
  - runoff;
  - grass/gravel/off-track;
- friction multiplier;
- rolling resistance multiplier;
- bump/instability scalar if simple;
- track-limit validity semantics preserved.

FastF1 calibration use:

- path comparison against position data;
- corner exit speed;
- realistic penalty for cutting too far.

Tests:

- surface lookup stable CPU/GPU;
- curb friction differs from asphalt;
- off-track invalidates lap as before;
- replay can visualize surface if useful.

### 6. Better Car Collision Shape

Purpose:

- Current collision is good enough for v1 search, but v2 should avoid unrealistic point-like survival.

Minimum v2 model:

- oriented rectangle or capsule collision;
- front/rear/side contact checks;
- still GPU-friendly;
- exact CPU/GPU parity.

Tests:

- point on centerline survives;
- side over boundary collides;
- rotated car collision behaves correctly;
- CPU/GPU collision classifications match.

### 7. DRS And Aero Drag/Downforce Split

Purpose:

- FastF1 telemetry includes DRS, and Monza top speed depends heavily on drag.
- Current v1 has simple drag and aero grip; v2 should separate drag and downforce more explicitly.

Minimum v2 model:

- drag force proportional to speed squared;
- downforce/grip contribution proportional to speed squared;
- optional DRS state that reduces drag and downforce in allowed zones;
- automatic DRS schedule from reference or track zones.

Initial actor output should remain throttle/brake/steer. DRS can be automatic/rule-based first.

FastF1 calibration use:

- top speed;
- acceleration on straights;
- DRS active regions;
- braking stability after DRS zones.

Tests:

- DRS reduces drag in allowed zones;
- DRS disabled in corners/braking zones if rule requires;
- top speed changes plausibly;
- CPU/GPU DRS behavior matches.

## CPU Implementation Plan

CPU v2 is the oracle.

Add:

- `PhysicsModelConfig` or equivalent.
- `CarParamsV2` or nested `PhysicsV2Params`.
- versioned physics dispatch in `src/f1rl/physics.py`.
- explicit v1/v2 path in `SimConfig`.
- v2 state variables as needed:
  - local velocity components;
  - yaw rate;
  - gear;
  - RPM;
  - wheel slip/lock flags;
  - surface id;
  - tire force diagnostics.

Rules:

- v1 default behavior remains unchanged.
- v2 diagnostics are optional in telemetry but available for debugging.
- If telemetry schema changes, keep old readers compatible.
- Do not require FastF1 network access for normal tests.

CPU unit tests:

- v1 unchanged smoke;
- v2 straight acceleration;
- v2 braking;
- v2 steady turning;
- v2 tire curve;
- v2 weight transfer;
- v2 gear/RPM;
- v2 brake lock;
- v2 surface lookup;
- v2 collision shape;
- v2 telemetry fields.

## GPU Implementation Plan

GPU v2 must match CPU v2 behaviorally before large ES.

Implementation sequence:

1. Add tensor/state support for v2 fields in GPU types.
2. Implement PyTorch batch v2 physics first.
3. Add CPU/GPU parity tests against the PyTorch path.
4. Implement fused/Warp v2 kernels only after PyTorch parity is stable.
5. Add Warp parity tests.
6. Add production GPU ES support with `--physics-model v2`.

Do not optimize before correctness.

GPU parity test categories:

- single-step fixed action parity;
- randomized state/action parity;
- short rollout parity;
- medium rollout parity;
- long rollout parity;
- high-speed braking edge cases;
- chicane/curb edge cases;
- off-track/collision edge cases;
- lap-complete/checkpoint edge cases;
- controller genome rollouts;
- phase genome rollouts;
- progress-phase genome rollouts;
- mixed surfaces;
- gear shift boundaries;
- tire saturation boundaries.

Acceptance should be behavioral, not bitwise:

- termination reason matches;
- checkpoint/lap validity matches;
- final progress delta within agreed tolerance;
- speed/heading/lateral error within agreed tolerance;
- score/rank correlation stable;
- selected winners CPU postcheck clean.

Do not hide mismatches by weakening tolerances. If a mismatch affects winner selection, fix the cause or rerank with CPU v2.

## FastF1 Calibration Implementation

Add or extend:

- `src/f1rl/fastf1_calibration.py`
- `src/f1rl/calibration.py`
- `tests/test_calibration.py`

Commands:

```powershell
uv run --no-sync python -m f1rl.fastf1_calibration fetch --year 2024 --event Monza --session Q --driver VER
uv run --no-sync python -m f1rl.fastf1_calibration summarize assets\reference\fastf1\monza_2024_Q_VER
uv run --no-sync python -m f1rl.fastf1_calibration compare --physics-model v2 --calibration-id monza_2024_Q_VER
uv run --no-sync python -m f1rl.calibration --json
```

Calibration artifacts:

```text
assets/reference/fastf1/
  monza_2024_Q_VER/
    raw_metadata.json
    car_data.parquet or csv
    pos_data.parquet or csv
    telemetry.csv
    summary.json
    distance_speed_trace.csv
    corner_targets.json
    braking_targets.json
```

If parquet adds dependency friction, use CSV/JSON first.

Calibration report must include:

- target values;
- v1 estimates;
- v2 estimates;
- error terms;
- pass/fail gates;
- config hash;
- calibration id.

## V2 ES Strategy

After CPU v2 and GPU v2 parity pass, rerun ES from scratch.

Do not seed v2 ES from v1 ES as truth. Optional v1-inspired genomes can be used only as labeled warm-start proposals after v2 random/search baselines exist.

Recommended initial sequence:

1. Tiny v2 GPU ES smoke.
2. 1000 x 5 v2 production smoke.
3. Deferred CPU v2 postcheck.
4. 1000 x 75 v2 run.
5. Deferred CPU v2 postcheck/rerank.
6. Inspect replay.
7. Tune scoring/selection for v2.
8. Scale only after storage and postcheck are healthy.

V2 scoring should initially reuse current speed-focused profiles:

- `fast_valid_lap`
- `time_attack`
- `lap_pace`
- `fast_frontier`
- `frontier_fast`
- `farthest_distance`
- `clean_distance`
- `max_progress`

Then adjust only with evidence from v2 bottlenecks.

Required v2 postcheck:

```powershell
uv run --no-sync python -m f1rl.evolution_postcheck artifacts\runs\v2-es-run `
  --top-k 8 `
  --candidate-pool-size 96 `
  --cpu-rerank `
  --workers 8 `
  --telemetry-compression gzip `
  --physics-model v2
```

The exact CLI may differ after implementation, but the behavior must exist.

## V2 Dataset Strategy

The v2 dataset must be fresh.

Rules:

- `physics_model=v2` required.
- `physics_calibration_id` required.
- v1 transition data must not be included.
- v1 policy checkpoints must not initialize v2 SAC unless explicitly labeled as transfer learning and evaluated separately.
- Dataset schema can match v1 learned-policy dataset, with added v2 fields.

Additional v2 transition fields:

- gear;
- RPM;
- DRS state if modeled;
- surface id;
- tire slip angles;
- tire force diagnostics;
- wheel lock flag;
- weight transfer/load estimate;
- local velocity components if part of state.

Do not add all diagnostics as policy inputs by default. Store them for analysis. Add them to policy observation only after tests and ablations show benefit.

## V2 Learned Policy Plan

Use the same high-level learned-policy pipeline as v1:

```text
v2 verified ES telemetry
  -> v2 transition dataset
  -> v2 BC actor
  -> v2 SAC actor
  -> v2 CPU eval
  -> v2 swarm replay
```

Policy inputs:

- start with v2 `racing_v2` equivalent or `learned_policy_v2` observation profile;
- include only stable, normalized features;
- include upcoming braking/curvature features if they are not already represented;
- avoid hidden private simulator internals as mandatory inputs.

Policy outputs:

```text
[throttle, brake, steer]
```

Gear and DRS:

- automatic gear first;
- automatic/rule-based DRS first;
- only add gear/DRS actor outputs if v2 experiments prove it is necessary.

SAC:

- use v2 ES dataset as replay prefill;
- initialize from v2 BC;
- evaluate under CPU v2;
- produce checkpoint swarm replays;
- compare to v2 ES, not v1 ES only.

## V2 Policy Swarm Replay Requirement

The user wants to watch learning improve visually.

Implement checkpoint-based swarm replay for v2 as well:

- run `1000` policy rollouts per checkpoint;
- group replay by checkpoint;
- support skip-checkpoint control;
- support speed up/down inside the replay window;
- highlight fastest valid, farthest, cleanest, and selected policy;
- preserve replay compatibility with existing `f1rl.replay` where possible.

This should look like the ES multi-car replay, but the grouping axis is policy checkpoint instead of ES generation.

## PPO Under V2

PPO remains part of the repo, but it is not the lead path.

V2 should still support:

- CPU PPO smoke under `physics_model=v2`;
- GPU PPO smoke under `physics_model=v2`;
- policy eval/replay under `physics_model=v2`.

Do not resume blind PPO tuning as the main route. Use v2 ES and v2 SAC as the primary path.

## Implementation Phases

### Phase 0: Protect V1

Tasks:

- add explicit `physics_model` config plumbing;
- default to `v1`;
- update metadata writers;
- add regression tests proving v1 outputs are unchanged for representative rollouts.

Validation:

```powershell
uv run --no-sync pytest -q tests/test_sim.py tests/test_gpu_physics.py tests/test_calibration.py
```

### Phase 1: FastF1 Calibration Tools

Tasks:

- add FastF1 fetch/cache command;
- add processed summary generation;
- add distance-indexed target traces;
- add calibration report comparing v1/v2 estimates;
- keep tests network-free by using checked-in reference summaries.

Validation:

```powershell
uv run --no-sync python -m f1rl.calibration --json
uv run --no-sync pytest -q tests/test_calibration.py
```

### Phase 2: CPU Physics V2

Tasks:

- implement v2 car parameters;
- implement v2 tire force model;
- implement weight transfer;
- implement gear/torque curve;
- implement brake bias/lock tendency;
- implement surface modifiers;
- implement better collision shape if in scope for first v2;
- add telemetry diagnostics.

Validation:

- focused unit tests for every component;
- v2 smoke lap with manual/scripted controls;
- calibration report generated.

### Phase 3: CPU V2 Calibration

Tasks:

- tune parameters to FastF1 targets;
- record calibration id;
- store calibration config;
- document accepted error bands.

Suggested calibration gates:

- lap capability band around reference target;
- max speed within reference band;
- mean speed within reference band;
- main braking distances plausible;
- corner min speeds plausible;
- lateral-g distribution plausible.

Do not require learned policy or ES to immediately match FastF1 exactly before physics is usable. Require the simulator capability envelope to be plausible and measured.

### Phase 4: GPU V2 Parity

Tasks:

- implement PyTorch batch v2;
- implement Warp/fused v2 only after PyTorch parity;
- add randomized parity battery;
- add edge-case tests;
- add postcheck safeguards.

Required parity battery:

- 100 to 1000 random start states;
- multiple fixed action tapes;
- controller, phase, and progress-phase genomes;
- short, medium, and long rollouts;
- collision/off-track/checkpoint/lap-complete edge cases;
- CPU vs GPU final state, termination, and score deltas.

### Phase 5: V2 GPU ES

Tasks:

- run tiny smoke;
- run 1000x5 smoke;
- CPU postcheck;
- replay selected telemetry;
- tune scoring only from evidence;
- scale to meaningful run.

Validation:

- GPU search finishes;
- postcheck clean;
- selected replay loads;
- generation metrics stream;
- storage stays controlled.

### Phase 6: V2 Dataset

Tasks:

- export verified v2 transitions;
- dataset manifest includes physics/calibration ids;
- v1/v2 incompatibility enforced;
- dataset QA reports generated.

Validation:

- dataset tests;
- corrupt/mismatched physics model fails clearly;
- replay source check passes.

### Phase 7: V2 BC And SAC

Tasks:

- train v2 BC;
- smoke/overfit test;
- train v2 SAC from v2 dataset;
- CPU v2 eval;
- swarm replay by checkpoint;
- compare v2 learned policy to v2 ES.

Validation:

- BC loss behaves;
- SAC smoke passes;
- CPU v2 eval works;
- selected telemetry loads;
- learned policy completes valid laps under v2.

### Phase 8: Documentation And Benchmarking

Tasks:

- update README;
- update `Documentation.md`;
- add commands;
- record current v1 and v2 scoreboard separately;
- document all caveats.

Final validation:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

## Artifact Layout

Recommended:

```text
artifacts\
  calibration\
    monza_2024_Q_VER_v2_calibration\
  v2-es\
    gpu-es-v2-YYYYMMDD\
  datasets\
    v2-es-policy-dataset-YYYYMMDD\
  learned\
    v2-bc-YYYYMMDD\
    v2-sac-YYYYMMDD\
```

Every folder should contain a manifest that names:

- physics model;
- physics version;
- calibration id;
- sim config hash;
- track hash;
- git commit;
- source artifacts;
- validation status.

## Rollback Plan

If v2 makes search or learning much harder:

1. Do not delete v2 work.
2. Keep v1 as the stable benchmark.
3. Identify which v2 component caused the regression:
   - tire model;
   - load transfer;
   - gear/torque;
   - brake lock;
   - curbs;
   - collision.
4. Disable one component at a time through versioned config.
5. Rerun calibration and small ES.
6. Keep the smallest v2 model that improves realism without destroying learnability.

Do not hide a broken v2 by relaxing validation.

## Metrics To Track

Calibration:

- target lap time;
- simulated capability lap time;
- max/mean/p10/p50/p90 speed;
- distance-speed RMSE;
- braking-point errors;
- corner min-speed errors;
- exit-speed errors;
- lateral-g distribution;
- gear/RPM errors;
- DRS/top-speed effects.

Physics:

- tire slip distribution;
- tire saturation frequency;
- lock events;
- gear shifts per lap;
- surface usage;
- collision/off-track rate.

ES:

- fastest valid lap;
- valid lap rate;
- average valid lap time;
- top-decile pace;
- generation average progress;
- gate pass rates;
- postcheck mismatch rates;
- CPU rerank pool stats.

Learned policy:

- BC action loss;
- SAC losses;
- entropy/alpha;
- valid lap rate;
- fastest valid lap;
- average valid lap time;
- terminal reason counts;
- swarm checkpoint improvement;
- replay-load status.

## Done Means Done

The implementing agent must continue through:

1. versioning;
2. FastF1 calibration tooling;
3. CPU v2 implementation;
4. CPU v2 tests;
5. GPU v2 implementation;
6. CPU/GPU parity tests;
7. v2 ES smoke;
8. v2 postcheck;
9. v2 dataset export;
10. v2 BC;
11. v2 SAC;
12. v2 policy eval;
13. v2 checkpoint swarm replay;
14. documentation;
15. full validation.

If any phase fails, isolate the failure, patch the root cause, rerun focused checks, then rerun the broader gate. Do not mark the goal complete while v2 exists only as partial physics, unverified GPU kernels, untrained policies, or non-replayable artifacts.

## Research Sources

- [FastF1 introduction](https://docs.fastf1.dev/)
- [FastF1 telemetry API reference](https://docs.fastf1.dev/api_reference/telemetry.html)
- [FastF1 core timing and telemetry data](https://docs.fastf1.dev/core.html)
- [FastF1 examples](https://docs.fastf1.dev/examples/index.html)
- [MathWorks Tire-Road Interaction Magic Formula](https://www.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html)
- [Pacejka, Tire and Vehicle Dynamics](https://books.google.com/books?id=926T_pblHqQC)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/v2.3.0/modules/sac.html)
- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
