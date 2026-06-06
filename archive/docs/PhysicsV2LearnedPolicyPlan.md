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

## Current Implementation Checkpoint

Status as of 2026-06-06:

- V1 is preserved as the default `physics_model`.
- V2 is explicitly selectable with `physics_model=v2`.
- V2 metadata is threaded through `SimConfig`, step telemetry, episode summaries, scripted runs, benchmark runs, SB3/GPU PPO smoke metadata, evolution summaries, selected telemetry manifests, postcheck manifests/summaries, dataset manifests, policy eval summaries/manifests, policy swarm manifests, and SAC run config.
- CPU V2 includes initial tire-slip, load-transfer, power-limited drive, brake-lock tendency, and FastF1-median-aligned automatic gear/RPM diagnostics.
- PyTorch GPU V2 implements the same initial contract and has focused CPU/GPU parity tests.
- FastF1 calibration reporting now includes V1 and V2 estimates, separate error terms from the checked-in Monza 2024 VER reference CSV/summary, focused sustained-corner diagnostics, and optional multi-lap section-distribution comparison.
- `src/f1rl/fastf1_calibration.py` provides offline `summarize`, `summarize-multi`, and `compare` commands, plus live FastF1 `fetch` and `fetch-multi` paths for the calibration extra.
- Current calibration id is `monza_2022_2024_fastf1_multilap_v2_manual_balance_fix`; current V2 report path is `artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json`.
- Current FastF1 calibration data includes the checked-in 2024 Italian GP Qualifying VER fastest lap (`79.662s`, `5745.669m`, max speed `348.0 kph`, mean speed `259.914 kph`, gear range `2..8`) plus `60` selected clean dry Q/FP2/FP3 laps from Monza `2024`, `2023`, and `2022` for VER/NOR/PIA/LEC/SAI/HAM/RUS where available.
- Multi-lap artifacts live under `artifacts\calibration\fastf1-multilap-20260605` and keep raw `get_car_data()`, raw `get_pos_data()`, processed `get_telemetry().add_distance()`, per-lap summaries, `manifest.json`, and `section_distribution_summary.json`.
- OpenF1 cross-checks live under `artifacts\calibration\openf1-monza-crosscheck-20260605`; `manifest.json` contains `42` selected Monza 2024/2023 Q/FP2/FP3 laps, `3` explicit 2022 skips because OpenF1 returned 404, lap p50 `80.8025s`, max-speed p90 `349.0 kph`, and speed-trap p90 `345.0 kph`. This is independent sanity evidence, not the primary calibration target.
- Current V2 calibration metrics: terminal-speed error `+3.0 kph`, acceleration trace p95 MAE about `1.15 m/s^2`, braking-zone distance MAE about `15.5m`, robust corner lateral-g margin min about `-0.45g`, robust margin mean about `+0.21g`, sustained-corner reference control pass rate `0.333`, max sustained-corner p95 lateral error about `17.95m`, minimum sustained p75/p90 margins `-2.173/-2.811`, gear match rate `1.0`, mean absolute RPM error under `9`. Raw point curvature/lateral-g spikes from interpolated FastF1 position data remain visible in the JSON report and are not hidden.
- Multi-lap sustained-section clusters are distance-based. Current p50 start/end and speed/lateral-g distributions: section 01 `2424.9..2590.4m`, speed p50/p90 `215.3/222.9 kph`, lateral-g p90 p50/p90 `5.15/5.85`; section 02 `2780.8..2885.5m`, `198.3/205.1 kph`, `4.74/5.29g`; section 03 `3952.9..4035.8m`, `216.0/249.8 kph`, `3.73/4.51g`; section 04 `5052.7..5317.4m`, `228.9/235.7 kph`, `5.35/6.15g`.
- Focused sustained-corner diagnostic starts before each section and reports reference-speed and controlled-speed runs at `150/180/200/220/230/240/250 kph`, including lateral error, heading error, steering saturation, actual/reference curvature, lateral-g, slip angles, front/rear lateral force, throttle/brake, tire saturation, and track-limit/off-track state.
- Manual mode now supports focused section starts with `--start-section sustained_corner_01|02|03`, `--start-section-lead-in-m`, `--start-progress-m`, and `--start-speed-kph`. Section starts align the FastF1 ghost to the matching reference time/distance and manual reset reuses the same start.
- Manual mode now swaps left/right keyboard steering in the renderer to match the observed on-screen response, and the HUD is right-aligned in free screen space instead of covering the left-side driving line.
- Mid-lap/manual-section telemetry summaries now report run-local `distance_traveled_m`; lap sectors already passed before the run start are left blank rather than producing invalid sector speeds.
- QC telemetry summaries now include `sustained_corner_diagnostics` for manual section runs: speed, ghost gap, lateral error, heading error, steering saturation, lateral-g, slip angles, front/rear lateral force, throttle/brake, tire saturation, off-track/collision flags, and a `manual_review_pass` flag.
- Manual mode accepts `--physics-model v2` and a headless smoke passed.
- Scripted, benchmark, SB3 PPO smoke, and GPU PPO smoke entrypoints accept `--physics-model v2`; tiny CLI smokes passed and emitted V2 metadata.
- Tiny V2 fused GPU parity smoke under the manual-approved balance retune wrote `artifacts\runs\v2-manual-balance-fix-gpu-parity-smoke-20260606` with v2.0.10 metadata, `gpu_parity_status=passed`, max CPU/GPU progress delta about `0.00020m`, and `0` reason/valid-lap mismatches.
- Tiny V2 persistent-controller parity smoke wrote `artifacts\runs\v2-recalibration-persistent-controller-parity-smoke-20260605` with `gpu_kernel_backend=warp_persistent_controller_open`, `gpu_parity_status=passed`, `0` CPU replay reason mismatches, and `0` valid-lap mismatches.
- `tools/tests/test_v2_metadata_contracts.py` now proves V2 metadata on CPU evolution summaries/checkpoints/bridge/selected telemetry, CPU postcheck summaries/manifests/rows, ES dataset manifests/source rows, BC/SAC policy checkpoints, policy eval manifests, policy swarm manifests, loaded telemetry rows, and the V1-labeled transfer export guard. `tools/tests/test_policy_io.py` also covers V2 PPO eval config reconstruction from `run_metadata.json`.
- The `127.183s` scripted lap is reclassified as a conservative V2 scripted smoke/debug baseline only, not `scripted_threshold`; it is archived with the other superseded V2 bulk artifacts.
- Any V2 `1000x5` or `1000x10` ES artifacts created before this recalibration are exploratory/pre-recalibration only and must not be used for target, dataset, threshold, or promotion decisions.
- Post-retune validation passed on 2026-06-06 for this manual-approved gate: manual headless V2 ghost section smoke wrote `artifacts\runs\manual-headless-20260606-012833-seed7-300975100`, flying-start smoke wrote `artifacts\runs\manual-headless-20260606-012839-seed7-378636300`, `ruff check .`, `pyright src/f1rl`, full `pytest -q`, `f1-hardware-check --json --warp-smoke`, focused V2 pytest, and V2 fused GPU parity smoke all passed.
- V2 manual handoff checklist: `artifacts\runs\qc-20260606-012846\manual_qc_checklist.md`.
- Manual mode is approved for v2.0.10 as of 2026-06-06. v2.0.3 through v2.0.9 failed or were superseded for handling/longitudinal feel. v2.0.10 keeps the manually approved high-speed cornering balance, slightly eases low-speed grip from v2.0.9, raises acceleration, and softens braking.
- V2 `scripted_threshold` is established from the fastest selected FastF1 Monza calibration lap, not from the old scripted debug lap or the slower simulator reference controller: `79.327s` from 2024 Monza Qualifying NOR lap 11, McLaren, SOFT, track status `1`. Local summary copy: `artifacts\highlights\physics2\calibration\summary.json`; original calibration tree: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`. The V2 ES target is CPU-verified `<=86.327s` (`scripted_threshold + 7s`).
- CPU V2 reference-control baseline `artifacts\runs\reference-control-20260606-022941-seed7-987680600` replay-loaded as valid (`116.2167s`, zero collisions/off-track), but it is not the threshold; the original run is in the D-drive archive.
- V2 staged GPU ES target is met. Run `artifacts\runs\v2-gpu-es-fastf1-1000x5-15000-20260606` was staged from `1000x5` through `1000x50` with `max_steps=15000`. Final CPU postcheck/rerank used `candidate_pool_size=512`; selected parity passed with `0` selected reason mismatches and `0` selected valid-lap mismatches. The best trusted selected V2 lap is generation `46`, candidate `724`, CPU-verified `77.6833s`; local replay copy: `artifacts\highlights\physics2\gpu-es\telemetry\postcheck-cpu_rerank_top_score-rank-000-gen-046-candidate-00724-steps.jsonl.gz`.
- Raw GPU proposal parity is still not clean at broad-pool scale: the final pool had `13` reason mismatches and `12` valid-lap mismatches. This does not invalidate the selected winner, but it reinforces that raw GPU winners are proposal data only and CPU postcheck/rerank is mandatory.
- V2 dataset export completed at `artifacts\datasets\v2-es-policy-dataset-fastf1-1000x50-20260606`: `16` CPU-replayed source candidates, `50128` transitions, `8` valid laps, fastest source lap `77.65s`, mean valid lap `85.925s`, physics `physics_v2.0.10-fastf1-manual-balance-fix`.
- V2 learned-policy target is met. The initial dominance-control BC/SAC attempt failed around the first chicane because the dataset contains simultaneous throttle/brake behavior; retraining with `control-mode independent` fixed closed-loop reproduction. BC source-3 CPU-evaluated at `77.6833s`. SAC now evaluates and preserves the initial BC policy at step `0`; promoted SAC workflow checkpoint CPU-evaluated at `77.6833s`, below the FastF1 threshold `79.327s`. Original checkpoints/evals are in the D-drive archive; local replay copy: `artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz`.
- Final V2 highlights are local under `artifacts\highlights\physics2`: `1006` replay traces total (`5` CPU-reranked GPU ES traces, `1` promoted learned-policy trace, `1000` deterministic best-policy swarm entries), plus exact-pygame `4x` GIFs at `artifacts\highlights\physics2\gpu-es\gifs\gpu-es-cpu-rerank-best-4x.gif` and `artifacts\highlights\physics2\gpu-rl\gifs\learned-policy-promotion-4x.gif`.

Manual handoff command:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_02 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_03 --start-section-lead-in-m 120
uv run --no-sync python -m f1rl.qc --telemetry artifacts\runs\manual-headless-20260606-012833-seed7-300975100 --output-dir artifacts\runs --max-telemetry-files 1
uv run --no-sync python -m f1rl.openf1_crosscheck --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\openf1-monza-crosscheck-20260605 --request-delay-s 1.0
```

Latest section-start smoke: `artifacts\runs\manual-headless-20260606-012833-seed7-300975100`. Latest QC report before threshold recording: `artifacts\runs\qc-20260606-012846`; its automated `manual_gate` block reports `scripted_threshold_status=unset` because it predates the FastF1 threshold correction, and the user manually approved v2.0.10 on 2026-06-06. The QC section-smoke diagnostic is metadata/checklist evidence only because the headless manual smoke runs straight; use the local copied report `artifacts\highlights\physics2\calibration\fastf1-v2-manual-balance-fix-20260606.json` for handling-balance metrics.

FastF1 threshold and CPU controller baseline after manual approval:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode control --physics-model v2 --steps 9000 --seed 7
uv run --no-sync f1-replay artifacts\runs\reference-control-20260606-022941-seed7-987680600\steps.jsonl --headless
```

The threshold source is FastF1 `79.327s` from `artifacts\highlights\physics2\calibration\summary.json` locally, with the original source preserved in the D-drive archive. The CPU controller baseline wrote `artifacts\runs\reference-control-20260606-022941-seed7-987680600`, replay-loaded headlessly, and records `lap_time_s=116.2167`; it is baseline evidence only. Earlier post-manual attempts are superseded failed diagnostics: `artifacts\runs\reference-control-20260606-014926-seed7-381937500` failed `off_track` at `14.0s`, and `artifacts\runs\scripted-20260606-014937-seed7-461099600` failed `collision` at `104.0s`. These original bulk run artifacts are now archived on `D:\`.

Large V2 ES, dataset export, BC, SAC workflow training, CPU learned-policy promotion, local highlight curation, exact-pygame `4x` GIF export, D-drive bulk offload, final validation, commit, and push have now passed their target gates. Final pushed commit: `1c09889 Complete Physics V2 learned policy pipeline`. The old `127.183s` debug lap remains invalid as a threshold. The detailed historical report is `archive/docs/PhysicsV2-Achieved.md`.

## Operating Plan

The future goal agent should execute the v2 work in this order:

1. Implement physics v2 in CPU and GPU together. CPU remains the oracle, but GPU parity work should be developed alongside CPU changes so the two models do not drift.
2. Calibrate physics v2 against FastF1 Monza telemetry as closely as practical. Calibration must produce reports with explicit error terms, not just visual confidence.
3. Add and run CPU/GPU parity tests until v2 behavior is aligned enough for search. Fix real mismatches instead of hiding them with weak tolerances.
4. Stop for human handoff after calibrated CPU/GPU v2 is implemented. The user should run manual mode and confirm that v2 driving feel is plausible before large search or learning starts.
5. Establish the fastest scripted/reference v2 lap after calibration. This lap defines the benchmark threshold for search and learning.
6. Run staged GPU ES under v2, CPU-verified, starting small and scaling only when metrics, telemetry, and storage are healthy.
7. Tune search algorithms, scoring, bottleneck handling, reward shaping, dataset selection, BC, SAC, and rollout collection as needed. After the scripted threshold is established, do not keep changing physics to make learning easier unless a calibration defect is found and documented.
8. Reach GPU ES target: CPU-verified v2 lap time at or below `scripted_threshold + 7s`.
9. Reach GPU RL target: CPU-verified learned v2 policy at or below `scripted_threshold`.
10. Create curated v2 highlights for GPU ES and GPU RL, around `1000` local replay traces total, stratified across generations/checkpoints and performance bands.
11. Export GIFs from the exact pygame replay renderer at `4x` speed, with no approximate custom rendering.
12. Compress/offload all non-highlight bulk artifacts to external storage.
13. Final sanity check, docs update, commit, and push.

The agent should not interpret this as a single straight-line training script. It should repeatedly inspect telemetry, identify bottlenecks, adjust algorithms and reward/scoring/data selection, validate, and continue until the explicit v2 ES and v2 RL targets are reached or a concrete blocker is documented.

## Final Success Criteria

Mark this goal complete only when all of these are true:

1. Simulator versioning exists and every run/dataset/policy/replay artifact records `physics_model`.
2. `physics_model=v1` preserves existing current-physics behavior and tests.
3. `physics_model=v2` exists in CPU `MonzaSim` and is selected explicitly.
4. Physics v2 exists in GPU batch/fused paths and is selected explicitly.
5. CPU and GPU v2 are developed as one contract, with shared parameters, shared calibration ids, and explicit parity tests.
6. Physics v2 is calibrated against FastF1 Monza telemetry targets and produces a calibration report with separate error terms.
7. CPU v2 has focused unit tests for each new physics component.
8. GPU v2 matches CPU v2 behaviorally across randomized parity batteries before any large ES run.
9. Manual mode works under v2 and the user has a handoff point to drive and approve the physics feel before learning/search scales up.
10. A fastest scripted/reference v2 lap is established after calibration and recorded as the benchmark threshold.
11. GPU v2 ES runs through staged scale-up, from small smoke runs toward meaningful production runs such as `1000 x 75`, with compressed telemetry and controlled storage.
12. V2 ES winners are CPU postchecked/reranked under CPU v2.
13. V2 ES reaches `scripted_threshold + 7s` or better under CPU v2 verification.
14. A v2 transition dataset is exported only from CPU-verified v2 trajectories.
15. A learned v2 policy is trained with BC plus SAC.
16. V2 policy evaluation uses CPU v2 as the promotion oracle.
17. V2 learned policy reaches `scripted_threshold` or better under CPU v2 verification.
18. Replay mode exists for v2 policy checkpoint swarms so the user can visually inspect many cars improving over checkpoints.
19. Curated v2 GPU ES and GPU RL highlights exist locally, around `1000` replay traces total, stratified across generations/checkpoints and performance bands.
20. V2 GIFs are exported from the exact pygame replay renderer at `4x` speed.
21. Bulk non-highlight artifacts are compressed/offloaded to external storage, and local highlights remain replayable.
22. Documentation clearly separates v1 and v2 results.
23. Final sanity checks, documentation updates, commit, and push are complete.
24. Full validation passes:

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
- Do not implement CPU v2 and GPU v2 as separate drifting models. They must share one documented physics contract, one parameter set, and one calibration id.
- Do not scale GPU v2 ES before CPU/GPU v2 parity passes.
- Do not trust raw GPU v2 winners without CPU v2 postcheck/rerank.
- Do not start large ES/RL runs before the manual handoff point is complete.
- Do not change calibrated physics after the scripted benchmark threshold is recorded unless a real calibration/parity bug is found, fixed, and documented.
- Do not drop replay compatibility.
- Do not remove CPU PPO, CPU ES, GPU ES, GPU PPO, telemetry, replay, or postcheck paths.
- Do not let v2 generate another uncontrolled local artifact pile. Bulk telemetry/checkpoints must be compressed early and offloaded after curation.
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
- `tools/tests/test_calibration.py`
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
5. Add Warp parity tests. Current V2 open-step Warp parity and V2 persistent-controller Warp parity are implemented and smoke-tested.
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
- `tools/tests/test_calibration.py`

Commands:

```powershell
uv run --no-sync python -m f1rl.fastf1_calibration fetch --year 2024 --event Monza --session Q --driver VER
uv run --no-sync python -m f1rl.fastf1_calibration summarize assets\reference\monza_2024_Q_VER_telemetry.csv --output artifacts\calibration\fastf1-summary.json
uv run --no-sync python -m f1rl.fastf1_calibration fetch-multi --years 2024,2023,2022 --sessions Q,FP2,FP3 --drivers VER,NOR,PIA,LEC,SAI,HAM,RUS --output-dir artifacts\calibration\fastf1-multilap-20260605 --max-laps-per-driver 1
uv run --no-sync python -m f1rl.fastf1_calibration summarize-multi --output-dir artifacts\calibration\fastf1-multilap-20260605
uv run --no-sync python -m f1rl.fastf1_calibration compare --physics-model v2 --multi-summary artifacts\calibration\fastf1-multilap-20260605\section_distribution_summary.json --output artifacts\calibration\fastf1-v2-manual-balance-fix-20260606.json
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
artifacts/calibration/fastf1-multilap-20260605/
  manifest.json
  section_distribution_summary.json
  monza_<year>_<session>/<driver>_lap<lap>/
    car_data_raw.csv
    pos_data_raw.csv
    telemetry_processed_add_distance.csv
    lap_metadata.json
    summary.json
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

After calibrated CPU v2, GPU v2 parity, manual handoff, and scripted benchmark threshold recording pass, rerun ES from scratch.

Do not seed v2 ES from v1 ES as truth. Optional v1-inspired genomes can be used only as labeled warm-start proposals after v2 random/search baselines exist.

Recommended initial sequence:

1. Tiny v2 GPU ES smoke.
2. 1000 x 5 v2 production smoke.
3. Deferred CPU v2 postcheck.
4. 1000 x 10 staged run with compressed telemetry and generation metrics.
5. Deferred CPU v2 postcheck/rerank.
6. Inspect replay, telemetry, gate failures, section bottlenecks, and storage footprint.
7. Tune scoring/selection/mutation/crossover only from evidence.
8. Scale through intermediate runs such as 1000 x 25 and 1000 x 50.
9. Run 1000 x 75 or larger only after postcheck, replay, and storage are healthy.
10. Stop the ES phase only when a CPU-verified winner reaches `scripted_threshold + 7s` or better.

Every staged ES run should write compressed telemetry by default. Keep enough candidate traces for debugging, reranking, bottleneck analysis, and replay curation, but avoid uncompressed full-run dumps unless there is a narrow debugging reason and a cleanup plan.

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

Likely v2 bottleneck work includes:

- braking-zone reward terms if cars are late-braking into invalid exits;
- section-specific progress/pace gates if evolution stalls before Ascari or Parabolica;
- traction/lock penalties if v2 tire physics encourages unrealistic steering while fully braking;
- line diversity pressure if all elites collapse into one fragile exploit;
- staged start positions only as diagnostics or curriculum, not as final proof;
- CPU rerank pool sizing if GPU winners are close but noisy;
- selected-telemetry sampling rules that preserve best, farthest, cleanest, and failure-mode examples.

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
- compare to the v2 scripted threshold and v2 ES, not v1 ES only.

The v2 learned-policy phase should copy the successful v1 operating style, not the old blind PPO loop:

- begin with a small dataset/export smoke and a BC overfit probe;
- use short CPU oracle evals for early diagnostics;
- save expensive multi-episode CPU promotion checks for promising checkpoints;
- keep SAC training on CUDA when available;
- preserve compressed replay telemetry for evals and checkpoint swarms;
- tune observation profile, reward terms, replay-buffer mix, eval cadence, and actor initialization from measured failures;
- iterate until a CPU-verified learned policy reaches `scripted_threshold` or better.

If RL lags ES, the implementing agent should inspect whether the issue is dataset coverage, observation features, reward shape, action distribution, SAC stability, termination distribution, or CPU/GPU mismatch. Do not respond by changing physics unless calibration/parity evidence says the physics is wrong.

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

Final local highlight curation should keep about `1000` replay traces total across v2 GPU ES and v2 GPU RL. Use a stratified strategy like the current RL1 highlight cleanup:

- early, middle, and late generations/checkpoints;
- best-performance band;
- upper-mid band;
- median band;
- lower-progress/failure-mode band;
- selected fastest valid / farthest / cleanest exemplars.

The full uncurated run can live only in compressed external storage. The local repo should keep the curated v2 highlight telemetry and GIFs replayable without restoring the bulk archive.

GIF export must use the exact pygame renderer path that the replay command opens. Approximate matplotlib/OpenCV reconstructions are not acceptable for the final deliverable. The GIF speed target is `4x`, with enough frames to inspect line, braking, crashes, and lap completion.

## PPO Under V2

PPO remains part of the repo, but it is not the lead path.

V2 should still support:

- CPU PPO smoke under `physics_model=v2` (tiny CLI smoke passed);
- GPU PPO smoke under `physics_model=v2` (tiny CPU-device CLI smoke passed);
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
uv run --no-sync pytest -q tools/tests/test_sim.py tools/tests/test_gpu_physics.py tools/tests/test_calibration.py
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
uv run --no-sync pytest -q tools/tests/test_calibration.py
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

### Phase 3: GPU V2 Parity Contract

Tasks:

- implement the same v2 parameters and formulas in GPU batch/fused paths while CPU v2 is being finalized;
- keep CPU v2 as oracle but avoid waiting until the end to discover GPU divergence;
- add shared parameter serialization for CPU/GPU;
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

### Phase 4: V2 Calibration And Manual Handoff

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

Manual handoff gate:

- manual mode can run with `physics_model=v2`;
- basic throttle/brake/steer behavior feels plausible enough for a human smoke;
- calibration report is available for review;
- CPU/GPU parity status is summarized;
- the user has a chance to drive v2 before ES/RL scale-up.

### Phase 4.5: Scripted V2 Benchmark

Tasks:

- establish the fastest scripted/reference v2 lap after calibration and manual handoff;
- record lap time, sector times, speed trace, braking zones, and telemetry summary;
- save replayable scripted telemetry;
- write benchmark metadata with physics version and calibration id.

This scripted time becomes `scripted_threshold`. It is the benchmark for v2 ES and v2 RL. After this point, algorithm work should improve search/learning against the threshold instead of moving the threshold by changing physics.

### Phase 5: V2 GPU ES

Tasks:

- run tiny smoke;
- run 1000x5 smoke;
- run 1000x10 staged search;
- CPU postcheck;
- replay selected telemetry;
- tune scoring only from evidence;
- scale through 1000x25, 1000x50, and toward 1000x75 when metrics justify it;
- keep telemetry compressed and manifests complete;
- inspect bottlenecks and update scoring/search operators as needed;
- reach `scripted_threshold + 7s` or better under CPU v2 verification.

Validation:

- GPU search finishes;
- postcheck clean;
- selected replay loads;
- generation metrics stream;
- storage stays controlled.
- fastest CPU-verified v2 ES lap meets the target.

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
- iterate observation/reward/data/replay-buffer/SAC settings until the learned policy reaches `scripted_threshold` or better.

Validation:

- BC loss behaves;
- SAC smoke passes;
- CPU v2 eval works;
- selected telemetry loads;
- learned policy completes valid laps under v2.
- promoted learned policy meets the scripted-threshold target under CPU v2.

### Phase 8: Highlights, GIFs, Storage, And Finalization

Tasks:

- curate about `1000` local v2 highlight traces total across GPU ES and GPU RL;
- stratify highlights across generations/checkpoints and performance bands;
- export exact-pygame GIFs at `4x`;
- archive/offload all non-highlight bulk artifacts to `D:`;
- verify local highlights replay without restoring bulk artifacts;
- update README;
- update `Documentation.md`;
- add commands;
- record current v1 and v2 scoreboard separately;
- document all caveats.
- commit and push.

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
  highlights\
    physics-v2-YYYYMMDD\
      gpu-es\
      gpu-rl\
      gifs\
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

Bulk v2 artifact policy:

- active runs may write locally while being debugged;
- telemetry should default to compressed JSON/gzip or equivalent compressed formats;
- large raw runs, datasets, checkpoints, and swarm dumps should be archived to `D:\f1-rl-artifacts\archives\...`;
- local repo should keep only curated v2 highlights and GIFs after the goal is complete;
- the full uncurated archives should remain restorable and should include verification metadata such as size, entry count, and SHA256 where practical.

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
5. GPU v2 implementation under the same physics contract;
6. CPU/GPU parity tests;
7. calibration report with accepted error bands;
8. manual-mode handoff;
9. scripted/reference v2 benchmark and saved telemetry;
10. staged v2 ES smoke and scale-up;
11. v2 ES CPU postcheck/rerank;
12. v2 ES target at `scripted_threshold + 7s` or better;
13. v2 dataset export;
14. v2 BC;
15. v2 SAC;
16. v2 policy eval;
17. v2 RL target at `scripted_threshold` or better;
18. v2 checkpoint swarm replay;
19. curated v2 GPU ES and GPU RL local highlights;
20. exact-pygame `4x` GIF exports;
21. external archive/offload of non-highlight bulk artifacts;
22. documentation;
23. full validation;
24. commit and push.

If any phase fails, isolate the failure, patch the root cause, rerun focused checks, then rerun the broader gate. Do not mark the goal complete while v2 exists only as partial physics, unverified GPU kernels, untrained policies, non-replayable artifacts, approximate GIFs, local bulk storage bloat, or undocumented results.

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
