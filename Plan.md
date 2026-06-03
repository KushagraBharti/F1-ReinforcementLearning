# F1RL Current Plan

## Current Status

The simplified Monza simulator and reinforcement-learning proof of concept is implemented and verified end to end.

Completed:

- Legacy implementation archived under `archive/legacy-20260424/`.
- Active runtime rebuilt around `track geometry -> physics -> simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay`.
- Track artifact generated at `assets/tracks/monza/track_spec.npz`.
- Manual driving works with keyboard controls, collision/off-track termination, ray sensors, HUD, reference ghost overlay, and flying-start comparison.
- Replay works with timestamp-based playback, `--speed`, and `--no-timing`.
- Fast-F1 reference ghost works as a perfect telemetry replay baseline.
- Scripted pure-pursuit baseline completes a valid slow lap.
- Gymnasium env and Stable-Baselines3 PPO training/eval work.
- Curriculum/segment spawning works.
- Benchmark harness writes JSON, CSV, per-episode JSONL, and selected telemetry.
- QC tool writes JSON/Markdown reports, HTML dashboard, and manual checklist.
- CUDA-required PPO smoke training works on the RTX 4060 Laptop GPU.
- TensorBoard logs load and render scalar curves.
- Lint/type/test validation passes.

Current proof-of-concept metrics:

- Observation dimension: `18`
- Discrete actions: `9`
- Sensor rays: `7`
- Checkpoints: `120`
- Boundary segments: `1800`
- Simulator step: `60 Hz`
- Fast-F1 reference ghost: `79.662s`, `259.9 kph` average, `348.0 kph` max
- Scripted baseline: valid lap, `214.47s`, `96.0 kph` average, `107.9 kph` max
- PPO scratch initial: `0.0m`, terminates by no-progress
- PPO final checkpoint: `431.4m`, passes `8` checkpoints, terminates off-track
- PPO best checkpoint: `797.6m`, passes `16` checkpoints, terminates collision
- CUDA smoke: `2048` timesteps, `device=cuda`, `vec_env=subproc`, `51.15` training FPS, `102.31` env steps/sec

Current honest gap:

- PPO has a clear learning signal, but no PPO policy has completed a clean or valid full lap yet.

## Active Goal

Train and evaluate PPO until the project has the target RL result.

Only success criterion:

- PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target, with lap time `<=80.0s`.

Required evidence:

- benchmark summaries,
- selected telemetry JSONL,
- replay command/path,
- TensorBoard artifacts,
- updated documentation.

Do not overclaim. Partial progress is useful evidence for the next experiment, but it is not goal completion.

Detailed operating plan: use `LearningPlan.md` for the next goal-mode run. It defines the TensorBoard-first workflow, scratch PPO baseline, full-lap PPO, segment curriculum PPO, checkpoint benchmarking, telemetry review, focused mini experiments, promotion rules, 8+ hour runtime expectation, and strict stop conditions.

Current route update:

- `fine-tuned learning plan.md` is now an inserted course-correction phase inside `LearningPlan.md`.
- Do not launch another serious PPO run until the early fine-tuned milestones are complete:
  - baseline audit,
  - metadata-faithful eval/benchmark,
  - failure-first telemetry observability,
  - racing observation profile,
  - successful-state curriculum,
  - second-chicane skill curriculum,
  - scaffolded/assisted section training with honest unassisted evaluation.
- The original success criterion is unchanged: valid normal-start PPO lap with lap time `<=80.0s`.

## Non-Goals

Do not implement these until the serious PPO path is exhausted or the user explicitly starts that phase:

- evolutionary search
- imitation learning
- offline RL
- image observations
- continuous action RL
- distributed training
- multi-track support
- advanced race-control/anti-cutting system
- world-record optimization

## Compute Policy

Keep the existing strict placement:

- CPU:
  - simulator stepping
  - physics
  - geometry
  - rendering
  - keyboard input
  - telemetry/logging
  - track preprocessing
  - vector env workers
- GPU:
  - PyTorch PPO policy training
  - PyTorch policy inference when requested/available

Important caveat:

- Stable-Baselines3 PPO with `MlpPolicy` and small numeric observations may not saturate the GPU.
- Low GPU utilization is acceptable if `--require-gpu` resolves model training/inference to CUDA.
- Do not move simulator/raycast/render/telemetry code to GPU unless a future measured batched simulator justifies it.

## Immediate Next Milestone: Fine-Tuned Learning Loop

The immediate milestone is no longer another long PPO launch. Follow `fine-tuned learning plan.md` until its infrastructure milestones are complete, then return to the serious PPO training loop below.

Completed in the fine-tuned route:

- Milestone 0 baseline audit:
  - robust best checkpoint: `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`
  - robust best normal-start result: `966.317m`, `20/120`, off-track, no valid lap
  - QC artifact: `artifacts\qc-20260602-221250`
- Milestone 1 metadata-faithful eval/benchmark:
  - PPO eval and benchmark resolve `run_metadata.json` from checkpoint paths or artifact directories by default.
  - Metadata restores action mode/action set/continuous scheme/observation profile/reward settings.
  - `best_vecnormalize.pkl` or `vecnormalize.pkl` is loaded when present.
  - model/environment observation and action spaces are validated before rollout.
- Milestone 2 failure-first observability:
  - QC now accepts telemetry files or folders and writes compact failure tables plus section summaries.
  - Current robust-best PPO failure is `throttle_during_brake_demand` in `rettifilo_chicane` at `521.4m`, `333.2kph`.
  - Section summary shows severe first-chicane overspeed: `rettifilo_chicane` entry `328.8kph`, min `325.0kph`, max speed surplus `232.2kph`, terminal off-track at `966.3m`.
  - Validation artifact: `artifacts\qc-20260602-223327`.
- Milestone 3 racing observation profile:
  - Existing `base`, `brake`, and `guidance` dimensions remain stable at `18`, `21`, and `23`.
  - New `racing` profile is `31` dimensions and adds signed lateral error, previous throttle/brake, brake/guidance features, lookahead target-speed features, and distance to the next braking gate.
  - Monza section/braking-gate definitions now live in `src\f1rl\track_sections.py` for shared simulator/QC use.
  - New profile passes Gymnasium env checker and bounded-observation tests.
- Milestone 4 successful-state curriculum:
  - Added serializable `StateSnapshot` records and `MonzaSim.reset(..., options={"state_snapshot": ...})`.
  - Added `f1rl.state_library` / `f1-state-library` to build state libraries from scripted rollouts or telemetry JSONL files/folders.
  - Added state-library curriculum sampling through `--curriculum segments --curriculum-state-library <path>`.
  - Scripted full-lap library artifact: `artifacts\state-library-scripted-full-m4-20260602\state_library.json`, `59` snapshots, final snapshot `5800.35m`, `valid_lap=True`, `completed_lap=True`.
  - State-library PPO smoke artifact: `artifacts\state-library-curriculum-smoke-20260602-225555`.
- Milestone 5 chicane-specific skill curriculum:
  - Added `--curriculum-preset chicane-skill` with `--curriculum-chicane rettifilo|roggia`.
  - Stages cover approach/brake, turn-in, apex, exit, post-exit, and full-chicane segments.
  - Stages can sample either progress ranges or filtered successful-state library snapshots.
  - Eval summaries now include `curriculum_stage_metrics` for per-stage completion/progress/termination evidence.
  - Chicane smoke artifact: `artifacts\chicane-skill-curriculum-smoke-20260602-230601`.
- Milestone 6 temporary scaffold rewards:
  - Added zero-default scaffold reward components for brake credit, no-throttle penalty, turn-in speed, apex cleanliness, exit alignment, and exit speed.
  - Added optional scaffold scale scheduling during training.
  - Added `--disable-scaffold-rewards` to eval/benchmark for honest scaffold-free scoring.
  - Scaffold smoke artifact: `artifacts\scaffold-reward-smoke-20260602-231335`.
  - Honest-disable benchmark artifact: `artifacts\benchmark-20260602-231438`.
- Milestone 7 forced exploration gates:
  - Added zero-default `AssistConfig` with explicit `--assist-*` flags for assisted section training.
  - Supports overspeed turn-in termination, throttle-during-brake-demand penalty, no-brake penalty, and virtual corridor penalty/termination.
  - Metadata records `training_assists_enabled` and full `assist_config`.
  - Added `--disable-training-assists` to eval/benchmark for honest unassisted scoring.
  - Assisted smoke artifact: `artifacts\forced-exploration-smoke-20260602-232142`.
  - Unassisted benchmark smoke artifact: `artifacts\benchmark-20260602-232229`.
- Milestone 8 segment elite search:
  - Added `f1rl.elite_search` / `f1-elite-search`.
  - Runner starts from a state library, runs short scripted/random/optional PPO attempts, scores exits, writes selected telemetry, and emits an elite state library.
  - Roggia elite artifact: `artifacts\elite-search-roggia-m8-20260602`, `6` attempts, `3` elite states.
  - QC display artifact for elite telemetry: `artifacts\qc-20260602-232815`.
- Milestone 9 continuous-control retry:
  - Hard-assisted continuous run rejected: `artifacts\ppo-continuous-racing-rettifilo-m9-20k-20260602-233058`, final normal-start eval collapsed to `43.6m`.
  - Discrete-expanded comparison improved section delta but honest transfer only reached `431.60m`.
  - Soft continuous run was useful but unpromoted: `artifacts\ppo-continuous-racing-rettifilo-soft-m9-10k-20260602-234003`.
  - Honest scaffold/assist-disabled benchmark: `artifacts\benchmark-20260602-234321`, `951.76m`, `19/120`, collision, no valid lap.
  - QC artifact: `artifacts\qc-20260602-234452`; failure remains throttle during brake demand at `521.1m`.
- Milestone 10 full-lap transfer/scaffold removal:
  - Resume/transfer run from the soft continuous best collapsed after training.
  - Artifact: `artifacts\ppo-continuous-racing-full-transfer-m10-20k-20260602-234628`.
  - Eval sequence: initial resume `951.8m`; final `60.3m`.
  - Preserved best benchmark: `artifacts\benchmark-20260602-235140`, `951.76m`, `19/120`, collision, no valid lap.
  - Decision: no M10 checkpoint promoted; return to the original `LearningPlan.md` loop with robust best still `966.32m`.

Next:

- Resume the original `LearningPlan.md` full-lap goal loop. Strict target remains a valid normal-start lap in `<=80.0s`; current robust best remains `966.32m`, `20/120`.
- New available tool for the resumed loop:
  - `--initialize-from-checkpoint` can transfer a trained PPO policy into a larger observation space, e.g. robust `base` checkpoint -> `racing` observation profile.
  - `--initialize-from-checkpoint` now also expands discrete action heads from `legacy` into richer `racing`/`expanded` action sets, preserving exact action logits and initializing new actions conservatively.
  - `exclusive_throttle_bias` is available for future continuous-control retries to avoid simultaneous throttle/brake during brake-demand zones.
  - `--ppo-stochastic` is available for eval/benchmark diagnostics; current robust-best stochastic benchmark is worse than deterministic evaluation (`129.31m` best across `4` episodes), so promotion remains deterministic honest normal-start performance.
  - Transfer-initialized racing-observation attempts preserved the `966.32m` baseline initially, but the first soft/hard assist schedules did not yet improve normal-start progress.
  - Action-head transfer probe preserved the robust baseline under `observation_profile=racing` and `action_set=racing`: `966.107m`, `20/120`, collision.
- Strategy after the fine-tuned closure audit:
  - avoid more tiny focus-window increments around `966m`;
  - keep the robust checkpoint as the benchmark to beat;
  - use `racing` observations and transfer initialization only when the initial deterministic eval reproduces the robust baseline;
  - keep hard assist termination out of promotion candidates because it repeatedly collapsed behavior;
  - mix normal starts with a minority of Rettifilo state-library starts so section training cannot erase launch/full-lap behavior;
  - promote only if deterministic, metadata-faithful, scaffold/assist-disabled normal-start benchmark clears `966.32m` by a meaningful margin or completes a valid lap.

## Deferred Serious PPO Training

### Step 1: Sanity Validation

Run before any long training:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.hardware --json
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Expected:

- ruff passes
- pyright passes
- pytest passes
- CUDA is visible on the RTX 4060
- scripted baseline completes `lap_complete`

### Step 2: Establish Baselines

Use explicit artifact paths. Do not use `latest` for reported results.

Random/scripted/reference:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random --episodes 20 --max-steps 3600 --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
```

Current PPO baseline checkpoints:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Record:

- completion rate
- valid lap rate
- average/best progress
- checkpoints passed
- reward
- termination reasons
- selected telemetry paths

### Step 3: Overnight Curriculum PPO

Current strategic reset before any longer overnight run:

- Treat normal-start valid full-lap progress as the primary metric.
- Do not promote checkpoints just because they complete easy short segments.
- Use segment eval only as diagnostics and force segment starts during segment eval.
- Prefer bigger strategy changes over local focus-window increments:
  - continuous normalized action experiments,
  - reward normalization,
  - speed-target/heading/lateral shaping,
  - full curriculum with normal-start exposure,
  - benchmarked promotion against the best known normal-start progress.
- Current strict target remains unchanged:
  - valid normal-start lap,
  - lap time `<=80.0s`,
  - telemetry/replay/benchmark evidence required.

Targeted continuous scratch command:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 310 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-mode continuous --continuous-action-scheme throttle_bias --normalize-reward --curriculum segments --curriculum-promotion-resets 500 --curriculum-normal-start-probability 0.20 --reward-lateral-penalty-scale 0.003 --reward-track-limit-penalty-scale 0.003 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 85 --reward-speed-target-max-kph 320 --reward-speed-target-heading-scale 3.0 --reward-speed-target-deadzone-kph 20 --reward-speed-target-penalty-scale 0.0015 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0001 --gamma 0.997 --ent-coef 0.004 --use-sde --sde-sample-freq 16 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-throttlebias-norm-scratch-goal-120k
```

If it fails to exceed the robust `966.32m` benchmark by a meaningful margin, reject it and move to a reward-normalized discrete/full-curriculum continuation from the best scratch-trained checkpoint.

Fast-stage discrete continuation command:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 410 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-mode discrete --action-set legacy --normalize-reward --curriculum segments --curriculum-start-stage-index 3 --curriculum-stage-count 4 --curriculum-promotion-resets 120 --curriculum-normal-start-probability 0.35 --resume-checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --reward-lateral-penalty-scale 0.003 --reward-track-limit-penalty-scale 0.003 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.0015 --reward-speed-target-min-kph 90 --reward-speed-target-max-kph 330 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 20 --reward-speed-target-penalty-scale 0.0015 --n-steps 512 --batch-size 256 --n-epochs 4 --learning-rate 0.00002 --gamma 0.997 --ent-coef 0.002 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-faststage-discrete-norm-best966-resume-goal-120k
```

Reject if normal-start progress collapses below the `966.32m` benchmark for two consecutive evals or if improvements remain tiny and local.

Primary command:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

If stable and promising, scale:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 3000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 100000 --eval-every 100000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-overnight-scratch
```

Required outputs:

- `run_metadata.json`
- `initial_model.zip`
- `final_model.zip`
- `best_model.zip` when available
- checkpoint files
- TensorBoard event file
- `eval/eval_metrics.jsonl`
- `eval/best_eval_summary.json`
- selected telemetry for representative evals

### Step 4: Full-Lap Fine-Tuning

If curriculum improves segments but strict normal-start eval remains weak, fine-tune from the best curriculum checkpoint on normal full-lap starts.

Use this only after selecting the strongest curriculum checkpoint:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-full-lap-finetune
```

If resume-from-checkpoint support is missing or insufficient, implement it explicitly before this step. Do not pretend a fresh full-lap run is fine-tuning.

### Step 5: Final Benchmark Matrix

For each serious run, benchmark:

- random
- scripted
- reference ghost
- PPO initial
- PPO final
- PPO best
- any fine-tuned checkpoint

Use at least `20` episodes for PPO policies.

Report:

- training timesteps
- wall-clock training time
- device and vector env backend
- completion rate
- valid lap rate
- finish crossed rate
- crash/off-track/no-progress rates
- average progress
- best progress
- average checkpoints passed
- average reward
- best lap time if a valid PPO lap exists
- selected replay path
- selected failure path

## Visual And Human Review

Manual visual checks remain useful but should not replace metrics.

Use replay commands for representative policies:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl" --speed 2
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl" --no-timing
```

Check:

- car orientation
- speed plausibility
- rays before crashes
- off-track/collision reason
- whether steering/braking behavior is improving
- whether the replay matches benchmark JSON

## Performance Polish

Low manual FPS is a usability issue, not an RL blocker.

Optimize without deleting useful visuals:

1. Skip `pygame.surfarray.array3d` during human rendering unless an RGB frame is explicitly requested.
2. Cache static track/background surfaces.
3. Cache car sprite rotations.
4. Buffer manual telemetry writes.
5. Add optional smaller-window/render-scale control.
6. Profile render buckets before larger changes.

Do not change physics while fixing FPS.

## Results And Recruiting Outputs

After serious PPO training:

1. Generate learning curves:
   - reward over timesteps
   - progress over timesteps
   - segment completion rate over timesteps
   - crash/off-track/no-progress rates
2. Generate benchmark comparison table.
3. Export or record replay video clips:
   - scripted clean lap
   - reference ghost
   - best PPO success or best PPO failure
4. Update `README.md` with a polished Results section.
5. Update `Documentation.md` with exact commands, artifacts, and metrics.
6. Keep resume claims honest:
   - claim clean PPO laps only if benchmark data proves them
   - otherwise claim simulator/RL pipeline plus measured PPO learning signal

## Definition Of Done For Next Goal

The next goal is complete when:

- PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target, with lap time `<=80.0s`.

Required evidence before reporting completion:

- benchmark summary,
- selected telemetry JSONL,
- replay command/path,
- TensorBoard curves,
- updated documentation,
- passing validations after any code changes made during the run.

The next goal is not complete when:

- serious PPO training merely runs for `1M+` timesteps,
- PPO improves beyond `797.6m`,
- PPO passes more checkpoints,
- PPO survives longer,
- PPO improves reward,
- PPO completes a valid lap slower than `80.0s`,
- PPO shows better segment-to-full-lap transfer,
- artifacts and TensorBoard curves exist but the strict target is missed.

Those are progress signals for the next experiment, not completion.

Blocked:

- mark blocked only if the run cannot continue because of a hard external blocker such as CUDA failure after repair attempts, broken training infrastructure, artifacts impossible to persist, or the user explicitly stops the run.

## Current Resumed-Goal Checkpoint - 2026-06-03

Fine-tuned-plan infrastructure status:

- Milestone 0 through Milestone 10 in `fine-tuned learning plan.md` are complete as infrastructure and controlled experiments.
- The strict `LearningPlan.md` goal is not complete: no PPO policy has completed a valid normal-start Monza lap, and no PPO lap is at or below `80.0s`.
- `LearningPlan.md` remains the goal contract; `fine-tuned learning plan.md` remains the route correction.

Current best honest PPO anchors:

- Robust legacy anchor: `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`.
  - Honest deterministic benchmark: `966.317m`, `20/120`, `off_track`, no finish, no valid lap.
- Racing action-head transfer candidate: `artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip`.
  - Honest deterministic benchmark: `970.775m`, `20/120`, `collision`, no finish, no valid lap.
  - Not promoted as meaningful progress because it remains the same Rettifilo failure and only adds about `4.5m`.

Current trusted QC signature:

- First bad event: `throttle_during_brake_demand`.
- Location: `rettifilo_chicane`, about `520.8m`.
- Speed: about `334kph` against a `115kph` target section.
- Actions before failure: almost all `throttle`, with only a few `brake_right` samples.
- Terminal: collision or off-track around `966-971m`.

Next experiment constraints:

- Preserve the metadata-faithful transfer path and the `racing` observation/action-space support.
- Do not run another long scaffold-heavy schedule without intermediate honest normal-start benchmarks; the last run collapsed after `20000` timesteps.
- Optimize for earlier braking and retention at Rettifilo, not tiny distance gains.
- Promotion requires either a valid lap or a meaningful deterministic normal-start breakthrough beyond the current `970.775m` candidate with QC showing the braking failure changed materially.

## Plateau Diagnosis And Next Strategy - 2026-06-03

Status:

- The user's concern that the project was still around `970m` after hours of work is correct.
- No PPO agent has completed a full valid normal-start lap.
- No PPO agent has produced a lap time.
- The best honest distance remains `970.775m`, `20/120`, collision, with first bad Rettifilo behavior around `520.8m` at about `334kph`.

Why the plateau persisted:

- The chicane-skill curriculum was using progress-only segment completion.
- A policy could start near Rettifilo, blast to the segment target at excessive speed, and receive segment completion.
- That meant segment metrics could look good while the policy was still learning a behavior that fails the real full-lap evaluation.

Completed correction:

- Added speed-gated segment completion through `segment_target_max_speed_kph`.
- Added chicane-skill target max speeds for Rettifilo and Roggia approach, turn-in, apex, exit, post-exit, and full-chicane stages.
- Preserved the target speed gate in state-library chicane training.
- Added tests proving overspeed segment target crossing no longer counts as completion.
- Validation passed:
  - `uv run --no-sync ruff check src/f1rl/curriculum.py src/f1rl/sim.py src/f1rl/train.py tests/test_curriculum.py`
  - `uv run --no-sync pyright src/f1rl`
  - `uv run --no-sync pytest tests/test_curriculum.py -q`
  - `uv run --no-sync pytest -q`

Next experiment:

- Train from the best racing-action transfer candidate with:
  - `observation_profile=racing_v2`;
  - `action_set=racing`;
  - speed-gated `chicane-skill` Rettifilo curriculum;
  - honest normal-start evals every short interval;
  - no promotion unless the QC failure changes materially or the normal-start lap progresses beyond the current plateau.
