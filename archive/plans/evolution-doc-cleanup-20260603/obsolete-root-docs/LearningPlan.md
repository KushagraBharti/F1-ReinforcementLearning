# F1RL Learning Plan

## Purpose

This file is the operating plan for the next goal-mode run. The objective is not more simulator scaffolding. The objective is to make PPO learn Monza from scratch and keep iterating for a long training session until the agent completes a valid normal-start Monza lap near the Fast-F1 reference ghost target.

The workflow is:

1. Keep TensorBoard open in the browser.
2. Train a from-scratch PPO agent.
3. Benchmark initial, final, and best checkpoints.
4. Inspect summaries, JSONL telemetry, TensorBoard curves, and replay behavior.
5. Identify what improved, what failed, and where the car failed.
6. Run targeted mini experiments for the failure mode.
7. Promote only checkpoints that improve benchmark evidence.
8. Repeat with full-lap and curriculum PPO until the strict completion target is met or a hard external blocker prevents continued work.

## Fine-Tuning Course Correction

If the goal-mode agent has already completed the initial simulator/RL setup, run baseline training, and reached a PPO plateau, treat `fine-tuned learning plan.md` as an inserted break in this plan.

That file does not replace this goal. The single success criterion below still stands. The purpose of the fine-tuned plan is to make the agent pause, audit the current best artifacts, improve eval truth and observability, add targeted section curricula, and then return to this original full-lap objective with a stronger learning loop.

When entering that course correction:

1. Do not mark this goal complete.
2. Do not abandon the original `<=80.0s` valid normal-start lap target.
3. Do not restart the repo from scratch.
4. Read `fine-tuned learning plan.md`.
5. Execute its Step-Back Protocol and milestones.
6. Resume this learning plan only after the fine-tuning infrastructure improves evaluation, telemetry, curriculum, and section training.

Status as of 2026-06-03:

- The fine-tuning course correction has been executed through its implementation, validation, and controlled-experiment milestones.
- The original success criterion is still unmet.
- Current best honest PPO evidence is `970.775m`, `20/120`, `collision`, no finish crossing, no valid lap, from `artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip`.
- The robust legacy anchor remains `966.317m`, `20/120`, no valid lap, from `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip`.
- Current trusted failure signature is late/no braking in `rettifilo_chicane`: `throttle_during_brake_demand` around `520.8m` at about `334kph`, then collision/off-track around `966-971m`.
- Continue this learning plan with the upgraded metadata-faithful eval, QC, racing observations, state-library curriculum, scaffold controls, training assists, and action-head transfer. Do not claim completion until PPO produces a valid normal-start lap with `lap_time <= 80.0s`.

Important correction:

- Do not treat `fine-tuned learning plan.md` as behavior-complete.
- It is only infrastructure-complete.
- The learning system still has not learned the required braking behavior.
- The next goal agent must read every file under `transcripts/` before choosing the next experiment.
- The next phase should follow the Yosh-style method from those transcripts: many targeted attempts, aggressive environment/reward changes, quick rejection of failed ideas, saved elite states, and honest scaffold-free promotion.

## Aggressive Mini-Experiment Mode

The project is past the point where another cautious single PPO run is a good use of time. The next agent should stop trying to design one perfect experiment. It should run a sequence of small, fast experiments that attack the Rettifilo failure from multiple angles.

Operating rule:

- Prefer five focused experiments where four fail and one teaches something over one over-planned experiment that preserves the same `970m` failure.
- Keep experiments short enough to reject quickly.
- Change reward scales, assist settings, segment gates, curriculum starts, action sets, observation profiles, and elite-state selection aggressively.
- Keep exact commands, artifacts, and rejection reasons in `Documentation.md`.
- Never promote scaffolded/assisted training results directly. Promotion still requires honest metadata-faithful normal-start eval with scaffold rewards and training assists disabled.

Immediate experiment backlog:

1. Speed-gated Rettifilo micro-curriculum:
   - use the new `segment_target_max_speed_kph` gates;
   - require low-enough speed at approach, turn-in, apex, and exit;
   - do not count target distance alone as segment success.
2. Hard forced-exploration run:
   - terminate or heavily penalize full throttle in Rettifilo brake demand;
   - make the old behavior fail immediately in training;
   - benchmark final candidates without assists.
3. Reward scale sweep:
   - test much stronger section-speed and overspeed-action penalties;
   - test lower progress reward inside braking zones;
   - test larger collision/overspeed-at-turn-in penalties;
   - reject if normal-start launch collapses.
4. Policy diagnostic at `520m`:
   - inspect action probabilities/logits at the first bad event;
   - verify whether the policy sees `racing_v2` brake-zone features but still strongly chooses throttle.
5. Rettifilo elite-state search:
   - generate elite states from successful braking attempts, not merely scripted/Roggia samples;
   - keep states only if they brake before turn-in and meet speed gates.
6. Section-first transfer:
   - solve Rettifilo from state-library starts before returning to normal starts;
   - then mix normal starts back in gradually.
7. Continuous-control retry only after section success:
   - use `exclusive_throttle_bias`;
   - do not allow simultaneous throttle/brake;
   - compare against the racing discrete action set.

## Rettifilo Breakthrough Mode

The current blocker is not subtle:

- At about `520.8m`, the policy is already in Rettifilo brake demand.
- It is doing about `334kph`.
- It should be preparing for roughly `115kph`.
- It still chooses full throttle.
- The car later collides around `966-971m`.

This must be treated as a failed training objective, not a small tuning issue. The training loop still makes "go far fast and crash" too attractive compared with "brake early and give up short-term progress." The next agent must change that objective aggressively.

Required working style for this phase:

1. Find the shortcoming from telemetry.
2. Change the metric, reward, assist, curriculum, observation, action space, or reset distribution around that shortcoming.
3. Run a short targeted experiment.
4. Read QC, eval rows, selected telemetry, and first-bad-event output.
5. Keep the change only if it materially changes behavior.
6. Reject quickly if it keeps the same `520m` full-throttle failure.
7. Immediately launch the next experiment.

Do not spend hours preserving a policy that is still choosing throttle at the braking gate. Do not celebrate a `+4m` improvement if the first bad event is identical. The milestone is not `970m`; the milestone is braking before Rettifilo turn-in and exiting the section alive.

Aggressive changes are allowed and expected:

- Make full throttle in the Rettifilo brake zone strongly negative.
- Temporarily terminate overspeed turn-in in section training.
- Reduce or zero progress reward while inside active brake demand.
- Increase speed-target and overspeed-action penalties by large factors for section experiments.
- Increase collision penalty during high-speed braking-zone failures.
- Require speed-gated segment completion at every Rettifilo gate.
- Reward braking only when speed surplus is real and the car is before turn-in.
- Penalize no-brake and throttle-through-brake-demand hard enough that the policy cannot ignore them.
- Run state-library starts from multiple Rettifilo approach states.
- Run stochastic/evolutionary/elite short attempts to discover any clean brake/exit state.
- Save elite states only when they meet speed, heading, lateral, validity, and no-collision criteria.

Guardrails:

- Keep final promotion honest: metadata-faithful normal start, scaffold rewards disabled, training assists disabled.
- Keep the simulator architecture simple.
- Do not revive archived complexity.
- Do not overfit to fake lap success from assisted training.
- Do not break scripted valid-lap sanity.
- Do not hide failed experiments; document them and move on.

Minimum experiment cadence:

- A short Rettifilo experiment should have an explicit hypothesis and a rejection condition before launch.
- If the first trained eval preserves `throttle_during_brake_demand` at about `520m`, stop or shorten the run.
- If the run collapses before Rettifilo, reject or repair immediately.
- If the run creates earlier braking, preserve telemetry and make an elite state library from the successful attempts.
- After every 3-5 mini experiments, summarize what changed and choose the next axis.

The next agent should behave like an experimentalist, not a caretaker. It should push the learning system until the first bad event changes.

## Success Criterion

There is exactly one success criterion:

- PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target, with lap time `<=80.0s`.

Required evidence for claiming that success:

- benchmark summary showing valid normal-start lap completion and `lap_time <= 80.0s`,
- selected telemetry JSONL proving the lap path, timing, speed profile, termination state, checkpoint progression, and lap validity,
- replay command/path for visual verification,
- TensorBoard curves and run metadata for training context.

The evidence requirements do not create alternate success criteria. They only prove the single success criterion above.

Progress milestones that are useful but do not count as completion:

- PPO improves materially over the current best checkpoint.
- Current best checkpoint is approximately `797.6m`, `16` checkpoints, collision termination.
- A new checkpoint should show at least one of:
  - valid full lap completion,
  - clean lap completion,
  - progress above `3000m`,
  - progress above `5000m`,
  - more than `60` checkpoints,
  - more than `100` checkpoints,
  - large crash-rate reduction,
  - clear TensorBoard reward improvement,
  - visibly better cornering/braking behavior in replay.
- The run must include JSON summaries, selected telemetry, TensorBoard curves, and replayable JSONL evidence.

These milestones guide the next experiment, but they are not completion criteria. Do not mark the goal complete just because training ran, reward improved, progress improved, checkpoints improved, or a slow valid lap happened. Completion requires the single success criterion above.

## Runtime Requirement

This is intended to be a long goal-mode run. The `8` hour number is a persistence expectation, not a success criterion.

- Expect to run for hours and hours, with `8+` hours as the rough planning reference unless the strict success criterion is reached earlier.
- In the current Rettifilo plateau phase, prefer many short targeted PPO/search experiments before any long run.
- Launch longer PPO only after a mini experiment changes the failure signature or produces a clear section skill.
- A single failed run is not a stopping condition.
- Partial improvements are not a stopping condition.
- A plateau is not a stopping condition until multiple follow-up experiments have tested the likely failure mode.
- If a run fails early, diagnose it, fix the immediate issue if appropriate, and launch the next experiment.
- If the best checkpoint improves but still misses the target, immediately plan and run the next full or focused experiment.
- Do not treat `8+` hours of runtime as completion. If the target is not met after 8 hours, report the best evidence, diagnose the remaining gap, and continue or ask only if the next step requires user approval or a hard blocker exists.

## Non-Negotiables

- Start from a random/scratch PPO policy for the first serious baseline. Do not initialize from the ghost reference, scripted controller, imitation data, or an existing trained model unless explicitly entering a fine-tuning stage after the scratch baseline has been benchmarked.
- Use the Fast-F1 reference ghost only as a benchmark, visualization reference, and target comparison. It is not a policy and must not be treated as learned behavior.
- Keep simulator stepping, physics, rendering, geometry, telemetry, and vector env workers on CPU.
- Require CUDA for PPO model training and inference unless explicitly running a CPU comparison experiment.
- If `--require-gpu` fails, stop, diagnose, fix CUDA/PyTorch/device setup, and rerun. Do not silently continue serious training on CPU.
- Keep artifacts tracked. Do not re-add `artifacts/` to `.gitignore`.
- Record every major command, run directory, benchmark result, decision, and failure mode in `Documentation.md`.
- Do not overfit docs to hope. Report the actual numbers.
- Do not claim a PPO clean lap or valid lap unless benchmark/eval artifacts prove it.

## Baseline Facts To Beat

Current known metrics:

- Observation dimension: `18`
- Discrete actions: `9`
- Sensor rays: `7`
- Checkpoints: `120`
- Boundary segments: `1800`
- Simulator rate: `60 Hz`
- Fast-F1 reference ghost: `79.662s`, `259.9 kph` average, `348.0 kph` max
- Scripted baseline: valid lap, `214.47s`, `96.0 kph` average, `107.9 kph` max
- PPO scratch initial: `0.0m`, no-progress termination
- PPO final checkpoint: `431.4m`, `8` checkpoints, off-track termination
- PPO best checkpoint: `797.6m`, `16` checkpoints, collision termination
- CUDA smoke: `2048` timesteps, `device=cuda`, `vec_env=subproc`, about `51.15` training FPS and `102.31` env steps/sec

The next serious result should beat `797.6m` and `16` checkpoints at minimum.

## Required Live Setup

### 1. Validate Environment

Run before any serious training:

```powershell
uv run --no-sync python -m f1rl.hardware --json
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Acceptance:

- Hardware output says CUDA is available.
- Device is the RTX 4060 Laptop GPU.
- Ruff passes.
- Pyright passes.
- Pytest passes.

If any validation fails, fix before training.

### 2. Start TensorBoard And Keep It Open

Start TensorBoard:

```powershell
uv run --no-sync tensorboard --logdir artifacts --host 127.0.0.1 --port 6006
```

Open:

```text
http://127.0.0.1:6006/
```

Use the browser to keep TensorBoard visible during training. If port `6006` is busy, use `6007` and record the port in `Documentation.md`.

Monitor at least:

- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `time/fps`
- `train/entropy_loss`
- `train/policy_gradient_loss`
- `train/value_loss`
- `train/approx_kl`
- `train/clip_fraction`

Interpretation:

- Rising episode reward is good.
- Rising episode length may mean better survival, but can also mean slow/no-progress behavior. Cross-check telemetry.
- Very low entropy too early can mean premature policy collapse.
- Very high KL or unstable value loss can mean learning rate or batch settings are too aggressive.

## Experiment Loop

Every experiment must follow this loop:

1. Name the hypothesis.
2. Run training or benchmark.
3. Record the artifact directory.
4. Read `run_metadata.json`, `summary.json`, `summary.csv`, `per_episode.jsonl`, and selected `steps.jsonl`.
5. Replay at least one representative selected telemetry file when behavior is unclear.
6. Classify failure mode.
7. Decide the next experiment.
8. Update `Documentation.md`.

Do not run endless training without reading the artifacts.

## Stage 0 - Scratch Baseline Benchmark

Goal: prove the starting PPO model knows nothing, so later improvements are real.

Run a tiny scratch train only to create initial/final model artifacts:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 2048 --seed 0 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name scratch-control-smoke
```

Benchmark the initial scratch model:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "PATH_TO_RUN\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected --telemetry-every 5
```

Expected:

- It should perform badly.
- It may no-progress, drive straight, crash, or do very little.
- This is the control condition.

Do not tune from this smoke run. It is only a sanity baseline.

## Stage 1 - Serious Full-Lap PPO From Scratch

Goal: test whether full-lap PPO can learn enough from normal starts without curriculum.

Run:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --seed 10 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-full-scratch-1m
```

Benchmark:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "PATH_TO_RUN\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "PATH_TO_RUN\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "PATH_TO_RUN\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected --telemetry-every 5
```

Read:

- `PATH_TO_RUN\run_metadata.json`
- latest benchmark `summary.json`
- latest benchmark `per_episode.jsonl`
- selected telemetry for first, best, median, and worst episode if available

Classify:

- no movement or no-progress,
- accelerates but does not steer,
- steers but overcorrects,
- fails first chicane,
- fails Lesmo,
- fails Ascari,
- fails Parabolica,
- survives but too slow,
- checkpoint progress improves but line is bad,
- reward increases but behavior is invalid.

Promote:

- If `best_model.zip` improves progress/checkpoints/reward over current known best, mark it as the new candidate.
- If full-lap PPO stalls under `1000m`, move to curriculum Stage 2.

## Stage 2 - Segment Curriculum PPO From Scratch

Goal: make the agent experience later parts of the lap and learn local driving skills that full-lap PPO may never reach early.

Run:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --seed 20 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-segments-scratch-1m
```

Benchmark the initial, final, and best models as in Stage 1.

Evaluation:

- Curriculum may improve local control but not transfer to normal full-lap starts.
- Always benchmark from the normal start to measure transfer.
- Also inspect segment eval telemetry to understand which section improved.

Promote:

- Promote if normal-start benchmark improves over Stage 1.
- If curriculum learns later sections but normal-start transfer is weak, run Stage 3 fine-tuning.

## Stage 3 - Full-Lap Fine-Tune From Best Curriculum Checkpoint

Goal: take the best segment-trained policy and adapt it to normal full-lap starts.

This stage is allowed to start from a previously trained PPO checkpoint because Stage 0 and Stage 2 already proved the from-scratch baseline.

Current CLI may need checkpoint resume support. If resume support is missing:

1. Implement the smallest safe resume option in `src/f1rl/train.py`.
2. Add a smoke test that loads a checkpoint and continues training.
3. Verify with CUDA.
4. Document the command.

Desired command shape after resume support exists:

```powershell
uv run --no-sync python -m f1rl.train --resume-checkpoint "PATH_TO_SEGMENT_RUN\best_model.zip" --timesteps 1000000 --seed 30 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum none --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --telemetry-every 10 --run-name ppo-full-finetune-from-segments-1m
```

If resume support is not implemented yet, do not fake this stage. Implement and validate it first.

## Stage 4 - Focused Mini Experiments

Run these only after reading telemetry from Stages 1 and 2.

Each focused experiment should be shorter, usually `100000` to `300000` timesteps, and should test one idea at a time.

### First Chicane Failure

Symptoms:

- high speed into first chicane,
- late braking,
- boundary collision,
- misses checkpoint around the chicane.

Actions:

- Use `--curriculum segments`.
- Inspect ray distances and heading error near failure.
- Try shorter max steps for faster iteration if failures happen early.
- Consider reward or termination tuning only if telemetry proves the signal is misleading.

Command:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 250000 --seed 101 --n-envs 8 --max-steps 1800 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 25000 --eval-every 25000 --eval-episodes 3 --telemetry selected --telemetry-every 5 --run-name focus-first-chicane
```

### Drives Straight Or Under-Steers

Symptoms:

- throttle action dominates,
- steering action rare,
- heading error grows,
- ray fan shows wall closing but action does not correct.

Actions:

- Inspect action distribution in telemetry.
- Check entropy curve in TensorBoard.
- If entropy collapses, tune PPO entropy or learning settings in code with a targeted test.
- Do not change physics unless behavior is impossible for scripted/manual/reference comparison.

### Oscillation Or Over-Steer

Symptoms:

- repeated left/right switching,
- high steering deltas,
- high lateral g spikes,
- line crosses track width unnecessarily.

Actions:

- Inspect control delta telemetry.
- Consider mild smoothness reward only if needed and only after documenting current behavior.
- Avoid making smoothness so strong that the car refuses to turn.

### Slow Safe Driving

Symptoms:

- survives longer,
- progresses slowly,
- low max speed,
- lap time nowhere near target.

Actions:

- Compare speed profile against reference ghost.
- Confirm reward does not overpay survival.
- Consider small progress scaling or no-progress tightening.
- Do not use raw speed as the main reward.

### Good Segment Skill But Poor Full-Lap Transfer

Symptoms:

- curriculum eval looks better,
- normal-start benchmark still crashes early.

Actions:

- Implement/resume Stage 3 full-lap fine-tuning from segment best.
- Use selected telemetry to compare segment behavior vs normal-start behavior.

## Stage 5 - Benchmark Matrix

Every candidate checkpoint must be compared against the same baselines:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random --episodes 20 --max-steps 3600 --telemetry selected --telemetry-every 10
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 5 --max-steps 18000 --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "PATH_TO_CANDIDATE\best_model.zip" --episodes 50 --max-steps 3600 --device auto --telemetry selected --telemetry-every 5
```

Capture:

- completion rate,
- valid lap rate,
- finish crossed rate,
- average progress,
- max progress,
- average checkpoints passed,
- max checkpoints passed,
- average reward,
- termination reasons,
- crash rate,
- no-progress rate,
- off-track rate,
- lap time if completed,
- average and max speed,
- reward components,
- ghost gap if available.

## Stage 6 - Telemetry Review

For each important benchmark, inspect:

- `summary.json`
- `summary.csv`
- `per_episode.jsonl`
- `selected_telemetry\ppo-episode-000-steps.jsonl`
- any selected best or longest episode telemetry

Use Python snippets or small one-off scripts as needed to compute:

- maximum progress,
- max checkpoint,
- first termination step,
- termination reason,
- speed at termination,
- distance to boundary before crash,
- action distribution,
- steering delta distribution,
- braking/throttle distribution,
- largest heading error,
- largest lateral error,
- minimum ray distance before crash,
- reward component totals,
- first section where progress stops improving.

Replay command:

```powershell
uv run --no-sync python -m f1rl.replay "PATH_TO_BENCHMARK\selected_telemetry\ppo-episode-000-steps.jsonl"
```

Use replay only after reading the numbers. Visuals explain behavior, but telemetry decides what actually happened.

## Stage 7 - QC And Dashboard

After each serious candidate:

```powershell
uv run --no-sync python -m f1rl.qc --telemetry "PATH_TO_SELECTED_STEPS.jsonl" --run-scripted --scripted-steps 18000
```

Open:

- `telemetry_dashboard.html`
- `qc_report.md`
- `manual_qc_checklist.md`

Use the QC dashboard to verify the behavior described by the benchmark.

## Stage 8 - Result Packaging

Once a candidate is meaningfully better:

1. Save the run directory path.
2. Save the benchmark directory path.
3. Save the best selected telemetry path.
4. Generate or preserve replay evidence.
5. Capture TensorBoard screenshot or curve summary.
6. Update `README.md` Results.
7. Update `Documentation.md`.
8. Keep the claims honest.

If video export tooling is missing and needed for recruiter output, implement it after the learning loop proves a useful policy.

## Decision Rules

Continue training if:

- reward is rising,
- progress is rising,
- checkpoints are rising,
- crash point moves later in lap,
- replay behavior improves,
- entropy remains healthy,
- eval metrics are not collapsing.

Change experiment if:

- progress plateaus for multiple eval intervals,
- the same corner fails repeatedly,
- policy collapses into no movement,
- action distribution becomes degenerate,
- reward improves but behavior worsens,
- curriculum improves segments but normal-start transfer stays bad.

Fix code only if:

- telemetry proves a simulator/accounting bug,
- benchmark aggregation is wrong,
- resume training support is required,
- GPU requirement is not enforced,
- logging lacks a metric needed to diagnose learning,
- a small PPO config exposure is needed for controlled experiments.

Do not change code just because PPO is slow to learn. First prove the failure mode from telemetry.

## Stop Conditions

Stop and report success if:

- PPO completes a valid normal-start lap in `<=80.0s`, with benchmark summary, telemetry, replay, and TensorBoard evidence.

Do not stop for these non-completion outcomes:

- PPO beats the previous `797.6m`/`16` checkpoint baseline.
- PPO completes more sectors or passes more checkpoints.
- PPO completes a valid lap slower than `80.0s`.
- PPO produces better TensorBoard curves.
- PPO shows a cleaner replay but still misses the target.
- PPO improves for one run and then plateaus.

For all non-completion outcomes, document the result, preserve artifacts, diagnose the next failure mode, and continue with another full, curriculum, fine-tuning, or focused experiment.

Stop and ask the user before continuing if:

- CUDA cannot be restored after diagnosis,
- machine stability, thermals, storage, or OS-level process failures prevent the 8+ hour run from continuing,
- the best next step requires a larger feature addition such as imitation learning, evolutionary search, or major reward redesign,
- the repo cannot commit/persist artifacts due to external storage problems.

Do not stop only because one training run failed. A failed run is expected evidence for the next experiment. Do not report goal completion unless the `<=80.0s` valid normal-start lap target is met. Running for 8+ hours is a persistence requirement, not a completion criterion.

## Current Plateau Note - 2026-06-03

The strict goal is still open:

- no PPO valid normal-start full lap;
- no PPO lap time;
- best honest normal-start candidate remains about `970.775m`, `20/120`, collision.

The plateau is now understood as more than insufficient runtime:

- the previous chicane-skill curriculum could count overspeed progress-only segment completions;
- this could reinforce reaching Rettifilo targets too fast instead of braking correctly;
- segment completion is now speed-gated, so future chicane-skill success must include being slow enough at the target.

Continue the original goal loop from the corrected curriculum, not by repeating long runs against the old progress-only segment signal.

## Concise Goal Prompt

Use this for Codex goal mode:

```text
Use `LearningPlan.md` as the operating plan. Start TensorBoard in the browser and keep it open, then iteratively train from a fully scratch/random PPO policy using full-lap and segment-curriculum PPO on CUDA for at least 8 hours unless the strict target is reached earlier. For each run, benchmark initial/final/best checkpoints, inspect `summary.json`, `per_episode.jsonl`, selected `steps.jsonl`, TensorBoard curves, and replay behavior, then diagnose what improved or failed. Run focused mini experiments around repeated failure sections, promote only checkpoints with measured improvement, update `Documentation.md`, and continue until PPO completes a valid normal-start Monza lap near the Fast-F1 ghost target (`<=80.0s`). Do not silently train on CPU, do not use ghost/scripted/imitation initialization for the scratch baseline, do not overclaim results, do not count partial progress as completion, do not treat 8+ hours of runtime as completion by itself, and do not stop after a single failed experiment.
```
