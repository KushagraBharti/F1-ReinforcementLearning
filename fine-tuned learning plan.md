# Fine-Tuned Learning Plan

## How To Use This Plan

This file is a course-correction prompt for an agent that is already working on the original F1RL goal. It does not replace `LearningPlan.md`, `Plan.md`, `Prompt.md`, or `AGENTS.md`.

The original goal stays active:

- Build and improve the simplified F1 reinforcement learning project.
- Preserve the active path: `track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> SB3 PPO -> eval/replay`.
- Keep pushing toward a valid normal-start Monza lap, eventually near the Fast-F1 reference target.
- Do not mark the goal complete until the original success criterion is actually met.

This plan changes the strategy, not the destination. The agent should pause the current training loop, step back, audit what has already been built, and then use this file to reorganize the next implementation and fine-tuning milestones.

## Execution Status - 2026-06-03

This course-correction plan has been executed through the infrastructure and controlled-experiment stages, but it is not complete in the behavioral sense that matters:

- Step-Back Protocol: complete.
- Milestone 0 through Milestone 9: complete as implemented, tested, and documented project capabilities.
- Milestone 10: implemented as the honest full-lap transfer/scaffold-removal loop, but its valid normal-start lap acceptance remains open because that acceptance is the original `LearningPlan.md` success criterion.

Current trusted PPO evidence:

- Robust legacy anchor: `966.317m`, `20/120`, no finish, no valid lap.
- Racing action-head transfer candidate: `970.775m`, `20/120`, collision, no finish, no valid lap.
- Current first bad event: `throttle_during_brake_demand` in `rettifilo_chicane` around `520.8m` at about `334kph`.

Post-status correction:

- Follow-up runs showed raw normal-start PPO progress was still flat around `970m`.
- The reason was structural: chicane-skill segment completion was progress-only and could reward overspeed target crossings.
- Segment completion is now speed-gated with `segment_target_max_speed_kph`, and future Rettifilo/Roggia section training must use that corrected signal.

Therefore this file has not served its full learning purpose yet. It has produced infrastructure, but the next agent must now use that infrastructure aggressively until the car actually learns the missing Rettifilo braking behavior.

## Mandatory Transcript Context

The transcript files have been copied into `transcripts/` and must be read before the next training decision:

- `transcripts/01-yosh-trackmania-2023.txt`
- `transcripts/02-yosh-noseboost.txt`
- `transcripts/03-yosh-a01.txt`
- `transcripts/04-yosh-a06.txt`
- `transcripts/05-f1rl-methods-summary.txt`
- `transcripts/README.md`

The required interpretation is practical, not inspirational:

- Yosh-style progress came from changing the training setup when the agent plateaued.
- Add missing observations only when telemetry proves they are missing.
- Use temporary rewards and remove them after the skill appears.
- Force exploration when the current behavior is lazy and locally rewarded.
- Spawn from useful states, not just the start line.
- Segment the problem and preserve elite states.
- Run many cheap attempts and keep the few that teach something.

## What Is Still Not Done

The following items are not done just because code exists:

- Chicane curriculum has not proven actual braking skill.
- Scaffold rewards exist, but previous scaffold-heavy runs either barely improved or collapsed.
- Forced exploration exists, but has not yet produced transferable normal-start behavior.
- Elite search exists, but current evidence is too small and not focused enough on Rettifilo high-speed braking.
- Continuous control was retried, but without first proving section success under speed-gated curriculum.
- Full-lap transfer is unsolved.
- `racing_v2` observation exists, but one short transfer run still collapsed.
- Speed-gated segment completion is implemented and validated, but has not yet driven a meaningful PPO experiment.

Do not report any of these as complete unless the artifact proves the behavior, not just the interface.

## Aggressive Push Directive

The next agent should stop being conservative. The right mode now is fast, targeted, and empirical.

Run mini experiments like Yosh:

1. Change one or two things hard.
2. Run a short experiment.
3. Read eval/QC/telemetry.
4. Keep the idea only if the first bad event changes materially.
5. Save elite states if the car brakes correctly.
6. Reject and move on quickly if it preserves the same `520m throttle_during_brake_demand` failure.

Allowed and encouraged changes:

- stronger speed-target penalties,
- stronger overspeed-throttle penalties,
- lower progress reward in braking zones,
- hard turn-in overspeed gates during training,
- no-throttle/full-brake requirements in Rettifilo brake demand,
- more aggressive speed-gated section targets,
- different normal-start/state-library mixing ratios,
- action-set changes,
- `racing_v2` and future observation additions,
- continuous `exclusive_throttle_bias` after section success,
- elite-state search from successful braking attempts.

Promotion remains strict:

- scaffold/assist-disabled,
- metadata-faithful,
- normal-start,
- deterministic PPO benchmark,
- either a valid lap or a materially different failure signature beyond the current `970.775m` plateau.

## Step-Back Protocol

When this file is introduced into an existing goal-mode thread, follow this protocol before doing more training:

1. Stop treating the next PPO run as the obvious next step.
2. Do not abandon the original goal.
3. Do not restart the repository from scratch.
4. Read the current docs and artifact history.
5. Identify the current best checkpoint and current failure mode.
6. Re-map the active code path from track geometry through eval/replay.
7. Execute the milestones in this file as a fine-tuning layer on top of the original goal.
8. Resume full-lap optimization only after eval truth, observability, and targeted section training are improved.

The point is to make the existing goal agent take a tactical pause. The project has already done enough work that blindly launching another long PPO run is lower value than improving the learning system around the known plateau.

## Relationship To The Original Learning Plan

`LearningPlan.md` remains the main success contract. This file should be read as an inserted break in that plan:

- `LearningPlan.md` defines the long-running goal and final success criterion.
- This file explains why the current approach should pause and be upgraded.
- After the upgrades here, return to the original goal loop with better evaluation, better observability, better curriculum, and better training experiments.

If there is a conflict:

1. Keep the original final goal from `LearningPlan.md`.
2. Keep the simplified architecture from `AGENTS.md`.
3. Use this file for the next implementation order and fine-tuning strategy.
4. Do not claim success from scaffolded or assisted training.

## Course-Correction Goal Prompt

Use the simplified F1 reinforcement learning project to build a much stronger fine-tuning loop for Monza. The goal is not to reintroduce the old complex architecture. The goal is to keep the small explicit path:

`track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> SB3 PPO -> eval/replay`

The project should learn from the attached Yosh/Trackmania-style transcripts: simple progress rewards are a starting point, but high-performing agents came from targeted observation design, curriculum resets, temporary reward scaffolds, forced exploration, reference-state starts, segmented search, and heavy telemetry.

Final target remains the original target from `LearningPlan.md`: a valid normal-start Monza lap, eventually near the Fast-F1 reference target. The immediate target is to break the current PPO plateau around the second-chicane region by teaching braking, turn-in, and exit as explicit section skills before transferring back to full-lap training.

## Non-Negotiable Constraints

- Preserve the active simplified architecture.
- Do not revive Ray/RLlib, image-based agents, imitation learning stacks, campaign systems, swarm systems, or archive-era complexity.
- Treat `archive/legacy-20260424/` as read-only reference unless the user explicitly asks otherwise.
- Use Python 3.11+, uv, Gymnasium, Stable-Baselines3 PPO, PyTorch, NumPy, OpenCV headless, and Pygame.
- Keep rendering and training decoupled.
- Keep simulator, renderer, and telemetry CPU-bound.
- Use CUDA only through PyTorch/SB3 when available.
- Persist telemetry as JSONL plus episode summaries.
- Update `Documentation.md` after every meaningful milestone with commands, results, decisions, and failures.
- Do not leave TODOs or placeholders for core functionality.
- If validation fails, fix it before moving on.

## Current Project Map

### Track Geometry

Main files:

- `src/f1rl/track_build.py`
- `src/f1rl/track_model.py`
- `src/f1rl/geometry.py`
- `src/f1rl/config.py`

Current behavior:

- Monza is built from image contours.
- The builder extracts outer and inner boundaries, creates a drivable mask, generates an approximate centerline, and creates checkpoints.
- The active track has 120 checkpoints, boundary segments, a centerline, checkpoint distances, start pose, finish line, meter scale, and real track length.
- The simulator projects car position onto the centerline with a continuity window to avoid huge progress jumps.

Important implication:

- The track model is good enough for scripted valid laps, so do not start by rebuilding track geometry.
- Geometry may still need better section labels and signed lateral information for learning.

### Car Physics

Main file:

- `src/f1rl/physics.py`

Current behavior:

- A bicycle-style model updates heading, yaw, speed, steering, throttle, and brake.
- It includes steering response, speed-sensitive steering, grip limits, aero grip, traction-circle-style longitudinal/lateral coupling, drag, rolling resistance, braking, engine acceleration, and max speed.
- Physics runs at 60 Hz.

Important implication:

- The physics model is already simple and suitable for PPO.
- Fine-tuning should focus on observations, rewards, curriculum, and evaluation before changing physics constants.

### Shared Simulator

Main file:

- `src/f1rl/sim.py`

Current behavior:

- `MonzaSim` is the center of the active project.
- It owns reset logic, curriculum reset options, centerline projection, checkpoint validity, lap validity, collision/offtrack checks, observations, action mapping, reward computation, termination, and telemetry emission.
- Reset options already support start checkpoint, start progress, start speed, position noise, heading noise, speed noise, segment length, and curriculum stage.
- Valid lap detection requires checkpoint progression and finish crossing.

Current observation profiles:

- `base`: speed, yaw rate, heading error, lateral error magnitude, progress ratio, last action, last steer, ray distances, and lookahead heading errors.
- `brake`: base plus target speed, speed-target error, and brake demand.

Current issue:

- Lateral error is effectively unsigned, so the policy does not know whether it is left or right of the centerline/racing line.
- Current lookahead mostly says "turn is coming" but not enough about "brake now, because the car is too fast for the upcoming section."

Current action modes:

- Discrete legacy action set.
- Discrete expanded action set.
- Continuous two-dimensional action with drive/brake and steering.

Current issue:

- Continuous control was tried before but likely too early, before the observation/curriculum/scaffold problem was solved.

Current rewards:

- Progress reward.
- Finish bonus.
- Collision/offtrack/no-progress penalties.
- Lateral, track-limit, heading, speed-target, overspeed-action, and smoothness components.
- Segment completion bonus for curriculum segments.

Current issue:

- Reward components exist, but they are not yet organized into a deliberate staged teaching system where temporary scaffolds are added, tested, and later removed.

### Manual And Scripted

Main files:

- `src/f1rl/manual.py`
- `src/f1rl/scripted.py`

Current behavior:

- Manual mode uses the same simulator and telemetry.
- Scripted mode uses a simple controller and completes a slow valid lap.

Important implication:

- Scripted driving should be used as a source of successful states and section reset snapshots.
- Scripted valid laps prove the simulator/track/lap-validity stack is viable.

### Telemetry

Main files:

- `src/f1rl/telemetry.py`
- `src/f1rl/qc.py`

Current behavior:

- Per-step JSONL telemetry includes speed, controls, progress, checkpoint/lap validity, rewards, ray distances, errors, collisions, offtrack state, and termination reason.
- Episode summaries include reward totals, sectors, braking zones, racing-line deviations, g-force aggregates, smoothness, ghost gaps, and corner summaries.
- QC can produce reports and simple plots.

Current issue:

- The raw telemetry is strong, but the analysis layer is not yet targeted enough.
- The next phase needs failure reports that answer:
  - Where was the first irreversible mistake?
  - Was the car too fast at turn-in?
  - Did it brake at all?
  - Did it brake too late?
  - Was throttle held during brake demand?
  - What was the action distribution before failure?
  - What section was failed?
  - What was the exit quality when it survived?

### Gymnasium And PPO

Main files:

- `src/f1rl/env.py`
- `src/f1rl/curriculum.py`
- `src/f1rl/train.py`

Current behavior:

- `MonzaEnv` wraps `MonzaSim`.
- PPO training supports curriculum, focus windows, action modes, action sets, observation profiles, reward overrides, checkpointing, VecNormalize, resume, eval callbacks, and telemetry.
- Curriculum stages include short control, longer segments, random checkpoint, flying lap, and normal lap.

Current issue:

- Curriculum currently samples mostly from arbitrary distances/checkpoints.
- The videos show that high-performing agents often improve when they spawn from successful reference states, not merely approximate locations.

### Eval, Benchmark, Replay

Main files:

- `src/f1rl/eval.py`
- `src/f1rl/benchmark.py`
- `src/f1rl/replay.py`
- `src/f1rl/policy_io.py`

Current behavior:

- Replay exists for visual inspection.
- Training callback evaluation is relatively sophisticated.
- Standalone eval/benchmark paths are weaker.

Current issue:

- `eval.py` and `benchmark.py` mostly instantiate default simulator configs.
- That can mis-evaluate models trained with non-default action modes, action sets, observation profiles, reward overrides, or VecNormalize stats.
- Before serious fine-tuning, evaluation must become faithful to the actual run metadata.

## Current Learning Diagnosis

The project is not failing because the simulator is too simple. It is failing because PPO is being asked to discover a high-speed racing skill from a mostly global progress setup.

The current PPO result is roughly:

- Scripted baseline: valid but slow lap.
- PPO: good progress compared to scratch, but no valid lap.
- Best normal-start behavior has reached around the second-chicane region.
- Failure mode is consistent with full-throttle or late-braking behavior.
- Recent brake-zone-focused runs showed some segment improvement but did not solve the normal-start lap.

This is the same pattern from the transcripts:

- A plain progress reward creates local competence.
- The agent discovers an easy but wrong behavior.
- More global training does not fix the wrong behavior.
- The creator watches failures, adds missing observations, forces exploration, uses staged rewards, and trains specific sections.

## Transcript Methods To Steal

### 1. Spawn Everywhere

Video lesson:

- Agents that only start at the beginning overfit early track behavior.
- Yosh-style training used many starts around the map so later sections were not starved.

Project translation:

- Keep full-lap normal starts for honest evaluation.
- Train from many Monza sections.
- Add weighted section starts for the current failure zone.
- Eventually maintain a library of start states for every major section.

### 2. Add Observations After Failures

Video lesson:

- Better results came from adding future turns, orientation, wheel contact, sliding, and other missing state after watching failure.

Project translation:

- Add signed lateral error.
- Add brake-demand and speed-surplus features.
- Add distance to next slow zone or next braking gate.
- Add lookahead curvature/target speed, not only heading errors.
- Add previous throttle and previous brake as separate features.
- Keep ray distances but do not expect rays alone to teach braking.

### 3. Temporary Reward Shaping

Video lesson:

- Rewards were used to teach special skills, then removed once the agent learned the behavior.

Project translation:

- Add chicane-specific training rewards only inside curriculum training.
- Reward correct braking before turn-in.
- Reward reaching apex/exit gates with target speed and heading.
- Reward clean throttle reapplication after the exit.
- Decay or disable these rewards for final full-lap evaluation.

### 4. Force Exploration

Video lesson:

- When the agent chose the lazy route, Yosh changed the environment so lazy behavior failed.

Project translation:

- In training-only chicane curricula, make full throttle past the brake point fail immediately or receive a large penalty.
- Add overspeed gates near chicane turn-in.
- Add section-specific no-progress/overspeed terminations.
- Never enable these assists for final evaluation.

### 5. Reference-State Curriculum

Video lesson:

- Good runs were extended by starting from states that already worked.

Project translation:

- Save exact states from scripted valid laps, manual laps, reference ghost poses, and best PPO attempts.
- Reset from these states directly, not just from approximate progress meters.
- Use successful exits as new starts for the next segment.

### 6. Risk Reward Engineering

Video lesson:

- Low-risk mediocre behavior must not dominate high-risk correct behavior.

Project translation:

- Penalize safe but useless behavior such as crawling, centerline hesitation, and throttle-through-chicane crashes.
- Make clean high-speed section exits much more valuable than merely surviving a few extra meters.
- Score exit quality, not just distance.

### 7. Explicit Exploit Hunting

Video lesson:

- The creator inspected weird behavior and patched the training setup around it.

Project translation:

- Build QC reports that expose action spam, full-throttle overspeed, brake avoidance, wall riding, checkpoint gaming, and low-speed crawling.
- Treat every plateau as a telemetry question, not as a reason to blindly train longer.

### 8. Multi-Stage Objectives

Video lesson:

- Strong runs came from stages: basic speed, special trick, target direction, then completion.

Project translation:

- Stage the Monza task:
  1. basic car control,
  2. braking,
  3. chicane entry,
  4. chicane apex,
  5. chicane exit,
  6. full sector,
  7. full lap,
  8. scaffold removal.

### 9. Segmented Elite Search

Video lesson:

- Keep best states and continue from them.

Project translation:

- Run many short attempts from a section start.
- Keep top exits by progress, speed, heading, track position, and validity.
- Store elite states.
- Train PPO from those states or use them to seed future curricula.

### 10. Bruteforce/Evolutionary Search

Video lesson:

- Some discoveries came from many attempts and perturbations, not from one smooth RL run.

Project translation:

- Start with simple segment action perturbation search, not complex genetic training.
- Use current PPO/scripted/manual/reference states.
- Perturb controls or starts around the failure zone.
- Keep elite transitions.

## Main Engineering Plan

### Milestone 0: Baseline Audit

Purpose:

Establish the exact current best model, current telemetry, and current failure point before changing behavior.

Tasks:

- Read `README.md`, `Prompt.md`, `Plan.md`, `Implement.md`, `LearningPlan.md`, `Documentation.md`, and this file.
- Inspect active files under `src/f1rl/`.
- Identify current best model artifact and current best normal-start result.
- Run tests and current QC if practical.
- Append findings to `Documentation.md`.

Acceptance:

- Clear statement of current best checkpoint.
- Clear statement of current best normal-start distance/checkpoints.
- Clear statement of current failure section.
- No code changes unless required to run existing validation.

Suggested commands:

```powershell
uv sync --active --all-extras --all-packages
uv run pytest
uv run f1-qc
```

### Milestone 1: Make Eval And Benchmark Truthful

Purpose:

Ensure all model comparisons use the same action mode, action set, observation profile, reward overrides, and normalization stats that the model was trained with.

Tasks:

- Make standalone eval load run metadata when given an artifact directory or model path.
- Make benchmark load run metadata for PPO policies.
- Load VecNormalize stats when present.
- Validate discrete legacy, discrete expanded, and continuous model paths.
- Fail loudly if model observation dimension does not match environment observation dimension.

Acceptance:

- A model trained with `observation_profile=brake` evaluates with brake observations.
- A model trained with continuous actions evaluates with continuous actions.
- Benchmark reports the config used for each PPO model.
- Tests cover metadata loading or dimension mismatch behavior.

Why this comes first:

If evaluation lies, every fine-tuning experiment after this is suspect.

### Milestone 2: Build Failure-First Observability

Purpose:

Turn telemetry into direct answers about why PPO fails.

Tasks:

- Add Monza section definitions by progress distance.
- Add per-section summaries:
  - entry speed,
  - minimum speed,
  - exit speed,
  - max speed,
  - brake start progress,
  - throttle reapplication progress,
  - average throttle,
  - average brake,
  - action histogram,
  - max speed surplus,
  - min ray distance,
  - lateral error,
  - heading error,
  - reward totals by component,
  - termination reason.
- Add first-bad-event detection:
  - overspeed at braking zone,
  - throttle during brake demand,
  - no brake before turn-in,
  - offtrack,
  - collision,
  - wrong heading,
  - excessive lateral error,
  - stalled/no-progress.
- Extend QC output with section charts and a compact failure table.

Acceptance:

- Given a PPO telemetry folder, QC says exactly which section failed and why.
- QC can compare scripted, PPO, and reference-style runs section by section.
- The report shows action distribution before failure.

### Milestone 3: Add Racing Observation Profile

Purpose:

Give the policy the state it needs to brake and place the car.

Tasks:

- Add signed lateral error.
- Keep the existing base/brake profiles stable.
- Add a new profile, likely `racing` or `brake_v2`.
- Include:
  - signed lateral error,
  - speed normalized,
  - heading error,
  - yaw rate,
  - progress ratio,
  - previous throttle,
  - previous brake,
  - previous steer,
  - ray distances,
  - lookahead heading errors,
  - lookahead curvature or target speeds,
  - target speed,
  - speed surplus,
  - brake demand,
  - distance to next slow zone/braking gate if available.
- Update env observation dimension handling.
- Add tests for observation shape/range.

Acceptance:

- Existing `base` and `brake` tests still pass.
- New profile passes Gymnasium env checker.
- Observation values remain bounded in `[-1, 1]`.
- Signed lateral error is verified on both sides of the centerline.

### Milestone 4: Add Successful-State Curriculum

Purpose:

Replace arbitrary hard-section starts with starts from states that already make physical/racing sense.

Tasks:

- Define a serializable simulator state snapshot:
  - x,
  - y,
  - heading,
  - speed,
  - yaw rate,
  - steering,
  - progress,
  - checkpoint,
  - lap,
  - lap validity flags where needed.
- Add ability to reset `MonzaSim` from a snapshot.
- Capture snapshots from:
  - scripted valid lap,
  - manual runs,
  - reference ghost/flying reference when possible,
  - best PPO telemetry,
  - segment elite search.
- Store state libraries in artifacts.
- Add curriculum sampler mode for state-library starts.

Acceptance:

- A scripted lap can generate a state library.
- The simulator can reset from a saved state and continue.
- Lap/checkpoint validity remains coherent after reset.
- State-library curriculum works through `MonzaEnv`.

### Milestone 5: Chicane-Specific Skill Curriculum

Purpose:

Teach the exact behavior currently blocking progress.

Tasks:

- Define second-chicane training zones using telemetry-derived progress boundaries.
- Add stages:
  - approach/brake zone,
  - turn-in,
  - apex,
  - exit,
  - post-exit straight,
  - full chicane segment,
  - mixed normal-start/chicane training.
- Start at low/medium speed first.
- Increase speed only after clean completions.
- Use successful-state starts from Milestone 4.
- Track promotion by actual success metrics, not only reset count.

Acceptance:

- Segment eval completes the chicane from multiple starts.
- Normal-start eval improves past the previous second-chicane plateau.
- Telemetry shows brake usage before turn-in and throttle reapplication after exit.

### Milestone 6: Add Temporary Brake/Exit Scaffold Rewards

Purpose:

Teach braking and exit quality, then remove the scaffold.

Tasks:

- Add training-only reward components for chicane curricula:
  - brake before turn-in when speed surplus is high,
  - target speed compliance at turn-in,
  - clean apex passage,
  - exit heading alignment,
  - exit speed within useful range,
  - no throttle during high brake demand.
- Add reward schedules so these components can be reduced over time.
- Record scaffold status in run metadata and telemetry.
- Ensure final full-lap eval can run with scaffold disabled.

Acceptance:

- Reward component totals appear in telemetry.
- Scaffold can be enabled for section training and disabled for final eval.
- Final model is evaluated without training-only rewards.

### Milestone 7: Add Training-Only Forced Exploration

Purpose:

Prevent the agent from repeatedly choosing the lazy full-throttle failure mode.

Tasks:

- Add optional training-only gates for specific sections:
  - overspeed at turn-in terminates or heavily penalizes,
  - throttle during strong brake demand is penalized,
  - no brake before required brake marker is penalized,
  - maybe narrower virtual corridor for targeted sections.
- Keep these behind explicit config flags.
- Record training assists in metadata.
- Never enable them in honest full-lap eval.

Acceptance:

- Full-throttle chicane behavior fails quickly during assisted section training.
- Same model can still be evaluated in normal unassisted environment.
- Metadata makes it impossible to confuse assisted training with honest evaluation.

### Milestone 8: Segment Elite Search

Purpose:

Use many short attempts to discover good section exits and store them.

Tasks:

- Implement a simple segment search runner.
- Start from a known state before the failure section.
- Run many short rollouts using:
  - deterministic PPO,
  - stochastic PPO,
  - scripted controller,
  - action perturbations,
  - start-state perturbations.
- Score exits by:
  - segment completion,
  - validity,
  - exit speed,
  - exit heading,
  - low lateral error,
  - no collision/offtrack,
  - good checkpoint progression.
- Save top K elite states and transition summaries.

Acceptance:

- Produces an elite state library for the second chicane.
- QC can display elite attempts.
- PPO curriculum can sample from elite states.

### Milestone 9: Retry Continuous Control Properly

Purpose:

Continuous controls are likely needed for clean racing behavior, but they should be retried only after the training problem is shaped correctly.

Tasks:

- Use the new racing observation profile.
- Use state-library curriculum.
- Use section scaffold initially.
- Use VecNormalize.
- Consider gSDE.
- Compare against discrete expanded action set.

Acceptance:

- Continuous model is evaluated with correct metadata.
- It improves section metrics, not only raw reward.
- It transfers to normal-start evaluation better than previous continuous attempts.

### Milestone 10: Full-Lap Transfer And Scaffold Removal

Purpose:

Convert section skill into honest lap performance.

Tasks:

- Mix normal starts with state-library starts.
- Gradually reduce chicane-specific scaffold.
- Disable forced exploration gates.
- Evaluate normal-start full lap after every meaningful training interval.
- Keep best model by valid full-lap performance first, then normal-start progress, then section metrics.

Acceptance:

- PPO passes the previous plateau.
- PPO completes a valid normal-start lap.
- Scaffold-free eval is clearly reported.
- Artifacts include checkpoint, metadata, telemetry, summary, QC report, and replay instructions.

## Observability Requirements

Every serious run should answer these questions:

- What model/config was used?
- What observation profile was used?
- What action mode/action set was used?
- Was VecNormalize used?
- Were training assists enabled?
- Were scaffold rewards enabled?
- What was the best normal-start distance?
- What was the best segment distance?
- What section failed?
- What was the first bad event?
- Did the agent brake before the chicane?
- Did it hold throttle during brake demand?
- What was speed at turn-in?
- What was min speed through the section?
- What was exit speed?
- What reward components dominated?
- What action distribution occurred before failure?
- Did the model exploit the reward?

Minimum artifact set for meaningful runs:

- `run_metadata.json`
- `checkpoints/`
- `best_model.zip`
- `final_model.zip`
- `vecnormalize.pkl` when used
- `eval/eval_metrics.jsonl`
- selected eval telemetry folders
- `episode_summary.json`
- `steps.jsonl`
- QC report
- replay command

## Suggested First Implementation Loop

Do this before more training:

1. Make eval and benchmark profile-aware.
2. Add a telemetry failure report for current best PPO artifacts.
3. Use that report to lock the exact second-chicane failure boundaries.
4. Add signed lateral error and the new racing observation profile.
5. Add tests for observation dimensions and signed lateral behavior.
6. Add state snapshot reset support.
7. Generate a scripted state library.
8. Train a short chicane-only PPO run.
9. Evaluate normal-start transfer.
10. Document the result honestly.

## Validation Commands

Use these as practical anchors, adjusting flags as the code evolves:

```powershell
uv sync --active --all-extras --all-packages
uv run pytest
uv run f1-build-track
uv run f1-scripted --headless
uv run f1-train --help
uv run f1-eval --help
uv run f1-benchmark --help
uv run f1-qc
```

For every new feature:

- Add or update focused tests.
- Run the focused test first.
- Run the broader test suite after related changes settle.
- Update `Documentation.md` with the exact command and result.

## Definition Of Success

Short-term success:

- Evaluation is faithful to model metadata.
- QC identifies the exact failure cause.
- The agent brakes before the second chicane in section eval.
- The agent exits the second chicane cleanly in section eval.

Medium-term success:

- PPO normal-start progress passes the current plateau.
- PPO reaches later Monza sections consistently.
- Training artifacts clearly distinguish scaffolded training from honest eval.

Final success:

- PPO completes a valid normal-start Monza lap without training assists.
- The run is reproducible from documented commands.
- Telemetry and replay clearly show why the model works.

## Handoff Prompt For The Existing Goal Agent

Paste this into the Codex thread that has already been working on the original F1RL goal:

```text
Pause the current loop and take a step back.

You are still pursuing the original F1-ReinforcementLearning goal. Do not abandon it, do not mark it complete, and do not replace it with a new project. The original success criterion still stands: improve the simplified F1RL stack until PPO can complete a valid normal-start Monza lap, eventually near the Fast-F1 reference target.

This is a course correction, not a reset.

Read these files again in order:

1. AGENTS.md
2. Prompt.md
3. Plan.md
4. Implement.md
5. LearningPlan.md
6. Documentation.md
7. fine-tuned learning plan.md

Treat LearningPlan.md as the original goal contract. Treat fine-tuned learning plan.md as an inserted break in the plan: it tells you to stop blindly launching more PPO runs, audit the current state, improve observability/evaluation/curriculum, and then return to the original lap-completion goal with a stronger fine-tuning loop.

After reading, inspect the active src/f1rl/ stack and map the actual code path:

track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> SB3 PPO -> eval/replay

Then execute fine-tuned learning plan.md starting from its Step-Back Protocol and Milestone 0.

Do not revive the archived complex system. Do not use Ray/RLlib, image agents, imitation stacks, campaign systems, swarm systems, or archive-era complexity. Keep the simplified project architecture.

Before doing another serious training run, complete the early correction work:

1. establish the current best baseline and failure mode,
2. make eval/benchmark faithful to trained model metadata,
3. add section/failure telemetry reports,
4. add signed lateral error plus a racing observation profile,
5. add state snapshot save/load/reset,
6. generate scripted/reference state libraries,
7. build the second-chicane skill curriculum,
8. add scaffolded brake/exit rewards only for training,
9. evaluate final progress only in the unassisted normal-start environment.

Work autonomously in milestone loops:

inspect -> plan -> implement -> validate -> fix failures -> update Documentation.md -> repeat

Keep diffs scoped. Add tests where behavior changes. If validation fails, fix it before moving on. Update Documentation.md with exact commands, results, artifact paths, decisions, and failures.

Use the transcript-derived methods as the north star: add missing observations based on failure telemetry, train hard sections directly, use temporary scaffolds only for teaching, force exploration only in training configs, save successful states, run segmented elite search, and always return to honest full-lap evaluation.

The goal did not change. The route to the goal is now the fine-tuned plan.
```
