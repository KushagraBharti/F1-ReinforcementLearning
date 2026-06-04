# Active Goal Prompt

## LOCK THE FUCK IN

Read `LearningPlan.md`, `fine-tuned learning plan.md`, and every file under `transcripts/` before launching more training. Use those files as the operating plan, not background reading. The transcripts are the playbook: isolate hard skills, instrument everything, make bad behavior impossible or unprofitable in focused setups, add missing observations, use temporary scaffolds, force exploration, save elite states, transfer learned behavior, and keep iterating until the whole task is solved.

The goal is unchanged and strict:

- Train a scratch/random SB3 PPO agent on CUDA.
- Complete a valid normal-start Monza lap.
- Reach near the Fast-F1 reference target, `<=80.0s`.
- Final success must be a real PPO policy, not scripted, ghost, imitation, assisted, or metadata-faked.
- Do not count partial distance, one good segment, assisted completion, infrastructure work, TensorBoard curves, or runtime as completion.

Scratch does not mean blind full-lap-only training. Scratch means the PPO weights start random and are not initialized from ghost/scripted/imitation policies. Curriculum training from random weights is allowed and required. If the agent has not learned the skills needed for a full lap, stop trying to rawdog the full lap and teach the missing skills directly.

## The Standard

Do not drift. Do not babysit one mediocre run. Do not keep repeating the same failure and calling it persistence. Really lock in.

Your job is to build a learning system that can keep finding the next blocker, isolate it, solve it, transfer the solution, and repeat until the car finishes a valid fast lap. Every repeated failure is a diagnosis target. Every artifact is evidence. Every experiment must teach you something or get rejected.

The `8+ hours` expectation means many aggressive measured curriculum loops, not one passive training job. Long runs are only justified after short experiments prove that the setup changes behavior, transfers a skill, or creates a checkpoint worth scaling.

## Non-Negotiable Loop

For every blocker, run this loop hard:

1. Find the current limiting failure from telemetry, section analysis, first-bad-event output, benchmark tables, TensorBoard, and replay.
2. State the hypothesis clearly before changing anything.
3. Isolate the failure in the smallest useful curriculum setup.
4. Change one or two relevant pieces aggressively.
5. Run a focused mini experiment first.
6. Inspect artifacts immediately.
7. Keep and scale the change only if behavior moves in the intended direction.
8. Reject quickly if the same failure signature remains.
9. Transfer proven skills into wider curriculum and full-lap training.
10. Update `Documentation.md` with hypothesis, command, artifacts, result, diagnosis, decision, and next action.
11. Repeat until final success is real.

Do not skip the inspection step. Do not launch another run just because it is easier than reading the artifacts. Do not promote a checkpoint because reward went up if replay and telemetry show broken driving.

## Curriculum First

Curriculum is not optional. It is the main tool.

Build the driver section by section and skill by skill. Use the repository's track-section model and curriculum tools. At minimum, reason over:

- `rettifilo_chicane`
- `curva_grande_roggia_run`
- `roggia_chicane`
- `lesmo_1`
- `lesmo_2_serraglio`
- `ascari_approach`
- `ascari_chicane`
- `parabolica_finish`

For each section or future bottleneck, train the actual sub-skills:

- approach setup;
- braking timing;
- speed control;
- turn-in;
- apex/line control;
- lateral stability;
- heading control;
- avoiding off-track/collision;
- exit speed;
- re-acceleration;
- handoff into the next section.

Use a ladder:

1. Micro stage: one braking gate, turn-in, exit, or recovery problem.
2. Section stage: full section from approach to exit.
3. Linked stage: previous section plus current section.
4. Rolling-window stage: randomized starts across learned sections.
5. Full-lap integration: normal starts plus randomized curriculum starts.
6. Final polish: honest normal-start PPO eval/benchmark.

Advance only when the behavior is real. Real means telemetry, QC, and replay agree that the car is driving correctly, not just collecting reward.

## Generalize Beyond The Current Blocker

The current known blocker may be Rettifilo braking, but that is only today's bottleneck. Do not overfit the plan to one failure. After each improvement, immediately search for the next limiting problem.

Use the same loop for anything that appears:

- late braking;
- early braking;
- no braking;
- throttle in a brake zone;
- weak exit speed;
- overspeed at turn-in;
- steering saturation;
- oscillation;
- bad racing line;
- wrong heading;
- lateral drift;
- off-track exits;
- collisions;
- invalid laps;
- slow but valid laps;
- reward hacking;
- segment-completion exploits;
- state-reset exploits;
- train/eval mismatch;
- missing observations;
- bad action discretization;
- continuous-control instability;
- PPO collapse after continuation;
- poor exploration;
- section-transfer failure;
- forgetting earlier skills while learning later sections.

The mission is not to clear one crash. The mission is to keep diagnosing and solving until the whole lap works.

## Change Things Boldly

If the evidence says the training system is paying for the wrong behavior, change the training system. Do not make tiny timid nudges when the same failure keeps happening.

Allowed and expected levers:

- reward scales;
- progress reward shaping;
- braking-zone progress suppression;
- speed-target penalties;
- overspeed-action penalties;
- collision/off-track penalties;
- invalid-lap penalties;
- section success thresholds;
- speed-gated completion;
- temporary early termination for impossible states;
- reset distributions;
- state-library starts;
- forced exploration;
- elite search;
- action perturbation;
- discrete action set redesign;
- continuous action experiments;
- observation features;
- normalization;
- PPO hyperparameters;
- rollout length;
- entropy/exploration pressure;
- vectorized environment count;
- curriculum promotion thresholds;
- benchmark/eval protocol if it hides the real behavior.

If a bad action is repeatedly selected, make it expensive in the focused setup. If PPO cannot discover a maneuver, force exploration or search for elite states. If the policy lacks information, add observations. If the action space blocks the maneuver, redesign or test another action space. If progress reward overpowers safety or validity, change the reward. If a curriculum can be gamed, tighten the success metric. If a section skill does not transfer, train linked stages and randomized starts until it does.

Be aggressive in experiments and strict in claims.

## Full-Lap PPO Has A Role

Full-lap scratch PPO is required, but it is not the default way to discover hard skills once the agent is stuck.

Use full-lap PPO to:

- measure end-to-end progress;
- expose the next bottleneck;
- test transfer;
- integrate learned skills;
- polish a policy that already has section competence.

If full-lap PPO repeats the same failure, stop and go back to curriculum. Do not burn hours raw full-lap training a policy that lacks the required section skill.

## Observability Requirements

Start TensorBoard in the browser and keep it open. Verify training uses CUDA and never silently runs on CPU.

For every serious run, inspect:

- `summary.json`;
- `per_episode.jsonl`;
- selected `steps.jsonl`;
- TensorBoard reward/loss/entropy/value/explained-variance curves;
- benchmark outputs;
- section summaries;
- first-bad-event analysis;
- replay behavior;
- checkpoint metadata;
- train/eval config differences.

Answer concrete questions from artifacts:

- Where is the first bad decision?
- What did the policy observe there?
- What action did it choose?
- What did the reward pay for?
- Was the section success real or gamed?
- Did the skill transfer to linked/full-lap starts?
- Did the change improve the root behavior or just move the crash?
- Did the checkpoint preserve earlier skills?
- Is the final eval metadata-faithful?

Do not guess when the data can tell you.

## Promotion Rules

Section promotion requires:

- stable section entry;
- correct braking or speed control;
- plausible turn-in/apex/exit speed;
- controlled heading and lateral error;
- no collision/off-track;
- exit state that can feed the next section.

Curriculum promotion requires:

- success across varied starts/seeds;
- success when linked with neighboring sections;
- no obvious reward hacking;
- no loss of earlier skills;
- replay that looks like plausible driving.

Full-lap checkpoint promotion requires:

- metadata-faithful benchmark/eval;
- real behavioral improvement;
- artifact proof;
- no hidden assist mismatch.

Final success requires:

- valid normal-start PPO lap;
- near `<=80.0s`;
- scaffold rewards disabled;
- training assists disabled;
- no ghost/scripted/imitation initialization;
- artifacts saved;
- `Documentation.md` updated with proof.

Do not claim final success before this.

## Guardrails

- Do not silently train on CPU.
- Do not revive archived complexity.
- Do not break the simplified path: track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> SB3 PPO -> eval/replay.
- Do not break scripted/manual sanity workflows.
- Do not hide failed experiments.
- Do not leave TODOs for core behavior.
- Do not stop after one failed experiment.
- Do not treat more runtime as progress.
- Do not let infrastructure replace behavioral proof.

## Final Instruction

Really, really lock in. Use the curriculum. Hunt the blocker. Isolate the skill. Change the setup. Run the experiment. Read the artifacts. Keep what works. Throw away what does not. Transfer the skill. Move to the next blocker. Keep doing that with urgency and discipline until PPO completes the valid normal-start Monza lap near `<=80.0s`.
