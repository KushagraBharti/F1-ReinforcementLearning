# Goal Agent Prompt

Use this as the active goal prompt.

You are still pursuing the original F1-ReinforcementLearning goal. Do not abandon it, do not mark it complete, and do not replace it with a new project.

Strict target:

- Train a PPO agent that completes a valid normal-start Monza lap.
- The target is near the Fast-F1 ghost reference: `<=80.0s`.
- Do not count partial distance, assisted completion, runtime duration, infrastructure completion, or segment-only success as completion.

Start by reading these files in full:

1. `AGENTS.md`
2. `Prompt.md`
3. `Plan.md`
4. `Implement.md`
5. `LearningPlan.md`
6. `fine-tuned learning plan.md`
7. every file under `transcripts/`

The transcript files are mandatory. Treat them as operating instructions, not optional context. The key lesson is that Yosh-style progress comes from aggressive isolated skill training, missing-observation fixes, temporary rewards, forced exploration, state starts, elite-state libraries, and fast iteration over many focused attempts.

Use all available resources:

- run training/eval/benchmark/replay commands yourself;
- use CUDA for PPO training and verify it is actually using CUDA;
- use web search when current API/version behavior matters;
- use the browser/TensorBoard UI for live curves and replay inspection;
- inspect raw artifacts instead of relying on console summaries.

Start TensorBoard in the browser and keep it open while working.

The baseline requirement still matters:

- Train from a fully scratch/random PPO policy for the honest scratch baseline.
- Do not initialize the promoted scratch baseline from ghost, scripted, imitation, or legacy policy weights.
- Segment curriculum, state-library starts, elite search, reward scaffolds, and assists are allowed as experiments and curriculum tools, but final promotion must be honest and metadata-faithful.

Current known failure:

- best honest PPO: about `970.775m`, `20/120`, collision, no valid lap;
- first bad event: `throttle_during_brake_demand`;
- section: `rettifilo_chicane`;
- progress: about `520.8m`;
- speed: about `334kph`;
- target section speed: about `115kph`;
- terminal outcome: collision around `966-971m`.

Main diagnosis:

This is not mainly an eval bug anymore. Eval/benchmark are now metadata-faithful enough to trust. PPO is stuck because the training objective still rewards "go far fast and crash" better than "sacrifice immediate progress to brake early." The task must be changed so the bad behavior is impossible, unprofitable, or quickly rejected during focused training.

Operating mode:

- Lock in.
- Stop being conservative.
- Stop doing long mild PPO continuations that preserve the same failure.
- Stop treating infrastructure completion as learning progress.
- Stop celebrating tiny distance improvements when the first bad event is unchanged.
- Run many fast, targeted experiments and keep only changes that alter behavior.

The rough persistence expectation is still `8+ hours`, but persistence means many measured experiment loops, not one blind run. Long runs are only justified after a mini experiment changes the failure signature, produces a real section skill, or creates a checkpoint worth scaling.

Required loop:

1. Benchmark the current initial/final/best checkpoints before and after each run.
2. Inspect `summary.json`, `per_episode.jsonl`, selected `steps.jsonl`, TensorBoard curves, section QC, first-bad-event output, and replay behavior.
3. Identify the repeated failure in telemetry.
4. Change the reward, metric, assist, curriculum, observation, action space, reset distribution, or elite-state process around that failure.
5. Run a short focused mini experiment.
6. Reject quickly if it keeps the same first bad event.
7. Promote only if the behavior materially improves under honest benchmark conditions.
8. Update `Documentation.md` with the hypothesis, command, result, diagnosis, and next action.
9. Repeat.

For the current Rettifilo plateau, the next milestone is not another `+4m`. The next milestone is changing the first bad event: the car must brake before Rettifilo turn-in and exit the section alive.

Aggressive changes are allowed and expected:

- make full throttle in Rettifilo brake demand strongly negative;
- temporarily terminate overspeed turn-in during section training;
- reduce or zero progress reward inside active brake demand;
- substantially increase speed-target and overspeed-action penalties in section experiments;
- increase collision penalty for high-speed braking-zone failures;
- require speed-gated segment completion at every Rettifilo gate;
- reward braking only when speed surplus is real and the car is before turn-in;
- penalize no-brake and throttle-through-brake-demand hard enough that PPO cannot ignore them;
- run state-library starts from multiple Rettifilo approach states;
- run stochastic, evolutionary, action-perturbation, and elite-search attempts to discover clean Rettifilo exits;
- save elite states only when speed, heading, lateral, validity, and no-collision criteria are actually met;
- train section-first, then transfer only after the section behavior is proven;
- retry continuous `exclusive_throttle_bias` only after section skill is proven, not as a blind replacement.

Full-lap and segment-curriculum PPO should both be used, but with discipline:

- full-lap scratch PPO measures whether the whole task is improving;
- segment curriculum isolates the current blocker;
- state starts test whether PPO can learn the skill when discovery is easier;
- elite search finds demonstrations/states/initializations for curriculum analysis, not fake final success;
- final benchmark must disable scaffold rewards and training assists.

Promotion rules:

- Promotion requires measured improvement in metadata-faithful benchmark/eval.
- Final success requires a valid normal-start lap with PPO, no scaffold rewards, no training assists, no ghost/scripted/imitation initialization, and lap time near `<=80.0s`.
- A section checkpoint can be promoted only to the next curriculum stage, not to final success.
- If a checkpoint reaches farther but still has `throttle_during_brake_demand` around `520m`, treat it as not solved.

Guardrails:

- Do not silently train on CPU.
- Do not revive archived complexity.
- Do not break the simplified path: track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium -> SB3 PPO -> eval/replay.
- Do not break scripted valid-lap sanity checks.
- Do not overfit to fake lap success from assisted training.
- Do not hide failed experiments.
- Do not stop after a single failed experiment.

The goal is a working PPO driver, not a neat sequence of cautious changes. Push hard, measure honestly, document what happened, and keep iterating until PPO completes the valid normal-start Monza lap near `<=80.0s`.
