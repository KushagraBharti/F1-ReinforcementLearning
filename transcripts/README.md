# Transcript Reading Guide

These transcript files are mandatory context for the F1RL goal agent. Read them before launching more PPO experiments.

## Files

1. `01-yosh-trackmania-2023.txt`
   - Core lesson: simple progress rewards plateau; the agent improved after adding better observations, curriculum starts, and temporary skill rewards.
2. `02-yosh-noseboost.txt`
   - Core lesson: when the target skill is rare, reshape the environment/reward so the desired behavior becomes discoverable instead of hoping PPO stumbles into it.
3. `03-yosh-a01.txt`
   - Core lesson: section-specific rewards, helper logic, segmented best-state continuation, and repeated attempts matter more than one polished global training run.
4. `04-yosh-a06.txt`
   - Core lesson: force the route/behavior you need, then optimize the section hard; late-stage performance comes from targeted track-specific pressure.
5. `05-f1rl-methods-summary.txt`
   - Project-specific translation: spawn everywhere, add observations from failure telemetry, use temporary scaffolds, force exploration, save successful states, run elite section search, and return to honest full-lap eval.

## Required Interpretation

Do not treat these as inspirational background. Treat them as operating instructions for escaping the current PPO plateau.

The current F1RL failure is not "needs more training." It is:

- first bad event: `throttle_during_brake_demand`,
- section: `rettifilo_chicane`,
- progress: about `520.8m`,
- speed: about `334kph`,
- target speed: `115kph`,
- terminal result: collision around `966-971m`.

Use the transcripts to justify aggressive mini experiments:

- make the Rettifilo brake zone impossible to ignore during training,
- run short experiments that test one hypothesis each,
- change rewards, assists, curriculum starts, action sets, and observation profiles quickly,
- reject failures quickly,
- keep elite successful section states,
- promote only scaffold-free normal-start improvements.

The desired behavior is not another tiny progress gain from `966m` to `970m`. The desired behavior is a materially different first bad event: braking before Rettifilo turn-in, lower entry speed, clean apex/exit, and eventual full-lap transfer.
