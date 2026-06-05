# Evolution Search Plan

This file documents the evolution-search implementation and the original pivot away from PPO micro-engineering. The active repeatable loop is now `workflow.md`, and the active target is `goal.md`.

The latest large run already produced valid slow laps, so do not treat the older Rettifilo-first target list as the current bottleneck unless new artifacts prove it has regressed. The current bottleneck is speed: get a valid normal-start evolved lap under `80.0s`.

## Why This Pivot Exists

The previous PPO/curriculum loop learned useful local subskills, but it did not improve the honest normal-start scoreboard:

- no valid full lap;
- no lap time;
- best honest normal-start PPO still around `970.775m`;
- repeated linked-transfer failure through Rettifilo;
- too much time spent on narrow micro-rungs and one-off gates/presets.

The lesson is not "give up on PPO." The lesson is that PPO is not discovering the needed trajectories fast enough. The next system must use Yosh-style population search as the discovery engine.

## New Core Method

Use elitist evolutionary search over driving behavior:

1. Spawn a large population of candidate drivers.
2. Let each candidate drive from the relevant start state or section state.
3. Score each candidate with strict progress, speed, line, stability, and validity gates.
4. Keep the best elites.
5. Mutate and cross over their action phases.
6. Add random immigrants so exploration does not collapse.
7. Repeat for generations.
8. Save elite telemetry and elite state libraries.
9. Use elite states/trajectories to train PPO curriculum only after the search proves a viable behavior exists.

This is not the old fixed `action_search.py` schedule enumeration. That tool is still useful for tiny diagnostics, but it is too manual and too slow as the main loop.

The active reusable tools are:

```powershell
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

Console scripts after `uv sync`:

```powershell
uv run f1-evolution-search --help
uv run f1-evolution-ladder --help
```

## Current Implementation

The new module is `src/f1rl/evolution_search.py`.

It provides:

- phase-based genomes;
- progress-phase genomes;
- observation-driven controller genomes;
- random population initialization;
- elite retention;
- mutation;
- crossover;
- random immigrants;
- chunked optional multiprocessing workers;
- strict segment target gates;
- named scoring profiles;
- multi-profile elite selection;
- live per-generation progress logging;
- streamed `attempts.jsonl`;
- streamed `generation_summary.jsonl`;
- rolling `best_so_far.json`;
- per-generation top genome files;
- resumable `population_checkpoint.json`;
- selected telemetry for top candidates;
- `attempts.jsonl`;
- `evolution_summary.json`;
- `elite_state_library.json`;
- `ppo_bridge.json`;
- `next_commands.md`;
- support for state-library starts or direct simulator starts.

The ladder module is `src/f1rl/evolution_ladder.py`.

It provides:

- `smoke`, `small`, `medium`, and `large` scales;
- normal-start/full-lap probes;
- focused Rettifilo rungs;
- major-milestone full-lap probe mode;
- shared genome/scoring/worker controls.

It is intentionally independent of PPO. Evolutionary trajectories are not final success. They are discovery artifacts and curriculum seeds.

## No More Micro-Engineering Drift

Do not respond to every failed transition by adding a tiny new action set, a one-off schedule preset, and another PPO rung.

Allowed:

- adjust evolutionary population size;
- adjust generation count;
- adjust phase length range;
- adjust mutation/crossover/immigrant rates;
- adjust scoring gates when telemetry proves the gate is accepting fake success;
- switch between `phase`, `progress_phase`, and `controller` genomes;
- run multiple scoring profiles in one search;
- scale from `100` to `1000` to `10000+` candidates only after artifacts look sane;
- add genuinely reusable controller/action primitives if the current action space cannot express the maneuver.

Not allowed:

- spending hours on one manually enumerated schedule preset;
- adding a custom gate for every failed trace without proving it generalizes;
- polishing isolated segment completions that do not transfer;
- calling search/scripted/evolutionary behavior final PPO success.

## First Search Targets

Use the evolutionary loop to solve linked Rettifilo transfer in progressively wider targets:

1. `945->1000`
   - Goal: immediate post-release stabilization through the heading transition.
   - Use strict line/heading/yaw/steering gates.

2. `945->1080`
   - Goal: stable exit handoff.
   - Reject candidates that reach progress but cannot continue.

3. `650->1220`
   - Goal: turn-in, rotation, stabilization, and exit.
   - This is the first meaningful full-Rettifilo successor target.

4. `520->1220`
   - Goal: brake, release, turn-in, rotate, stabilize, exit.
   - This is the key transfer proof before serious full-lap PPO.

5. Normal-start honest eval
   - Goal: verify whether full-lap behavior changes beyond `~970.775m`.

## Example Commands

Tiny smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search `
  --output-dir artifacts\evolution-smoke `
  --start-progress-m 500 `
  --start-speed-kph 80 `
  --target-progress-m 510 `
  --action-set straight `
  --observation-profile base `
  --max-steps 24 `
  --population 8 `
  --generations 2 `
  --elite-count 2 `
  --random-immigrants 1 `
  --top-k 2 `
  --workers 1 `
  --genome-type phase `
  --scoring-profiles max_progress,clean_exit `
  --progress-every-generation
```

Smoke ladder:

```powershell
uv run --no-sync python -m f1rl.evolution_ladder `
  --output-dir artifacts\evolution-ladder-smoke `
  --scale smoke `
  --workers 1 `
  --full-probe-mode none `
  --genome-type phase `
  --action-set straight `
  --observation-profile base `
  --progress-every-generation
```

Rettifilo stabilization search:

```powershell
uv run --no-sync python -m f1rl.evolution_search `
  --state-library artifacts\state-library-rettifilo-clean944_946-rank000-20260603\state_library.json `
  --output-dir artifacts\evolution-rettifilo-945-1000 `
  --target-progress-m 1000 `
  --target-max-speed-kph 175 `
  --target-max-lateral-error-m 6 `
  --target-max-heading-error-deg 18 `
  --target-max-abs-yaw-rate-rps 0.5 `
  --target-max-abs-steering 0.25 `
  --action-set turnin_power `
  --observation-profile racing_v2 `
  --max-steps 160 `
  --population 1000 `
  --generations 20 `
  --elite-count 64 `
  --random-immigrants 128 `
  --min-phases 2 `
  --max-phases 8 `
  --min-phase-steps 3 `
  --max-phase-steps 36 `
  --top-k 24 `
  --workers 8 `
  --worker-chunk-size 0 `
  --genome-type controller `
  --scoring-profiles max_progress,clean_exit,risk_seeking `
  --checkpoint-every-generations 1 `
  --progress-every-generation
```

Wider Rettifilo search:

```powershell
uv run --no-sync python -m f1rl.evolution_search `
  --state-library artifacts\state-library-rettifilo-turnin-650_760-from930-20260603-linegate\state_library.json `
  --output-dir artifacts\evolution-rettifilo-650-1220 `
  --target-progress-m 1220 `
  --target-max-speed-kph 230 `
  --target-max-lateral-error-m 10 `
  --target-max-heading-error-deg 24 `
  --target-max-abs-yaw-rate-rps 0.8 `
  --action-set racing `
  --observation-profile racing_v2 `
  --max-steps 520 `
  --population 2000 `
  --generations 30 `
  --elite-count 96 `
  --random-immigrants 192 `
  --min-phases 3 `
  --max-phases 12 `
  --min-phase-steps 4 `
  --max-phase-steps 60 `
  --top-k 32 `
  --workers 8 `
  --worker-chunk-size 0 `
  --genome-type controller `
  --scoring-profiles max_progress,clean_exit,apex,exit_speed,risk_seeking `
  --checkpoint-every-generations 1 `
  --progress-every-generation
```

## What Counts As Progress

Evolution progress:

- higher best progress to target;
- target completion with strict gates;
- clean selected telemetry;
- top candidates that continue into the next section;
- saved elite states that PPO can use.

PPO progress:

- PPO learns from those elite starts;
- PPO reproduces the behavior from varied starts;
- PPO transfers to wider linked starts;
- honest normal-start eval improves beyond the old `~970.775m` crash.

Final success:

- PPO completes a valid normal-start Monza lap;
- near `<=80.0s`;
- no scaffold rewards;
- no training assists;
- no ghost/scripted/imitation/evolutionary policy counted as the final driver;
- artifacts and `Documentation.md` prove it.

## Validation Already Run

After adding `evolution_search.py`:

- `uv run --no-sync ruff check .` passed.
- `uv run --no-sync pytest -q` passed.
- `uv run --no-sync pyright src/f1rl` passed.
- CLI smoke search wrote `artifacts\evolution-smoke-20260603\evolution_summary.json`.
- Smoke search produced `attempts.jsonl`, selected telemetry, and `elite_state_library.json`.

After the Yosh-scale upgrade:

- `phase`, `progress_phase`, and `controller` genome CLI smokes passed.
- `--workers 8` CLI smoke passed.
- stop/resume CLI smoke passed.
- ladder smoke passed.
- focused evolution/search tests passed.

## Next Agent Rule

The next agent should not resume the old micro-PPO goal loop. It should run population search first, inspect elite telemetry, save useful state libraries, and only then use PPO to learn from search-discovered behavior.
