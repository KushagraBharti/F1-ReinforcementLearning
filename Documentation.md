# Documentation

## Current Status

Date: 2026-06-04.

The project has pivoted from PPO micro-rung tuning to reusable elitist evolutionary search.

Honest scoreboard:

- Valid normal-start PPO lap: no.
- Lap time: none.
- Best honest normal-start PPO progress: about `970.775m`.
- Final target: valid normal-start PPO lap near `<=80.0s`.

Why the pivot happened:

- The previous PPO/curriculum loop produced local Rettifilo subskills.
- It did not improve honest normal-start behavior.
- It drifted into narrow gates, one-off action sets, and slow hand-written schedule searches.
- The project now needs population search: spawn candidates, score them, keep elites, mutate, repeat.

## Active Direction

Use `EvolutionSearchPlan.md` and `EvolutionGoal.md`.

Immediate method:

1. Run elitist evolutionary search to discover viable section trajectories.
2. Save elite telemetry and elite state libraries.
3. Run the evolution ladder to alternate honest full-lap probes with focused segment discovery.
4. Use PPO curriculum to learn from those search-discovered states.
5. Retest honest normal-start PPO.

Do not resume the old PPO micro-engineering loop.

## Active Implementation

Evolution search module:

- `src/f1rl/evolution_search.py`

CLI:

- `python -m f1rl.evolution_search`
- `f1-evolution-search`
- `python -m f1rl.evolution_ladder`
- `f1-evolution-ladder`

Outputs:

- `evolution_summary.json`
- `attempts.jsonl`
- `generation_summary.jsonl`
- `best_so_far.json`
- `population_checkpoint.json`
- `top_genomes/*.json`
- `selected_telemetry/*.jsonl`
- `elite_state_library.json`
- `ppo_bridge.json`
- `next_commands.md`

Core behavior:

- phase-based genomes;
- progress-phase genomes;
- observation-driven controller genomes;
- population initialization;
- elite retention;
- mutation;
- crossover;
- random immigrants;
- chunked optional multiprocessing;
- strict target gates;
- scoring profiles;
- multi-profile elite selection;
- streaming artifacts;
- checkpoint/resume.

Evolution ladder module:

- `src/f1rl/evolution_ladder.py`

It runs smoke/small/medium/large search ladders that start with normal-start probes, search Rettifilo rungs, and return to full-lap probes at major milestones.

Legacy diagnostics:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`

They remain available for tiny investigations, but they are not the active main path.

## Archived Context

The previous context-heavy docs were archived under:

- `archive/plans/evolution-doc-cleanup-20260603/`
- `archive/plans/evolution-pivot-20260603/`
- `archive/legacy-20260424/`

Do not read those by default. They are historical reference only.

## Latest Validation

After the Yosh-scale evolution implementation:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv run --no-sync pyright src/f1rl
```

Result:

- Ruff passed.
- Pytest passed.
- Pyright passed.

Focused validations run:

```powershell
uv run --no-sync pytest tests/test_evolution_search.py -q
uv run --no-sync pytest tests/test_evolution_search.py tests/test_action_search.py tests/test_elite_search.py -q
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

Result:

- Evolution search tests passed.
- Legacy search tests still passed.
- Phase, progress-phase, and controller CLI smokes completed.
- `--workers 8` CLI smoke completed.
- CLI stop/resume completed.
- Ladder smoke completed at `artifacts\evolution-ladder-smoke-20260603`.

## Next Step

Run real search at increasing scale:

1. `smoke`: validate commands and artifacts.
2. `small`: population around `100`, inspect behavior.
3. `medium`: population around `1000`, compare profiles/genomes.
4. `large`: long-running search only after streaming, resume, and telemetry look sane.

Suggested command shapes are in `EvolutionSearchPlan.md`.

After search finds useful elites:

1. Inspect selected telemetry.
2. Save/use the elite state library.
3. Train PPO curriculum from those states.
4. Test linked transfer.
5. Test honest normal-start eval.

Do not claim final success until PPO completes a valid normal-start lap near `<=80.0s` with scaffolds and assists disabled.

## Generation Replay Update - 2026-06-04

Added generation-by-generation replay for evolution swarm artifacts.

Use:

```powershell
uv run --no-sync f1-replay 'artifacts\evolution-swarm-100x5-visible-next\selected_telemetry' --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Behavior:

- replays one manifest generation at a time;
- keeps candidates sorted by best progress within each generation;
- shows the primary/best candidate plus the rest of that generation as ghost cars;
- displays generation index, raw generation number, car count, and generation best distance.
- press `N` or `Space` during generation replay to skip the current generation.

Validation:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Result:

- Ruff passed.
- Pyright passed.
- Pytest passed.

## Evolution Streaming Telemetry Update - 2026-06-04

`--telemetry-selection all` now preserves all candidate telemetry without returning full step traces through multiprocessing. Workers stream replay-compatible JSONL files directly into `selected_telemetry/` using per-candidate temporary files, then atomically rename them when complete. The parent process keeps only compact scoring rows in memory and writes the manifest after the run.

This prevents the `MemoryError` seen when long-horizon all-candidate telemetry tried to pickle large trace payloads through `concurrent.futures.process`.

Recommended for large debug runs:

- write `--output-dir` outside OneDrive, e.g. `C:\f1rl-artifacts\...`;
- keep `--telemetry-selection all` when full debugging/replay is needed;
- use `leaders` for high-throughput search when full replay for every candidate is not required.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py::test_evolution_search_can_save_all_candidate_telemetry_for_swarm_replay
```

Result:

- Ruff passed.
- Multi-worker all-candidate telemetry test passed.

Run:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-0to1220-100x10-stream-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1220 --action-set racing --observation-profile racing_v2 --max-steps 10000 --population 100 --generations 10 --elite-count 16 --random-immigrants 20 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles risk_seeking,max_progress,clean_exit --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation
```

Result:

- completed without the previous multiprocessing telemetry `MemoryError`;
- wrote 1000 candidate traces;
- best distance reached `1130.4m`;
- no `1220m` segment completion yet.

## Open-Distance Evolution Update - 2026-06-04

Added `--no-target-termination` for evolutionary search. In this mode, `--target-progress-m` is only a milestone/completion-rate threshold for scoring and reporting. It does not install `segment_length_m`, so candidates keep driving until normal simulator termination: lap complete, collision, off-track, no-progress, or `--max-steps`.

This fixes the capped-search behavior where a run could reach the target and stop before showing how much farther the controller could brute-force.

Additional search pressure changes:

- open-distance scoring now rewards progress beyond the milestone instead of capping base progress at `target_progress_m`;
- `frontier` and `risk_seeking` scoring profiles are more distance/speed-biased;
- controller mutation is more aggressive and resets more weights;
- parent selection is more rank-biased;
- generation logs now print `leader_progress_m`, `generation_best_distance_m`, `generation_average_distance_m`, `global_best_distance_m`, `completion_rate`, and `milestone_rate`.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --headless
```

Result:

- Ruff passed.
- Pyright passed.
- Focused evolution tests passed.
- Replay loaded 10 generation groups and 1000 traces.

Run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 10000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time: `104.03s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2409.611m`;
- best candidate: generation `9`, candidate `72`;
- best termination: `collision`, not simulator segment completion (`sim_segment_complete=false`);
- final speed: `300.68 kph`;
- final lateral error: `17.34m`;
- final heading error: `-4.20deg`;
- milestone counts across all attempts: `88 >=1000m`, `54 >=1220m`, `14 >=1500m`, `14 >=2000m`, `1 >=2400m`.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-aggressive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## General 100x30 Compute Probe - 2026-06-04

Ran a fresh normal-start open-distance evolutionary search from scratch to test whether more compute solves the frontier without segment initialization.

Command:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x30-general-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 30 --elite-count 12 --random-immigrants 24 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 30 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,frontier,risk_seeking,max_progress,early_pace,clean_distance,clean_exit,exit_speed,frontier_recovery,frontier_novelty --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2000 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 80 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time including 3000-trace packaging: `999.76s`;
- attempts: `3000`;
- saved telemetry traces: `3000`;
- best distance: `5223.047m`;
- best candidate: generation `16`, candidate `73`;
- source: `offspring` from `cleanest_distance`;
- parent: generation `15`, candidate `1`;
- mutation: `controller_masked_gaussian`;
- termination: `collision`;
- final speed: `214.33 kph`;
- no valid lap completion yet (`0 >=5793m`);
- counts: `414 >=1500m`, `385 >=2000m`, `275 >=2200m`, `245 >=2400m`, `151 >=2600m`, `72 >=3000m`, `66 >=4000m`, `54 >=5000m`;
- best reached `parabolica_finish`, about `37m` before its braking gate at `5260m`;
- best final behavior: full throttle, no brake, lateral error drifted to `20.59m`, heading error about `-33.8deg`, then collision.

Diagnosis:

- More compute did solve multiple previous frontiers. The run broke past Rettifilo/Roggia/Lesmo and reached the final sector.
- It did not finish the lap. The new blocker is Parabolica entry braking/positioning.
- Plateau mode preserved the best but was disruptive after activation; average distance dropped immediately after plateau-triggered generations. For late full-lap refinement, plateau extra mutation should probably be reduced or only applied to a smaller branch.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x30-general-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## Frontier Recovery / Roggia Segment Update - 2026-06-04

Implemented frontier-specific tools after the normal-start search plateaued around Roggia exit:

- added `frontier_recovery` scoring to prefer alive, controlled exits through a configurable focus band;
- added `frontier_novelty` scoring to keep useful variation near the frontier;
- added a `frontier_distance` parent bucket for rare far candidates;
- added plateau mode, which detects flat best distance with improving average distance, reduces elite-copy pressure, and applies extra mutations to frontier parents;
- added ladder rungs for `roggia-exit-lesmo-1900-2800` and `roggia-lesmo-transfer-2000-3030`;
- exposed frontier and plateau knobs through `f1-evolution-search` and `f1-evolution-ladder`.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py src/f1rl/evolution_ladder.py tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py src/f1rl/evolution_ladder.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-frontier-cli-smoke --start-progress-m 2200 --start-speed-kph 180 --target-progress-m 2600 --action-set straight --observation-profile base --max-steps 32 --population 8 --generations 2 --elite-count 2 --random-immigrants 2 --top-k 2 --workers 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,clean_distance --frontier-focus-start-m 2200 --frontier-focus-end-m 2600 --frontier-parent-min-progress-m 2200 --progress-every-generation --telemetry-selection leaders
uv run --no-sync python -m f1rl.evolution_ladder --output-dir artifacts\evolution-ladder-frontier-smoke --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base --scoring-profiles max_progress,frontier_recovery,frontier_novelty --progress-every-generation
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Result:

- focused Ruff passed;
- focused Pyright passed;
- focused evolution tests passed;
- frontier evolution CLI smoke passed;
- ladder smoke with Roggia rungs passed;
- full Ruff passed;
- full Pyright passed;
- full pytest passed.

Focused segment run:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-roggia-1900-2800-frontier-100x10-20260604 --start-progress-m 1900 --start-speed-kph 315 --target-progress-m 2800 --target-min-speed-kph 120 --target-max-speed-kph 245 --target-max-lateral-error-m 16 --target-max-heading-error-deg 34 --action-set racing --observation-profile racing_v2 --max-steps 1200 --population 100 --generations 10 --elite-count 12 --random-immigrants 20 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 24 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,clean_distance,exit_speed,max_progress,risk_seeking --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 2000,2200,2400,2600,2800 --survival-floor-pass-rate 0.25 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2200 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 60 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
```

Focused segment result:

- elapsed time: `69.98s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2800.629m`;
- best selected candidate: generation `8`, candidate `85`;
- selected candidate source: `offspring` from `fastest_pace`;
- selected candidate ended with `segment_complete`;
- final speed: `216.92 kph`;
- completion rate improved to `17%`;
- replay loaded `10` groups and `1000` traces.

Normal-start transfer probe:

```powershell
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-frontier-plateau-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.98 --crossover-rate 0.65 --top-k 24 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_recovery,frontier_novelty,frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --frontier-focus-start-m 2200 --frontier-focus-end-m 2800 --frontier-parent-min-progress-m 2000 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4 --plateau-generations 3 --plateau-distance-epsilon-m 10 --plateau-average-improvement-m 80 --plateau-elite-fraction 0.45 --plateau-extra-mutations 2
```

Normal-start transfer result:

- elapsed time: `185.25s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2157.566m`;
- best selected candidate: generation `5`, candidate `15`;
- selected candidate source: `offspring` from `frontier_distance`;
- selected candidate ended with `collision`;
- final speed: `137.18 kph`;
- `61` attempts reached `>=1500m`;
- `45` attempts reached `>=2000m`;
- `0` attempts reached `>=2200m`;
- replay loaded `10` groups and `1000` traces.

Diagnosis:

- The focused segment search solved a local Roggia-exit skill.
- The normal-start probe did not transfer that skill and regressed from the previous `2457.633m` best.
- The frontier-recovery profiles are useful for isolated segment discovery but too local/recovery-biased when used as the leading normal-start full-lap objective.
- Next full-lap search should keep `frontier_recovery`/`frontier_novelty` as secondary diversity profiles, while restoring `frontier_fast`/`farthest_distance` as the primary full-lap profiles.

Follow-up refinement:

- adaptive immigrant pressure now uses the broader quality pass rate as well as the active survival floor pass rate;
- this keeps more slots assigned to offspring/smart near-elite mutation after the population stops dying early, even when the active parent floor has moved outward.

Second run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-adaptive-quality-v2-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Second run result:

- elapsed time: `132.72s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2457.633m`;
- best candidate: generation `6`, candidate `50`;
- best candidate source: `offspring` from `survival_gate`;
- best parent: generation `5`, candidate `83`;
- best mutation: `controller_masked_gaussian`;
- best termination: `no_progress`;
- average distance improved from `71.08m` in generation 0 to `840.55m` in generation 9;
- total counts: `589 >=100m`, `473 >=450m`, `78 >=1000m`, `41 >=1220m`, `39 >=1500m`, `35 >=2000m`, `7 >=2400m`;
- all-candidate replay loaded `10` groups and `1000` traces.

Second run replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-adaptive-quality-v2-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.

## Adaptive Evolution Selection Update - 2026-06-04

Implemented the next aggressive evolution-search pass:

- performance-based dynamic survival floor for parent selection;
- tiered parent buckets: survival gate, farthest distance, fastest pace, cleanest distance;
- adaptive immigrant budget so later generations spend more slots on offspring when survival improves;
- smart immigrants split from pure random immigrants, mostly near current elites with some global archive support;
- conservative pace-sensitive scoring profiles: `frontier_fast`, `early_pace`, `clean_distance`, `farthest_distance`;
- lineage logging for parent, crossover partner, mutation type/sigma/reset count, genome distance, immigrant type, and source bucket;
- generation summaries now include survival floor/pass rates, lineage source counts, progress threshold counts, average pace, and next-population reproduction counts.

Research basis:

- performance-based exploit/explore mirrors Population Based Training style adaptation;
- keeping multiple useful buckets follows quality-diversity / MAP-Elites style pressure;
- smart archive/current elite reuse follows the hard-exploration lesson of returning to promising states before exploring.

Validation:

```powershell
uv run --no-sync ruff check src/f1rl/evolution_search.py tests/test_evolution_search.py
uv run --no-sync pytest -q tests/test_evolution_search.py
uv run --no-sync pyright src/f1rl/evolution_search.py
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-adaptive-cli-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 2 --top-k 2 --workers 1 --genome-type controller --scoring-profiles frontier_fast,early_pace,clean_distance,farthest_distance --progress-every-generation --telemetry-selection leaders
uv run --no-sync python -m f1rl.evolution_ladder --output-dir artifacts\evolution-ladder-adaptive-smoke --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base --progress-every-generation
```

Result:

- focused Ruff passed;
- focused evolution tests passed;
- focused Pyright passed;
- full Ruff passed;
- full Pyright passed;
- full pytest passed;
- evolution-search adaptive CLI smoke passed;
- evolution-ladder smoke passed.

Run:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-open-100x10-adaptive-20260604 --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 15000 --population 100 --generations 10 --elite-count 12 --random-immigrants 24 --mutation-rate 0.97 --crossover-rate 0.60 --top-k 20 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed --telemetry-selection all --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400 --survival-floor-pass-rate 0.30 --smart-immigrant-fraction 0.50 --smart-immigrant-current-fraction 0.90 --min-random-immigrants 4
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

Result:

- elapsed time: `157.20s`;
- attempts: `1000`;
- saved telemetry traces: `1000`;
- best distance: `2448.570m`;
- best candidate: generation `7`, candidate `80`;
- best candidate source: `smart_immigrant` from `smart_current_elite`;
- best parent: generation `6`, candidate `55`;
- best mutation: `controller_weight_reset`;
- best termination: `collision`;
- final speed: `169.45 kph`;
- average distance improved from `71.08m` in generation 0 to `766.60m` in generation 9;
- final generation counts: `63 >=450m`, `18 >=1000m`, `15 >=1220m`, `11 >=1500m`, `10 >=2000m`, `1 >=2400m`;
- all-candidate replay loaded `10` groups and `1000` traces.

Replay:

```powershell
uv run --no-sync f1-replay C:\f1rl-artifacts\evolution-open-100x10-adaptive-20260604\selected_telemetry --sort best-progress --by-generation --generation-limit 100 --speed 1
```

Press `N` or `Space` to skip a generation.
