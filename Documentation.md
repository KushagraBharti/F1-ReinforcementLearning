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
