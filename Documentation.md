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
