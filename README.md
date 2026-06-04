# F1 Reinforcement Learning

Top-down 2D Monza simulator with telemetry, manual/scripted driving, Gymnasium/SB3 PPO, replay, benchmark/QC tools, and an evolution-first search path.

<p align="center">
  <img src="./pygame-window-gen49-fastest-89s-all-150-cars-slow.gif" alt="Monza simulator replay GIF" width="900" />
</p>

The current strategy is not blind full-lap PPO. It is:

1. Discover viable section trajectories with elitist evolutionary search.
2. Save elite telemetry and state libraries.
3. Run a repeatable ladder that alternates normal-start probes with focused segment search.
4. Train PPO curriculum from those search-discovered states.
5. Retest honest normal-start PPO.

Final target: a valid normal-start Monza PPO lap near `<=80.0s`.

## Active Docs

- `AGENTS.md`: agent operating rules.
- `EvolutionSearchPlan.md`: current implementation plan.
- `EvolutionGoal.md`: goal-thread prompt.
- `Documentation.md`: concise live status and validation log.

Old context-heavy docs were archived under:

- `archive/plans/evolution-doc-cleanup-20260603/`
- `archive/plans/evolution-pivot-20260603/`
- `archive/legacy-20260424/`

Do not use old archived plans as active instructions.

## Setup

```powershell
cd "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning"
uv sync --active --all-extras --all-packages
```

For repeated commands after setup:

```powershell
uv run --no-sync python -m f1rl.hardware --json
```

## Core Commands

Manual driving:

```powershell
uv run --no-sync python -m f1rl.manual
```

Scripted baseline:

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Evolutionary search help:

```powershell
uv run --no-sync python -m f1rl.evolution_search --help
uv run --no-sync python -m f1rl.evolution_ladder --help
```

After `uv sync`, console scripts are also available:

```powershell
uv run f1-evolution-search --help
uv run f1-evolution-ladder --help
```

Tiny evolutionary smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

Smoke ladder:

```powershell
uv run --no-sync python -m f1rl.evolution_ladder --output-dir artifacts\evolution-ladder-smoke --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base --progress-every-generation
```

Resume a search from a checkpoint:

```powershell
uv run --no-sync python -m f1rl.evolution_search --resume artifacts\evolution-smoke\population_checkpoint.json --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase
```

PPO smoke:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name ppo-smoke
```

Evaluate:

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint latest --steps 600 --device auto
```

Benchmark:

```powershell
uv run --no-sync python -m f1rl.benchmark --help
```

Replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<run>\steps.jsonl"
```

## Validation

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv run --no-sync pyright src/f1rl
```

Latest known validation after the evolution-search pivot:

- `ruff check .`: pass.
- `pytest -q`: pass.
- `pyright src/f1rl`: pass.
- evolution-search CLI smoke: pass.
- evolution-ladder CLI smoke: pass.

## Current Status

No valid PPO full lap yet.

Best honest normal-start PPO remains about `970.775m`, no valid lap, no lap time.

The old PPO/curriculum loop learned useful local Rettifilo subskills but did not transfer to honest normal-start driving. The active path is reusable population-based evolutionary search and ladder probes, not more one-off PPO micro-rungs.

`f1rl.action_search` and `f1rl.elite_search` remain available as legacy diagnostics, but the main search path is `f1rl.evolution_search` plus `f1rl.evolution_ladder`.
