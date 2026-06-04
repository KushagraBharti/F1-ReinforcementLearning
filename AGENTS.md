# AGENTS.md

## Mission

This repo is a simplified top-down 2D Monza driving simulator and learning project.

Keep the runtime path explicit:

`track geometry -> car physics -> shared simulator -> manual/scripted -> telemetry -> Gymnasium/SB3 PPO -> eval/replay`

The current learning strategy is evolution-first discovery, then PPO transfer:

1. Use elitist evolutionary search to discover viable driving trajectories and elite state libraries.
2. Use the evolution ladder to alternate full-lap probes with focused segment search.
3. Use PPO curriculum to learn from those search-discovered states.
4. Promote only honest normal-start PPO results.

Final target remains a valid normal-start Monza PPO lap near `<=80.0s`.

## Active Docs

Read these first:

- `README.md`
- `goal.md`
- `workflow.md`
- `EvolutionSearchPlan.md`
- `EvolutionGoal.md`
- `Documentation.md`

Only read archived docs, old plans, transcripts, or artifact reports when a task explicitly needs historical detail.

Archived/old material lives under:

- `archive/`
- `artifacts/`
- `transcripts/`

Do not bulk-read those folders by default.

`archive/` and `artifacts/` are ignored by `rg` through `.rgignore` to keep default searches useful. Use `rg --no-ignore` only when historical artifacts are explicitly needed.

## Current Non-Negotiables

- Do not resume the old PPO micro-engineering loop.
- Do not add one-off action sets, gates, or schedule presets unless they are clearly reusable or a short diagnostic.
- Do not treat local segment success as final success.
- Do not treat evolutionary/search/scripted trajectories as final PPO success.
- Do not initialize final scratch PPO policy weights from ghost/scripted/imitation/evolutionary policies.
- Do use evolutionary search to discover trajectories, states, action primitives, and curriculum targets.
- Do use `f1rl.evolution_ladder` for repeatable Yosh-style search ladders instead of hand-inventing every rung.
- Do use PPO only after search has found behavior worth learning.
- Do keep artifacts and validation explicit.
- Do compress verified old analyzed large-run folders under `C:\f1rl-artifacts\archives` and delete the original folders after the next changes have been implemented/validated and immediately before launching another full `100x30` or larger run.
- Do not delete active, unanalyzed, or unverified runs.
- Treat `action_search.py` and `elite_search.py` as legacy diagnostics. They are not the main path.

## Work Loop

Operate autonomously:

1. Inspect.
2. Form a short plan.
3. Implement.
4. Validate.
5. Fix failures.
6. Update `Documentation.md`.
7. Repeat.

If validation fails, fix it before moving on.

## Tech Expectations

- Python `>=3.11,<3.13`.
- Use `uv`.
- Simulator, renderer, geometry, and telemetry stay CPU-bound unless deliberately redesigned.
- PyTorch/SB3 PPO training uses CUDA when available and requested.
- Keep rendering and training decoupled.
- Persist per-step JSONL telemetry and episode summaries.

## Validation

Use the smallest validation that proves the change, then broaden when needed.

Common checks:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest -q
uv run --no-sync pyright src/f1rl
```

For evolutionary search changes, also run a tiny CLI smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1
```

For ladder changes, also run a smoke ladder:

```powershell
uv run --no-sync python -m f1rl.evolution_ladder --scale smoke --workers 1 --full-probe-mode none --genome-type phase --action-set straight --observation-profile base
```

## Documentation Rule

`Documentation.md` is now a concise live status file. Do not turn it back into a giant transcript. Archive long logs under `archive/plans/` or keep them as artifacts.
