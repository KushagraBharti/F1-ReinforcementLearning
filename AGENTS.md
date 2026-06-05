# AGENTS.md

## Mission

This repo is a simplified top-down 2D Monza driving simulator and learning/search project.

Keep the runtime path explicit:

`track assets -> track geometry -> simulator -> telemetry -> replay/eval -> learning/search`

The current strongest path is:

1. Use GPU evolutionary search to discover fast valid laps.
2. CPU-verify/rerank selected winners.
3. Save replayable selected telemetry.
4. Use verified ES data for learned-policy work.

The old blind PPO micro-rung loop is not the active lead path.

## Active Docs

Read these first:

- `README.md`
- `Documentation.md`
- `AGENTS.md`

Archived/old material lives under:

- `archive/`
- `artifacts/`
- `transcripts/`

Do not bulk-read archived docs, old plans, transcripts, or artifacts unless the task explicitly needs historical detail.

`archive/` and `artifacts/` are ignored by `rg` through `.rgignore` to keep default searches useful. Use `rg --no-ignore` only when historical artifacts are explicitly needed.

## Current Non-Negotiables

- Do not resume the old PPO micro-engineering loop.
- Do not treat evolutionary/search/scripted trajectories as PPO success.
- Do not treat raw GPU winners as trusted without CPU postcheck/rerank.
- Do use CPU `MonzaSim` replay/eval as the promotion oracle.
- Do keep replay, telemetry, and selected artifacts explicit.
- Do keep root docs small.
- Do not delete active, unanalyzed, or unverified large runs.
- Treat `action_search.py` and `elite_search.py` as legacy diagnostics, not the main path.

## Work Loop

Operate autonomously:

1. Inspect.
2. Form a short plan.
3. Implement.
4. Validate.
5. Fix failures.
6. Update `Documentation.md` when status or workflow changes.
7. Repeat until the task is actually handled.

If validation fails, fix it before moving on.

## Tech Expectations

- Python `>=3.11,<3.13`.
- Use `uv`.
- Keep rendering and training/search decoupled.
- Keep selected telemetry replay-compatible.
- Preserve CPU/GPU verification paths when changing search or GPU code.

## Common Validation

Use the smallest validation that proves the change, then broaden when needed.

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

For evolutionary search changes, also run a tiny CLI smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

For replay changes, run a headless replay load check against selected telemetry.

## Documentation Rule

`Documentation.md` is a concise live status file. Do not turn it back into a giant transcript.

Archive long logs, old plans, and historical context under `archive/`.
