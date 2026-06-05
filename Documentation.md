# Documentation

Last updated: 2026-06-05.

This is the concise live status file. Historical prompts, long plans, and transcript-derived notes are archived under `archive/`.

## Scoreboard

Best current evolved result:

- Run: `C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605`
- Scale: `2000 x 150 = 300,000` candidates
- Backend: GPU fused evolutionary search
- Max steps: `25000`
- Target termination: disabled
- Fastest selected valid lap: `81.233s`
- CPU-verified selected candidate: generation `143`, candidate `1864`
- CPU verification reason: `lap_complete`
- Reason mismatches: `0`
- Valid-lap mismatches: `0`
- Max final-progress delta in selected CPU verification: about `0.0306m`
- Replay telemetry: `C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top`

Previous milestone:

- CPU evolutionary search reached about `89s`.
- The README keeps both the 89s GIF and the newer 81.2s GIF.

PPO status:

- CPU/Gym/SB3 PPO exists and works mechanically.
- GPU PPO exists as a separate experimental path.
- No PPO policy is currently the headline result.
- The old blind PPO loop should not be resumed as the primary path.

## Current Architecture

Core runtime:

`track assets -> track geometry -> simulator -> telemetry -> replay/eval -> learning/search`

Main modules:

- `src/f1rl/sim.py`: CPU simulator and promotion oracle.
- `src/f1rl/env.py`: Gymnasium wrapper.
- `src/f1rl/train.py`: SB3 PPO training path.
- `src/f1rl/gpu_ppo.py`: custom GPU PPO experiments.
- `src/f1rl/evolution_search.py`: CPU/GPU evolutionary search.
- `src/f1rl/evolution_postcheck.py`: CPU postcheck/rerank for GPU search.
- `src/f1rl/evolution_backend.py`: shared CPU/GPU evolution backend routing.
- `src/f1rl/gpu_batch.py`: batched GPU simulation.
- `src/f1rl/gpu_fused_warp.py`: Warp fused GPU kernels.
- `src/f1rl/replay.py`: visual/headless replay.
- `src/f1rl/telemetry.py`: telemetry schema and summaries.

Legacy diagnostics:

- `src/f1rl/action_search.py`
- `src/f1rl/elite_search.py`

They remain available, but they are not the main search path.

## Current Method

The practical loop is:

1. Run GPU evolutionary search in speed-first production mode.
2. Keep search-time telemetry compact.
3. CPU-rerank/postcheck selected candidates.
4. Generate replayable selected telemetry only for verified winners.
5. Inspect telemetry and replay.
6. Use verified ES data for the next learned-policy step.

Raw GPU winners are proposal data. CPU-verified selected winners are trusted promotion data.

## GPU ES Contract

The production GPU search path uses:

- `--backend gpu`
- `--gpu-engine fused`
- `--gpu-run-mode production`
- `--gpu-device cuda`
- `--gpu-dtype float32`
- `--gpu-fast-geometry local_window`
- `--gpu-collision-mode exact_grid`
- `--gpu-cpu-replay-top-k 0` for raw speed-first search
- deferred `f1rl.evolution_postcheck` for CPU verification/rerank

Deferred postcheck supports:

- candidate pool selection;
- CPU rerank;
- duplicate replay-key caching;
- parallel compact CPU pool replay;
- incremental pool progress files;
- full telemetry only for final selected winners.

Recent verified postcheck examples:

- `C:\f1rl-artifacts\gpu-speed-deferred-rerank-1000x75-25k-20260605`
  - candidate pool: `96`
  - unique CPU replays: `16`
  - cache hits / duplicate skips: `80`
  - selected winners: `8`
  - final selected parity: passed
  - reason mismatches: `0`
  - valid-lap mismatches: `0`

- `C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605`
  - selected summary-top CPU verification: passed
  - selected winner: generation `143`, candidate `1864`
  - selected replay telemetry written under `selected_telemetry_summary_top`

## Current RL Direction

The next learned-policy path should use the ES result instead of ignoring it:

1. Export a broad verified transition dataset from ES telemetry.
2. Include fast valid laps, consistent valid laps, near-valid failures, section specialists, and diverse lineages.
3. Train behavior cloning from observations to controls.
4. Fine-tune with off-policy RL, likely SAC or TD3.
5. CPU-evaluate learned policies before promotion.
6. Optionally inject the learned actor back into ES as a smart candidate source.

Do not train from only the single fastest lap. The useful dataset is a curated library of verified behavior and near-miss behavior.

Detailed goal docs:

- `docs/CurrentPhysicsLearnedPolicyPlan.md`: current physics v1 ES data -> transition dataset -> BC -> SAC -> learned policy.
- `docs/PhysicsV2LearnedPolicyPlan.md`: FastF1-calibrated physics v2 -> GPU ES v2 -> v2 dataset -> SAC learned policy v2.

## Important Commands

Replay the current best selected telemetry:

```powershell
uv run --no-sync python -m f1rl.replay "C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top"
```

Headless replay smoke:

```powershell
uv run --no-sync python -m f1rl.replay "C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top" --headless --limit 1
```

CPU evolution smoke:

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

Deferred CPU postcheck example:

```powershell
uv run --no-sync python -m f1rl.evolution_postcheck C:\f1rl-artifacts\example-gpu-es --top-k 8 --candidate-pool-size 96 --cpu-rerank --workers 8 --telemetry-compression gzip
```

## Validation

Standard full checks:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

Before launching large search, also run a small CLI smoke and a replay-load check for any selected telemetry that will be used in docs or demos.

## Storage And Artifacts

Keep root clean:

- Do not place large run artifacts in repo root.
- Put local experiment outputs in `artifacts/` or `C:\f1rl-artifacts`.
- Keep only small demo media in root when directly referenced by README.
- Archive old root documents under `archive/`.

The current large run folder is about `19.86 GB`:

`C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605`

The selected replay telemetry for the best candidate is small and replayable:

`C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top`

## Root Documentation Policy

Root markdown is intentionally limited to:

- `README.md`
- `Documentation.md`
- `AGENTS.md`

Archived root markdown from the previous planning phase lives under:

`archive/repo-cleanup-20260605/docs/`

Archived root media that is no longer referenced by README lives under:

`archive/repo-cleanup-20260605/media/`

Do not restore long transcript-style status logs to `Documentation.md`.
