# F1 Reinforcement Learning

Top-down 2D F1-style driving simulator with:
- manual keyboard gameplay in Pygame
- Gymnasium-compatible RL environment
- RLlib PPO training/eval on PyTorch
- GPU-aware local training on Windows
- benchmark/champion workflow with sampled clip artifacts
- swarm diagnostics and stage-aware campaign tooling
- lightweight inference artifacts for eval/benchmark without full algorithm restore

## Requirements
- Windows laptop workflow is the primary target
- Python `3.12.x`
- Conda recommended
- `uv`
- NVIDIA GPU for the default local optimization flow

## Setup
```powershell
conda create -n f1rl python=3.12 -y
conda activate f1rl
pip install uv
uv sync --active --all-extras
```

If the repo lives under OneDrive, set:
```powershell
$env:UV_LINK_MODE='copy'
```
before repeated `uv run ...` commands. This avoids Windows hardlink issues on cloud-synced paths.

## Runtime Checks
Validate CUDA and a local RL smoke path:
```powershell
uv run python -m f1rl.hardware_check
```

Quick Torch check:
```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Manual Mode
```powershell
uv run python -m f1rl.manual
```

Headless scripted smoke:
```powershell
uv run python -m f1rl.manual --headless --controller scripted --max-steps 120
```

Controls:
- `W` / `Up`: throttle
- `S` / `Down`: brake / reverse
- `A` / `Left`, `D` / `Right`: steering
- `Esc`: quit

## Training
Smoke:
```powershell
uv run python -m f1rl.train --mode smoke --device gpu --require-gpu
```
Default shape: `1 env runner x 2 envs`, sync vectorization.

Stage-aware logical swarm collection:
```powershell
uv run python -m f1rl.train --mode smoke --swarm-stage competence --logical-cars 32 --device gpu --require-gpu
```
This keeps the single-car environment but scales rollout collection across a logical swarm target.

Benchmark:
```powershell
uv run python -m f1rl.train --mode benchmark --device gpu --require-gpu
```
Default shape: `2 env runners x 3 envs`, sync vectorization.

Longer local training:
```powershell
uv run python -m f1rl.train --mode performance --device gpu --require-gpu
```
Default shape: `4 env runners x 4 envs`, sync vectorization.

Override local parallelism explicitly:
```powershell
uv run python -m f1rl.train --mode smoke --num-env-runners 1 --num-envs-per-env-runner 4 --vector-mode sync
```

Candidate env override example:
```powershell
uv run python -m f1rl.train --mode benchmark --device gpu --env-config-json "{\"reward\":{\"progress_reward\":3.0}}"
```

Candidate PPO override example:
```powershell
uv run python -m f1rl.train --mode benchmark --device gpu --ppo-config-json "{\"entropy_coeff\":0.003,\"minibatch_size\":256}"
```

## Eval, Rollout, Benchmark
Checkpoint eval:
```powershell
uv run python -m f1rl.eval --checkpoint latest --headless --steps 300
```
Default backend: lightweight inference artifact.

Checkpoint rollout:
```powershell
uv run python -m f1rl.rollout --policy checkpoint --checkpoint latest --headless --steps 300
```
Default backend: lightweight inference artifact.

Benchmark a checkpoint and export sampled clips:
```powershell
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --promote-if-best
```
Default backend: lightweight inference artifact.

Quick benchmark without clip generation:
```powershell
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --no-clips
```

Promotion / diagnostic / holdout seed tiers:
```powershell
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier promotion --no-clips
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier diagnostic --rotation-index 2 --no-clips
uv run python -m f1rl.benchmark --checkpoint latest --profile quick --seed-tier holdout --no-clips
```

Compatibility mode if you explicitly need full RLlib restore:
```powershell
uv run python -m f1rl.eval --checkpoint latest --headless --legacy-algorithm-restore
```

Run the older recursive candidate loop:
```powershell
uv run python -m f1rl.optimize --max-candidates 2 --iterations 3 --device gpu --profile quick --no-clips
```

## Swarm Diagnostics
Random swarm smoke:
```powershell
uv run python -m f1rl.swarm --policy random --headless --cars 8 --steps 40
```

Checkpoint-backed swarm diagnostics with clips:
```powershell
uv run python -m f1rl.swarm --policy checkpoint --checkpoint latest --cars 32 --swarm-stage competence --headless --export-clips
```

Swarm mode uses:
- one shared policy
- many independent cars
- no car-car collision
- same track and spawn semantics
- lightweight Trackmania-style visualization for diagnostics

## Autonomous Campaign
Run a bounded overnight campaign:
```powershell
uv run python -m f1rl.campaign --hours 7 --device gpu
```

Small bounded campaign smoke:
```powershell
uv run python -m f1rl.campaign --hours 0.02 --max-waves 1 --start-cars 8 --max-cars 8 --device auto
```

Campaign behavior:
- stage ladder: competence -> lap -> stability -> performance
- one subsystem family per wave
- fixed promotion seeds + rotating diagnostic seeds + periodic holdout seeds
- single-car benchmark remains the promotion authority
- swarm traces are diagnostics only
- per-candidate training throughput is recorded from the training run artifacts

## Validation
```powershell
uv run ruff check .
uv run pyright src/f1rl
uv run pytest -q
uv run python -m f1rl.validate
```

## Artifacts
- `artifacts/train-*/`
  checkpoints, lightweight inference artifact, metrics, logs, run metadata
- `artifacts/swarm-*/`
  `swarm_summary.json`, optional clips, top-k trace payload
- `artifacts/campaign-*/`
  `campaign_manifest.json` with wave-by-wave keep/reject decisions
- `artifacts/track-cache/`
  cached preprocessed track geometry used to reduce worker startup cost on repeated runs
- `artifacts/benchmark-*/`
  `benchmark_summary.json` and sampled GIF clips
- `artifacts/champions/current.json`
  current promoted benchmark summary
- `artifacts/optimize-*/optimize_manifest.json`
  candidate loop keep/reject history
- `artifacts/renders/`
  manual/eval/rollout exported frames

Protected runs are marked with `.pin`.

## Architecture
```text
src/f1rl/
  env.py            # environment, reward logic, termination rules
  dynamics.py       # kinematic bicycle-style dynamics and observations
  track.py          # track extraction, goals, centerline metadata
  train.py          # PPO train profiles with GPU-aware config
  benchmark.py      # benchmark metrics, clips, champion promotion
  optimize.py       # candidate loop orchestration
  hardware_check.py # CUDA/runtime validation
```

## Notes
- The env now includes richer observations than raw wall rays alone: heading error, goal distance, centerline offset, and previous action signals.
- Reverse is still supported in the simulator, but RL reward/termination now make sustained reverse behavior non-competitive.
- Stage-aware early termination now exists for:
  - checkpoint deadlines
  - heading-away streaks
  - wall-hugging streaks
- The benchmark decision rule is fixed:
  1. higher completion rate
  2. lower lap time
  3. lower collision / instability
- Eval and benchmark no longer restore full `Algorithm` objects by default, which removes the old logger deprecation cascade from those commands.
- RLlib still emits internal deprecation warnings from Ray 2.54.x internals during training, and RLModule construction still warns once during lightweight inference in this Ray build.
- The benchmark profile is materially heavier than smoke on Windows; smoke is the stable default for fast iteration.
- Campaign-scale runs on Windows are materially heavier than the old smoke loop. The orchestration is implemented, but short “smoke” budgets are dominated by RLlib startup and are not representative of overnight throughput.
