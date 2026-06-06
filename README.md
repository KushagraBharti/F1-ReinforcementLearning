# F1 Reinforcement Learning

A top-down 2D Monza driving simulator and learning project. The repo explores how far a simple racing environment can be pushed with physics calibration, GPU evolutionary search, replayable telemetry, and learned neural driving policies.

The project started as a PPO reinforcement-learning experiment. The final strongest path became: search for fast laps at large scale, verify the best candidates in the CPU simulator, turn those verified laps into training data, and train a learned policy that can reproduce the fast driving line.

<p align="center">
  <strong>1) Physics 1 GPU ES</strong><br />
  <img src="./artifacts/highlights/physics1/gpu-es/gifs/gpu-es-2000x150-all6-stratified150.gif" alt="Physics 1 GPU ES replay reel" width="900" />
</p>

<p align="center">
  <strong>2) Physics 2 GPU RL</strong><br />
  <img src="./artifacts/highlights/physics2/gpu-rl/gifs/learned-policy-promotion-4x.gif" alt="Physics 2 GPU RL learned policy replay" width="900" />
</p>

## What This Shows

The first replay shows the earlier Physics 1 evolutionary-search stage. Many candidate cars are evaluated at once, and the best behaviors gradually move from rough progress around the track toward fast, valid Monza laps.

The second replay shows the final Physics 2 learned policy. Physics 2 was recalibrated against real FastF1 Monza telemetry, then GPU evolutionary search found strong trajectories, and a neural policy was trained from CPU-verified data.

## Final Result

The final promoted Physics 2 learned policy completed a valid CPU-verified Monza lap in `77.6833s`. The target reference was the selected FastF1 Monza lap threshold of `79.327s`, so the learned policy beat the target under the project simulator's CPU oracle.

Earlier milestones are preserved too: Physics 1 CPU evolutionary search reached roughly `89s`, Physics 1 GPU evolutionary search reached `81.233s`, and the first Physics 1 learned policy reached `79.750s`.

## How It Works

The simulator stores Monza track geometry, physics parameters, car state, observations, controls, telemetry, and replay data explicitly. That makes each run inspectable instead of being only a training log.

Evolutionary search runs many candidate controllers and keeps the best ones. GPU search provides scale, but raw GPU winners are not trusted by default; selected winners are replayed through the CPU simulator for final verification.

The learned-policy pipeline uses verified search trajectories as demonstrations. Behavior cloning gives the policy a strong starting point, and the project-native SAC workflow fine-tunes it while CPU evaluation remains the promotion gate.

Physics 2 adds a more realistic calibration pass. FastF1 telemetry, manual driving checks, CPU/GPU parity tests, and replay validation were used to tune the simulator before running the final search and learned-policy workflow.

## Repository Map

- `src/f1rl/`: simulator, physics, learning, search, replay, telemetry, and CLI code.
- `tools/`: utility scripts and moved test suite under `tools/tests/`.
- `assets/`: track images, reference telemetry, track assets, and README GIFs.
- `artifacts/`: curated replay telemetry, highlight GIFs, manifests, and final local artifacts.
- `archive/docs/`: detailed historical plans and achievement reports.
- `Documentation.md`: technical status, commands, validation notes, storage layout, and exact artifact paths.

## Try It

Install dependencies:

```powershell
uv sync --active --all-extras --all-packages
```

Replay the final learned policy:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion" --speed 4
```

Run manual driving:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
```

Run the standard validation suite:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

For the full technical record, read `Documentation.md`, `archive/docs/RL1-Achieved.md`, and `archive/docs/PhysicsV2-Achieved.md`.
