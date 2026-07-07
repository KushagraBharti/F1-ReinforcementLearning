# F1 Reinforcement Learning

A Formula 1 racing-AI lab built around Monza. Everything under the learning algorithms is project-built: the 2D simulator, the vehicle physics, the telemetry and replay stack, and the calibration against real F1 data. On top of that sits CUDA evolutionary search that evaluated 300,000 controller candidates, and a learned neural policy that laps Monza in **77.683s** against a **79.327s** benchmark taken from Lando Norris's 2024 Italian GP qualifying lap.

<p align="center">
  <img src="./artifacts/highlights/physics2/gpu-rl/gifs/learned-policy-promotion-4x.gif" alt="Promoted learned policy lapping Monza at 4x speed" width="900" />
  <br />
  <em>The promoted learned policy. One valid lap, 77.683s, 270.8 kph across the line, rendered from the exact replay telemetry.</em>
</p>

## The lap-time story

Every number below is CPU-verified: the lap was replayed through the deterministic CPU simulator before it counted.

| Stage | Lap time |
|---|---:|
| Physics V1, CPU evolutionary search | ~89s |
| Physics V1, GPU evolutionary search | 81.233s |
| Physics V1, learned policy | 79.750s |
| Physics V2, GPU evolutionary search | 77.683s |
| Physics V2, learned policy (promoted) | **77.683s** |

The final policy beats the 79.327s FastF1 threshold by 1.6 seconds inside a simulator that was calibrated to make that threshold mean something.

One thing I want to be upfront about, because the docs are upfront about it: the promoted checkpoint is a behavior-cloned reproduction of a single search-discovered lap, carried through the SAC workflow and promoted at step 0, evaluated deterministically from a normal start. It is imitation of a search-discovered trajectory, not converged reinforcement learning. `docs/RL2-TrueRL-Plan.md` defines the bar a future from-scratch RL result has to clear before I call it RL.

<p align="center">
  <img src="./artifacts/highlights/physics1/gpu-es/gifs/gpu-es-2000x150-all6-stratified150.gif" alt="Physics V1 GPU evolutionary search replay reel with 150 stratified candidates" width="900" />
  <br />
  <em>Earlier in the ladder: Physics V1 GPU evolutionary search, 150 stratified candidates from a 2,000 x 150 run, from rough progress to full laps.</em>
</p>

## How it works

The pipeline: FastF1-calibrated physics grounds the simulator, CUDA evolutionary search discovers fast laps, CPU reranking verifies the candidates, behavior cloning captures the winning trajectory as a policy, and a project-native PyTorch SAC workflow promotes the final checkpoint.

### The simulator

A Gymnasium-compatible Monza environment (5,793 m lap, 60 Hz timestep) over a shared simulator core, with discrete, continuous, and multidiscrete control modes, checkpoint-validated laps, SB3 hooks, benchmark tooling, and full replay support. The package spans 55 modules with 27 console entry points. Track geometry, physics parameters, car state, observations, controls, and telemetry are all stored explicitly, so any run can be replayed and inspected later.

### Physics V2

The current physics model covers:

- Tire slip-angle behavior with force saturation
- Weight transfer under braking, acceleration, and cornering
- An 8-gear torque curve on a 798 kg car
- Brake-bias instability, so late braking can actually lose the rear
- Distinct asphalt, curb, grass, and wall surface behavior
- Oriented car-body collision checks

Curb use and track limits carry real tradeoffs. V2 is an explicit opt-in (`physics_model=v2`) with the physics version and calibration ID stamped into every artifact, so V1 and V2 results never silently mix benchmark categories.

### Calibration against real telemetry

V2 is calibrated against 60 clean dry FastF1 Monza laps (2022 to 2024, seven drivers), with a 42-lap OpenF1 cross-check as an independent sanity path. The benchmark threshold of 79.327s comes from Norris's 2024 Italian GP qualifying lap 11. Calibration compares terminal speeds, acceleration traces, braking-zone distances, corner lateral-g margins, and gear/RPM behavior between the simulator and the reference laps. Every claim in this repo stays scoped to a calibrated top-down 2D simulator.

### GPU evolutionary search

The discovery engine is CUDA-scale evolutionary search. Large controller populations are scored on full-lap behavior; elites survive, and mutation plus crossover generate the next population, so each generation gets faster at completing valid laps. The V2 campaign ran staged from 1000x5 through 1000x50, and the largest run used a fused Warp kernel at 2,000 population x 150 generations x 25k steps, which comes out to 300,000 controller candidates evaluated.

<p align="center">
  <img src="./artifacts/highlights/physics2/gpu-es/gifs/gpu-es-cpu-rerank-best-4x.gif" alt="Best CPU-reranked V2 GPU search candidate at 4x speed" width="900" />
  <br />
  <em>The CPU-reranked V2 search winner: generation 46, candidate 724, verified at 77.683s.</em>
</p>

### GPU proposes, CPU verifies

GPU search is a proposal generator, not an oracle. The trusted path is GPU proposals, then CPU postcheck and rerank, then selected telemetry, then replay. The 77.683s source lap passed parity with zero reason mismatches and zero valid-lap mismatches. The final broad-pool GPU postcheck failed parity (13 reason mismatches, 12 valid-lap mismatches), which is exactly why only CPU-verified candidates get promoted. That failure is the correctness contract working.

### From search laps to a policy

- **Dataset export.** 16 CPU-replayed source candidates produced a V2-only dataset of 50,128 transitions and 8 valid laps, with a fastest raw source lap of 77.65s (training data only; the 77.683s headline lap is the parity-verified selected candidate). Physics model, calibration ID, observation profile, and action schema are recorded on every row.
- **Action representation.** The source telemetry used meaningful simultaneous throttle and brake, so the learned-policy path uses independent three-channel throttle, brake, and steer control instead of dominance heuristics.
- **Behavior cloning.** 120 epochs on CUDA, best validation action error 3.85e-5. The BC policy preserves the source behavior through CPU V2 evaluation.
- **SAC.** A project-native PyTorch SAC workflow with twin critics, target networks, replay buffers preloaded from verified search transitions, online CPU rollouts, and checkpoint selection by valid CPU eval lap. The promoted checkpoint is the BC-initialized policy at step 0; the SAC machinery is built and exercised, and CPU evaluation remains the promotion gate.

### Observations

The `racing_v2` profile is 35 dimensions: speed, heading and lateral error, lap progress, seven ray distances, lookahead heading, target speed and target-speed drops, braking-gate proximity, curvature, and section-aware features.

### PPO infrastructure

The project started as a Gymnasium/SB3 PPO experiment. That path still exists and works mechanically (vectorized environments, checkpointing, TensorBoard, curriculum and state-library starts, metadata-aware eval), and a separate experimental GPU PPO path exists too. Neither is the headline result. `docs/RL2-TrueRL-Plan.md` includes a post-mortem of why the blind PPO loop failed.

### Testing and validation

243 pytest functions across the test suite cover physics, GPU parity, calibration, telemetry, metadata contracts, replay, and the policy training paths. Ruff and pyright run clean, and a hardware check exercises CUDA and NVIDIA Warp interop. The curated highlights under `artifacts/highlights/physics2/` hold 1,006 replay traces, including a 1,000-trace deterministic policy swarm, all rendered from real replay telemetry.

## Repository map

- `src/f1rl/`: simulator, physics, learning, search, replay, telemetry, and CLI code
- `tools/`: GIF export and recovery utilities, plus the test suite under `tools/tests/`
- `assets/`: track images, reference telemetry, and archived README media
- `artifacts/`: curated replay telemetry, highlight GIFs, manifests
- `docs/`: forward plans (`PhysicsV3-Plan.md`, `RL2-TrueRL-Plan.md`)
- `archive/docs/`: achievement reports (`RL1-Achieved.md`, `PhysicsV2-Achieved.md`) and historical plans
- `Documentation.md`: live technical status, validation notes, storage layout, and exact artifact paths

## Setup and commands

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Install everything, including the optional train, GPU, calibration, and dev extras:

```powershell
uv sync --active --all-extras --all-packages
```

### Watch the results

Replay the promoted learned policy:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion" --speed 4
```

Replay the curated V2 GPU search winners, best score first:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-es\telemetry" --sort score --speed 4
```

Replay the 1,000-car policy swarm:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_swarm_1000" --by-checkpoint --speed 4
```

Export an exact-pygame GIF from any replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion\policy-eval-episode-000-steps.jsonl.gz" --export-gif "artifacts\highlights\physics2\gpu-rl\gifs\learned-policy-promotion-4x.gif" --speed 4 --gif-fps 12
```

### Drive it yourself

Manual V2 driving with the FastF1 ghost reference and a flying start:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --flying-start
```

You can also start at a specific corner section, with the ghost aligned to the matching reference time:

```powershell
uv run --no-sync python -m f1rl.manual --physics-model v2 --ghost-reference --start-section sustained_corner_01 --start-section-lead-in-m 120
```

### Run search and training

Small CPU evolution smoke (safe on any machine):

```powershell
uv run --no-sync python -m f1rl.evolution_search --output-dir artifacts\runs\evolution-smoke --start-progress-m 500 --start-speed-kph 80 --target-progress-m 510 --action-set straight --observation-profile base --max-steps 24 --population 8 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type phase --scoring-profiles max_progress,clean_exit --progress-every-generation
```

Production GPU search uses `--backend gpu --gpu-engine fused --gpu-run-mode production --gpu-device cuda`, then verifies candidates with the deferred CPU postcheck:

```powershell
uv run --no-sync python -m f1rl.evolution_postcheck artifacts\runs\example-gpu-es --top-k 8 --candidate-pool-size 96 --cpu-rerank --workers 8 --telemetry-compression gzip
```

The learned-policy chain runs through console entry points: `f1-export-es-dataset` (CPU-replayed transition datasets), `f1-train-bc` (behavior cloning), `f1-train-sac` (project-native SAC), `f1-policy-eval` (deterministic CPU promotion oracle), and `f1-policy-swarm-eval`. SB3 PPO training is `f1-train`; the experimental GPU PPO path is `f1-gpu-ppo`. Calibration tooling is `f1-fastf1-calibration` and `f1-openf1-crosscheck`. Run any of them with `--help` for options, and see `Documentation.md` for the exact commands from the final validated campaign.

### Validate

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

For the full technical record, read `Documentation.md`, then `archive/docs/RL1-Achieved.md` and `archive/docs/PhysicsV2-Achieved.md`.
