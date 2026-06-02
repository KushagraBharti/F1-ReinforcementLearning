# F1RL Current Plan

## Current Status

The simplified Monza simulator and reinforcement-learning proof of concept is implemented and verified end to end.

Completed:

- Legacy implementation archived under `archive/legacy-20260424/`.
- Active runtime rebuilt around `track geometry -> physics -> simulator -> manual/scripted -> telemetry -> Gymnasium -> PPO -> eval/replay`.
- Track artifact generated at `assets/tracks/monza/track_spec.npz`.
- Manual driving works with keyboard controls, collision/off-track termination, ray sensors, HUD, reference ghost overlay, and flying-start comparison.
- Replay works with timestamp-based playback, `--speed`, and `--no-timing`.
- Fast-F1 reference ghost works as a perfect telemetry replay baseline.
- Scripted pure-pursuit baseline completes a valid slow lap.
- Gymnasium env and Stable-Baselines3 PPO training/eval work.
- Curriculum/segment spawning works.
- Benchmark harness writes JSON, CSV, per-episode JSONL, and selected telemetry.
- QC tool writes JSON/Markdown reports, HTML dashboard, and manual checklist.
- CUDA-required PPO smoke training works on the RTX 4060 Laptop GPU.
- TensorBoard logs load and render scalar curves.
- Lint/type/test validation passes.

Current proof-of-concept metrics:

- Observation dimension: `18`
- Discrete actions: `9`
- Sensor rays: `7`
- Checkpoints: `120`
- Boundary segments: `1800`
- Simulator step: `60 Hz`
- Fast-F1 reference ghost: `79.662s`, `259.9 kph` average, `348.0 kph` max
- Scripted baseline: valid lap, `214.47s`, `96.0 kph` average, `107.9 kph` max
- PPO scratch initial: `0.0m`, terminates by no-progress
- PPO final checkpoint: `431.4m`, passes `8` checkpoints, terminates off-track
- PPO best checkpoint: `797.6m`, passes `16` checkpoints, terminates collision
- CUDA smoke: `2048` timesteps, `device=cuda`, `vec_env=subproc`, `51.15` training FPS, `102.31` env steps/sec

Current honest gap:

- PPO has a clear learning signal, but no PPO policy has completed a clean or valid full lap yet.

## Active Goal

Train and evaluate PPO until the project has a stronger RL result:

1. Preferably a PPO clean/valid lap from strict normal start.
2. At minimum, a much stronger proof of learning than the current `797.6m` best checkpoint, backed by benchmark summaries, telemetry, TensorBoard curves, and replayable artifacts.

Do not overclaim. If PPO still fails after serious training, document the failure mode precisely and preserve the evidence.

## Non-Goals

Do not implement these until the serious PPO path is exhausted or the user explicitly starts that phase:

- evolutionary search
- imitation learning
- offline RL
- image observations
- continuous action RL
- distributed training
- multi-track support
- advanced race-control/anti-cutting system
- world-record optimization

## Compute Policy

Keep the existing strict placement:

- CPU:
  - simulator stepping
  - physics
  - geometry
  - rendering
  - keyboard input
  - telemetry/logging
  - track preprocessing
  - vector env workers
- GPU:
  - PyTorch PPO policy training
  - PyTorch policy inference when requested/available

Important caveat:

- Stable-Baselines3 PPO with `MlpPolicy` and small numeric observations may not saturate the GPU.
- Low GPU utilization is acceptable if `--require-gpu` resolves model training/inference to CUDA.
- Do not move simulator/raycast/render/telemetry code to GPU unless a future measured batched simulator justifies it.

## Immediate Next Milestone: Serious PPO Training

### Step 1: Sanity Validation

Run before any long training:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync python -m f1rl.hardware --json
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Expected:

- ruff passes
- pyright passes
- pytest passes
- CUDA is visible on the RTX 4060
- scripted baseline completes `lap_complete`

### Step 2: Establish Baselines

Use explicit artifact paths. Do not use `latest` for reported results.

Random/scripted/reference:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random --episodes 20 --max-steps 3600 --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
```

Current PPO baseline checkpoints:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Record:

- completion rate
- valid lap rate
- average/best progress
- checkpoints passed
- reward
- termination reasons
- selected telemetry paths

### Step 3: Overnight Curriculum PPO

Primary command:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

If stable and promising, scale:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 3000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 100000 --eval-every 100000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-overnight-scratch
```

Required outputs:

- `run_metadata.json`
- `initial_model.zip`
- `final_model.zip`
- `best_model.zip` when available
- checkpoint files
- TensorBoard event file
- `eval/eval_metrics.jsonl`
- `eval/best_eval_summary.json`
- selected telemetry for representative evals

### Step 4: Full-Lap Fine-Tuning

If curriculum improves segments but strict normal-start eval remains weak, fine-tune from the best curriculum checkpoint on normal full-lap starts.

Use this only after selecting the strongest curriculum checkpoint:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-full-lap-finetune
```

If resume-from-checkpoint support is missing or insufficient, implement it explicitly before this step. Do not pretend a fresh full-lap run is fine-tuning.

### Step 5: Final Benchmark Matrix

For each serious run, benchmark:

- random
- scripted
- reference ghost
- PPO initial
- PPO final
- PPO best
- any fine-tuned checkpoint

Use at least `20` episodes for PPO policies.

Report:

- training timesteps
- wall-clock training time
- device and vector env backend
- completion rate
- valid lap rate
- finish crossed rate
- crash/off-track/no-progress rates
- average progress
- best progress
- average checkpoints passed
- average reward
- best lap time if a valid PPO lap exists
- selected replay path
- selected failure path

## Visual And Human Review

Manual visual checks remain useful but should not replace metrics.

Use replay commands for representative policies:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl" --speed 2
uv run --no-sync python -m f1rl.replay "artifacts\<benchmark-run>\selected_telemetry\ppo-episode-000-steps.jsonl" --no-timing
```

Check:

- car orientation
- speed plausibility
- rays before crashes
- off-track/collision reason
- whether steering/braking behavior is improving
- whether the replay matches benchmark JSON

## Performance Polish

Low manual FPS is a usability issue, not an RL blocker.

Optimize without deleting useful visuals:

1. Skip `pygame.surfarray.array3d` during human rendering unless an RGB frame is explicitly requested.
2. Cache static track/background surfaces.
3. Cache car sprite rotations.
4. Buffer manual telemetry writes.
5. Add optional smaller-window/render-scale control.
6. Profile render buckets before larger changes.

Do not change physics while fixing FPS.

## Results And Recruiting Outputs

After serious PPO training:

1. Generate learning curves:
   - reward over timesteps
   - progress over timesteps
   - segment completion rate over timesteps
   - crash/off-track/no-progress rates
2. Generate benchmark comparison table.
3. Export or record replay video clips:
   - scripted clean lap
   - reference ghost
   - best PPO success or best PPO failure
4. Update `README.md` with a polished Results section.
5. Update `Documentation.md` with exact commands, artifacts, and metrics.
6. Keep resume claims honest:
   - claim clean PPO laps only if benchmark data proves them
   - otherwise claim simulator/RL pipeline plus measured PPO learning signal

## Definition Of Done For Next Goal

The next goal is complete when:

- serious PPO training has run for at least `1M` timesteps, or a shorter run is explicitly documented due to a blocker
- all required artifacts are written
- PPO results are benchmarked against random/scripted/reference/initial PPO baselines
- TensorBoard curves are available
- selected telemetry can be replayed
- README and Documentation are updated
- validations pass

Strong success:

- PPO completes at least one strict valid lap.

Acceptable proof of progress:

- PPO does not complete a lap but significantly improves beyond `797.6m`, passes more checkpoints, survives longer, improves reward, or shows better segment-to-full-lap transfer with clear artifacts.

Blocked:

- mark blocked only if validations pass, training infrastructure works, multiple serious PPO/curriculum attempts fail to improve, and the next step requires a deeper algorithm/design decision.
