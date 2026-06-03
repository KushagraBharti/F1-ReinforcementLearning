# Implement

Run all commands from:

```powershell
cd "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning"
```

Use `uv run --no-sync ...` for repeated local commands after the environment is installed.

## Setup

```powershell
uv sync --active --all-extras --all-packages
```

If the editable project package is stale:

```powershell
uv pip install --no-deps -e .
```

The project targets Python `>=3.11,<3.13`. The validated local runtime uses Python `3.12.12`.

## Environment Checks

```powershell
uv run --no-sync python -c "import f1rl, pygame, gymnasium, torch; print('ok', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
uv run --no-sync python -m f1rl.hardware --json
```

Expected:

- imports succeed
- CUDA is visible on the RTX 4060 Laptop GPU
- hardware policy reports PyTorch training/inference on CUDA and sim/render/telemetry on CPU

## Validation

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

Latest verified result:

- `ruff`: passed
- `pyright`: `0` errors
- `pytest`: passed; known benign SB3 `VecMonitor` warning in PPO smoke tests
- Latest full gate after failure-first QC work:
  - `uv run --no-sync ruff check .` -> passed
  - `uv run --no-sync pyright src/f1rl` -> `0` errors
  - `uv run --no-sync pytest -q` -> passed; known benign SB3 `VecMonitor` warning only

## Track Build

```powershell
uv run --no-sync python -m f1rl.track_build
```

Outputs:

- `assets/tracks/monza/track_spec.npz`
- `assets/tracks/monza/track_manifest.json`

## Manual Mode

```powershell
uv run --no-sync python -m f1rl.manual
uv run --no-sync python -m f1rl.manual --ghost-reference
uv run --no-sync python -m f1rl.manual --ghost-reference --flying-start
```

Controls:

- `W` / Up: throttle
- `S` / Down: brake
- `A` / Left: steer left
- `D` / Right: steer right
- `R`: reset
- `Esc`: quit

Headless smoke:

```powershell
uv run --no-sync python -m f1rl.manual --headless --max-steps 60
```

Manual mode currently works visually. Known issue: FPS can be improved by optimizing render frame capture, sprite caching, static surface caching, and telemetry buffering.

## Scripted Baseline

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000 --no-telemetry
```

Expected:

- `reason=lap_complete`
- valid slow lap around `214.47s`

Generate telemetry and replay:

```powershell
uv run --no-sync python -m f1rl.scripted --steps 18000
uv run --no-sync python -m f1rl.replay "artifacts\<scripted-run>\steps.jsonl"
```

## State Libraries

Generate a successful-state library from the scripted valid lap:

```powershell
uv run --no-sync python -m f1rl.state_library --source scripted --steps 18000 --sample-every-m 100 --sample-every-steps 0 --output artifacts\state-library-scripted-full-m4-20260602\state_library.json
```

Generate from existing telemetry, including manual, reference, PPO benchmark, or selected eval JSONL files/folders:

```powershell
uv run --no-sync python -m f1rl.state_library --source telemetry --telemetry "artifacts\<run>\selected_telemetry" --sample-every-m 100 --output "artifacts\<state-library-run>\state_library.json"
```

Use a state library during curriculum training:

```powershell
uv run --no-sync python -m f1rl.train --curriculum segments --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --curriculum-state-library-segment-length-m 900 --timesteps 2048 --n-envs 2 --max-steps 900 --device auto --checkpoint-every 1024 --run-name state-library-smoke
```

Use the chicane-skill curriculum with state-library starts:

```powershell
uv run --no-sync python -m f1rl.train --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 2 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --observation-profile racing --timesteps 2048 --n-envs 2 --max-steps 900 --device auto --checkpoint-every 1024 --eval-every 1024 --eval-episodes 2 --run-name chicane-skill-smoke
```

Add temporary scaffold rewards for section training:

```powershell
uv run --no-sync python -m f1rl.train --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --observation-profile racing --reward-scaffold-brake-reward-scale 0.4 --reward-scaffold-no-throttle-penalty-scale 0.6 --reward-scaffold-turn-in-speed-penalty-scale 0.2 --reward-scaffold-apex-clean-reward-scale 0.05 --reward-scaffold-exit-alignment-reward-scale 0.05 --reward-scaffold-exit-speed-reward-scale 0.05 --reward-scaffold-final-scale 0.0 --reward-scaffold-schedule-timesteps 200000 --timesteps 200000 --n-envs 8 --max-steps 900 --device auto --checkpoint-every 20000 --eval-every 20000 --eval-episodes 3 --run-name scaffolded-chicane-training
```

Honest scaffold-free evaluation:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\<scaffolded-run>" --episodes 20 --max-steps 5000 --device auto --telemetry selected --metadata-mode require --disable-scaffold-rewards
```

Add training-only forced exploration gates:

```powershell
uv run --no-sync python -m f1rl.train --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --observation-profile racing --assist-enabled --assist-overspeed-turn-in-terminate --assist-overspeed-turn-in-margin-kph 20 --assist-throttle-brake-demand-penalty-scale 0.8 --assist-no-brake-penalty -8 --timesteps 200000 --n-envs 8 --max-steps 900 --device auto --checkpoint-every 20000 --eval-every 20000 --eval-episodes 3 --run-name assisted-chicane-training
```

Honest assist-free evaluation:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\<assisted-run>" --episodes 20 --max-steps 5000 --device auto --telemetry selected --metadata-mode require --disable-training-assists
```

Run segment elite search and create an elite state library:

```powershell
uv run --no-sync python -m f1rl.elite_search --state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --output-dir artifacts\elite-search-roggia-m8-20260602 --attempts 20 --max-steps 600 --top-k 5 --segment-length-m 800 --start-min-progress-m 1850 --start-max-progress-m 2020 --policy scripted --action-noise 0.05 --seed 80
```

Inspect elite attempts with QC:

```powershell
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\elite-search-roggia-m8-20260602\selected_telemetry" --max-telemetry-files 5
```

State-library starts are training/curriculum starts only. Reported final PPO progress still needs honest normal-start benchmark/eval unless a run is explicitly labeled as section diagnostics.

## Fast-F1 Reference Ghost

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode ghost
```

Expected:

- target lap: `79.662s`
- average speed: `259.9 kph`
- max speed: `348.0 kph`

Replay:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\<reference-ghost-run>\steps.jsonl"
```

Diagnostic control chase:

```powershell
uv run --no-sync python -m f1rl.reference_agent --mode control --steps 7200
```

## Benchmark

Reference baselines:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies random --episodes 20 --max-steps 3600 --telemetry selected --telemetry-every 5
uv run --no-sync python -m f1rl.benchmark --policies scripted --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
uv run --no-sync python -m f1rl.benchmark --policies reference_ghost --episodes 1 --max-steps 18000 --telemetry selected --telemetry-every 1
```

Current PPO checkpoints:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\initial_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\final_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-curriculum-balanced-scratch-20260602-085522\best_model.zip" --episodes 20 --max-steps 3600 --device auto --telemetry selected
```

Metadata-aware PPO benchmark:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514" --episodes 20 --max-steps 5000 --device auto --telemetry selected --metadata-mode require
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-brakeobs-continuous-throttlebias-norm-scratch-goal-80k-20260602-211740" --episodes 3 --max-steps 5000 --device auto --telemetry selected --metadata-mode require
```

For PPO policies, `--metadata-mode auto` is the default. It loads `run_metadata.json` from the artifact directory or checkpoint parent, restores the trained action mode/action set/observation profile/reward settings, loads `best_vecnormalize.pkl` or `vecnormalize.pkl` when present, and fails if the model observation/action space does not match the eval environment. Use `--metadata-mode ignore` only for intentional compatibility tests with explicit CLI flags.

PPO benchmark action selection is deterministic by default. Use stochastic rollout only as a diagnostic, not as the default promotion metric:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --episodes 4 --max-steps 1000 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --ppo-stochastic
```

Current diagnostic result:

- `artifacts\benchmark-20260603-011203`
- `0/4` valid laps
- average progress `73.74m`
- best progress `129.31m`
- all episodes ended by `max_steps`
- decision: stochastic inference is worse than the deterministic robust checkpoint and is not promoted.

Artifacts:

- `summary.json`
- `summary.csv`
- `per_episode.jsonl`
- selected telemetry JSONL

## Replay PPO Telemetry

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154222\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154253\selected_telemetry\ppo-episode-000-steps.jsonl"
uv run --no-sync python -m f1rl.replay "artifacts\benchmark-20260602-154555\selected_telemetry\ppo-episode-000-steps.jsonl"
```

Expected:

- initial PPO: no movement
- final PPO: drives straight, exits track
- best PPO: drives straight, begins correcting, collides later

## Train PPO

CUDA smoke:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 2048 --n-envs 2 --max-steps 600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 1024 --eval-every 1024 --eval-episodes 1 --telemetry selected --run-name manual-qc-smoke
```

Serious curriculum:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 1000000 --n-envs 8 --max-steps 3600 --device auto --require-gpu --vec-env subproc --curriculum segments --checkpoint-every 50000 --eval-every 50000 --eval-episodes 5 --telemetry selected --run-name ppo-curriculum-serious-scratch
```

Strategy-reset continuous PPO experiment:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 310 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-mode continuous --continuous-action-scheme throttle_bias --normalize-reward --curriculum segments --curriculum-promotion-resets 500 --curriculum-normal-start-probability 0.20 --reward-lateral-penalty-scale 0.003 --reward-track-limit-penalty-scale 0.003 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 85 --reward-speed-target-max-kph 320 --reward-speed-target-heading-scale 3.0 --reward-speed-target-deadzone-kph 20 --reward-speed-target-penalty-scale 0.0015 --n-steps 1024 --batch-size 512 --n-epochs 5 --learning-rate 0.0001 --gamma 0.997 --ent-coef 0.004 --use-sde --sde-sample-freq 16 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-continuous-throttlebias-norm-scratch-goal-120k
```

Key options:

- `--action-mode continuous`: use Gymnasium `Box([-1, 1], shape=(2,))`.
- `--continuous-action-scheme throttle_bias`: policy mean zero maps to forward drive instead of no-progress coast.
- `--continuous-action-scheme exclusive_throttle_bias`: policy mean zero maps to a small launch throttle, positive drive maps to throttle only, and negative drive maps to brake only. Use this when continuous PPO is holding throttle during brake-demand zones.
- `--observation-profile racing`: use the Milestone 3 profile with signed lateral error, previous throttle/brake, target-speed lookaheads, brake/guidance features, and distance to the next braking gate.
- `--initialize-from-checkpoint PATH`: initialize a fresh PPO model from a compatible checkpoint without requiring the same observation shape. This is useful for expanding a trained `base` policy into `racing` observations; matching tensors are copied and first-layer observation inputs are expanded with new columns initialized to zero.
- `--initialize-from-checkpoint PATH` also expands discrete PPO action heads when moving from `legacy` to `racing` or `expanded` action sets. Exact action rows are copied; new soft/half actions are initialized from the nearest source action with an initial bias penalty so the richer action set starts from the old deterministic behavior instead of random logits.
- `--curriculum-state-library PATH`: sample exact saved simulator states during segment curriculum; metadata records the path and snapshot count.
- `--curriculum-preset chicane-skill --curriculum-chicane rettifilo|roggia`: use approach/brake, turn-in, apex, exit, post-exit, and full-chicane section stages.
- `--reward-scaffold-*`: enable zero-default temporary brake/exit teaching rewards; metadata records whether scaffold rewards were active.
- `--reward-scaffold-final-scale --reward-scaffold-schedule-timesteps`: linearly reduce scaffold reward strength during training.
- `--disable-scaffold-rewards` on eval/benchmark: zero training-only scaffold rewards for honest scoring.
- `--assist-*`: enable explicit training-only gates. Metadata records `training_assists_enabled` and full `assist_config`.
- `--disable-training-assists` on eval/benchmark: zero training-only gates for honest scoring.
- `python -m f1rl.elite_search`: run short section attempts and emit `elite_state_library.json` for later curriculum sampling.
- `--normalize-reward`: enable SB3 `VecNormalize` reward normalization with observation normalization disabled.
- `--use-sde --sde-sample-freq 16`: use gSDE exploration for continuous control.

Training writes `vecnormalize.pkl` when reward normalization is enabled. Keep that file with the matching checkpoint if continuing normalized training.

Fast-stage discrete continuation:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 120000 --seed 410 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --action-mode discrete --action-set legacy --normalize-reward --curriculum segments --curriculum-start-stage-index 3 --curriculum-stage-count 4 --curriculum-promotion-resets 120 --curriculum-normal-start-probability 0.35 --resume-checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --reward-lateral-penalty-scale 0.003 --reward-track-limit-penalty-scale 0.003 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.0015 --reward-speed-target-min-kph 90 --reward-speed-target-max-kph 330 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 20 --reward-speed-target-penalty-scale 0.0015 --n-steps 512 --batch-size 256 --n-epochs 4 --learning-rate 0.00002 --gamma 0.997 --ent-coef 0.002 --checkpoint-every 10000 --eval-every 10000 --eval-episodes 1 --telemetry selected --telemetry-every 1 --run-name ppo-faststage-discrete-norm-best966-resume-goal-120k
```

Key options:

- `--curriculum-start-stage-index 3 --curriculum-stage-count 4`: skip low-speed A0/A1/B and train on `C-long` through `F-normal-lap`.
- `--curriculum-normal-start-probability 0.35`: keep direct pressure on normal-start full-lap behavior.
- `--normalize-reward`: use reward normalization for the continuation; keep `vecnormalize.pkl` with any promoted checkpoint.

Training writes:

- `run_metadata.json`
- `initial_model.zip`
- `final_model.zip`
- checkpoint files
- `vecnormalize.pkl` when reward normalization is active
- TensorBoard event files
- `eval/eval_metrics.jsonl`
- `eval/best_eval_summary.json`
- selected telemetry

## Evaluate

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint latest --steps 600 --device auto
```

For reported metrics, prefer explicit checkpoint paths or artifact directories over `latest`.

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --steps 5000 --device auto --metadata-mode require
```

`f1rl.eval` uses the same metadata-aware PPO loading as benchmark. It reports `config_source=run_metadata` when trained-run metadata was applied and reports the VecNormalize stats path when loaded.

Eval is also deterministic by default. Use `--ppo-stochastic` only for diagnostic rollouts:

```powershell
uv run --no-sync python -m f1rl.eval --checkpoint "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\checkpoints\ppo_monza_40000_steps.zip" --steps 1000 --device auto --metadata-mode require --ppo-stochastic
```

## QC Report

```powershell
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260602-154555\selected_telemetry\ppo-episode-000-steps.jsonl" --run-scripted --scripted-steps 18000
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000\selected_telemetry" --max-telemetry-files 2 --run-scripted --scripted-steps 18000
```

Outputs:

- `qc_report.json`
- `qc_report.md`
- `telemetry_dashboard.html`
- `manual_qc_checklist.md`

QC accepts either a single telemetry JSONL file or a folder of selected telemetry files. The current failure-first report includes `telemetry_reports`, a compact `failure_table`, section summaries, action/control histograms before failure, and scripted comparison when `--run-scripted` is enabled.

Telemetry written after 2026-06-03 includes `action_name`. QC prefers that field so reports remain truthful for non-legacy discrete action sets such as `racing` and `expanded`. Older telemetry files are still supported through the legacy action-id fallback, but action labels in old QC reports should be treated cautiously when the policy did not use `action_set=legacy`.

Current trustworthy Rettifilo diagnosis command:

```powershell
uv run --no-sync python -m f1rl.benchmark --policies ppo --checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341" --episodes 1 --max-steps 1500 --seed 9800 --device auto --telemetry selected --telemetry-every 1 --metadata-mode require --disable-scaffold-rewards --disable-training-assists --ppo-deterministic
uv run --no-sync python -m f1rl.qc --telemetry "artifacts\benchmark-20260603-015016\selected_telemetry" --max-telemetry-files 1
```

Expected current failure signature:

- no valid PPO lap;
- best progress about `970.8m`, `20/120` checkpoints;
- first bad event `throttle_during_brake_demand` around `520.8m` at about `334kph`;
- actions before failure dominated by `throttle`, with only a few `brake_right` samples;
- terminal collision in `rettifilo_chicane`.

Current curriculum caveat:

- Chicane-skill segment completion is now speed-gated through `segment_target_max_speed_kph`.
- Do not compare old chicane-skill segment completion rates directly with new ones.
- Old segment completions may have included overspeed target crossings and should be treated as diagnostic only.
- New segment completion requires reaching the target progress while also being at or below the stage target max speed.

Recommended next Rettifilo experiment shape:

```powershell
uv run --no-sync python -m f1rl.train --timesteps 30000 --seed 990 --n-envs 8 --max-steps 5000 --device auto --require-gpu --vec-env subproc --initialize-from-checkpoint "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_model.zip" --vec-normalize-path "artifacts\ppo-transfer-racing-actionhead-rettifilo-mix-goal-60k-20260603-013341\best_vecnormalize.pkl" --action-mode discrete --action-set racing --observation-profile racing_v2 --curriculum segments --curriculum-preset chicane-skill --curriculum-chicane rettifilo --curriculum-stage-count 6 --curriculum-state-library "artifacts\state-library-scripted-full-m4-20260602\state_library.json" --curriculum-promotion-resets 80 --curriculum-normal-start-probability 0.45 --reward-lateral-penalty-scale 0.004 --reward-track-limit-penalty-scale 0.004 --reward-heading-deadzone-deg 8 --reward-heading-penalty-scale 0.001 --reward-speed-target-min-kph 80 --reward-speed-target-max-kph 340 --reward-speed-target-heading-scale 2.5 --reward-speed-target-deadzone-kph 18 --reward-speed-target-penalty-scale 0.001 --reward-overspeed-throttle-penalty-scale 0.003 --reward-overspeed-brake-reward-scale 0.001 --normalize-reward --n-steps 512 --batch-size 256 --n-epochs 2 --learning-rate 0.00001 --gamma 0.997 --ent-coef 0.002 --checkpoint-every 5000 --eval-every 5000 --eval-episodes 2 --telemetry selected --telemetry-every 1 --run-name ppo-racingv2-speedgated-rettifilo-goal-30k
```

Promotion rule:

- Promote only if normal-start full-lap eval moves materially beyond `970.775m` or QC shows the first bad Rettifilo behavior has changed from late/no braking at about `520.8m`.
- Reject immediately if deterministic normal-start eval collapses for two consecutive eval intervals.

## TensorBoard

```powershell
uv run --no-sync tensorboard --logdir "C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts" --host 127.0.0.1 --port 6006
Start-Process "http://127.0.0.1:6006"
```

If TensorBoard fails with missing `pkg_resources`, repair setuptools:

```powershell
uv pip install --no-deps setuptools==80.9.0
```

## Documentation

After any meaningful implementation, validation, benchmark, training, or blocker event, update:

- `README.md`
- `Plan.md`
- `Implement.md`
- `Documentation.md`

Archived plans live under:

- `archive/plans/`
