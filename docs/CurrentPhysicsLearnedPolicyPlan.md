# Current Physics Learned Policy Goal

This is a goal prompt for implementing the next learned-policy path under the current simulator physics.

The target is not to replace the current GPU evolutionary search result. The target is to turn the verified behavior discovered by GPU ES into a reusable neural policy that can drive from a normal start under the current `v1` physics, be visually inspectable, and optionally feed stronger candidates back into ES.

Do not stop at a prototype. Do not stop at a smoke test. Continue until the dataset export, behavior cloning, SAC fine-tuning, evaluation, replay, documentation, and validation gates are implemented end to end and pass.

## Objective

Build the full pipeline:

```text
current physics v1
  -> CPU-verified GPU ES telemetry
  -> transition dataset
  -> behavior-cloned policy
  -> SAC fine-tuned policy
  -> CPU-verified normal-start learned-policy laps
  -> replayable 1000-car policy checkpoint swarm visualizations
  -> optional actor injection back into ES
```

The project has already proved that search can find fast laps:

- CPU ES reached about `89s`.
- GPU ES reached a CPU-verified selected valid lap of `81.233s`.
- The current best run is `C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605`.
- The current strongest selected replay path is `C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top`.

The learned policy goal is stricter than "ES found a lap." A trained neural policy must complete valid normal-start laps by itself under the CPU oracle simulator.

## Final Success Criteria

Mark this goal complete only when all of these are true:

1. A reproducible transition export tool exists and can export verified ES data into a versioned dataset.
2. The dataset contains observations, actions, rewards, next observations, done flags, terminal reasons, lap validity, timing, source candidate metadata, and all hashes needed to reproduce the source simulator configuration.
3. A behavior-cloning trainer exists and can overfit a small shard, train on the full curated dataset, and save actor checkpoints with full metadata.
4. A SAC trainer exists and can initialize from the behavior-cloned actor, use the ES transition dataset as replay-buffer prefill, collect additional rollouts, and save policy checkpoints.
5. SAC uses the current physics contract and outputs real continuous car controls, not discrete action IDs.
6. Evaluation uses CPU `MonzaSim` as the promotion oracle.
7. The best learned policy completes a valid normal-start lap under current physics.
8. The target lap goal remains near `<=80.0s`. If the first learned policy only reaches the `81-85s` band, continue improving the dataset, weighting, SAC training, reward, and actor injection loop until the learned policy is competitive with the ES target or the remaining blocker is documented with concrete failed experiments.
9. A replay mode exists for policy checkpoints where the user can visually inspect many cars from the same policy checkpoint, including `1000` parallel policy rollouts grouped by checkpoint.
10. The README and `Documentation.md` describe how to export the dataset, train BC, train SAC, evaluate, replay, and verify results.
11. Full validation passes:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

## Non-Negotiables

- Do not change current physics in this plan.
- Do not silently alter reward semantics for existing PPO/ES commands unless the change is explicitly versioned, tested, and documented.
- Do not treat ES behavior as learned-policy success.
- Do not treat raw GPU ES winners as trusted unless they pass CPU postcheck/rerank.
- Do not train from only the single fastest lap.
- Do not throw away near-miss and diverse behavior. The learned policy needs a broad dataset, not a one-lap clone.
- Do not use TD3 in this plan. Use SAC only.
- Do not make replay optional. Replay is part of the product.
- Do not remove existing CPU PPO, GPU PPO, CPU ES, GPU ES, replay, or telemetry functionality.
- Do not break old selected telemetry replay.
- Do not put large datasets in the repo root.

## Why SAC

SAC is the best immediate fit for this codebase because:

- The action space is continuous: throttle, brake, and steer.
- SAC is off-policy, so it can learn from a replay buffer seeded by verified ES transitions.
- SAC's entropy objective keeps exploration alive instead of collapsing immediately to one brittle clone of the fastest ES line.
- SAC can combine offline prefill data with online rollouts under the same simulator.
- The project already has PyTorch, CUDA, GPU batch simulation, replay, and CPU verification infrastructure.

Research grounding:

- Haarnoja et al. describe SAC as an off-policy actor-critic method under a maximum entropy framework, targeting both return and entropy for robust continuous-control learning: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).
- Stable-Baselines3 documents SAC as a continuous-action off-policy algorithm with a replay buffer and automatic entropy tuning options: [SB3 SAC documentation](https://stable-baselines3.readthedocs.io/en/v2.3.0/modules/sac.html).

This plan should use project-native SAC where practical, because the repo's strongest simulator path is now batched GPU simulation plus CPU oracle verification. SB3 SAC can remain useful for reference or smoke checks, but the production-quality route should avoid a NumPy/Gym loop that prevents GPU-resident rollout collection.

## Library And Setup Decision

Do not install a separate SAC-specific library for the main path.

Use:

- custom PyTorch SAC for the production implementation;
- the repo's existing PyTorch/CUDA setup;
- the repo's existing GPU batch simulator for rollout collection;
- the repo's existing CPU `MonzaSim` for promotion eval;
- Stable-Baselines3 SAC only as a reference or smoke baseline.

Reasoning:

- `torch` is already part of the `train` extra.
- `stable-baselines3` is already part of the `train` extra and includes SAC, but SB3's Gym/VecEnv path is CPU/NumPy oriented.
- The real goal is a GPU-resident rollout collector and replay buffer, which is cleaner with project-native PyTorch SAC.
- Adding another RL framework before the custom SAC path is proven would increase complexity without solving the main bottleneck.

Required setup before this goal:

```powershell
uv sync --active --all-extras --all-packages
uv run --no-sync f1-hardware-check --json --warp-smoke
```

No FastF1 dependency is required for the current-physics learned-policy goal unless the implementation also chooses to regenerate calibration reports.

## Current Simulator Contract

The learned policy must respect the current v1 simulator contract.

Current physics and controls:

- CPU oracle: `src/f1rl/sim.py`, `MonzaSim`.
- Physics step: `src/f1rl/physics.py`.
- GPU batch path: `src/f1rl/gpu_batch.py`, `src/f1rl/gpu_physics.py`, `src/f1rl/gpu_fused_warp.py`.
- Control surface: `MonzaSim.step_controls(throttle, brake, steer)`.
- Output controls:
  - `throttle`: continuous, clipped to `[0.0, 1.0]`.
  - `brake`: continuous, clipped to `[0.0, 1.0]`.
  - `steer`: continuous, clipped to `[-1.0, 1.0]`.

The current controller genome has exactly three outputs and uses the following feature contract:

- `bias`
- `speed_norm`
- `target_speed_norm`
- `speed_error_norm`
- `brake_demand`
- `future_brake_demand`
- `target_speed_drop_norm`
- `brake_gate_proximity`
- `brake_gate_distance_norm`
- `lookahead_abs_max`
- `signed_lateral_error_norm`
- `heading_error_norm`
- `yaw_rate_norm`
- `curvature_norm`
- `target_steer`
- `last_throttle`
- `last_brake`
- `last_steer`
- `segment_progress_ratio`
- `lookahead_0`
- `lookahead_1`
- `lookahead_2`
- `lookahead_3`

The learned policy should start from the public observation profile, not the private controller feature map, unless a deliberate learned-policy observation profile is created.

Recommended first observation contract:

- Use `observation_profile="racing_v2"` as the initial learned-policy input.
- Store the exact observation vector in the dataset.
- Store the observation profile name and observation dimension in the dataset manifest.
- Add a named `learned_policy_v1` observation profile only if `racing_v2` is missing necessary features for SAC.

Do not bind the learned-policy dataset to private simulator internals. Use `MonzaSim.observation()` or a shared batch observation implementation so CPU/GPU paths can remain testable.

## Policy Inputs And Outputs

### Inputs

The first SAC actor should consume the normalized `racing_v2` observation vector.

Required input metadata:

- `observation_profile`
- `observation_dim`
- ordered observation feature schema if available
- normalization strategy
- simulator config hash
- track spec hash
- physics model id, initially `v1`

Potential future learned-policy-specific additions:

- explicit progress ratio
- upcoming target-speed drop
- distance to next braking demand
- lookahead heading errors
- local curvature
- last throttle/brake/steer
- checkpoint or section id encoded continuously

Do not add these casually. If a new observation profile is added, CPU observation tests, GPU observation tests, dataset export tests, and replay/eval metadata must be updated.

### Outputs

The learned actor outputs exactly:

```text
[throttle, brake, steer]
```

Output interpretation:

- `throttle = sigmoid/logistic or squashed value -> [0, 1]`
- `brake = sigmoid/logistic or squashed value -> [0, 1]`
- `steer = tanh -> [-1, 1]`

Training behavior:

- SAC actor is stochastic during training.
- Evaluation actor is deterministic by default, using the mean action or deterministic squashed action.
- Evaluation may optionally run stochastic samples for debugging, but promotion requires deterministic CPU evaluation.

Throttle/brake handling:

- Do not use the old two-action PPO mapping as the learned-policy target.
- Store throttle and brake as independent outputs.
- During environment stepping, apply the same dominance/compatibility semantics used by `step_controls` and current controller handling.
- Add tests for simultaneous throttle/brake behavior and ensure it is not silently different between dataset export, BC training, SAC rollout, CPU eval, and GPU rollout.

## Dataset Strategy

The dataset is the bridge from ES to learned policy.

Do not export only the fastest lap. The dataset should include enough variety that SAC can learn:

- how to be fast;
- how to recover from imperfect states;
- how to finish safely;
- how to avoid the slow-crawl behavior;
- where braking must happen;
- how good lines differ from near-miss lines;
- how different valid lineages solve the lap.

### Source Candidate Buckets

Export a curated union from CPU-verified ES data:

- fastest valid laps;
- top valid laps by lap time;
- valid laps with high top-decile pace;
- valid laps from distinct generations;
- valid laps from distinct lineage branches;
- near-valid failures after `5000m`;
- near-valid failures after `5500m`;
- high-speed frontier failures;
- clean frontier candidates with low heading and lateral error;
- section specialists from braking/corner exit profiles;
- a small amount of early-lap failure data for negative/terminal learning.

Do not overrepresent duplicated clones. Deduplicate by:

- genome hash;
- source snapshot hash;
- sim config hash;
- lineage root;
- generation/candidate identity.

Keep duplicates only if they produce distinct CPU replay trajectories or distinct terminal outcomes.

### Source Artifacts

The exporter must accept:

- a GPU ES output directory;
- selected telemetry folders;
- postchecked attempts;
- pool diagnostics when available;
- optional multiple run directories.

For the current best run, the exporter should support:

```powershell
C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605
```

and selected telemetry:

```powershell
C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top
```

### Transition Schema

Each transition must include:

- `obs`
- `action`
- `reward`
- `next_obs`
- `done`
- `terminated`
- `truncated`
- `terminal_reason`
- `valid_lap`
- `completed_lap`
- `lap_time_s` if known
- `sim_time_s`
- `progress_m`
- `progress_delta_m`
- `speed_kph`
- `throttle`
- `brake`
- `steer`
- `heading_error_deg`
- `lateral_error_m`
- `checkpoint_index`
- `source_run_id`
- `source_generation`
- `source_candidate_index`
- `source_selection_reason`
- `source_profile_scores` when available
- `genome_hash`
- `lineage_root`
- `physics_model`
- `sim_config_hash`
- `track_hash`

Recommended storage:

- compressed `.npz` shards for numeric arrays;
- JSON manifest for metadata;
- JSONL summary for per-source-candidate stats;
- no heavy new dependency for v1 unless array size demands it.

Dataset manifest should include:

- schema version;
- dataset id;
- creation timestamp;
- source run list;
- source artifact paths;
- postcheck status;
- physics model id;
- sim config;
- reward config;
- assist config;
- observation profile;
- action schema;
- track spec hash;
- exporter git commit;
- candidate selection counts by bucket;
- total transitions;
- total source candidates;
- valid lap count;
- fastest source lap;
- mean valid lap time;
- average/median/max progress;
- storage size.

### Transition Generation

Preferred export method:

1. Load selected/postchecked candidate metadata.
2. Reconstruct the candidate genome and start snapshot.
3. Replay through CPU `MonzaSim`.
4. At each step, record `obs`, selected action, reward, `next_obs`, and done.
5. Emit sharded transitions.

Reason:

- CPU replay is the oracle.
- It guarantees dataset actions and transitions match trusted promotion behavior.
- It avoids trusting raw GPU drift.

Do not export raw GPU transitions as training truth unless the manifest labels them as unverified proposal data. For training targets, use CPU-verified replay.

## Behavior Cloning Phase

Behavior cloning is a controlled pretraining phase.

Goal:

- Learn a policy that imitates verified ES controls from the same observation vectors SAC will later use.
- Create a strong actor initialization for SAC.
- Detect dataset/action schema mistakes before RL adds noise.

BC trainer requirements:

- CLI entrypoint, for example `f1-train-bc`.
- Reads the transition dataset manifest.
- Uses train/validation split by source candidate, not by random row only. This prevents leaking the same lap into train and validation.
- Saves checkpoints, metrics, config, and normalization stats.
- Supports resume.
- Logs per-action loss: throttle, brake, steer.
- Logs derived control metrics: simultaneous throttle/brake rate, mean brake in braking zones, steering smoothness, action saturation rates.
- Supports weighted sampling by source bucket.
- Supports overfit smoke on one tiny shard.

Suggested losses:

- throttle: Huber or MSE;
- brake: Huber or MSE with higher weight in braking zones;
- steer: Huber or MSE;
- optional action-smoothness auxiliary loss;
- optional speed-conditioned brake demand auxiliary target if added later.

Suggested sample weights:

- fastest valid laps: high;
- diverse valid laps: high;
- clean near-valid frontier: medium;
- terminal failures: low for imitation, but kept for SAC replay;
- very early failures: very low for imitation unless used for recovery labels.

BC validation:

- Overfit 1 source lap until action loss is near zero.
- Run deterministic CPU eval from normal start.
- Run multiple seed evals if stochastic wrappers are used.
- Export replay telemetry for the best BC checkpoint.
- Confirm replay loads headlessly and visually.

BC is not final success. It is a bootstrapping phase.

## SAC Phase

SAC is the main learned-policy training phase.

Goal:

- Use the BC actor as initialization.
- Seed replay buffer from the verified ES transition dataset.
- Continue learning through environment interaction.
- Keep enough entropy to discover improvements beyond pure imitation.

Implementation options:

1. Project-native SAC with batched GPU rollout collection.
2. SB3 SAC as a reference/smoke path.

Recommended production route:

- Implement project-native SAC so rollout collection can use the batched GPU simulator without per-step Gym/NumPy overhead.
- Keep CPU eval/postcheck as the promotion oracle.
- Use SB3 SAC only as a baseline or smoke if it is faster to validate algorithm wiring.

SAC trainer requirements:

- CLI entrypoint, for example `f1-train-sac`.
- Reads BC checkpoint optionally.
- Reads ES transition dataset for replay prefill.
- Supports `--device cuda`.
- Supports `--n-envs` for batched rollout collection.
- Supports checkpointing and resume.
- Saves replay buffer metadata.
- Logs actor loss, critic loss, alpha/entropy, Q estimates, reward, episode length, lap validity, lap time, progress, terminal reason, speed metrics, and off-track/collision rate.
- Evaluates deterministic CPU policy on a schedule.
- Writes replayable telemetry for promoted eval checkpoints.
- Supports early stopping only after validation passes, not because a training metric looks good.

Replay-buffer prefill:

- Store verified ES transitions before online collection starts.
- Preserve source bucket labels.
- Allow weighted sampling or prioritized sampling later, but start simple.
- Do not overwrite source metadata.

Reward:

- Start with current simulator reward.
- Add learned-policy-specific logging before modifying reward.
- If speed remains weak, tune reward in versioned configs, not hidden constants.
- Any reward change must be documented in dataset/training manifests.

Promotion:

- A SAC checkpoint is promoted only if CPU eval proves valid normal-start lap completion.
- For the <=80s target, use deterministic CPU eval under the saved normalization stats and exact sim config.
- Save the promoted policy, eval telemetry, summary JSON, and replay command.

## SAC Swarm Replay Requirement

The user wants the same visual feeling as ES: many cars visible together, gradually improving.

For RL, these are not 1000 independent evolving genomes. They are 1000 rollout instances from one policy checkpoint, optionally with stochastic sampling, start perturbations, or fixed seed variations.

Implement this explicitly.

Required behavior:

- Every configured SAC evaluation interval, run a policy swarm evaluation:
  - default `1000` cars;
  - same policy checkpoint;
  - deterministic mode for promotion;
  - optional stochastic mode for exploration visualization;
  - fixed seed list for comparability across checkpoints.
- Save replayable telemetry grouped by policy checkpoint.
- Add replay mode to play checkpoint by checkpoint:
  - checkpoint 000;
  - checkpoint 001;
  - checkpoint 002;
  - etc.
- Add controls equivalent to ES replay:
  - skip checkpoint;
  - speed up/down in-window;
  - pause/resume;
  - sort by best progress or fastest valid lap;
  - limit displayed cars if needed;
  - highlight fastest valid, farthest, and selected promoted policy.

Suggested command shape:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay artifacts\sac-run\checkpoint_swarms --by-checkpoint --checkpoint-limit 1000 --speed 1
```

Storage policy:

- Store checkpoint swarm summaries compactly.
- Store full per-step telemetry for selected/highlighted cars.
- Store compressed all-candidate swarm telemetry if full debug is requested.
- Keep manifests uncompressed and human-readable.

Visual success:

- The user can open replay and see many cars driving at once.
- Checkpoints can be skipped without waiting for slow loading.
- Replay communicates policy improvement over training, not just a final one-car lap.

## Optional Actor Injection Back Into ES

After SAC produces a useful actor, inject it back into ES as a smart candidate source.

Do not do this before BC/SAC evaluation works.

Possible injection modes:

- actor-controlled candidates with action noise;
- actor plus small residual controller genome;
- actor-generated phase/controller seeds;
- actor used as smart immigrant source;
- actor used to create new state-library starts.

Rules:

- ES success remains ES success, not learned-policy success.
- Actor-injected ES candidates must still pass CPU postcheck.
- Actor injection should be measured against a no-actor ES baseline.

## Implementation Phases

### Phase 0: Repo Audit And Interfaces

Inspect:

- `src/f1rl/sim.py`
- `src/f1rl/env.py`
- `src/f1rl/gpu_batch.py`
- `src/f1rl/evolution_search.py`
- `src/f1rl/evolution_postcheck.py`
- `src/f1rl/replay.py`
- `src/f1rl/telemetry.py`
- `src/f1rl/gpu_ppo.py`
- current best artifact folder

Deliverables:

- Confirm source artifacts exist.
- Confirm selected telemetry replay loads.
- Confirm dataset export requirements against actual `StepTelemetry`.
- Confirm action and observation schemas.

Validation:

```powershell
uv run --no-sync python -m f1rl.replay "C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top" --headless --limit 1
```

### Phase 1: Dataset Export

Add:

- `src/f1rl/es_dataset.py` or equivalent.
- CLI entrypoint, for example `f1-export-es-dataset`.
- Sharded transition writer.
- Dataset manifest.
- Dataset loader.
- Unit tests.

Minimum command shape:

```powershell
uv run --no-sync python -m f1rl.es_dataset export `
  --run-dir "C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605" `
  --selected-telemetry "C:\f1rl-artifacts\gpu-speed-speedprofiles-2000x150-25k-20260605\selected_telemetry_summary_top" `
  --output-dir "C:\f1rl-artifacts\datasets\v1-es-policy-dataset" `
  --observation-profile racing_v2 `
  --physics-model v1
```

Tests:

- schema round trip;
- manifest hash fields exist;
- action bounds;
- observation dimension;
- `done` semantics;
- source candidate metadata;
- deterministic re-export from same source;
- corrupt source artifact fails clearly.

### Phase 2: Dataset QA

Add:

- dataset report command;
- histograms or JSON summaries;
- per-bucket counts;
- fastest valid source stats;
- progress distribution;
- terminal reason distribution;
- action distribution;
- braking-zone action checks.

Minimum command:

```powershell
uv run --no-sync python -m f1rl.es_dataset report "C:\f1rl-artifacts\datasets\v1-es-policy-dataset"
```

Do not train until the dataset report looks sane.

### Phase 3: Behavior Cloning

Add:

- `src/f1rl/bc_train.py` or equivalent.
- actor network shared with SAC if possible.
- checkpoint format.
- eval command.
- overfit test.

Minimum commands:

```powershell
uv run --no-sync python -m f1rl.bc_train `
  --dataset "C:\f1rl-artifacts\datasets\v1-es-policy-dataset" `
  --output-dir "C:\f1rl-artifacts\learned\v1-bc" `
  --device cuda `
  --epochs 1 `
  --smoke
```

```powershell
uv run --no-sync python -m f1rl.policy_eval `
  --policy "C:\f1rl-artifacts\learned\v1-bc\best_policy.pt" `
  --output-dir "C:\f1rl-artifacts\learned\v1-bc-eval" `
  --episodes 3 `
  --observation-profile racing_v2 `
  --physics-model v1
```

### Phase 4: SAC

Add:

- `src/f1rl/sac_train.py` or equivalent.
- replay buffer seeded from dataset.
- actor/critic networks.
- entropy temperature handling.
- checkpointing/resume.
- deterministic CPU eval callback.
- policy swarm eval callback.

Minimum smoke:

```powershell
uv run --no-sync python -m f1rl.sac_train `
  --dataset "C:\f1rl-artifacts\datasets\v1-es-policy-dataset" `
  --bc-checkpoint "C:\f1rl-artifacts\learned\v1-bc\best_policy.pt" `
  --output-dir "C:\f1rl-artifacts\learned\v1-sac-smoke" `
  --device cuda `
  --timesteps 1024 `
  --n-envs 64 `
  --max-steps 512 `
  --eval-every 512 `
  --swarm-every 512 `
  --swarm-size 128
```

Full training should scale only after smoke, overfit, dataset QA, and eval all pass.

### Phase 5: CPU Promotion Eval

Add/extend:

- deterministic CPU policy eval;
- normal-start valid lap checks;
- lap time summary;
- replay telemetry writing;
- benchmark comparison against current ES and PPO baselines.

Promotion command shape:

```powershell
uv run --no-sync python -m f1rl.policy_eval `
  --policy "C:\f1rl-artifacts\learned\v1-sac\best_policy.pt" `
  --output-dir "C:\f1rl-artifacts\learned\v1-sac-promotion-eval" `
  --episodes 20 `
  --deterministic `
  --normal-start `
  --observation-profile racing_v2 `
  --physics-model v1 `
  --write-telemetry gzip
```

Required output:

- fastest valid learned-policy lap;
- valid lap count;
- valid lap rate;
- average valid lap time;
- terminal reason counts;
- selected replay command.

### Phase 6: SAC Swarm Visualization

Add:

- policy checkpoint swarm evaluator;
- replay grouping by checkpoint;
- replay UI controls for checkpoint skipping and speed adjustment;
- headless replay tests.

Command shape:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_eval `
  --policy-dir "C:\f1rl-artifacts\learned\v1-sac" `
  --output-dir "C:\f1rl-artifacts\learned\v1-sac-swarms" `
  --swarm-size 1000 `
  --checkpoints all `
  --observation-profile racing_v2 `
  --physics-model v1
```

Replay:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "C:\f1rl-artifacts\learned\v1-sac-swarms" --by-checkpoint --speed 1
```

### Phase 7: Optional Actor Injection

Only after learned-policy eval works:

- add actor smart immigrant mode to ES;
- compare ES with and without actor injection;
- CPU postcheck all promoted ES candidates;
- document whether actor injection improves speed or consistency.

## Metrics To Track

Dataset:

- total transitions;
- total source candidates;
- valid lap count;
- fastest source lap;
- source lineage diversity;
- terminal reason distribution;
- progress distribution;
- action distribution;
- braking-zone action quality;
- dataset size.

BC:

- train/validation action loss;
- per-action loss;
- overfit-shard loss;
- deterministic CPU eval progress;
- valid lap rate;
- fastest lap;
- replay-load pass/fail.

SAC:

- actor loss;
- critic loss;
- entropy/alpha;
- Q-value scale;
- replay-buffer source mix;
- online/offline sample ratio;
- average return;
- average progress;
- valid lap count/rate;
- fastest valid lap;
- average valid lap time;
- top-decile pace;
- collision/off-track/no-progress rates;
- action saturation;
- throttle/brake overlap;
- CPU eval metrics;
- swarm replay metrics.

Promotion:

- learned policy fastest valid lap;
- learned policy average valid lap time;
- learned policy valid lap rate;
- comparison to current `81.233s` ES;
- comparison to CPU PPO status;
- comparison to CPU ES `89s` milestone.

## Validation Plan

Use focused tests first, then full repo checks.

Dataset tests:

- manifest round trip;
- transition array shapes;
- action bounds;
- observation bounds;
- done semantics;
- source metadata;
- deterministic CPU replay export;
- corrupt input handling;
- replay source hash mismatch detection.

BC tests:

- model forward output bounds;
- one-batch train step;
- tiny overfit;
- checkpoint save/load;
- deterministic eval smoke.

SAC tests:

- replay buffer prefill;
- actor/critic forward;
- target-network update;
- alpha update;
- one training step;
- checkpoint save/load;
- CPU eval callback smoke;
- policy swarm eval smoke.

Replay tests:

- selected policy telemetry load;
- by-checkpoint replay manifest;
- headless replay for 2 traces;
- skip-checkpoint metadata works.

CLI smokes:

```powershell
uv run --no-sync python -m f1rl.es_dataset export --help
uv run --no-sync python -m f1rl.bc_train --help
uv run --no-sync python -m f1rl.sac_train --help
uv run --no-sync python -m f1rl.policy_eval --help
uv run --no-sync python -m f1rl.policy_swarm_eval --help
```

Full checks:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

## Artifact Layout

Recommended:

```text
C:\f1rl-artifacts\
  datasets\
    v1-es-policy-dataset-YYYYMMDD\
      dataset_manifest.json
      source_candidates.jsonl
      transition_shards\
      reports\
  learned\
    v1-bc-YYYYMMDD\
      config.json
      checkpoints\
      eval\
    v1-sac-YYYYMMDD\
      config.json
      checkpoints\
      replay_buffer_manifest.json
      cpu_eval\
      checkpoint_swarms\
```

Keep repo root clean. Do not store large datasets or raw training buffers in the repo.

## Documentation Updates Required

When implemented, update:

- `README.md` with the learned-policy path and commands.
- `Documentation.md` with live status, current best learned-policy metrics, and caveats.
- `AGENTS.md` only if the active working loop changes.

Do not turn `Documentation.md` into a transcript.

## Done Means Done

The implementing agent must continue through:

1. code implementation;
2. focused tests;
3. CLI smokes;
4. dataset export smoke;
5. BC smoke and overfit;
6. SAC smoke;
7. CPU eval;
8. replay-load check;
9. documentation update;
10. full repo validation.

If a learned policy does not yet hit the final lap-time target, do not mark the learned-policy performance goal complete. Keep improving the dataset selection, BC weighting, SAC rollout, evaluation cadence, and optional actor-injection loop with explicit experiments and artifact summaries.

If a blocker appears, isolate it with the smallest reproducible command, fix it, retest, and continue. Do not hand back a partially wired pipeline as complete.

## Research Sources

- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/v2.3.0/modules/sac.html)
- [FastF1 telemetry API reference](https://docs.fastf1.dev/api_reference/telemetry.html)
- [FastF1 core timing and telemetry data](https://docs.fastf1.dev/core.html)
- [FastF1 getting started examples](https://docs.fastf1.dev/examples/index.html)
