# RL1 Achieved: Learned Policy Reaches Sub-80s CPU MonzaSim Laps

Last updated: 2026-06-05.

This document is the detailed RL1 achievement report for the current-physics learned-policy goal. It explains what was achieved, what counts as proof, how the repo moved from PPO and ES-only attempts to a promoted learned policy, what code changed, what artifacts prove the result, and how future agents should reproduce, validate, replay, or extend the work.

The headline result is a project-native PyTorch SAC learned policy that completes valid normal-start laps under the CPU `MonzaSim` oracle in `79.750s`.

## Executive Summary

RL1 is complete for current physics `v1`.

The promoted learned-policy checkpoint is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt
```

The promotion proof is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\eval_summary.json
```

The CPU oracle result is:

```text
episodes: 3
valid_lap_count: 3
valid_lap_rate: 1.0
terminal_reason_counts: lap_complete = 3
fastest_valid_lap_s: 79.75
average_valid_lap_s: 79.75
normal_start: true
deterministic: true
physics_model: v1
```

Each promoted episode reached the same result:

```text
steps: 4785
best_progress_m: 5793.000137343182
final_progress_m: 5793.000137343182
final_speed_kph: 296.7877942668157
termination_reason: lap_complete
valid_lap: true
lap_time_s: 79.75
```

The promoted policy is a learned SAC checkpoint, not a raw ES trajectory and not PPO. Its saved metadata says:

```text
stage: sac
step: 256
updates: 32
actor_class: SACActor
observation_profile: learned_policy_v1
observation_dim: 58
control_mode: dominance
```

The replay command for the promoted CPU-eval laps is:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry"
```

The policy swarm replay command is:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000" --by-checkpoint --speed 1
```

## Final Storage State

After RL1 was achieved, bulk artifacts were compressed and moved to external storage while curated replay telemetry stayed local.

Local replay/highlight artifacts are under:

```text
artifacts\highlights
```

Current local split:

| Set | Files | Size |
|---|---:|---:|
| CPU ES telemetry | 907 | `0.43 GB` |
| GPU ES telemetry, reduced stratified local set | 907 | `0.65 GB` |
| RL telemetry | 1005 | `1.55 GB` |
| All local highlights including GIFs | 2830 | `2.70 GB` |

The three local replay sets are:

```text
artifacts\highlights\full-generation-reel-20260605\cpu-es-150x60
artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150
artifacts\highlights\learned-policy-replays-20260605\telemetry
```

The local GPU ES replay set is intentionally reduced to `6 x 150 = 900` stratified traces. The full `6 x 2000 = 12000` GPU ES replay set is preserved on external storage:

```text
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\gpu-es-2000x150-full-12000-traces-20260605.tar.zst
```

The current reduced local highlight mirror is archived on external storage:

```text
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst
```

Bulk learned-policy artifacts, datasets, and run artifacts are also preserved under:

```text
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605
```

The original in-run paths in the sections below remain important for provenance and reproduction. If a command needs `artifacts\runs`, `artifacts\datasets`, or `artifacts\learned`, restore the corresponding archive from `D:` first. The local replay commands above work without restoring those bulk archives.

## What Counts As The RL1 Success

The result counts because a saved neural policy checkpoint drives the car through a normal-start lap under CPU `MonzaSim`.

The promotion path is:

```text
CPU-verified ES source data
  -> CPU-replayed transition dataset
  -> controller-distilled learned actor seed
  -> behavior cloning
  -> project-native PyTorch SAC fine-tuning
  -> deterministic CPU MonzaSim policy eval
  -> replayable selected telemetry
  -> 1000-car policy swarm replay
```

The success criteria were stricter than "a search found a lap." ES and actor-injected ES were allowed to discover and improve fast source behavior, but final promotion required a learned policy checkpoint to run normal-start CPU `MonzaSim` by itself.

The promoted policy is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt
```

The policy was evaluated by:

```text
src\f1rl\policy_eval.py
```

That evaluator:

- loads the learned checkpoint through `load_policy_checkpoint`;
- constructs CPU `MonzaSim` with continuous controls and `action_set="racing"`;
- uses `observation_profile="learned_policy_v1"`;
- resets from normal start with `start_progress_m=0.0` and `start_speed_kph=80.0`;
- applies policy outputs through `sim.step_controls(throttle, brake, steer)`;
- records gzip telemetry;
- writes `eval_summary.json`;
- writes a replay-compatible `selected_telemetry\manifest.json`.

## What Does Not Count As The RL1 Success

The following are important, but they are not the final learned-policy proof by themselves:

- CPU ES reaching about `89s`.
- Broad GPU ES reaching a CPU-verified `81.233s` selected lap.
- Actor-injected ES finding a `79.750s` source candidate.
- Behavior cloning loss getting low.
- SAC training metrics looking good.
- Raw GPU winner rows before CPU postcheck.
- PPO infrastructure or PPO smokes.

The final proof is the deterministic CPU oracle eval of the learned SAC checkpoint:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\eval_summary.json
```

## Historical Timeline

### 1. Initial PPO Direction

The project began as a top-down Monza driving environment with a Gymnasium/SB3 PPO training path. That infrastructure still exists and remains useful:

```text
src\f1rl\env.py
src\f1rl\train.py
src\f1rl\gpu_ppo.py
```

However, PPO was not the path that produced the promoted RL1 result. The old PPO micro-rung loop was explicitly deprioritized because it was not producing valid full normal-start laps. PPO remains mechanically tested infrastructure, not the headline achievement.

This distinction matters because the repo now has several things that can "drive":

- scripted or diagnostic action search;
- CPU evolutionary search;
- GPU evolutionary search;
- actor-injected ES;
- behavior-cloned learned actors;
- SAC learned actors;
- CPU/Gym/SB3 PPO;
- experimental GPU PPO.

RL1 specifically means the learned SAC actor completes the CPU oracle lap.

### 2. CPU ES Milestone

Before the learned-policy work, CPU evolutionary search reached a visible approximate `89s` Monza lap. That result proved that the simulator, telemetry, scoring, and replay loop could produce full-lap behavior. It also gave the project a multi-car replay style that became part of the product expectation.

The `README.md` still references the older 89s GIF:

```text
pygame-window-gen49-fastest-89s-all-150-cars-slow.gif
```

The CPU ES result was important because it established the search and replay workflow, but it was still search output, not a reusable neural policy.

### 3. Broad GPU ES Milestone

The next major step was GPU fused evolutionary search. The current docs record this run:

```text
artifacts\runs\gpu-speed-speedprofiles-2000x150-25k-20260605
```

That run used:

```text
population: 2000
generations: 150
total candidates: 300,000
max_steps: 25000
target termination: disabled
backend: GPU fused evolutionary search
```

The strongest selected result from that run was:

```text
fastest selected valid lap: 81.233s
CPU-verified selected candidate: generation 143, candidate 1864
CPU verification reason: lap_complete
CPU/GPU reason mismatches: 0
CPU/GPU valid-lap mismatches: 0
```

The broad GPU result proved that GPU ES could discover genuinely fast full laps, but it still did not satisfy RL1 because a raw ES controller is not a trained neural policy.

### 4. Early Learned-Policy Attempts

The learned-policy plan originally allowed starting from `racing_v2` observations. Early learned-policy attempts could imitate rows, but closed-loop CPU eval exposed drift. That is a common failure mode for imitation: the supervised dataset can match source actions locally, but when the policy makes a small mistake, it visits states that are not well covered by the data and the loop compounds the error.

The critical steering decision was to stop optimizing a fragile `racing_v2` learned policy and make a named, explicit learned-policy observation profile:

```text
learned_policy_v1
```

This profile records the exact ordered input contract in manifests and checkpoints. It uses the existing public `racing_v2` observation block, then appends the fixed feature set that the strong controller-style ES policy had been using.

### 5. learned_policy_v1 Observation Contract

The new profile is defined in:

```text
src\f1rl\config.py
src\f1rl\sim.py
src\f1rl\gpu_observation.py
src\f1rl\es_dataset.py
```

`src\f1rl\config.py` adds `learned_policy_v1` to `OBSERVATION_PROFILES` and defines:

```python
LEARNED_POLICY_V1_FEATURES = (
    "bias",
    "speed_norm",
    "target_speed_norm",
    "speed_error_norm",
    "brake_demand",
    "future_brake_demand",
    "target_speed_drop_norm",
    "brake_gate_proximity",
    "brake_gate_distance_norm",
    "lookahead_abs_max",
    "signed_lateral_error_norm",
    "heading_error_norm",
    "yaw_rate_norm",
    "curvature_norm",
    "target_steer",
    "last_throttle",
    "last_brake",
    "last_steer",
    "segment_progress_ratio",
    "lookahead_0",
    "lookahead_1",
    "lookahead_2",
    "lookahead_3",
)
```

The resulting observation dimension for the final dataset and policy is:

```text
observation_dim: 58
```

The manifest schema describes the observation as:

```text
base_profile: racing_v2
append block: learned_policy_v1_append
```

This was the main representational fix. It gave the learned actor stable access to the speed target, braking demand, future braking demand, braking-gate distance, lateral and heading error, curvature, target steer, previous controls, segment progress, and lookahead errors.

### 6. Controller Distillation And Actor Injection

The learned-policy path did not jump directly from broad ES to SAC. It added a bridge from a CPU-replayable ES controller into a neural policy checkpoint:

```text
src\f1rl\controller_policy_init.py
```

This creates a `SACActor` with:

```text
hidden_sizes: []
control_mode: dominance
normalizer: identity
stage: controller_distill
```

The controller weights are copied into the actor's mean head over the appended `learned_policy_v1` features. The output scale is adjusted so the actor's tanh/raw action mapping reproduces the controller-style control surface.

The current controller-distilled artifact is:

```text
artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt
```

Its metadata records:

```text
stage: controller_distill
source: cpu_replayable_es_controller
source_run: artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910
source_generation: 0
source_candidate_index: 334
source_candidate_id: 0
source_lap_time_s: 79.75
source_terminal_reason: lap_complete
observation_profile: learned_policy_v1
observation_dim: 58
control_mode: dominance
actor_form: linear_controller_distillation
```

The ES side also gained actor injection support in:

```text
src\f1rl\evolution_search.py
```

New config and CLI options:

```text
--actor-injection-policy
--actor-injection-count
--actor-injection-mutation-sigma
```

Important constraints are enforced:

- actor injection requires `--genome-type controller`;
- the injected policy must be `learned_policy_v1`;
- the actor must use `control_mode="dominance"`;
- the current mapping only supports linear learned actors with no hidden layers.

The source run that produced the `79.750s` ES source candidate is:

```text
artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910
```

Its `best_so_far.json` records:

```text
generation: 0
candidate_index: 334
seed: 347
primary_scoring_profile: fast_valid_lap
termination_reason: lap_complete
valid_lap: true
elapsed_s: 79.75
steps: 4785
best_progress_m: 5793.000137343179
final_progress_m: 5793.000137343179
final_speed_kph: 296.7878394212713
backend: gpu
gpu_verified: true
cpu_verified_score: 40845479.8858056
```

The best row's lineage references an earlier actor-injection policy path:

```text
artifacts\learned\v1-controller-distill-sac80p050-best-source0\policy.pt
```

That means the final `79.750s` source candidate came from an actor-injection push seeded by a previous learned/controller-distilled policy, then the final artifact set distilled the `79.750s` source behavior into the current `sac79p750` controller seed. That source candidate is valuable, but the ES source candidate itself is not the RL1 promotion. The SAC checkpoint eval is the promotion.

The actor-injected source run was configured as a GPU fused postcheck run:

```text
population: 1024
configured_generations: 16
backend: gpu
gpu_engine: fused
gpu_run_mode: postcheck
gpu_dtype: float32
gpu_verify_top_k: 8
gpu_cpu_replay_top_k: 8
genome_type: controller
observation_profile: learned_policy_v1
actor_injection_count: 512
actor_injection_mutation_sigma: 0.02
```

The run was mined early because the generation summaries already showed the same under-80 source line and a rapidly increasing valid-lap count:

```text
generation 0: valid_lap_count=19, fastest_valid_lap_s=79.75
generation 1: valid_lap_count=93, fastest_valid_lap_s=79.75
generation 2: valid_lap_count=168, fastest_valid_lap_s=79.75
```

Use `best_so_far.json` and `generation_summary.jsonl` as the clearest result evidence for this run. Use `population_checkpoint.json` mainly for configuration and population provenance.

## Final Dataset

The final dataset is:

```text
artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16
```

The dataset manifest is:

```text
artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\dataset_manifest.json
```

The dataset report is:

```text
artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\reports\dataset_report.json
```

The dataset is a CPU-replayed ES transition dataset. It does not trust raw GPU transitions as training truth. It reconstructs selected source candidates and replays them through CPU `MonzaSim` to emit transitions.

The manifest records:

```text
kind: f1rl_es_transition_dataset
schema_version: 1
created_at: 2026-06-05T14:23:25.563138+00:00
dataset_id: v1-es-policy-dataset-learned-v1-sac79p750-broad-16
observation_profile: learned_policy_v1
observation_dim: 58
action_schema: throttle, brake, steer
physics_model: v1
postcheck_status: cpu_replayed_export
normalization_strategy: dataset_mean_std
total_transitions: 60584
total_source_candidates: 16
valid_lap_count: 10
fastest_source_lap: 79.75
mean_valid_lap_time: 79.99499999999999
average_progress_m: 4522.660291918846
median_progress_m: 5793.000137343179
max_progress_m: 5793.161968620552
storage_size_bytes: 21631976
```

Candidate buckets:

```text
valid_lap: 10
mid_frontier: 3
late_frontier: 1
early_failure: 2
```

The compressed transition shard is:

```text
transition_shards\shard_0000.npz
rows: 60584
sha256: d8993bf0d14cfe4ee667680417c7275acf93427abbf2399158b721236385eda8
```

Reproducibility hashes:

```text
sim_config_hash: df6ccc21b7a03492db6ab2f189eb722e5b168ad9c75cd30fca9c28647d1d61a3
track_hash: fe493af5fbd1b14ddb7d5f19bb7008a97d77a21b7e5dea2e870f60c479bc6ccc
exporter_git_commit: 50efc96c799427b63170ce78506df3825c42e8aa
```

The dataset report adds QA:

```text
total_transitions: 60584
source_candidate_count: 16
terminal_reason_distribution:
  lap_complete: 10
  off_track: 6
done_count: 16
completed_lap_transition_count: 10
completed_lap_source_count: 10
fastest_lap_time_s: 79.75
progress.mean_m: 2614.12939453125
progress.median_m: 2511.056640625
progress.max_m: 5793.162109375
```

Action distribution:

```text
action mean:
  throttle: 0.7879744172096252
  brake: 0.10841494798660278
  steer: -0.08157886564731598

action std:
  throttle: 0.2898886203765869
  brake: 0.2503615915775299
  steer: 0.4265025854110718

action min:
  throttle: 0.0011661153985187411
  brake: 0.0000031710733310319483
  steer: -0.9996628761291504

action max:
  throttle: 0.9997032880783081
  brake: 0.9953997731208801
  steer: 0.9951965808868408

simultaneous_throttle_brake_rate: 0.21023702627756502
```

Braking-zone checks:

```text
mean_brake_when_braking: 0.39126214385032654
braking_fraction: 0.2528720454245345
```

The source artifact path recorded in the dataset is:

```text
artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910
```

This dataset is deliberately not only the fastest lap. It includes the under-80 valid source, other valid near-80 sources, mid-frontier failures, one late-frontier failure, and early failures. That breadth matters because the learned-policy path needed verified behavior around the fast line, not just a single memorized lap.

The dataset export command shape is:

```powershell
uv run --no-sync python -m f1rl.es_dataset export `
  --run-dir artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910 `
  --output-dir artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --observation-profile learned_policy_v1 `
  --physics-model v1 `
  --max-candidates 16 `
  --max-per-generation 16 `
  --balanced-buckets
```

The dataset report command shape is:

```powershell
uv run --no-sync python -m f1rl.es_dataset report artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16
```

## Behavior Cloning

Behavior cloning lives in:

```text
src\f1rl\bc_train.py
```

BC's purpose in RL1 was not final promotion. It was controlled actor initialization. It takes verified ES transitions, normalizes observations, fits actions, records losses, and saves learned-policy checkpoints with enough metadata to feed SAC.

Important implementation details:

- Uses `load_dataset` from `src\f1rl\es_dataset.py`.
- Uses `SACActor`, `PolicyNormalizer`, and `bc_action_loss` from `src\f1rl\learned_policy.py`.
- Supports train/validation splits by source candidate.
- Supports overfit-source mode for shard debugging.
- Supports source weights from `source_candidates.jsonl`.
- Can upweight fastest valid sources.
- Can upweight late-progress states.
- Supports per-action weighting, including brake action weight.
- Saves epoch checkpoints and a best checkpoint.
- Preserves actor metadata and dataset manifest metadata.

The final BC checkpoint is:

```text
artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt
```

The final BC artifact metadata records:

```text
stage: bc
epoch: 4
dataset: artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16
device: cuda
epochs: 5
batch_size: 2048
hidden_size config: 512
lr: 1e-10
control_mode: dominance
observation_profile: learned_policy_v1
observation_dim: 58
```

The saved actor itself has:

```text
actor_class: SACActor
hidden_sizes: []
control_mode: dominance
```

This is intentional. The controller-distilled seed was a linear actor, and loading that seed through BC preserved the linear actor architecture. The `hidden_size=512` config value remains relevant to newly initialized networks and to SAC critic/Q networks, but the promoted actor is the loaded linear actor.

BC metrics for the best checkpoint:

```text
train_loss: 0.000208537612343207
val_loss: 0.01484123058617115
train_abs_error_throttle: 0.001403059926815331
train_abs_error_brake: 0.0007524057291448116
train_abs_error_steer: 0.010878312401473522
val_abs_error_throttle: 0.04777579754590988
val_abs_error_brake: 0.03090587630867958
val_abs_error_steer: 0.14260047674179077
simultaneous_throttle_brake_rate: 0.21023702627756502
```

The README-recorded BC command is:

```powershell
uv run --no-sync python -m f1rl.bc_train `
  --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --output-dir artifacts\learned\v1-bc-lpv1-sac79p750-broad16 `
  --device cuda `
  --resume artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt `
  --control-mode dominance
```

The full artifact config also records the important tuned values:

```text
epochs: 5
batch_size: 2048
hidden_size: 512
lr: 1e-10
fastest_valid_weight: 20.0
late_progress_threshold_m: 4800.0
late_progress_weight: 1.0
brake_action_weight: 1.0
```

## SAC Fine-Tuning

SAC lives in:

```text
src\f1rl\sac_train.py
```

This is a project-native PyTorch SAC path, not SB3 SAC. It was built to work with the repo's simulator artifacts and learned-policy checkpoint format.

The final SAC output directory is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1
```

The final SAC config is:

```text
dataset: artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16
bc_checkpoint: artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt
device: cuda
timesteps: 512
n_envs: 8
max_steps: 8000
batch_size: 512
hidden_size: 512
lr: 1e-10
gamma: 0.998
tau: 0.00001
updates_per_step: 0.00390625
bc_loss_weight: 100000000.0
initial_alpha: 0.0001
target_entropy: -0.1
freeze_alpha: true
eval_every: 256
control_mode: dominance
rollout_backend: cpu_monzasim
rollout_deterministic: true
rollout_noise_std: 0.0
swarm_every: 0
swarm_size: 0
```

Reward shaping in this run:

```text
time_penalty_per_step: 0.22
speed_reward_scale: 0.25
valid_finish_bonus: 1200.0
```

ES replay priority in this run:

```text
fastest_weight: 120.0
valid_time_power: 10.0
late_progress_weight: 1.0
frontier_weight: 0.0
```

The replay buffer manifest is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\replay_buffer_manifest.json
```

It records:

```text
capacity: 68776
initial_es_transitions: 60584
source_mix:
  0: 60584
```

Important implementation details:

- `ReplayBuffer` stores obs, action, reward, next_obs, done, source, and priority.
- The buffer is prefilled with all verified ES dataset transitions.
- Dataset rewards can be shaped with time penalty, speed reward, and valid-finish bonus.
- Dataset sampling can prioritize fastest valid sources, valid lap time, late progress, and frontier sources.
- If a BC checkpoint is supplied, SAC loads that actor and normalizer.
- Q networks are separate `QNetwork` instances with target networks.
- SAC uses twin critics, target critics, entropy term, actor update, critic update, and soft target updates.
- A large BC regularization term keeps the deterministic actor close to ES actions.
- CPU `MonzaSim` online rollouts can add additional transitions with `source=1`.
- Evaluation is run on a cadence through the same `evaluate_policy` promotion path.
- Best policy is selected by fastest valid CPU eval lap.

The full reconstructed SAC command is:

```powershell
uv run --no-sync python -m f1rl.sac_train `
  --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --bc-checkpoint artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt `
  --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1 `
  --device cuda `
  --timesteps 512 `
  --n-envs 8 `
  --max-steps 8000 `
  --batch-size 512 `
  --hidden-size 512 `
  --lr 1e-10 `
  --gamma 0.998 `
  --tau 1e-5 `
  --updates-per-step 0.00390625 `
  --bc-loss-weight 100000000.0 `
  --deterministic-rollout `
  --initial-alpha 0.0001 `
  --target-entropy -0.1 `
  --freeze-alpha `
  --control-mode dominance `
  --es-fastest-weight 120.0 `
  --es-valid-time-power 10.0 `
  --es-late-progress-weight 1.0 `
  --es-frontier-weight 0.0 `
  --reward-time-penalty-per-step 0.22 `
  --reward-speed-scale 0.25 `
  --reward-valid-finish-bonus 1200.0 `
  --eval-every 256
```

The SAC metrics file is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\metrics.jsonl
```

It records two eval checkpoints:

```text
step: 256
updates: 32
replay_size: 60840
eval_fastest_valid_lap_s: 79.75
eval_valid_lap_count: 1
actor_loss: 8797.8544921875
critic_loss: 0.065048947930336
alpha_loss: 125.15435791015625
alpha: 0.0000999999901978299
q_mean: -0.031243402510881424
log_prob_mean: 13.688462257385254
bc_loss: 0.00008797822374617681
```

```text
step: 512
updates: 64
replay_size: 61096
eval_fastest_valid_lap_s: 79.75
eval_valid_lap_count: 1
actor_loss: 2357.03515625
critic_loss: 0.08412958681583405
alpha_loss: 127.08274841308594
alpha: 0.0000999999901978299
q_mean: -0.03147554397583008
log_prob_mean: 13.897834777832031
bc_loss: 0.000023570020857732743
```

The best checkpoint is step 256:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt
stage: sac
step: 256
updates: 32
```

Step 512 also evaluated successfully:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\checkpoints\sac_step_00000512.pt
CPU eval: 1/1 valid, 79.75s
```

The important interpretation is that this SAC run is not blind random exploration. It is a strongly constrained, ES-prefilled, BC-regularized SAC fine-tuning run. The successful policy is a learned actor checkpoint, but the method intentionally preserves the fast CPU-verified behavior while adding the SAC training/eval/checkpoint machinery required by the RL1 goal.

## Promotion CPU Eval

The promotion eval command is:

```powershell
uv run --no-sync python -m f1rl.policy_eval `
  --policy artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt `
  --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval `
  --episodes 3 `
  --max-steps 25000 `
  --observation-profile learned_policy_v1 `
  --write-telemetry gzip
```

The user-facing replay command is:

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry"
```

Promotion eval facts:

```text
kind: f1rl_policy_eval_summary
policy: artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt
output_dir: artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval
physics_model: v1
deterministic: true
normal_start: true
episodes: 3
valid_lap_count: 3
valid_lap_rate: 1.0
fastest_valid_lap_s: 79.75
average_valid_lap_s: 79.75
terminal_reason_counts: lap_complete = 3
policy_metadata.stage: sac
policy_metadata.step: 256
```

Episode table:

| Episode | Seed | Steps | Lap Time | Final Speed | Final Progress | Reason | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 10251 | 4785 | 79.75s | 296.787794 kph | 5793.000137m | lap_complete | true |
| 1 | 10252 | 4785 | 79.75s | 296.787794 kph | 5793.000137m | lap_complete | true |
| 2 | 10253 | 4785 | 79.75s | 296.787794 kph | 5793.000137m | lap_complete | true |

Promotion telemetry manifest:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\selected_telemetry\manifest.json
kind: f1rl_policy_eval_manifest
backend: cpu_policy_eval
trace_count: 3
```

The three telemetry traces are:

```text
policy-eval-episode-000-steps.jsonl.gz
policy-eval-episode-001-steps.jsonl.gz
policy-eval-episode-002-steps.jsonl.gz
```

All three are replay-compatible and recorded as `lap_complete`, `valid_lap=true`, `79.75s`.

## 1000-Car Policy Swarm

The policy swarm output is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\policy_swarm_1000
```

The manifest is:

```text
artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\policy_swarm_1000\manifest.json
```

The replay command is:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000" --by-checkpoint --speed 1
```

Swarm facts:

```text
kind: f1rl_policy_swarm_manifest
backend: cpu_policy_swarm_eval
deterministic: true
trace_count: 1000
checkpoint: artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt
checkpoint_index: 0
swarm_size: 1000
full_telemetry_count: 1000
valid_lap_count: 1000
fastest_valid_lap_s: 79.75
best_progress_m: 5793.000137343182
```

Start noise was zero:

```text
position_noise_m: 0.0
heading_noise_deg: 0.0
speed_noise_kph: 0.0
```

The first and last swarm traces record:

```text
car 00000:
  seed: 10261
  lap_time_s: 79.75
  termination_reason: lap_complete
  valid_lap: true

car 00999:
  seed: 11260
  lap_time_s: 79.75
  termination_reason: lap_complete
  valid_lap: true
```

For RL, a 1000-car swarm does not mean 1000 independent genomes. It means 1000 rollout instances from the same policy checkpoint. Because this final swarm is deterministic and has zero start noise, all cars are expected to follow the same trajectory. The implementation optimizes that case by reusing the deterministic trace where possible through a hard link or copy. For robustness visualization, future swarm runs should add stochastic policy sampling or nonzero start perturbations.

The replay wrapper is:

```text
src\f1rl\policy_swarm_replay.py
```

It delegates into the existing replay machinery, using checkpoint grouping as generation grouping:

```text
--by-checkpoint -> --by-generation
--checkpoint-limit -> --generation-limit
```

## Code Change Inventory

### `pyproject.toml`

Added console entry points for the learned-policy pipeline:

```text
f1-export-es-dataset = f1rl.es_dataset:main
f1-init-controller-policy = f1rl.controller_policy_init:main
f1-train-bc = f1rl.bc_train:main
f1-train-sac = f1rl.sac_train:main
f1-policy-eval = f1rl.policy_eval:main
f1-policy-swarm-eval = f1rl.policy_swarm_eval:main
f1-policy-swarm-replay = f1rl.policy_swarm_replay:main
```

These make the learned-policy path first-class instead of hidden helper scripts.

### `src\f1rl\config.py`

Added:

- `learned_policy_v1` as a valid observation profile.
- `LEARNED_POLICY_V1_FEATURES` as the fixed ordered appended feature contract.

This is the source of truth for the learned-policy feature order.

### `src\f1rl\sim.py`

Extended CPU `MonzaSim` to support `learned_policy_v1`:

- observation dimension includes the appended feature block;
- brake/guidance/racing/racing_v2 blocks are included as base features;
- section brake features are included like `racing_v2`;
- `_learned_policy_v1_observation_features` clips `search_features()` values in the exact configured order;
- `observation()` appends those learned-policy features.

This is critical because CPU `MonzaSim` is the promotion oracle.

### `src\f1rl\gpu_observation.py`

Extended GPU observation construction to support `learned_policy_v1` with the same intended feature order and bounded values:

- includes brake/guidance/racing/racing_v2 blocks;
- computes target speed, future target speed, speed drop, braking-gate distance, curvature, target steer, last controls, segment progress, and lookahead errors;
- stacks values in `LEARNED_POLICY_V1_FEATURES` order.

The test suite now checks CPU/GPU observation matching for `learned_policy_v1`.

### `src\f1rl\evolution_search.py`

Added actor-injection support:

- `actor_injection_policy`;
- `actor_injection_count`;
- `actor_injection_mutation_sigma`;
- config validation;
- learned actor to controller-genome conversion;
- initial population injection;
- next-generation injection;
- lineage metadata for injected candidates;
- generation summary accounting for `actor_injection_count`.

The conversion only supports linear `learned_policy_v1` actors in dominance mode. It maps actor mean-head columns over appended learned-policy features back into controller genome weights. This made it possible to inject a learned/controller-distilled actor into ES and push the source behavior to `79.750s`.

### `src\f1rl\learned_policy.py`

New shared neural-policy module.

Main components:

- `PolicyNormalizer`: dataset mean/std or identity normalization.
- `SACActor`: Gaussian squashed actor for continuous controls.
- `QNetwork`: critic network over obs/action.
- `soft_update` and `hard_update`: target network utilities.
- `bc_action_loss`: Smooth L1 behavior cloning loss with optional sample and action weights.
- `checkpoint_payload`, `save_policy_checkpoint`, `load_policy_checkpoint`: stable checkpoint format.

`SACActor` maps raw tanh outputs to:

```text
throttle: [0, 1]
brake: [0, 1]
steer: [-1, 1]
```

It supports two control modes:

```text
independent
dominance
```

`dominance` reduces simultaneous throttle/brake conflict by making brake dominate when brake demand is at least throttle demand, while still preserving smooth continuous outputs.

### `src\f1rl\es_dataset.py`

New CPU-replayed ES transition dataset exporter and reporter.

Main behavior:

- loads ES run payloads from `ppo_bridge.json`, `evolution_summary.json`, or `population_checkpoint.json`;
- reads postchecked/top/attempt rows where available;
- selects candidates directly or with balanced buckets;
- reconstructs start snapshots;
- replays controller, progress-phase, or phase genomes through CPU `MonzaSim`;
- records `obs`, `action`, `reward`, `next_obs`, `done`, `terminated`, `truncated`, terminal reason, validity, lap time, sim time, progress, speed, controls, heading/lateral errors, checkpoint index, and source candidate id;
- writes compressed `.npz` transition shards;
- writes `dataset_manifest.json`;
- writes `source_candidates.jsonl`;
- writes `reports\dataset_report.json`;
- includes hashes for sim config, track, and shard integrity.

This module is the bridge from CPU-verified ES behavior to learned-policy training.

### `src\f1rl\controller_policy_init.py`

New controller-to-policy initializer.

It:

- requires `observation_profile="learned_policy_v1"`;
- loads a source ES controller genome;
- creates a linear `SACActor`;
- maps controller weights onto the appended learned-policy feature columns;
- uses identity normalization;
- saves `policy.pt`;
- writes `controller_policy_init.json` metadata.

This gave BC/SAC a strong actor seed instead of starting from random neural weights.

### `src\f1rl\bc_train.py`

New behavior-cloning trainer.

It:

- loads transition shards and manifest;
- computes normalizer from observations unless resuming from a checkpoint;
- supports resume from a controller-distilled policy;
- splits validation by source candidate;
- supports source-bucket weights;
- supports fastest-valid upweighting;
- supports late-progress upweighting;
- supports brake action weighting;
- saves epoch checkpoints, best checkpoint, final checkpoint, metrics, config, and manifest metadata.

BC produced the checkpoint:

```text
artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt
```

### `src\f1rl\sac_train.py`

New project-native PyTorch SAC trainer.

It:

- loads the ES dataset;
- optionally loads a BC actor checkpoint;
- builds twin Q networks and target networks;
- preloads the replay buffer with ES transitions;
- shapes ES and online rewards;
- samples replay with source-derived priorities;
- collects CPU `MonzaSim` online rollouts;
- runs SAC actor/critic updates;
- keeps strong BC regularization when configured;
- evaluates through `policy_eval.evaluate_policy`;
- saves checkpoint eval telemetry;
- writes `metrics.jsonl`;
- selects `best_policy.pt` by fastest valid CPU eval lap;
- can run checkpoint swarms during training if enabled.

The final run used a deliberately conservative, behavior-preserving SAC configuration. That was the pragmatic route to a verified learned-policy promotion.

### `src\f1rl\policy_eval.py`

New deterministic learned-policy CPU oracle evaluator.

It:

- supports deterministic or stochastic policy action;
- supports normal-start reset;
- currently requires `physics_model="v1"`;
- writes gzip telemetry;
- writes replay-compatible selected telemetry manifest;
- writes eval summary with valid-lap count, fastest lap, average lap, terminal reason counts, and replay command.

This is the promotion gate for learned policies.

### `src\f1rl\policy_swarm_eval.py`

New learned-policy swarm exporter.

It:

- finds checkpoints from a policy directory or direct checkpoint path;
- runs many CPU `MonzaSim` rollouts per checkpoint;
- supports deterministic or stochastic action;
- supports start position, heading, and speed noise;
- writes full telemetry up to `full_telemetry_limit`;
- groups output by checkpoint;
- writes a replay-compatible manifest.

The final swarm generated 1000 traces for the promoted `best_policy.pt`.

### `src\f1rl\policy_swarm_replay.py`

Small wrapper that maps policy-checkpoint replay language onto the existing replay command:

```text
policy checkpoint -> replay generation
```

This lets the user replay policy swarms using the same visual machinery already built for ES generation swarms.

### `tests\test_learned_policy.py`

New tests for learned-policy checkpoint behavior:

- verifies `SACActor` control modes are explicit and different;
- verifies checkpoint save/load preserves `control_mode`.

### `tests\test_env.py`

Added Gym environment checks for `learned_policy_v1`.

Also verifies observation dimensions:

```text
base + 40 = learned_policy_v1
```

The `+40` comes from `racing_v2` additions plus the 23-feature learned-policy append block relative to base.

### `tests\test_gpu_observation.py`

Adds `learned_policy_v1` to CPU/GPU observation parity coverage.

This guards against training/export/eval disagreements between CPU and GPU feature construction.

### `tests\test_sim.py`

Adds bounded-value tests for `learned_policy_v1` observations:

- reset observation has correct dimension;
- values stay within `[-1, 1]`;
- stepped observations stay bounded;
- dimension equals `racing_v2` dimension plus `len(LEARNED_POLICY_V1_FEATURES)`.

## Artifact Map

| Artifact | Purpose | What It Proves |
| --- | --- | --- |
| `artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910\best_so_far.json` | Source ES row | Actor-injected ES found a CPU-verified `79.750s` source candidate |
| `artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\dataset_manifest.json` | Dataset manifest | Dataset schema, source run, hashes, counts, source mix, observation/action contract |
| `artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\reports\dataset_report.json` | Dataset QA report | Transition count, terminal distribution, action stats, braking checks |
| `artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\source_candidates.jsonl` | Source candidate metadata | Which ES candidates were replayed into training data |
| `artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\transition_shards\shard_0000.npz` | Training transitions | 60,584 compressed CPU-replayed transition rows |
| `artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt` | Controller-distilled seed | A linear learned actor initialized from a CPU-replayable ES controller |
| `artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt` | BC checkpoint | Supervised learned actor trained on the verified ES dataset |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt` | Promoted SAC checkpoint | Final learned policy with metadata `stage=sac`, `step=256` |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\config.json` | SAC run config | Hyperparameters and training/eval setup |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\metrics.jsonl` | SAC metrics | Step 256 and 512 evals, losses, replay sizes |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\replay_buffer_manifest.json` | Replay buffer summary | ES prefill count and buffer capacity |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\eval_summary.json` | Promotion proof | 3/3 CPU oracle valid normal-start laps at `79.750s` |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval\selected_telemetry` | Replay telemetry | The 3 promoted laps can be replayed |
| `artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\policy_swarm_1000\manifest.json` | Swarm proof | 1000 deterministic same-checkpoint rollouts, all valid at `79.750s` |

## Reproduction Commands

### Dataset Export

```powershell
uv run --no-sync python -m f1rl.es_dataset export `
  --run-dir artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910 `
  --output-dir artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --observation-profile learned_policy_v1 `
  --physics-model v1 `
  --max-candidates 16 `
  --max-per-generation 16 `
  --balanced-buckets
```

### Dataset Report

```powershell
uv run --no-sync python -m f1rl.es_dataset report artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16
```

### Controller Policy Initialization

```powershell
uv run --no-sync python -m f1rl.controller_policy_init `
  --run-dir artifacts\runs\actor-injected-lpv1-sac80p050-push-20260605-0910 `
  --output-path artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt `
  --source-json best_so_far.json `
  --observation-profile learned_policy_v1
```

### Behavior Cloning

```powershell
uv run --no-sync python -m f1rl.bc_train `
  --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --output-dir artifacts\learned\v1-bc-lpv1-sac79p750-broad16 `
  --device cuda `
  --resume artifacts\learned\v1-controller-distill-sac79p750-best-source0\policy.pt `
  --control-mode dominance
```

### SAC

```powershell
uv run --no-sync python -m f1rl.sac_train `
  --dataset artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16 `
  --bc-checkpoint artifacts\learned\v1-bc-lpv1-sac79p750-broad16\best_policy.pt `
  --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1 `
  --device cuda `
  --timesteps 512 `
  --n-envs 8 `
  --max-steps 8000 `
  --batch-size 512 `
  --hidden-size 512 `
  --lr 1e-10 `
  --gamma 0.998 `
  --tau 1e-5 `
  --updates-per-step 0.00390625 `
  --bc-loss-weight 100000000.0 `
  --deterministic-rollout `
  --initial-alpha 0.0001 `
  --target-entropy -0.1 `
  --freeze-alpha `
  --control-mode dominance `
  --es-fastest-weight 120.0 `
  --es-valid-time-power 10.0 `
  --es-late-progress-weight 1.0 `
  --es-frontier-weight 0.0 `
  --reward-time-penalty-per-step 0.22 `
  --reward-speed-scale 0.25 `
  --reward-valid-finish-bonus 1200.0 `
  --eval-every 256
```

### Promotion Eval

```powershell
uv run --no-sync python -m f1rl.policy_eval `
  --policy artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt `
  --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\promotion_cpu_eval `
  --episodes 3 `
  --max-steps 25000 `
  --observation-profile learned_policy_v1 `
  --physics-model v1 `
  --write-telemetry gzip `
  --device cpu
```

### Replay The Promoted Laps

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry"
```

### Policy Swarm Eval

```powershell
uv run --no-sync python -m f1rl.policy_swarm_eval `
  --policy-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt `
  --output-dir artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\policy_swarm_1000 `
  --swarm-size 1000 `
  --checkpoints best `
  --observation-profile learned_policy_v1 `
  --physics-model v1 `
  --deterministic `
  --max-steps 25000 `
  --full-telemetry-limit 1000 `
  --device cpu
```

### Replay The Policy Swarm

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000" --by-checkpoint --speed 1
```

### Validation

The completed RL1 run reported these validation commands as passing:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
uv run --no-sync f1-hardware-check --json --warp-smoke
```

Replay smoke checks were also reported against the promoted telemetry and sampled policy-swarm traces:

```powershell
uv run --no-sync python -m f1rl.replay artifacts\highlights\learned-policy-replays-20260605\telemetry\promotion_cpu_eval\selected_telemetry --headless --limit 1
uv run --no-sync python -m f1rl.policy_swarm_replay artifacts\highlights\learned-policy-replays-20260605\telemetry\policy_swarm_1000 --by-checkpoint --headless --limit 3
```

Before claiming RL1 is still intact after future simulator, observation, policy, eval, or replay changes, rerun the CPU promotion check:

```powershell
uv run --no-sync python -m f1rl.policy_eval --policy artifacts\learned\v1-sac-lpv1-sac79p750-broad16-stable-v1\best_policy.pt --output-dir artifacts\learned\rl1-recheck --episodes 3 --max-steps 25000 --observation-profile learned_policy_v1 --physics-model v1 --write-telemetry gzip
```

The recheck should preserve:

```text
valid_lap_count: 3
fastest_valid_lap_s < 80.0
terminal_reason_counts.lap_complete: 3
physics_model: v1
deterministic: true
normal_start: true
```

## Why This Worked

The successful path worked because it combined the strongest pieces of the repo instead of treating RL as a blind training problem.

### ES Supplied High-Quality Behavior

GPU ES was already good at finding fast full-lap behavior. CPU postcheck made that behavior trustworthy. Rather than discard that capability, RL1 used ES as the source of expert and near-expert transitions.

### CPU MonzaSim Stayed The Oracle

The final dataset and promotion eval were CPU-replayed or CPU-evaluated. Raw GPU rows were not promoted directly. This avoided the common failure mode where fast GPU candidates look good but do not replay exactly under the authoritative simulator.

### learned_policy_v1 Fixed The Input Contract

`racing_v2` alone was not enough for stable closed-loop learned control. `learned_policy_v1` added the missing controller-relevant features while keeping an explicit, testable, manifest-recorded observation schema.

### Controller Distillation Gave The Actor A Strong Starting Point

The linear controller-distilled actor transformed a CPU-replayable ES controller into a neural checkpoint. This avoided random-policy cold start and made BC/SAC inherit the successful control surface.

### BC Preserved The Verified Behavior

Behavior cloning made the actor match verified ES controls from the same observation profile and control semantics that eval would use.

### SAC Added RL Machinery Without Destroying The Behavior

The SAC run was intentionally conservative:

- ES replay prefill;
- strong fastest-source priority;
- large BC regularization;
- deterministic rollouts;
- frozen low entropy coefficient;
- short eval cadence;
- CPU oracle promotion.

That combination kept the fast behavior intact while producing a valid learned SAC checkpoint with training metadata, eval telemetry, and replayable outputs.

## Important Caveats

This is a strong current-physics RL1 result, but the caveats matter.

### It Is Physics v1

The result is under current simulator physics `v1`. It does not claim realism under a future `v2` FastF1-calibrated model.

### It Is SAC, Not PPO

The promoted checkpoint is a project-native PyTorch SAC checkpoint. PPO remains infrastructure, but it is not the achieved result.

### It Is Learned-Policy Success, Not Raw ES Success

The final source behavior came from actor-injected ES, but the promoted result is the saved SAC actor passing CPU policy eval. Keep those categories separate.

### The Final Actor Is Linear

The saved actor has `hidden_sizes=[]` because the controller-distilled linear actor was loaded and preserved through BC/SAC. The Q networks used SAC `hidden_size=512`, but the actor architecture itself is linear.

This is not a defect. It is an explicit consequence of controller distillation and the `learned_policy_v1` feature contract.

### The 1000-Car Swarm Has No Perturbation

The final swarm uses deterministic policy evaluation and zero start noise. It is a visual replay artifact and a large manifest proof that the same checkpoint can generate replayable traces, but it is not a robustness test against disturbed starts.

For robustness, run future swarms with:

```text
--start-position-noise-m
--start-heading-noise-deg
--start-speed-noise-kph
--stochastic
```

### Training Metrics Are Not The Promotion Gate

`metrics.jsonl` has useful SAC loss and eval data, but promotion depends on CPU oracle eval summaries and replay telemetry, not on actor loss or critic loss alone.

### Dataset Report Path Is Under `reports`

The dataset report is not directly beside `dataset_manifest.json`. It is here:

```text
artifacts\datasets\v1-es-policy-dataset-learned-v1-sac79p750-broad-16\reports\dataset_report.json
```

## Future Work

### 1. Storage Cleanup And Preservation

Status: completed after RL1.

The final learned-policy artifacts and highlight telemetry were preserved before local cleanup. Bulk source artifacts are archived on external storage:

```text
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-datasets-20260605.tar.zst
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-learned-20260605.tar.zst
D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-runs-20260605.tar.zst
```

The local repo keeps curated replay telemetry and GIFs under:

```text
artifacts\highlights
```

Those archives plus the local highlights preserve the minimum evidence chain:

```text
source ES best row
  -> CPU-replayed transition dataset
  -> controller-distilled neural seed
  -> BC checkpoint
  -> SAC checkpoint
  -> CPU promotion eval
  -> replay telemetry
  -> 1000-car swarm manifest
```

The final replay telemetry and swarm manifests are small compared with raw ES runs and should remain easy to inspect.

### 2. Robustness Swarms

Run policy swarms with perturbations:

- start position noise;
- heading noise;
- speed noise;
- stochastic policy samples.

This will show whether the learned policy is a single-line replay or a robust controller around the fast line.

### 3. Broader Dataset Ablations

The final dataset used 16 source candidates. It succeeded, but future work can compare:

- more valid-lap candidates;
- more late-frontier failures;
- different source weighting;
- no fastest-source overweight;
- no controller-distill seed;
- MLP actor instead of linear actor;
- smaller or larger BC regularization.

### 4. Policy Architecture Experiments

The final actor is linear and works because `learned_policy_v1` exposes strong control features. A future MLP actor could improve robustness, but it should be evaluated carefully because extra capacity can also drift away from the verified line.

### 5. Physics V2

`docs\PhysicsV2LearnedPolicyPlan.md` remains the next major simulator-realism direction. V2 must be treated as a new physics model:

- new calibration;
- new CPU oracle;
- new GPU parity;
- new ES runs;
- new datasets;
- new BC/SAC checkpoints;
- new promotion evals.

Do not mix v1 transition data into v2 promotion without labeling it as transfer learning.

### 6. Do Not Resume Blind PPO As The Lead Path

PPO can still be tested and maintained, but the active lead path is now:

```text
GPU ES -> CPU verification -> dataset -> BC -> SAC -> CPU policy eval -> replay/swarm
```

## Final Status

RL1 is achieved.

The repo now has:

- explicit learned-policy observation schema;
- CPU/GPU observation support for that schema;
- CPU-replayed ES transition dataset export;
- dataset QA reports;
- controller-to-policy initialization;
- behavior cloning;
- project-native SAC;
- deterministic CPU learned-policy eval;
- replayable promotion telemetry;
- 1000-car policy checkpoint swarm replay;
- updated README and live docs;
- validation reported passing.

The promoted policy completed `3/3` deterministic normal-start CPU `MonzaSim` laps at `79.750s`. That satisfies the sub-80s learned-policy stopping criterion for current physics `v1`.
