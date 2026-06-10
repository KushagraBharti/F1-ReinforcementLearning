# RL2 Plan: Real Reinforcement Learning, From Scratch

Status: design document, 2026-06-10.
Depends on: `docs/PhysicsV3-Plan.md` (dynamic model, real track widths, stochastic hooks, GPU batch sim).

The one-sentence goal: **a policy trained from random initialization by reinforcement learning alone — no demonstrations, no behavior cloning, no guidance features — that laps Monza robustly under randomized conditions and drives circuits it never saw in training.** Plus the swarm visual, done honestly: thousands of *actual training environments* rendered live, not hardlinked replays of one trajectory.

---

## 1. What went wrong before, stated as rules

The RL1/V2 record (this repo's own manifests) shows the failure modes precisely. They become hard rules:

| # | Rule | What it prevents (evidence from this repo) |
|---|---|---|
| R1 | No demonstrations anywhere in the training loop. ES may exist only as a *baseline* to beat. | The "learned policy" being a clone (BC `overfit_source_id=3`, val MAE 0.004) |
| R2 | No solution-bearing observation features: no `target_steer`, no target speeds, no brake gates, no per-section anything. Track *map* in, driving *decisions* out. | The linear "controller" genome riding a hand-built racing assistant |
| R3 | Every reported number is a **distribution over ≥ 500 stochastic episodes** (randomized starts, physics, latency). Deterministic single-episode evals are debug tools, never results. | 77.6833 s == its ES source to four decimals; swarm of 1,000 hardlinked copies of one trace |
| R4 | A checkpoint may be promoted only if it **improves on its own initialization** under the stochastic eval. | SAC promoting at `step: 0, updates: 0` |
| R5 | One generic action space, one generic reward, all tracks. Any track-specific constant in reward or obs is a bug. | 14 hand-built action sets (`delayed_left`, `exit_tiny`, …), the scaffold/assist graveyard |
| R6 | Keep RL1's *good* habits: CPU-oracle verification of GPU results, full config lineage in checkpoints, seeded reproducibility. | (the part that was already right) |

---

## 2. Environment specification

### 2.1 Observation (the honest interface)

All features computable by a real car with a map and state estimation — nothing that encodes *what to do*.

**Ego block (10):**
```
v_x/v_max,  v_y/20,  r/3,  β = atan2(v_y, v_x)/1.0,
α_f/0.3,  α_r/0.3,                 # slip awareness → slide-catching is learnable
a_x/50, a_y/50 (filtered),
gear/8,  rpm/rpm_max
```

**Pose-relative-to-track block (4):**
```
n / (w(s)/2),                      # normalized lateral offset, ±1 = edge
sin(Δψ), cos(Δψ),                  # heading error vs centerline tangent
ṡ / v_max                          # progress rate
```

**Track preview block (2×K, K=20):** curvature and width sampled ahead at log-spaced arc-length offsets
```
d_k ∈ {5, 7, 10, 14, 20, 28, 39, 55, 77, 108, 151, 211, 296, 414, 580, 800, 1000, 1250, 1500, 1800} m
features: κ(s+d_k)·30,  w(s+d_k)/20
```
Log spacing: fine resolution where the next apex is, coarse awareness of what's coming at 300 kph. This is a *map interface* — the standard formulation in the autonomous-racing literature (e.g. Sony GT Sophy, ETH/UZH drone racing use exactly this pattern) — and it generalizes across tracks, which fixed ray sensors and per-track features do not.

**Last action (2)** for smoothness awareness. Total: **56 floats.** Running mean/std normalization learned during training, frozen at eval, stored in the checkpoint.

### 2.2 Action

2-D continuous, 60 Hz with optional action-repeat 2 (30 Hz decisions):
```
a₁ ∈ [−1, 1]: steering rate or position (position first; rate is a refinement)
a₂ ∈ [−1, 1]: longitudinal axis — a₂>0 throttle, a₂<0 brake
```
The single longitudinal axis kills the simultaneous-throttle-and-brake pathology *structurally* (RL1's dataset had 13.6% overlapping pedal steps) and needs no "dominance" post-processing.

### 2.3 Reward (minimal, generic, shaping-free)

```
r_t = w_s·Δs_t                     # progress along centerline this step [m]; w_s = 0.1
    − w_a·‖a_t − a_{t−1}‖²         # actuator smoothness; w_a = 0.05
    + terminal: 0 on lap complete (progress already paid for it)
               −10 on off_track / spin / stalled
```

That is the whole reward. Reasoning:
- `Δs` is (up to a constant) a **potential-based shaping** of lap completion with potential Φ = s, so it does not change the optimal policy ordering — maximizing total progress per unit time *is* minimizing lap time. No speed targets, corridors, or brake bonuses: those encode the engineer's driving theory, and the entire point is for the policy to derive its own.
- The smoothness term is the one concession to realism (real actuators, and it visibly improves learned lines); it is track-generic.
- Termination penalty small relative to a lap's progress (~580): crashing is bad mostly because progress stops, which keeps the incentive landscape simple.

### 2.4 Episodes, randomization, horizon

- Reset: `s₀ ~ U(0, L)`, lateral `n₀ ~ U(−0.6, 0.6)·w/2`, heading error `U(±8°)`, speed `U(10, 70) m/s` clipped to the local quasi-steady envelope. Random starts everywhere on track are an *exploration* device (the Atari random-start trick), not a guidance device — final evaluation still includes standing starts.
- Physics randomization + latency + obs noise per `PhysicsV3-Plan.md` §5, on during all training.
- Truncation at 2,400 decision steps (~80 s at 30 Hz); **bootstrap `V(s_T)` on truncation** — getting this wrong silently caps the horizon and is the most common PPO bug.
- `γ = 0.998`, effective horizon ≈ 500 steps ≈ 17 s — enough to credit a braking decision with its corner-exit consequence; full-lap credit flows through the value function.

---

## 3. Algorithm: PPO, implemented from scratch, end-to-end on GPU

PPO is the right choice: on-policy stability at massive parallelism is exactly what a 4,096-env GPU sim feeds best (the Isaac Gym / OpenAI Five / GT Sophy recipe). The repo already has a from-scratch GPU PPO skeleton (`gpu_ppo.py`) — RL2 rewrites it against the V3 batch sim.

### 3.1 The math (written down so the implementation can be audited against it)

Advantages by GAE:
```
δ_t = r_t + γ·V_θ(s_{t+1})·(1−done_t) − V_θ(s_t)
Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l},   λ = 0.95
```
Clipped surrogate with entropy bonus, ratio ρ_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t):
```
L(θ) = E_t[ min(ρ_t·Â_t, clip(ρ_t, 1−ε, 1+ε)·Â_t) ] − c_v·E_t[(V_θ(s_t) − V_targ,t)²] + c_e·E_t[H(π_θ(·|s_t))]
ε = 0.2,  c_v = 0.5,  c_e: 0.004 → 0 linear
```
Policy head: diagonal Gaussian over pre-tanh actions, state-independent learnable `log σ` init −0.5; tanh-squash with the log-det-Jacobian correction (already correct in `learned_policy.py` — reuse).

### 3.2 Implementation details that decide success (the checklist)

From the "37 implementation details of PPO" literature plus this project's history; each is a known silent failure if skipped:

1. Orthogonal init (gain √2 hidden, 0.01 policy head, 1.0 value head).
2. Running obs normalization; clip normalized obs to ±10; freeze stats at eval.
3. Reward scaling by running std of *returns* (not rewards).
4. Advantage normalization per update batch.
5. Truncation bootstrapping (§2.4) — distinct `terminated` vs `truncated` paths.
6. LR 3e-4 → 0 linear anneal; Adam ε = 1e-5.
7. Grad-norm clip 0.5.
8. 4 epochs, minibatch 65,536 from a 4,096-env × 256-step rollout (~1.05 M transitions per update).
9. Early-stop epoch when approx-KL > 0.02 (cheap insurance against destructive updates).
10. Network: MLP 2×512 (separate policy/value trunks); optional GRU-256 variant once latency/noise are on (partial observability makes recurrence earn its cost).
11. Everything resident on GPU: sim state, rollout buffer, learner. The only CPU traffic is logging and periodic oracle evals.
12. Determinism contract: seeded runs reproduce learning curves within noise; every checkpoint stores config lineage + normalizer (keep RL1's discipline).

### 3.3 Compute reality check

V3 Tier-1 physics ≈ 60 fused FLOPs-heavy ops/step; with 4,096 envs at 4 substeps expect ≥ 200k env-steps/s on the local RTX-class GPU (the V2 Warp kernel did far more work per step at scale). PPO learner adds ~20%. Budget:
- Single-track competence: ~200–500 M steps → **half a day to 2 days**, one GPU.
- Multi-track + robustness: ~1–2 B steps → about a week of background training. Feasible; plan checkpoints/resume accordingly (the ES infra already has the pattern).

---

## 4. The swarm visual, done honestly

The training loop *is* the swarm: 4,096 real environments at different track positions, speeds, and noise draws.

- **Live view**: render a decimated 1,024 cars from actual GPU state every N updates, colored by return percentile (red = bottom, green = top). Early training is glorious chaos — cars spinning everywhere — converging over hours into a coherent fast river of traffic. This is *better* footage than ES generations, and every car is real.
- **Training timelapse**: auto-capture the live view at updates {0, 50, 200, 1k, 5k, 20k, final} → one GIF of "chaos → racing." This replaces the hardlinked swarm as the README hero.
- **Honest swarm replay**: 1,000 *distinct* evaluation episodes (different seeds/noise) rendered together. Spread between cars now visualizes the policy's actual robustness distribution instead of being a rendering trick.
- Keep the exact-pygame GIF exporter; it was good.

---

## 5. Evaluation protocol and baselines

### 5.1 Stochastic evaluation (the only kind that produces reportable numbers)

1,000 episodes, randomized per §2.4, CPU-oracle verified (rule R6). Report:
```
valid-lap rate | lap time mean / p5 / p95 | spin rate | off-track rate
```
plus a robustness matrix: each cell = valid-lap rate under {μ −10%, mass +3%, +2-step latency, obs noise ×2, wind 8 m/s} × {trained track, held-out track}.

### 5.2 Baselines (a result is a comparison)

| Baseline | Why it's there |
|---|---|
| **ES pipeline (this repo)** re-run on V3 physics | the fair fight RL1 never had; prediction: ES matches RL deterministically but collapses under randomization — *show it* |
| Racing-line + pure-pursuit/PID controller | the classical-controls strawman every RL paper needs |
| Human (manual mode) | fun, relatable bar |
| BC-from-ES (RL1's method, ported) | quantifies exactly what the old approach was worth |

### 5.3 Headline claims being built (the brag list)

1. *From scratch*: random network → full-speed Monza laps, no demonstrations, no racing line, reward = progress only. Learning-curve plot from step 0.
2. *Robust*: ≥ 95% valid-lap rate under full randomization (where deterministic-replay approaches score ~0).
3. *General*: trained on {Monza, Spielberg, Silverstone, Budapest}, **zero-shot laps on Spa** (held out, never seen). This is the claim no search/cloning pipeline can fake, because Spa's trajectories were never available to memorize.
4. *Emergent racecraft*: telemetry shows trail-braking (front ellipse usage through entry), throttle-on rotation, slide-catching countersteer — none rewarded explicitly. Plot ellipse utilization vs corner phase as evidence.
5. *Scale*: thousands of envs, end-to-end GPU, from-scratch PPO with the full modern checklist — and the timelapse to show for it.

---

## 6. Milestones with kill criteria

| M | Deliverable | Gate (objective, falsifiable) | If it fails |
|---|---|---|---|
| M0 | V3 env behind Gym API; unit tests; random policy renders | seeded determinism; obs/reward bounds tests | fix env, nothing downstream starts |
| M1 | PPO core validated on Pendulum + a single-corner mini-env | matches SB3 PPO learning curve within 20% on both | debug PPO against checklist §3.2 — do not touch env/reward |
| M2 | Full Monza from scratch, randomization ON | ≥ 90% valid-lap rate stochastic eval | inspect entropy/KL/value-loss curves; widen randomization *gradually*, never add guidance |
| M3 | Beat baselines | ≥ ES lap time −0% … +3% deterministic AND ≥ 95% vs ES's < 50% under randomization | profile failure modes from telemetry; consider GRU variant |
| M4 | 4-track training, Spa zero-shot | ≥ 80% valid laps on Spa, no fine-tuning | add 2 more training tracks (generalization scales with track diversity) |
| M5 (stretch) | Multi-agent racing | 16 cars/track instance, collision discs; self-play league vs past checkpoints; overtakes without contact in eval | ship M4 results regardless; M5 is a sequel, not a dependency |

Rules R1–R5 are standing kill criteria at every gate: the moment a fix involves a demonstration, a guidance feature, or a track-specific constant, the fix is rejected by construction.

### Honest risk register

- **Reward hacking**: progress reward + real widths can still find cuts; the off-track check must use real `w(s)` from day one (the 24 m corridor is what made V2 times fake).
- **PPO sensitivity**: M1 exists precisely so algorithm bugs are caught on cheap envs, not after a day of GPU time.
- **The wall at M2**: if from-scratch never laps, the honest fallbacks (in order) are: longer horizon γ=0.999, action repeat 3, curriculum on randomization *strength* (not on track sections), GRU. BC restart is not on the list.
- **Chaotic-parity noise**: stochastic eval makes this moot at the results level; per-step parity tests cover correctness (PhysicsV3 §6.6).

---

## Appendix A: Why the earlier PPO didn't work (post-mortem)

It wasn't that PPO can't solve this — it's that it was set up to fail in about four compounding ways. The repo's own config is the fossil record.

**1. The exploration problem at Monza specifically.** From the start line you have ~600m of flat-out straight into the Turn 1 chicane, which is a near-stop corner from ~340 kph. A fresh policy maximizing progress reward quickly learns "full throttle = more progress"... and then dies at the chicane, every single episode. To discover that braking pays off, Gaussian/discrete dithering at 60 Hz would need to randomly produce a *coherent 2–3 second sustained brake* followed by a precise turn-in. Per-step noise self-cancels — the odds of that happening by chance are effectively zero. So the agent never *experiences* "survived the chicane at speed," which means the value function has nothing to bootstrap from. The reward isn't sparse, but it's **deceptive**: the gradient points toward the cliff.

The config is the fossil record of this exact fight: `overspeed_turn_in_terminate`, `no_brake_penalty`, `brake_zone_progress_multiplier`, `virtual_corridor`, scaffold brake rewards, and eventually 14 hand-built action sets like `delayed_left` and `exit_tiny`. Every one of those is a patch for "the agent won't brake for the chicane." And each patch made the MDP *more* deceptive — assist gates that terminate episodes add new cliffs, scaffold rewards add new local optima to hack — so the project ended up in a shaping arms race instead of fixing exploration.

**2. The data distribution collapsed onto the first 600 meters.** With episodes starting at the line and terminating on off-track/no-progress (`-60`/`-90` penalties, 3-second no-progress limit), early training generates thousands of episodes that all die in sector 1. The agent gets essentially zero data about Lesmo, Ascari, or Parabolica for ages. The standard fix — **random start states uniformly along the track** — was sitting right there in the ES code (`start_progress_m`) but the PPO lap-training loop started from the line. The segment-PPO experiments (`--start-progress-m 500 --target-progress-m 510`) were a hand-rolled version of this, but with gates and per-segment action sets, which turned it back into scaffolding.

**3. The sample budget was off by ~100×.** The SB3 path ran on CPU `MonzaSim` with a handful of envs. From-scratch continuous-control driving at this difficulty needs hundreds of millions to billions of steps — that's the regime GT Sophy and every Isaac Gym result lives in. At CPU throughput, training realistically lived in the 1–10M step range. The GPU PPO existed but stayed "experimental" (the docs' smoke tests run it for 8 timesteps). So PPO never actually got the one resource it trades everything else for: data volume.

**4. Horizon details.** 60 Hz decisions over a ~4,700-step lap with γ=0.995 gives an effective credit horizon of ~200 steps (~3 seconds). Enough to learn "braking prevents the crash 2s from now," not enough to learn "this entry line costs me time all the way down the next straight." No action repeat, no truncation-bootstrapping guarantees in the custom loop. None of these alone kills you; stacked on 1–3, they do.

So the honest one-liner: **PPO didn't fail — it was starved of data, trapped in a deceptive first corner with no exploration mechanism, and then buried under scaffolding meant to compensate.** This plan's answers map one-to-one: random starts everywhere (kills #1 and #2), GPU-vectorized envs at 4,096× (kills #3), action repeat + γ=0.998 + proper truncation handling (kills #4), and a hard rule against scaffolds (R5) so the arms race can't restart.

## Appendix B: Does PPO give the swarm visual?

Not inherently — classic PPO with 8 CPU envs is 8 lonely cars — but **GPU-vectorized PPO absolutely does**, and honestly a cooler version of it than ES gave:

- **During training**, you have 4,096 real environments stepping simultaneously on the GPU. Render 1,000 of them and you get dense traffic scattered all around the circuit (random reset points, so it's a flowing river rather than a pack), evolving live from "spinning chaos everywhere" early in training to "coherent racing line" hours later. Color the cars by return percentile and you can literally *watch* the policy improve. ES can't give you this — its swarm only changes between generations; this one is learning in front of you.
- **The ES aesthetic** — everyone launches together from the grid and fans out — is still available, honestly: periodically pause training, reset 1,000 envs to a synchronized start with different noise/randomization seeds, and roll out the current policy. Because each car gets different physics randomization and noise draws, they genuinely diverge — the visual spread *is* the policy's robustness distribution, instead of being 1,000 hardlinked copies of one trace like the old swarm GIF.

So the swarm isn't a property of the algorithm — it's a property of vectorized simulation. ES needed it for population evaluation; PPO exploits it for throughput. Same visual, except with PPO every car in frame is part of the actual learning process, which is a much better story to tell over the footage.
