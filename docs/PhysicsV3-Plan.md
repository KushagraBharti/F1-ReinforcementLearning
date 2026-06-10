# Physics V3 Plan: A True Dynamic Vehicle Model

Status: design document, 2026-06-10.
Scope: replaces the V2 "stabilized arcade" model with a genuinely dynamic single-track model that can understeer, oversteer, slide, spin, lock wheels, and be caught — because a policy that learns car control needs a car that can actually be lost.

---

## 1. Diagnosis: why V2 is not dynamic

V2 (`physics.py::apply_physics_v2`) looks physical but is kinematically constrained in the one place that matters:

1. **No lateral velocity state.** The car state is `(x, y, ψ, v, r)` where `v` is scalar speed along the heading. There is no body-frame sideslip `v_y`. The car always travels exactly where its nose points.
2. **Yaw rate is set algebraically, not integrated.** V2 computes `requested_yaw = v·tan(δ)/L`, caps it by tire capacity, and assigns it directly. There is no yaw moment balance and no yaw inertia. Consequences:
   - The car **cannot oversteer or spin** — yaw rate is definitionally whatever the front wheels request, saturated.
   - The Pacejka-lite tire curves only ever act as a *cap* on a kinematic formula. Front/rear force balance never creates a yaw moment, so front/rear grip distribution barely matters dynamically.
3. **Slip angles are synthetic.** `α_f = atan2(r·a, v) − δ` uses yaw rate alone (no `v_y` term), so the computed slips are an echo of the commanded steering, not an independent state of the vehicle.
4. **No combined slip.** Longitudinal capacity is derived from a *car-level* grip budget (`sqrt(a_max² − a_lat²)`), not per-axle friction ellipses. Trail-braking, throttle-on oversteer, and lift-off oversteer physically cannot exist.
5. **Patched parameters betray the structure.** `front_peak_mu = 3.28` with `mechanical_grip_low_speed_scale = 0.1925` means effective low-speed mu ≈ 0.63 and high-speed mu ≈ 2.3. Real slick mu is ~1.6–2.0 roughly constant; the speed dependence of cornering g in real F1 comes from **downforce**, which V2 under-represents (6.35 N per (m/s)² ≈ 4.9 kN at 280 kph; real F1 is ~16–20 kN). The grip-scale knob is compensating for missing aero.
6. **Gear hysteresis is dead code.** `apply_physics_v2` calls `gear_and_rpm_v2(speed, 1, v2)` with `previous_gear=1` every step, so the shift up/down logic with hysteresis never runs; gear is a pure function of speed.
7. **Pixel-space integration.** Positions integrate in pixels via `meters_per_pixel` inside the physics step. Physics should live entirely in SI units; pixels are a render concern.

None of this is a bug in execution — it was a sound choice for stability under search. But it caps what learning can demonstrate: a policy in V2 only learns *where to slow down*, never *how to control a car*.

---

## 2. Goals and non-goals

**Goals**

- G1: 3-DOF dynamic single-track (bicycle) model: `(v_x, v_y, r)` body-frame dynamics with a yaw moment balance. The car can oversteer, spin, and be caught by countersteer.
- G2: Combined-slip tire model per axle with load sensitivity. Trail-braking and traction-limited corner exit emerge from the model, not from reward shaping.
- G3: Aero done properly so cornering-g vs speed matches FastF1 *without* mu hacks.
- G4: Numerically robust at all speeds (including standstill) and stable on GPU in float32 at large batch.
- G5: Stochastic hooks (parameter randomization, control latency, noise) so that learned policies must be robust — this is a physics-layer requirement for the RL plan.
- G6: Keep the FastF1 calibration pipeline; add dynamic validation targets (g-g envelope, step-steer response).

**Non-goals**

- Per-wheel double-track model, suspension kinematics, tire thermal model, kerb 3D geometry, ERS energy management. Each is a possible V3.5+ extension; none is needed for credible dynamics.

---

## 3. The model

### 3.1 State

```
q = (x, y, ψ,  v_x, v_y, r,  δ,  gear,  [ω_f, ω_r])
```

- `x, y` position of CoG in **meters** (world frame), `ψ` yaw.
- `v_x, v_y` body-frame longitudinal/lateral velocities, `r = ψ̇` yaw rate.
- `δ` road-wheel steering angle (first-order actuator state, as in V2).
- `gear` integer state (real hysteresis this time).
- `ω_f, ω_r` wheel angular velocities — **Tier 2 only** (see §3.5).

Pixels appear only in the renderer: `px = x / meters_per_pixel`.

### 3.2 Equations of motion (3-DOF single track)

Let `a` = CoG-to-front-axle distance, `b` = CoG-to-rear-axle, `L = a + b`, `m` mass, `I_z` yaw inertia.

```
m (v̇_x − r·v_y) = F_x,r + F_x,f·cos δ − F_y,f·sin δ − F_drag − F_roll
m (v̇_y + r·v_x) = F_y,r + F_y,f·cos δ + F_x,f·sin δ
I_z · ṙ          = a·(F_y,f·cos δ + F_x,f·sin δ) − b·F_y,r

ẋ = v_x·cos ψ − v_y·sin ψ
ẏ = v_x·sin ψ + v_y·cos ψ
ψ̇ = r
```

The left-hand `−r·v_y` / `+r·v_x` terms are the centripetal coupling that V2 lacks entirely. The third equation is the yaw moment balance — **this single line is what makes oversteer exist**: if the rear axle saturates (`F_y,r` stops growing) while the front still has grip, `ṙ` grows, sideslip builds, and the car rotates beyond the kinematic rate.

### 3.3 Slip quantities

```
α_f = atan( (v_y + a·r) / max(|v_x|, v_ε) ) − δ
α_r = atan( (v_y − b·r) / max(|v_x|, v_ε) )
β   = atan( v_y / max(|v_x|, v_ε) )          # body sideslip, telemetry + spin detection
```

with `v_ε ≈ 0.5 m/s` to regularize the singularity at standstill (see §3.7 for the full low-speed treatment). Note the slips now depend on `v_y` — a genuinely independent state — unlike V2's yaw-rate-only synthetic slips.

### 3.4 Tire model: Pacejka lateral + friction-ellipse combined slip (Tier 1)

Per axle, pure lateral force from the Magic Formula:

```
F_y0(α, F_z) = D · sin( C · atan( B·α − E·(B·α − atan(B·α)) ) )

D = μ_y(F_z) · F_z                       # peak force
B = C_α / (C · D)                        # stiffness factor, C_α = cornering stiffness [N/rad]
C ≈ 1.35  (shape),  E ≈ −0.8 … 0  (curvature; controls post-peak falloff)
```

Load-sensitive friction (slicks lose mu as load grows):

```
μ_y(F_z) = μ_0 · (1 − k_μ · (F_z − F_z0) / F_z0),   k_μ ≈ 0.05–0.10
```

**Combined slip (the part V2 cannot do).** Tier 1 treats longitudinal force as a *demand* (no wheel-spin states) and derates lateral grip with a friction ellipse:

```
F_x,max(axle) = μ_x · F_z
F_x = clip(F_x,demand, −F_x,max, +F_x,max)
F_y = F_y0(α, F_z) · sqrt( max(0, 1 − (F_x / F_x,max)²) )
```

Demands come from the driver model:

- Drive: `F_x,r,demand = throttle · min(F_engine(v_x, gear), μ_x·F_z,r)` (rear-wheel drive; front gets 0).
- Brake: `F_x,f,demand = −brake · γ_bias · F_brake,max`, rear gets `−brake·(1−γ_bias)·F_brake,max`, each clipped per-axle by `μ_x·F_z`.

Why this matters: braking deep into a corner consumes the front ellipse (less `F_y,f` → understeer on entry), releasing the brake restores it (turn-in), and throttle consumes the rear ellipse (power oversteer on exit). All three canonical racing behaviors fall out of two equations.

Everything here is smooth except the `clip`/`sqrt(max(0,·))` saturations — fine for GPU and fine for RL (no gradient flows through the sim anyway).

### 3.5 Tier 2 (stretch): wheel speed states and slip ratio

Add `ω_f, ω_r` with wheel spin dynamics:

```
κ_i = (R_e·ω_i − v_x) / max(|v_x|, v_ε)            # slip ratio
I_w · ω̇_i = T_drive,i − T_brake,i − R_e · F_x,i
F_x,i = F_x0(κ_i, F_z,i)                            # Magic Formula in κ, same structure as F_y0
```

with combined slip via the normalized-slip vector `s = (κ/κ_peak, tan α / tan α_peak)`, `F = F(|s|)·ŝ`. This buys lockups (κ → −1, lateral grip collapses), wheelspin, and flat-spot-style behavior. Cost: stiff dynamics (`I_w` small) needing ~1 kHz substeps, plus two more states per car. **Do Tier 1 first; only add Tier 2 if the RL work wants lockup/launch behavior.** V2's `brake_lock_threshold` heuristic is deleted either way — Tier 1 approximates lockup by the ellipse already.

### 3.6 Loads and aero

Static + aero + longitudinal transfer (same structure as V2 but with honest aero):

```
F_z,f = m·g·(b/L) + ½·ρ·C_L·A_f·v_x²·χ_f − m·a_x·(h/L)
F_z,r = m·g·(a/L) + ½·ρ·C_L·A_f·v_x²·(1−χ_f) + m·a_x·(h/L)
```

- `a_x` is the **previous step's** filtered longitudinal acceleration (first-order low-pass, τ ≈ 0.1 s) to avoid the algebraic loop force→load→force. This also crudely mimics suspension pitch dynamics.
- Drag: `F_drag = ½·ρ·C_D·A·v_x²`; rolling resistance `F_roll = c_rr·m·g·sign(v_x)`.
- Lateral load transfer in a single-track model can't be split across left/right wheels; keep V2's idea but as a *documented* axle-grip derate: `μ_eff = μ_0·(1 − k_llt·|a_y|·h/(g·w_track))` with `k_llt ≈ 0.05`, or drop it entirely at first. Do not let it become a hidden tuning knob again.

**Target numbers (the reasoning behind G3).** With `ρ = 1.2 kg/m³`:

| Quantity | Value | Why |
|---|---|---|
| `C_L·A` | ≈ 4.2 m² | downforce ≈ ½·1.2·4.2·(83.3)² ≈ 17.5 kN at 300 kph ≈ 2.2× weight — matches modern F1 |
| `C_D·A` | ≈ 1.15 m² | drag ≈ 5.5 kN at 320 kph; with P ≈ 780 kW gives v_max ≈ 350 kph — matches Monza speed traps |
| `μ_0` (slick) | 1.7–1.9 | physically real; high-speed cornering ≈ μ·(mg + F_down)/(m·g) ≈ 1.8·(1+2.2) ≈ 5.7 g available, matching Parabolica/Lesmo telemetry **without any speed-dependent grip scale** |
| `I_z` | ≈ 1,200 kg·m² | m·k² with radius of gyration k ≈ 1.25 m for a 798 kg car |
| `h` (CoG) | 0.30 m | published F1 estimates |
| `χ_f` (aero balance) | 0.44–0.47 | slightly rearward of weight distribution for stability |

Sanity check that kills the V2 hack: low-speed (no aero) max lateral g = μ_0 ≈ 1.8 g; Lesmo-speed (200 kph) ≈ μ_0·(1 + F_down/mg) ≈ 1.8·1.95 ≈ 3.5 g; both match the FastF1 sustained-corner distributions V2 needed `mechanical_grip_*` to fake.

### 3.7 Low-speed regularization

The slip-angle `atan(·/v_x)` blows up as `v_x → 0`, and Pacejka forces chatter at a standstill. Standard two-part fix:

1. Regularized denominators: `max(|v_x|, v_ε)` with `v_ε = 0.5 m/s` everywhere.
2. Kinematic blend: below `v_lo = 3 m/s`, blend the dynamic yaw/lateral solution toward the kinematic bicycle (`r_kin = v_x·tan δ/L`, `v_y,kin = r·b`) with weight `w = smoothstep(v_x; 1, 3)`. Above 3 m/s the car is fully dynamic; the blend only sanitizes pit-speed behavior, which no lap time depends on.

Optional refinement instead of (2): first-order tire relaxation `σ·α̇ = v_x·(α_ss − α)` with relaxation length `σ ≈ 0.3 m`, which both fixes low-speed chatter and adds realistic transient tire lag. Choose one; document which.

### 3.8 Powertrain and brakes

- Keep V2's gear ratios / torque curve, but make gear a true state: shift up when `rpm > shift_up`, down when `rpm < shift_down` **using the persisted gear**, with a 50 ms shift cut (drive force = 0 during shift — adds a real cost to bad gearing, visible in telemetry).
- Engine force at the wheels: `F_engine = min(T_engine(rpm)·g_ratio·g_final·η / R_e,  P_max·η / max(v_x, v_pm))`.
- Brakes: `F_brake,max ≈ 60 kN` total (well above grip limit — braking is always tire-limited, as in reality), split by bias `γ_bias ≈ 0.56`.

### 3.9 Integration

- Semi-implicit (symplectic) Euler on the velocity states, then positions, at `dt_phys = 1/240 s`; control held for 4 substeps per 60 Hz action. Reasoning: the fastest Tier 1 time constant is the yaw mode, `τ ≈ I_z/(C_α,f·a² + C_α,r·b²)/v_x`-ish ≈ 5–15 ms at speed; 240 Hz resolves it with margin. Tier 2 wheel dynamics would need ~1 kHz.
- The CPU oracle may optionally use RK4 at the same `dt` as a cross-check, but **CPU and GPU must ship the same integrator** — parity by construction, not by tolerance gymnastics.
- All state in float32 on GPU; accumulate progress/time in float64 on CPU oracle only.

### 3.10 Termination semantics (new, physics-level)

- `off_track`: CoG lateral offset beyond `w(s)/2 + 0.5 m` (see §4 — real widths, not 24 m).
- `spin`: `|β| > 60°` sustained for > 0.3 s, or `|r| > 4 rad/s` — the car *can* spin now, so the env must detect it. (Recoverable slides below the threshold are allowed: catching a slide is exactly the behavior worth learning.)
- `stalled`: `v_x < 1 m/s` for > 2 s away from the start.

---

## 4. Track model upgrade

- Replace the contour-image-derived "extra wide" track with the **TUMFTM racetrack database** (github.com/TUMFTM/racetrack-database): real centerlines with per-point left/right widths for ~20 F1 circuits including Monza. Build `track_spec` directly in meters: `(s_i, x_i, y_i, w_left,i, w_right,i, κ_i)` with arc-length `s`, curvature `κ` from smoothed splines.
- Real Monza width is ~10–12 m. The current 24 m checkpoint lateral limit and extra-wide raster are a large part of why 77.6 s "beat" a 79.3 s real lap. Honest widths will slow the sim down toward realistic times — accept it.
- Multi-track support is a physics-layer deliverable because the RL plan trains on several circuits and holds out tracks for zero-shot evaluation. The loader, curvature pipeline, and start grids must be track-generic.
- Optional: kerb strips as `μ` modulation bands (`μ × 0.9` on kerb, `× 0.6` on grass before termination margin) — cheap, adds texture to line choice.

---

## 5. Stochastic hooks (required by the RL plan)

All sampled per-episode at reset, all off by default for the deterministic oracle:

| Hook | Distribution | Why |
|---|---|---|
| `μ_0` scale | U(0.95, 1.05) | policies must not memorize one grip level |
| mass | U(−2%, +3%) | fuel-load proxy |
| `C_L·A`, `C_D·A` | U(±4%) | setup variation |
| wind | constant world vector, speed U(0, 8 m/s) | breaks straight-line memorization |
| control latency | 0–2 control steps (action delay buffer) | real systems have latency; kills open-loop replay strategies |
| observation noise | per-feature Gaussian, σ from a config table | same |

These belong in the physics/env layer so that CPU oracle evaluation can replay the exact seeds.

---

## 6. Calibration and validation

Keep the FastF1 multilap pipeline and add dynamics-specific targets:

1. **g-g envelope match**: compute (a_x, a_y) scatter from FastF1 laps; sim's quasi-steady envelope (sweep of max-effort arcs at each speed) must contain ~95% of the real cloud and overshoot its hull by < 10% in any direction. This is the single test that replaces all of V2's grip-scale tuning.
2. **Speed-trace MAE** on the reference laps (keep) plus **braking-distance** checks (keep) — both should now pass with physical μ and aero.
3. **Steady-state handling**: constant-radius sweep; understeer gradient `K = m·(b·C_α,r − a·C_α,f)/(L·C_α,f·C_α,r)` should be slightly positive (mild understeer) and the limit behavior should transition to oversteer with throttle — assert both in tests.
4. **Step-steer response**: at 50 m/s, 2° step; yaw-rate rise time 0.15–0.35 s, overshoot < 30% — validates `I_z` and stiffnesses jointly.
5. **Energy audit property test**: per step, `P_engine·η ≥ Δ(½mv²)/dt + P_drag + P_roll + P_brake` within tolerance; catches force-budget bugs forever.
6. **CPU/GPU parity, redefined honestly**: a chaotic dynamic model diverges exponentially under float32 reorderings, so demand (a) bitwise-equal single steps from identical states, (b) trajectory agreement < 0.05 m over 2 s horizons, and (c) *distributional* parity over 1,000-car batches (lap-time and termination-reason histograms). Drop the expectation of long-horizon trajectory parity — V2 could promise it only because it was nearly linear.
7. Manual driving QC stays: V3 must feel like a car (slides announce themselves, catchable with countersteer at moderate speed). Keyboard input gets a steering-rate filter so it remains drivable.

---

## 7. Phased roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | SI-units refactor; pixels out of physics; track loader from TUM DB with real widths | all existing tests green on V2 |
| P1 | 3-DOF EOM + Pacejka lateral, no combined slip; CPU only | step-steer + skidpad tests pass; car can spin in manual |
| P2 | Friction-ellipse combined slip; brakes/powertrain rework; gear-state fix | trail-brake/power-oversteer behavior tests; energy audit |
| P3 | Aero recalibration vs FastF1; delete `mechanical_grip_*` knobs | g-g envelope test passes with μ ≤ 1.9 |
| P4 | GPU port (Warp + eager parity twin); batch 4,096+ | parity per §6.6; ≥ 200k env-steps/s on local GPU |
| P5 | Stochastic hooks + spin/off-track termination; env API freeze for RL2 | seeded replay determinism test |
| P6 (opt) | Tier 2 wheel-speed states | only if RL2 wants lockup behavior |

Pitfalls to expect, recorded so they're not relearned: instability at low speed if §3.7 is skipped; reward-hacking of kerb μ bands; float32 `atan2` discontinuities at ±π in heading (wrap once per step, test it); load-transfer algebraic loop oscillation if `a_x` isn't filtered.
