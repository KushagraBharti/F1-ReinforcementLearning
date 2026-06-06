# Final Highlights

This folder is the local replay/highlight set for the final project state.

## Buckets

1. `physics1/cpu-es`
   - Physics 1 CPU evolutionary search.
   - Replay telemetry: `physics1/cpu-es/telemetry`
   - GIFs: `physics1/cpu-es/gifs`

2. `physics1/gpu-es`
   - Physics 1 GPU evolutionary search, reduced stratified local set.
   - Replay telemetry: `physics1/gpu-es/telemetry`
   - GIFs: `physics1/gpu-es/gifs`
   - The full 12,000-trace GPU ES archive remains on `D:`.

3. `physics1/gpu-rl`
   - Physics 1 learned-policy replay telemetry.
   - Replay telemetry: `physics1/gpu-rl/telemetry`
   - GIFs: `physics1/gpu-rl/gifs`

4. `physics2/gpu-es`
   - Physics 2 FastF1-calibrated GPU ES winner, CPU-reranked.
   - Replay telemetry: `physics2/gpu-es/telemetry`
   - GIFs: `physics2/gpu-es/gifs`

5. `physics2/gpu-rl`
   - Physics 2 learned-policy promotion and deterministic swarm.
   - Replay telemetry: `physics2/gpu-rl/telemetry`
   - GIFs: `physics2/gpu-rl/gifs`

Physics 2 calibration summaries live under `physics2/calibration`.

## Replay Commands

```powershell
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics1\cpu-es\telemetry\gen-000" "artifacts\highlights\physics1\cpu-es\telemetry\gen-005" "artifacts\highlights\physics1\cpu-es\telemetry\gen-010" "artifacts\highlights\physics1\cpu-es\telemetry\gen-029" "artifacts\highlights\physics1\cpu-es\telemetry\gen-049" "artifacts\highlights\physics1\cpu-es\telemetry\gen-059" --sort score --speed 2
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics1\gpu-es\telemetry\gen-000" "artifacts\highlights\physics1\gpu-es\telemetry\gen-014" "artifacts\highlights\physics1\gpu-es\telemetry\gen-020" "artifacts\highlights\physics1\gpu-es\telemetry\gen-061" "artifacts\highlights\physics1\gpu-es\telemetry\gen-072" "artifacts\highlights\physics1\gpu-es\telemetry\gen-143" --sort score --speed 2
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics1\gpu-rl\telemetry\promotion_cpu_eval\selected_telemetry" --speed 4
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-es\telemetry" --sort score --speed 4
uv run --no-sync python -m f1rl.replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_promotion" --speed 4
```

Policy swarm replays:

```powershell
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\physics1\gpu-rl\telemetry\policy_swarm_1000\checkpoint_0000" --speed 1
uv run --no-sync python -m f1rl.policy_swarm_replay "artifacts\highlights\physics2\gpu-rl\telemetry\learned_policy_swarm_1000\checkpoint_0000" --speed 4
```

## Archive Sources

- Physics 1 restored local highlight set: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\artifacts-highlights-20260605-local-reduced-gpu-es.tar.zst`
- Physics 1 full GPU ES 12,000-trace archive: `D:\f1-rl-artifacts\archives\rl1-postgoal-20260605\gpu-es-2000x150-full-12000-traces-20260605.tar.zst`
- Physics 2 bulk archive: `D:\f1-rl-artifacts\archives\physics-v2-20260606\artifacts-bulk-excluding-v2-final-highlights-20260606.tar.zst`
