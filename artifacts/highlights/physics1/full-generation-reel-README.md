# Full-Generation Highlight Reel

Created: 2026-06-05T10:01:41.346630+00:00

This reel preserves curated full generations from the main CPU ES and GPU ES runs.

## Contents

- `cpu-es-150x60`: generations 0, 5, 10, 29, 49, 59 (900 traces)
- `gpu-es-2000x150`: generations 0, 14, 20, 61, 72, 143 (12000 traces)

Each run folder has a replay manifest, and each `gen-XXX` folder has its own manifest.

## Replay

```powershell
uv run --no-sync python -m f1rl.replay artifacts\highlights\full-generation-reel-20260605\cpu-es-150x60 --by-generation --generation-limit 150
uv run --no-sync python -m f1rl.replay artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150 --by-generation --generation-limit 2000
```

For smoother preview, reduce `--generation-limit`:

```powershell
uv run --no-sync python -m f1rl.replay artifacts\highlights\full-generation-reel-20260605\gpu-es-2000x150 --by-generation --generation-limit 20 --sort-by score
```
