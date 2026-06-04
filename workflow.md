# Workflow

This workflow is the repeatable loop for pushing the evolutionary driving system from "valid slow lap" to "valid fast lap under `80.0s`."

## Current Baseline

Latest completed large run:

`C:\f1rl-artifacts\evolution-speed-100x30-fastlap-v1-20260604`

Known results:

- `3000` candidates;
- `30` generations;
- `39` valid/lap-complete candidates;
- fastest valid lap: `161.95s`;
- final generation valid laps: `22/100`;
- final generation average distance: `2733.5m`;
- final generation average pace: `117.2 kph`;
- final generation top-decile pace: `168.4 kph`;
- final generation `5000m+` gate count: `33/100`.

The historical best valid evolved lap is still `148.3s` from `C:\f1rl-artifacts\evolution-open-100x30-avg-speed-aggressive-20260604`, now archived. The latest run improved breadth and made fastest valid laps line up with highest scores, but it is not fast enough. The active weakness is speed, especially fastest valid lap time, average valid-lap pace, and top-decile pace.

## Storage Contract

Large runs must split hot control artifacts from cold trace payloads.

Keep hot artifacts on `C:\f1rl-artifacts\<run-name>`:

- `evolution_summary.json`;
- `generation_summary.jsonl`;
- `attempts.jsonl`;
- `best_so_far.json`;
- `population_checkpoint.json`;
- `selected_telemetry\manifest.json`;
- `elite_state_library.json`;
- `ppo_bridge.json`;
- `next_commands.md`;
- `top_genomes\*.json`;
- selected/top traces when not using all-candidate telemetry.

Put bulky cold data on `D:\f1-rl-artifacts` when the external drive is available:

- all-candidate step telemetry under `D:\f1-rl-artifacts\cold-telemetry\<run-name>\*.jsonl.gz`;
- verified old-run archives under `D:\f1-rl-artifacts\archives\*.tar.gz`.

The C: manifest must point to the D: `.jsonl.gz` trace paths. Replay and state-library extraction should start from the C: `selected_telemetry` folder and follow the manifest to D:. Do not drop telemetry fields to save space; use lossless gzip compression.

## One Iteration

Every serious iteration must follow this order.

### 1. Read The Latest Artifacts

Inspect:

- `evolution_summary.json`;
- `generation_summary.jsonl`;
- `attempts.jsonl`;
- `best_so_far.json`;
- selected telemetry traces;
- replay behavior;
- checkpoint metadata;
- disk usage;
- compressed telemetry readability;
- C: hot artifact paths and D: cold telemetry/archive paths when the external drive is available.

Answer these questions:

- What is the fastest valid lap?
- How many valid laps exist?
- Which generation has the best average distance?
- Which generation has the best valid-lap count?
- Which generation has the best top-decile pace?
- Did average score improve or only top elites?
- Are slow crawlers surviving?
- Are many candidates still dying before `3000m` or `5000m`?
- Are the fastest candidates also the highest-scored candidates?
- Do selected elites look like plausible racing or just safe completion?

### 2. Diagnose Bottlenecks

Use the current expected bottleneck list as a starting point, then update it from evidence:

- speed is the main bottleneck;
- scoring still rewards safe completion too much relative to fast completion;
- average population quality is volatile;
- too many candidates fail before the final sector;
- elites improve while population breadth lags;
- slow-crawl behavior still survives;
- best-distance metrics are saturated and should not drive decisions;
- oversized step caps waste runtime and make slow candidates too viable;
- artifact size can slow iteration if old runs are not archived;
- all-candidate telemetry must be preserved losslessly but stored compressed.

Do not fix stale bottlenecks from old runs unless the new artifacts prove they still matter.

### 3. Mark The Analyzed Run As Archive-Eligible

After the latest full run has been read and summarized, mark it as eligible to archive later.

Do not compress or delete it yet. The artifacts are still needed while designing, implementing, and validating the next set of changes.

The run becomes safe to archive only after:

- the artifacts have been read;
- the bottlenecks have been written down;
- the new changes have been implemented;
- the artifacts have been re-read once more to verify the changes still target the real bottlenecks.

At that point it is safe to start compression in the background, but it is not yet safe to delete the original folder.

### 4. Implement Aggressive, Measured Changes

Allowed changes include:

- stronger valid-lap-time scoring;
- stronger fast-frontier continuation scoring before valid laps exist;
- stronger top-speed and average-pace scoring;
- penalties for slow valid laps;
- conservative but meaningful early-pace pressure;
- reduced credit for max-step/no-progress survival;
- selection pressure toward fastest valid laps and far-fast candidates;
- parent buckets for fastest valid, top-decile pace, clean fast distance, and late-frontier finishers;
- dynamic survival floors that move with the frontier;
- fewer pure elite clones when breadth collapses;
- more near-elite offspring when top elites keep improving;
- smarter immigrants near current elites and global fastest-valid archive;
- controller features that expose braking demand, target-speed drops, lookahead curvature, and lap-progress context;
- run settings such as population, generation count, worker count, telemetry selection, and max-step cap;
- artifact routing such as compressed all-candidate telemetry on the external drive.

Everything in the search loop is malleable when evidence justifies it. That includes frontier scoring, clean-distance scoring, risk-seeking scoring, parent bucket weights, elite-copy counts, plateau behavior, immigrants, mutation scale, controller features, and step caps. Preserve what works; rewrite what is blocking speed.

The default full comparison run may now scale to `150x60` once compressed telemetry and cold storage are validated. Do not scale above that until the `150x60` artifact size, runtime, and run quality are understood.

### 5. Validate And Mini-Test The Changes

Run focused checks appropriate to the change:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pyright src/f1rl
uv run --no-sync pytest -q
```

For evolution changes, also run smoke or tiny runs before the full experiment:

```powershell
uv run --no-sync f1-evolution-search --output-dir artifacts\evolution-smoke-speed --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 2000 --population 10 --generations 2 --elite-count 2 --random-immigrants 2 --top-k 2 --workers 1 --genome-type controller --scoring-profiles frontier_fast,farthest_distance,early_pace,clean_distance --telemetry-selection top --progress-every-generation
```

For compression/cold-storage changes, also run a tiny all-candidate telemetry smoke. It must keep summaries/manifests on C: while writing compressed all-candidate traces to D:

```powershell
uv run --no-sync f1-evolution-search --output-dir artifacts\evolution-smoke-gzip --start-progress-m 0 --start-speed-kph 80 --target-progress-m 200 --no-target-termination --action-set straight --observation-profile base --max-steps 80 --population 6 --generations 2 --elite-count 2 --random-immigrants 1 --top-k 2 --workers 1 --genome-type controller --scoring-profiles fast_valid_lap,time_attack,fast_frontier --telemetry-selection all --telemetry-compression gzip --all-candidate-telemetry-dir D:\f1-rl-artifacts\cold-telemetry\evolution-smoke-gzip --progress-every-generation
```

If validation fails, fix it before launching a full run.

After fixing failures, rerun the relevant checks. Do not proceed with known failing tests, broken CLIs, invalid artifacts, or smoke results that contradict the intended change.

### 6. Re-Read The Latest Full Run Against The Changes

Before archiving the old large run and before launching a new full run, re-read the latest full-run artifacts again.

This second read is a sanity check:

- Did the implemented changes actually target the measured bottleneck?
- Did the mini experiments test the right behavior?
- Did the scoring/selection change match the failure mode visible in telemetry?
- Is the next full run likely to answer a useful question?
- Is the old run now fully analyzed and safe to archive?

If the answer is no, adjust the implementation and repeat validation/mini tests.

If the answer is yes, start compressing the analyzed old large run to `D:\f1-rl-artifacts\archives` immediately. Let the compression run while final validation continues. Do not delete the original folder yet.

### 7. Final Validation Pass While Archiving

Run the validators, smoke tests, and mini/small searches one more time after any final adjustment.

This pass should prove:

- the repo is not broken;
- the evolution CLI works;
- the scoring/selection changes produce artifacts;
- the next full run is worth the disk space and time.
- any background archive has finished and can be listed before deletion.

### 8. Verify Archive And Delete Originals Immediately Before Full Compute

By this point, compression should usually already be running or complete. Now verify the archive and delete the original analyzed folder.

This is mandatory. Do not skip it. The run folders can become huge, and skipping this step can kill the next experiment with `No space left on device`.

Rules:

- archive only runs that have already been read and summarized;
- do not archive or delete the active run being resumed;
- start archive compression after the second artifact review when the run is confirmed safe;
- continue final checks while compression runs;
- do not delete an original folder until the archive exists and can be listed;
- keep archives under `D:\f1-rl-artifacts\archives` when the external drive is available;
- fall back to `C:\f1rl-artifacts\archives` only if `D:\f1-rl-artifacts` is unavailable;
- after deletion, check free disk space.

Default external-drive archive pattern:

```powershell
$runName = "RUN_NAME"
$artifactRoot = "C:\f1rl-artifacts"
$archiveRoot = "D:\f1-rl-artifacts\archives"
New-Item -ItemType Directory -Force -Path $archiveRoot
tar -czf "$archiveRoot\$runName.tar.gz" -C $artifactRoot $runName
tar -tzf "$archiveRoot\$runName.tar.gz" | Select-Object -First 5
Remove-Item -LiteralPath "$artifactRoot\$runName" -Recurse -Force
Get-PSDrive C | Select-Object @{Name='FreeGB';Expression={[math]::Round($_.Free/1GB,2)}}
Get-PSDrive D | Select-Object @{Name='FreeGB';Expression={[math]::Round($_.Free/1GB,2)}}
```

If archive verification fails, stop and fix that. Do not delete the original.

### 9. Run The Full Comparison

Use a full normal-start run after the mini checks. Default to `150x60` and around `18000` max steps once compressed telemetry has passed validation:

```powershell
$sw = [System.Diagnostics.Stopwatch]::StartNew()
uv run --no-sync f1-evolution-search --output-dir C:\f1rl-artifacts\evolution-speed-150x60-YYYYMMDD-HHMM --start-progress-m 0 --start-speed-kph 80 --target-progress-m 1500 --no-target-termination --action-set racing --observation-profile racing_v2 --max-steps 18000 --population 150 --generations 60 --elite-count 12 --random-immigrants 20 --mutation-rate 0.98 --crossover-rate 0.76 --start-mutation-rate 0.08 --top-k 40 --workers 0 --worker-chunk-size 1 --genome-type controller --scoring-profiles fast_valid_lap,time_attack,fast_frontier,lap_pace,frontier_fast,farthest_distance,early_pace,clean_distance,frontier,risk_seeking,max_progress,clean_exit,exit_speed,frontier_recovery,frontier_novelty --telemetry-selection all --telemetry-compression gzip --all-candidate-telemetry-dir D:\f1-rl-artifacts\cold-telemetry\evolution-speed-150x60-YYYYMMDD-HHMM --checkpoint-every-generations 1 --progress-every-generation --survival-floor-stages-m 450,1000,1220,1500,2000,2400,3000,4000,5000 --survival-floor-pass-rate 0.20 --parent-pool-size 44 --frontier-focus-start-m 3000 --frontier-focus-end-m 5793 --frontier-parent-min-progress-m 2400 --late-frontier-trigger-m 4000 --smart-immigrant-fraction 0.70 --smart-immigrant-current-fraction 0.92 --min-random-immigrants 2 --plateau-generations 3 --plateau-distance-epsilon-m 12 --plateau-average-improvement-m 100 --plateau-elite-fraction 0.60 --plateau-extra-mutations 1
$sw.Stop()
"elapsed_s={0:N2}" -f $sw.Elapsed.TotalSeconds
```

If the CLI has changed, update the command from `--help` instead of blindly using this template.

### 10. Read And Compare The New Run

Compare against the previous baseline:

- fastest valid lap time;
- valid-lap count;
- final generation average distance;
- best generation average distance;
- top-decile pace;
- average score trend;
- top-decile score trend;
- `3000m`, `4000m`, `5000m`, and lap-complete counts;
- failure reasons;
- replay quality.

A change is good only if it improves the target behavior or clearly reveals the next bottleneck.

### 11. Mark The New Run For The Next Iteration

After the new run has been read, compared, and documented, it becomes the latest baseline and is eligible to archive later in the next iteration.

Do not immediately delete the only copy if analysis is incomplete. The archive/delete gate happens after the next change-and-validation cycle, immediately before the next large run.

### 12. Update Documentation

Update `Documentation.md` with:

- exact command;
- run folder;
- validation results;
- fastest valid lap;
- valid-lap count;
- average and top-decile metrics;
- what changed versus the previous run;
- what bottleneck remains;
- next planned change.
- which old run was archived;
- archive path;
- free disk space after deletion.

Keep it concise. Put long logs in artifacts, not in `Documentation.md`.

## Completion Rule

Stop only when the artifacts show a valid normal-start evolved lap with elapsed time `<=80.0s`.

Do not stop for:

- a lap-distance plateau;
- a slow valid lap;
- one prettier replay;
- infrastructure completion;
- runtime duration;
- average distance alone;
- a checkpoint that merely preserves old behavior.

The goal is a valid fast lap.
