# Goal

Use `workflow.md` as the operating loop. The evolved-controller search has already proven that a valid Monza lap is reachable. The historical best valid evolved lap is `148.3s`; the latest speed-focused `100x30` run produced `39` valid laps with a fastest valid lap of `161.95s` and much better late-run population breadth. The objective is not merely to finish. The objective is to drive much faster.

Keep iterating until the evolutionary search produces a valid normal-start Monza lap under `80.0s`, with artifacts proving the result. Do not mark this goal complete for runtime, infrastructure, partial progress, best-distance saturation, or one lucky slow finish. Completion requires a saved artifact showing a valid lap with elapsed time `<=80.0s`.

The required loop is:

1. Read the latest run artifacts in detail.
2. Diagnose the bottlenecks from data, not guesses.
3. Mark the analyzed run as eligible for archive later, but do not compress or delete it yet.
4. Change the evolution system aggressively but coherently: scoring, selection, survival floors, mutation, crossover, genome features, telemetry, replay, or run settings.
5. Run focused smoke tests, mini experiments, repo checks, and small searches.
6. Fix failures immediately and rerun the relevant checks.
7. Re-read the latest full-run artifacts after the changes to verify the implemented changes still line up with the real bottlenecks.
8. Make any final adjustment needed by that second artifact review.
9. Rerun validators, smoke tests, and mini/small experiments until the code and behavior are ready for a full comparison.
10. After the second artifact review confirms the old run is safe, start compressing every already-analyzed old large run to the external artifact drive when available, and let that archive work run while final validation continues.
11. Confirm disk space is healthy.
12. Immediately before launching the full comparison run, verify the archive can be listed, hard-delete the original archived folder, then run the full comparison experiment, defaulting to `150x60` after compressed telemetry and cold storage pass validation.
13. Read and compare the new artifacts.
14. Update documentation and repeat until the `<=80.0s` valid-lap target is reached.

Disk management is not optional, but it happens at the correct time. Do not delete the current baseline before it has been fully read and used to drive the code changes. Once the second artifact review confirms a run is fully analyzed and safe, start compressing it in the background. Continue final validation while compression runs. Immediately before the next full comparison run, verify the compressed archive can be listed, then delete the original folder. Never delete the active run, an unanalyzed run, or a run whose archive verification failed.

Use lossless compressed telemetry for large evolutionary runs. Keep summaries, manifests, checkpoints, `attempts.jsonl`, `generation_summary.jsonl`, `best_so_far.json`, and other hot control artifacts on `C:\f1rl-artifacts`. Write all-candidate step telemetry as `.jsonl.gz`. When the external drive is available, use `D:\f1-rl-artifacts` for cold all-candidate telemetry and old archives. Manifest files on C: must point to the cold trace paths so replay and analysis can still load everything. Do not drop telemetry fields to save space unless a separate, explicit decision is made later.

For future full runs, use a max-step cap around `18000` unless artifacts justify changing it. The previous `25000` cap was too generous for slow crawlers and produced excessive telemetry. A cap around `18000` leaves a healthy buffer over the current fastest valid run while making slow/no-progress candidates less attractive.

The main metrics now are:

- fastest valid lap time;
- average valid lap time;
- valid lap count;
- top-decile lap pace;
- average distance;
- top-decile distance;
- gate pass rates, especially `3000m`, `4000m`, and `5000m`;
- average score and top-decile score;
- average pace and top-decile pace;
- termination reasons;
- replay behavior.

Best distance is no longer a primary metric once candidates cross lap distance. A `5800m+` plateau means the car is crossing the finish and terminating. Optimize speed, valid-lap count, lap-time distribution, and population breadth instead.

The next speed push should emphasize:

- valid lap count;
- fastest valid lap time;
- top-decile pace;
- average valid-lap pace;
- fewer slow valid finishers;
- more offspring from fast valid and fast frontier parents.

Be aggressive. If the artifacts show that scoring overvalues safe slow finishes, change scoring. If elites improve but the population average collapses, change selection. If too many candidates die early, make the survival pressure more ruthless while preserving a small exploration pool. If slow-crawl behavior survives, add stronger pace and lap-time pressure. If a genome feature is missing, add it. If a run setting wastes compute, change it. Do not spend hours protecting a weak setup.

All scoring and selection profiles are malleable. Frontier scoring, clean-distance scoring, lap-time scoring, parent buckets, plateau behavior, immigrant strategy, and survival floors can all be changed when artifacts justify it. Keep changes evidence-driven, but do not preserve an old scoring shape merely because it exists.

Do not count PPO, search, scripted, or replay artifacts as interchangeable. This goal is specifically about the evolutionary search loop reaching a valid `<=80.0s` lap first. PPO transfer remains important later, but it is not the completion condition for this goal.

Keep `Documentation.md` updated with what changed, which commands ran, what failed, what improved, and the next bottleneck.
