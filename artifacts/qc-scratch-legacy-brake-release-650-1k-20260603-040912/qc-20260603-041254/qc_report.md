# F1RL QC Report

- run: `qc-20260603-041254`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-release-650-goal-3k-20260603-040912\eval\selected_telemetry\ppo_curriculum_segment_train_00001000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-release-650-1k-20260603-040912\qc-20260603-041254\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-release-650-1k-20260603-040912\qc-20260603-041254\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `max_steps`
- best progress: `630.7m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `319.5kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `526.1m`
- first bad speed: `319.5kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `max_steps`
- terminal progress: `630.7m`
- terminal speed: `0.0kph`

## Section Summary
- `rettifilo_chicane`: entry `319.5kph`, min `0.0kph`, exit `0.0kph`, avg brake `1.00`, termination `max_steps`
