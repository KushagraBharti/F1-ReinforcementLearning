# F1RL QC Report

- run: `qc-20260603-033504`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-highspeed-brakegate-goal-10k-20260603-033227\eval\selected_telemetry\ppo_curriculum_segment_train_00002000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-highspeed-2k-20260603-033227\qc-20260603-033504\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-highspeed-2k-20260603-033227\qc-20260603-033504\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `547.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.6kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `521.1m`
- first bad speed: `284.0kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake_right': 27, 'throttle_left': 1}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `547.6m`
- terminal speed: `264.5kph`

## Section Summary
- `rettifilo_chicane`: entry `315.6kph`, min `264.5kph`, exit `264.5kph`, avg brake `0.54`, termination `assist_virtual_corridor`
