# F1RL QC Report

- run: `qc-20260603-033046`
- telemetry: `artifacts\ppo-scratch-rettifilo-4k-fullbrake-line-goal-8k-20260603-032828\eval\selected_telemetry\ppo_curriculum_segment_train_00008000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-4k-fullbrake-line-8k-20260603-032828\qc-20260603-033046\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-4k-fullbrake-line-8k-20260603-032828\qc-20260603-033046\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `543.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.6kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `543.3m`
- first bad speed: `315.6kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'half_throttle_left': 1}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `543.3m`
- terminal speed: `315.6kph`

## Section Summary
- `rettifilo_chicane`: entry `315.6kph`, min `315.6kph`, exit `315.6kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
