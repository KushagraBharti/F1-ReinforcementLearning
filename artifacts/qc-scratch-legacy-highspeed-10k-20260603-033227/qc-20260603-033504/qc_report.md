# F1RL QC Report

- run: `qc-20260603-033504`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-highspeed-brakegate-goal-10k-20260603-033227\eval\selected_telemetry\ppo_curriculum_segment_train_00010000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-highspeed-10k-20260603-033227\qc-20260603-033504\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-highspeed-10k-20260603-033227\qc-20260603-033504\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `520.4m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `313.1kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `520.4m`
- first bad speed: `302.1kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'throttle_left': 11}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `520.4m`
- terminal speed: `302.1kph`

## Section Summary
- `rettifilo_chicane`: entry `313.1kph`, min `302.1kph`, exit `302.1kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
