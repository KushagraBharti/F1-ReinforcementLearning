# F1RL QC Report

- run: `qc-20260603-030948`
- telemetry: `artifacts\ppo-aggressive-rettifilo-backchain-library-goal-8k-20260603-030432\eval\selected_telemetry\ppo_curriculum_segment_train_00008000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-aggressive-backchain-segment-20260603-031000\qc-20260603-030948\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-aggressive-backchain-segment-20260603-031000\qc-20260603-030948\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `523.8m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `240.3kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `521.6m`
- first bad speed: `240.3kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake_left': 1}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `523.8m`
- terminal speed: `238.5kph`

## Section Summary
- `rettifilo_chicane`: entry `240.3kph`, min `237.8kph`, exit `238.5kph`, avg brake `0.67`, termination `assist_throttle_brake_demand`
