# F1RL QC Report

- run: `qc-20260603-025618`
- telemetry: `artifacts\ppo-aggressive-rettifilo-highspeed-focus-goal-6k-20260603-025224\eval\selected_telemetry\ppo_curriculum_segment_train_00006000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-aggressive-highspeed-focus-segment-20260603-025650\qc-20260603-025618\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-aggressive-highspeed-focus-segment-20260603-025650\qc-20260603-025618\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `523.5m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `317.6kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `520.6m`
- first bad speed: `314.9kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake_left': 2, 'brake_right': 1, 'throttle': 24}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `523.5m`
- terminal speed: `311.8kph`

## Section Summary
- `rettifilo_chicane`: entry `313.8kph`, min `310.4kph`, exit `311.8kph`, avg brake `0.14`, termination `assist_throttle_brake_demand`
