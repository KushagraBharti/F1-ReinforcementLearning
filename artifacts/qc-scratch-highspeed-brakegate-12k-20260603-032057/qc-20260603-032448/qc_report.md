# F1RL QC Report

- run: `qc-20260603-032448`
- telemetry: `artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\eval\selected_telemetry\ppo_curriculum_segment_train_00012000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-highspeed-brakegate-12k-20260603-032057\qc-20260603-032448\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-highspeed-brakegate-12k-20260603-032057\qc-20260603-032448\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `520.1m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `335.7kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `520.1m`
- first bad speed: `314.8kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'half_throttle_right': 18}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `520.1m`
- terminal speed: `314.8kph`

## Section Summary
- `rettifilo_chicane`: entry `335.7kph`, min `314.8kph`, exit `314.8kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
