# F1RL QC Report

- run: `qc-20260603-034344`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-straight-micro-goal-6k-20260603-033723\eval\selected_telemetry\ppo_curriculum_segment_train_00006000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-straight-micro-6k-20260603-033723\qc-20260603-034344\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-straight-micro-6k-20260603-033723\qc-20260603-034344\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `521.4m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `318.6kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `521.4m`
- first bad speed: `318.6kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'throttle': 14}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `521.4m`
- terminal speed: `318.6kph`

## Section Summary
- `rettifilo_chicane`: entry `314.9kph`, min `314.9kph`, exit `318.6kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
