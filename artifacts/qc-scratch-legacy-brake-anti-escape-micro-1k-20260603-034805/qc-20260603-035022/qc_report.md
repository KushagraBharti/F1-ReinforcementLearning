# F1RL QC Report

- run: `qc-20260603-035022`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k-20260603-034805\eval\selected_telemetry\ppo_curriculum_segment_train_00001000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-1k-20260603-034805\qc-20260603-035022\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-1k-20260603-034805\qc-20260603-035022\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `585.0m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `322.9kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `525.3m`
- first bad speed: `322.9kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'right': 1}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `585.0m`
- terminal speed: `276.6kph`

## Section Summary
- `rettifilo_chicane`: entry `322.9kph`, min `276.6kph`, exit `276.6kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
