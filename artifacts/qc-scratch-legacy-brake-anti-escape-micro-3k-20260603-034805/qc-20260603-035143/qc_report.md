# F1RL QC Report

- run: `qc-20260603-035143`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k-20260603-034805\eval\selected_telemetry\ppo_curriculum_segment_train_00003000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-3k-20260603-034805\qc-20260603-035143\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-3k-20260603-034805\qc-20260603-035143\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `597.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `327.1kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `527.2m`
- first bad speed: `327.1kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake_right': 1}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `597.3m`
- terminal speed: `269.0kph`

## Section Summary
- `rettifilo_chicane`: entry `327.1kph`, min `269.0kph`, exit `269.0kph`, avg brake `1.00`, termination `assist_virtual_corridor`
