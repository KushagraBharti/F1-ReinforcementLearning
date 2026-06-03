# F1RL QC Report

- run: `qc-20260603-034344`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-straight-micro-goal-6k-20260603-033723\eval\selected_telemetry\ppo_curriculum_segment_train_00001000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-straight-micro-1k-20260603-033723\qc-20260603-034344\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-straight-micro-1k-20260603-033723\qc-20260603-034344\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `564.8m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.3kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `521.0m`
- first bad speed: `309.5kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'left': 6}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `564.8m`
- terminal speed: `276.3kph`

## Section Summary
- `rettifilo_chicane`: entry `315.3kph`, min `276.3kph`, exit `276.3kph`, avg brake `0.00`, termination `assist_virtual_corridor`
