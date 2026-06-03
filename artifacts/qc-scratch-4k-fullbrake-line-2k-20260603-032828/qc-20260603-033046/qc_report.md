# F1RL QC Report

- run: `qc-20260603-033046`
- telemetry: `artifacts\ppo-scratch-rettifilo-4k-fullbrake-line-goal-8k-20260603-032828\eval\selected_telemetry\ppo_curriculum_segment_train_00002000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-4k-fullbrake-line-2k-20260603-032828\qc-20260603-033046\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-4k-fullbrake-line-2k-20260603-032828\qc-20260603-033046\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `591.5m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `322.0kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `539.5m`
- first bad speed: `322.0kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'soft_brake_soft_right': 1}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `591.5m`
- terminal speed: `281.9kph`

## Section Summary
- `rettifilo_chicane`: entry `322.0kph`, min `281.9kph`, exit `281.9kph`, avg brake `0.35`, termination `assist_virtual_corridor`
