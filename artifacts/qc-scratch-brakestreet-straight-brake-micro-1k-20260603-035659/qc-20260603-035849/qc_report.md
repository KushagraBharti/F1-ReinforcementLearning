# F1RL QC Report

- run: `qc-20260603-035849`
- telemetry: `artifacts\ppo-scratch-brakestreet-rettifilo-straight-brake-micro-goal-4k-20260603-035659\eval\selected_telemetry\ppo_curriculum_segment_train_00001000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-brakestreet-straight-brake-micro-1k-20260603-035659\qc-20260603-035849\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-brakestreet-straight-brake-micro-1k-20260603-035659\qc-20260603-035849\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `588.0m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `329.7kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `521.1m`
- first bad speed: `329.7kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'left': 1}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `588.0m`
- terminal speed: `277.0kph`

## Section Summary
- `rettifilo_chicane`: entry `329.7kph`, min `277.0kph`, exit `277.0kph`, avg brake `0.00`, termination `assist_virtual_corridor`
