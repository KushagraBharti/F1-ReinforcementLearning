# F1RL QC Report

- run: `qc-20260603-032448`
- telemetry: `artifacts\ppo-scratch-rettifilo-highspeed-brakegate-goal-12k-20260603-032057\eval\selected_telemetry\ppo_curriculum_segment_train_00004000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-highspeed-brakegate-4k-20260603-032057\qc-20260603-032448\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-highspeed-brakegate-4k-20260603-032057\qc-20260603-032448\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `collision`
- best progress: `600.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `313.3kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `526.6m`
- first bad speed: `313.3kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'soft_brake_soft_right': 1}`
- terminal event: `collision`
- terminal progress: `600.6m`
- terminal speed: `258.7kph`

## Section Summary
- `rettifilo_chicane`: entry `313.3kph`, min `258.7kph`, exit `258.7kph`, avg brake `0.35`, termination `collision`
