# F1RL QC Report

- run: `qc-20260603-062135`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-link520-brakecurve-softover-1024-20260603-061914\eval\selected_telemetry\ppo_curriculum_segment_train_00000384-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-link520-brakecurve-softover-384-20260603-061914\qc-20260603-062135\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-link520-brakecurve-softover-384-20260603-061914\qc-20260603-062135\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `651.0m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `329.4kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `530.8m`
- first bad speed: `329.4kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'trail_brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `651.0m`
- terminal speed: `180.9kph`

## Section Summary
- `rettifilo_chicane`: entry `329.4kph`, min `180.9kph`, exit `180.9kph`, avg brake `0.35`, termination `segment_complete`
