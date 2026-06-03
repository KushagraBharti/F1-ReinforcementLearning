# F1RL QC Report

- run: `qc-20260603-070026`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-rel540-highspeed-softnobrake-1024-20260603-065156\eval\selected_telemetry\ppo_curriculum_segment_train_00001024-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel540-highspeed-softnobrake-1024-20260603-065156\qc-20260603-070026\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel540-highspeed-softnobrake-1024-20260603-065156\qc-20260603-070026\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.7m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `272.5kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `552.6m`
- first bad speed: `272.5kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `650.7m`
- terminal speed: `157.4kph`

## Section Summary
- `rettifilo_chicane`: entry `272.5kph`, min `157.4kph`, exit `157.4kph`, avg brake `0.32`, termination `segment_complete`
