# F1RL QC Report

- run: `qc-20260603-063944`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-rel570-releasebrake-gated-scratch-1024-20260603-063822\eval\selected_telemetry\ppo_curriculum_segment_initial_scratch_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel570-releasebrake-initial-20260603-063822\qc-20260603-063944\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel570-releasebrake-initial-20260603-063822\qc-20260603-063944\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.5m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `203.7kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `582.0m`
- first bad speed: `203.7kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `650.5m`
- terminal speed: `157.0kph`

## Section Summary
- `rettifilo_chicane`: entry `203.7kph`, min `157.0kph`, exit `157.0kph`, avg brake `0.12`, termination `segment_complete`
