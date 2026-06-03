# F1RL QC Report

- run: `qc-20260603-070238`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-link520-releasebrake-softnobrake-transfer-1024-20260603-070121\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-link520-releasebrake-softnobrake-initial-20260603-070121\qc-20260603-070238\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-link520-releasebrake-softnobrake-initial-20260603-070121\qc-20260603-070238\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.8m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `313.6kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `528.9m`
- first bad speed: `313.6kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `650.8m`
- terminal speed: `154.5kph`

## Section Summary
- `rettifilo_chicane`: entry `313.6kph`, min `154.5kph`, exit `154.5kph`, avg brake `0.37`, termination `segment_complete`
