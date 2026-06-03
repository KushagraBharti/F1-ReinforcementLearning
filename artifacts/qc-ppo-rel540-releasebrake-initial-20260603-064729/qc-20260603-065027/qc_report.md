# F1RL QC Report

- run: `qc-20260603-065027`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-rel540-releasebrake-gated-transfer-1024-20260603-064729\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel540-releasebrake-initial-20260603-064729\qc-20260603-065027\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rel540-releasebrake-initial-20260603-064729\qc-20260603-065027\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_no_brake_gate`
- best progress: `571.1m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `231.3kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `571.1m`
- first bad speed: `231.3kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'coast': 1}`
- terminal event: `assist_no_brake_gate`
- terminal progress: `571.1m`
- terminal speed: `231.3kph`

## Section Summary
- `rettifilo_chicane`: entry `231.3kph`, min `231.3kph`, exit `231.3kph`, avg brake `0.00`, termination `assist_no_brake_gate`
