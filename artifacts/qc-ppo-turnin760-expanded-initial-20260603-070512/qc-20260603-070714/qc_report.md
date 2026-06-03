# F1RL QC Report

- run: `qc-20260603-070714`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-turnin760-expanded-transfer-1024-20260603-070512\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-turnin760-expanded-initial-20260603-070512\qc-20260603-070714\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-turnin760-expanded-initial-20260603-070512\qc-20260603-070714\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `760.2m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `325.7kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `531.7m`
- first bad speed: `325.7kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `760.2m`
- terminal speed: `129.4kph`

## Section Summary
- `rettifilo_chicane`: entry `325.7kph`, min `129.4kph`, exit `129.4kph`, avg brake `0.15`, termination `segment_complete`
