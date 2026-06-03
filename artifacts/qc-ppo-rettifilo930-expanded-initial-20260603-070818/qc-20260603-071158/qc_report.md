# F1RL QC Report

- run: `qc-20260603-071158`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-rettifilo930-expanded-transfer-1024-20260603-070818\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rettifilo930-expanded-initial-20260603-070818\qc-20260603-071158\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rettifilo930-expanded-initial-20260603-070818\qc-20260603-071158\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `930.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `330.2kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `521.3m`
- first bad speed: `309.4kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 7}`
- terminal event: `segment_complete`
- terminal progress: `930.6m`
- terminal speed: `75.3kph`

## Section Summary
- `rettifilo_chicane`: entry `330.2kph`, min `75.3kph`, exit `75.3kph`, avg brake `0.06`, termination `segment_complete`
