# F1RL QC Report

- run: `qc-20260603-071850`
- telemetry: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\ppo-rettifilo1220-expanded-transfer-1024-20260603-071315\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rettifilo1220-expanded-initial-20260603-071315\qc-20260603-071850\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-ppo-rettifilo1220-expanded-initial-20260603-071315\qc-20260603-071850\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `964.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `311.5kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `520.3m`
- first bad speed: `294.9kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 6}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `964.3m`
- terminal speed: `64.1kph`

## Section Summary
- `rettifilo_chicane`: entry `311.5kph`, min `64.1kph`, exit `64.1kph`, avg brake `0.05`, termination `assist_virtual_corridor`
