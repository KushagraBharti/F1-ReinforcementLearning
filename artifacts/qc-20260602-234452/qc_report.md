# F1RL QC Report

- run: `qc-20260602-234452`
- telemetry: `artifacts\benchmark-20260602-234321\selected_telemetry\ppo-episode-000-steps.jsonl`
- telemetry files analyzed: `3`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-234452\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-234452\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `collision`
- best progress: `951.8m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `252.9kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `521.1m`
- first bad speed: `244.0kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'continuous': 180}`
- terminal event: `collision`
- terminal progress: `951.8m`
- terminal speed: `252.9kph`

## Section Summary
- `start_finish_straight`: entry `0.7kph`, min `0.7kph`, exit `239.7kph`, avg brake `0.00`, termination `None`
- `rettifilo_chicane`: entry `239.8kph`, min `239.8kph`, exit `252.9kph`, avg brake `0.00`, termination `collision`
