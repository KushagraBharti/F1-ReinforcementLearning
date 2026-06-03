# F1RL QC Report

- run: `qc-20260603-020532`
- telemetry: `artifacts\benchmark-20260603-020500\selected_telemetry\ppo-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-020532\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-020532\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `collision`
- best progress: `968.7m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `341.8kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `520.8m`
- first bad speed: `334.2kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'brake_right': 2, 'throttle': 178}`
- terminal event: `collision`
- terminal progress: `968.7m`
- terminal speed: `265.1kph`

## Section Summary
- `start_finish_straight`: entry `0.0kph`, min `0.0kph`, exit `325.7kph`, avg brake `0.03`, termination `None`
- `rettifilo_chicane`: entry `325.9kph`, min `265.1kph`, exit `265.1kph`, avg brake `0.13`, termination `collision`
