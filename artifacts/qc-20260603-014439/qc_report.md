# F1RL QC Report

- run: `qc-20260603-014439`
- telemetry: `artifacts\benchmark-20260603-014242\selected_telemetry\ppo-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-014439\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-014439\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `collision`
- best progress: `970.8m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `338.7kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `520.8m`
- first bad speed: `334.1kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'action_19': 3, 'brake': 177}`
- terminal event: `collision`
- terminal progress: `970.8m`
- terminal speed: `273.8kph`

## Section Summary
- `start_finish_straight`: entry `0.0kph`, min `0.0kph`, exit `325.6kph`, avg brake `0.03`, termination `None`
- `rettifilo_chicane`: entry `325.8kph`, min `273.8kph`, exit `273.8kph`, avg brake `0.12`, termination `collision`
