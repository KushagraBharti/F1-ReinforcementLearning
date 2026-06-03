# F1RL QC Report

- run: `qc-20260603-012346`
- telemetry: `artifacts\benchmark-20260603-012314\selected_telemetry\ppo-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-012346\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-012346\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `off_track`
- best progress: `966.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `347.2kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `521.4m`
- first bad speed: `333.2kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'brake_right': 3, 'throttle': 177}`
- terminal event: `off_track`
- terminal progress: `966.3m`
- terminal speed: `325.0kph`

## Section Summary
- `start_finish_straight`: entry `1.3kph`, min `1.3kph`, exit `328.7kph`, avg brake `0.01`, termination `None`
- `rettifilo_chicane`: entry `328.8kph`, min `325.0kph`, exit `325.0kph`, avg brake `0.04`, termination `off_track`
