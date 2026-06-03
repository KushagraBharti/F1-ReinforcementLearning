# F1RL QC Report

- run: `qc-20260603-024747`
- telemetry: `artifacts\ppo-aggressive-rettifilo-brakegate-goal-5k-20260603-024536\eval\selected_telemetry\ppo_full_lap_train_00005008-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-024747\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-024747\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_throttle_brake_demand`
- best progress: `520.9m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `333.8kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `throttle_during_brake_demand`
- first bad progress: `520.9m`
- first bad speed: `333.8kph`
- reason: `car is overspeed in a braking zone while still applying throttle`
- actions before failure: `{'brake_right': 4, 'throttle': 176}`
- terminal event: `assist_throttle_brake_demand`
- terminal progress: `520.9m`
- terminal speed: `333.8kph`

## Section Summary
- `start_finish_straight`: entry `0.0kph`, min `0.0kph`, exit `325.2kph`, avg brake `0.03`, termination `None`
- `rettifilo_chicane`: entry `325.4kph`, min `325.4kph`, exit `333.8kph`, avg brake `0.00`, termination `assist_throttle_brake_demand`
