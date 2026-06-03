# F1RL QC Report

- run: `qc-20260603-025559`
- telemetry: `artifacts\ppo-aggressive-rettifilo-highspeed-focus-goal-6k-20260603-025224\eval\selected_telemetry\ppo_full_lap_train_00006000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-025559\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260603-025559\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `off_track`
- best progress: `152.1m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `157.7kph`

## Failure Analysis
- failed section: `start_finish_straight`
- first bad event: `boundary_contact_risk`
- first bad progress: `150.7m`
- first bad speed: `82.3kph`
- reason: `minimum ray distance is below 1m at speed`
- actions before failure: `{'brake_left': 42, 'brake_right': 69, 'throttle': 69}`
- terminal event: `off_track`
- terminal progress: `152.1m`
- terminal speed: `79.5kph`

## Section Summary
- `start_finish_straight`: entry `0.0kph`, min `0.0kph`, exit `79.5kph`, avg brake `0.39`, termination `off_track`
