# F1RL QC Report

- run: `qc-20260603-044022`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-postbrake-release-650-fast-goal-2k-20260603-043842\eval\selected_telemetry\ppo_curriculum_segment_train_00000768-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-postbrake-release-650-fast-768-20260603-043842\qc-20260603-044022\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-postbrake-release-650-fast-768-20260603-043842\qc-20260603-044022\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `off_track`
- best progress: `643.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `158.8kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `boundary_contact_risk`
- first bad progress: `643.1m`
- first bad speed: `140.7kph`
- reason: `minimum ray distance is below 1m at speed`
- actions before failure: `{'left': 67}`
- terminal event: `off_track`
- terminal progress: `643.6m`
- terminal speed: `140.5kph`

## Section Summary
- `rettifilo_chicane`: entry `158.8kph`, min `140.5kph`, exit `140.5kph`, avg brake `0.00`, termination `off_track`
