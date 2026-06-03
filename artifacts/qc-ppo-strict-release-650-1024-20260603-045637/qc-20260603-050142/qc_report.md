# F1RL QC Report

- run: `qc-20260603-050142`
- telemetry: `artifacts\ppo-strict-release-650-2200-20260603-045637\eval\selected_telemetry\ppo_curriculum_segment_train_00001024-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-ppo-strict-release-650-1024-20260603-045637\qc-20260603-050142\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-ppo-strict-release-650-1024-20260603-045637\qc-20260603-050142\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_overbrake_gate`
- best progress: `605.7m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `327.2kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `525.3m`
- first bad speed: `327.2kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `assist_overbrake_gate`
- terminal progress: `605.7m`
- terminal speed: `133.9kph`

## Section Summary
- `rettifilo_chicane`: entry `327.2kph`, min `133.9kph`, exit `133.9kph`, avg brake `1.00`, termination `assist_overbrake_gate`
