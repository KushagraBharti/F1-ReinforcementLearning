# F1RL QC Report

- run: `qc-20260603-044354`
- telemetry: `artifacts\ppo-scratch-straight-rettifilo-postbrake-release-650-goal-1k-20260603-044245\eval\selected_telemetry\ppo_curriculum_segment_train_00000128-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-straight-postbrake-release-650-128-20260603-044245\qc-20260603-044354\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-straight-postbrake-release-650-128-20260603-044245\qc-20260603-044354\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `158.8kph`

## Failure Analysis
- terminal event: `segment_complete`
- terminal progress: `650.6m`
- terminal speed: `139.3kph`

## Section Summary
- `rettifilo_chicane`: entry `158.8kph`, min `139.3kph`, exit `139.3kph`, avg brake `0.00`, termination `segment_complete`
