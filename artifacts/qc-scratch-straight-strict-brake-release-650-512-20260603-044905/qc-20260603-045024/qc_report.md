# F1RL QC Report

- run: `qc-20260603-045024`
- telemetry: `artifacts\ppo-scratch-straight-rettifilo-strict-brake-release-650-goal-3k-20260603-044905\eval\selected_telemetry\ppo_curriculum_segment_train_00000512-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-straight-strict-brake-release-650-512-20260603-044905\qc-20260603-045024\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-straight-strict-brake-release-650-512-20260603-044905\qc-20260603-045024\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `max_steps`
- best progress: `620.1m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.1kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `520.2m`
- first bad speed: `308.3kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 3}`
- terminal event: `max_steps`
- terminal progress: `620.1m`
- terminal speed: `0.0kph`

## Section Summary
- `rettifilo_chicane`: entry `315.1kph`, min `0.0kph`, exit `0.0kph`, avg brake `1.00`, termination `max_steps`
