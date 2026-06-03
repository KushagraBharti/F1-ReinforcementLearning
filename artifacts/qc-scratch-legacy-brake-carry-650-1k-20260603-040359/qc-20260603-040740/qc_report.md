# F1RL QC Report

- run: `qc-20260603-040740`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-carry-650-goal-5k-20260603-040359\eval\selected_telemetry\ppo_curriculum_segment_train_00001000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-carry-650-1k-20260603-040359\qc-20260603-040740\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-carry-650-1k-20260603-040359\qc-20260603-040740\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `no_progress`
- best progress: `615.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.4kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `520.8m`
- first bad speed: `295.3kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 7}`
- terminal event: `no_progress`
- terminal progress: `615.3m`
- terminal speed: `0.0kph`

## Section Summary
- `rettifilo_chicane`: entry `315.4kph`, min `0.0kph`, exit `0.0kph`, avg brake `1.00`, termination `no_progress`
