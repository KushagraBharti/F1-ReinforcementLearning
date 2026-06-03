# F1RL QC Report

- run: `qc-20260603-034344`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-straight-micro-goal-6k-20260603-033723\eval\selected_telemetry\ppo_curriculum_segment_initial_scratch_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-straight-micro-initial-20260603-033723\qc-20260603-034344\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-straight-micro-initial-20260603-033723\qc-20260603-034344\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `assist_virtual_corridor`
- best progress: `580.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `315.9kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `527.0m`
- first bad speed: `315.9kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'left': 1}`
- terminal event: `assist_virtual_corridor`
- terminal progress: `580.6m`
- terminal speed: `275.0kph`

## Section Summary
- `rettifilo_chicane`: entry `315.9kph`, min `275.0kph`, exit `275.0kph`, avg brake `0.00`, termination `assist_virtual_corridor`
