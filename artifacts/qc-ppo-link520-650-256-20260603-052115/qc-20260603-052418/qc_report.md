# F1RL QC Report

- run: `qc-20260603-052418`
- telemetry: `artifacts\ppo-link520-650-2048-20260603-052115\eval\selected_telemetry\ppo_curriculum_segment_train_00000256-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-ppo-link520-650-256-20260603-052115\qc-20260603-052418\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-ppo-link520-650-256-20260603-052115\qc-20260603-052418\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_speed_gate_failed`
- best progress: `650.9m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `328.2kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `529.4m`
- first bad speed: `328.2kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'coast': 1}`
- terminal event: `segment_speed_gate_failed`
- terminal progress: `650.9m`
- terminal speed: `240.5kph`

## Section Summary
- `rettifilo_chicane`: entry `328.2kph`, min `240.5kph`, exit `240.5kph`, avg brake `0.00`, termination `segment_speed_gate_failed`
