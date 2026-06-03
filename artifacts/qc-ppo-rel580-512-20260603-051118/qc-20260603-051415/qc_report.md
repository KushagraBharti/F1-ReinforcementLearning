# F1RL QC Report

- run: `qc-20260603-051415`
- telemetry: `artifacts\ppo-rel580-1024-20260603-051118\eval\selected_telemetry\ppo_curriculum_segment_train_00000512-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-ppo-rel580-512-20260603-051118\qc-20260603-051415\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-ppo-rel580-512-20260603-051118\qc-20260603-051415\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.5m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `166.4kph`

## Failure Analysis
- terminal event: `segment_complete`
- terminal progress: `650.5m`
- terminal speed: `144.9kph`

## Section Summary
- `rettifilo_chicane`: entry `166.4kph`, min `144.9kph`, exit `144.9kph`, avg brake `0.00`, termination `segment_complete`
