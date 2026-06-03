# F1RL QC Report

- run: `qc-20260603-051954`
- telemetry: `artifacts\ppo-rel570-1024-20260603-051546\eval\selected_telemetry\ppo_curriculum_segment_train_00000384-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-ppo-rel570-384-20260603-051546\qc-20260603-051954\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-ppo-rel570-384-20260603-051546\qc-20260603-051954\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `650.2m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `188.2kph`

## Failure Analysis
- terminal event: `segment_complete`
- terminal progress: `650.2m`
- terminal speed: `160.5kph`

## Section Summary
- `rettifilo_chicane`: entry `188.2kph`, min `160.5kph`, exit `160.5kph`, avg brake `0.00`, termination `segment_complete`
