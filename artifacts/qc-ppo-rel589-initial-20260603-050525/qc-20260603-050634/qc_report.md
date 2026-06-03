# F1RL QC Report

- run: `qc-20260603-050634`
- telemetry: `artifacts\ppo-rel589-512-20260603-050525\eval\selected_telemetry\ppo_curriculum_segment_initial_transfer_00000000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-ppo-rel589-initial-20260603-050525\qc-20260603-050634\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-ppo-rel589-initial-20260603-050525\qc-20260603-050634\manual_qc_checklist.md`

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
- max speed: `186.2kph`

## Failure Analysis
- terminal event: `segment_complete`
- terminal progress: `650.5m`
- terminal speed: `158.9kph`

## Section Summary
- `rettifilo_chicane`: entry `186.2kph`, min `158.9kph`, exit `158.9kph`, avg brake `0.00`, termination `segment_complete`
