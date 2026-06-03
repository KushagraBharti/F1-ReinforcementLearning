# F1RL QC Report

- run: `qc-20260602-232815`
- telemetry: `artifacts\elite-search-roggia-m8-20260602\selected_telemetry\elite-rank-000-attempt-002-steps.jsonl`
- telemetry files analyzed: `3`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-232815\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-232815\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `max_steps`
- best progress: `2028.9m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `107.6kph`

## Failure Analysis
- terminal event: `max_steps`
- terminal progress: `2028.9m`
- terminal speed: `106.2kph`

## Section Summary
- `roggia_chicane`: entry `106.8kph`, min `106.1kph`, exit `106.2kph`, avg brake `0.02`, termination `max_steps`
