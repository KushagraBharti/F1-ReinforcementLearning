# F1RL QC Report

- run: `qc-20260602-130020`
- telemetry: `artifacts\benchmark-20260602-091800\selected_telemetry\ppo-episode-000-steps.jsonl`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-130020\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-130020\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `collision`
- best progress: `797.6m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `346.1kph`
