# F1RL QC Report

- run: `qc-20260602-221250`
- telemetry: `artifacts\ppo-focus-850-legacy-resume-goal-120k-20260602-195514\benchmark_40000\selected_telemetry\ppo_focus_40000_full_lap-episode-000-steps.jsonl`
- dashboard: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-221250\telemetry_dashboard.html`
- manual checklist: `C:\Users\kushagra\OneDrive\Documents\CS Projects\F1-ReinforcementLearning\artifacts\qc-20260602-221250\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `off_track`
- best progress: `966.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `347.2kph`
