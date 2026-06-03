# F1RL QC Report

- run: `qc-20260603-040142`
- telemetry: `artifacts\ppo-scratch-legacy-rettifilo-brake-anti-escape-micro-goal-8k-20260603-034805\eval\selected_telemetry\ppo_curriculum_segment_train_00008000-episode-000-steps.jsonl`
- telemetry files analyzed: `1`
- dashboard: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-8k-20260603-034805\qc-20260603-040142\telemetry_dashboard.html`
- manual checklist: `artifacts\qc-scratch-legacy-brake-anti-escape-micro-8k-20260603-034805\qc-20260603-040142\manual_qc_checklist.md`

## Key Checks
- checkpoint count: `120`
- boundary segments: `1800`
- observation/action: `18` / `9`
- huge progress jump invalidates lap: `True`
- wide checkpoint crossing invalidates lap: `True`

## Telemetry
- termination: `segment_complete`
- best progress: `600.3m`
- valid lap: `False`
- missed checkpoints: `0`
- max speed: `328.1kph`

## Failure Analysis
- failed section: `rettifilo_chicane`
- first bad event: `overspeed_at_braking_zone`
- first bad progress: `529.2m`
- first bad speed: `328.1kph`
- reason: `car is far above the section target speed in the braking zone`
- actions before failure: `{'brake': 1}`
- terminal event: `segment_complete`
- terminal progress: `600.3m`
- terminal speed: `159.2kph`

## Section Summary
- `rettifilo_chicane`: entry `328.1kph`, min `159.2kph`, exit `159.2kph`, avg brake `1.00`, termination `segment_complete`
