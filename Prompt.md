# Prompt

## Mission
Push this repository beyond a working baseline into a measured, GPU-enabled RL optimization workflow for a top-down F1-style driving simulator.

## Primary Objective
Optimize for:
1. clean lap completion without barrier hits
2. continuous lap-time improvement
3. maximum speed only after stable control is established

## Scope
- F1 RL repository only
- manual driving, environment realism, reward shaping, observation design, PPO training, evaluation, benchmarking, artifacts, and documentation
- GPU enablement on the local RTX 4060 laptop

## Constraints
- Python 3.12.x on Windows
- Ray/RLlib + PyTorch
- Gymnasium-compatible environment API
- Use `uv` for dependency management
- Keep training deterministic where practical with explicit seeds and reproducible commands
- Only keep candidate changes that beat the current benchmark champion

## Required Outcomes
- CUDA-enabled Torch is installed and used by the default local training profile
- RLlib training supports GPU-aware configuration with CPU env runners by default
- Eval, rollout, and benchmark use lightweight inference artifacts by default instead of full algorithm restore
- Formal benchmark/evaluation workflow exists with machine-readable metrics
- Candidate-vs-champion comparison logic exists and follows the project acceptance metric ordering
- Environment physics, observations, rewards, and terminations are more realistic and more informative for RL
- Structured train profiles exist for smoke, benchmark, and performance runs
- Short visual review clips are emitted for benchmark candidates
- README and repo memory docs describe the optimization loop, commands, and artifact locations

## Success Criteria
- Full validation suite passes
- GPU validation command passes
- Smoke training passes on the upgraded stack
- Benchmark workflow produces metrics and clips
- At least one benchmarked candidate is promoted over the baseline using explicit comparison rules
