"""Run repeatable evolution-search ladders without hand-building every command."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1rl.config import ARTIFACTS_DIR, MONZA_LENGTH_METERS
from f1rl.evolution_search import (
    EvolutionGates,
    EvolutionSearchConfig,
    run_evolution_search,
)


@dataclass(frozen=True, slots=True)
class LadderScale:
    population: int
    generations: int
    elite_count: int
    random_immigrants: int
    max_steps: int
    full_probe_steps: int
    top_k: int


@dataclass(frozen=True, slots=True)
class LadderRung:
    name: str
    start_progress_m: float
    target_progress_m: float
    start_speed_kph: float
    max_steps: int
    target_min_speed_kph: float | None = None
    target_max_speed_kph: float | None = None
    target_max_lateral_error_m: float | None = None
    target_max_heading_error_deg: float | None = None
    target_max_abs_yaw_rate_rps: float | None = None
    target_max_abs_steering: float | None = None
    use_previous_elites: bool = False


SCALES: dict[str, LadderScale] = {
    "smoke": LadderScale(
        population=8,
        generations=1,
        elite_count=2,
        random_immigrants=1,
        max_steps=40,
        full_probe_steps=80,
        top_k=2,
    ),
    "small": LadderScale(
        population=100,
        generations=4,
        elite_count=12,
        random_immigrants=16,
        max_steps=180,
        full_probe_steps=900,
        top_k=8,
    ),
    "medium": LadderScale(
        population=1000,
        generations=10,
        elite_count=64,
        random_immigrants=128,
        max_steps=360,
        full_probe_steps=3000,
        top_k=24,
    ),
    "large": LadderScale(
        population=10000,
        generations=20,
        elite_count=256,
        random_immigrants=512,
        max_steps=600,
        full_probe_steps=18000,
        top_k=48,
    ),
}


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return ARTIFACTS_DIR / f"evolution-ladder-{timestamp}"


def _rungs(scale: LadderScale) -> list[LadderRung]:
    return [
        LadderRung(
            name="brake-discovery-520-650",
            start_progress_m=520.0,
            target_progress_m=650.0,
            start_speed_kph=320.0,
            max_steps=min(scale.max_steps, 180),
            target_max_speed_kph=230.0,
            target_max_lateral_error_m=14.0,
            target_max_heading_error_deg=28.0,
        ),
        LadderRung(
            name="turnin-first-apex-650-900",
            start_progress_m=650.0,
            target_progress_m=900.0,
            start_speed_kph=210.0,
            max_steps=min(scale.max_steps, 260),
            target_max_speed_kph=210.0,
            target_max_lateral_error_m=10.0,
            target_max_heading_error_deg=24.0,
            use_previous_elites=True,
        ),
        LadderRung(
            name="rotation-stabilization-900-1000",
            start_progress_m=900.0,
            target_progress_m=1000.0,
            start_speed_kph=165.0,
            max_steps=min(scale.max_steps, 180),
            target_max_speed_kph=190.0,
            target_max_lateral_error_m=8.0,
            target_max_heading_error_deg=20.0,
            target_max_abs_yaw_rate_rps=0.8,
            use_previous_elites=True,
        ),
        LadderRung(
            name="exit-1000-1220",
            start_progress_m=1000.0,
            target_progress_m=1220.0,
            start_speed_kph=160.0,
            max_steps=min(scale.max_steps, 280),
            target_min_speed_kph=130.0,
            target_max_speed_kph=240.0,
            target_max_lateral_error_m=12.0,
            target_max_heading_error_deg=24.0,
            use_previous_elites=True,
        ),
        LadderRung(
            name="linked-rettifilo-520-1220",
            start_progress_m=520.0,
            target_progress_m=1220.0,
            start_speed_kph=320.0,
            max_steps=scale.max_steps,
            target_min_speed_kph=120.0,
            target_max_speed_kph=250.0,
            target_max_lateral_error_m=16.0,
            target_max_heading_error_deg=32.0,
        ),
        LadderRung(
            name="normal-start-transfer-0-1220",
            start_progress_m=0.0,
            target_progress_m=1220.0,
            start_speed_kph=0.0,
            max_steps=scale.full_probe_steps,
            target_max_lateral_error_m=18.0,
            target_max_heading_error_deg=35.0,
        ),
        LadderRung(
            name="roggia-exit-lesmo-1900-2800",
            start_progress_m=1900.0,
            target_progress_m=2800.0,
            start_speed_kph=315.0,
            max_steps=max(scale.max_steps, 420),
            target_min_speed_kph=120.0,
            target_max_speed_kph=245.0,
            target_max_lateral_error_m=16.0,
            target_max_heading_error_deg=34.0,
        ),
        LadderRung(
            name="roggia-lesmo-transfer-2000-3030",
            start_progress_m=2000.0,
            target_progress_m=3030.0,
            start_speed_kph=260.0,
            max_steps=max(scale.max_steps, 520),
            target_min_speed_kph=135.0,
            target_max_speed_kph=250.0,
            target_max_lateral_error_m=16.0,
            target_max_heading_error_deg=34.0,
            use_previous_elites=True,
        ),
    ]


def _search_config(
    *,
    scale: LadderScale,
    rung: LadderRung,
    args: argparse.Namespace,
) -> EvolutionSearchConfig:
    return EvolutionSearchConfig(
        action_set=args.action_set,
        observation_profile=args.observation_profile,
        max_steps=rung.max_steps,
        population=scale.population,
        generations=scale.generations,
        elite_count=scale.elite_count,
        random_immigrants=scale.random_immigrants,
        min_phases=args.min_phases,
        max_phases=args.max_phases,
        min_phase_steps=args.min_phase_steps,
        max_phase_steps=args.max_phase_steps,
        min_phase_progress_m=args.min_phase_progress_m,
        max_phase_progress_m=args.max_phase_progress_m,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        start_mutation_rate=args.start_mutation_rate,
        seed=args.seed,
        top_k=scale.top_k,
        workers=args.workers,
        worker_chunk_size=args.worker_chunk_size,
        genome_type=args.genome_type,
        scoring_profiles=tuple(args.scoring_profiles),
        checkpoint_every_generations=args.checkpoint_every_generations,
        progress_every_generation=args.progress_every_generation,
        frontier_focus_start_m=args.frontier_focus_start_m,
        frontier_focus_end_m=args.frontier_focus_end_m,
        frontier_parent_min_progress_m=args.frontier_parent_min_progress_m,
        plateau_mode=not args.no_plateau_mode,
        plateau_generations=args.plateau_generations,
        plateau_distance_epsilon_m=args.plateau_distance_epsilon_m,
        plateau_average_improvement_m=args.plateau_average_improvement_m,
        plateau_elite_fraction=args.plateau_elite_fraction,
        plateau_extra_mutations=args.plateau_extra_mutations,
    )


def _gates(rung: LadderRung) -> EvolutionGates:
    return EvolutionGates(
        target_progress_m=rung.target_progress_m,
        target_min_speed_kph=rung.target_min_speed_kph,
        target_max_speed_kph=rung.target_max_speed_kph,
        target_max_lateral_error_m=rung.target_max_lateral_error_m,
        target_max_heading_error_deg=rung.target_max_heading_error_deg,
        target_max_abs_yaw_rate_rps=rung.target_max_abs_yaw_rate_rps,
        target_max_abs_steering=rung.target_max_abs_steering,
        segment_fail_on_speed_gate_miss=True,
    )


def _summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "evolution_summary.json").read_text(encoding="utf-8"))


def _full_probe(
    *,
    root: Path,
    scale: LadderScale,
    args: argparse.Namespace,
    name: str,
) -> dict[str, Any]:
    rung = LadderRung(
        name=name,
        start_progress_m=0.0,
        target_progress_m=MONZA_LENGTH_METERS,
        start_speed_kph=0.0,
        max_steps=scale.full_probe_steps,
    )
    run_dir = root / name
    run_evolution_search(
        output_dir=run_dir,
        config=_search_config(scale=scale, rung=rung, args=args),
        gates=_gates(rung),
        start_progress_m=0.0,
        start_speed_kph=0.0,
    )
    return {"name": name, "run_dir": str(run_dir), "summary": _summary(run_dir)}


def run_ladder(args: argparse.Namespace) -> Path:
    scale = SCALES[args.scale]
    root = args.output_dir or default_output_dir()
    root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    previous_elite_library: Path | None = None

    if args.full_probe_mode != "none":
        runs.append(_full_probe(root=root, scale=scale, args=args, name="full-probe-before"))

    for rung in _rungs(scale):
        run_dir = root / rung.name
        state_library = previous_elite_library if rung.use_previous_elites else None
        run_evolution_search(
            output_dir=run_dir,
            config=_search_config(scale=scale, rung=rung, args=args),
            gates=_gates(rung),
            state_library=state_library,
            start_progress_m=None if state_library is not None else rung.start_progress_m,
            start_speed_kph=rung.start_speed_kph,
        )
        summary = _summary(run_dir)
        runs.append({"name": rung.name, "run_dir": str(run_dir), "summary": summary})
        elite_path = summary.get("elite_state_library")
        previous_elite_library = Path(elite_path) if elite_path else None
        if args.full_probe_mode == "every-rung":
            runs.append(_full_probe(root=root, scale=scale, args=args, name=f"full-probe-after-{rung.name}"))
        elif args.full_probe_mode == "major" and rung.name in {
            "linked-rettifilo-520-1220",
            "normal-start-transfer-0-1220",
            "roggia-exit-lesmo-1900-2800",
            "roggia-lesmo-transfer-2000-3030",
        }:
            runs.append(_full_probe(root=root, scale=scale, args=args, name=f"full-probe-after-{rung.name}"))

    payload = {
        "kind": "evolution_ladder",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scale": args.scale,
        "genome_type": args.genome_type,
        "scoring_profiles": list(args.scoring_profiles),
        "runs": runs,
        "final_success_rule": "Only honest normal-start PPO full-lap completion counts as final success.",
    }
    (root / "ladder_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Yosh-style evolution-search ladders.")
    parser.add_argument("--scale", choices=sorted(SCALES), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--worker-chunk-size", type=int, default=0)
    parser.add_argument("--genome-type", choices=["phase", "progress_phase", "controller"], default="controller")
    parser.add_argument(
        "--scoring-profiles",
        default="max_progress,clean_exit,risk_seeking,frontier_recovery,frontier_novelty",
    )
    parser.add_argument("--action-set", default="racing")
    parser.add_argument("--observation-profile", default="racing_v2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-phases", type=int, default=2)
    parser.add_argument("--max-phases", type=int, default=8)
    parser.add_argument("--min-phase-steps", type=int, default=4)
    parser.add_argument("--max-phase-steps", type=int, default=72)
    parser.add_argument("--min-phase-progress-m", type=float, default=8.0)
    parser.add_argument("--max-phase-progress-m", type=float, default=140.0)
    parser.add_argument("--mutation-rate", type=float, default=0.8)
    parser.add_argument("--crossover-rate", type=float, default=0.4)
    parser.add_argument("--start-mutation-rate", type=float, default=0.10)
    parser.add_argument("--checkpoint-every-generations", type=int, default=1)
    parser.add_argument("--progress-every-generation", action="store_true")
    parser.add_argument("--full-probe-mode", choices=["none", "major", "every-rung"], default="major")
    parser.add_argument("--frontier-focus-start-m", type=float, default=2200.0)
    parser.add_argument("--frontier-focus-end-m", type=float, default=2600.0)
    parser.add_argument("--frontier-parent-min-progress-m", type=float, default=2000.0)
    parser.add_argument("--no-plateau-mode", action="store_true")
    parser.add_argument("--plateau-generations", type=int, default=3)
    parser.add_argument("--plateau-distance-epsilon-m", type=float, default=8.0)
    parser.add_argument("--plateau-average-improvement-m", type=float, default=80.0)
    parser.add_argument("--plateau-elite-fraction", type=float, default=0.50)
    parser.add_argument("--plateau-extra-mutations", type=int, default=1)
    args = parser.parse_args(argv)
    args.scoring_profiles = tuple(item.strip() for item in args.scoring_profiles.split(",") if item.strip())
    return args


def main(argv: list[str] | None = None) -> int:
    run_root = run_ladder(parse_args(argv))
    print(f"evolution_ladder_complete run={run_root} summary={run_root / 'ladder_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
