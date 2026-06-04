"""Named Monza sections and braking gates shared by simulator and analysis code."""

from __future__ import annotations

from dataclasses import dataclass

from f1rl.config import MONZA_LENGTH_METERS

BRAKING_GATE_EPSILON_M = 1e-3


@dataclass(frozen=True, slots=True)
class TrackSection:
    name: str
    start_m: float
    end_m: float
    target_speed_kph: float
    brake_start_m: float | None = None
    turn_in_m: float | None = None
    exit_m: float | None = None


MONZA_SECTIONS: tuple[TrackSection, ...] = (
    TrackSection("start_finish_straight", 0.0, 450.0, 340.0),
    TrackSection("rettifilo_chicane", 450.0, 1150.0, 115.0, brake_start_m=520.0, turn_in_m=720.0, exit_m=1080.0),
    TrackSection("curva_grande_roggia_run", 1150.0, 1850.0, 315.0),
    TrackSection("roggia_chicane", 1850.0, 2500.0, 135.0, brake_start_m=1950.0, turn_in_m=2140.0, exit_m=2420.0),
    TrackSection("lesmo_1", 2500.0, 3150.0, 185.0, brake_start_m=2620.0, turn_in_m=2780.0, exit_m=3030.0),
    TrackSection("lesmo_2_serraglio", 3150.0, 3850.0, 195.0, brake_start_m=3250.0, turn_in_m=3430.0, exit_m=3730.0),
    TrackSection("ascari_approach", 3850.0, 4450.0, 320.0),
    TrackSection("ascari_chicane", 4450.0, 5150.0, 165.0, brake_start_m=4560.0, turn_in_m=4740.0, exit_m=5070.0),
    TrackSection("parabolica_finish", 5150.0, MONZA_LENGTH_METERS, 205.0, brake_start_m=5260.0, turn_in_m=5450.0),
)


def section_for_progress(progress_m: float) -> TrackSection:
    lap_progress_m = progress_m % MONZA_LENGTH_METERS
    for section in MONZA_SECTIONS:
        if section.start_m <= lap_progress_m < section.end_m:
            return section
    return MONZA_SECTIONS[-1]


def distance_to_next_braking_gate(progress_m: float, *, epsilon_m: float = BRAKING_GATE_EPSILON_M) -> float:
    gates = tuple(section.brake_start_m for section in MONZA_SECTIONS if section.brake_start_m is not None)
    if not gates:
        return MONZA_LENGTH_METERS
    lap_progress_m = progress_m % MONZA_LENGTH_METERS
    epsilon_m = max(0.0, float(epsilon_m))
    for gate_m in gates:
        if abs(lap_progress_m - gate_m) <= epsilon_m:
            return 0.0
    for gate_m in gates:
        if lap_progress_m < gate_m:
            return float(gate_m - lap_progress_m)
    return float((MONZA_LENGTH_METERS - lap_progress_m) + gates[0])
