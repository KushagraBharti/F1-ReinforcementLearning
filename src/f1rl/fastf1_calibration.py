"""FastF1 reference fetch, summarize, and compare CLI."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

from f1rl.calibration import (
    REFERENCE_CSV,
    REFERENCE_SUMMARY,
    calibration_report,
    reference_summary_from_csv,
)
from f1rl.config import ARTIFACTS_DIR, ASSETS_DIR, PHYSICS_MODELS

DEFAULT_SOURCE = "Fast-F1 2024 Italian Grand Prix Qualifying VER fastest lap"
DEFAULT_MULTI_DRIVERS = ("VER", "NOR", "PIA", "LEC", "SAI", "HAM", "RUS")
DEFAULT_MULTI_YEARS = (2024, 2023, 2022)
DEFAULT_MULTI_SESSIONS = ("Q", "FP2", "FP3")
SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _safe_segment(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "NaT", "nan", "None"}


def _truthy(value: object) -> bool:
    if _is_missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _jsonable_value(value: object) -> Any:
    if _is_missing(value):
        return None
    dynamic_value = cast(Any, value)
    if hasattr(dynamic_value, "item"):
        try:
            return _jsonable_value(dynamic_value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(dynamic_value, "isoformat"):
        try:
            return dynamic_value.isoformat()
        except (TypeError, ValueError):
            pass
    if hasattr(dynamic_value, "total_seconds"):
        try:
            return float(dynamic_value.total_seconds())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _lap_time_seconds(lap: Any) -> float | None:
    lap_time = cast(Any, lap.get("LapTime") if hasattr(lap, "get") else None)
    if _is_missing(lap_time):
        return None
    if hasattr(lap_time, "total_seconds"):
        return float(lap_time.total_seconds())
    return None


def _lap_metadata(lap: Any) -> dict[str, Any]:
    if not hasattr(lap, "items"):
        return {}
    return {str(key): _jsonable_value(value) for key, value in lap.items()}


def _clean_lap_reject_reasons(lap: Any) -> list[str]:
    reasons: list[str] = []
    if _lap_time_seconds(lap) is None:
        reasons.append("missing_lap_time")
    if not _is_missing(lap.get("PitInTime")):
        reasons.append("pit_in_lap")
    if not _is_missing(lap.get("PitOutTime")):
        reasons.append("pit_out_lap")
    if "IsAccurate" in lap and not _truthy(lap.get("IsAccurate")):
        reasons.append("not_fastf1_accurate")
    if _truthy(lap.get("Deleted")):
        reasons.append("deleted_lap")
    track_status = "" if _is_missing(lap.get("TrackStatus")) else str(lap.get("TrackStatus")).strip()
    if track_status and set(track_status) != {"1"}:
        reasons.append(f"non_green_track_status_{track_status}")
    compound = "" if _is_missing(lap.get("Compound")) else str(lap.get("Compound")).upper()
    if compound and compound not in SLICK_COMPOUNDS:
        reasons.append(f"non_slick_compound_{compound}")
    return reasons


def _select_fast_clean_laps(
    loaded_session: Any,
    *,
    drivers: list[str],
    max_laps_per_driver: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    selected: list[Any] = []
    rejected: list[dict[str, Any]] = []
    laps = loaded_session.laps
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver)
        if len(driver_laps) == 0:
            rejected.append({"driver": driver, "reasons": ["no_laps_for_driver"]})
            continue
        sorted_laps = driver_laps.sort_values("LapTime")
        clean_for_driver = 0
        for _, lap in sorted_laps.iterrows():
            reasons = _clean_lap_reject_reasons(lap)
            if reasons:
                rejected.append(
                    {
                        "driver": driver,
                        "lap_number": _jsonable_value(lap.get("LapNumber")),
                        "lap_time_s": _lap_time_seconds(lap),
                        "reasons": reasons,
                    }
                )
                continue
            selected.append(lap)
            clean_for_driver += 1
            if clean_for_driver >= max_laps_per_driver:
                break
        if clean_for_driver == 0:
            rejected.append({"driver": driver, "reasons": ["no_clean_dry_lap_selected"]})
    return selected, rejected


def _ensure_time_column(lap: Any, telemetry: Any) -> None:
    if "Time" in telemetry.columns or "SessionTime" not in telemetry.columns:
        return
    lap_start = lap.get("LapStartTime") if hasattr(lap, "get") else None
    if lap_start is not None:
        telemetry["Time"] = telemetry["SessionTime"] - lap_start


def _lap_number_segment(lap: Any) -> str:
    lap_number = lap.get("LapNumber") if hasattr(lap, "get") else "unknown"
    try:
        return f"{int(float(lap_number)):03d}"
    except (TypeError, ValueError):
        return _safe_segment(lap_number)


def _session_priority(session_name: str) -> str:
    normalized = session_name.upper()
    if normalized == "Q":
        return "primary_qualifying_clean_dry"
    if normalized in {"FP2", "FP3"}:
        return "secondary_practice_clean_dry"
    if normalized in {"R", "RACE"}:
        return "lower_priority_race_clean_dry"
    return "calibration_candidate_clean_dry"


def _save_fastf1_lap_reference(
    lap: Any,
    *,
    year: int,
    event: str,
    session_name: str,
    output_dir: Path,
    fastf1_version: str,
) -> dict[str, Any]:
    driver = str(lap.get("Driver"))
    lap_number = _lap_number_segment(lap)
    lap_dir = output_dir / f"monza_{year}_{_safe_segment(session_name)}" / f"{_safe_segment(driver)}_lap{lap_number}"
    lap_dir.mkdir(parents=True, exist_ok=True)

    car_data = lap.get_car_data()
    pos_data = lap.get_pos_data()
    telemetry = lap.get_telemetry().add_distance()
    _ensure_time_column(lap, telemetry)

    raw_car_path = lap_dir / "car_data_raw.csv"
    raw_pos_path = lap_dir / "pos_data_raw.csv"
    processed_path = lap_dir / "telemetry_processed_add_distance.csv"
    summary_path = lap_dir / "summary.json"
    metadata_path = lap_dir / "lap_metadata.json"

    car_data.to_csv(raw_car_path, index=False)
    pos_data.to_csv(raw_pos_path, index=False)
    telemetry.to_csv(processed_path, index=False)

    source = (
        f"FastF1 {year} {event} {session_name} {driver} lap {lap_number} "
        f"{_session_priority(session_name)}"
    )
    summarize_reference(input_path=processed_path, output_path=summary_path, source=source)
    metadata = _lap_metadata(lap)
    metadata.update(
        {
            "fastf1_version": fastf1_version,
            "year": year,
            "event": event,
            "session": session_name,
            "driver": driver,
            "priority": _session_priority(session_name),
            "clean_filter_reasons": [],
            "raw_car_data_path": str(raw_car_path),
            "raw_pos_data_path": str(raw_pos_path),
            "processed_telemetry_path": str(processed_path),
            "summary_path": str(summary_path),
        }
    )
    _write_json(metadata_path, metadata)
    return {
        "year": year,
        "event": event,
        "session": session_name,
        "driver": driver,
        "team": _jsonable_value(lap.get("Team")),
        "lap_number": _jsonable_value(lap.get("LapNumber")),
        "lap_time_s": _lap_time_seconds(lap),
        "compound": _jsonable_value(lap.get("Compound")),
        "stint": _jsonable_value(lap.get("Stint")),
        "track_status": _jsonable_value(lap.get("TrackStatus")),
        "priority": _session_priority(session_name),
        "raw_car_data_path": str(raw_car_path),
        "raw_pos_data_path": str(raw_pos_path),
        "processed_telemetry_path": str(processed_path),
        "summary_path": str(summary_path),
        "metadata_path": str(metadata_path),
    }


def _quantile(values: list[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _metric_distribution(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "p10": _quantile(clean, 0.10),
        "p50": _quantile(clean, 0.50),
        "p90": _quantile(clean, 0.90),
        "max": max(clean),
        "mean": sum(clean) / len(clean),
    }


def _append_metric(bucket: dict[str, list[float]], name: str, value: object) -> None:
    if value is None:
        return
    try:
        bucket.setdefault(name, []).append(float(cast(Any, value)))
    except (TypeError, ValueError):
        return


def _cluster_sustained_targets(
    targets: list[dict[str, Any]],
    *,
    center_gap_m: float = 250.0,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    entries: list[tuple[float, dict[str, Any]]] = []
    for target in targets:
        try:
            start = float(target["start_distance_m"])
            end = float(target["end_distance_m"])
        except (KeyError, TypeError, ValueError):
            continue
        entries.append(((start + end) * 0.5, target))
    entries.sort(key=lambda item: item[0])

    clusters: list[list[tuple[float, dict[str, Any]]]] = []
    for center, target in entries:
        if not clusters:
            clusters.append([(center, target)])
            continue
        current_centers = [item[0] for item in clusters[-1]]
        current_center = sum(current_centers) / len(current_centers)
        if abs(center - current_center) <= center_gap_m:
            clusters[-1].append((center, target))
        else:
            clusters.append([(center, target)])

    clustered: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for index, cluster in enumerate(clusters, start=1):
        metrics: dict[str, list[float]] = {}
        for center, target in cluster:
            _append_metric(metrics, "center_distance_m", center)
            for name in (
                "start_distance_m",
                "end_distance_m",
                "length_m",
                "mean_speed_kph",
                "min_speed_kph",
                "max_speed_kph",
                "lateral_g_p75",
                "lateral_g_p90",
                "lateral_g_max",
                "curvature_abs_p90_rad_per_m",
                "radius_p10_m",
            ):
                _append_metric(metrics, name, target.get(name))
        clustered[f"section_{index:02d}"] = {
            name: _metric_distribution(values) for name, values in sorted(metrics.items())
        }
    return clustered


def summarize_multi_reference(
    *,
    output_dir: Path,
    output_path: Path | None = None,
    summary_paths: list[Path] | None = None,
) -> dict[str, Any]:
    paths = summary_paths or sorted(output_dir.glob("**/summary.json"))
    lap_metrics: dict[str, list[float]] = {}
    sustained_targets: list[dict[str, Any]] = []
    braking_metrics: dict[str, list[float]] = {}
    corner_metrics: dict[str, list[float]] = {}
    sources: list[dict[str, Any]] = []

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("trace_features")
        if not isinstance(features, dict):
            continue
        sources.append(
            {
                "source": data.get("source"),
                "summary_path": str(path),
                "lap_time_s": data.get("lap_time_s"),
                "distance_m": data.get("distance_m"),
                "max_speed_kph": data.get("max_speed_kph"),
                "mean_speed_kph": data.get("mean_speed_kph"),
            }
        )
        for name in ("lap_time_s", "distance_m", "max_speed_kph", "mean_speed_kph", "p10_speed_kph", "p50_speed_kph", "p90_speed_kph"):
            _append_metric(lap_metrics, name, data.get(name))
        for target in features.get("sustained_corner_targets", []):
            sustained_targets.append(target)
        for zone in features.get("braking_zones", []):
            for name in ("start_distance_m", "length_m", "start_speed_kph", "min_speed_kph", "speed_drop_kph", "mean_decel_mps2"):
                _append_metric(braking_metrics, name, zone.get(name))
        for corner in features.get("corner_speed_targets", []):
            for name in ("min_speed_kph", "entry_speed_kph", "exit_speed_kph", "lateral_g_p75", "lateral_g_p90"):
                _append_metric(corner_metrics, name, corner.get(name))

    aggregate = {
        "output_dir": str(output_dir),
        "summary_count": len(sources),
        "sources": sources,
        "lap_distributions": {
            name: _metric_distribution(values) for name, values in sorted(lap_metrics.items())
        },
        "section_distributions": {
            "sustained_high_speed_curves": _cluster_sustained_targets(sustained_targets),
            "sustained_high_speed_curve_raw_target_count": len(sustained_targets),
            "braking_zones": {
                name: _metric_distribution(values) for name, values in sorted(braking_metrics.items())
            },
            "corner_min_exit_speeds": {
                name: _metric_distribution(values) for name, values in sorted(corner_metrics.items())
            },
        },
    }
    if output_path is not None:
        _write_json(output_path, aggregate)
    return aggregate


def summarize_reference(
    *,
    input_path: Path = REFERENCE_CSV,
    output_path: Path | None = None,
    source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    summary = reference_summary_from_csv(input_path, source=source)
    if output_path is not None:
        _write_json(output_path, summary)
    return summary


def _distribution_value(distribution: dict[str, Any], key: str) -> float | None:
    value = distribution.get(key)
    if value is None:
        return None
    return float(value)


def _sustained_multi_lap_comparison(model: dict[str, Any], multi_summary: dict[str, Any]) -> dict[str, Any]:
    diagnostic = model.get("estimates", {}).get("sustained_corner_diagnostic", {})
    rows = diagnostic.get("rows", [])
    clusters = multi_summary.get("section_distributions", {}).get("sustained_high_speed_curves", {})
    if not rows or not clusters:
        return {"rows": [], "matched_sections": 0}

    cluster_rows: list[tuple[str, dict[str, Any], float]] = []
    for name, metrics in clusters.items():
        center = _distribution_value(metrics.get("center_distance_m", {}), "p50")
        if center is not None:
            cluster_rows.append((str(name), metrics, center))

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        row_center = (float(row["reference_start_distance_m"]) + float(row["reference_end_distance_m"])) * 0.5
        nearest = min(cluster_rows, key=lambda item: abs(item[2] - row_center), default=None)
        if nearest is None:
            continue
        cluster_name, metrics, cluster_center = nearest
        speed_distribution = metrics.get("mean_speed_kph", {})
        lateral_g_distribution = metrics.get("lateral_g_p90", {})
        reference_run = row["reference_run"]
        run_speed = float(reference_run.get("mean_speed_kph", 0.0))
        run_lateral_g_p90 = float(reference_run.get("lateral_g_p90", 0.0))
        speed_p10 = _distribution_value(speed_distribution, "p10")
        speed_p90 = _distribution_value(speed_distribution, "p90")
        lateral_g_p50 = _distribution_value(lateral_g_distribution, "p50")
        comparisons.append(
            {
                "reference_section": row["name"],
                "matched_multi_lap_section": cluster_name,
                "center_distance_error_m": row_center - cluster_center,
                "sim_mean_speed_kph": run_speed,
                "multi_lap_mean_speed_p10_kph": speed_p10,
                "multi_lap_mean_speed_p50_kph": _distribution_value(speed_distribution, "p50"),
                "multi_lap_mean_speed_p90_kph": speed_p90,
                "sim_speed_within_multi_lap_p10_p90": (
                    speed_p10 is not None and speed_p90 is not None and speed_p10 <= run_speed <= speed_p90
                ),
                "sim_lateral_g_p90": run_lateral_g_p90,
                "multi_lap_lateral_g_p90_p50": lateral_g_p50,
                "multi_lap_lateral_g_p90_p90": _distribution_value(lateral_g_distribution, "p90"),
                "sim_lateral_g_p90_margin_to_multi_lap_p50": (
                    run_lateral_g_p90 - lateral_g_p50 if lateral_g_p50 is not None else None
                ),
                "sim_p95_abs_lateral_error_m": reference_run.get("p95_abs_lateral_error_m"),
                "sim_steering_saturation_rate": reference_run.get("steering_saturation_rate"),
                "sim_control_pass": reference_run.get("control_pass"),
            }
        )
    if not comparisons:
        return {"rows": [], "matched_sections": 0}
    speed_pass_rate = sum(1 for row in comparisons if row["sim_speed_within_multi_lap_p10_p90"]) / len(comparisons)
    margins = [
        float(row["sim_lateral_g_p90_margin_to_multi_lap_p50"])
        for row in comparisons
        if row["sim_lateral_g_p90_margin_to_multi_lap_p50"] is not None
    ]
    return {
        "rows": comparisons,
        "matched_sections": len(comparisons),
        "sim_speed_within_multi_lap_p10_p90_rate": speed_pass_rate,
        "min_sim_lateral_g_p90_margin_to_multi_lap_p50": min(margins) if margins else None,
    }


def compare_reference(*, physics_model: str = "both", multi_summary_path: Path | None = None) -> dict[str, Any]:
    if physics_model != "both" and physics_model not in PHYSICS_MODELS:
        valid = ", ".join([*sorted(PHYSICS_MODELS), "both"])
        raise ValueError(f"physics_model must be one of: {valid}.")
    report = calibration_report()
    if physics_model == "both":
        result = report
    else:
        result = {
            "targets": report["targets"],
            "reference_trace_features": report["reference_trace_features"],
            "physics_models": {
                physics_model: report["physics_models"][physics_model],
            },
        }
    if multi_summary_path is not None:
        multi_summary = json.loads(multi_summary_path.read_text(encoding="utf-8"))
        result["multi_lap_reference_summary_path"] = str(multi_summary_path)
        result["multi_lap_reference_summary"] = multi_summary
        for model in result["physics_models"].values():
            model["multi_lap_sustained_corner_comparison"] = _sustained_multi_lap_comparison(
                model,
                multi_summary,
            )
    return result


def fetch_reference(
    *,
    year: int,
    event: str,
    session: str,
    driver: str,
    output_dir: Path,
    cache_dir: Path,
) -> dict[str, str]:
    try:
        fastf1 = importlib.import_module("fastf1")
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("FastF1 is required. Install the calibration extra before using fetch.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    loaded_session = fastf1.get_session(year, event, session)
    loaded_session.load()
    laps = loaded_session.laps.pick_drivers(driver)
    lap = laps.pick_fastest()
    telemetry = lap.get_telemetry().add_distance()
    if "Time" not in telemetry.columns and "SessionTime" in telemetry.columns:
        lap_start = lap.get("LapStartTime")
        if lap_start is not None:
            telemetry["Time"] = telemetry["SessionTime"] - lap_start
    csv_path = output_dir / f"monza_{year}_{session}_{driver}_telemetry.csv"
    summary_path = output_dir / f"monza_{year}_{session}_{driver}_summary.json"
    columns = [
        "Time",
        "SessionTime",
        "Date",
        "Distance",
        "X",
        "Y",
        "Speed",
        "RPM",
        "nGear",
        "Throttle",
        "Brake",
        "DRS",
        "Source",
    ]
    available_columns = [column for column in columns if column in telemetry.columns]
    telemetry[available_columns].to_csv(csv_path, index=False)
    source = f"Fast-F1 {year} {event} {session} {driver} fastest lap"
    summary = summarize_reference(input_path=csv_path, output_path=summary_path, source=source)
    return {
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "source": str(summary["source"]),
    }


def fetch_multi_reference(
    *,
    years: list[int],
    event: str,
    sessions: list[str],
    drivers: list[str],
    output_dir: Path,
    cache_dir: Path,
    max_laps_per_driver: int,
    include_race_clean_air: bool = False,
) -> dict[str, Any]:
    try:
        fastf1 = importlib.import_module("fastf1")
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("FastF1 is required. Install the calibration extra before using fetch-multi.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    fastf1_version = str(getattr(fastf1, "__version__", "unknown"))
    requested_sessions = [session.upper() for session in sessions]
    if include_race_clean_air and "R" not in requested_sessions:
        requested_sessions.append("R")

    manifest: dict[str, Any] = {
        "fastf1_version": fastf1_version,
        "event": event,
        "years": years,
        "sessions": requested_sessions,
        "drivers": drivers,
        "clean_filter": {
            "track_status": "green_flag_only_status_1",
            "compounds": sorted(SLICK_COMPOUNDS),
            "requires_fastf1_is_accurate": True,
            "excludes_deleted_laps": True,
            "excludes_pit_in_out_laps": True,
        },
        "selected_laps": [],
        "rejected_laps": [],
        "session_errors": [],
    }
    for year in years:
        for session_name in requested_sessions:
            try:
                loaded_session = fastf1.get_session(year, event, session_name)
                loaded_session.load(laps=True, telemetry=True, weather=False, messages=True)
            except Exception as exc:  # pragma: no cover - live FastF1/network dependent.
                manifest["session_errors"].append(
                    {
                        "year": year,
                        "event": event,
                        "session": session_name,
                        "error": str(exc),
                    }
                )
                continue
            selected, rejected = _select_fast_clean_laps(
                loaded_session,
                drivers=drivers,
                max_laps_per_driver=max_laps_per_driver,
            )
            for rejected_lap in rejected:
                rejected_lap.update({"year": year, "event": event, "session": session_name})
                manifest["rejected_laps"].append(rejected_lap)
            for lap in selected:
                try:
                    manifest["selected_laps"].append(
                        _save_fastf1_lap_reference(
                            lap,
                            year=year,
                            event=event,
                            session_name=session_name,
                            output_dir=output_dir,
                            fastf1_version=fastf1_version,
                        )
                    )
                except Exception as exc:  # pragma: no cover - live FastF1/network dependent.
                    manifest["session_errors"].append(
                        {
                            "year": year,
                            "event": event,
                            "session": session_name,
                            "driver": _jsonable_value(lap.get("Driver")),
                            "lap_number": _jsonable_value(lap.get("LapNumber")),
                            "error": str(exc),
                        }
                    )
    summary_paths = [Path(row["summary_path"]) for row in manifest["selected_laps"]]
    aggregate_path = output_dir / "section_distribution_summary.json"
    aggregate = summarize_multi_reference(
        output_dir=output_dir,
        output_path=aggregate_path,
        summary_paths=summary_paths,
    )
    manifest["aggregate_summary_path"] = str(aggregate_path)
    manifest["aggregate_summary_count"] = aggregate["summary_count"]
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, summarize, or compare FastF1 calibration references.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch a FastF1 fastest-lap telemetry CSV and summary.")
    fetch.add_argument("--year", type=int, default=2024)
    fetch.add_argument("--event", default="Monza")
    fetch.add_argument("--session", default="Q")
    fetch.add_argument("--driver", default="VER")
    fetch.add_argument("--output-dir", type=Path, default=ASSETS_DIR / "reference" / "fastf1" / "monza_2024_Q_VER")
    fetch.add_argument("--cache-dir", type=Path, default=ARTIFACTS_DIR / "fastf1-cache")
    fetch.add_argument("--json", action="store_true")

    fetch_multi = subparsers.add_parser(
        "fetch-multi",
        help="Fetch clean dry FastF1 Monza laps with raw car/position data and aggregate section summaries.",
    )
    fetch_multi.add_argument("--years", default=",".join(str(year) for year in DEFAULT_MULTI_YEARS))
    fetch_multi.add_argument("--event", default="Monza")
    fetch_multi.add_argument("--sessions", default=",".join(DEFAULT_MULTI_SESSIONS))
    fetch_multi.add_argument("--drivers", default=",".join(DEFAULT_MULTI_DRIVERS))
    fetch_multi.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "calibration" / "fastf1-multilap")
    fetch_multi.add_argument("--cache-dir", type=Path, default=ARTIFACTS_DIR / "fastf1-cache")
    fetch_multi.add_argument("--max-laps-per-driver", type=int, default=1)
    fetch_multi.add_argument("--include-race-clean-air", action="store_true")
    fetch_multi.add_argument("--json", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize a cached FastF1 telemetry CSV.")
    summarize.add_argument("input", nargs="?", type=Path, default=REFERENCE_CSV)
    summarize.add_argument("--output", type=Path, default=REFERENCE_SUMMARY)
    summarize.add_argument("--source", default=DEFAULT_SOURCE)
    summarize.add_argument("--json", action="store_true")

    summarize_multi = subparsers.add_parser(
        "summarize-multi",
        help="Aggregate section distributions from an existing fetch-multi output directory.",
    )
    summarize_multi.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "calibration" / "fastf1-multilap")
    summarize_multi.add_argument("--output", type=Path)
    summarize_multi.add_argument("--json", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare simulator physics estimates to the reference summary.")
    compare.add_argument("--physics-model", choices=[*sorted(PHYSICS_MODELS), "both"], default="both")
    compare.add_argument("--multi-summary", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_summary(summary: dict[str, Any]) -> None:
    trace = summary["trace_features"]
    print(f"reference_summary source={summary['source']}")
    print(
        f"lap={summary['lap_time_s']:.3f}s distance={summary['distance_m']:.1f}m "
        f"max={summary['max_speed_kph']:.1f}kph mean={summary['mean_speed_kph']:.1f}kph"
    )
    print(
        f"braking_zones={len(trace['braking_zones'])} "
        f"corners={len(trace['corner_speed_targets'])} "
        f"gear_shifts={len(trace['gear_rpm']['shift_points'])}"
    )


def _print_multi_manifest(manifest: dict[str, Any]) -> None:
    print(
        f"fastf1_fetch_multi_complete selected={len(manifest['selected_laps'])} "
        f"rejected={len(manifest['rejected_laps'])} "
        f"session_errors={len(manifest['session_errors'])}"
    )
    print(
        f"manifest={manifest['manifest_path']} "
        f"section_summary={manifest['aggregate_summary_path']}"
    )


def _print_multi_summary(summary: dict[str, Any]) -> None:
    lap_times = summary["lap_distributions"].get("lap_time_s", {})
    max_speeds = summary["lap_distributions"].get("max_speed_kph", {})
    print(
        f"fastf1_multi_summary summaries={summary['summary_count']} "
        f"lap_p50={lap_times.get('p50')}s max_speed_p90={max_speeds.get('p90')}kph"
    )
    print(f"section_summary_output={summary['output_dir']}")


def _print_compare(report: dict[str, Any]) -> None:
    print(f"reference_compare source={report['targets']['source']}")
    for name, model in report["physics_models"].items():
        errors = model["error_terms"]
        print(
            f"{name}: max_speed_error={errors['max_speed_error_kph']:.1f}kph "
            f"trace_accel_mae={errors['speed_trace_accel_p95_mae_mps2']:.2f}m/s^2 "
            f"brake_zone_mae={errors['mean_abs_braking_zone_distance_error_m']:.1f}m "
            f"robust_corner_margin_min={errors['min_robust_corner_lateral_g_margin']:.2f}g "
            f"sustained_pass={errors['sustained_corner_reference_control_pass_rate']:.2f} "
            f"sustained_laterr_p95={errors['max_sustained_corner_reference_p95_abs_lateral_error_m']:.2f}m"
        )
        multi = model.get("multi_lap_sustained_corner_comparison")
        if multi:
            margin = multi.get("min_sim_lateral_g_p90_margin_to_multi_lap_p50")
            print(
                f"{name}: multi_lap_sustained_matches={multi['matched_sections']} "
                f"speed_in_p10_p90={multi['sim_speed_within_multi_lap_p10_p90_rate']:.2f} "
                f"latg_p90_margin_to_p50={margin if margin is not None else 'n/a'}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        result = fetch_reference(
            year=args.year,
            event=args.event,
            session=args.session,
            driver=args.driver,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"fastf1_fetch_complete csv={result['csv_path']} summary={result['summary_path']}")
        return 0
    if args.command == "fetch-multi":
        result = fetch_multi_reference(
            years=_parse_csv_ints(args.years),
            event=args.event,
            sessions=_parse_csv_strings(args.sessions),
            drivers=_parse_csv_strings(args.drivers),
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            max_laps_per_driver=args.max_laps_per_driver,
            include_race_clean_air=args.include_race_clean_air,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_multi_manifest(result)
        return 0
    if args.command == "summarize":
        summary = summarize_reference(input_path=args.input, output_path=args.output, source=args.source)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            _print_summary(summary)
        return 0
    if args.command == "summarize-multi":
        output_path = args.output or args.output_dir / "section_distribution_summary.json"
        summary = summarize_multi_reference(output_dir=args.output_dir, output_path=output_path)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            _print_multi_summary(summary)
        return 0
    report = compare_reference(physics_model=args.physics_model, multi_summary_path=args.multi_summary)
    if args.output is not None:
        _write_json(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_compare(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
