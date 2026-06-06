"""OpenF1 Monza telemetry cross-check helpers.

OpenF1 is not the primary calibration source for Physics V2. This module keeps
an independent raw-data sanity check for lap timing, speed, gear, throttle,
brake, and approximate position-derived curvature.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from f1rl.config import ARTIFACTS_DIR, MONZA_LENGTH_METERS

OPENF1_BASE_URL = "https://api.openf1.org/v1"
DEFAULT_DRIVERS = ("VER", "NOR", "PIA", "LEC", "SAI", "HAM", "RUS")
DEFAULT_YEARS = (2024, 2023)
DEFAULT_SESSIONS = ("Q", "FP2", "FP3")
SESSION_NAME_ALIASES = {
    "Q": "Qualifying",
    "QUALIFYING": "Qualifying",
    "FP1": "Practice 1",
    "PRACTICE 1": "Practice 1",
    "FP2": "Practice 2",
    "PRACTICE 2": "Practice 2",
    "FP3": "Practice 3",
    "PRACTICE 3": "Practice 3",
    "R": "Race",
    "RACE": "Race",
}
ApiGet = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _safe_segment(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "-" for char in text)
    return safe.strip("-") or "unknown"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _parse_openf1_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _format_openf1_datetime(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _api_get(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    url = f"{OPENF1_BASE_URL}/{endpoint}?{query}"
    payload: Any = None
    for attempt in range(5):
        try:
            with urlopen(url, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            retry_after = exc.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay_s = float(retry_after)
                except ValueError:
                    delay_s = 2.5 * (attempt + 1)
            else:
                delay_s = 2.5 * (attempt + 1)
            time.sleep(min(delay_s, 20.0))
    if not isinstance(payload, list):
        raise ValueError(f"OpenF1 {endpoint} returned a non-list payload")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _delayed_api_get(delay_s: float) -> ApiGet:
    last_call_time = 0.0

    def get(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal last_call_time
        now_s = time.perf_counter()
        wait_s = max(float(delay_s) - (now_s - last_call_time), 0.0)
        if wait_s > 0.0:
            time.sleep(wait_s)
        result = _api_get(endpoint, params)
        last_call_time = time.perf_counter()
        return result

    return get


def _quantile(values: list[float], quantile: float) -> float | None:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return None
    return float(np.quantile(clean, quantile))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(value)]
    if not clean:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "p10": _quantile(clean, 0.10),
        "p50": _quantile(clean, 0.50),
        "p90": _quantile(clean, 0.90),
        "max": max(clean),
        "mean": float(sum(clean) / len(clean)),
    }


def _session_label(session_name: str) -> str:
    normalized = session_name.strip().upper()
    return SESSION_NAME_ALIASES.get(normalized, session_name.strip())


def _find_monza_session(
    *,
    year: int,
    session_name: str,
    api_get: ApiGet = _api_get,
) -> tuple[dict[str, Any] | None, str | None]:
    sessions = api_get("sessions", {"year": year, "country_name": "Italy"})
    wanted_session = _session_label(session_name)
    for session in sessions:
        if session.get("location") == "Monza" and session.get("session_name") == wanted_session:
            return session, None
    return None, f"no_openf1_monza_session_for_{year}_{wanted_session}"


def _driver_number_map(session_key: int, *, api_get: ApiGet = _api_get) -> dict[str, dict[str, Any]]:
    drivers = api_get("drivers", {"session_key": session_key})
    by_acronym: dict[str, dict[str, Any]] = {}
    for row in drivers:
        acronym = str(row.get("name_acronym", "")).upper()
        if acronym:
            by_acronym[acronym] = row
    return by_acronym


def _clean_fastest_lap(laps: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for lap in laps:
        duration = _optional_float(lap.get("lap_duration"))
        if duration is None:
            continue
        if bool(lap.get("is_pit_out_lap")):
            continue
        if not lap.get("date_start"):
            continue
        candidates.append(lap)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["lap_duration"]))


def _time_seconds(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float64)
    start = _parse_openf1_datetime(str(rows[0]["date"]))
    return np.asarray(
        [(_parse_openf1_datetime(str(row["date"])) - start).total_seconds() for row in rows],
        dtype=np.float64,
    )


def _car_data_summary(car_data: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [_optional_float(row.get("speed")) for row in car_data]
    speed_values = [value for value in speeds if value is not None]
    throttle_values = [
        value
        for value in (_optional_float(row.get("throttle")) for row in car_data)
        if value is not None
    ]
    brake_values = [value for value in (_optional_float(row.get("brake")) for row in car_data) if value is not None]
    rpm_values = [value for value in (_optional_float(row.get("rpm")) for row in car_data) if value is not None]
    gear_values = [
        int(value)
        for value in (_optional_float(row.get("n_gear")) for row in car_data)
        if value is not None
    ]
    return {
        "sample_count": len(car_data),
        "speed_kph": _stats(speed_values),
        "throttle_pct": _stats(throttle_values),
        "brake_sample_rate": float(sum(1 for value in brake_values if value > 0.5) / len(brake_values))
        if brake_values
        else None,
        "rpm": _stats(rpm_values),
        "gear_min": min(gear_values) if gear_values else None,
        "gear_max": max(gear_values) if gear_values else None,
    }


def _location_summary(
    location_data: list[dict[str, Any]],
    car_data: list[dict[str, Any]],
    *,
    target_track_length_m: float,
) -> dict[str, Any]:
    if len(location_data) < 5 or len(car_data) < 2:
        return {
            "sample_count": len(location_data),
            "usable": False,
            "reason": "not_enough_samples",
        }
    location_sorted = sorted(location_data, key=lambda row: str(row.get("date", "")))
    car_sorted = sorted(car_data, key=lambda row: str(row.get("date", "")))
    x = np.asarray([_optional_float(row.get("x")) or 0.0 for row in location_sorted], dtype=np.float64)
    y = np.asarray([_optional_float(row.get("y")) or 0.0 for row in location_sorted], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 5:
        return {"sample_count": len(location_data), "usable": False, "reason": "invalid_location_samples"}
    x = x[valid]
    y = y[valid]
    location_sorted = [row for row, keep in zip(location_sorted, valid, strict=True) if bool(keep)]
    deltas = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    raw_distance = np.concatenate(([0.0], np.cumsum(deltas)))
    raw_length = float(raw_distance[-1])
    if raw_length <= 1e-6:
        return {"sample_count": len(location_data), "usable": False, "reason": "zero_path_length"}

    scale = target_track_length_m / raw_length
    x_m = x * scale
    y_m = y * scale
    distance_m = raw_distance * scale
    location_t = _time_seconds(location_sorted)
    car_t = _time_seconds(car_sorted)
    car_speed = np.asarray([_optional_float(row.get("speed")) or 0.0 for row in car_sorted], dtype=np.float64)
    speed_kph = np.interp(location_t, car_t, car_speed)

    heading = np.unwrap(np.arctan2(np.gradient(y_m), np.gradient(x_m)))
    if heading.size >= 9:
        heading = np.convolve(heading, np.ones(9, dtype=np.float64) / 9.0, mode="same")
    ds = np.gradient(distance_m)
    curvature = np.divide(
        np.gradient(heading),
        ds,
        out=np.zeros_like(ds),
        where=np.abs(ds) > 1e-6,
    )
    lateral_g = (speed_kph / 3.6) ** 2 * np.abs(curvature) / 9.81
    return {
        "sample_count": len(location_data),
        "usable": True,
        "raw_path_length_units": raw_length,
        "scale_to_monza_m_per_unit": scale,
        "scaled_path_length_m": float(distance_m[-1]),
        "curvature_abs_rad_per_m": _stats(np.abs(curvature).tolist()),
        "lateral_g_from_scaled_location": _stats(lateral_g.tolist()),
    }


def _lap_window(lap: dict[str, Any]) -> tuple[str, str] | None:
    duration = _optional_float(lap.get("lap_duration"))
    date_start = lap.get("date_start")
    if duration is None or not date_start:
        return None
    start = _parse_openf1_datetime(str(date_start))
    end = start + timedelta(seconds=duration)
    return _format_openf1_datetime(start), _format_openf1_datetime(end)


def _fetch_lap_streams(
    *,
    session_key: int,
    driver_number: int,
    lap: dict[str, Any],
    api_get: ApiGet,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    window = _lap_window(lap)
    if window is None:
        return [], [], "lap_missing_time_window"
    start, end = window
    params = {
        "session_key": session_key,
        "driver_number": driver_number,
        "date>": start,
        "date<": end,
    }
    car_data = api_get("car_data", params)
    location_data = api_get("location", params)
    return car_data, location_data, None


def _selected_row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    lap = summary.get("lap", {})
    session = summary.get("session", {})
    driver_info = summary.get("driver_info", {})
    car_summary = summary.get("car_data_summary", {})
    speed_summary = car_summary.get("speed_kph", {}) if isinstance(car_summary, dict) else {}
    return {
        "year": summary.get("year"),
        "session": session.get("session_name") if isinstance(session, dict) else None,
        "driver": summary.get("driver"),
        "driver_number": driver_info.get("driver_number") if isinstance(driver_info, dict) else None,
        "lap_number": lap.get("lap_number") if isinstance(lap, dict) else None,
        "lap_duration_s": summary.get("lap_duration_s"),
        "max_speed_kph": speed_summary.get("max") if isinstance(speed_summary, dict) else None,
        "mean_speed_kph": speed_summary.get("mean") if isinstance(speed_summary, dict) else None,
        "gear_min": car_summary.get("gear_min") if isinstance(car_summary, dict) else None,
        "gear_max": car_summary.get("gear_max") if isinstance(car_summary, dict) else None,
        "speed_traps_kph": summary.get("speed_traps_kph"),
        "summary_path": summary.get("summary_path"),
        "reused_existing_summary": True,
    }


def fetch_openf1_crosscheck(
    *,
    years: list[int],
    sessions: list[str],
    drivers: list[str],
    output_dir: Path,
    target_track_length_m: float = MONZA_LENGTH_METERS,
    api_get: ApiGet = _api_get,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": "OpenF1 REST API",
        "openf1_base_url": OPENF1_BASE_URL,
        "role": "independent_calibration_cross_check",
        "notes": [
            "OpenF1 is used as an independent sanity check, not as the primary Physics V2 calibration target.",
            "Location curvature is scaled to the configured Monza length because OpenF1 location coordinates have arbitrary origin/scale.",
        ],
        "requested_years": years,
        "requested_sessions": sessions,
        "requested_drivers": drivers,
        "target_track_length_m": target_track_length_m,
        "selected_laps": [],
        "skipped": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        for requested_session in sessions:
            try:
                session, reason = _find_monza_session(
                    year=year,
                    session_name=requested_session,
                    api_get=api_get,
                )
            except Exception as exc:
                manifest["skipped"].append(
                    {
                        "year": year,
                        "session": requested_session,
                        "reason": "openf1_session_fetch_error",
                        "error": str(exc),
                    }
                )
                continue
            if session is None:
                manifest["skipped"].append(
                    {"year": year, "session": requested_session, "reason": reason}
                )
                continue
            session_key = int(session["session_key"])
            try:
                driver_map = _driver_number_map(session_key, api_get=api_get)
            except Exception as exc:
                manifest["skipped"].append(
                    {
                        "year": year,
                        "session": requested_session,
                        "session_key": session_key,
                        "reason": "openf1_drivers_fetch_error",
                        "error": str(exc),
                    }
                )
                continue
            session_dir = output_dir / f"monza_{year}_{_safe_segment(session['session_name'])}"
            _write_json(session_dir / "session.json", session)
            _write_json(session_dir / "drivers.json", list(driver_map.values()))
            for driver in drivers:
                driver_info = driver_map.get(driver.upper())
                if driver_info is None:
                    manifest["skipped"].append(
                        {
                            "year": year,
                            "session": requested_session,
                            "driver": driver,
                            "reason": "driver_not_in_openf1_session",
                        }
                    )
                    continue
                driver_number = int(driver_info["driver_number"])
                try:
                    laps = api_get("laps", {"session_key": session_key, "driver_number": driver_number})
                except Exception as exc:
                    manifest["skipped"].append(
                        {
                            "year": year,
                            "session": requested_session,
                            "driver": driver,
                            "reason": "openf1_laps_fetch_error",
                            "error": str(exc),
                        }
                    )
                    continue
                lap = _clean_fastest_lap(laps)
                if lap is None:
                    manifest["skipped"].append(
                        {
                            "year": year,
                            "session": requested_session,
                            "driver": driver,
                            "reason": "no_clean_timed_openf1_lap",
                        }
                    )
                    continue
                lap_number = int(float(lap["lap_number"]))
                lap_dir = session_dir / f"{driver.upper()}_lap{lap_number:03d}"
                raw_laps_path = lap_dir / "laps_raw.json"
                raw_car_path = lap_dir / "car_data_raw.json"
                raw_location_path = lap_dir / "location_raw.json"
                summary_path = lap_dir / "summary.json"
                if reuse_existing and summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    manifest["selected_laps"].append(_selected_row_from_summary(summary))
                    continue
                try:
                    car_data, location_data, stream_error = _fetch_lap_streams(
                        session_key=session_key,
                        driver_number=driver_number,
                        lap=lap,
                        api_get=api_get,
                    )
                except Exception as exc:
                    car_data = []
                    location_data = []
                    stream_error = f"openf1_stream_fetch_error:{exc}"
                if stream_error is not None:
                    manifest["skipped"].append(
                        {
                            "year": year,
                            "session": requested_session,
                            "driver": driver,
                            "lap_number": lap_number,
                            "reason": stream_error,
                        }
                    )
                    continue
                _write_json(raw_laps_path, laps)
                _write_json(raw_car_path, car_data)
                _write_json(raw_location_path, location_data)
                summary = {
                    "year": year,
                    "session": session,
                    "driver": driver.upper(),
                    "driver_info": driver_info,
                    "lap": lap,
                    "lap_duration_s": _optional_float(lap.get("lap_duration")),
                    "sector_times_s": [
                        _optional_float(lap.get("duration_sector_1")),
                        _optional_float(lap.get("duration_sector_2")),
                        _optional_float(lap.get("duration_sector_3")),
                    ],
                    "speed_traps_kph": {
                        "i1": _optional_float(lap.get("i1_speed")),
                        "i2": _optional_float(lap.get("i2_speed")),
                        "st": _optional_float(lap.get("st_speed")),
                    },
                    "car_data_summary": _car_data_summary(car_data),
                    "location_summary": _location_summary(
                        location_data,
                        car_data,
                        target_track_length_m=target_track_length_m,
                    ),
                    "raw_laps_path": str(raw_laps_path),
                    "raw_car_data_path": str(raw_car_path),
                    "raw_location_path": str(raw_location_path),
                    "summary_path": str(summary_path),
                }
                _write_json(summary_path, summary)
                manifest["selected_laps"].append(
                    {
                        "year": year,
                        "session": session["session_name"],
                        "driver": driver.upper(),
                        "driver_number": driver_number,
                        "lap_number": lap_number,
                        "lap_duration_s": summary["lap_duration_s"],
                        "max_speed_kph": summary["car_data_summary"]["speed_kph"]["max"],
                        "mean_speed_kph": summary["car_data_summary"]["speed_kph"]["mean"],
                        "gear_min": summary["car_data_summary"]["gear_min"],
                        "gear_max": summary["car_data_summary"]["gear_max"],
                        "speed_traps_kph": summary["speed_traps_kph"],
                        "summary_path": str(summary_path),
                    }
                )

    selected = manifest["selected_laps"]
    lap_times = [float(row["lap_duration_s"]) for row in selected if row.get("lap_duration_s") is not None]
    max_speeds = [float(row["max_speed_kph"]) for row in selected if row.get("max_speed_kph") is not None]
    mean_speeds = [float(row["mean_speed_kph"]) for row in selected if row.get("mean_speed_kph") is not None]
    speed_trap_values: list[float] = []
    for row in selected:
        traps = row.get("speed_traps_kph", {})
        if isinstance(traps, dict):
            speed_trap_values.extend(
                float(value) for value in traps.values() if isinstance(value, int | float)
            )
    manifest["summary"] = {
        "selected_lap_count": len(selected),
        "skipped_count": len(manifest["skipped"]),
        "lap_duration_s": _stats(lap_times),
        "max_speed_kph": _stats(max_speeds),
        "mean_speed_kph": _stats(mean_speeds),
        "speed_traps_kph": _stats(speed_trap_values),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OpenF1 Monza calibration cross-check data.")
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--drivers", default=",".join(DEFAULT_DRIVERS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "calibration" / "openf1-monza-crosscheck",
    )
    parser.add_argument("--target-track-length-m", type=float, default=MONZA_LENGTH_METERS)
    parser.add_argument(
        "--request-delay-s",
        type=float,
        default=0.5,
        help="Minimum delay between live OpenF1 requests.",
    )
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_manifest(manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    print(
        f"openf1_crosscheck_complete selected={summary['selected_lap_count']} "
        f"skipped={summary['skipped_count']} manifest={manifest['manifest_path']}"
    )
    print(
        f"lap_p50={summary['lap_duration_s']['p50']}s "
        f"max_speed_p90={summary['max_speed_kph']['p90']}kph "
        f"trap_p90={summary['speed_traps_kph']['p90']}kph"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_get = _delayed_api_get(float(args.request_delay_s)) if args.request_delay_s > 0.0 else _api_get
    manifest = fetch_openf1_crosscheck(
        years=_parse_csv_ints(args.years),
        sessions=_parse_csv_strings(args.sessions),
        drivers=_parse_csv_strings(args.drivers),
        output_dir=args.output_dir,
        target_track_length_m=float(args.target_track_length_m),
        api_get=api_get,
        reuse_existing=not bool(args.no_reuse_existing),
    )
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        _print_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
