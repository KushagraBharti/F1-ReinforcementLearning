import json
from typing import Any

from f1rl.openf1_crosscheck import fetch_openf1_crosscheck


def test_openf1_crosscheck_saves_raw_and_summarizes_lap(tmp_path) -> None:
    car_rows = [
        {
            "date": f"2024-08-31T14:05:0{index}.000+00:00",
            "speed": speed,
            "rpm": 9000 + index * 200,
            "n_gear": min(8, 4 + index),
            "throttle": 100 if index < 4 else 20,
            "brake": 1 if index == 4 else 0,
        }
        for index, speed in enumerate([180, 220, 260, 300, 250, 210])
    ]
    location_rows = [
        {
            "date": f"2024-08-31T14:05:0{index}.000+00:00",
            "x": float(index * 10),
            "y": float(index * index),
        }
        for index in range(6)
    ]

    def fake_api_get(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if endpoint == "sessions":
            return [
                {
                    "session_key": 9586,
                    "session_name": "Qualifying",
                    "location": "Monza",
                    "year": 2024,
                }
            ]
        if endpoint == "drivers":
            return [{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}]
        if endpoint == "laps":
            return [
                {
                    "driver_number": 1,
                    "lap_number": 1,
                    "date_start": "2024-08-31T14:03:00.000+00:00",
                    "lap_duration": None,
                    "is_pit_out_lap": True,
                },
                {
                    "driver_number": 1,
                    "lap_number": 2,
                    "date_start": "2024-08-31T14:05:00.000+00:00",
                    "lap_duration": 80.226,
                    "duration_sector_1": 26.605,
                    "duration_sector_2": 26.993,
                    "duration_sector_3": 26.628,
                    "i1_speed": 324,
                    "i2_speed": 335,
                    "st_speed": 342,
                    "is_pit_out_lap": False,
                },
            ]
        if endpoint == "car_data":
            assert params["date>"].startswith("2024-08-31T14:05:00")
            return car_rows
        if endpoint == "location":
            return location_rows
        raise AssertionError(endpoint)

    manifest = fetch_openf1_crosscheck(
        years=[2024],
        sessions=["Q"],
        drivers=["VER"],
        output_dir=tmp_path,
        api_get=fake_api_get,
    )

    assert manifest["summary"]["selected_lap_count"] == 1
    assert manifest["summary"]["max_speed_kph"]["max"] == 300.0
    selected = manifest["selected_laps"][0]
    summary_path = tmp_path / "monza_2024_Qualifying" / "VER_lap002" / "summary.json"
    raw_car_path = tmp_path / "monza_2024_Qualifying" / "VER_lap002" / "car_data_raw.json"
    raw_location_path = tmp_path / "monza_2024_Qualifying" / "VER_lap002" / "location_raw.json"
    assert selected["summary_path"] == str(summary_path)
    assert raw_car_path.exists()
    assert raw_location_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["car_data_summary"]["gear_max"] == 8
    assert summary["car_data_summary"]["brake_sample_rate"] == 1 / 6
    assert summary["location_summary"]["usable"] is True
