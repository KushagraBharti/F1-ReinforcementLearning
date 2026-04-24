from f1rl.calibration import calibration_report, load_targets


def test_fastf1_reference_targets_available() -> None:
    targets = load_targets()
    assert 79.0 < targets.lap_time_s < 81.0
    assert 340.0 < targets.max_speed_kph < 355.0
    assert 250.0 < targets.mean_speed_kph < 270.0


def test_default_car_speed_is_near_monza_target() -> None:
    report = calibration_report()
    terminal = report["sim_estimates"]["terminal_speed_kph"]
    assert 340.0 <= terminal <= 360.0
