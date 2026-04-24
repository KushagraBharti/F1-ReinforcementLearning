from f1rl.campaign import _diagnostic_collapse, _select_family_candidates, candidate_library, parse_args


def test_diagnostic_collapse_flags_large_competence_regression() -> None:
    candidate = {
        "first_checkpoint_rate": 0.2,
        "avg_checkpoints_reached": 2.0,
        "collision_rate": 1.0,
    }
    baseline = {
        "first_checkpoint_rate": 1.0,
        "avg_checkpoints_reached": 8.0,
        "collision_rate": 0.6,
    }
    collapsed, reason = _diagnostic_collapse("competence", candidate, baseline)
    assert collapsed is True
    assert reason


def test_candidate_library_contains_all_wave_families() -> None:
    library = candidate_library("competence", logical_cars=32, max_cars=128)
    assert set(library) == {
        "reward",
        "termination",
        "observations",
        "action_control",
        "curriculum",
        "ppo",
        "performance",
    }
    assert all(len(candidates) == 3 for candidates in library.values())


def test_select_family_candidates_rotates_single_candidate_waves() -> None:
    candidates = candidate_library("competence", logical_cars=32, max_cars=128)["reward"]
    first = _select_family_candidates(candidates, limit=1, offset=0)
    second = _select_family_candidates(candidates, limit=1, offset=1)
    third = _select_family_candidates(candidates, limit=1, offset=2)

    assert [item.name for item in first] == ["reward_progress_bias"]
    assert [item.name for item in second] == ["reward_safety_bias"]
    assert [item.name for item in third] == ["reward_alignment_bias"]


def test_parse_args_accepts_start_family() -> None:
    args = parse_args(["--start-family", "termination"])
    assert args.start_family == "termination"
