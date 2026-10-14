import json
from pathlib import Path

from subset.sampler import build, PER_SUITE_USERS

ROOT = Path(__file__).resolve().parent.parent


def test_pairs_file_300_and_strata():
    data = json.loads((ROOT / "subset" / "pairs.json").read_text())
    pairs = data["pairs"]
    assert len(pairs) == 300
    counts = {}
    for p in pairs:
        counts[p["suite"]] = counts.get(p["suite"], 0) + 1
    assert counts == {"workspace": 140, "travel": 56, "banking": 54, "slack": 50}


def test_all_35_injection_tasks_present():
    data = json.loads((ROOT / "subset" / "pairs.json").read_text())
    inj = {(p["suite"], p["injection_task"]) for p in data["pairs"]}
    assert len(inj) == 35


def test_deterministic():
    a = build(seed=20261002)
    b = build(seed=20261002)
    assert a == b


def test_seed_changes_sample():
    a = build(seed=1)
    b = build(seed=2)
    assert a != b


def test_nested_150_is_strict_subset_and_strata():
    from subset.sampler import build_nested
    data = json.loads((ROOT / "subset" / "pairs.json").read_text())
    pairs = data["pairs"]
    nested = build_nested(pairs)
    assert len(nested) == 150
    counts = {}
    for p in nested:
        counts[p["suite"]] = counts.get(p["suite"], 0) + 1
    assert counts == {"workspace": 70, "travel": 28, "banking": 27, "slack": 25}
    # every nested pair is one of the 300 (strict nesting)
    keyset = {(p["suite"], p["user_task"], p["injection_task"]) for p in pairs}
    assert all((p["suite"], p["user_task"], p["injection_task"]) in keyset for p in nested)
    # all 35 injection tasks still present
    assert len({(p["suite"], p["injection_task"]) for p in nested}) == 35


def test_pairs_150_file_matches_builder():
    data = json.loads((ROOT / "subset" / "pairs_150.json").read_text())
    assert data["meta"]["n_pairs"] == 150
    assert len(data["pairs"]) == 150
