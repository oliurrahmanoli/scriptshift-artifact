"""T2 on the REAL pilot no-attack traces must reproduce the utilities the pilot notebooks logged
(`*_pilot_result.json`, DECISIONS D20/D21). This checks the pipeline, it does not create results."""
import json
from pathlib import Path

import pytest

from analysis import tables
from analysis.ingest import build_episodes, load_traces

KAGGLE = Path(__file__).resolve().parents[2] / "kaggle"
RESULTS = sorted(KAGGLE.glob("pilot-*-v1/*_pilot_result.json"))


@pytest.mark.skipif(not RESULTS, reason="pilot artefacts not present")
def test_t2_matches_pilot_result_json():
    df = build_episodes(load_traces(sorted(str(p) for p in KAGGLE.glob("pilot-*-v1/runs"))))
    t2 = tables.t2_utility(df, RESULTS)
    checked = 0
    for fp in RESULTS:
        r = json.loads(fp.read_text())
        label = r.get("label") or [d.name for d in (fp.parent / "runs").iterdir() if d.is_dir()][0]
        for suite, u in r["utility"].items():
            row = t2[(t2["model"] == label) & (t2["suite"] == suite)].iloc[0]
            assert row["utility"] == pytest.approx(u, abs=1e-9), (label, suite)
            assert row["n"] == r["n"][suite]
            checked += 1
    assert checked >= 6
    assert (t2["parse_error_rate"] == 0).all()                 # D20/D21: strict parse 0/37
    assert t2["utility_floor_ok"].all() and not t2["synthetic"].any()


@pytest.mark.skipif(not RESULTS, reason="pilot artefacts not present")
def test_cli_on_pilot_writes_only_utility(tmp_path):
    from analysis.run_analysis import build_parser, run
    args = build_parser().parse_args(["--traces", *[str(p) for p in KAGGLE.glob("pilot-*-v1/runs")],
                                      "--out", str(tmp_path)])
    m = run(args)
    assert not m["synthetic"]
    assert set(m["written"]) == {"episodes_tidy", "T0_episode_accounting", "T2_utility_no_attack"}   # no attack tables
