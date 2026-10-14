"""paper/RESULTS_SUMMARY.md is generated from the committed fast-track CSVs and is up to date."""
from pathlib import Path

import pandas as pd
import pytest

from analysis.write_results_summary import build

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "paper" / "results" / "fasttrack_v1_20261007"


@pytest.mark.skipif(not (ROOT / "primary" / "MANIFEST.json").exists(), reason="fast-track results not present")
def test_summary_matches_csvs_and_is_current():
    txt = build(ROOT, REPO)
    assert txt == (REPO / "paper" / "RESULTS_SUMMARY.md").read_text(encoding="utf-8"), "rerun write_results_summary"
    g = pd.read_csv(ROOT / "primary" / "F1_asr_grid.csv")
    for _, r in g.iterrows():   # every headline cell appears with its CI
        assert f"{100 * r.asr:.1f} [{100 * r.asr_lo:.1f}, {100 * r.asr_hi:.1f}]" in txt
    assert "NOT RUN" in txt and "POST HOC" in txt
