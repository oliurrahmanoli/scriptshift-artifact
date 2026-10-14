"""End-to-end test on the SYNTHETIC fixture (planted effects). Outputs go to a pytest tmp dir only."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.run_analysis import build_parser, run

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
spec = importlib.util.spec_from_file_location("make_fixture", HERE / "fixtures_synthetic" / "make_fixture.py")
make_fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_fixture)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    fx = make_fixture.make(tmp_path_factory.mktemp("synth_fixture"))
    out = tmp_path_factory.mktemp("synth_out")
    args = build_parser().parse_args([
        "--traces", str(fx / "runs"), "--scores", str(fx / "scores" / "*.jsonl"),
        "--judge", str(fx / "judge.jsonl"), "--probe", str(fx / "probe.jsonl"),
        "--detectors", str(fx / "detectors.csv"), "--audit", str(fx / "audit.csv"),
        "--ratings", str(fx / "ratings.csv"), "--payloads", str(CODE / "data" / "payloads.json"),
        "--pairs", str(CODE / "subset" / "pairs.json"),
        "--flip-pair", "repeat:SYNTH-model-A:SYNTH-model-A-rep1",
        "--exclude-from-main", "SYNTH-model-A-rep1", "--n-boot", "300", "--out", str(out)])
    return run(args), out


def _t(out, name):
    return pd.read_csv(out / f"SYNTHETIC_{name}.csv")


def test_everything_labelled_synthetic(synth):
    m, out = synth
    assert m["synthetic"] is True
    files = [p.name for p in out.iterdir()]
    assert files and all(f.startswith("SYNTHETIC_") for f in files)
    for f in out.glob("*.csv"):
        t = pd.read_csv(f)
        if "synthetic" in t and f.name != "SYNTHETIC_T1_dataset.csv":
            assert t["synthetic"].all(), f.name
    for stem in ("F1_asr_heatmap", "F2_contrast_cis"):
        for ext in ("png", "pdf"):
            assert (out / f"SYNTHETIC_{stem}.{ext}").stat().st_size > 1000


def test_all_tables_present(synth):
    m, _ = synth
    for k in ("T1_dataset", "T2_utility_no_attack", "F1_asr_grid", "T3_primary_contrasts", "T3_pooled_or",
              "T3_vs_en_all", "T4_undercount", "T4_banking_precedence", "T4_judge_audit", "T5_subset_vs_full",
              "T5_suite_reweighted", "T6_comprehension", "T6_pooled_with_comprehension", "T7_detectors",
              "T8_defences", "T9_register", "T10a_nllb_vs_google", "T10b_flip_rates", "T10c_output_language"):
        assert k in m["written"], k


def test_planted_primary_effects_recovered(synth):
    _, out = synth
    t3 = _t(out, "T3_primary_contrasts")
    pooled = t3[t3["level"] == "pooled_bootstrap"].set_index("contrast")
    for c in ("C1", "C3"):                       # planted bn_native +15 pp
        assert pooled.loc[c, "diff_pp"] > 5 and pooled.loc[c, "ci_lo_pp"] > 0
    assert abs(pooled.loc["C2", "diff_pp"]) < 5   # planted null
    assert bool(pooled.loc["C2", "equivalent"])   # TOST ±5 pp
    per = t3[t3["level"] == "per_model"]
    assert per["p_holm"].ge(per["mcnemar_p"]).all()
    assert per[per["contrast"] == "C1"]["holm_reject"].all()
    gee = _t(out, "T3_pooled_or")
    g = gee[gee["model_fit"] == "GEE_exchangeable"].set_index("contrast")
    assert g.loc["C1", "or"] > 1 and g.loc["C1", "p_holm"] < 0.05
    assert g.loc["C2", "p_holm"] > 0.05
    vs = _t(out, "T3_vs_en_all")
    assert {"p_holm", "p_bh"} <= set(vs.columns)


def test_undercount_banking_comprehension_detectors(synth):
    _, out = synth
    t4 = _t(out, "T4_undercount")
    bangla = t4[t4["condition"].str.startswith(("bn", "banglish", "codemix"))]
    assert (bangla["undercount_pp_norm_minus_raw"].mean()) > -5
    assert (t4["judge_aug_security"] >= t4["normalised_security"] - 1e-12).all()
    bk = _t(out, "T4_banking_precedence")
    assert (bk["corrected_asr"] <= bk["raw_asr"] + 1e-12).all()
    t6 = _t(out, "T6_comprehension")
    assert (t6["asr_given_comp_both"].mean()) > (t6["asr_given_notcomp_both"].mean())
    t7 = _t(out, "T7_detectors").set_index(["detector", "condition"])
    assert t7.loc[("SYNTH-D1", "en"), "tpr_default"] > t7.loc[("SYNTH-D1", "bn_native"), "tpr_default"]


def test_robustness_and_defences(synth):
    _, out = synth
    fl = _t(out, "T10b_flip_rates")
    assert (fl["flip_rate"] < 0.15).all()        # planted ~5 % redraw
    t8 = _t(out, "T8_defences")
    tf = t8[t8["defense"] == "tool_filter"].set_index("condition")["asr"]
    nd = t8[(t8["defense"] == "none") & (t8["model"] == "SYNTH-model-A")].set_index("condition")["asr"]
    assert (tf < nd.reindex(tf.index)).all()     # planted x0.3
    t5 = _t(out, "T5_subset_vs_full")
    assert set(t5["scope"]) == {"subset_300", "all_pairs"}


def test_refuses_synthetic_into_paper(tmp_path):
    fx = make_fixture.make(tmp_path / "fx")
    bad = tmp_path / "paper" / "results"
    args = build_parser().parse_args(["--scores", str(fx / "scores" / "*.jsonl"), "--n-boot", "50",
                                      "--no-glmm", "--out", str(bad)])
    with pytest.raises(SystemExit):
        run(args)
    assert not bad.exists()
