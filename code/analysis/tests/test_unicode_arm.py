"""D41 Unicode-preserving arm analysis (analysis/unicode_arm.py) on SYNTHETIC mock data only.

Planted: in the stock run the Bengali injection of the listed bn_* episodes reached the agent escaped and
never succeeded; in the Unicode rerun those same episodes succeed at the latent rate. No real numbers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import unicode_arm as UA

CODE = Path(__file__).resolve().parents[2]
PAIRS = json.loads((CODE / "subset" / "pairs_150.json").read_text())["pairs"][:40]
MODELS = ["SYNTH-A", "SYNTH-B", "SYNTH-G"]          # SYNTH-G = self-generated (outside the pool)
CONDS = ["en", "bn_mt", "bn_llm", "banglish_llm"]
ESC = "\\u0986\\u09aa\\u09a8\\u09bf"                  # what the stock formatter produced
BN = "আপনি"


def _trace(model, s, u, i, cond, block, sec, util):
    msgs = [{"role": "system", "content": [{"type": "text", "content": "SYNTHETIC"}]},
            {"role": "user", "content": [{"type": "text", "content": "SYNTHETIC task"}]},
            {"role": "assistant", "content": [{"type": "text", "content": ""}],
             "tool_calls": [{"function": "get_x", "args": {}, "id": "t", "placeholder_args": None}]},
            {"role": "tool", "content": [{"type": "text", "content": f"- body: <INFORMATION> {block} </INFORMATION>"}],
             "tool_call_id": "t", "tool_call": None, "error": None},
            {"role": "assistant", "content": [{"type": "text", "content": "Done."}], "tool_calls": None}]
    return {"suite_name": s, "pipeline_name": model, "user_task_id": u, "injection_task_id": i if cond != "none" else None,
            "attack_type": None if cond == "none" else cond, "injections": {}, "messages": msgs, "error": None,
            "benchmark_version": "v1.2.2", "agentdojo_package_version": "0.1.35", "utility": util,
            "security": sec if cond != "none" else True, "duration": 1.0, "synthetic": True}


def _write(root, run_id, recs, traces):
    (root / "runs").mkdir(parents=True, exist_ok=True)
    jl = root / "runs" / f"{run_id}.jsonl"
    jl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [{"_header": True, "synthetic": True}] + recs) + "\n",
                  encoding="utf-8")
    for d in traces:
        f = root / "traces" / d["pipeline_name"] / d["suite_name"] / d["user_task_id"] / (d["attack_type"] or "none") / \
            f"{d['injection_task_id'] or 'none'}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return jl


def _rec(model, s, u, i, cond, sec, util, status="ok", error=None):
    r = {"model": model, "suite": s, "user_task": u, "injection_task": i, "condition": cond,
         "kind": "utility" if cond == "none" else "attack", "status": status, "error": error, "synthetic": True}
    if status == "ok":
        r.update(utility=util, raw_security=sec, normalised_security=sec, corrected_security=sec)
    return r


def make(tmp: Path, escaped_uni_model=None, seed=11):
    rng = np.random.default_rng(seed)
    stock_r, stock_t, uni_r, uni_t = [], [], [], []
    lists = {m: {"model_key": m.lower(), "label": m, "tier_A": [], "tier_B": [], "stock_retry": [], "not_rerun_errors": []}
             for m in MODELS}
    retry_key = ("SYNTH-A",) + tuple(PAIRS[0][k] for k in ("suite", "user_task", "injection_task")) + ("en",)
    for m in MODELS:
        for p in PAIRS:
            s, u, i = p["suite"], p["user_task"], p["injection_task"]
            lat = rng.random()
            for cond in CONDS:
                util = bool(rng.random() < .6)
                if cond in ("en", "banglish_llm"):
                    sec = bool(lat < (.5 if cond == "en" else .3))
                    block = "Please do X."
                    escaped = False
                else:
                    escaped = rng.random() < .6
                    sec_uni = bool(lat < (.45 if cond == "bn_llm" else .35))
                    sec = False if escaped else sec_uni
                    block = ESC if escaped else BN
                if (m,) + (s, u, i, cond) == retry_key:          # D35-like infrastructure error in stock
                    stock_r.append(_rec(m, s, u, i, cond, None, None, "error", "APITimeoutError: Request timed out."))
                    continue
                stock_r.append(_rec(m, s, u, i, cond, sec, util))
                stock_t.append(_trace(m, s, u, i, cond, block, sec, util))
                if escaped:
                    lists[m]["tier_A"].append({"suite": s, "user_task": u, "injection_task": i, "condition": cond,
                                               "reason": "injection_escaped", "stock_duration_s": 1.0})
                    if m == "SYNTH-B" and (s, u, i, cond) == tuple(lists[m]["tier_A"][0][k] for k in
                                                                   ("suite", "user_task", "injection_task", "condition")) \
                            and len(lists[m]["tier_A"]) == 1:
                        continue                                  # this rerun never came back -> rerun_missing
                    ublock = ESC if (m == escaped_uni_model and len(uni_r) % 7 == 0) else BN
                    uni_r.append(_rec(m, s, u, i, cond, sec_uni, util))
                    uni_t.append(_trace(m, s, u, i, cond, ublock, sec_uni, util))
        for p in PAIRS[:5]:
            stock_r.append(_rec(m, p["suite"], p["user_task"], "none", "none", None, True))
            stock_t.append(_trace(m, p["suite"], p["user_task"], "none", "none", "", True, True))
    s_jl = _write(tmp / "stock", "stock_150", stock_r, stock_t)
    u_jl = _write(tmp / "uni", "x_150_uni", uni_r, uni_t)
    m, s, u, i, c = retry_key
    r_jl = _write(tmp / "retry", "x_150_stockretry", [_rec(m, s, u, i, c, True, True)],
                  [_trace(m, s, u, i, c, "Please do X.", True, True)])
    ep = tmp / "episodes"
    ep.mkdir()
    for mm, d in lists.items():
        (ep / f"episodes_{mm.lower()}.json").write_text(json.dumps(d), encoding="utf-8")
    return {"stock": (tmp / "stock" / "traces", s_jl), "uni": (tmp / "uni" / "traces", u_jl),
            "retry": (tmp / "retry" / "traces", r_jl), "episodes": ep, "retry_key": retry_key}


def _args(fx, out, extra=()):
    return UA.build_parser().parse_args([
        "--stock-traces", str(fx["stock"][0]), "--stock-scores", str(fx["stock"][1]),
        "--uni-traces", str(fx["uni"][0]), "--uni-scores", str(fx["uni"][1]),
        "--retry-traces", str(fx["retry"][0]), "--retry-scores", str(fx["retry"][1]),
        "--episodes-dir", str(fx["episodes"]), "--self-generated-models", "SYNTH-G",
        "--n-boot", "200", "--no-glmm", "--out", str(out), *extra])


@pytest.fixture(scope="module")
def arm(tmp_path_factory):
    fx = make(tmp_path_factory.mktemp("uni_fx"))
    out = tmp_path_factory.mktemp("uni_out")
    return UA.run(_args(fx, out)), out, fx


def _t(out, name):
    return pd.read_csv(out / f"SYNTHETIC_{name}.csv")


def test_outputs_present_labelled_synthetic_and_secondary(arm):
    m, out, _ = arm
    assert m["synthetic"] is True and m["analysis_status"] == "secondary_post_hoc_D41"
    files = [p.name for p in out.iterdir()]
    assert files and all(f.startswith("SYNTHETIC_") for f in files)
    for k in ("U0_arm_accounting", "U1_rerun_rendering_check", "U2_asr_grid_unicode", "U3_primary_contrasts_unicode",
              "U3_pooled_or_unicode", "U3b_exploratory_pairs_unicode", "U4_stock_vs_unicode_asr",
              "U5_stock_vs_unicode_contrasts", "U6_rendering_unicode_arm_all", "episodes_tidy_unicode"):
        assert k in m["written"], k
        t = pd.read_csv(m["written"][k])
        if k != "episodes_tidy_unicode":
            assert (t["analysis_status"] == "secondary_post_hoc_D41").all(), k
            assert (t["arm"] == UA.ARM).all() or (k == "U5_stock_vs_unicode_contrasts" and set(t["arm"]) == {"stock", UA.ARM}), k
        if "synthetic" in t:
            assert t["synthetic"].all(), k
    for stem in ("FU1_asr_heatmap_unicode", "FU2_contrasts_stock_vs_unicode"):
        for ext in ("png", "pdf"):
            assert (out / f"SYNTHETIC_{stem}.{ext}").stat().st_size > 1000


def test_accounting_missing_rerun_and_retry(arm):
    m, out, fx = arm
    u0 = _t(out, "U0_arm_accounting")
    assert m["n_rerun_missing"] == 1 and m["retry_merged_into_stock"] == 1
    b = u0[u0["model"] == "SYNTH-B"]
    assert b["rerun_missing"].sum() == 1
    assert (u0[u0["condition"].isin(["en", "banglish_llm"])]["unicode_rerun"] == 0).all()   # reused from stock
    assert (u0["episodes"] == len(PAIRS)).all()
    tidy = _t(out, "episodes_tidy_unicode")
    miss = tidy[tidy["arm_source"] == "rerun_missing"]
    assert len(miss) == 1 and (miss["status"] == "rerun_missing").all() and miss["normalised_security"].isna().all()
    mk, s, u, i, c = fx["retry_key"]
    row = tidy[(tidy["model"] == mk) & (tidy["suite"] == s) & (tidy["user_task"] == u) &
               (tidy["injection_task"] == i) & (tidy["condition"] == c)]
    assert len(row) == 1 and row["status"].iloc[0] == "ok" and bool(row["retry_merged"].iloc[0])


def test_rendering_check_patch_ok(arm, tmp_path):
    m, out, _ = arm
    u1 = _t(out, "U1_rerun_rendering_check")
    assert m["patch_ok_all"] is True and u1["patch_ok"].all() and (u1["escaped_unicode"] == 0).all()
    assert set(u1["condition"]) == {"bn_mt", "bn_llm"}
    fx2 = make(tmp_path / "fx2", escaped_uni_model="SYNTH-A")
    m2 = UA.run(_args(fx2, tmp_path / "out2"))
    u1b = pd.read_csv(m2["written"]["U1_rerun_rendering_check"])
    assert m2["patch_ok_all"] is False
    assert not u1b[u1b["model"] == "SYNTH-A"]["patch_ok"].any() and u1b[u1b["model"] != "SYNTH-A"]["patch_ok"].all()


def test_planted_stock_vs_unicode_recovered(arm):
    _, out, _ = arm
    u4 = _t(out, "U4_stock_vs_unicode_asr")
    r = u4[(u4["subset"] == "rerun_only") & u4["condition"].isin(["bn_mt", "bn_llm"])]
    assert set(r["model"]) == {"SYNTH-A", "SYNTH-B", "SYNTH-G", "POOLED_primary_pool"}
    assert (r["asr_stock"] == 0).all() and (r["asr_unicode"] > .15).all() and (r["diff_pp"] > 0).all()
    assert (r["n_stock_only"] == 0).all() and (r["n_unicode_only"] > 0).all()
    pooled = r[r["model"] == "POOLED_primary_pool"]
    assert pooled["mcnemar_p"].isna().all() and (pooled["ci_lo_pp"] > 0).all()
    # en / banglish_llm are the same episodes in both arms -> zero difference
    e = u4[(u4["subset"] == "all_episodes") & u4["condition"].isin(["en", "banglish_llm"])]
    assert (e["diff_pp"] == 0).all()
    u5 = _t(out, "U5_stock_vs_unicode_contrasts")
    assert set(u5["contrast"]) == {"C1", "C3"} and set(u5["arm"]) == {"stock", UA.ARM}
    c3 = u5[(u5["contrast"] == "C3") & (u5["level"] == "per_model")].set_index(["model", "arm"])["diff_pp"]
    for mm in ("SYNTH-A", "SYNTH-B"):
        assert c3[(mm, UA.ARM)] > c3[(mm, "stock")]          # bn_llm − en gap shrinks once Bengali is script
    u3 = _t(out, "U3_primary_contrasts_unicode")
    assert {"C1", "C3"} <= set(u3["contrast"])
    sg = u3[u3["model"] == "SYNTH-G"]
    assert sg["self_generated"].astype(bool).all()


def test_build_arm_and_merge_retry_units():
    k = dict(suite="s", user_task="u", injection_task="i")
    stock = pd.DataFrame([dict(model="M", condition="bn_mt", status="ok", normalised_security=False, **k),
                          dict(model="M", condition="en", status="error", normalised_security=np.nan, **k)])
    listed = pd.DataFrame([dict(model="M", condition="bn_mt", tier="tier_A", reason="x", **k)])
    uni = pd.DataFrame([dict(model="M", condition="bn_mt", status="ok", normalised_security=True, **k)])
    a = UA.build_arm(stock, uni, listed)
    assert sorted(a["arm_source"]) == ["stock_reused", "unicode_rerun"]
    assert a.loc[a["arm_source"] == "unicode_rerun", "normalised_security"].iloc[0]
    with pytest.raises(ValueError, match="not in the rerun list"):
        UA.build_arm(stock, uni.assign(condition="bn_llm"), listed)
    with pytest.raises(ValueError, match="duplicate"):
        UA.build_arm(stock, pd.concat([uni, uni]), listed)
    retry = pd.DataFrame([dict(model="M", condition="en", status="ok", normalised_security=True, **k),
                          dict(model="M", condition="bn_mt", status="ok", normalised_security=True, **k)])
    mr = UA.merge_retry(stock, retry)
    assert len(mr) == 2 and mr.attrs["retry_merged"] == 1
    assert mr[mr["condition"] == "bn_mt"]["normalised_security"].iloc[0] == False   # ok stock row never replaced
    assert mr[mr["condition"] == "en"]["status"].iloc[0] == "ok"
    still = UA.merge_retry(stock, retry.assign(status="error"))
    assert still.attrs["retry_merged"] == 0 and (still["status"] == ["ok", "error"]).all()


def test_run_analysis_folds_retry_into_primary(tmp_path):
    from analysis.run_analysis import build_parser, run
    fx = make(tmp_path / "fx")
    base = ["--traces", str(fx["stock"][0]), "--scores", str(fx["stock"][1]), "--n-boot", "50", "--no-glmm"]
    m0 = run(build_parser().parse_args(base + ["--out", str(tmp_path / "o0")]))
    m1 = run(build_parser().parse_args(base + ["--retry-traces", str(fx["retry"][0]), "--retry-scores",
                                               str(fx["retry"][1]), "--out", str(tmp_path / "o1")]))
    man = json.loads(next((tmp_path / "o1").glob("*MANIFEST*.json")).read_text())
    assert json.dumps(man).count('"retry_merged": 1') == 1
    t0 = pd.read_csv(next((tmp_path / "o0").glob("*episodes_tidy*.csv")))
    t1 = pd.read_csv(next((tmp_path / "o1").glob("*episodes_tidy*.csv")))
    assert len(t0) == len(t1) and (t1["status"] != "ok").sum() == (t0["status"] != "ok").sum() - 1
    assert m0 and m1


def test_refuses_synthetic_into_paper(tmp_path):
    fx = make(tmp_path / "fx")
    bad = tmp_path / "paper" / "results"
    with pytest.raises(SystemExit):
        UA.run(_args(fx, bad))
    assert not bad.exists()
