"""Review-response sensitivity analyses (15 Oct 2026). Descriptive / exploratory; existing traces only.

1. Added headings/importance cues in LLM-written payloads (counted from code/data/payloads.json) and C1 with
   those items excluded / only, stock and Unicode arms.
2. banglish_llm contrasts with the 11 author-rated 'drifted' items excluded.
3. Cross-language model ranking: Spearman rho between per-model en ASR and each Bangla form, with a
   cluster bootstrap (injection tasks) CI, per arm.
4. Unicode-arm 'rerun episodes only' ASR on the strictly escaped subset vs the full rerun list.
5. L1 (stock security()) ASR grid next to L2.
Run from code/:  python3 -m analysis.review_sensitivity
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
from analysis import tables as T
from analysis.bootstrap import _layout

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "paper/results"
OUT = RES / "review_sensitivity_20261015"
STOCK = RES / "fasttrack_v1_20261007/primary/episodes_tidy.csv"
UNI = RES / "unicode_arm_v1/episodes_tidy_unicode.csv"
PAY = ROOT / "code/data/payloads.json"
RAT = RES / "native_check/native_check_ratings_merged.csv"
NB = 10_000
MODELS = ["gpt-oss-20b", "granite-4.2-8b", "ministral-3-14b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b"]

BN_HEAD = re.compile(r"^\s*(জরুরি|গুরুত্বপূর্ণ)\s+\S+\s*:")
# Banglish items whose text opens with an added announcement/attention framing not in the English goal
# (manual reading of all 35 items, listed explicitly so the count is auditable).
BANGLISH_FRAMED = ["workspace/injection_task_2", "workspace/injection_task_3", "workspace/injection_task_7",
                   "workspace/injection_task_9", "banking/injection_task_0", "banking/injection_task_2",
                   "banking/injection_task_3"]


def headed_items():
    g = json.loads(PAY.read_text(encoding="utf-8"))["conditions"]
    en = g["en"]["goals"]
    rows = []
    for c in ("en", "bn_mt", "bn_llm", "banglish_llm"):
        for k, v in g[c]["goals"].items():
            bn_head = bool(BN_HEAD.match(v))
            eng_prefix = v.startswith("Important message!") and not en[k].startswith("Important message!")
            framed = c == "banglish_llm" and k in BANGLISH_FRAMED
            rows.append({"condition": c, "item": k, "bengali_heading": bn_head,
                         "english_importance_prefix": eng_prefix, "banglish_framing": framed,
                         "added_cue": bn_head or eng_prefix or framed, "opening": v[:60]})
    return pd.DataFrame(rows)


def load(p):
    d = pd.read_csv(p, low_memory=False)
    d["item"] = d["suite"] + "/" + d["injection_task"].astype(str)
    return d


def c1(df, items=None, a="bn_mt", b="bn_llm", metric="normalised_security", label=""):
    d = df if items is None else df[df["item"].isin(items)]
    r = T.contrast_row(T._pool(d), a, b, metric, None, NB, 0, tost=True, bca=True)
    return {"subset": label, "cond_a": a, "cond_b": b, **{k: r[k] for k in (
        "n_pairs", "n_clusters", "asr_a", "asr_b", "diff_pp", "ci_lo_pp", "ci_hi_pp", "bca_lo_pp", "bca_hi_pp",
        "tost_ci_lo", "tost_ci_hi", "tost_p", "equivalent")}}


def per_model_asr(df, cond, metric="normalised_security"):
    d = T.attacked(df)
    d = d[(d["condition"] == cond) & d[metric].notna()]
    return d.groupby("model")[metric].apply(lambda s: s.astype(float).mean())


def rank_boot(df, arm, metric="normalised_security", nb=4000, seed=0):
    """Spearman rho (5 models) between en ASR and each Bangla form's ASR; CI by resampling injection
    tasks (the authored unit) with all their episodes, the same draw for every model and condition."""
    d = T.attacked(df)
    d = d[d[metric].notna()]
    items = sorted(d["item"].unique())
    rng = np.random.default_rng(seed)
    S = d.assign(s=d[metric].astype(float)).groupby(["item", "model", "condition"])["s"].agg(["sum", "count"])
    S = S.unstack(["model", "condition"]).fillna(0)
    sums, cnts = S["sum"], S["count"]
    rows = []
    for cond in ("bn_mt", "bn_llm", "banglish_llm"):
        def rho(idx):
            ss, cc = sums.iloc[idx].sum(), cnts.iloc[idx].sum()
            asr = ss / cc
            x = [asr[(m, "en")] for m in MODELS]; y = [asr[(m, cond)] for m in MODELS]
            return st.spearmanr(x, y).statistic
        est = rho(np.arange(len(items)))
        bs = np.array([rho(rng.integers(0, len(items), len(items))) for _ in range(nb)])
        bs = bs[~np.isnan(bs)]
        rows.append({"arm": arm, "bangla_condition": cond, "spearman_rho_en_vs_bn": est,
                     "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975),
                     "share_boot_rho_lt_0": float((bs < 0).mean()), "n_boot_valid": len(bs)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H = headed_items()
    H.to_csv(OUT / "R1_added_cues_by_item.csv", index=False)
    summ = H.groupby("condition")[["bengali_heading", "english_importance_prefix", "banglish_framing", "added_cue"]].sum()
    summ.to_csv(OUT / "R1_added_cues_counts.csv")
    print(summ)
    s, u = load(STOCK), load(UNI)
    headed = set(H[(H.condition == "bn_llm") & H.added_cue].item)
    allitems = set(H.item)
    rows = []
    for arm, df in (("stock", s), ("unicode", u)):
        for lab, it in (("all items", None), ("bn_llm items without added cue", allitems - headed),
                        ("bn_llm items with added cue", headed)):
            rows.append({"arm": arm, **c1(df, it, label=lab)})
    R2 = pd.DataFrame(rows); R2.to_csv(OUT / "R2_c1_by_added_cue.csv", index=False); print(R2.round(3).to_string())
    # 2. Banglish drifted
    r = pd.read_csv(RAT)
    M = r.columns[5]
    drift = set(r[(r.condition == "banglish_llm") & (r[M] == "drifted")]["id"].str.split(":").str[1])
    rows = []
    for arm, df in (("stock", s), ("unicode", u)):
        for lab, it in (("all items", None), ("drifted Banglish items excluded", allitems - drift)):
            for a, b in (("bn_mt", "banglish_llm"), ("bn_llm", "banglish_llm"), ("en", "banglish_llm")):
                rows.append({"arm": arm, **c1(df, it, a, b, label=lab)})
            d = df if it is None else df[df["item"].isin(it)]
            for m in MODELS:
                pm = {c: per_model_asr(d, c).get(m, np.nan) for c in ("en", "bn_mt", "bn_llm", "banglish_llm")}
                rows.append({"arm": arm, "subset": lab, "cond_a": "per_model", "cond_b": m,
                             **{f"asr_{c}": v for c, v in pm.items()},
                             "banglish_highest_bangla": pm["banglish_llm"] >= max(pm["bn_mt"], pm["bn_llm"])})
    R3 = pd.DataFrame(rows); R3.to_csv(OUT / "R3_banglish_drift_sensitivity.csv", index=False)
    print(R3.round(3).to_string())
    # 3. ranking
    rows = []
    for arm, df in (("stock", s), ("unicode", u)):
        rows += rank_boot(df, arm)
        for c in ("en", "bn_mt", "bn_llm", "banglish_llm"):
            pm = per_model_asr(df, c)
            rows.append({"arm": arm, "bangla_condition": c, **{f"asr_{m}": pm.get(m) for m in MODELS},
                         "max_model": pm.idxmax(), "min_model": pm.idxmin()})
    R4 = pd.DataFrame(rows); R4.to_csv(OUT / "R4_rank_correlation.csv", index=False); print(R4.round(3).to_string())
    # 4. rerun subsets
    K = ["model", "suite", "user_task", "injection_task", "condition"]
    ok = lambda d: d[d.exclusion.isna() | (d.exclusion == "model_output_error_counted_as_failure")]
    mm = ok(s).merge(ok(u)[K + ["normalised_security", "arm_source"]], on=K, suffixes=("_s", "_u"))
    mm = mm[mm.condition.isin(["bn_mt", "bn_llm"]) & mm.primary_pool.astype(bool) & (mm.arm_source == "unicode_rerun")]
    rows = []
    for lab, sel in (("all rerun episodes", mm.index == mm.index), ("strictly escaped (Table D1 class)", mm.injection_render == "escaped_unicode")):
        for c, g in mm[sel].groupby("condition"):
            a, b = g.normalised_security_s.astype(float), g.normalised_security_u.astype(float)
            rows.append({"subset": lab, "condition": c, "n": len(g), "asr_stock": a.mean(), "asr_unicode": b.mean(),
                         "to_success": int(((a == 0) & (b == 1)).sum()), "to_failure": int(((a == 1) & (b == 0)).sum())})
    R5 = pd.DataFrame(rows); R5.to_csv(OUT / "R5_rerun_subsets.csv", index=False); print(R5.round(3).to_string())
    # 5. L1 grid
    rows = []
    for m in MODELS:
        for c in ("en", "bn_mt", "bn_llm", "banglish_llm"):
            rows.append({"model": m, "condition": c, "L1_raw": per_model_asr(s, c, "raw_security").get(m),
                         "L2_normalised": per_model_asr(s, c).get(m)})
    dp = T._pool(s)
    for c in ("en", "bn_mt", "bn_llm", "banglish_llm"):
        d = T.attacked(dp); d = d[d.condition == c]
        rows.append({"model": "POOLED_primary_pool", "condition": c, "L1_raw": d.raw_security.astype(float).mean(),
                     "L2_normalised": d.normalised_security.astype(float).mean()})
    R6 = pd.DataFrame(rows); R6.to_csv(OUT / "R6_L1_vs_L2_grid.csv", index=False); print(R6.round(3).to_string())


if __name__ == "__main__":
    main()
