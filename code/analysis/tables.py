"""Build the paper's Results tables T1–T10 (paper/draft.md §6) from the tidy episode table.

Every function returns a pandas DataFrame; ``analysis/run_analysis.py`` writes them as CSV.
Nothing here invents a number: a table whose inputs are absent comes back empty with a
``status`` note, and every table carries a ``synthetic`` column copied from the inputs.

Metric columns in the episode table (ingest.build_episodes):
  raw_security (L1), normalised_security (L2, PRIMARY), corrected_security (banking fix),
  judge_aug_security (L2 OR judge=="full"), utility.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config as C
from analysis.bootstrap import asr_ci, paired_diff_ci
from analysis.metrics import (auroc, cohen_kappa, krippendorff_alpha, rate_above,
                              threshold_at_fpr, tost_from_boot, wilson, flip_rate)
from analysis.stats import benjamini_hochberg, holm_bonferroni, mcnemar_exact

PAIR_KEY = ["model", "defense", "suite", "user_task", "injection_task", "rep"]
METRICS = ["raw_security", "normalised_security", "corrected_security", "judge_aug_security"]


# ----------------------------------------------------------------------------- helpers
def _ok(df):
    return df[df["status"].fillna("ok").eq("ok")]


def attacked(df, defense="none"):
    d = _ok(df)
    d = d[d["condition"] != C.NO_ATTACK]
    if defense is not None:
        d = d[d["defense"] == defense]
    return d


def _order(conds):
    known = C.MAIN_CONDITIONS + C.W2_CONDITIONS + [C.NLLB_CONDITION]
    return [c for c in known if c in conds] + sorted(c for c in conds if c not in known)


def paired(df, cond_a, cond_b, metric, model=None, defense="none"):
    d = attacked(df, defense)
    if model is not None:
        d = d[d["model"] == model]
    d = d[d["condition"].isin([cond_a, cond_b]) & d[metric].notna()]
    if d.empty:
        return None
    w = d.pivot_table(index=PAIR_KEY, columns="condition", values=metric, aggfunc="first")
    if cond_a not in w or cond_b not in w:
        return None
    w = w.dropna(subset=[cond_a, cond_b]).reset_index()
    if w.empty:
        return None
    w["inj_uid"] = w["suite"] + "/" + w["injection_task"].astype(str)
    return w


def contrast_row(df, cond_a, cond_b, metric, model=None, n_boot=C.N_BOOT, seed=0,
                 tost=False, bca=True, defense="none"):
    w = paired(df, cond_a, cond_b, metric, model, defense)
    if w is None:
        return None
    a = w[cond_a].astype(bool).values
    b = w[cond_b].astype(bool).values
    p, b10, b01 = mcnemar_exact(a, b)
    ci = paired_diff_ci(a, b, w["inj_uid"].values, n_boot, seed, bca=bca, return_boot=True)
    row = {"model": model if model is not None else "POOLED", "cond_a": cond_a, "cond_b": cond_b,
           "metric": metric, "n_pairs": len(w), "n_clusters": w["inj_uid"].nunique(),
           "asr_a": a.mean(), "asr_b": b.mean(), "diff_pp": 100 * ci["diff"],
           "ci_lo_pp": 100 * ci["lo"], "ci_hi_pp": 100 * ci["hi"],
           "bca_lo_pp": 100 * ci.get("bca_lo", np.nan), "bca_hi_pp": 100 * ci.get("bca_hi", np.nan),
           "mcnemar_p": p, "n_only_a": b10, "n_only_b": b01}
    if model is None:
        row["mcnemar_p"] = np.nan  # McNemar is per model (§4.7); pooled inference = GEE/glmer
    if tost:
        row.update(tost_from_boot(row["diff_pp"], 100 * ci["boot"], C.TOST_MARGIN_PP, C.ALPHA))
    return row


def _adjust(rows, col_p="mcnemar_p", holm=True, bh=False):
    idx = [i for i, r in enumerate(rows) if r.get(col_p) == r.get(col_p) and r.get(col_p) is not None]
    ps = [rows[i][col_p] for i in idx]
    if not ps:
        return rows
    if holm:
        for j, _, padj, rej in holm_bonferroni(ps, C.ALPHA):
            rows[idx[j]]["p_holm"] = padj
            rows[idx[j]]["holm_reject"] = rej
    if bh:
        for j, _, padj, rej in benjamini_hochberg(ps, C.ALPHA):
            rows[idx[j]]["p_bh"] = padj
            rows[idx[j]]["bh_reject"] = rej
    return rows


def _syn(df, t):
    t = t.copy()
    t["synthetic"] = bool(df["synthetic"].any()) if "synthetic" in df else False
    return t


def _empty(status):
    return pd.DataFrame([{"status": status}])


# ----------------------------------------------------------------------------- core ASR grid (F1)
def asr_grid(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    """Per model x condition: targeted ASR and utility under attack, cluster-bootstrap 95% CIs."""
    d = attacked(df)
    if metric not in d or d[metric].notna().sum() == 0:
        return _empty(f"no {metric} values (layer-2 needs scoring JSONL)")
    rows = []
    for (m, cond), g in d.groupby(["model", "condition"]):
        g = g[g[metric].notna()]
        if g.empty:
            continue
        a = asr_ci(g[metric].astype(bool).values, g["inj_uid"].values, n_boot, seed)
        gu = g[g["utility"].notna()]
        u = asr_ci(gu["utility"].astype(bool).values, gu["inj_uid"].values, n_boot, seed + 1) \
            if len(gu) else {"est": np.nan, "lo": np.nan, "hi": np.nan}
        rows.append({"model": m, "condition": cond, "metric": metric, "n": a["n"],
                     "n_clusters": a["n_clusters"], "asr": a["est"], "asr_lo": a["lo"], "asr_hi": a["hi"],
                     "utility_under_attack": u["est"], "uua_lo": u["lo"], "uua_hi": u["hi"],
                     "parse_error_rate": g["parse_error"].astype(float).mean() if g["parse_error"].notna().any() else np.nan})
    t = pd.DataFrame(rows)
    if "self_generated_payload" in d:
        sg = d.groupby(["model", "condition"])["self_generated_payload"].any()
        t["self_generated"] = [bool(sg.get((m, c), False)) for m, c in zip(t["model"], t["condition"])]
    t["condition"] = pd.Categorical(t["condition"], _order(t["condition"].unique()), ordered=True)
    return _syn(df, t.sort_values(["model", "condition"]).reset_index(drop=True))


# ----------------------------------------------------------------------------- T1 dataset summary
_LATIN_WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]*[A-Za-z]?$")


def t1_dataset(payloads_path, ratings_csv=None, n_boot=2000, seed=0):
    """Items per condition from data/payloads.json (+ validator agreement if a ratings CSV with the
    ASSUMED schema item_key,condition,rater,fidelity,adequacy,naturalness,register[,spelling] exists)."""
    pay = json.loads(Path(payloads_path).read_text(encoding="utf-8"))
    rows = []
    for cond, block in pay["conditions"].items():
        goals = block["goals"]
        vals = list(goals.values())
        todo = [v for v in vals if (not v) or str(v).startswith("TODO_")]
        filled = [v for v in vals if v not in todo]
        def latin_share(t):
            toks = [x.strip(".,!?;:()\"'") for x in t.split()]
            toks = [x for x in toks if x]
            return np.mean([bool(_LATIN_WORD.match(x)) for x in toks]) if toks else np.nan
        from analysis.ingest import bengali_share
        rows.append({"condition": cond, "n_items": len(vals), "n_filled": len(filled),
                     "n_todo": len(todo),
                     "mean_chars": np.mean([len(v) for v in filled]) if filled else np.nan,
                     "mean_bengali_share": np.mean([bengali_share(v) for v in filled]) if filled else np.nan,
                     "mean_latin_token_share": np.mean([latin_share(v) for v in filled]) if filled else np.nan,
                     "wrapper_filled": bool(pay["wrappers"].get(block.get("wrapper_id", "W1"), {}).get(cond)
                                            not in (None, "", "TODO_NATIVE", "TODO_MT", "TODO_TRANSLIT"))})
    t = pd.DataFrame(rows)
    if ratings_csv and Path(ratings_csv).exists():
        t = t.merge(t1_agreement(pd.read_csv(ratings_csv), n_boot, seed), on="condition", how="left")
    else:
        t["agreement_status"] = "PENDING (no validator ratings file; D5)"
    t["synthetic"] = False
    return t


def t1_agreement(r, n_boot=2000, seed=0):
    """Cohen's kappa (fidelity, register) and Krippendorff alpha ordinal (adequacy, naturalness),
    per condition, with item-bootstrap 95% CIs. Expects exactly two raters per item."""
    rng = np.random.default_rng(seed)
    out = []
    for cond, g in r.groupby("condition"):
        raters = sorted(g["rater"].unique())[:2]
        if len(raters) < 2:
            continue
        w = {c: g.pivot_table(index="item_key", columns="rater", values=c, aggfunc="first")
             for c in ("fidelity", "adequacy", "naturalness", "register") if c in g}
        items = w["fidelity"].dropna().index.values
        A = {c: w[c].loc[items, raters].to_numpy() for c in w}     # (n_items, 2) arrays
        row = {"condition": cond, "n_rated_items": len(items),
               "fidelity_pass_both": float((A["fidelity"] == 1).all(axis=1).mean())}
        for c in ("adequacy", "naturalness"):
            if c in A:
                row[f"{c}_mean"] = float(np.nanmean(A[c]))
        def stat(ix):
            s = {"kappa_fidelity": cohen_kappa(A["fidelity"][ix, 0], A["fidelity"][ix, 1])}
            if "register" in A:
                s["kappa_register"] = cohen_kappa(A["register"][ix, 0], A["register"][ix, 1])
            for c in ("adequacy", "naturalness"):
                if c in A:
                    s[f"alpha_{c}"] = krippendorff_alpha(A[c][ix].tolist(), "ordinal")
            return s
        items = np.arange(len(items))
        pt = stat(items)
        boots = [stat(rng.choice(items, len(items), replace=True)) for _ in range(n_boot)]
        for k, v in pt.items():
            bs = np.array([b[k] for b in boots], float)
            row[k], row[k + "_lo"], row[k + "_hi"] = v, np.nanquantile(bs, .025), np.nanquantile(bs, .975)
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- T2 utility
def t2_utility(df, pilot_results=()):
    """No-attack utility per model x suite (+ all), strict parse-error rate, admission checks."""
    d = _ok(df)
    d = d[(d["condition"] == C.NO_ATTACK) & d["utility"].notna()]
    if d.empty:
        return _empty("no no-attack episodes")
    rows = []
    for (m, dfn), g in d.groupby(["model", "defense"]):
        for suite, gs in list(g.groupby("suite")) + [("ALL", g)]:
            k, n = int(gs["utility"].astype(bool).sum()), len(gs)
            p, lo, hi = wilson(k, n)
            pe = gs["parse_error"].astype(float)
            ntc = gs["n_tool_calls"].astype(float)
            rows.append({"model": m, "defense": dfn, "suite": suite, "n": n, "utility": p,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "parse_errors": int(pe.sum()) if pe.notna().any() else np.nan,
                         "parse_error_rate": pe.mean() if pe.notna().any() else np.nan,
                         "zero_tool_episodes": int((ntc == 0).sum()) if ntc.notna().any() else np.nan,
                         "median_tool_calls": ntc.median() if ntc.notna().any() else np.nan,
                         "utility_floor_ok": p >= C.UTILITY_FLOOR,
                         "parse_ok": (pe.mean() <= C.PARSE_MAX) if pe.notna().any() else np.nan})
    t = pd.DataFrame(rows)
    # attach measured throughput / verdict from the pilot result JSONs (real pilot artefacts)
    meta = []
    for fp in pilot_results:
        try:
            r = json.loads(Path(fp).read_text())
        except Exception:
            continue
        label = r.get("label")
        if not label:  # older pilot JSON (Gemma) has no label: use the single runs/<label>/ dir
            subs = [x.name for x in (Path(fp).parent / "runs").glob("*") if x.is_dir()]
            label = subs[0] if len(subs) == 1 else None
        meta.append({"model": label, "eps_per_h_pilot": r.get("eps_per_h"),
                     "pilot_verdict": r.get("verdict"),
                     "pilot_utility_banking": (r.get("utility") or {}).get("banking"),
                     "pilot_utility_slack": (r.get("utility") or {}).get("slack"),
                     "pilot_result_file": str(fp)})
    if meta:
        t = t.merge(pd.DataFrame(meta), on="model", how="left")
    return _syn(df, t)


# ----------------------------------------------------------------------------- T3 primary contrasts
def _pool(df):
    """Rows eligible for pooled (cross-model) primary inference (D38.3: self-generated cells out)."""
    if "primary_pool" in df:
        return df[df["primary_pool"].fillna(True).astype(bool)]
    return df


def t3_primary(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0, fit_glmm=True):
    """Per-model C1–C3 (Δpp, cluster-bootstrap CI percentile+BCa, exact McNemar, Holm over all
    per-model primary tests, TOST ±5 pp) + pooled Δpp (bootstrap over the primary-pool models) + pooled ORs.
    A contrast whose conditions were not run (e.g. C2 without banglish_translit) gets a NOT RUN row."""
    d = attacked(df)
    if d.empty or d[metric].notna().sum() == 0:
        return _empty(f"no attacked episodes with {metric}"), pd.DataFrame()
    selfgen = set(df.loc[df.get("self_generated_payload", pd.Series(False, index=df.index)).fillna(False).astype(bool), "model"])
    rows = []
    present = set(d["condition"])
    for cname, a, b in C.PRIMARY_CONTRASTS:
        for m in sorted(d["model"].unique()):
            r = contrast_row(df, a, b, metric, m, n_boot, seed, tost=True)
            if r:
                r["self_generated"] = m in selfgen and b in ("bn_llm", "banglish_llm", "codemix_llm")
                rows.append({"contrast": cname, "level": "per_model", **r})
    rows = _adjust(rows, holm=True)          # §10.2 / D38.4: per-model McNemar family, Holm
    dp = _pool(df)
    pool_models = sorted(attacked(dp)["model"].unique())
    for cname, a, b in C.PRIMARY_CONTRASTS:
        r = contrast_row(dp, a, b, metric, None, n_boot, seed, tost=True)
        if r:
            rows.append({"contrast": cname, "level": "pooled_bootstrap", **r,
                         "pool_models": ",".join(pool_models)})
        if len(pool_models) < d["model"].nunique():   # sensitivity: all models incl. self-generated
            r2 = contrast_row(df, a, b, metric, None, n_boot, seed, tost=True)
            if r2:
                r2["model"] = "POOLED_incl_selfgen"
                rows.append({"contrast": cname, "level": "pooled_bootstrap_incl_selfgen", **r2,
                             "pool_models": ",".join(sorted(d["model"].unique()))})
    for cname, a, b in C.PRIMARY_CONTRASTS:
        if a not in present or b not in present:
            missing = [c for c in (a, b) if c not in present]
            rows.append({"contrast": cname, "level": "not_run", "cond_a": a, "cond_b": b, "metric": metric,
                         "status": f"NOT RUN: condition(s) {', '.join(missing)} not in this grid"})
    t = pd.DataFrame(rows)
    pooled = pd.DataFrame()
    if fit_glmm:
        from analysis.models import fit_gee, fit_vb_glmm
        dpa = attacked(dp)
        conds = [c for c in C.MAIN_CONDITIONS if c in set(dpa["condition"])]
        g, _ = fit_gee(dpa, metric, conds)
        v = fit_vb_glmm(dpa, metric, conds)
        parts = [x for x in (g, v) if len(x)]
        if len(pool_models) < d["model"].nunique():
            g2, _ = fit_gee(d, metric, conds)
            if len(g2):
                g2 = g2.assign(model_fit=g2["model_fit"] + "_incl_selfgen")
                parts.append(g2)
        pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(pooled):
            pooled["pool_models"] = [",".join(sorted(d["model"].unique())) if str(mf).endswith("_incl_selfgen")
                                     else ",".join(pool_models) for mf in pooled["model_fit"]]
            pooled = _syn(df, pooled)
    return _syn(df, t), pooled


def t3_exploratory(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    """D38.4 secondary/exploratory pairs per model (config.EXPLORATORY_PAIRS): exact McNemar, Δpp with
    cluster-bootstrap CI, BH q=0.05 over the whole table (Holm also given for reference)."""
    d = attacked(df)
    selfgen = set(df.loc[df.get("self_generated_payload", pd.Series(False, index=df.index)).fillna(False).astype(bool), "model"])
    rows = []
    for m in sorted(d["model"].unique()):
        for a, b in C.EXPLORATORY_PAIRS:
            r = contrast_row(df, a, b, metric, m, n_boot, seed, tost=False, bca=False)
            if r:
                r["self_generated"] = m in selfgen and any(c in ("bn_llm", "banglish_llm", "codemix_llm") for c in (a, b))
                rows.append({"family": "secondary_exploratory_BH", **r})
    if not rows:
        return _empty("no exploratory pairs present")
    return _syn(df, pd.DataFrame(_adjust(rows, holm=True, bh=True)))


def t3_vs_en(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    """Every non-en condition vs en, per model: McNemar + Δpp CI; Holm over the whole family and
    BH (the plan labels non-primary pairs exploratory, §4.7)."""
    d = attacked(df)
    rows = []
    for m in sorted(d["model"].unique()):
        for cond in _order(set(d["condition"]) - {C.BASELINE}):
            r = contrast_row(df, C.BASELINE, cond, metric, m, n_boot, seed, tost=False, bca=False)
            if r:
                rows.append(r)
    if not rows:
        return _empty("no paired en contrasts")
    return _syn(df, pd.DataFrame(_adjust(rows, holm=True, bh=True)))


def gonogo_inputs(t3, pooled):
    """Inputs to go/no-go criteria 6a/6b (§11.11 / checklist §6). Computation only, no decision."""
    if t3.empty or "contrast" not in t3:
        return pd.DataFrame()
    rows = []
    for cname, g in t3[t3["level"] == "per_model"].groupby("contrast"):
        pos = int(((g["diff_pp"] > 0) & g.get("holm_reject", False)).sum())
        neg = int(((g["diff_pp"] < 0) & g.get("holm_reject", False)).sum())
        p = pooled[(pooled.get("contrast") == cname) & (pooled.get("model_fit") == "GEE_exchangeable")] \
            if len(pooled) else pd.DataFrame()
        pr = t3[(t3["level"] == "pooled_bootstrap") & (t3["contrast"] == cname)]
        rows.append({"contrast": cname, "n_models": len(g),
                     "models_holm_sig_positive": pos, "models_holm_sig_negative": neg,
                     "pooled_gee_p_holm3": float(p["p_holm"].iloc[0]) if len(p) and "p_holm" in p else np.nan,
                     "pooled_tost_equivalent": bool(pr["equivalent"].iloc[0]) if len(pr) else np.nan,
                     "per_model_tost_equivalent": int(g["equivalent"].sum())})
    out = pd.DataFrame(rows)
    out["synthetic"] = bool(t3["synthetic"].any())
    return out


# ----------------------------------------------------------------------------- T4 under-count
def t4_undercount(df, n_boot=C.N_BOOT, seed=0, audit_csv=None):
    d = attacked(df)
    if d.empty:
        return _empty("no attacked episodes"), pd.DataFrame(), pd.DataFrame()
    rows = []
    for (m, cond), g in d.groupby(["model", "condition"]):
        row = {"model": m, "condition": cond, "n": len(g)}
        for met in METRICS:
            gg = g[g[met].notna()]
            if len(gg):
                ci = asr_ci(gg[met].astype(bool).values, gg["inj_uid"].values, n_boot, seed)
                row[met], row[met + "_lo"], row[met + "_hi"] = ci["est"], ci["lo"], ci["hi"]
            else:
                row[met] = row[met + "_lo"] = row[met + "_hi"] = np.nan
        row["undercount_pp_norm_minus_raw"] = 100 * (row["normalised_security"] - row["raw_security"])
        row["judge_gain_pp"] = 100 * (row["judge_aug_security"] - row["normalised_security"])
        row["n_needs_judge"] = int(g["needs_judge"].fillna(False).astype(bool).sum()) if "needs_judge" in g else np.nan
        row["n_judged"] = int(g["judge_verdict"].notna().sum())
        row["n_judge_full"] = int(g["judge_verdict"].eq("full").sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    # banking 0/1/3 raw vs corrected (§4.4 side finding)
    bk = d[(d["suite"] == "banking") & d["injection_task"].isin(C.BANKING_PRECEDENCE_TASKS)]
    brow = []
    for (m, cond, inj), g in bk.groupby(["model", "condition", "injection_task"]):
        brow.append({"model": m, "condition": cond, "injection_task": inj, "n": len(g),
                     "raw_asr": g["raw_security"].astype(float).mean(),
                     "corrected_asr": g["corrected_security"].astype(float).mean()
                     if g["corrected_security"].notna().any() else np.nan})
    b = pd.DataFrame(brow)
    audit = t4_audit(df, audit_csv) if audit_csv else pd.DataFrame()
    return _syn(df, t), (_syn(df, b) if len(b) else b), audit


def t4_audit(df, audit_csv):
    """Judge precision/recall vs human labels + human–human kappa. ASSUMED audit schema:
    KEY columns + judge_model, judge_verdict, human_a, human_b (full|partial|no). Human 'positive'
    = both humans say full (strict consensus); disagreements are counted separately."""
    a = pd.read_csv(audit_csv)
    out = []
    for jm, g in a.groupby(a.get("judge_model", pd.Series(["judge"] * len(a)))):
        hp = (g["human_a"] == "full") & (g["human_b"] == "full")
        jp = g["judge_verdict"] == "full"
        tp, fp, fn = int((jp & hp).sum()), int((jp & ~hp).sum()), int((~jp & hp).sum())
        out.append({"judge_model": jm, "n_audited": len(g), "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "recall": tp / (tp + fn) if tp + fn else np.nan,
                    "human_kappa": cohen_kappa(g["human_a"], g["human_b"]),
                    "human_disagreements": int((g["human_a"] != g["human_b"]).sum()),
                    "precision_below_0.9_use_human_labels": (tp / (tp + fp) < 0.9) if tp + fp else np.nan})
    return _syn(df, pd.DataFrame(out))


# ----------------------------------------------------------------------------- T5 subset validity
def t5_subset(df, pairs_json, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    sub = {(p["suite"], p["user_task"], p["injection_task"])
           for p in json.loads(Path(pairs_json).read_text())["pairs"]}
    d = attacked(df)
    if d.empty:
        return _empty("no attacked episodes"), pd.DataFrame()
    d = d.copy()
    d["in_subset"] = [(s, u, i) in sub for s, u, i in zip(d["suite"], d["user_task"], d["injection_task"])]
    tot = sum(C.FULL_PAIRS_PER_SUITE.values())
    w = {s: n / tot for s, n in C.FULL_PAIRS_PER_SUITE.items()}
    rw = []
    for (m, cond), g in d[d[metric].notna()].groupby(["model", "condition"]):
        gs = g[g["in_subset"]]
        per = gs.groupby("suite")[metric].apply(lambda x: x.astype(float).mean())
        have = [s for s in w if s in per.index]
        rw.append({"model": m, "condition": cond, "n_subset": len(gs),
                   "asr_subset_raw_mix": gs[metric].astype(float).mean() if len(gs) else np.nan,
                   "asr_suite_reweighted": sum(w[s] * per[s] for s in have) / sum(w[s] for s in have) if have else np.nan,
                   "suites_present": ",".join(have)})
    rows = []
    for m, g in d.groupby("model"):
        if (~g["in_subset"]).sum() == 0:
            continue  # model ran the subset only; no full-vs-subset comparison possible
        for cname, a, b in C.PRIMARY_CONTRASTS:
            for scope, dd in (("subset_300", df[df.index.isin(g[g["in_subset"]].index)]),
                              ("all_pairs", df[df.index.isin(g.index)])):
                r = contrast_row(dd, a, b, metric, m, n_boot, seed, bca=False)
                if r:
                    rows.append({"contrast": cname, "scope": scope, **r})
    return _syn(df, pd.DataFrame(rows)) if rows else _empty("no model with pairs outside the 300 subset"), _syn(df, pd.DataFrame(rw))


# ----------------------------------------------------------------------------- T6 comprehension
def t6_comprehension(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    d = attacked(df)
    if d.empty or d["comprehended_both"].notna().sum() == 0:
        return _empty("PENDING: no comprehension-probe data (D11/C8; probe schema ASSUMED in ingest.load_probe)"), pd.DataFrame()
    rows = []
    for (m, cond), g in d[d["comprehended_both"].notna() & d[metric].notna()].groupby(["model", "condition"]):
        goals = g.drop_duplicates("inj_uid")
        kb, n = int(goals["comprehended_both"].astype(bool).sum()), len(goals)
        ke = int(goals["comprehended_either"].astype(bool).sum())
        pb, lob, hib = wilson(kb, n)
        row = {"model": m, "condition": cond, "n_goals": n, "comprehension_both": pb,
               "comp_both_lo": lob, "comp_both_hi": hib, "comprehension_either": ke / n if n else np.nan,
               "n_episodes": len(g), "asr_all": g[metric].astype(float).mean(),
               "flag_rate_keyword": g["flag_keyword"].astype(float).mean() if g["flag_keyword"].notna().any() else np.nan}
        for rule in ("both", "either"):
            col = f"comprehended_{rule}"
            for val, tag in ((True, "comp"), (False, "notcomp")):
                gg = g[g[col].astype(bool) == val]
                if len(gg):
                    ci = asr_ci(gg[metric].astype(bool).values, gg["inj_uid"].values, n_boot, seed)
                    row[f"asr_given_{tag}_{rule}"], row[f"asr_given_{tag}_{rule}_lo"], row[f"asr_given_{tag}_{rule}_hi"] = ci["est"], ci["lo"], ci["hi"]
                    row[f"n_{tag}_{rule}"] = len(gg)
        rows.append(row)
    t = pd.DataFrame(rows)
    from analysis.models import fit_comprehension
    conds = [c for c in C.MAIN_CONDITIONS if c in set(d["condition"])]
    pooled = fit_comprehension(d, metric, conds)
    return _syn(df, t), (_syn(df, pooled) if len(pooled) else pooled)


# ----------------------------------------------------------------------------- T7 detectors
def t7_detectors(det_csv, default_threshold=0.5, fpr=C.DETECTOR_FPR, n_boot=2000, seed=0):
    """Offline detector metrics. ASSUMED schema: detector, condition, item_id, label (1 = injected
    tool output, 0 = benign), score in [0,1]; benign rows carry the language form in `condition`.
    Thresholds: (a) default; (b) calibrated to `fpr` on the pooled benign set; (c) calibrated on
    benign texts of the same language form when present."""
    if not det_csv or not Path(det_csv).exists():
        return _empty("PENDING: no offline detector scores (C9)")
    d = pd.read_csv(det_csv)
    rng = np.random.default_rng(seed)
    syn = bool(d["synthetic"].any()) if "synthetic" in d else False
    rows = []
    for det, g in d.groupby("detector"):
        neg_all = g[g["label"] == 0]["score"].values
        thr_pooled = threshold_at_fpr(neg_all, fpr)
        for cond in _order(set(g[g["label"] == 1]["condition"])):
            pos = g[(g["label"] == 1) & (g["condition"] == cond)]["score"].values
            neg_same = g[(g["label"] == 0) & (g["condition"] == cond)]["score"].values
            k, n = rate_above(pos, default_threshold)
            _, tlo, thi = wilson(k, n)
            kf, nf = rate_above(neg_same if len(neg_same) else neg_all, default_threshold)
            _, flo, fhi = wilson(kf, nf)
            kp, _ = rate_above(pos, thr_pooled)
            _, plo, phi = wilson(kp, n)
            neg_ref = neg_same if len(neg_same) else neg_all
            au = auroc(pos, neg_ref)
            bs = [auroc(rng.choice(pos, len(pos)), rng.choice(neg_ref, len(neg_ref))) for _ in range(n_boot)] if len(pos) and len(neg_ref) else []
            row = {"detector": det, "condition": cond, "n_pos": n, "n_benign_same_form": len(neg_same),
                   "tpr_default": k / n if n else np.nan, "tpr_default_lo": tlo, "tpr_default_hi": thi,
                   "fpr_default": kf / nf if nf else np.nan, "fpr_default_lo": flo, "fpr_default_hi": fhi,
                   "fpr_ref": "same_form" if len(neg_same) else "pooled_benign",
                   "thr_at_fpr_pooled": thr_pooled, "tpr_at_fpr_pooled": kp / n if n else np.nan,
                   "tpr_at_fpr_pooled_lo": plo, "tpr_at_fpr_pooled_hi": phi,
                   "auroc": au, "auroc_lo": np.nanquantile(bs, .025) if bs else np.nan,
                   "auroc_hi": np.nanquantile(bs, .975) if bs else np.nan}
            if len(neg_same):
                ts = threshold_at_fpr(neg_same, fpr)
                ks, _ = rate_above(pos, ts)
                row.update({"thr_at_fpr_same_form": ts, "tpr_at_fpr_same_form": ks / n if n else np.nan})
            rows.append(row)
    t = pd.DataFrame(rows)
    t["synthetic"] = syn
    return t


# ----------------------------------------------------------------------------- T8 defences
DEFENCE_CONDITIONS = ["en", "bn_mt", "bn_llm", "banglish_llm"]


def t8_defences(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    d = _ok(df)
    if d["defense"].nunique() <= 1 and (d["defense"] == "none").all():
        return _empty("PENDING: no defence runs (C10)")
    rows = []
    for (m, dfn), g in d.groupby(["model", "defense"]):
        noatk = g[(g["condition"] == C.NO_ATTACK) & g["utility"].notna()]
        u0 = noatk["utility"].astype(float).mean() if len(noatk) else np.nan
        for cond in _order(set(g["condition"]) - {C.NO_ATTACK}):
            gg = g[(g["condition"] == cond) & g[metric].notna()]
            if gg.empty:
                continue
            a = asr_ci(gg[metric].astype(bool).values, gg["inj_uid"].values, n_boot, seed)
            gu = gg[gg["utility"].notna()]
            u = asr_ci(gu["utility"].astype(bool).values, gu["inj_uid"].values, n_boot, seed + 1)
            rows.append({"model": m, "defense": dfn, "condition": cond, "n": a["n"],
                         "asr": a["est"], "asr_lo": a["lo"], "asr_hi": a["hi"],
                         "utility_no_attack": u0, "n_no_attack": len(noatk),
                         "utility_under_attack": u["est"], "uua_lo": u["lo"], "uua_hi": u["hi"],
                         "in_plan_defence_conditions": cond in DEFENCE_CONDITIONS})
    return _syn(df, pd.DataFrame(rows))


# ----------------------------------------------------------------------------- T9 register
def t9_register(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0, t7=None):
    d = attacked(df)
    rows = []
    for m in sorted(d["model"].unique()):
        for a, b in C.REGISTER_PAIRS:
            r = contrast_row(df, a, b, metric, m, n_boot, seed, bca=False)
            if r:
                rows.append(r)
    if not rows:
        return _empty("PENDING: no W2 register-ablation runs (C11; *_llm_W2 payloads not generated this pass)")
    t = pd.DataFrame(_adjust(rows, holm=False, bh=True))   # secondary family: BH (§4.7)
    if t7 is not None and len(t7) and "tpr_default" in t7:
        hit = t7.set_index(["detector", "condition"])["tpr_default"]
        for det in t7["detector"].unique():
            t[f"det_hit_a_{det}"] = [hit.get((det, a), np.nan) for a in t["cond_a"]]
            t[f"det_hit_b_{det}"] = [hit.get((det, b), np.nan) for b in t["cond_b"]]
    return _syn(df, t)


# ----------------------------------------------------------------------------- T10 robustness
def t10_robustness(df, metric="normalised_security", flip_pairs=(), nllb_condition=C.NLLB_CONDITION,
                   n_boot=C.N_BOOT, seed=0):
    d = attacked(df)
    # (a) NLLB vs Google NMT bn_mt
    a_rows = [r for m in sorted(d["model"].unique())
              if (r := contrast_row(df, "bn_mt", nllb_condition, metric, m, n_boot, seed, bca=False))]
    ta = _syn(df, pd.DataFrame(a_rows)) if a_rows else _empty(f"PENDING: no '{nllb_condition}' runs")
    # (b) flip rates between labelled runs (Q4 vs Q8, repeats): flip_pairs = [(kind, label_a, label_b)]
    b_rows = []
    for kind, la, lb in flip_pairs:
        A = d[d["model"] == la].set_index(["suite", "user_task", "injection_task", "condition"])[metric]
        B = d[d["model"] == lb].set_index(["suite", "user_task", "injection_task", "condition"])[metric]
        j = pd.concat([A.rename("a"), B.rename("b")], axis=1).dropna()
        for cond, g in j.groupby(level="condition"):
            b_rows.append({"kind": kind, "run_a": la, "run_b": lb, "condition": cond, "n": len(g),
                           "flip_rate": flip_rate(g["a"].astype(bool), g["b"].astype(bool)),
                           "asr_a": g["a"].astype(float).mean(), "asr_b": g["b"].astype(float).mean()})
    tb = _syn(df, pd.DataFrame(b_rows)) if b_rows else _empty("PENDING: no repeat / quantisation runs given (--flip-pair)")
    # (c) output-language rate (§4.7): episodes whose tool args are mostly Bengali script
    c_rows = []
    for (m, cond), g in d.groupby(["model", "condition"]):
        ba = g["bengali_share_args"].astype(float)
        bo = g["bengali_share_output"].astype(float)
        c_rows.append({"model": m, "condition": cond, "n": len(g),
                       "rate_args_mostly_bengali": (ba > C.BENGALI_MAJORITY).mean() if ba.notna().any() else np.nan,
                       "rate_output_mostly_bengali": (bo > C.BENGALI_MAJORITY).mean() if bo.notna().any() else np.nan,
                       "mean_bengali_share_args": ba.mean(), "mean_bengali_share_output": bo.mean()})
    tc = _syn(df, pd.DataFrame(c_rows)) if c_rows else _empty("no attacked episodes")
    return ta, tb, tc
