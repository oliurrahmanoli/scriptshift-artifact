"""Exploratory tables added for the fast-track grid (DECISIONS D40). NOT pre-registered; labelled so.

* T11_injection_rendering — how the injected <INFORMATION> block reached the agent. AgentDojo 0.1.35
  formats structured tool results with ``yaml.safe_dump`` (``allow_unicode=False``), so Bengali-script
  payloads placed in YAML-dumped fields arrive as ``\\uXXXX`` escapes, while plain-string tool results keep
  the script. Banglish and English are ASCII and unaffected. Counts per model x condition.
* T11b_asr_by_rendering_stratum — ASR per condition within pair strata defined by how the *Bengali-script*
  payload of the same (model, pair) was rendered (stratum = rendering of the bn_mt episode, else bn_llm):
  ``raw_script`` vs ``escaped`` vs ``not_seen``. Because the stratum is a property of the pair (injection
  vector), all four conditions of a pair share it, so within-stratum contrasts stay paired.
* T11c_contrasts_by_stratum — paired Δpp (cluster bootstrap) + exact McNemar within each stratum, BH over
  the table. Exploratory.
* T12_asr_by_scoring_group — ASR per model x condition x scoring group (A identifiers / B verbatim copy /
  C attacker free text, plan §4.4), raw and normalised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import config as C
from analysis.bootstrap import asr_ci
from analysis.tables import _adjust, _empty, _syn, attacked, contrast_row

STRATUM_CONDS = ("bn_mt", "bn_llm")


def pair_strata(df: pd.DataFrame) -> pd.Series:
    """Map (model, suite, user_task, injection_task) -> rendering stratum of the Bengali-script payload."""
    d = attacked(df)
    if "injection_render" not in d:
        return pd.Series(dtype=object)
    out = {}
    for cond in reversed(STRATUM_CONDS):          # bn_mt wins (assigned last)
        g = d[d["condition"] == cond]
        for k, r in zip(zip(g["model"], g["suite"], g["user_task"], g["injection_task"]), g["injection_render"]):
            if isinstance(r, str):
                out[k] = {"raw_bengali_script": "raw_script", "escaped_unicode": "escaped"}.get(r, r)
    return pd.Series(out)


def with_stratum(df):
    s = pair_strata(df)
    d = df.copy()
    keys = list(zip(d["model"], d["suite"], d["user_task"], d["injection_task"]))
    d["render_stratum"] = [s.get(k, np.nan) if c != C.NO_ATTACK else np.nan for k, c in zip(keys, d["condition"])]
    return d


def t11_rendering(df):
    d = attacked(df)
    if "injection_render" not in d or d["injection_render"].isna().all():
        return _empty("no trace messages: rendering unknown")
    t = pd.crosstab([d["model"], d["condition"]], d["injection_render"]).reset_index()
    t.columns.name = None
    t["exploratory"] = True
    return _syn(df, t)


def t11b_asr_by_stratum(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    d = with_stratum(df)
    a = attacked(d)
    a = a[a[metric].notna() & a["render_stratum"].notna()]
    if a.empty:
        return _empty("no strata")
    rows = []
    for (m, st, cond), g in a.groupby(["model", "render_stratum", "condition"]):
        ci = asr_ci(g[metric].astype(bool).values, g["inj_uid"].values, n_boot, seed)
        rows.append({"model": m, "render_stratum": st, "condition": cond, "n": ci["n"],
                     "n_clusters": ci["n_clusters"], "asr": ci["est"], "asr_lo": ci["lo"], "asr_hi": ci["hi"]})
    # pooled over the primary-pool models
    ap = a[a["primary_pool"].fillna(True).astype(bool)] if "primary_pool" in a else a
    for (st, cond), g in ap.groupby(["render_stratum", "condition"]):
        ci = asr_ci(g[metric].astype(bool).values, g["inj_uid"].values, n_boot, seed)
        rows.append({"model": "POOLED_primary_pool", "render_stratum": st, "condition": cond, "n": ci["n"],
                     "n_clusters": ci["n_clusters"], "asr": ci["est"], "asr_lo": ci["lo"], "asr_hi": ci["hi"]})
    t = pd.DataFrame(rows)
    t["exploratory"] = True
    return _syn(df, t)


def t11c_contrasts_by_stratum(df, metric="normalised_security", n_boot=C.N_BOOT, seed=0):
    d = with_stratum(df)
    rows = []
    pairs = [("en", "bn_llm"), ("en", "bn_mt"), ("bn_mt", "bn_llm"), ("bn_llm", "banglish_llm"), ("en", "banglish_llm")]
    for st in ("raw_script", "escaped"):
        ds = d[(d["render_stratum"] == st) | (d["condition"] == C.NO_ATTACK)]
        for m in sorted(attacked(ds)["model"].unique()):
            for a_, b_ in pairs:
                r = contrast_row(ds, a_, b_, metric, m, n_boot, seed, bca=False)
                if r:
                    rows.append({"render_stratum": st, **r})
        dp = ds[ds["primary_pool"].fillna(True).astype(bool)] if "primary_pool" in ds else ds
        for a_, b_ in pairs:
            r = contrast_row(dp, a_, b_, metric, None, n_boot, seed, bca=False)
            if r:
                r["model"] = "POOLED_primary_pool"
                rows.append({"render_stratum": st, **r})
    if not rows:
        return _empty("no strata")
    t = pd.DataFrame(_adjust(rows, holm=False, bh=True))
    t["exploratory"] = True
    return _syn(df, t)


def t12_by_group(df, n_boot=C.N_BOOT, seed=0):
    d = attacked(df)
    if "scoring_group" not in d or d["scoring_group"].isna().all():
        return _empty("no scoring_group (needs scoring JSONL)")
    rows = []
    for (m, cond, grp), g in d.groupby(["model", "condition", "scoring_group"]):
        row = {"model": m, "condition": cond, "scoring_group": grp, "n": len(g),
               "n_injection_tasks": g["inj_uid"].nunique()}
        for met in ("raw_security", "normalised_security"):
            gg = g[g[met].notna()]
            row[met] = gg[met].astype(float).mean() if len(gg) else np.nan
        rows.append(row)
    return _syn(df, pd.DataFrame(rows))
