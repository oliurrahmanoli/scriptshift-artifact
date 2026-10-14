"""Pooled models for the primary contrasts (research-plan.md §4.7, §10.2).

Pre-registered fit: ``success ~ condition * model + (1 | injection_task) + (1 | user_task)``
with ``lme4::glmer`` in R (see ``analysis/glmm.R``; R is not installed on the box, so that script
is provided but untested here). Python fits provided:

* **GEE** (statsmodels), binomial, exchangeable working correlation clustered on injection task —
  the plan's stated robustness check. Primary Python estimate for the pooled ORs.
* **Bayesian mixed GLM** (statsmodels ``BinomialBayesMixedGLM``, variational Bayes) with crossed
  random intercepts for injection task and user task — an *approximation* to glmer; mean-field
  posterior, so contrast SEs ignore posterior correlation between fixed effects. Reported as a
  cross-check only.

Coding: condition = Treatment(reference "en"); model = Sum (deviation) coding, so each condition
coefficient is the across-model average log-odds ratio vs en — the "pooled" contrast.
C1 = b[bn_llm] - b[bn_mt]; C2 = b[banglish_llm] - b[banglish_translit]; C3 = b[bn_llm] (reframed names, §12.9).
Holm over the 3 pooled contrasts (§10.2 primary family).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as _st

from analysis.config import BASELINE, PRIMARY_CONTRASTS
from analysis.stats import holm_bonferroni


def _cname(level):
    return f"C(condition, Treatment('{BASELINE}'))[T.{level}]"


def _L(names, cond_b, cond_a):
    L = np.zeros(len(names))
    for lvl, sgn in ((cond_b, 1.0), (cond_a, -1.0)):
        if lvl == BASELINE:
            continue
        nm = _cname(lvl)
        if nm not in names:
            return None
        L[names.index(nm)] += sgn
    return L


def _prep(df, metric, conditions):
    d = df[df["condition"].isin(conditions) & df[metric].notna()].copy()
    d["y"] = d[metric].astype(bool).astype(int)
    return d


def _formula(d, extra=""):
    rhs = f"C(condition, Treatment('{BASELINE}'))"
    if d["model"].nunique() > 1:
        rhs += " * C(model, Sum)"
    return f"y ~ {rhs}{extra}"


def _contrast_rows(names, beta, cov, label, notes="", contrasts=PRIMARY_CONTRASTS):
    rows = []
    for cname, a, b in contrasts:
        L = _L(names, b, a)
        if L is None:
            rows.append({"model_fit": label, "contrast": cname, "cond_a": a, "cond_b": b,
                         "log_or": np.nan, "se": np.nan, "or": np.nan, "or_lo": np.nan,
                         "or_hi": np.nan, "p": np.nan, "notes": "condition missing"})
            continue
        est = float(L @ beta)
        se = float(np.sqrt(L @ cov @ L))
        z = est / se if se > 0 else np.nan
        p = float(2 * (1 - _st.norm.cdf(abs(z)))) if se > 0 else np.nan
        rows.append({"model_fit": label, "contrast": cname, "cond_a": a, "cond_b": b,
                     "log_or": est, "se": se, "or": float(np.exp(est)),
                     "or_lo": float(np.exp(est - 1.96 * se)), "or_hi": float(np.exp(est + 1.96 * se)),
                     "p": p, "notes": notes})
    ps = [r["p"] for r in rows]
    ok = [i for i, p in enumerate(ps) if p == p]
    if ok:
        adj = holm_bonferroni([ps[i] for i in ok])
        for (j, _, padj, rej) in adj:
            rows[ok[j]]["p_holm"] = padj
            rows[ok[j]]["holm_reject"] = rej
    return rows


def fit_gee(df, metric="normalised_security", conditions=None, extra_terms=""):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    conditions = conditions or sorted(df["condition"].unique())
    d = _prep(df, metric, conditions)
    if d.empty or d["condition"].nunique() < 2:
        return pd.DataFrame(), None
    note = ""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            mod = smf.gee(_formula(d, extra_terms), groups="inj_uid", data=d,
                          family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            res = mod.fit(maxiter=200)
        except Exception as e:  # separation / singular design
            return pd.DataFrame([{"model_fit": "GEE_exchangeable", "notes": f"fit failed: {e}"}]), None
        if w:
            note = "; ".join(sorted({str(x.message)[:80] for x in w}))[:300]
    names = list(res.params.index)
    rows = _contrast_rows(names, res.params.values, res.cov_params().values,
                          "GEE_exchangeable", note)
    return pd.DataFrame(rows), res


def fit_vb_glmm(df, metric="normalised_security", conditions=None, seed=0):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    conditions = conditions or sorted(df["condition"].unique())
    d = _prep(df, metric, conditions)
    if d.empty or d["condition"].nunique() < 2:
        return pd.DataFrame()
    np.random.seed(seed)
    try:
        mod = BinomialBayesMixedGLM.from_formula(
            _formula(d), {"inj": "0 + C(inj_uid)", "user": "0 + C(user_uid)"}, d)
        res = mod.fit_vb()
    except Exception as e:
        return pd.DataFrame([{"model_fit": "VB_GLMM_approx", "notes": f"fit failed: {e}"}])
    names = list(mod.exog_names)
    k = len(names)
    beta = np.asarray(res.fe_mean)[:k]
    cov = np.diag(np.asarray(res.fe_sd)[:k] ** 2)
    rows = _contrast_rows(names, beta, cov, "VB_GLMM_approx",
                          "variational Bayes; mean-field SEs; glmer (glmm.R) is the registered fit")
    return pd.DataFrame(rows)


def fit_comprehension(df, metric="normalised_security", conditions=None, comp_col="comprehended_both"):
    """RQ2 descriptive refit (§4.7): pooled GEE without / with comprehension, and with a
    condition x comprehended interaction. Returns a long table of contrast estimates."""
    d = df[df[comp_col].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["comprehended"] = d[comp_col].astype(bool).astype(int)
    out = []
    for label, extra in (("without_comprehension", ""), ("with_comprehension", " + comprehended"),
                         ("condition_x_comprehended",
                          f" + comprehended + C(condition, Treatment('{BASELINE}')):comprehended")):
        t, _ = fit_gee(d, metric, conditions, extra)
        if len(t):
            t.insert(0, "specification", label)
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
