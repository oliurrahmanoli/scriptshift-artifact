"""Small, dependency-light metric functions used by analysis/tables.py.

* Wilson score interval (detector TPR/FPR, utility in tiny pilot cells; §4.7 "Wilson 95% CIs").
* TOST equivalence from a bootstrap distribution (±5 pp, §4.7).
* Cohen's kappa and Krippendorff's alpha (nominal / ordinal / interval) for validator agreement
  (§4.3 T1 and the human audit in T4).
* Detector metrics: threshold at a target FPR, TPR at that threshold, FPR at a fixed threshold,
  AUROC (Mann–Whitney) (§4.6, T7).
* Flip rate between two runs on matched episodes (repeats, Q4 vs Q8; T10).
McNemar / Holm / BH come from analysis/stats.py (unchanged).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st


def wilson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = _st.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def tost_from_boot(diff_pp: float, boot_pp, margin_pp: float = 5.0, alpha: float = 0.05):
    """TOST at level alpha: equivalent iff the (1-2*alpha) percentile CI lies inside ±margin.
    Also returns a normal-approximation TOST p-value using the bootstrap SE."""
    boot_pp = np.asarray(boot_pp, dtype=float)
    lo, hi = np.quantile(boot_pp, [alpha, 1 - alpha])
    se = float(boot_pp.std(ddof=1)) if len(boot_pp) > 1 else np.nan
    if se and se > 0:
        p_lower = 1 - _st.norm.cdf((diff_pp + margin_pp) / se)   # H0: diff <= -m
        p_upper = _st.norm.cdf((diff_pp - margin_pp) / se)       # H0: diff >= +m
        p = float(max(p_lower, p_upper))
    else:
        p = 0.0 if abs(diff_pp) < margin_pp else 1.0
    return {"tost_ci_lo": float(lo), "tost_ci_hi": float(hi), "tost_margin_pp": margin_pp,
            "tost_p": p, "equivalent": bool(lo > -margin_pp and hi < margin_pp)}


# ----------------------------------------------------------------------------- agreement
def cohen_kappa(r1, r2) -> float:
    r1, r2 = list(r1), list(r2)
    if len(r1) != len(r2) or not r1:
        raise ValueError("ratings must be non-empty and the same length")
    cats = sorted(set(r1) | set(r2), key=str)
    n = len(r1)
    po = sum(a == b for a, b in zip(r1, r2)) / n
    pe = sum((r1.count(c) / n) * (r2.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def krippendorff_alpha(data, level: str = "ordinal") -> float:
    """data: list of units, each a list of ratings (None/NaN = missing). Krippendorff (2011)."""
    units = []
    for u in data:
        vals = [v for v in u if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(vals) >= 2:
            units.append(vals)
    values = sorted({v for u in units for v in u})
    if len(values) < 2:
        return 1.0
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V))
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[idx[a], idx[b]] += 1.0 / (m - 1)
    nc = o.sum(axis=1)
    n = nc.sum()
    d2 = np.zeros((V, V))
    for c in range(V):
        for k in range(V):
            if level == "nominal":
                d2[c, k] = 0.0 if c == k else 1.0
            elif level == "interval":
                d2[c, k] = (values[c] - values[k]) ** 2
            elif level == "ordinal":
                lo, hi = min(c, k), max(c, k)
                d2[c, k] = (nc[lo:hi + 1].sum() - (nc[c] + nc[k]) / 2.0) ** 2
            else:
                raise ValueError(level)
    Do = (o * d2).sum() / n
    De = (np.outer(nc, nc) * d2).sum() / (n * (n - 1))
    return 1.0 if De == 0 else float(1 - Do / De)


# ----------------------------------------------------------------------------- detectors
def auroc(scores_pos, scores_neg) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann–Whitney U / (n_pos n_neg))."""
    pos, neg = np.asarray(scores_pos, float), np.sort(np.asarray(scores_neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lt = np.searchsorted(neg, pos, side="left")
    le = np.searchsorted(neg, pos, side="right")
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg)))


def threshold_at_fpr(scores_neg, fpr: float = 0.02) -> float:
    """Smallest threshold t (flag if score > t) with benign FPR <= fpr."""
    neg = np.sort(np.asarray(scores_neg, float))
    if len(neg) == 0:
        return np.nan
    k = int(math.floor(fpr * len(neg)))       # number of benign allowed above t
    return float(neg[len(neg) - k - 1]) if k < len(neg) else float(neg[0]) - 1e-9


def rate_above(scores, t):
    s = np.asarray(scores, float)
    k = int((s > t).sum())
    return k, len(s)


# ----------------------------------------------------------------------------- flips
def flip_rate(x, y) -> float:
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    return float(np.mean(x != y)) if len(x) else np.nan
