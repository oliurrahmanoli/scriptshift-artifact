"""Statistics for the multilingual IPI study (research-plan.md 4.7, 10.2).

Implements:
  * paired McNemar test (exact) on success indicators for a condition contrast,
  * cluster bootstrap CIs resampling INJECTION TASKS (the authored unit) then user
    tasks within them (percentile + BCa-style),
  * Holm-Bonferroni and Benjamini-Hochberg multiple-comparison correction,
  * comprehension-conditional ASR: P(success | comprehended),
  * TOST equivalence test (+/-5 pp) for planned null results.

Input is a tidy list of records, one per (model, user_task, injection_task, condition):
  {"model","suite","user_task","injection_task","condition","success"(bool),
   "comprehended"(bool, optional)}
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# ----------------------------------------------------------------- paired data
def _paired(records, cond_a, cond_b, model=None):
    """Return aligned (a_success, b_success, injection_task) arrays for the pairs
    present in BOTH conditions (matched on model,suite,user_task,injection_task)."""
    def keyset(cond):
        d = {}
        for r in records:
            if r["condition"] != cond:
                continue
            if model is not None and r["model"] != model:
                continue
            k = (r["model"], r["suite"], r["user_task"], r["injection_task"])
            d[k] = r
        return d

    A, B = keyset(cond_a), keyset(cond_b)
    keys = sorted(set(A) & set(B))
    a = np.array([bool(A[k]["success"]) for k in keys])
    b = np.array([bool(B[k]["success"]) for k in keys])
    clusters = [k[3] for k in keys]  # injection_task id
    return a, b, clusters


# ----------------------------------------------------------------- McNemar
@dataclass
class ContrastResult:
    contrast: str
    model: str | None
    n: int
    n_clusters: int
    asr_a: float
    asr_b: float
    diff_pp: float
    p_value: float
    ci_low: float
    ci_high: float
    b01: int = 0  # a=0,b=1 discordant
    b10: int = 0  # a=1,b=0 discordant
    extra: dict = field(default_factory=dict)


def mcnemar_exact(a, b):
    """Exact (binomial) McNemar on paired booleans. Returns (p, b10, b01)."""
    b10 = int(np.sum(a & ~b))  # success only in a
    b01 = int(np.sum(~a & b))  # success only in b
    n = b10 + b01
    if n == 0:
        return 1.0, b10, b01
    k = min(b10, b01)
    # two-sided exact binomial p at 0.5
    p = min(1.0, 2 * stats.binom.cdf(k, n, 0.5))
    return p, b10, b01


# ----------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap_diff(a, b, clusters, n_boot=10000, seed=0, alpha=0.05):
    """Percentile CI for ASR difference (b - a) resampling injection-task clusters."""
    rng = random.Random(seed)
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    clusters = np.asarray(clusters, dtype=object)
    uniq = sorted(set(clusters.tolist()))
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    diffs = np.empty(n_boot)
    point = b.mean() - a.mean()
    for i in range(n_boot):
        chosen = [rng.choice(uniq) for _ in uniq]  # resample clusters w/ replacement
        idxs = np.concatenate([idx_by_cluster[c] for c in chosen])
        # resample user tasks within clusters too
        samp = np.array([idxs[rng.randrange(len(idxs))] for _ in range(len(idxs))])
        diffs[i] = b[samp].mean() - a[samp].mean()
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi, diffs


def contrast(records, cond_a, cond_b, model=None, n_boot=10000, seed=0, alpha=0.05):
    a, b, clusters = _paired(records, cond_a, cond_b, model)
    if len(a) == 0:
        raise ValueError(f"No paired observations for {cond_a} vs {cond_b} (model={model})")
    p, b10, b01 = mcnemar_exact(a, b)
    point, lo, hi, _ = cluster_bootstrap_diff(a, b, clusters, n_boot, seed, alpha)
    return ContrastResult(
        contrast=f"{cond_b}_vs_{cond_a}", model=model, n=len(a),
        n_clusters=len(set(clusters)), asr_a=float(a.mean()), asr_b=float(b.mean()),
        diff_pp=100 * point, p_value=p, ci_low=100 * lo, ci_high=100 * hi,
        b01=b01, b10=b10)


# ----------------------------------------------------------------- corrections
def holm_bonferroni(pvals, alpha=0.05):
    """Return list of (index, p, adjusted_p, reject) in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        a = (m - rank) * pvals[i]
        running = max(running, a)
        adj[i] = min(1.0, running)
    reject = [adj[i] < alpha for i in range(m)]
    return [(i, pvals[i], adj[i], reject[i]) for i in range(m)]


def benjamini_hochberg(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        a = pvals[i] * m / (rank + 1)
        prev = min(prev, a)
        adj[i] = min(1.0, prev)
    reject = [adj[i] < q for i in range(m)]
    return [(i, pvals[i], adj[i], reject[i]) for i in range(m)]


# ----------------------------------------------------------------- comprehension
def comprehension_conditional_asr(records, condition, model=None):
    """P(success | comprehended) and refusal/base rates for a condition."""
    rel = [r for r in records if r["condition"] == condition
           and (model is None or r["model"] == model)
           and "comprehended" in r]
    comp = [r for r in rel if r["comprehended"]]
    out = {
        "condition": condition, "model": model, "n": len(rel),
        "comprehension_rate": (sum(r["comprehended"] for r in rel) / len(rel)) if rel else None,
        "asr_all": (sum(bool(r["success"]) for r in rel) / len(rel)) if rel else None,
        "asr_given_comprehended": (sum(bool(r["success"]) for r in comp) / len(comp)) if comp else None,
    }
    return out


# ----------------------------------------------------------------- TOST
def tost_equivalence(records, cond_a, cond_b, model=None, margin_pp=5.0,
                     n_boot=10000, seed=0):
    """Two one-sided tests via the bootstrap distribution of the paired diff.
    Equivalent if the 90% CI (for alpha=0.05 TOST) lies within +/- margin."""
    a, b, clusters = _paired(records, cond_a, cond_b, model)
    _, _, _, diffs = cluster_bootstrap_diff(a, b, clusters, n_boot, seed)
    diffs_pp = 100 * diffs
    lo = float(np.percentile(diffs_pp, 5))
    hi = float(np.percentile(diffs_pp, 95))
    equivalent = (lo > -margin_pp) and (hi < margin_pp)
    return {"contrast": f"{cond_b}_vs_{cond_a}", "model": model,
            "margin_pp": margin_pp, "ci90_low": lo, "ci90_high": hi,
            "equivalent": equivalent}
