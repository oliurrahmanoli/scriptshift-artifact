"""Vectorised two-stage cluster bootstrap (research-plan.md §4.7).

"95% CI: a cluster bootstrap (10,000 resamples) that resamples injection tasks (35 clusters,
the authored unit), and the user tasks within them. Use percentile intervals, and BCa as a check."

Stage 1 draws C clusters with replacement; stage 2 redraws each chosen cluster's units with
replacement (within-cluster). Paired contrasts resample the *same* unit indices for both
conditions, so the pairing is preserved. Statistics are ratios of resampled sums to resampled
counts (cluster sizes differ: workspace 10 users, travel 8, banking 6, slack 10 in the subset).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def _layout(clusters):
    clusters = np.asarray(clusters, dtype=object)
    uniq, inv = np.unique(clusters.astype(str), return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    maxn = int(sizes.max())
    pad = np.full((len(uniq), maxn), -1, dtype=np.int64)
    fill = np.zeros(len(uniq), dtype=np.int64)
    for i, c in enumerate(inv):
        pad[c, fill[c]] = i
        fill[c] += 1
    return uniq, inv, sizes, pad


def resampled_means(V, clusters, n_boot=10_000, seed=0, two_stage=True, chunk=500):
    """V: (n, k) array of per-unit values. Returns (n_boot, k) resampled column means."""
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    uniq, _, sizes, pad = _layout(clusters)
    C, maxn = pad.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, V.shape[1]))
    ar = np.arange(maxn)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        chosen = rng.integers(0, C, size=(b, C))
        sz = sizes[chosen]                                    # (b, C)
        if two_stage:
            within = np.floor(rng.random((b, C, maxn)) * sz[..., None]).astype(np.int64)
        else:
            within = np.broadcast_to(ar, (b, C, maxn))
        mask = ar[None, None, :] < sz[..., None]              # (b, C, maxn)
        unit = pad[chosen[..., None], within]                 # (b, C, maxn)
        unit = np.where(mask, unit, 0)
        vals = V[unit] * mask[..., None]                      # (b, C, maxn, k)
        sums = vals.sum(axis=(1, 2))
        cnt = mask.sum(axis=(1, 2))[:, None]
        out[done:done + b] = sums / cnt
        done += b
    return out


def _bca(theta_hat, boots, jack, alpha):
    """BCa interval from bootstrap replicates and leave-one-cluster-out jackknife values."""
    boots = np.asarray(boots)
    prop = np.mean(boots < theta_hat) + 0.5 * np.mean(boots == theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _st.norm.ppf(prop)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den > 0 else 0.0
    zl, zh = _st.norm.ppf(alpha / 2), _st.norm.ppf(1 - alpha / 2)
    def adj(z):
        return _st.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boots, [adj(zl), adj(zh)])
    return float(lo), float(hi)


def _jackknife(V, clusters, fn):
    _, inv, _, _ = _layout(clusters)
    out = []
    for c in range(inv.max() + 1):
        keep = inv != c
        if keep.sum() == 0:
            continue
        out.append(fn(V[keep].mean(axis=0)))
    return np.asarray(out)


def asr_ci(x, clusters, n_boot=10_000, seed=0, alpha=0.05, bca=False):
    """Point estimate and cluster-bootstrap CI for one proportion (e.g. ASR or utility)."""
    x = np.asarray(x, dtype=float)
    point = float(x.mean()) if len(x) else np.nan
    if len(x) == 0:
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0}
    B = resampled_means(x, clusters, n_boot, seed)[:, 0]
    lo, hi = np.quantile(B, [alpha / 2, 1 - alpha / 2])
    res = {"est": point, "lo": float(lo), "hi": float(hi), "n": int(len(x)),
           "n_clusters": int(len(set(map(str, clusters))))}
    if bca:
        jack = _jackknife(x[:, None], clusters, lambda m: m[0])
        res["bca_lo"], res["bca_hi"] = _bca(point, B, jack, alpha)
    return res


def paired_diff_ci(a, b, clusters, n_boot=10_000, seed=0, alpha=0.05, bca=False,
                   return_boot=False):
    """Paired difference mean(b) - mean(a) with a two-stage cluster bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    V = np.column_stack([a, b])
    M = resampled_means(V, clusters, n_boot, seed)
    D = M[:, 1] - M[:, 0]
    point = float(b.mean() - a.mean())
    lo, hi = np.quantile(D, [alpha / 2, 1 - alpha / 2])
    res = {"diff": point, "lo": float(lo), "hi": float(hi), "se_boot": float(D.std(ddof=1))}
    if bca:
        jack = _jackknife(V, clusters, lambda m: m[1] - m[0])
        res["bca_lo"], res["bca_hi"] = _bca(point, D, jack, alpha)
    if return_boot:
        res["boot"] = D
    return res
