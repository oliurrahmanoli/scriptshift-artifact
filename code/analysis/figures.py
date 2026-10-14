"""F1 (ASR heat map, models x conditions) and F2 (contrast CIs, forest plot).

Both are written as PNG and PDF. If any input episode is synthetic the figure carries a large
"SYNTHETIC — NOT RESULTS" watermark (and run_analysis prefixes the file name).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import config as C  # noqa: E402

WATERMARK = "SYNTHETIC — NOT RESULTS"


def _watermark(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, WATERMARK, fontsize=34, color="red", alpha=0.25, ha="center",
                 va="center", rotation=25, transform=fig.transFigure, zorder=100)


def _save(fig, path_stem):
    out = []
    for ext in ("png", "pdf"):
        p = f"{path_stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", metadata={"Title": "", "Subject": ""} if ext == "pdf" else None)
        out.append(p)
    plt.close(fig)
    return out


def f1_heatmap(grid: pd.DataFrame, path_stem: str, conditions=None):
    """grid = tables.asr_grid output."""
    if grid.empty or "asr" not in grid:
        return []
    conditions = conditions or [c for c in C.MAIN_CONDITIONS if c in set(grid["condition"].astype(str))]
    models = sorted(grid["model"].unique())
    M = np.full((len(models), len(conditions)), np.nan)
    N = np.zeros_like(M)
    SG = np.zeros_like(M, dtype=bool)
    for _, r in grid.iterrows():
        c = str(r["condition"])
        if c in conditions:
            M[models.index(r["model"]), conditions.index(c)] = 100 * r["asr"]
            N[models.index(r["model"]), conditions.index(c)] = r["n"]
            SG[models.index(r["model"]), conditions.index(c)] = bool(r.get("self_generated", False))
    fig, ax = plt.subplots(figsize=(1.3 * len(conditions) + 2.5, 0.6 * len(models) + 1.8))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="viridis", vmin=0, vmax=max(10, np.nanmax(M) if np.isfinite(M).any() else 10), aspect="auto")
    ax.set_xticks(range(len(conditions)), conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(conditions)):
            txt = "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.1f}{'*' if SG[i, j] else ''}\n(n={int(N[i, j])})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] < np.nanmax(M) * 0.6) else "black")
    metric = grid["metric"].iloc[0] if "metric" in grid else ""
    fig.colorbar(im, ax=ax, label=f"targeted ASR (%) [{metric}]")
    ax.set_title("F1 — targeted ASR by model and language form"
                 + ("\n* = payload written by the same model (self-generated)" if SG.any() else ""), fontsize=10)
    _watermark(fig, bool(grid.get("synthetic", pd.Series([False])).any()))
    return _save(fig, path_stem)


def f2_forest(t3: pd.DataFrame, path_stem: str):
    """t3 = tables.t3_primary per-model + pooled_bootstrap rows."""
    if t3.empty or "diff_pp" not in t3:
        return []
    t = t3[t3["diff_pp"].notna()].copy() if "diff_pp" in t3 else t3.copy()
    if t.empty:
        return []
    t["model"] = t["model"].astype(str) + np.where(t.get("self_generated", pd.Series(False, index=t.index)).fillna(False).astype(bool), " (self-gen)", "")
    t["label"] = t["contrast"] + "  " + t["cond_b"] + " − " + t["cond_a"] + "  [" + t["model"] + "]"
    t = t.sort_values(["contrast", "level", "model"], ascending=[True, False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.5))
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        pooled = r["level"] != "per_model"
        ax.errorbar(r["diff_pp"], yi, xerr=[[r["diff_pp"] - r["ci_lo_pp"]], [r["ci_hi_pp"] - r["diff_pp"]]],
                    fmt="D" if pooled else "o", color="black" if pooled else "tab:blue", capsize=2, ms=5 if pooled else 4)
    ax.axvline(0, color="grey", lw=1)
    for s in (-C.TOST_MARGIN_PP, C.TOST_MARGIN_PP):
        ax.axvline(s, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y, t["label"], fontsize=7)
    ax.set_xlabel("Δ targeted ASR (pp), cluster-bootstrap 95% CI; dashed = ±5 pp TOST margin")
    ax.set_title("F2 — primary contrasts (per model and pooled)")
    _watermark(fig, bool(t.get("synthetic", pd.Series([False])).any()))
    return _save(fig, path_stem)
