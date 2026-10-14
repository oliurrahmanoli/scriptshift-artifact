"""Unicode-preserving arm (DECISIONS D41): SECONDARY, POST HOC. Not part of the pre-registered primary
analysis (which stays the stock AgentDojo 0.1.35 run, D38).

The arm is the stock grid with every rerun episode replaced by its Unicode-preserving rerun
(``kaggle/uni_rerun_<model>.ipynb``: same harness, seed and serve settings; only the tool-output formatter
patched to ``allow_unicode=True``). Episodes outside the rerun list are reused from the stock run (all
``en`` and ``banglish_llm`` episodes, whose injected text is ASCII, and the ``bn_*`` episodes whose
injection already reached the agent as script or never reached it). A listed episode without a rerun
result is **missing** in the arm (never silently replaced by its escaped stock version).

Outputs (``--out``; ``SYNTHETIC_`` prefix + watermark if any input is synthetic; every table carries
``arm`` and ``analysis_status = secondary_post_hoc_D41``):

* ``U0_arm_accounting``            per model × condition: stock reused / rerun / rerun missing / errors.
* ``U1_rerun_rendering_check``     how the injected block reached the agent in the rerun episodes
                                   (must be raw script; ``patch_ok`` False flags any escaped rerun).
* ``U2_asr_grid_unicode`` + ``FU1_asr_heatmap_unicode``      ASR / utility under attack (as F1).
* ``U3_primary_contrasts_unicode`` + ``U3_pooled_or_unicode`` C1 and C3 exactly as T3 (per-model McNemar +
                                   Holm, pooled bootstrap, GEE/VB cross-checks; C2 NOT RUN).
* ``U3b_exploratory_pairs_unicode`` the D38.4 exploratory pairs (BH) on the arm.
* ``U4_stock_vs_unicode_asr``      per model × condition (+ pooled primary pool): stock vs Unicode ASR, paired
                                   Δpp with cluster-bootstrap CI over all episodes and over the rerun episodes
                                   only, exact McNemar on the rerun episodes.
* ``U5_stock_vs_unicode_contrasts`` C1/C3 per model and pooled, stock and Unicode arm side by side.
* ``FU2_contrasts_stock_vs_unicode`` forest plot of U5.
* ``episodes_tidy_unicode.csv`` (input for ``Rscript analysis/glmm.R … pool|all``), ``MANIFEST.json``.

Run (once ``results/uni/<model>/`` exist; see scripts/run_unicode_arm_analysis.sh)::

    python -m analysis.unicode_arm --stock-traces results/main/*/runs/traces \
        --stock-scores 'results/main/*/runs/*_150.jsonl' \
        --uni-traces results/uni/*/runs/traces_uni --uni-scores 'results/uni/*/runs/*_150_uni.jsonl' \
        --retry-traces results/uni/*/runs/traces_stockretry --retry-scores 'results/uni/*/runs/*_stockretry.jsonl' \
        --episodes-dir data/uni_rerun --self-generated-models gemma-4-26b-a4b --out ../paper/results/unicode_arm_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import config as C  # noqa: E402
from analysis import extra_tables, figures, tables  # noqa: E402
from analysis.bootstrap import paired_diff_ci  # noqa: E402
from analysis.episode_rules import accounting, apply_rules  # noqa: E402
from analysis.ingest import load_all  # noqa: E402
from analysis.stats import mcnemar_exact  # noqa: E402

ARM = "unicode_preserving"
STATUS = "secondary_post_hoc_D41"
EKEY = ["model", "suite", "user_task", "injection_task", "condition"]
U_CONTRASTS = [c for c in C.PRIMARY_CONTRASTS if c[0] in ("C1", "C3")]


def _expand(pats):
    out = []
    for p in pats or []:
        out.extend(sorted(glob.glob(p)))
    return out


def _keys(df):
    return list(zip(*(df[c] for c in EKEY)))


def load_lists(episodes_dir, tiers=("tier_A",)) -> pd.DataFrame:
    """Listed rerun episodes (data/uni_rerun/episodes_<model>.json) keyed by model label."""
    rows = []
    for fp in sorted(Path(episodes_dir).glob("episodes_*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        for t in tiers:
            for e in d.get(t, []):
                rows.append({"model": d["label"], "suite": e["suite"], "user_task": e["user_task"],
                             "injection_task": e["injection_task"], "condition": e["condition"],
                             "tier": t, "reason": e.get("reason")})
    return pd.DataFrame(rows, columns=EKEY + ["tier", "reason"])


def merge_retry(stock: pd.DataFrame, retry: pd.DataFrame | None) -> pd.DataFrame:
    """Fold a stock-formatting retry (D35/D41) into the stock table: a retried key replaces the stock row
    only if the stock row was an error and the retry succeeded (status ok). Marks ``retry_merged``."""
    s = stock.copy()
    s["retry_merged"] = False
    if retry is None or not len(retry):
        return s
    ok = retry[retry["status"].fillna("ok").eq("ok")]
    rk = set(_keys(ok))
    sk = pd.Series(_keys(s), index=s.index)
    replace = sk.isin(rk) & s["status"].fillna("ok").ne("ok")
    add = ok[pd.Series(_keys(ok), index=ok.index).isin(set(sk[replace]))].copy()
    add["retry_merged"] = True
    out = pd.concat([s[~replace], add], ignore_index=True)
    out.attrs = dict(stock.attrs)
    out.attrs["retry_merged"] = int(len(add))
    return out


def build_arm(stock: pd.DataFrame, uni: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """Stock rows, with every listed key replaced by its rerun row; listed keys without a rerun row become
    ``status = rerun_missing``. Rerun rows that are not listed are an error (wrong inputs)."""
    lk = set(_keys(listed))
    uk = pd.Series(_keys(uni), index=uni.index) if len(uni) else pd.Series(dtype=object)
    extra = sorted(set(uk) - lk)
    if extra:
        raise ValueError(f"{len(extra)} rerun episodes are not in the rerun list, e.g. {extra[:3]}")
    if len(uk) and uk.duplicated().any():
        raise ValueError(f"duplicate rerun episodes: {list(uk[uk.duplicated()])[:3]}")
    s = stock.copy()
    sk = pd.Series(_keys(s), index=s.index)
    s["arm_source"] = np.where(sk.isin(lk), "stock_escaped_replaced", "stock_reused")
    keep = s[s["arm_source"] == "stock_reused"].copy()
    u = uni.copy()
    u["arm_source"] = "unicode_rerun"
    have = set(uk)
    miss = s[(s["arm_source"] == "stock_escaped_replaced") & ~sk.isin(have)].copy()
    miss["arm_source"] = "rerun_missing"
    miss["status"] = "rerun_missing"
    for c in ("raw_security", "normalised_security", "corrected_security", "judge_aug_security", "utility"):
        if c in miss:
            miss[c] = np.nan
    out = pd.concat([keep, u, miss], ignore_index=True)
    out["arm"] = ARM
    out.attrs = dict(stock.attrs)
    return out


def _tag(t: pd.DataFrame, arm=ARM) -> pd.DataFrame:
    t = t.copy()
    if "arm" not in t:          # U5 carries its own stock / unicode_preserving column
        t["arm"] = arm
    t["analysis_status"] = STATUS
    return t


def arm_accounting(arm: pd.DataFrame) -> pd.DataFrame:
    a = arm[arm["condition"] != C.NO_ATTACK]
    rows = []
    for (m, cond), g in a.groupby(["model", "condition"]):
        rr = g[g["arm_source"] == "unicode_rerun"]
        rows.append({"model": m, "condition": cond, "episodes": len(g),
                     "stock_reused": int((g["arm_source"] == "stock_reused").sum()),
                     "unicode_rerun": len(rr), "rerun_missing": int((g["arm_source"] == "rerun_missing").sum()),
                     "rerun_errors": int(rr["status"].fillna("ok").ne("ok").sum()),
                     "analysed": int(g["status"].eq("ok").sum())})
    t = pd.DataFrame(rows)
    acc = accounting(a.assign(defense=a["defense"].fillna("none")))
    t = t.merge(acc.drop(columns=["defense"]), on=["model", "condition"], how="left", suffixes=("", "_rules"))
    return tables._syn(arm, t)


def rendering_check(arm: pd.DataFrame) -> pd.DataFrame:
    r = arm[(arm["arm_source"] == "unicode_rerun") & (arm["condition"] != C.NO_ATTACK)]
    if r.empty or "injection_render" not in r:
        return tables._empty("no rerun episodes")
    t = pd.crosstab([r["model"], r["condition"]], r["injection_render"].fillna("unknown")).reset_index()
    t.columns.name = None
    for c in ("escaped_unicode", "raw_bengali_script", "not_seen", "ascii_only"):
        if c not in t:
            t[c] = 0
    t["patch_ok"] = t["escaped_unicode"].eq(0)
    return tables._syn(arm, t)


def _pairwise(stock_ok, arm_ok, metric, model, cond, subset, keys_rerun, n_boot, seed):
    s = stock_ok[(stock_ok["condition"] == cond) & stock_ok[metric].notna()]
    u = arm_ok[(arm_ok["condition"] == cond) & arm_ok[metric].notna()]
    if model is not None:
        s, u = s[s["model"] == model], u[u["model"] == model]
    w = s[EKEY + ["inj_uid", metric]].merge(u[EKEY + [metric]], on=EKEY, suffixes=("_stock", "_uni"))
    if subset == "rerun_only":
        w = w[pd.Series(_keys(w), index=w.index).isin(keys_rerun)]
    if w.empty:
        return None
    a = w[f"{metric}_stock"].astype(bool).values
    b = w[f"{metric}_uni"].astype(bool).values
    p, n10, n01 = mcnemar_exact(a, b)
    ci = paired_diff_ci(a, b, w["inj_uid"].values, n_boot, seed, bca=False)
    return {"model": model or "POOLED_primary_pool", "condition": cond, "subset": subset, "metric": metric,
            "n": len(w), "n_clusters": w["inj_uid"].nunique(), "asr_stock": a.mean(), "asr_unicode": b.mean(),
            "diff_pp": 100 * ci["diff"], "ci_lo_pp": 100 * ci["lo"], "ci_hi_pp": 100 * ci["hi"],
            "mcnemar_p": p if model is not None else np.nan, "n_stock_only": n10, "n_unicode_only": n01}


def stock_vs_unicode_asr(stock, arm, metric, n_boot, seed):
    so, ao = tables.attacked(stock), tables.attacked(arm)
    rerun_keys = set(_keys(arm[arm["arm_source"] == "unicode_rerun"]))
    pool = set(tables._pool(stock)["model"])
    rows = []
    for subset in ("all_episodes", "rerun_only"):
        for cond in tables._order(set(ao["condition"])):
            for m in sorted(ao["model"].unique()):
                r = _pairwise(so, ao, metric, m, cond, subset, rerun_keys, n_boot, seed)
                if r:
                    rows.append(r)
            r = _pairwise(so[so["model"].isin(pool)], ao[ao["model"].isin(pool)], metric, None, cond, subset,
                          rerun_keys, n_boot, seed)
            if r:
                rows.append(r)
    if not rows:
        return tables._empty("no paired stock/unicode episodes")
    return tables._syn(arm, pd.DataFrame(rows))


def stock_vs_unicode_contrasts(t3_stock, t3_uni):
    cols = ["contrast", "level", "model", "cond_a", "cond_b", "n_pairs", "asr_a", "asr_b", "diff_pp",
            "ci_lo_pp", "ci_hi_pp", "mcnemar_p", "p_holm", "holm_reject", "equivalent", "self_generated"]
    parts = []
    for name, t in (("stock", t3_stock), (ARM, t3_uni)):
        if "diff_pp" not in t:
            continue
        x = t[t["diff_pp"].notna()]
        x = x[[c for c in cols if c in x]].copy()
        x["arm"] = name
        parts.append(x)
    if not parts:
        return tables._empty("no contrasts")
    t = pd.concat(parts, ignore_index=True)
    t = t[t["contrast"].isin([c[0] for c in U_CONTRASTS])]
    t["analysis_status"] = STATUS
    t["synthetic"] = bool(t3_uni.get("synthetic", pd.Series([False])).any() or
                          t3_stock.get("synthetic", pd.Series([False])).any())
    return t.sort_values(["contrast", "level", "model", "arm"]).reset_index(drop=True)


def fu2_forest(u5: pd.DataFrame, path_stem: str):
    if u5.empty or "diff_pp" not in u5:
        return []
    t = u5.copy()
    t["label"] = t["contrast"] + "  " + t["cond_b"] + " − " + t["cond_a"] + "  [" + t["model"].astype(str) + "]"
    labels = list(dict.fromkeys(t.sort_values(["contrast", "level", "model"], ascending=[True, False, True])["label"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(labels) + 1.6))
    ypos = {lab: i for i, lab in enumerate(labels[::-1])}
    for arm_name, off, color in (("stock", 0.15, "tab:grey"), (ARM, -0.15, "tab:red")):
        g = t[t["arm"] == arm_name]
        for _, r in g.iterrows():
            y = ypos[r["label"]] + off
            ax.errorbar(r["diff_pp"], y, xerr=[[r["diff_pp"] - r["ci_lo_pp"]], [r["ci_hi_pp"] - r["diff_pp"]]],
                        fmt="D" if r["level"] != "per_model" else "o", color=color, capsize=2, ms=4,
                        label=arm_name if _ == g.index[0] else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()), fontsize=7)
    ax.set_xlabel("Δ targeted ASR (pp), cluster-bootstrap 95% CI")
    ax.set_title("FU2 — C1/C3: stock AgentDojo vs Unicode-preserving arm (secondary, post hoc, D41)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    figures._watermark(fig, bool(t["synthetic"].any()))
    return figures._save(fig, path_stem)


def run(args) -> dict:
    from analysis.run_analysis import _git_commit, _sha_inputs, guard_outdir
    sg = args.self_generated_models
    stock_raw = load_all(_expand(args.stock_traces), _expand(args.stock_scores))
    retry_raw = (load_all(_expand(args.retry_traces), _expand(args.retry_scores))
                 if (args.retry_traces or args.retry_scores) and _expand(args.retry_scores) else None)
    stock_raw = merge_retry(stock_raw, retry_raw)
    uni_raw = load_all(_expand(args.uni_traces), _expand(args.uni_scores))
    listed = load_lists(args.episodes_dir, args.tiers)
    arm_raw = build_arm(stock_raw, uni_raw, listed)
    rules = dict(model_error_rule=args.model_error_rule, overflow_rule=args.overflow_rule, self_generated_models=sg)
    stock = apply_rules(stock_raw, **rules)
    arm = apply_rules(arm_raw, **rules)
    arm.loc[arm["arm_source"] == "rerun_missing", "status"] = "rerun_missing"   # rules must not resurrect them
    if "exclusion" in arm:
        arm.loc[arm["arm_source"] == "rerun_missing", "exclusion"] = "unicode_rerun_missing"
    synthetic = bool(arm["synthetic"].any() or stock["synthetic"].any())
    out = Path(args.out)
    guard_outdir(out, synthetic)
    out.mkdir(parents=True, exist_ok=True)
    pre = "SYNTHETIC_" if synthetic else ""
    written = {}
    metric, nb, seed = args.metric, args.n_boot, args.seed

    def w(name, t):
        if t is None:
            return
        p = out / f"{pre}{name}.csv"
        _tag(t).to_csv(p, index=False)
        written[name] = str(p)

    arm.to_csv(out / f"{pre}episodes_tidy_unicode.csv", index=False,
               columns=[c for c in arm.columns if c != "messages"])
    written["episodes_tidy_unicode"] = str(out / f"{pre}episodes_tidy_unicode.csv")
    w("U0_arm_accounting", arm_accounting(arm))
    w("U1_rerun_rendering_check", rendering_check(arm))
    grid = tables.asr_grid(arm, metric, nb, seed)
    w("U2_asr_grid_unicode", grid)
    t3u, pooled_u = tables.t3_primary(arm, metric, nb, seed, fit_glmm=not args.no_glmm)
    t3u = t3u[t3u["contrast"].isin([c[0] for c in C.PRIMARY_CONTRASTS])] if "contrast" in t3u else t3u
    w("U3_primary_contrasts_unicode", t3u)
    w("U3_pooled_or_unicode", pooled_u if len(pooled_u) else tables._empty("no pooled fit"))
    w("U3b_exploratory_pairs_unicode", tables.t3_exploratory(arm, metric, nb, seed))
    w("U4_stock_vs_unicode_asr", stock_vs_unicode_asr(stock, arm, metric, nb, seed))
    t3s, _ = tables.t3_primary(stock, metric, nb, seed, fit_glmm=False)
    u5 = stock_vs_unicode_contrasts(t3s, t3u)
    w("U5_stock_vs_unicode_contrasts", u5)
    w("U6_rendering_unicode_arm_all", extra_tables.t11_rendering(arm))
    written["figures"] = (figures.f1_heatmap(grid, str(out / f"{pre}FU1_asr_heatmap_unicode"))
                          + fu2_forest(u5, str(out / f"{pre}FU2_contrasts_stock_vs_unicode")))
    rc = pd.read_csv(written["U1_rerun_rendering_check"])
    manifest = {"arm": ARM, "analysis_status": STATUS, "synthetic": synthetic, "metric": metric,
                "label": args.label, "tiers": list(args.tiers), "rules": arm.attrs.get("rules"),
                "n_listed": int(len(listed)), "n_rerun_rows": int(len(uni_raw)),
                "n_rerun_missing": int((arm["arm_source"] == "rerun_missing").sum()),
                "retry_merged_into_stock": int(stock_raw.attrs.get("retry_merged", 0)),
                "patch_ok_all": bool(rc["patch_ok"].all()) if "patch_ok" in rc else None,
                "code_git_commit": _git_commit(),
                "input_sha256": _sha_inputs(_expand(args.stock_scores) + _expand(args.uni_scores)
                                            + _expand(args.retry_scores)),
                "n_boot": nb, "seed": seed, "written": written,
                "note": "Secondary, post hoc arm (D41). Primary inference stays the stock run (D38)."}
    (out / f"{pre}MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["patch_ok_all"] is False:
        print("[unicode_arm] WARNING: some rerun episodes still saw escaped Bengali (U1 patch_ok=False)",
              file=sys.stderr)
    return manifest


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock-traces", nargs="*", default=[])
    ap.add_argument("--stock-scores", nargs="*", default=[])
    ap.add_argument("--uni-traces", nargs="*", default=[])
    ap.add_argument("--uni-scores", nargs="*", default=[])
    ap.add_argument("--retry-traces", nargs="*", default=[])
    ap.add_argument("--retry-scores", nargs="*", default=[])
    ap.add_argument("--episodes-dir", default=str(ROOT / "data" / "uni_rerun"))
    ap.add_argument("--tiers", nargs="*", default=["tier_A"], choices=["tier_A", "tier_B"])
    ap.add_argument("--metric", default="normalised_security")
    ap.add_argument("--model-error-rule", choices=["failure", "exclude"], default="failure")
    ap.add_argument("--overflow-rule", choices=["pair", "episode"], default="pair")
    ap.add_argument("--self-generated-models", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--no-glmm", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    return ap


if __name__ == "__main__":
    m = run(build_parser().parse_args())
    print(json.dumps({k: v for k, v in m.items() if k != "written"}, indent=2, default=str))
