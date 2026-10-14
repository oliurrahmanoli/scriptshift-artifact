"""CLI: traces + scoring outputs -> Results tables T1–T10 (CSV) and figures F1/F2 (PNG+PDF).

Example (main grid, once results exist):
  python -m analysis.run_analysis \
      --traces /path/to/kaggle/runs --scores 'runs/*.jsonl' --judge 'runs/judge/*.jsonl' \
      --probe runs/probe.jsonl --detectors runs/detectors.csv --out analysis/out/main

Pilot no-attack utility only (real pilot traces):
  python -m analysis.run_analysis --traces kaggle/pilot-*-v1/runs \
      --pilot-results 'kaggle/pilot-*-v1/*_pilot_result.json' --out analysis/out/pilot

Safety rules (D12, D23): if ANY input is synthetic, every output file is prefixed ``SYNTHETIC_``,
carries ``synthetic=True`` and a figure watermark, and the CLI refuses to write under ``paper/``
or next to ``DECISIONS.md``.
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

import pandas as pd  # noqa: E402

from analysis import config as C  # noqa: E402
from analysis import extra_tables, figures, tables  # noqa: E402
from analysis.ingest import load_all  # noqa: E402
from analysis.episode_rules import accounting, apply_rules  # noqa: E402

FORBIDDEN_FOR_SYNTHETIC = ("paper",)


def _expand(pats):
    out = []
    for p in pats or []:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else [])
    return out


def _git_commit():
    import subprocess
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "analysis", "scoring"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return c.stdout.strip() + ("+dirty(analysis/scoring)" if dirty else "")
    except Exception:
        return None


def _sha_inputs(files):
    import hashlib
    out = {}
    for f in files:
        try:
            out[str(f)] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        except OSError:
            pass
    return out


def _versions():
    import importlib.metadata as md
    v = {}
    for pkg in ("numpy", "scipy", "pandas", "statsmodels", "matplotlib"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return v


def guard_outdir(out: Path, synthetic: bool):
    """Synthetic outputs may never land in paper/ or in a directory holding DECISIONS.md."""
    res = out.resolve()
    if synthetic and (any(f in res.parts for f in FORBIDDEN_FOR_SYNTHETIC)
                      or (res / "DECISIONS.md").exists()):
        raise SystemExit(f"Refusing to write SYNTHETIC outputs to {res} (paper/ or DECISIONS location).")


def run(args) -> dict:
    trace_dirs = _expand(args.traces)
    score_files = _expand(args.scores)
    judge_files = _expand(args.judge)
    aliases = json.loads(Path(args.aliases).read_text()) if args.aliases else None
    df = load_all(trace_dirs, score_files, judge_files, args.probe, aliases)
    retry_scores = _expand(args.retry_scores)
    if retry_scores:   # D41: stock-formatting retry of infrastructure-error episodes (kept in separate files)
        from analysis.unicode_arm import merge_retry
        df = merge_retry(df, load_all(_expand(args.retry_traces), retry_scores, judge_files, args.probe, aliases))
    df = apply_rules(df, args.model_error_rule, args.overflow_rule, args.self_generated_models)  # D38
    synthetic = bool(df["synthetic"].any())
    df_all = df
    if args.exclude_from_main:
        df = df[~df["model"].isin(args.exclude_from_main)].copy()
        df.attrs = df_all.attrs
    out = Path(args.out)
    guard_outdir(out, synthetic)
    out.mkdir(parents=True, exist_ok=True)
    pre = "SYNTHETIC_" if synthetic else ""
    written = {}

    def w(name, t):
        if t is None:
            return
        p = out / f"{pre}{name}.csv"
        t.to_csv(p, index=False)
        written[name] = str(p)

    metric = args.metric
    if metric not in df or df[metric].notna().sum() == 0:
        fallback = "raw_security"
        print(f"[analysis] {metric} unavailable (needs scoring JSONL); attack tables use {fallback} "
              f"and are labelled so.", file=sys.stderr)
        metric = fallback
    nb, seed = args.n_boot, args.seed
    df.to_csv(out / f"{pre}episodes_tidy.csv", index=False,
              columns=[c for c in df.columns if c not in ("messages",)])
    written["episodes_tidy"] = str(out / f"{pre}episodes_tidy.csv")

    w("T0_episode_accounting", accounting(df))
    if args.payloads:
        w("T1_dataset", tables.t1_dataset(args.payloads, args.ratings, n_boot=min(nb, 2000), seed=seed))
    w("T2_utility_no_attack", tables.t2_utility(df, _expand(args.pilot_results)))

    has_attack = (df["condition"] != C.NO_ATTACK).any()
    if has_attack:
        grid = tables.asr_grid(df, metric, nb, seed)
        w("F1_asr_grid", grid)
        t3, pooled = tables.t3_primary(df, metric, nb, seed, fit_glmm=not args.no_glmm)
        w("T3_primary_contrasts", t3)
        w("T3_pooled_or", pooled if len(pooled) else tables._empty("no pooled fit"))
        w("T3_vs_en_all", tables.t3_vs_en(df, metric, nb, seed))
        w("T3_exploratory_pairs", tables.t3_exploratory(df, metric, nb, seed))
        w("T3_gonogo_6a_6b_inputs", tables.gonogo_inputs(t3, pooled))
        t4, t4b, t4c = tables.t4_undercount(df, nb, seed, args.audit)
        w("T4_undercount", t4)
        w("T4_banking_precedence", t4b)
        if len(t4c):
            w("T4_judge_audit", t4c)
        if args.pairs:
            t5, t5b = tables.t5_subset(df, args.pairs, metric, nb, seed)
            w("T5_subset_vs_full", t5)
            w("T5_suite_reweighted", t5b)
        t6, t6b = tables.t6_comprehension(df, metric, nb, seed)
        w("T6_comprehension", t6)
        if len(t6b):
            w("T6_pooled_with_comprehension", t6b)
        t7 = tables.t7_detectors(args.detectors, seed=seed)
        w("T7_detectors", t7)
        w("T8_defences", tables.t8_defences(df, metric, nb, seed))
        w("T9_register", tables.t9_register(df, metric, nb, seed, t7))
        fp = [tuple(x.split(":")) for x in (args.flip_pair or [])]
        ta, tb, tc = tables.t10_robustness(df_all, metric, fp, args.nllb_condition, nb, seed)
        w("T10a_nllb_vs_google", ta)
        w("T10b_flip_rates", tb)
        w("T10c_output_language", tc)
        # exploratory, not pre-registered (D40)
        w("T11_injection_rendering", extra_tables.t11_rendering(df))
        w("T11b_asr_by_rendering_stratum", extra_tables.t11b_asr_by_stratum(df, metric, nb, seed))
        w("T11c_contrasts_by_rendering_stratum", extra_tables.t11c_contrasts_by_stratum(df, metric, nb, seed))
        w("T12_asr_by_scoring_group", extra_tables.t12_by_group(df, nb, seed))
        written["figures"] = (figures.f1_heatmap(grid, str(out / f"{pre}F1_asr_heatmap"))
                              + figures.f2_forest(t3[t3.get("level", pd.Series(dtype=str)).notna()] if "level" in t3 else t3,
                                                  str(out / f"{pre}F2_contrast_cis")))
    else:
        print("[analysis] no attacked episodes: only T1/T2 written (utility).", file=sys.stderr)
    manifest = {"synthetic": synthetic, "metric_used": metric, "n_episodes": int(len(df)),
                "label": args.label, "rules": df.attrs.get("rules"),
                "code_git_commit": _git_commit(), "input_sha256": _sha_inputs(score_files + _expand(args.judge) + retry_scores),
                "python_packages": _versions(),
                "raw_mismatch_trace_vs_score": df.attrs.get("raw_mismatch", 0),
                "retry_merged": df.attrs.get("retry_merged", 0),
                "retry_inputs": {"traces": _expand(args.retry_traces), "scores": retry_scores},
                "inputs": {"traces": trace_dirs, "scores": score_files, "judge": judge_files,
                           "probe": args.probe, "detectors": args.detectors},
                "n_boot": nb, "seed": seed, "written": written}
    (out / f"{pre}MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="*", default=[], help="AgentDojo trace roots (globs ok)")
    ap.add_argument("--scores", nargs="*", default=[], help="run_benchmark.py JSONL files (globs ok)")
    ap.add_argument("--judge", nargs="*", default=[], help="scoring/run_judge.py JSONL files")
    ap.add_argument("--retry-traces", nargs="*", default=[],
                    help="D41: trace dirs of a STOCK-formatting retry (e.g. results/uni/ministral/runs/traces_stockretry)")
    ap.add_argument("--retry-scores", nargs="*", default=[],
                    help="D41: JSONL of that retry (*_stockretry.jsonl); an ok retry replaces the errored stock episode")
    ap.add_argument("--probe", default=None, help="comprehension probe JSONL (ASSUMED schema)")
    ap.add_argument("--detectors", default=None, help="offline detector scores CSV (ASSUMED schema)")
    ap.add_argument("--audit", default=None, help="human audit CSV (ASSUMED schema)")
    ap.add_argument("--ratings", default=None, help="validator ratings CSV (ASSUMED schema)")
    ap.add_argument("--payloads", default=None, help="data/payloads.json for T1")
    ap.add_argument("--pairs", default=None, help="subset/pairs.json for T5")
    ap.add_argument("--pilot-results", nargs="*", default=[], help="*_pilot_result.json for T2")
    ap.add_argument("--aliases", default=None, help="JSON map pipeline_name -> model label")
    ap.add_argument("--metric", default="normalised_security",
                    choices=["normalised_security", "raw_security", "corrected_security", "judge_aug_security"])
    ap.add_argument("--flip-pair", action="append", help="kind:labelA:labelB, e.g. quant:ministral-3-14b:ministral-3-14b-q8")
    ap.add_argument("--exclude-from-main", nargs="*", default=[],
                    help="model labels kept out of T1–T9/F1/F2 (e.g. repeat or Q8 runs used only in T10b)")
    ap.add_argument("--nllb-condition", default=C.NLLB_CONDITION)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--no-glmm", action="store_true")
    ap.add_argument("--model-error-rule", choices=["failure", "exclude"], default="failure",
                    help="D38.2: model-output errors count as attack+utility failure (primary) or are excluded")
    ap.add_argument("--overflow-rule", choices=["pair", "episode"], default="pair",
                    help="D38.2: context overflow excludes the whole pair for that model (primary) or the episode only")
    ap.add_argument("--self-generated-models", nargs="*", default=[],
                    help="D38.3: models whose *_llm payloads they wrote themselves; kept out of pooled primary fits")
    ap.add_argument("--label", default=None, help="free-text run label stored in MANIFEST.json")
    ap.add_argument("--out", required=True)
    return ap


if __name__ == "__main__":
    m = run(build_parser().parse_args())
    print(json.dumps({k: v for k, v in m.items() if k != "written"}, indent=2))
    print("wrote", len(m["written"]), "artefacts to", Path(build_parser().parse_known_args()[0].out).resolve())
