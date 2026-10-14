"""Side-by-side of the D38 primary rule and its two sensitivity variants (fast-track, D38.2).

Reads <root>/{primary,sensitivity_model_error_excluded,sensitivity_overflow_episode_level}/ and writes
<root>/SENSITIVITY_SUMMARY.csv (ASR per model x condition + C1/C3 per model and pooled, every variant)
and <root>/SENSITIVITY_MAX_ABS_DIFF.json (largest change vs primary). Numbers are copied from the
variant CSVs, nothing is recomputed here.

  python -m analysis.fasttrack_summary ../paper/results/fasttrack_v1_20261007
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

VARIANTS = ["primary", "sensitivity_model_error_excluded", "sensitivity_overflow_episode_level"]


def summarise(root: Path) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        d = root / v
        g = pd.read_csv(d / "F1_asr_grid.csv")
        for _, r in g.iterrows():
            rows.append({"variant": v, "kind": "asr", "model": r["model"], "item": r["condition"],
                         "n": r["n"], "value": r["asr"], "lo": r["asr_lo"], "hi": r["asr_hi"], "p_adj": None})
        t = pd.read_csv(d / "T3_primary_contrasts.csv")
        t = t[t["level"] != "not_run"]
        for _, r in t.iterrows():
            rows.append({"variant": v, "kind": "contrast_pp", "model": r["model"], "item": r["contrast"],
                         "n": r["n_pairs"], "value": r["diff_pp"], "lo": r["ci_lo_pp"], "hi": r["ci_hi_pp"],
                         "p_adj": r.get("p_holm")})
    return pd.DataFrame(rows)


def main(argv=None):
    root = Path((argv or sys.argv[1:])[0])
    s = summarise(root)
    s.to_csv(root / "SENSITIVITY_SUMMARY.csv", index=False)
    p = s[s.variant == "primary"].set_index(["kind", "model", "item"])["value"]
    out = {}
    for v in VARIANTS[1:]:
        q = s[s.variant == v].set_index(["kind", "model", "item"])["value"]
        diff = (q - p.reindex(q.index)).abs()
        out[v] = {k: {"max_abs_diff": float(diff.xs(k, level=0).max()),
                      "at": list(diff.xs(k, level=0).idxmax())} for k in ("asr", "contrast_pp")}
    (root / "SENSITIVITY_MAX_ABS_DIFF.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
