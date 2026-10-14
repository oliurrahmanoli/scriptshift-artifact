"""Aggregate run JSONL logs into tidy records and headline tables (section 4.7).

Reads one or more runs/*.jsonl files (written by runners/run_benchmark.py), builds
the tidy record list that analysis/stats.py consumes, and prints per-condition ASR
(raw / normalised / corrected) with the size of the scoring under-count.

Usage:  python analysis/aggregate.py runs/*.jsonl
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_runs(paths):
    recs = []
    for pat in paths:
        for fp in glob.glob(pat):
            for line in Path(fp).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("_header") or r.get("status") != "ok":
                    continue
                recs.append(r)
    return recs


def to_tidy(recs, model_from="pipeline"):
    tidy = []
    for r in recs:
        tidy.append({
            "model": r.get("model") or r.get("condition", "") and r.get(model_from, "model"),
            "suite": r["suite"], "user_task": r["user_task"],
            "injection_task": r["injection_task"], "condition": r["condition"],
            "success": bool(r["normalised_security"]),
            "raw": bool(r["raw_security"]), "corrected": bool(r["corrected_security"]),
        })
    return tidy


def summarise(recs):
    by_cond = {}
    for r in recs:
        c = r["condition"]
        d = by_cond.setdefault(c, {"n": 0, "raw": 0, "norm": 0, "corr": 0})
        d["n"] += 1
        d["raw"] += int(r["raw_security"])
        d["norm"] += int(r["normalised_security"])
        d["corr"] += int(r["corrected_security"])
    print(f"{'condition':20s} {'n':>4} {'raw%':>7} {'norm%':>7} {'corr%':>7} {'undercount_pp':>13}")
    for c, d in sorted(by_cond.items()):
        n = d["n"]
        raw, norm = 100 * d["raw"] / n, 100 * d["norm"] / n
        corr = 100 * d["corr"] / n
        print(f"{c:20s} {n:>4} {raw:7.1f} {norm:7.1f} {corr:7.1f} {norm - raw:13.1f}")
    return by_cond


if __name__ == "__main__":
    paths = sys.argv[1:] or ["runs/*.jsonl"]
    recs = load_runs(paths)
    print(f"Loaded {len(recs)} ok records from {paths}\n")
    if recs:
        summarise(recs)
