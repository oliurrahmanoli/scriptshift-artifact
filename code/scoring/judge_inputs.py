"""Pre-rendered judge inputs for the self-contained Kaggle judge notebooks (DECISIONS D39).

The full AgentDojo trace JSONs of the 2,992 attacked main-grid episodes are ~60 MB; the judge only
ever sees ``render_trace(messages)`` (assistant tool calls + arguments + final answer; tool results
excluded by default, D23). This module renders that exact text once on the box and writes one JSONL
record per attacked, status-ok episode, so the notebook embeds ~0.4 MB (xz) instead of the traces:

  {model, defense, suite, user_task, injection_task, condition, rep,      <- analysis.ingest.KEY
   gated, normalised_security, scoring_group, judge_id,
   trace_text, trace_chars, trace_sha256, render: {include_tool_results, max_chars}}

``gated`` = the in-run ``needs_judge`` flag (§4.4 gate), recomputed from the trace only if absent.
``trace_sha256`` is the SHA-256 of the source trace JSON file, so every verdict can be tied back to
the trace in ``results/main/<m>/<m>_150_traces.zip``. ``run_judge.py --inputs`` consumes this file;
``tests/test_judge_notebook.py`` checks the text equals a fresh ``render_trace`` of the trace.

  python -m scoring.judge_inputs --scores 'results/main/*/runs/*_150.jsonl' \
      --traces results/main/*/runs/traces --config scoring/judge_config.fasttrack.json --out X.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ingest import KEY, load_scores, load_traces  # noqa: E402
from scoring.judge import load_judge_config, render_trace, select_judge_id  # noqa: E402

RENDER = {"include_tool_results": False, "max_chars": 12000}


def _sha(fp) -> str:
    return hashlib.sha256(Path(fp).read_bytes()).hexdigest()


def build_records(scores, traces, config: dict | None = None) -> list[dict]:
    """One record per attacked, status-ok scored episode that has a trace. Sorted by KEY."""
    from scoring.run_judge import select_episodes  # local import: run_judge imports this module
    eps = select_episodes(scores, traces, ungated=True)
    eps = eps.merge(traces[KEY + ["trace_path"]], on=KEY, how="left")
    recs = []
    for _, r in eps.iterrows():
        if not isinstance(r.get("messages"), list):
            raise ValueError(f"no trace for {tuple(r[k] for k in KEY)}")
        text = render_trace(r["messages"], **RENDER)
        rec = {k: (int(r[k]) if k == "rep" else r[k]) for k in KEY}
        rec.update(gated=bool(r["_gate"]), normalised_security=bool(r["normalised_security"]),
                   scoring_group=r.get("scoring_group"),
                   judge_id=select_judge_id(r["model"], config) if config else None,
                   trace_text=text, trace_chars=len(text), trace_sha256=_sha(r["trace_path"]),
                   render=RENDER)
        recs.append(rec)
    recs.sort(key=lambda d: tuple(str(d[k]) for k in KEY))
    return recs


def write_jsonl(recs, out) -> bytes:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs).encode("utf-8")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)
    return data


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = load_judge_config(a.config) if a.config else None
    recs = build_records(load_scores(a.scores), load_traces(a.traces, keep_messages=True), cfg)
    write_jsonl(recs, a.out)
    print(json.dumps({"records": len(recs), "gated": sum(r["gated"] for r in recs), "out": a.out}))


if __name__ == "__main__":
    main()
