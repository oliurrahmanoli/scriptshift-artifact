"""Select the episodes for the Unicode-preserving rerun arm (DECISIONS D41) from the STOCK main grid.

Reads each model's stock scoring JSONL (``results/main/<m>/runs/<m>_150.jsonl``) and the trace each
episode left (``<m>_150_traces.zip``, the hashed artefact; falls back to the unpacked
``runs/traces/``) and writes ``data/uni_rerun/episodes_<model>.json`` + ``SELECTION.json``. Nothing is sampled or judged by hand.

Per stock attack episode, from the tool messages the agent actually received:

* ``injection_escaped`` — some ``<INFORMATION>`` block arrived with ``\\uXXXX``/``\\xXX`` escapes
  (AgentDojo 0.1.35 ``yaml.safe_dump`` default ``allow_unicode=False``, D40). Checked for EVERY
  condition, so an ``en`` or ``banglish_llm`` episode would be caught too if its injected text were
  not ASCII (it is ASCII: 0 non-ASCII characters in their goals and W1 wrappers, recorded in meta).
* ``benign_escape``     — escapes outside the injected block (environment data such as ``’``/``é``,
  or the agent's own text echoed back in a structured result).
* ``nonascii_args``     — the agent emitted non-ASCII tool-call arguments that were replayed to the
  server as ``json.dumps`` escapes (``openai_llm._tool_call_to_openai``) in a later turn.

Tiers:

* **A (the rerun arm, run by default)** = every ``bn_mt``/``bn_llm``/``banglish_llm`` (and, if it
  ever happened, ``en``) episode with ``injection_escaped``; plus an errored episode that left no
  messages (model-output error) whose same-model sibling Bengali-script episode of the same pair was
  ``injection_escaped`` (reason ``error_no_trace_sibling_escaped``). Context-overflow errors are not
  rerun (D36/D38.2: deterministic; the pair is excluded under the primary rule).
* **B (optional, OFF by default)** = remaining attack episodes whose agent input would still change
  under the patch only through ``benign_escape``/``nonascii_args``.
* **stock_retry** = infrastructure-error episodes (D35), rerun under STOCK formatting to complete
  the primary pre-registered set.

Usage (from code/):  python3 scripts/select_uni_rerun.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.config import ERROR_CLASS_PATTERNS  # noqa: E402
from analysis.ingest import message_features  # noqa: E402
from runners.main_grid import MODELS, SUITE_ORDER, interleave_by_suite  # noqa: E402

RESULTS = ROOT / "results" / "main"
MODEL_KEYS = ["gemma", "gptoss", "granite", "ministral", "qwen"]
CONDITIONS = ["en", "bn_mt", "bn_llm", "banglish_llm"]
RERUN_CONDITIONS = ["bn_mt", "bn_llm", "banglish_llm"]   # the arm's conditions (en reused if ASCII)
BN_SCRIPT = ("bn_mt", "bn_llm")
STOCK_EMBED_SHA256 = "ee7b4195e694977b3f4e060155d66ca373027cdc2dee98ee88750b5305169334"

ESC = re.compile(r"\\(u[0-9a-fA-F]{4}|x[89a-fA-F][0-9a-fA-F]|U[0-9a-fA-F]{8})")
INFO_RE = re.compile(r"<INFORMATION>.*?(</INFORMATION>|$)", re.S)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _text(msg) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    return " ".join(b.get("content") or "" for b in (c or []) if isinstance(b, dict))


def classify_error(text):
    if not text:
        return None
    for name, pat in ERROR_CLASS_PATTERNS:
        if re.search(pat, str(text), re.IGNORECASE):
            return name
    return "other"


def trace_flags(messages) -> dict:
    """How non-ASCII text reached the agent in one stock episode (see module doc)."""
    inj_esc = benign = args_na = False
    msgs = messages or []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            t = _text(m)
            blocks = [b.group(0) for b in INFO_RE.finditer(t)]
            if any(ESC.search(b) for b in blocks):
                inj_esc = True
            if ESC.search(INFO_RE.sub("", t)):
                benign = True
        elif m.get("role") == "assistant" and i < len(msgs) - 1:
            for tc in m.get("tool_calls") or []:
                s = json.dumps(tc.get("args", {}) if isinstance(tc, dict) else {}, ensure_ascii=False)
                if any(ord(ch) > 127 for ch in s):
                    args_na = True
    return {"injection_escaped": inj_esc, "benign_escape": benign, "nonascii_args": args_na,
            "render": message_features(msgs)["injection_render"]}


class TraceSource:
    def __init__(self, mkey: str):
        self.mkey, self.label = mkey, MODELS[mkey]["label"]
        self.zip = RESULTS / mkey / f"{mkey}_150_traces.zip"
        self.dir = RESULTS / mkey / "runs" / "traces"
        self.zf = zipfile.ZipFile(self.zip) if self.zip.exists() else None
        if self.zf is None and not self.dir.is_dir():
            raise FileNotFoundError(f"no traces for {mkey}: {self.zip} / {self.dir}")

    def get(self, suite, ut, cond, it):
        rel = f"{self.label}/{suite}/{ut}/{cond}/{it}.json"
        if self.zf is not None:
            return json.loads(self.zf.read(rel).decode("utf-8"))
        return json.loads((self.dir / rel).read_text(encoding="utf-8"))

    def source_info(self):
        return ({"traces_zip": str(self.zip.relative_to(ROOT)), "traces_zip_sha256": _sha(self.zip)}
                if self.zf is not None else {"traces_dir": str(self.dir.relative_to(ROOT))})


def _records(mkey):
    p = RESULTS / mkey / "runs" / f"{mkey}_150.jsonl"
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("_header"):
                recs.append(r)
    return p, recs


def payload_ascii_check() -> dict:
    pay = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))
    out = {}
    for c in CONDITIONS:
        goals = pay["conditions"][c]["goals"]
        w = pay["wrappers"]["W1"][c]
        out[c] = {"goals_with_non_ascii": sum(any(ord(ch) > 127 for ch in v) for v in goals.values()),
                  "n_goals": len(goals), "wrapper_non_ascii_chars": sum(ord(ch) > 127 for ch in w)}
    return out


def pair_order() -> dict:
    pairs = json.loads((ROOT / "subset" / "pairs_150.json").read_text(encoding="utf-8"))["pairs"]
    return {(p["suite"], p["user_task"], p["injection_task"]): i for i, p in enumerate(interleave_by_suite(pairs))}


def select() -> dict:
    order = pair_order()
    cidx = {c: i for i, c in enumerate(CONDITIONS)}
    models, counts, sources = OrderedDict(), {}, {}
    for mk in MODEL_KEYS:
        ts = TraceSource(mk)
        jl, recs = _records(mk)
        sources[mk] = {"jsonl": str(jl.relative_to(ROOT)), "jsonl_sha256": _sha(jl), **ts.source_info()}
        att = [r for r in recs if r.get("kind", "attack") == "attack" and r["condition"] in CONDITIONS]
        flags = {}
        for r in att:
            k = (r["suite"], r["user_task"], r["injection_task"], r["condition"])
            d = ts.get(*[r["suite"], r["user_task"], r["condition"], r["injection_task"]])
            f = trace_flags(d.get("messages"))
            f["status"], f["error_class"] = r.get("status", "ok"), classify_error(r.get("error"))
            f["no_messages"] = not (d.get("messages") or [])
            f["duration_s"] = r.get("duration_s")
            flags[k] = f
        tier_a, tier_b, retry = [], [], []
        for k, f in flags.items():
            s, ut, it, c = k
            ep = {"suite": s, "user_task": ut, "injection_task": it, "condition": c,
                  "stock_render": f["render"], "stock_duration_s": f["duration_s"]}
            if f["error_class"] == "infrastructure":
                retry.append({**ep, "reason": "infrastructure_error_stock_retry"})
                continue
            if f["injection_escaped"]:
                mixed = f["render"] == "raw_bengali_script"
                tier_a.append({**ep, "reason": "injection_escaped" + ("_mixed_with_script" if mixed else "")})
                continue
            if f["status"] != "ok" and f["no_messages"] and f["error_class"] == "model_output" and c in BN_SCRIPT:
                sib = [flags.get((s, ut, it, o)) for o in BN_SCRIPT if o != c]
                if any(x and x["injection_escaped"] for x in sib):
                    tier_a.append({**ep, "reason": "error_no_trace_sibling_escaped"})
                    continue
            if f["status"] == "ok" and (f["benign_escape"] or f["nonascii_args"]):
                why = "+".join(n for n in ("benign_escape", "nonascii_args") if f[n])
                tier_b.append({**ep, "reason": why})
        key = lambda e: (order[(e["suite"], e["user_task"], e["injection_task"])], cidx[e["condition"]])
        tier_a.sort(key=key)
        tier_b.sort(key=key)
        retry.sort(key=key)
        bad = [e for e in tier_a if e["condition"] not in RERUN_CONDITIONS]
        models[mk] = {"label": MODELS[mk]["label"], "tier_A": tier_a, "tier_B": tier_b, "stock_retry": retry,
                      "not_rerun_errors": sorted(
                          [{"suite": k[0], "user_task": k[1], "injection_task": k[2], "condition": k[3],
                            "error_class": f["error_class"]}
                           for k, f in flags.items() if f["status"] != "ok"
                           and not any(e["suite"] == k[0] and e["user_task"] == k[1] and e["injection_task"] == k[2]
                                       and e["condition"] == k[3] for e in tier_a + retry)],
                          key=lambda e: (e["suite"], e["user_task"], e["injection_task"], e["condition"])),
                      "en_or_other_in_tier_A": len(bad)}
        cnt = {}
        for tier in ("tier_A", "tier_B", "stock_retry"):
            cnt[tier] = dict(Counter(e["condition"] for e in models[mk][tier]))
            cnt[tier]["total"] = len(models[mk][tier])
        cnt["tier_A_reasons"] = dict(Counter(e["reason"] for e in tier_a))
        counts[mk] = cnt
    return {"_meta": {
        "decision": "D41", "arm": "unicode_preserving (secondary, post hoc)",
        "generated_by": "scripts/select_uni_rerun.py",
        "stock_embed_sha256": STOCK_EMBED_SHA256,
        "criterion_tier_A": "stock attack episode in which some <INFORMATION> block reached the agent with "
                            "\\uXXXX/\\xXX escapes (any condition); + model-output-error episodes without "
                            "messages whose same-model sibling Bengali-script episode was escaped",
        "criterion_tier_B": "other attack episodes whose agent input still differs under the patch "
                            "(escapes outside the injected block, or non-ASCII tool-call args replayed)",
        "criterion_stock_retry": "infrastructure-error episodes (APITimeout etc.), rerun with the patch OFF",
        "payload_ascii_check": payload_ascii_check(), "sources": sources, "counts": counts},
        "models": models}


def _dump_model(mk: str, block: dict) -> str:
    """One episode per line (readable, diff-friendly, small enough to embed)."""
    lines = ["{", f' "model_key": {json.dumps(mk)},', f' "label": {json.dumps(block["label"])},']
    keys = ["tier_A", "tier_B", "stock_retry", "not_rerun_errors"]
    for i, k in enumerate(keys):
        eps = block[k]
        body = ",\n".join("  " + json.dumps(e, ensure_ascii=False, sort_keys=True) for e in eps)
        lines.append(f' "{k}": [' + ("\n" + body + "\n ]" if eps else "]") + ("," if i < len(keys) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_files(sel: dict) -> dict:
    """{relative path: text} for data/uni_rerun/: one episodes_<model>.json per model + SELECTION.json."""
    files, shas = {}, {}
    for mk, block in sel["models"].items():
        rel = f"data/uni_rerun/episodes_{mk}.json"
        files[rel] = _dump_model(mk, block)
        shas[mk] = {"path": rel, "sha256": hashlib.sha256(files[rel].encode("utf-8")).hexdigest()}
    files["data/uni_rerun/SELECTION.json"] = json.dumps({"_meta": sel["_meta"], "files": shas},
                                                        ensure_ascii=False, indent=1) + "\n"
    return files


def load_model_episodes(mk: str, root: Path = ROOT) -> dict:
    return json.loads((root / "data" / "uni_rerun" / f"episodes_{mk}.json").read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed files differ from a fresh selection")
    a = ap.parse_args(argv)
    sel = select()
    files = render_files(sel)
    if a.check:
        stale = [rel for rel, txt in files.items()
                 if not (ROOT / rel).exists() or (ROOT / rel).read_text(encoding="utf-8") != txt]
        if stale:
            raise SystemExit(f"stale: {stale}; rerun scripts/select_uni_rerun.py")
        print("data/uni_rerun is current")
        return 0
    for rel, txt in files.items():
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).write_text(txt, encoding="utf-8")
        print(f"wrote {rel} ({len(txt.encode())} B)")
    print(json.dumps(sel["_meta"]["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
