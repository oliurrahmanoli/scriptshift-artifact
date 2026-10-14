"""Post-freeze checks (DECISIONS D26): round-trip, PAYLOAD_SHA256SUMS.txt, subset coverage.

  * payloads.json == the frozen sources (NLLB json, revalidated _gen_meta.json texts), the
    W1 wrappers match up to surrounding whitespace, and every OTHER condition/wrapper is
    unchanged versus the pre-freeze backup;
  * writes data/conditions/PAYLOAD_SHA256SUMS.txt (every installed file of the fast-track
    conditions en, bn_mt, bn_llm, banglish_llm + their W1 wrappers + _gen_meta.json, plus a
    canonical hash per condition over the payloads.json goal map);
  * coverage of subset/pairs_150.json and subset/pairs.json per condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
from make_mt import force_ascii_digits  # noqa: E402  (D27: §4.3 ASCII digits)
C = ROOT / "data" / "conditions"
FAST = ["en", "bn_mt", "bn_llm", "banglish_llm"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(d) -> str:
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def filled(v) -> bool:
    return isinstance(v, str) and v.strip() != "" and not v.startswith("TODO_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default=str(ROOT / "data" / "payloads.json.bak-20261006-freeze"))
    ap.add_argument("--nllb-json", default=str(C / "_provenance" / "2026-10-06_nllb33b_v1" / "bn_mt_nllb-3.3b.json"))
    ap.add_argument("--nllb-rerun-json", default=str(C / "_provenance" / "2026-10-06_nllb33b_rerun" / "bn_mt_nllb_rerun.json"),
                    help="D27 rerun overlay: its goals supersede the v1 NLLB json for those keys (skipped if absent)")
    args = ap.parse_args()
    pay = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))
    old = json.loads(Path(args.backup).read_text(encoding="utf-8"))
    nllb = json.loads(Path(args.nllb_json).read_text(encoding="utf-8"))
    if args.nllb_rerun_json and Path(args.nllb_rerun_json).exists():
        rerun = json.loads(Path(args.nllb_rerun_json).read_text(encoding="utf-8"))
        nllb["goals"] = {**nllb["goals"], **rerun["goals"]}
    errors = []

    # --- round trip ---
    # D27: payloads carry the §4.3 ASCII-digit normalisation; the frozen NLLB source is raw,
    # so compare against its digit-normalised form (the only permitted transform).
    for k, v in pay["conditions"]["bn_mt"]["goals"].items():
        if v != "TODO_MT" and v != force_ascii_digits(nllb["goals"][k])[0]:
            errors.append(f"bn_mt {k} differs from NLLB output (after D27 ASCII-digit norm)")
    for cond in ("bn_llm", "banglish_llm"):
        meta = json.loads((C / cond / "_gen_meta.json").read_text(encoding="utf-8"))
        for k, it in meta["items"].items():
            if pay["conditions"][cond]["goals"][k] != force_ascii_digits(it["text"])[0]:
                errors.append(f"{cond} {k} differs from _gen_meta text (after D27 ASCII-digit norm)")
            if it["status"] != "ok" and filled(pay["conditions"][cond]["goals"][k]):
                errors.append(f"{cond} {k} pending but filled in payloads")
        w = (C / "_wrappers" / f"W1__{cond}.txt").read_text(encoding="utf-8")
        if w.split("# STATUS: filled\n\n", 1)[1].strip() != pay["wrappers"]["W1"][cond].strip():
            errors.append(f"W1 {cond} wrapper mismatch")
    if nllb["wrapper_W1"].strip() != pay["wrappers"]["W1"]["bn_mt"].strip():
        errors.append("W1 bn_mt wrapper mismatch")
    for cond, blk in pay["conditions"].items():
        if cond in ("bn_mt", "bn_llm", "banglish_llm"):
            continue
        if blk != old["conditions"][cond]:
            errors.append(f"condition {cond} changed unexpectedly")
    for wid, blk in pay["wrappers"].items():
        for cond, v in blk.items():
            if not (wid == "W1" and cond in ("bn_mt", "bn_llm", "banglish_llm")) and v != old["wrappers"][wid].get(cond):
                errors.append(f"wrapper {wid}/{cond} changed unexpectedly")
    if pay["meta"] != old["meta"]:
        errors.append("payloads meta changed")

    # --- sums ---
    lines = ["# SHA-256 of every installed fast-track payload file (DECISIONS D26).",
             "# Files named *.TODO_*.txt are placeholders (NOT payloads; the attack class refuses them).",
             "# CANONICAL lines: sha256 of json.dumps(payloads.json conditions[<cond>].goals,"
             " ensure_ascii=False, sort_keys=True, separators=(',',':')).encode('utf-8')."]
    for cond in FAST:
        for p in sorted((C / cond).glob("*")):
            if p.is_file():
                lines.append(f"{sha(p)}  {p.relative_to(C)}")
    for cond in FAST:
        p = C / "_wrappers" / f"W1__{cond}.txt"
        lines.append(f"{sha(p)}  {p.relative_to(C)}")
    for cond in FAST:
        lines.append(f"{canon(pay['conditions'][cond]['goals'])}  CANONICAL_GOALS_JSON({cond})")
    (C / "PAYLOAD_SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- coverage ---
    cov = {}
    for name in ("pairs_150.json", "pairs.json"):
        pairs = json.loads((ROOT / "subset" / name).read_text(encoding="utf-8"))["pairs"]
        tasks = sorted({f"{p['suite']}/{p['injection_task']}" for p in pairs})
        cov[name] = {"n_pairs": len(pairs), "n_injection_tasks": len(tasks), "per_condition": {}}
        for cond in FAST:
            g = pay["conditions"][cond]["goals"]
            w_ok = filled(pay["wrappers"]["W1"].get(cond))
            missing = [t for t in tasks if not filled(g.get(t))]
            covered = sum(1 for p in pairs if filled(g.get(f"{p['suite']}/{p['injection_task']}")) and w_ok)
            cov[name]["per_condition"][cond] = {"pairs_covered": covered, "wrapper_W1": w_ok,
                                                "tasks_missing": missing,
                                                "pairs_missing": len(pairs) - covered}
        allc = sum(1 for p in pairs if all(filled(pay["conditions"][c]["goals"].get(f"{p['suite']}/{p['injection_task']}"))
                                           for c in FAST))
        cov[name]["pairs_covered_in_all_4"] = allc
    (C / "_provenance" / "2026-10-06_coverage.json").write_text(json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cov, indent=1))
    print("ERRORS:", errors or "none")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
