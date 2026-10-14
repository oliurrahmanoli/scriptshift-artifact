"""Merge the two Kaggle regen outputs into data/conditions/ (DECISIONS D27).

Given the downloaded output folders of
  * `bangla-pi-gen-llm-regen`  (gen_llm_payloads_regen_pending.ipynb -> llm_payloads.zip), and
  * `bangla-pi-nllb-bn-mt-rerun` (make_bn_mt_nllb_rerun.ipynb -> bn_mt_nllb_rerun.zip),
this script, deterministically and box-only (no GPU, no network):

  1. VALIDATES every regenerated item on the box:
     * LLM items (bn_llm / banglish_llm): force ASCII digits (§4.3, D27), normalise any
       digit-bearing literal token, then re-run the literal + script validators; the
       back-translation verdict is REUSED from the regen run's _gen_meta (the D26 attempt-
       bookkeeping fix means the recorded verdict belongs to the saved candidate). An item is
       accepted only if literal + script pass AND the run recorded a passing (non-skipped)
       back-translation. Strict "adds_content" drift (D27) is already enforced inside the run.
     * bn_mt rerun items: force ASCII digits, then confirm every D26 literal
       (G.extract_literals: regex + mandated Subject/Body + ws4 override + §4.3 hotel name)
       is present verbatim; otherwise the item stays TODO_MT.
  2. MERGES only the passing items into data/payloads.json, item by item, keeping every frozen
     item untouched; re-exports data/conditions/ and updates the installed _gen_meta.json.
  3. REGENERATES data/conditions/PAYLOAD_SHA256SUMS.txt and reports per-condition counts and
     150-/300-pair coverage (target 150/150 in all four conditions; anything still missing is
     listed).

Usage (after download + unzip):
  python scripts/merge_regen_outputs.py \
    --gen-dir   /path/to/llm_payloads \
    --nllb-rerun-dir /path/to/bn_mt_nllb_rerun
Add --dry-run to validate and report without writing anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from payloads import generate_llm as G  # noqa: E402
from make_mt import force_ascii_digits  # noqa: E402
import check_payload_freeze as CPF  # noqa: E402 (sha, canon, filled, FAST, C)

COND_DIR = ROOT / "data" / "conditions"
PAY_PATH = ROOT / "data" / "payloads.json"
GOALS_PATH = ROOT / "data" / "goals_en.json"
LLM_CONDS = ["bn_llm", "banglish_llm"]
NLLB_RERUN_KEYS = ["workspace/injection_task_13", "workspace/injection_task_4",
                   "workspace/injection_task_9", "workspace/injection_task_12",
                   "travel/injection_task_0", "travel/injection_task_6"]


def _en_goals():
    goals = json.loads(GOALS_PATH.read_text(encoding="utf-8"))
    return {k: goals["goals"][k]["goal_en"] for k in goals["order"]}, goals["order"]


def validate_llm_candidate(cond, key, candidate, recorded_validation, en_goal):
    """Deterministic box re-validation of one regenerated LLM candidate (D27).

    Returns (status, final_text, detail). status 'ok' only if literal+script pass on the
    digit-normalised candidate AND the run recorded a passing, non-skipped back-translation."""
    lits = G.extract_literals(en_goal, key)
    text, n_dig = force_ascii_digits(candidate or "")
    text, norm = G.normalize_literal_digits(text, lits)
    lv = G.validate_literals(text, lits)
    sv = G.validate_script(text, lits, cond)
    bt = (recorded_validation or {}).get("backtranslation") or {}
    bt_ok = (not bt.get("skipped", True)) and bool(bt.get("ok"))
    if lv["ok"] and sv["ok"] and bt_ok:
        status = "ok"
    elif not bt_ok:
        status = "NEEDS_REVIEW_backtranslation_not_ok"
    else:
        status = "NEEDS_REVIEW_validation_failed"
    detail = {"literals": lv, "script": sv,
              "backtranslation": {"source": "REUSED from regen run _gen_meta",
                                  "ok": bt.get("ok"), "skipped": bt.get("skipped", True),
                                  "verdict": bt.get("verdict")},
              "digit_normalizations": n_dig, "normalizations": norm}
    return status, (text if status == "ok" else G.TODO_LLM), detail


def validate_nllb_rerun(key, text):
    """Confirm every D26 literal is present verbatim in the digit-normalised rerun text."""
    en = _en_goals()[0][key]
    t, n_dig = force_ascii_digits(text or "")
    lits = G.extract_literals(en, key)
    not_verbatim = [l for l in lits if G.literal_found(t, l) is None]
    status = "ok" if not not_verbatim else "NEEDS_FIX_literal_lost"
    return status, (t if status == "ok" else "TODO_MT"), {"literals": lits,
                                                          "not_verbatim": not_verbatim,
                                                          "digit_normalizations": n_dig}


def write_sums_and_coverage(pay):
    """Regenerate PAYLOAD_SHA256SUMS.txt and the coverage json (same format as
    check_payload_freeze), and return the coverage dict."""
    lines = ["# SHA-256 of every installed fast-track payload file (DECISIONS D26/D27).",
             "# Files named *.TODO_*.txt are placeholders (NOT payloads; the attack class refuses them).",
             "# CANONICAL lines: sha256 of json.dumps(payloads.json conditions[<cond>].goals,"
             " ensure_ascii=False, sort_keys=True, separators=(',',':')).encode('utf-8')."]
    for cond in CPF.FAST:
        for p in sorted((CPF.C / cond).glob("*")):
            if p.is_file():
                lines.append(f"{CPF.sha(p)}  {p.relative_to(CPF.C)}")
    for cond in CPF.FAST:
        p = CPF.C / "_wrappers" / f"W1__{cond}.txt"
        lines.append(f"{CPF.sha(p)}  {p.relative_to(CPF.C)}")
    for cond in CPF.FAST:
        lines.append(f"{CPF.canon(pay['conditions'][cond]['goals'])}  CANONICAL_GOALS_JSON({cond})")
    (CPF.C / "PAYLOAD_SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cov = {}
    for name in ("pairs_150.json", "pairs.json"):
        pairs = json.loads((ROOT / "subset" / name).read_text(encoding="utf-8"))["pairs"]
        tasks = sorted({f"{p['suite']}/{p['injection_task']}" for p in pairs})
        cov[name] = {"n_pairs": len(pairs), "n_injection_tasks": len(tasks), "per_condition": {}}
        for cond in CPF.FAST:
            g = pay["conditions"][cond]["goals"]
            w_ok = CPF.filled(pay["wrappers"]["W1"].get(cond))
            missing = [t for t in tasks if not CPF.filled(g.get(t))]
            covered = sum(1 for p in pairs
                          if CPF.filled(g.get(f"{p['suite']}/{p['injection_task']}")) and w_ok)
            cov[name]["per_condition"][cond] = {"pairs_covered": covered, "wrapper_W1": w_ok,
                                                "tasks_missing": missing,
                                                "pairs_missing": len(pairs) - covered}
        allc = sum(1 for p in pairs
                   if all(CPF.filled(pay["conditions"][c]["goals"].get(f"{p['suite']}/{p['injection_task']}"))
                          for c in CPF.FAST))
        cov[name]["pairs_covered_in_all_4"] = allc
    (CPF.C / "_provenance" / "2026-10-06_coverage.json").write_text(
        json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="./scratch/kaggle_out/gen_regen/llm_payloads",
                    help="unzipped llm_payloads.zip from bangla-pi-gen-llm-regen")
    ap.add_argument("--nllb-rerun-dir", default="./scratch/kaggle_out/nllb_rerun/bn_mt_nllb_rerun",
                    help="unzipped bn_mt_nllb_rerun.zip from bangla-pi-nllb-bn-mt-rerun")
    ap.add_argument("--date", default="2026-10-06")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    gen, nllb = Path(args.gen_dir), Path(args.nllb_rerun_dir)

    EN, order = _en_goals()
    pay = json.loads(PAY_PATH.read_text(encoding="utf-8"))
    report = {"run": dt.datetime.now().isoformat(timespec="seconds"), "dry_run": args.dry_run,
              "gen_dir": str(gen), "nllb_rerun_dir": str(nllb),
              "merged": {"bn_llm": [], "banglish_llm": [], "bn_mt": []},
              "rejected": {"bn_llm": {}, "banglish_llm": {}, "bn_mt": {}}}

    # ---- 1. LLM regen -----------------------------------------------------------------
    installed_meta = {}
    for cond in LLM_CONDS:
        mp = gen / "data" / "conditions" / cond / "_gen_meta.json"
        inst_path = COND_DIR / cond / "_gen_meta.json"
        installed_meta[cond] = json.loads(inst_path.read_text(encoding="utf-8"))
        if not mp.exists():
            continue
        regen = json.loads(mp.read_text(encoding="utf-8"))
        for key, it in regen["items"].items():
            cand = it.get("candidate_text") or it.get("text") or ""
            status, final, detail = validate_llm_candidate(cond, key, cand,
                                                           it.get("validation"), EN[key])
            if status == "ok":
                pay["conditions"][cond]["goals"][key] = final
                report["merged"][cond].append(key)
                installed_meta[cond]["items"][key] = {
                    **{k: v for k, v in it.items()},
                    "status": "ok", "text": final,
                    "merge_revalidation": detail,
                    "merged": args.date, "decision": "DECISIONS.md D27"}
            else:
                report["rejected"][cond][key] = {"status": status,
                                                 "missing": detail["literals"]["missing"],
                                                 "script_reasons": detail["script"]["reasons"]}

    # ---- 2. bn_mt rerun ---------------------------------------------------------------
    rerun_json = nllb / "bn_mt_nllb_rerun.json"
    if rerun_json.exists():
        rr = json.loads(rerun_json.read_text(encoding="utf-8"))
        for key, val in rr["goals"].items():
            if val == "TODO_MT":
                report["rejected"]["bn_mt"][key] = {"status": "TODO_MT (run)"}
                continue
            status, final, detail = validate_nllb_rerun(key, val)
            if status == "ok":
                pay["conditions"]["bn_mt"]["goals"][key] = final
                report["merged"]["bn_mt"].append(key)
            else:
                report["rejected"]["bn_mt"][key] = {"status": status,
                                                    "not_verbatim": detail["not_verbatim"]}

    # ---- 3. write, export, sums, coverage ---------------------------------------------
    if not args.dry_run:
        bak = PAY_PATH.with_suffix(".json.bak-20261006-regen")
        if not bak.exists():
            bak.write_text(PAY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        PAY_PATH.write_text(json.dumps(pay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for cond in LLM_CONDS:
            m = installed_meta[cond]
            m["filled"] = sum(1 for v in m["items"].values() if v.get("status") == "ok")
            (COND_DIR / cond / "_gen_meta.json").write_text(
                json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # re-export condition dirs from the merged payloads.json
        import export_condition_dirs as EXP  # noqa: E402
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            EXP.main()
        # copy rerun provenance
        prov = COND_DIR / "_provenance" / f"{args.date}_nllb33b_rerun"
        prov.mkdir(parents=True, exist_ok=True)
        for n in ("bn_mt_nllb_rerun.json", "mt_report_nllb_rerun.json", "SHA256SUMS.txt"):
            src = nllb / n
            if src.exists():
                (prov / n).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        gp = COND_DIR / "_provenance" / f"{args.date}_gemma_gen_regen"
        gp.mkdir(parents=True, exist_ok=True)
        for cond in LLM_CONDS:
            src = gen / "data" / "conditions" / cond / "_gen_meta.json"
            if src.exists():
                (gp / f"_gen_meta.{cond}.regen.json").write_text(src.read_text(encoding="utf-8"),
                                                                 encoding="utf-8")
        cov = write_sums_and_coverage(pay)
    else:
        cov = {}
        for name in ("pairs_150.json", "pairs.json"):
            pairs = json.loads((ROOT / "subset" / name).read_text(encoding="utf-8"))["pairs"]
            tasks = sorted({f"{p['suite']}/{p['injection_task']}" for p in pairs})
            cov[name] = {"n_pairs": len(pairs), "per_condition": {}}
            for cond in CPF.FAST:
                g = pay["conditions"][cond]["goals"]
                w_ok = CPF.filled(pay["wrappers"]["W1"].get(cond))
                covered = sum(1 for p in pairs
                              if CPF.filled(g.get(f"{p['suite']}/{p['injection_task']}")) and w_ok)
                cov[name]["per_condition"][cond] = {"pairs_covered": covered,
                                                    "pairs_missing": len(pairs) - covered}
            cov[name]["pairs_covered_in_all_4"] = sum(
                1 for p in pairs
                if all(CPF.filled(pay["conditions"][c]["goals"].get(f"{p['suite']}/{p['injection_task']}"))
                       for c in CPF.FAST))

    # ---- report -----------------------------------------------------------------------
    counts = {cond: sum(1 for v in pay["conditions"][cond]["goals"].values() if CPF.filled(v))
              for cond in CPF.FAST}
    report["counts_filled_of_35"] = counts
    report["coverage"] = cov
    print(json.dumps({"merged": {k: len(v) for k, v in report["merged"].items()},
                      "rejected": {k: list(v) for k, v in report["rejected"].items() if v},
                      "counts_filled_of_35": counts,
                      "coverage_150_all4": cov.get("pairs_150.json", {}).get("pairs_covered_in_all_4"),
                      "coverage_150_per_condition": {c: cov.get("pairs_150.json", {}).get("per_condition", {}).get(c, {}).get("pairs_covered")
                                                     for c in CPF.FAST}},
                     ensure_ascii=False, indent=2))
    for name in ("pairs_150.json", "pairs.json"):
        pc = cov.get(name, {}).get("per_condition", {})
        for cond in CPF.FAST:
            miss = pc.get(cond, {}).get("tasks_missing")
            if miss:
                print(f"  MISSING {name} {cond}: {miss}")
    if not args.dry_run:
        (COND_DIR / "_provenance" / f"{args.date}_merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
