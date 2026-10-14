"""Freeze the fast-track payload sets (DECISIONS D26): bn_mt (NLLB-200-3.3B) + bn_llm /
banglish_llm (Gemma-4-26B generator), from the downloaded Kaggle outputs.

Deterministic, box-only (no GPU, no network). Steps:
  0. verify both downloads against their SHA256SUMS.txt (incl. the NLLB canonical-goals hash);
  1. bn_mt audit: re-derive each item's protected literals with the AS-RUN make_mt.py (git
     commit --asrun-commit), re-check the NLLB raw output with the FIXED restore() (which also
     catches DROPPED sentinels) and confirm the restored text equals the shipped text; a lost
     literal => TODO_MT (the documented MT rule, make_mt.py docstring). Report-only: which of
     the LLM validator's literals (D26 rules) are not verbatim in bn_mt;
  2. LLM revalidation of ALL 70 candidates with the FIXED literal + script validators
     (payloads/generate_llm.py, D26). Back-translation verdicts are NOT recomputed: they are
     reused from the Kaggle run, and only where the recorded validation provably belongs to
     the candidate text (recorded literal + script results reproduce exactly on it);
  3. install into data/conditions/ (bn_mt, bn_llm, banglish_llm, W1 wrappers, _gen_meta.json)
     and copy provenance into data/conditions/_provenance/.
Then run:  import_condition_dirs.py --sync-todo ; export_condition_dirs.py ;
           check_payload_freeze.py   (SHA256SUMS + coverage + round-trip checks).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from payloads import generate_llm as G  # noqa: E402
import make_mt as MT_FIXED  # noqa: E402
from export_condition_dirs import en_header  # noqa: E402

COND_DIR = ROOT / "data" / "conditions"
LLM_CONDS = ["bn_llm", "banglish_llm"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(d) -> str:
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_sums(root: Path, extra_canonical=None) -> dict:
    ok, bad, virtual = 0, [], []
    for ln in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not ln.strip():
            continue
        h, name = ln.split("  ", 1)
        if name.startswith("CANONICAL_GOALS_JSON("):
            got = extra_canonical() if extra_canonical else None
            virtual.append({"line": name, "expected": h, "recomputed": got, "match": got == h})
            continue
        if sha256_file(root / name) == h:
            ok += 1
        else:
            bad.append(name)
    return {"files_ok": ok, "files_bad": bad, "virtual": virtual}


def load_asrun_make_mt(commit: str):
    src = subprocess.run(["git", "show", f"{commit}:code/scripts/make_mt.py"], cwd=ROOT.parent,
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "make_mt_asrun.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_mt_asrun", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def legacy_extract(goal_en, patterns):
    found = []
    for _k, pat in patterns:
        for m in pat.finditer(goal_en):
            if m.group(0) not in found:
                found.append(m.group(0))
    return found


def legacy_strip(text, literals):
    out = text
    for lit in sorted(literals, key=len, reverse=True):
        out = out.replace(lit, " ")
    return out


def filled_file(cond, key, goal_en, text):
    return (f"# condition: {cond}\n# key: {key}\n# STATUS: filled\n"
            f"{en_header(goal_en)}\n\n{text}\n")


def todo_file(cond, key, goal_en, sentinel, reason):
    return (f"# condition: {cond}\n# key: {key}\n# STATUS: {sentinel}\n# REASON: {reason}\n"
            f"{en_header(goal_en)}\n#\n{sentinel}\n")


def wrapper_file(wid, cond, text):
    return (f"# Wrapper {wid} for condition={cond}\n# Placeholders: {{goal}}, {{user}}, {{model}}\n"
            f"# STATUS: filled\n\n{text if text.endswith(chr(10)) else text + chr(10)}")


def kaggle_body(path: Path) -> str:
    """Body of a Kaggle-written file: everything after the first blank line that follows
    the '# EN:' header block (the Kaggle writer does not comment multi-line EN goals)."""
    raw = path.read_text(encoding="utf-8")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="./scratch/kaggle_out/gen/llm_payloads")
    ap.add_argument("--nllb-dir", default="./scratch/kaggle_out/nllb/bn_mt_nllb-3.3b")
    ap.add_argument("--gen-zip", default="./downloads/llm_payloads.zip")
    ap.add_argument("--nllb-zip", default="./downloads/bn_mt_nllb-3.3b.zip")
    ap.add_argument("--server-log", default="./downloads/server_8099_swa_full+no_think.log")
    ap.add_argument("--asrun-commit", default="491b4a8",
                    help="commit whose make_mt.py / generate_llm.py were embedded in the run notebooks")
    ap.add_argument("--date", default="2026-10-06")
    args = ap.parse_args()
    gen, nllb = Path(args.gen_dir), Path(args.nllb_dir)

    goals = json.loads((ROOT / "data" / "goals_en.json").read_text(encoding="utf-8"))
    order = goals["order"]
    EN = {k: goals["goals"][k]["goal_en"] for k in order}
    report = {"run": dt.datetime.now().isoformat(timespec="seconds"), "asrun_commit": args.asrun_commit,
              "generate_llm_sha256_fixed": sha256_file(ROOT / "payloads" / "generate_llm.py"),
              "make_mt_sha256_fixed": sha256_file(ROOT / "scripts" / "make_mt.py")}

    # ---- 0. input integrity ------------------------------------------------------------
    nllb_json = json.loads((nllb / "bn_mt_nllb-3.3b.json").read_text(encoding="utf-8"))
    report["inputs"] = {
        "gemma_gen_v1": {"zip": args.gen_zip, "zip_sha256": sha256_file(Path(args.gen_zip)),
                         "sums": verify_sums(gen)},
        "nllb33b_v1": {"zip": args.nllb_zip, "zip_sha256": sha256_file(Path(args.nllb_zip)),
                       "sums": verify_sums(nllb, lambda: canonical_sha(nllb_json["goals"]))},
        "server_log_sha256": sha256_file(Path(args.server_log)),
    }
    for k, v in report["inputs"].items():
        if isinstance(v, dict):
            assert not v["sums"]["files_bad"], f"{k}: checksum mismatch {v['sums']['files_bad']}"
            assert all(x["match"] for x in v["sums"]["virtual"]), f"{k}: canonical hash mismatch"

    # ---- 1. bn_mt audit -----------------------------------------------------------------
    MT_OLD, mt_old_sha = load_asrun_make_mt(args.asrun_commit)
    report["make_mt_sha256_asrun"] = mt_old_sha
    mt_rep = json.loads((nllb / "mt_report_nllb-3.3b.json").read_text(encoding="utf-8"))
    bn_mt, bn_mt_audit = {}, {}
    for key in order:
        it = mt_rep["items"][key]
        _prot, lit_map = MT_OLD.protect(EN[key])
        assert list(lit_map.values()) == it["protected_literals"], f"bn_mt {key}: literal map differs"
        restored, missing = MT_FIXED.restore(it["raw_mt"], lit_map)
        assert restored == it["bn_mt"] == nllb_json["goals"][key], f"bn_mt {key}: restore differs"
        new_lits = G.extract_literals(EN[key], key)
        not_verbatim = [l for l in new_lits if G.literal_found(restored, l) is None]
        status = "ok" if not missing else "NEEDS_FIX_literal_lost"
        bn_mt[key] = restored if status == "ok" else "TODO_MT"
        bn_mt_audit[key] = {"kaggle_status": it["status"], "status": status,
                            "protected_literals_asrun": it["protected_literals"],
                            "lost_literals_fixed_restore": missing,
                            "llm_rule_literals_not_verbatim_REPORT_ONLY": not_verbatim}
    report["bn_mt"] = {"filled": sum(v != "TODO_MT" for v in bn_mt.values()), "n": len(order),
                       "kaggle_filled": mt_rep["filled"],
                       "withdrawn": [k for k, v in bn_mt.items() if v == "TODO_MT"],
                       "llm_rule_not_verbatim_REPORT_ONLY": {k: v["llm_rule_literals_not_verbatim_REPORT_ONLY"]
                                                            for k, v in bn_mt_audit.items()
                                                            if v["llm_rule_literals_not_verbatim_REPORT_ONLY"]}}

    # ---- 2. LLM revalidation (all 70) -----------------------------------------------------
    llm_final, llm_meta = {}, {}
    for cond in LLM_CONDS:
        kmeta = json.loads((gen / "data" / "conditions" / cond / "_gen_meta.json").read_text(encoding="utf-8"))
        items, final = {}, {}
        for key in order:
            it = kmeta["items"][key]
            cand = it["candidate_text"]
            rec = it["validation"]
            safe = key.replace("/", "__")
            # (a) does the recorded validation belong to this candidate text?
            old_lits = it["literals"]
            assert old_lits == legacy_extract(EN[key], MT_OLD.PATTERNS), f"{cond} {key}: literal drift"
            leg_missing = [l for l in old_lits if l not in cand]
            leg_shares = G.script_shares(legacy_strip(cand, old_lits))
            consistent = (leg_missing == rec["literals"]["missing"] and leg_shares == rec["script"]["shares"])
            if it["status"] == "ok":
                f = gen / "data" / "conditions" / cond / f"{safe}.txt"
                file_ok = f.read_text(encoding="utf-8").endswith("\n\n" + cand + "\n")
                assert consistent and file_ok, f"{cond} {key}: accepted item inconsistent"
            bt = rec.get("backtranslation") or {}
            bt_applies = consistent and not bt.get("skipped", True)
            # (b) fixed validators on the candidate (with logged digit normalisation)
            lits = G.extract_literals(EN[key], key)
            text, norm = G.normalize_literal_digits(cand, lits)
            lv = G.validate_literals(text, lits)
            sv = G.validate_script(text, lits, cond)
            if lv["ok"] and sv["ok"] and bt_applies and bt.get("ok"):
                status = "ok"
            elif lv["ok"] and sv["ok"] and not bt_applies:
                status = "NEEDS_REVIEW_backtranslation_not_recorded_for_candidate"
            else:
                status = "NEEDS_REVIEW_validation_failed"
            final[key] = text if status == "ok" else G.TODO_LLM
            items[key] = {
                "status": status,
                "text": final[key],
                "candidate_text": cand,
                "normalized_text": text if norm else None,
                "normalizations": norm,
                "literals": lits,
                "literal_override": G.literal_override_reason(key),
                "revalidation": {
                    "literals": lv, "script": sv,
                    "backtranslation": {
                        "source": "REUSED from Kaggle run (Gemma self back-translation + self-judge); not recomputed",
                        "applies_to_candidate": bt_applies,
                        "ok": bt.get("ok") if bt_applies else None,
                        "back_en": bt.get("back_en") if bt_applies else None,
                        "verdict": bt.get("verdict") if bt_applies else None,
                        "note": None if bt_applies else (
                            "Kaggle generate_item kept attempt-0 text as candidate_text but recorded the "
                            "LAST attempt's validation (bookkeeping bug fixed in D26); the recorded verdict "
                            "belongs to a different, unsaved text, so no verdict exists for this candidate."),
                    },
                },
                "kaggle": it,  # the original Kaggle record, verbatim
            }
        llm_final[cond] = final
        top = {k: v for k, v in kmeta.items() if k != "items"}
        n_ok = sum(1 for v in items.values() if v["status"] == "ok")
        llm_meta[cond] = {**top, "kaggle_filled": kmeta["filled"], "filled": n_ok,
                          "frozen": dt.date.today().isoformat(), "decision": "DECISIONS.md D26",
                          "revalidated_with": {"generate_llm_sha256": report["generate_llm_sha256_fixed"],
                                               "make_mt_sha256": report["make_mt_sha256_fixed"]},
                          "items": items}
        report[cond] = {"kaggle_filled": kmeta["filled"], "filled": n_ok, "n": len(order),
                        "previously_accepted_still_ok": sum(1 for v in items.values()
                                                            if v["kaggle"]["status"] == "ok" and v["status"] == "ok"),
                        "previously_accepted": sum(1 for v in items.values() if v["kaggle"]["status"] == "ok"),
                        "pending": {k: v["status"] for k, v in items.items() if v["status"] != "ok"},
                        "normalized": {k: v["normalizations"] for k, v in items.items() if v["normalizations"]},
                        "relaxed_matches": {k: v["revalidation"]["literals"]["matched_relaxed"]
                                            for k, v in items.items() if v["revalidation"]["literals"]["matched_relaxed"]},
                        "fixed_vs_kaggle_validator": {k: {"kaggle_missing": v["kaggle"]["validation"]["literals"]["missing"],
                                                          "kaggle_script_reasons": v["kaggle"]["validation"]["script"]["reasons"],
                                                          "now_literals_ok": v["revalidation"]["literals"]["ok"],
                                                          "now_script_ok": v["revalidation"]["script"]["ok"],
                                                          "now_bengali_share": round(v["revalidation"]["script"]["shares"]["bengali"], 3)}
                                                      for k, v in items.items() if v["kaggle"]["status"] != "ok"}}

    # ---- 3. install ---------------------------------------------------------------------
    for cond, final in [("bn_mt", bn_mt)] + [(c, llm_final[c]) for c in LLM_CONDS]:
        cdir = COND_DIR / cond
        for old in cdir.glob("*.txt"):
            old.unlink()
        for key in order:
            safe = key.replace("/", "__")
            val = final[key]
            if val.startswith("TODO_"):
                reason = (bn_mt_audit[key]["status"] + " lost=" + json.dumps(bn_mt_audit[key]["lost_literals_fixed_restore"], ensure_ascii=False)
                          if cond == "bn_mt" else llm_meta[cond]["items"][key]["status"])
                (cdir / f"{safe}.{val}.txt").write_text(todo_file(cond, key, EN[key], val, reason), encoding="utf-8")
            else:
                (cdir / f"{safe}.txt").write_text(filled_file(cond, key, EN[key], val), encoding="utf-8")
    for cond in LLM_CONDS:
        (COND_DIR / cond / "_gen_meta.json").write_text(
            json.dumps(llm_meta[cond], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wdir = COND_DIR / "_wrappers"
    w_bnmt = nllb_json["wrapper_W1"]
    assert all(p in w_bnmt for p in ("{goal}", "{user}", "{model}"))
    (wdir / "W1__bn_mt.txt").write_text(wrapper_file("W1", "bn_mt", w_bnmt), encoding="utf-8")
    for cond in LLM_CONDS:
        src = (gen / "data" / "conditions" / "_wrappers" / f"W1__{cond}.txt").read_text(encoding="utf-8")
        assert "# STATUS: filled\n\n" in src
        body = src.split("# STATUS: filled\n\n", 1)[1].rstrip("\n")
        assert all(p in body for p in ("{goal}", "{user}", "{model}"))
        (wdir / f"W1__{cond}.txt").write_text(wrapper_file("W1", cond, body), encoding="utf-8")

    # provenance
    pg = COND_DIR / "_provenance" / f"{args.date}_gemma_gen_v1"
    pn = COND_DIR / "_provenance" / f"{args.date}_nllb33b_v1"
    for d in (pg, pn):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gen / "gen_report.json", pg / "gen_report.json")
    shutil.copy2(gen / "SHA256SUMS.txt", pg / "SHA256SUMS.txt")
    shutil.copy2(args.server_log, pg / Path(args.server_log).name)
    shutil.copy2(args.gen_zip, pg / Path(args.gen_zip).name)
    for cond in LLM_CONDS:
        shutil.copy2(gen / "data" / "conditions" / cond / "_gen_meta.json", pg / f"_gen_meta.{cond}.kaggle_original.json")
    for n in ("SHA256SUMS.txt", "mt_report_nllb-3.3b.json", "bn_mt_nllb-3.3b.json"):
        shutil.copy2(nllb / n, pn / n)
    shutil.copy2(args.nllb_zip, pn / Path(args.nllb_zip).name)
    (pn / "bn_mt_audit.json").write_text(json.dumps(bn_mt_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for d, z in ((pg, args.gen_zip), (pn, args.nllb_zip)):
        (d / "ORIGINAL_ZIP_SHA256.txt").write_text(f"{sha256_file(Path(z))}  {Path(z).name}\n")
    (COND_DIR / "_provenance" / f"{args.date}_freeze_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {c: {k: report[c][k] for k in ("kaggle_filled", "filled", "n")} for c in ["bn_mt"] + LLM_CONDS}
    print(json.dumps(summary))
    for c in LLM_CONDS:
        print(c, "previously accepted still ok:", report[c]["previously_accepted_still_ok"], "/", report[c]["previously_accepted"])
        for k, s in report[c]["pending"].items():
            print("  PENDING", c, k, s)
    print("bn_mt withdrawn:", report["bn_mt"]["withdrawn"])
    print("bn_mt LLM-rule literals not verbatim (report only):", json.dumps(report["bn_mt"]["llm_rule_not_verbatim_REPORT_ONLY"], ensure_ascii=False))


if __name__ == "__main__":
    main()
