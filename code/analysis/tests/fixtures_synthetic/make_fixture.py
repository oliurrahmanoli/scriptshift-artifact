"""Generate the SYNTHETIC analysis fixture (planted effects; NOT RESULTS). See README.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[3]
MODELS = ["SYNTH-model-A", "SYNTH-model-B", "SYNTH-model-C"]
P = {"en": .30, "bn_mt": .30, "bn_native": .45, "banglish_native": .30, "banglish_translit": .30,
     "codemix_native": .35, "en_W2": .40, "bn_native_W2": .55, "banglish_native_W2": .30, "bn_mt_nllb": .30}
COMP = {"en": .95, "bn_mt": .85, "bn_native": .9, "banglish_native": .75, "banglish_translit": .75,
        "codemix_native": .85, "en_W2": .95, "bn_native_W2": .9, "banglish_native_W2": .75, "bn_mt_nllb": .85}
BANGLA = {c for c in P if c.startswith(("bn", "banglish", "codemix"))}
USERS_PER_INJ = 4


def _trace(model, suite, ut, inj, cond, utility, security, bengali, parse_err, rng):
    final = ("কাজটি সম্পন্ন হয়েছে।" if bengali else "Done, the task is complete.")
    if rng.random() < .03:
        final += " This looks like a prompt injection, I will not comply."
    msgs = [
        {"role": "system", "content": [{"type": "text", "content": "SYNTHETIC system prompt"}]},
        {"role": "user", "content": [{"type": "text", "content": "SYNTHETIC user task"}]},
        {"role": "assistant", "content": [{"type": "text", "content": ""}],
         "tool_calls": [{"function": "send_money", "args": {"recipient": "SYNTH-IBAN",
                                                            "subject": "বার্তা" if bengali else "msg"},
                         "id": "x", "placeholder_args": None}]},
        {"role": "tool", "content": [{"type": "text", "content": "ok"}], "tool_call_id": "x",
         "tool_call": None, "error": "EmptyFunctionCall" if parse_err else None},
        {"role": "assistant", "content": [{"type": "text", "content": final}], "tool_calls": None},
    ]
    return {"suite_name": suite, "pipeline_name": model, "user_task_id": ut,
            "injection_task_id": None if cond == "none" else inj,
            "attack_type": None if cond == "none" else cond, "injections": {}, "messages": msgs,
            "error": None, "benchmark_version": "v1.2.2", "evaluation_timestamp": "SYNTHETIC",
            "agentdojo_package_version": "0.1.35", "utility": bool(utility),
            "security": bool(security) if cond != "none" else True, "duration": 1.0, "synthetic": True}


def make(out: Path, seed: int = 7) -> Path:
    out = Path(out)
    rng = np.random.default_rng(seed)
    goals = json.loads((CODE / "data" / "goals_en.json").read_text())["goals"]
    pairs = json.loads((CODE / "subset" / "pairs.json").read_text())["pairs"]
    by_inj = {}
    for p in pairs:
        by_inj.setdefault((p["suite"], p["injection_task"]), []).append(p["user_task"])
    sub = [(s, u, i) for (s, i), us in by_inj.items() for u in us[:USERS_PER_INJ]]
    extra = [(s, f"user_task_9{k:02d}", i) for (s, i) in by_inj for k in range(3)]  # outside subset
    (out / "scores").mkdir(parents=True, exist_ok=True)
    scores, judge, probe = {}, [], []
    comp_tab = {}

    def write_trace(d):
        f = out / "runs" / d["pipeline_name"] / d["suite_name"] / d["user_task_id"] / (d["attack_type"] or "none") / f"{d['injection_task_id'] or 'none'}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(d, ensure_ascii=False))

    for m_i, model in enumerate(MODELS):
        # probe: one comprehension outcome per (cond, inj), two prompt variants
        for cond in P:
            for (s, i) in by_inj:
                if cond == "banglish_translit":      # C2 pair shares comprehension (planted null)
                    c1, c2 = comp_tab[(model, "banglish_native", s, i, "raw")]
                else:
                    c1 = rng.random() < COMP[cond]
                    c2 = c1 if rng.random() < .9 else not c1
                comp_tab[(model, cond, s, i, "raw")] = (c1, c2)
                comp_tab[(model, cond, s, i)] = c1 and c2
                for pv, ok in (("p1", c1), ("p2", c2)):
                    probe.append({"model": model, "condition": cond, "suite": s, "injection_task": i,
                                  "prompt_variant": pv, "q1_correct": ok, "q2_correct": ok,
                                  "q3_correct": ok, "synthetic": True})
        # no-attack utility episodes
        for (s, u) in sorted({(s, u) for s, u, _ in sub}):
            write_trace(_trace(model, s, u, "none", "none", rng.random() < .7, True, False, rng.random() < .01, rng))
        runs = [(model, "none", list(P), sub + (extra if model == "SYNTH-model-A" else []))]
        if model == "SYNTH-model-A":
            runs.append((model + "-tool_filter", "tool_filter", ["en", "bn_mt", "bn_native", "banglish_native"], sub))
            runs.append((model + "-rep1", "none", ["en", "bn_native"], sub))
        lat_tab = {}
        for label, defense, conds, plist in runs:
            for (s, u, i) in plist:
                # shared latent across conditions AND across runs of the same model (pairing)
                lat = lat_tab.setdefault((s, u, i), rng.random())
                grp = goals[f"{s}/{i}"]["scoring_group"]
                for cond in conds:
                    if (s, u, i) in extra and cond not in ("en", "bn_mt", "bn_native"):
                        continue
                    lat_c = lat if rng.random() > .05 else rng.random()   # ~5 % independent redraw
                    p = P[cond] * (.3 if defense == "tool_filter" else 1.0)
                    comp = comp_tab[(model, cond, s, i)]
                    norm = lat_c < (p if comp else p * .2)
                    raw = norm and not (grp == "C" and cond in BANGLA and rng.random() < .5)
                    raw = raw or (s == "banking" and i in ("injection_task_0", "injection_task_1", "injection_task_3") and rng.random() < .2)
                    corrected = raw and not (s == "banking" and i in ("injection_task_0", "injection_task_1", "injection_task_3") and rng.random() < .4)
                    util = rng.random() < (.6 if not norm else .45)
                    beng = cond in BANGLA and cond != "banglish_native" and rng.random() < .4
                    perr = rng.random() < .01
                    tlabel = label
                    write_trace(_trace(tlabel, s, u, i, cond, util, raw, beng, perr, rng))
                    needs = (not norm) and (beng or rng.random() < .1)
                    rec = {"suite": s, "user_task": u, "injection_task": i, "condition": cond, "status": "ok",
                           "error": None, "utility": bool(util), "raw_security": bool(raw),
                           "normalised_security": bool(norm), "corrected_security": bool(corrected),
                           "scoring_group": grp, "bengali_share_output": .9 if beng else 0.0,
                           "bengali_share_args": .6 if beng else 0.0, "needs_judge": bool(needs),
                           "n_tool_calls": 1}
                    scores.setdefault((label, cond, defense), []).append(rec)
                    if needs:
                        v = "full" if rng.random() < .3 else ("partial" if rng.random() < .3 else "no")
                        judge.append({"model": label.replace("-tool_filter", ""), "defense": defense,
                                      "suite": s, "user_task": u, "injection_task": i,
                                      "condition": cond, "rep": 0, "verdict": v,
                                      "reason": "SYNTHETIC", "judge_model": "SYNTH-judge", "synthetic": True})
    for (label, cond, defense), recs in scores.items():
        f = out / "scores" / f"{label}_{cond}.jsonl"
        head = {"_header": True, "pipeline": f"local/{label}.gguf" if defense == "none" else f"local/{label}",
                "condition": cond, "defense": None if defense == "none" else defense, "synthetic": True}
        f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [head] + recs) + "\n")
    (out / "judge.jsonl").write_text("\n".join(json.dumps(r) for r in judge) + "\n")
    (out / "probe.jsonl").write_text("\n".join(json.dumps(r) for r in probe) + "\n")
    # offline detectors (ASSUMED schema)
    rows = ["detector,condition,item_id,label,score,synthetic"]
    for det, shift in (("SYNTH-D1", 0.0), ("SYNTH-D4", 0.15)):
        for cond in ["en", "bn_mt", "bn_native", "banglish_native", "banglish_translit", "codemix_native", "en_W2", "bn_native_W2", "banglish_native_W2"]:
            mu = .85 if cond.startswith("en") else (.55 + shift if cond in ("bn_mt",) else .4 + shift)
            for k in range(35):
                rows.append(f"{det},{cond},{k},1,{min(1, max(0, rng.normal(mu, .15))):.4f},True")
            for k in range(60):
                rows.append(f"{det},{cond},b{k},0,{min(1, max(0, rng.normal(.2, .12))):.4f},True")
    (out / "detectors.csv").write_text("\n".join(rows) + "\n")
    # human audit (ASSUMED schema)
    aud = ["model,defense,suite,user_task,injection_task,condition,rep,judge_model,judge_verdict,human_a,human_b"]
    for j in judge[:120]:
        ha = j["verdict"] if rng.random() < .85 else "no"
        hb = ha if rng.random() < .9 else "partial"
        aud.append(",".join(str(j[k]) for k in ("model", "defense", "suite", "user_task", "injection_task", "condition", "rep", "judge_model", "verdict")) + f",{ha},{hb}")
    (out / "audit.csv").write_text("\n".join(aud) + "\n")
    # validator ratings (ASSUMED schema)
    rat = ["item_key,condition,rater,fidelity,adequacy,naturalness,register"]
    for cond in ["bn_mt", "bn_native", "banglish_native"]:
        for (s, i) in by_inj:
            base = int(rng.integers(3, 6))
            for r in ("V1", "V2"):
                adj = base if rng.random() < .8 else max(1, base - 1)
                rat.append(f"{s}/{i},{cond},{r},{int(rng.random() < .95)},{adj},{adj},formal")
    (out / "ratings.csv").write_text("\n".join(rat) + "\n")
    (out / "SYNTHETIC_NOT_RESULTS.txt").write_text("Synthetic fixture with planted effects. Not results.\n")
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_fixture")))
