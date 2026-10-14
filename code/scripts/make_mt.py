"""Produce the bn_mt condition with NLLB-200 (facebook/nllb-200-distilled-600M).

Implements the literal-preservation rule (research-plan.md 4.3): identifiers and
quoted attacker literals are swapped for placeholder tokens before translation and
restored afterwards. Any placeholder that fails to round-trip is logged and the
affected item is left as TODO_MT so it can be fixed by hand (never silently wrong).

Writes:
  * data/payloads.json      (fills condition bn_mt, if available)
  * data/mt_report.json     (per-item engine, date, protected-literal audit)

Usage:
  MT venv:  .venv-mt/bin/python scripts/make_mt.py --model facebook/nllb-200-distilled-600M
The CC-BY-NC licence of NLLB-200 applies to the generated text (released
separately under CC BY-NC 4.0 per section 4.3).
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAY = ROOT / "data" / "payloads.json"
REPORT = ROOT / "data" / "mt_report.json"

# Literal patterns to protect (ASCII-only entities). Order matters.
PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    # URL: must not end in sentence punctuation (. , ; : ! ? or the Bangla danda U+0964),
    # so "... to www.my-website-234.com/random." yields "www.my-website-234.com/random"
    # (the old greedy class swallowed the full stop; slack/injection_task_2,3,4). D26.
    ("URL", re.compile(r"(?:https?://|www\.)[^\s'\"]*[^\s'\".,;:!?\u0964]")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{6,}\b")),
    # QUOTED: opening quote must NOT follow a letter, so in-word apostrophes
    # (wife's, it's) are not mistaken for an opening delimiter (fixes the
    # over-long travel/injection_task_5 match). DECISIONS D24.
    ("QUOTED", re.compile(r"(?<![A-Za-z])'[^']*'")),
    ("MONEY", re.compile(r"\$[\d,]+(?:\.\d+)?")),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("TIME", re.compile(r"\b\d{1,2}:\d{2}\b")),
]


# ASCII <- Bengali digits (U+09E6..U+09EF). research-plan.md §4.3 mandates Latin digits,
# never Bengali digits, in every rendered condition (D27). This is a deterministic 1:1 map
# applied as a post-processing step to the RESTORED text (so sentinels/literals, which are
# already ASCII, are untouched) and to installed payloads when the rule is applied uniformly.
_BN2ASCII = str.maketrans("\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef",
                          "0123456789")


def force_ascii_digits(text):
    """Map Bengali digits (U+09E6..U+09EF) to ASCII 0-9. Returns (text, n_replaced)."""
    n = sum(1 for c in text if "\u09e6" <= c <= "\u09ef")
    return text.translate(_BN2ASCII), n


def protect(text):
    literals = {}
    i = 0

    _NESTED = re.compile(r" (LLZ\d+ZLL) ")

    def repl(kind):
        def f(m):
            nonlocal i
            i += 1
            tok = f" LLZ{i}ZLL "   # ascii sentinel that survives sentencepiece
            # Strip one pad space on each side of any sentinel already nested inside
            # this match, so restoring an inner literal does not leave inner spaces.
            literals[tok.strip()] = _NESTED.sub(r"\1", m.group(0))
            return tok
        return f

    for kind, pat in PATTERNS:
        text = pat.sub(repl(kind), text)
    return text, literals


def restore(text, literals):
    # A sentinel the MT model DROPPED (absent from the output and not nested inside
    # another literal) is a lost literal too. The old check only caught sentinels left
    # un-substituted, so a dropped one passed silently (bn_mt workspace/injection_task_13
    # lost https://bit.ly/sUf7tOJ this way). D26.
    nested_in_values = " ".join(literals.values())
    dropped = [tok for tok in literals if tok not in text and tok not in nested_in_values]
    # Sentinels can nest (e.g. an email inside a quoted string), so restore to a
    # fixpoint: keep substituting until no known sentinel remains.
    for _ in range(len(literals) + 2):
        changed = False
        for tok, val in literals.items():
            if tok in text:
                text = text.replace(tok, val); changed = True
        if not changed:
            break
    missing = [literals[tok] for tok in literals if tok in text] + [literals[t] for t in dropped]
    text = re.sub(r"\s+", " ", text).strip()
    # Tidy a space introduced before ASCII punctuation by sentinel padding
    # (e.g. "janeLong@google.com , use" -> "janeLong@google.com, use"). DECISIONS D24.
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--src", default="eng_Latn")
    ap.add_argument("--tgt", default="ben_Beng")
    ap.add_argument("--cache", default=str(ROOT.parent / ".cache" / "hf"))
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache, src_lang=args.src)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache)
    model.eval()
    bos = tok.convert_tokens_to_ids(args.tgt)

    def translate(text):
        prot, literals = protect(text)
        enc = tok(prot, return_tensors="pt", truncation=True, max_length=args.max_new)
        with torch.no_grad():
            out = model.generate(
                **enc, forced_bos_token_id=bos, num_beams=args.beam,
                max_new_tokens=args.max_new,
            )
        raw = tok.batch_decode(out, skip_special_tokens=True)[0]
        restored, missing = restore(raw, literals)
        return restored, raw, literals, missing

    pay = json.loads(PAY.read_text(encoding="utf-8"))
    order = list(pay["conditions"]["en"]["goals"].keys())
    report = {"model": args.model, "src": args.src, "tgt": args.tgt,
              "run_date": dt.date.today().isoformat(), "beam": args.beam, "items": {}}

    filled = 0
    for key in order:
        en = pay["conditions"]["en"]["goals"][key]
        bn, raw, literals, missing = translate(en)
        if not missing:
            bn, _ndig = force_ascii_digits(bn)  # D27: §4.3 ASCII digits, never Bengali digits
        if missing:
            pay["conditions"]["bn_mt"]["goals"][key] = "TODO_MT"
            status = "NEEDS_FIX_literal_lost"
        else:
            pay["conditions"]["bn_mt"]["goals"][key] = bn
            status = "ok"
            filled += 1
        report["items"][key] = {
            "status": status, "protected_literals": list(literals.values()),
            "missing": [literals[m] for m in missing], "bn_mt": bn,
        }

    # translate the W1 wrapper too (protect placeholders {goal}/{user}/{model})
    w1 = pay["wrappers"]["W1"]["en"]
    w1_prot = w1.replace("{goal}", " LLGOALLL ").replace("{user}", " LLUSERLL ").replace("{model}", " LLMODELLL ")
    enc = tok(w1_prot, return_tensors="pt", truncation=True, max_length=512)
    import torch as _t
    with _t.no_grad():
        o = model.generate(**enc, forced_bos_token_id=bos, num_beams=args.beam, max_new_tokens=512)
    w1_bn = tok.batch_decode(o, skip_special_tokens=True)[0]
    w1_bn = (w1_bn.replace("LLGOALLL", "{goal}").replace("LLUSERLL", "{user}")
             .replace("LLMODELLL", "{model}"))
    if all(p in w1_bn for p in ("{goal}", "{user}", "{model}")):
        pay["wrappers"]["W1"]["bn_mt"] = w1_bn
        report["wrapper_W1_bn_mt"] = "ok"
    else:
        report["wrapper_W1_bn_mt"] = f"NEEDS_FIX placeholders lost: {w1_bn!r}"

    PAY.write_text(json.dumps(pay, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bn_mt filled: {filled}/{len(order)}  (wrapper: {report['wrapper_W1_bn_mt'][:40]})")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
