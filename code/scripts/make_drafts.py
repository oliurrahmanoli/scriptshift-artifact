"""Generate data/drafts_for_native_review.json: MACHINE DRAFTS only.

These are NOT the native dataset. They are machine-produced starting points that a
native speaker must rewrite and validate (research-plan.md 4.3, 4.10). Nothing here
may be used as 'native' data. The drafts are:

  * draft_bn        : NLLB-200 machine translation (= the bn_mt condition).
  * draft_banglish  : rule-based/IndicXlit transliteration of draft_bn.
  * draft_codemix   : a heuristic keep-English-nouns pass over draft_bn (very rough).

Every record carries loud DO_NOT_USE_AS_NATIVE flags. The purpose is to save the
native writer typing, never to substitute for their work.
"""
import json
from pathlib import Path

from translit import transliterate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAY = ROOT / "data" / "payloads.json"
GOALS = ROOT / "data" / "goals_en.json"
OUT = ROOT / "data" / "drafts_for_native_review.json"

# English office/technical words that commonly stay English in Bangladeshi code-mixed
# writing. Heuristic only; the native writer decides what actually stays English.
KEEP_ENGLISH_HINTS = [
    "email", "password", "transaction", "IBAN", "account", "calendar", "event",
    "file", "security code", "subject", "link", "website", "Slack", "channel",
    "message", "reservation", "hotel", "passport", "credit card",
]


def main():
    pay = json.loads(PAY.read_text(encoding="utf-8"))
    goals = json.loads(GOALS.read_text(encoding="utf-8"))["goals"]
    en = pay["conditions"]["en"]["goals"]
    bn = pay["conditions"]["bn_mt"]["goals"]

    records = {}
    for key in en:
        draft_bn = bn.get(key, "")
        draft_banglish, eng = transliterate(draft_bn) if draft_bn not in ("", "TODO_MT") else ("", "n/a")
        records[key] = {
            "suite": goals[key]["suite"],
            "scoring_group": goals[key]["scoring_group"],
            "goal_en": en[key],
            "draft_bn_FROM_NLLB_MT": draft_bn,
            "draft_banglish_FROM_translit": draft_banglish,
            "draft_banglish_engine": eng,
            "draft_codemix_HEURISTIC": draft_bn,  # writer code-mixes from here
            "review_status": "DRAFT_UNREVIEWED",
            "must_do": "Native speaker REWRITES bn_native, banglish_native, codemix_native "
                       "from goal_en (not from these drafts) and validates per section 4.3.",
        }

    payload = {
        "WARNING": "MACHINE DRAFTS ONLY. NOT native data. Do not use as bn_native / "
                   "banglish_native / codemix_native. Native speakers must rewrite and "
                   "validate every item (research-plan.md 4.3 and 4.10).",
        "draft_sources": {
            "draft_bn_FROM_NLLB_MT": "facebook/nllb-200-distilled-600M (same as bn_mt)",
            "draft_banglish_FROM_translit": "scripts/translit.py (IndicXlit or rule-based fallback)",
            "draft_codemix_HEURISTIC": "copy of draft_bn; writer code-mixes manually",
        },
        "keep_english_hints": KEEP_ENGLISH_HINTS,
        "n_items": len(records),
        "items": records,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {len(records)} DRAFT records (clearly labelled unreviewed).")


if __name__ == "__main__":
    main()
