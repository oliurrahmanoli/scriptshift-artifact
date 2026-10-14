"""Fill the banglish_translit condition by transliterating bn_native (section 4.3).

banglish_translit = machine transliteration of the *human-authored* bn_native text,
word by word (top-1), with ASCII literals passed through unchanged. Until bn_native
exists it stays TODO_TRANSLIT. Engine: IndicXlit if available, else the documented
rule-based fallback in scripts/translit.py (recorded per item).

Usage:  python scripts/make_translit.py
"""
import json
from pathlib import Path

from translit import transliterate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAY = ROOT / "data" / "payloads.json"
REPORT = ROOT / "data" / "translit_report.json"
TODO = ("TODO_NATIVE", "TODO_MT", "TODO_TRANSLIT", "")


def main():
    pay = json.loads(PAY.read_text(encoding="utf-8"))
    src = pay["conditions"]["bn_native"]["goals"]
    dst = pay["conditions"]["banglish_translit"]["goals"]
    report = {"engine_per_item": {}, "filled": 0, "skipped_todo": 0}
    for key, bn in src.items():
        if bn in TODO:
            dst[key] = "TODO_TRANSLIT"
            report["skipped_todo"] += 1
            continue
        roman, eng = transliterate(bn)
        dst[key] = roman
        report["engine_per_item"][key] = eng
        report["filled"] += 1
    # wrapper
    wsrc = pay["wrappers"]["W1"].get("bn_native")
    if wsrc and wsrc not in TODO:
        roman, eng = transliterate(wsrc)
        pay["wrappers"]["W1"]["banglish_translit"] = roman
        report["wrapper_engine"] = eng
    PAY.write_text(json.dumps(pay, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"banglish_translit filled {report['filled']}, skipped(TODO) {report['skipped_todo']}")
    if report["filled"] == 0:
        print("NOTE: bn_native is not authored yet, so banglish_translit stays TODO_TRANSLIT.")


if __name__ == "__main__":
    main()
