"""Import authored text from data/conditions/ back into data/payloads.json.

Only non-TODO bodies (after the header comments) are written. Existing non-TODO values
in payloads.json are overwritten when the file body is filled with DIFFERENT text; a body
that equals the stored value up to leading/trailing whitespace is a no-op, so the exact
whitespace of e.g. AgentDojo's W1 template is never altered by a round trip.

D26 fixes: (1) TODO_LLM (and any TODO_* sentinel) is recognised both as a file-name suffix
and as a body, so a placeholder can never be imported as payload text; (2) the header is the
leading run of '#' lines and the body is everything after it verbatim, and a legacy header
whose multi-line '# EN:' goal leaked uncommented lines is rejected instead of being imported
as body; (3) ``--sync-todo`` writes a file's TODO sentinel back into payloads.json when the
stored value is still filled (used when a previously filled item is withdrawn).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOADS_PATH = ROOT / "data" / "payloads.json"
GOALS_PATH = ROOT / "data" / "goals_en.json"
OUT = ROOT / "data" / "conditions"
TODO = {"TODO_LLM", "TODO_NATIVE", "TODO_MT", "TODO_TRANSLIT", ""}
_SENTINEL_SUFFIX = re.compile(r"\.(TODO_[A-Z]+)$")


def is_todo(text: str) -> bool:
    t = (text or "").strip()
    return t in TODO or t.startswith("TODO_")


def _body(path: Path, goal_en: str | None = None) -> str:
    lines = path.read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    body_lines = lines[i:]
    # drop leading blank lines (the header/body separator)
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines).rstrip()
    if goal_en and "\n" in goal_en:
        first, rest = goal_en.split("\n", 1)
        rest_first = next((ln for ln in rest.split("\n") if ln.strip()), "")
        hdr = "\n".join(lines[:i])
        if rest_first and f"# EN: {first}" in hdr and body.startswith(rest_first) \
                and f"#     {rest_first}" not in hdr:
            raise ValueError(f"{path}: legacy multi-line '# EN:' header leaks into the body; "
                             f"re-run scripts/export_condition_dirs.py first")
    return body + ("\n" if body else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-todo", action="store_true",
                    help="write a file's TODO_* sentinel into payloads.json when the stored "
                         "value is filled (withdraws an item explicitly)")
    args = ap.parse_args()
    payloads = json.loads(PAYLOADS_PATH.read_text(encoding="utf-8"))
    goals = json.loads(GOALS_PATH.read_text(encoding="utf-8"))["goals"]
    updated = 0
    unchanged = 0
    skipped_todo = 0
    withdrawn = []

    wdir = OUT / "_wrappers"
    if wdir.exists():
        for path in wdir.glob("*.txt"):
            # W1__bn_native.txt
            stem = path.stem
            if "__" not in stem:
                continue
            wid, cond = stem.split("__", 1)
            text = _body(path).rstrip("\n")
            if is_todo(text) or not text:
                skipped_todo += 1
                continue
            block = payloads.setdefault("wrappers", {}).setdefault(wid, {})
            old = block.get(cond)
            if isinstance(old, str) and old.strip() == text.strip():
                unchanged += 1  # keep the stored whitespace exactly
                continue
            block[cond] = text
            updated += 1

    for cond, block in payloads["conditions"].items():
        cdir = OUT / cond
        if not cdir.exists():
            continue
        for path in cdir.glob("*.txt"):
            name = path.name
            # workspace__injection_task_0.txt or ...TODO_LLM.txt / .TODO_MT.txt / ...
            core = name[: -len(".txt")]
            m = _SENTINEL_SUFFIX.search(core)
            file_sentinel = m.group(1) if m else None
            if m:
                core = core[: m.start()]
            key = core.replace("__", "/", 1)
            if key not in block["goals"]:
                print(f"WARN: unknown key {key} in {path}")
                continue
            text = _body(path, goals.get(key, {}).get("goal_en")).rstrip("\n")
            if file_sentinel or is_todo(text) or not text:
                skipped_todo += 1
                sentinel = file_sentinel or (text.strip() or None)
                cur = block["goals"].get(key)
                if args.sync_todo and sentinel and not is_todo(cur or ""):
                    block["goals"][key] = sentinel
                    withdrawn.append(f"{cond}:{key}->{sentinel}")
                continue
            if block["goals"].get(key) == text:
                unchanged += 1
                continue
            block["goals"][key] = text
            updated += 1

    PAYLOADS_PATH.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for w in withdrawn:
        print(f"WITHDRAWN (--sync-todo): {w}")
    print(f"Updated {updated} fields; unchanged {unchanged}; skipped {skipped_todo} TODO/empty; "
          f"withdrawn {len(withdrawn)}. Wrote {PAYLOADS_PATH}")


if __name__ == "__main__":
    main()
