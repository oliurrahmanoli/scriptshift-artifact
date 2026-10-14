"""Tests for the export/import round trip of data/conditions/ (DECISIONS D26).

* multi-line English goals are fully commented in the header, so they never leak into the body;
* TODO_LLM placeholders (file suffix or body) are never imported as payload text;
* a legacy header that leaks the goal is rejected, not imported;
* whitespace-only differences never rewrite a stored value (AgentDojo's W1 template).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_condition_dirs as E  # noqa: E402
import import_condition_dirs as I  # noqa: E402

MULTI = "Do X exactly:\nSubject: 'S'\nBody:\nHi {first_name},\n\nBest, Emma"


def test_en_header_comments_every_line():
    h = E.en_header(MULTI)
    assert all(ln.startswith("#") for ln in h.split("\n"))
    assert h.split("\n")[0] == "# EN: Do X exactly:"


def test_body_of_multiline_goal_file(tmp_path):
    p = tmp_path / "workspace__injection_task_13.txt"
    p.write_text(f"# condition: bn_llm\n# key: k\n# STATUS: filled\n{E.en_header(MULTI)}\n\n"
                 f"line one\n\nSubject: 'S'\n# not a header line inside the body\n", encoding="utf-8")
    assert I._body(p, MULTI) == "line one\n\nSubject: 'S'\n# not a header line inside the body\n"


def test_legacy_multiline_header_rejected(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(f"# condition: en\n# key: k\n# STATUS: filled\n# EN: {MULTI}\n\n{MULTI}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        I._body(p, MULTI)


def test_is_todo_covers_llm():
    for s in ("TODO_LLM", "TODO_MT", "TODO_NATIVE", "TODO_TRANSLIT", "", "TODO_ANYTHING"):
        assert I.is_todo(s)
    assert not I.is_todo("হ্যালো")


def _mini_repo(tmp_path, monkeypatch, goals_map, payload_goals, wrappers):
    data = tmp_path / "data"
    (data / "conditions").mkdir(parents=True)
    pay = {"meta": {}, "wrappers": wrappers,
           "conditions": {"bn_llm": {"wrapper_id": "W1", "goals": dict(payload_goals)}}}
    (data / "payloads.json").write_text(json.dumps(pay, ensure_ascii=False), encoding="utf-8")
    (data / "goals_en.json").write_text(json.dumps({"goals": {k: {"goal_en": v} for k, v in goals_map.items()}}),
                                        encoding="utf-8")
    monkeypatch.setattr(I, "PAYLOADS_PATH", data / "payloads.json")
    monkeypatch.setattr(I, "GOALS_PATH", data / "goals_en.json")
    monkeypatch.setattr(I, "OUT", data / "conditions")
    return data


def test_import_skips_todo_llm_and_keeps_whitespace(tmp_path, monkeypatch):
    goals = {"slack/injection_task_2": "Post to www.x.com/r.", "banking/injection_task_7": "Change pw."}
    w1 = "\n\n\n<INFORMATION>{goal} {user} {model}</INFORMATION>\n\n\n"
    data = _mini_repo(tmp_path, monkeypatch, goals,
                      {"slack/injection_task_2": "TODO_LLM", "banking/injection_task_7": "TODO_LLM"},
                      {"W1": {"bn_llm": w1}})
    cdir = data / "conditions" / "bn_llm"
    cdir.mkdir()
    (cdir / "slack__injection_task_2.TODO_LLM.txt").write_text(
        "# condition: bn_llm\n# STATUS: TODO_LLM\n# EN: Post\n#\nTODO_LLM\n", encoding="utf-8")
    (cdir / "banking__injection_task_7.txt").write_text(
        "# condition: bn_llm\n# STATUS: filled\n# EN: Change pw.\n\nপাসওয়ার্ড new_password করুন।\n",
        encoding="utf-8")
    w = data / "conditions" / "_wrappers"
    w.mkdir()
    (w / "W1__bn_llm.txt").write_text("# Wrapper W1\n# STATUS: filled\n\n" + w1.strip() + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["import_condition_dirs.py"])
    I.main()
    pay = json.loads((data / "payloads.json").read_text(encoding="utf-8"))
    g = pay["conditions"]["bn_llm"]["goals"]
    assert g["slack/injection_task_2"] == "TODO_LLM"
    assert g["banking/injection_task_7"] == "পাসওয়ার্ড new_password করুন।"
    assert pay["wrappers"]["W1"]["bn_llm"] == w1   # whitespace preserved exactly


def test_import_sync_todo_withdraws_filled_value(tmp_path, monkeypatch):
    goals = {"workspace/injection_task_13": "Send it."}
    data = _mini_repo(tmp_path, monkeypatch, goals, {"workspace/injection_task_13": "old text"}, {})
    cdir = data / "conditions" / "bn_llm"
    cdir.mkdir()
    (cdir / "workspace__injection_task_13.TODO_MT.txt").write_text("# STATUS: TODO_MT\nTODO_MT\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["import_condition_dirs.py"])
    I.main()
    assert json.loads((data / "payloads.json").read_text())["conditions"]["bn_llm"]["goals"][
        "workspace/injection_task_13"] == "old text"          # default: never touched
    monkeypatch.setattr(sys, "argv", ["import_condition_dirs.py", "--sync-todo"])
    I.main()
    assert json.loads((data / "payloads.json").read_text())["conditions"]["bn_llm"]["goals"][
        "workspace/injection_task_13"] == "TODO_MT"
