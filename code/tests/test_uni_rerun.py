"""D41: Unicode-preserving rerun arm — episode selection, self-contained notebooks, stock retry, budget.

No Kaggle, no GPU: the notebooks' own cells run against the test-only mock endpoint (tests/_mock_llm.py)."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_grid_notebook as B  # noqa: E402
import build_uni_rerun_notebook as UB  # noqa: E402
import select_uni_rerun as S  # noqa: E402
from runners import uni_rerun as R  # noqa: E402
from runners.main_grid import task_key  # noqa: E402
from runners import unicode_patch as U  # noqa: E402
from tests._mock_llm import MockLLM  # noqa: E402
from tests.test_main_grid import DRIVER, _cells, _run_cells, _sub  # noqa: E402

NB = {k: ROOT / "kaggle" / UB.nb_filename(k) for k in UB.LAUNCH_ORDER}
SEL = json.loads((ROOT / "data/uni_rerun/SELECTION.json").read_text(encoding="utf-8"))
T11 = ROOT.parent / "paper/results/fasttrack_v1_20261007/primary/T11_injection_rendering.csv"
EXPECTED_A = {"gemma": (95, 95), "gptoss": (84, 85), "granite": (100, 99), "ministral": (88, 88), "qwen": (94, 94)}


# ----------------------------------------------------------------------------- selection
def test_selection_counts_payload_ascii_and_file_hashes():
    pa = SEL["_meta"]["payload_ascii_check"]
    # verified on the data: en / banglish_llm injected text (goals + W1 wrapper) is pure ASCII -> reused
    for c in ("en", "banglish_llm"):
        assert pa[c]["goals_with_non_ascii"] == 0 and pa[c]["wrapper_non_ascii_chars"] == 0
    for c in ("bn_mt", "bn_llm"):
        assert pa[c]["goals_with_non_ascii"] == pa[c]["n_goals"] == 35
    for mk, (n_mt, n_llm) in EXPECTED_A.items():
        f = ROOT / SEL["files"][mk]["path"]
        assert S._sha(f) == SEL["files"][mk]["sha256"]
        ep = R.load_episodes(f)
        c = Counter(e["condition"] for e in ep["tier_A"])
        assert c == Counter({"bn_mt": n_mt, "bn_llm": n_llm}), mk          # no en / banglish in the arm
        assert len(R.arm_tasks(ep)) == n_mt + n_llm
        assert all(e["reason"].startswith(("injection_escaped", "error_no_trace")) for e in ep["tier_A"])
        assert not {task_key(t) for t in R.arm_tasks(ep)} & {task_key(t) for t in R.retry_tasks(ep)}
    rt = R.retry_tasks(R.load_episodes(ROOT / SEL["files"]["ministral"]["path"]))
    assert [(t["suite"], t["user_task"], t["injection_task"], t["condition"]) for t in rt] == \
        [("travel", "user_task_5", "injection_task_1", "en")]                      # D35
    assert all(not R.retry_tasks(R.load_episodes(ROOT / SEL["files"][m]["path"]))
               for m in EXPECTED_A if m != "ministral")


def test_selection_agrees_with_committed_T11_rendering_table():
    from analysis.episode_rules import classify_error
    rows = list(csv.DictReader(T11.open(encoding="utf-8")))
    for mk in EXPECTED_A:
        ep = R.load_episodes(ROOT / SEL["files"][mk]["path"])
        stock = [json.loads(l) for l in (S.RESULTS / mk / "runs" / f"{mk}_150.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        # T11 is computed after D38.2 rules: a context overflow excludes the whole (model, pair); those
        # escaped bn episodes are still in the arm (the overflow rule is a sensitivity knob)
        overflow = {(r["suite"], r["user_task"], r["injection_task"]) for r in stock
                    if r.get("status") == "error" and classify_error(r.get("error")) == "context_overflow"}
        kept = [e for e in ep["tier_A"] if (e["suite"], e["user_task"], e["injection_task"]) not in overflow]
        for cond in ("bn_mt", "bn_llm", "en", "banglish_llm"):
            row = next(r for r in rows if r["model"] == ep["label"] and r["condition"] == cond)
            by = Counter(e["reason"] for e in kept if e["condition"] == cond)
            # T11 counts an episode as escaped only if no raw script was seen; "mixed" episodes (escaped
            # in one tool output, raw in another) are T11 raw_bengali_script but are rerun too (D41).
            assert by["injection_escaped"] == int(row["escaped_unicode"]), (mk, cond)
            assert by["injection_escaped_mixed_with_script"] <= int(row["raw_bengali_script"]), (mk, cond)


def test_arm_tasks_reject_duplicates_and_unknown_tiers():
    ep = R.load_episodes(ROOT / SEL["files"]["granite"]["path"])
    with pytest.raises(ValueError):
        R.arm_tasks(ep, ("tier_C",))
    bad = dict(ep, tier_A=ep["tier_A"] + ep["tier_A"][:1])
    with pytest.raises(ValueError):
        R.arm_tasks(bad)
    assert len(R.arm_tasks(ep, ("tier_A", "tier_B"))) == len(ep["tier_A"]) + len(ep["tier_B"])


@pytest.mark.skipif(not all((S.RESULTS / m / f"{m}_150_traces.zip").exists() or (S.RESULTS / m / "runs" / "traces").exists()
                            for m in S.MODEL_KEYS), reason="stock traces not on this machine (gitignored)")
def test_selection_files_are_current():
    assert S.main(["--check"]) == 0


# ----------------------------------------------------------------------------- notebooks
def test_notebooks_current_titles_and_under_limit(tmp_path):
    written, est = UB.write_notebooks(UB.LAUNCH_ORDER, tmp_path)
    assert [w["title"] for w in written] == [f"bangla-pi-uni-{k}" for k in ["gptoss", "granite", "ministral", "qwen", "gemma"]]
    for w in written:
        committed = ROOT / "kaggle" / Path(w["path"]).name
        assert committed.read_bytes() == Path(w["path"]).read_bytes(), f"rebuild {committed.name}"
        assert w["bytes"] < B.KAGGLE_LIMIT_BYTES
        txt = committed.read_text(encoding="utf-8")
        assert w["embed_sha256"] in txt and UB.STOCK_EMBED_SHA256 in txt and f"bangla-pi-uni-{w['model']}" in txt
    assert est["total_est_session_h"] < 10


def test_serve_cells_identical_to_main_grid_and_params():
    for k, path in NB.items():
        uni = _cells(path)
        main = _cells(ROOT / "kaggle" / f"{B.NB_NAME[k]}.ipynb")
        assert [s for t, s in uni if t == "gpu"] == [s for t, s in main if t == "gpu"]   # same servers, -np 1, seeds
        p = dict(uni)["params"]
        assert f"MODEL_KEY  = {k!r}" in p and "TIERS      = ['tier_A']" in p and "STREAMS_PER_SERVER = 1" in p
        assert ("RUN_STOCK_RETRY = True" in p) == (k == "ministral")
        assert "RUN_STOCK_RETRY = False" in p or k == "ministral"
        run = dict(uni)["run"]
        assert "unicode=True" in run and "run_stock_retry" in run


def test_embed_contains_stock_files_unchanged_plus_new_files():
    for k in UB.LAUNCH_ORDER:
        files = UB.embed_files(k)
        assert set(B.embed_file_list()) < set(files)
        assert set(files) - set(B.embed_file_list()) == set(UB.NEW_FILES) | {f"data/uni_rerun/episodes_{k}.json"}
    man = json.loads(UB.stock_manifest_bytes())
    assert set(man["files"]) == set(B.embed_file_list())


def test_unpack_cell_stock_check_passes_and_fails_on_tamper(tmp_path):
    unpack = dict(_cells(NB["ministral"]))["unpack"]
    r = _run_cells(_sub(unpack, tmp_path / "a"), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "EMBED OK" in r.stdout and "STOCK HARNESS OK: 162 files identical" in r.stdout
    assert "runners/unicode_patch.py" in r.stdout
    # a stock harness file modified inside the arm's tarball -> refuses to run
    files = UB.embed_files("ministral")
    src = tmp_path / "src"
    for rel in files:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, src / rel)
    p = src / "runners/run_benchmark.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    blob, info = B.build_tarball(src, files)
    r2 = _run_cells(_sub(UB.unpack_cell(blob, info), tmp_path / "b"), tmp_path)
    assert r2.returncode != 0 and "stock harness files differ" in r2.stderr


# ----------------------------------------------------------------------------- dry run
def _dry(nb, work, port, smoke):
    work.mkdir(parents=True, exist_ok=True)
    (work / "driver.py").write_text(DRIVER, encoding="utf-8")
    cfg = json.dumps({"nb": str(nb), "work": str(work), "input": str(work / "noinput"), "port": port,
                      "smoke": smoke, "tags": ["unpack", "params", "run", "finalize"]})
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(work / "driver.py"), cfg], capture_output=True, text=True,
                          cwd=str(work), env=env, timeout=900)


BN_ESC = re.compile(r"\\u09[89a-fA-F][0-9a-fA-F]")


def test_ministral_notebook_dry_run_separate_arms(tmp_path):
    work = tmp_path / "w"
    with MockLLM() as mock:
        r = _dry(NB["ministral"], work, mock.port, smoke=3)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        bodies = list(mock.requests)
    out = r.stdout
    assert "unicode arm 176 episodes {'bn_mt': 88, 'bn_llm': 88}" in out and "stock retry 1" in out
    assert "SMOKE_N=3: unicode arm limited to 3 episodes" in out
    uni_line = [l for l in out.splitlines() if l.startswith("UNI_RESULT ")][-1]
    ret_line = [l for l in out.splitlines() if l.startswith("STOCK_RETRY_RESULT ")][-1]
    assert "run_id=ministral_150_uni" in uni_line and "episodes_done=3/3" in uni_line and "complete=True" in uni_line
    assert "run_id=ministral_150_stockretry" in ret_line and "episodes_done=1/1" in ret_line and "per_condition=en:1/1" in ret_line
    runs = work / "runs"
    uni = [json.loads(l) for l in (runs / "ministral_150_uni.jsonl").read_text(encoding="utf-8").splitlines()]
    ret = [json.loads(l) for l in (runs / "ministral_150_stockretry.jsonl").read_text(encoding="utf-8").splitlines()]
    hu, hr = uni[0], ret[0]
    assert hu["_header"] and hu["arm"] == R.ARM_UNI and hu["unicode_patch"]["enabled"] is True
    assert hu["unicode_self_check"]["tool_output"] == "script" and hu["decision"] == "D41"
    assert hr["arm"] == R.ARM_STOCK_RETRY and hr["unicode_patch"]["enabled"] is False
    assert hr["unicode_self_check"]["tool_output"] == "escaped"
    assert hu["stock_embed_sha256"] == UB.STOCK_EMBED_SHA256 and hu["selection_sha256"] == SEL["files"]["ministral"]["sha256"]
    eu = [x for x in uni if not x.get("_header")]
    er = [x for x in ret if not x.get("_header")]
    assert len(eu) == 3 and {e["condition"] for e in eu} <= {"bn_mt", "bn_llm"} and all(e["status"] == "ok" for e in eu)
    assert len(er) == 1 and er[0]["condition"] == "en" and er[0]["status"] == "ok"
    assert (runs / "traces_uni").is_dir() and (runs / "traces_stockretry").is_dir() and not (runs / "traces").exists()
    for rid in ("ministral_150_uni", "ministral_150_stockretry"):
        man = json.loads((work / f"runs_manifest_{rid}.json").read_text(encoding="utf-8"))
        assert man["arm"] in (R.ARM_UNI, R.ARM_STOCK_RETRY) and man["summary"]["complete"]
        assert man["unicode_patch"]["enabled_for_arm"] == (man["arm"] == R.ARM_UNI)
        assert zipfile.ZipFile(work / f"{rid}_traces.zip").namelist()
    # what the 'server' received: no escaped Bengali in any tool message of the unicode arm
    # (the stock retry is en = ASCII), and the traces say the same
    tool_msgs = [m["content"] for b in bodies for m in b["messages"] if m["role"] == "tool"]
    assert tool_msgs and not any(BN_ESC.search(t if isinstance(t, str) else json.dumps(t)) for t in tool_msgs)
    from analysis.ingest import load_traces
    tr = load_traces(runs / "traces_uni")
    assert len(tr) == 3 and "escaped_unicode" not in set(tr["injection_render"])


def test_stock_retry_budget_abandons_and_is_infrastructure(tmp_path):
    ep = R.load_episodes(ROOT / SEL["files"]["ministral"]["path"])
    tasks = R.retry_tasks(ep)
    jl = tmp_path / "retry.jsonl"
    with MockLLM(delay_s=30) as mock:
        st = R.run_stock_retry(tasks, [mock.port], label="mock", out_jsonl=jl, trace_dir=tmp_path / "tr",
                               header={"run_id": "x"}, budget_s=2, log=lambda *a: None)
    assert st["abandoned"] and st["error"] == 1 and not U.is_enabled()
    rec = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if '"_header"' not in l][-1]
    assert rec["status"] == "error" and "RetryBudgetExceeded" in rec["error"]
    from analysis.episode_rules import classify_error
    assert classify_error(rec["error"]) == "infrastructure"           # D38.2: missing, never a failure
    with U.unicode_preserving(True), pytest.raises(RuntimeError):
        R.run_stock_retry(tasks, [9], label="m", out_jsonl=tmp_path / "y.jsonl", trace_dir=tmp_path / "t2")


def test_run_arm_restores_patch_and_checks_pipeline(tmp_path):
    ep = R.load_episodes(ROOT / SEL["files"]["gptoss"]["path"])
    with MockLLM() as mock:
        st = R.run_arm(R.arm_tasks(ep)[:1], [mock.port], unicode=True, label="mock", out_jsonl=tmp_path / "u.jsonl",
                       trace_dir=tmp_path / "tr", header={"run_id": "u"}, log=lambda *a: None)
    assert st["ok"] == 1 and not U.is_enabled()


def test_estimate_session_h():
    ep = {"tier_A": [{"stock_duration_s": 3600, "suite": "s", "user_task": "u", "injection_task": "i"},
                     {"stock_duration_s": 1800, "suite": "s", "user_task": "u2", "injection_task": "i"}], "tier_B": []}
    e = R.estimate_session_h(ep, 2, 0.5, en_durations={("s", "u2", "i"): 3600})
    assert e["n_episodes"] == 2 and e["episode_h"] == 0.75 and e["est_session_h"] == round(0.75 * 1.1 + 0.55, 2)
    assert e["episode_h_high"] == 1.0
    e2 = R.estimate_session_h(ep, 1, 0.0, retry_budget_s=1800, boot_h=0)
    assert e2["est_session_h"] == round(1.5 * 1.1 + 0.5, 2)
    est = UB.estimates()
    assert list(est["models"]) == UB.LAUNCH_ORDER
    assert est["models"]["ministral"]["retry_budget_h"] == round(1500 / 3600, 3)
    assert all(v["retry_budget_h"] == 0 for k, v in est["models"].items() if k != "ministral")
