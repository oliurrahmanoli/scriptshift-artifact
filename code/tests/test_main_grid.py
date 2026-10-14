"""Self-contained main-grid notebooks (DECISIONS D30): embedded tarball, in-notebook
PAYLOAD_SHA256SUMS verification, per-model serve settings, 12-h budget, and a CPU dry run of the
notebook's own cells against a test-only mock OpenAI endpoint (tests/_mock_llm.py)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_grid_notebook as B  # noqa: E402
from runners import main_grid as G  # noqa: E402
from tests._mock_llm import MockLLM  # noqa: E402

NB = {k: ROOT / "kaggle" / f"{B.NB_NAME[k]}.ipynb" for k in B.LAUNCH_ORDER}


def _cells(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(c["metadata"].get("tags", [""])[0], "".join(c["source"]))
            for c in nb["cells"] if c["cell_type"] == "code"]


# ----------------------------------------------------------------------------- tarball
def test_tarball_deterministic_contents_and_size():
    b1, i1 = B.build_tarball()
    b2, i2 = B.build_tarball()
    assert b1 == b2 and i1["sha256"] == i2["sha256"]
    files = set(i1["files"])
    for need in ("runners/main_grid.py", "runners/run_benchmark.py", "runners/endpoints.py",
                 "attacks/multilingual_attack.py", "scoring/scoring.py", "scoring/judge.py",
                 "scoring/normalise.py", "data/payloads.json", "data/accept_lists.json",
                 "subset/pairs_150.json", "subset/pairs.json", "data/conditions/PAYLOAD_SHA256SUMS.txt",
                 "data/conditions/bn_llm/_gen_meta.json", "data/conditions/_wrappers/W1__bn_mt.txt"):
        assert need in files, need
    sums = (ROOT / "data/conditions/PAYLOAD_SHA256SUMS.txt").read_text(encoding="utf-8")
    listed = [l.split("  ", 1)[1] for l in sums.splitlines()
              if l.strip() and not l.startswith("#") and "CANONICAL" not in l]
    assert listed and all(f"data/conditions/{n}" in files for n in listed)
    for f in files:
        assert not re.search(r"(^|/)tests?/|\.bak|\.log$|traces|\.ipynb$|__pycache__", f), f
    assert i1["bytes"] < 400_000


def test_embed_refuses_forbidden_paths(monkeypatch):
    monkeypatch.setattr(B, "CODE_FILES", B.CODE_FILES + ["tests/test_attacks.py"])
    with pytest.raises(ValueError):
        B.embed_file_list()


def test_committed_notebooks_are_current_and_under_kaggle_limit(tmp_path):
    written, info, est = B.write_notebooks(B.LAUNCH_ORDER, tmp_path, generic=True)
    assert len(written) == 6
    for w in written:
        name = Path(w["path"]).name
        committed = ROOT / "kaggle" / name
        assert committed.read_bytes() == Path(w["path"]).read_bytes(), f"rebuild {name}"
        assert w["bytes"] < B.KAGGLE_LIMIT_BYTES
        assert info["sha256"] in committed.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- settings
def test_per_model_serve_settings_match_pilots():
    g = G.server_commands(G.MODELS["gemma"], "srv", "m.gguf")
    assert len(g) == 1 and g[0]["env"] == {}
    c = g[0]["cmd"]
    assert c[c.index("--tensor-split") + 1] == "1,1" and "--swa-full" in c
    assert c[c.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    q = G.server_commands(G.MODELS["qwen"], "srv", "m.gguf")
    assert len(q) == 1 and "--tensor-split" in q[0]["cmd"] and "--swa-full" not in q[0]["cmd"]
    for k in ("granite", "ministral", "gptoss"):
        r = G.server_commands(G.MODELS[k], "srv", "m.gguf")
        assert [s["env"]["CUDA_VISIBLE_DEVICES"] for s in r] == ["0", "1"]
        assert [s["port"] for s in r] == [8099, 8100]
        assert all("--tensor-split" not in s["cmd"] for s in r)
    gr = G.server_commands(G.MODELS["granite"], "srv", "m.gguf")[0]["cmd"]
    assert gr[gr.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    assert "--chat-template-kwargs" not in G.server_commands(G.MODELS["ministral"], "srv", "m.gguf")[0]["cmd"]
    go = G.server_commands(G.MODELS["gptoss"], "srv", "m.gguf")[0]["cmd"]
    assert go[go.index("--chat-template-kwargs") + 1] == '{"reasoning_effort":"low"}'
    for k, m in G.MODELS.items():
        cmd = G.server_commands(m, "srv", "m.gguf")[0]["cmd"]
        for flag, val in (("-c", "24576"), ("-np", "1"), ("--temp", "0"), ("--seed", "20261002")):
            assert cmd[cmd.index(flag) + 1] == val
        # D32: llama-server splits -c across -np slots; travel/workspace prompts reach ~6.6k tokens,
        # so each slot must keep the full 24576 (1 slot per server = 1 agent stream per server).
        assert m["ctx"] // m["nslots"] >= 16384
        assert "--jinja" in cmd and len(m["sha256"]) == 64 and m["size"] > 1e9
    assert {m["variant"] for m in G.MODELS.values()} == {
        "swa_full+no_think", "no_think", "no_think_kwargs", "plain", "low_effort_kwargs"}


def test_template_check():
    M = G.MODELS
    assert G.template_ok(M["granite"], "<|im_start|>assistant\n<think></think>")
    assert not G.template_ok(M["granite"], "<|im_start|>assistant\n")
    assert G.template_ok(M["gptoss"], "...Reasoning: low\n...")
    assert G.template_ok(M["qwen"], "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not G.template_ok(M["qwen"], "<|im_start|>assistant\n<think>\n")
    assert G.template_ok(M["gemma"], "<start_of_turn>model\n")
    assert not G.template_ok(M["gemma"], "<|think|> model")
    assert not G.template_ok(M["ministral"], "<apply-template failed: boom>")


def test_notebook_parameters_per_model():
    for k, path in NB.items():
        params = dict(_cells(path))["params"]
        assert f"MODEL_KEY  = {k!r}" in params
        assert "SUBSET     = '150'" in params
        assert "CONDITIONS = ['en', 'bn_mt', 'bn_llm', 'banglish_llm']" in params
        assert re.search(r"^DEFENSE\s+= None", params, re.M)
        assert "SHARD, N_SHARDS = 1, 1" in params
    assert not list((ROOT / "kaggle").glob("main_grid_*_shard*.ipynb"))


def test_utility_task_count_matches_agentdojo():
    from agentdojo.task_suite.load_suites import get_suites
    s = get_suites(G.BENCHMARK_VERSION)
    assert {k: len(s[k].user_tasks) for k in G.SUITE_ORDER} == G.UTILITY_TASKS_V122
    assert len(G.utility_task_list(s)) == 97


# ----------------------------------------------------------------------------- tasks / budget
def test_build_tasks_pair_major_and_shards_partition():
    pairs = json.loads((ROOT / "subset/pairs_150.json").read_text())["pairs"]
    util = [("banking", f"user_task_{i}") for i in range(16)]
    full = G.build_tasks(pairs, B.FAST_CONDITIONS, util)
    assert len(full) == 150 * 4 + 16
    assert [t["condition"] for t in full[:4]] == B.FAST_CONDITIONS
    assert len({G.task_key(t) for t in full}) == len(full)
    assert {t["suite"] for t in full[:16]} == {"workspace", "travel", "banking", "slack"}
    assert all(t["kind"] == "utility" and t["condition"] == "none" for t in full[-16:])
    parts = [G.build_tasks(pairs, B.FAST_CONDITIONS, util, s, 3) for s in (1, 2, 3)]
    keys = [set(map(G.task_key, p)) for p in parts]
    assert sum(map(len, keys)) == len(full) and set().union(*keys) == set(map(G.task_key, full))
    with pytest.raises(ValueError):
        G.build_tasks(pairs, ["en"], [], 4, 3)


def test_estimates_fit_12h_without_sharding():
    est = B.estimates("150")
    assert est["episodes_per_model"] == 150 * 4 + 97 == G.episode_count(150, 4)
    for k, e in est["models"].items():
        assert e["shards"] == 1 and e["est_session_h"] < G.MAX_SESSION_H
        assert e["est_session_h"] == round(697 / G.MODELS[k]["pilot_eps_h"] * 1.10 + 0.25, 2)
        assert e["est_session_h_at_half_rate"] < G.MAX_SESSION_H
    assert B.LAUNCH_ORDER == sorted(G.MODELS, key=lambda k: -G.MODELS[k]["pilot_eps_h"])
    assert G.shards_needed(11.5) == 2 and G.shards_needed(10.9) == 1


# ----------------------------------------------------------------------------- unpack cell
def _run_cells(code_text, work):
    script = Path(work) / "cells.py"
    script.write_text(code_text, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["BANGLA_PI_PAYLOADS"] = "/nonexistent/should_be_popped.json"
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          cwd=str(work), env=env, timeout=600)


def _sub(src, work):
    return src.replace("/kaggle/working", f"{work}/working").replace("/kaggle/input", f"{work}/input")


def test_unpack_cell_unpacks_and_verifies(tmp_path):
    unpack = dict(_cells(NB["gemma"]))["unpack"]
    r = _run_cells(_sub(unpack, tmp_path) + "\nprint('VERIFY', json.dumps(PAYLOAD_VERIFY))\n", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "EMBED OK" in r.stdout and "PAYLOAD_SHA256SUMS OK: 146 files + 4 canonical" in r.stdout
    v = json.loads(r.stdout.split("VERIFY ", 1)[1].splitlines()[0])
    assert v["conditions"] == sorted(B.FAST_CONDITIONS) and v["goal_files_vs_payloads_json"] == 140
    code_dir = tmp_path / "working" / "code"
    assert (code_dir / "runners/main_grid.py").read_bytes() == (ROOT / "runners/main_grid.py").read_bytes()
    assert not (code_dir / "tests").exists()


def test_unpack_cell_fails_on_corrupt_tarball(tmp_path):
    unpack = dict(_cells(NB["gemma"]))["unpack"]
    bad = re.sub(r"EMBED_SHA256 = '[0-9a-f]{64}'", "EMBED_SHA256 = '" + "0" * 64 + "'", unpack)
    r = _run_cells(_sub(bad, tmp_path), tmp_path)
    assert r.returncode != 0 and "EMBEDDED TARBALL CORRUPT" in r.stderr


@pytest.mark.parametrize("tamper", ["payload_file", "payloads_json_goal", "payloads_json_wrapper"])
def test_unpack_cell_fails_loudly_on_payload_mismatch(tmp_path, tamper):
    files = B.embed_file_list()
    src = tmp_path / "src"
    for rel in files:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, src / rel)
    if tamper == "payload_file":
        p = src / "data/conditions/bn_llm/banking__injection_task_0.txt"
        p.write_text(p.read_text(encoding="utf-8") + " extra", encoding="utf-8")
    else:
        pj = src / "data/payloads.json"
        d = json.loads(pj.read_text(encoding="utf-8"))
        if tamper == "payloads_json_goal":
            d["conditions"]["bn_mt"]["goals"]["banking/injection_task_0"] += " x"
        else:
            d["wrappers"]["W1"]["banglish_llm"] = d["wrappers"]["W1"]["banglish_llm"].replace("Signed", "Signd")
        pj.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    blob, info = B.build_tarball(src, files)
    r = _run_cells(_sub(B.unpack_cell(blob, info), tmp_path), tmp_path)
    assert r.returncode != 0, r.stdout
    assert "PAYLOAD/EMBED VERIFICATION FAILED" in r.stderr
    assert ("PAYLOAD_SHA256SUMS:" in r.stdout) or ("!= sha-verified file" in r.stdout)


# ----------------------------------------------------------------------------- dry run
DRIVER = r'''
import json, sys
cfg = json.loads(sys.argv[1])
nb = json.load(open(cfg["nb"], encoding="utf-8"))
ns = {"__name__": "__main__"}
injected = False
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    tag = c["metadata"]["tags"][0]
    if tag not in cfg["tags"] and tag != "gpu":
        continue
    if tag == "gpu":   # GPU / llama.cpp / GGUF / serve cells: replaced by the mock endpoint
        if not injected:
            ns.update(SERVER="(test)", MODEL_PATH="(test)", GGUF_SHA="TEST-ONLY-NO-GGUF", PROCS=[],
                      PORTS=[cfg["port"]], MODEL_ID="mock-model", TEMPLATE_OK=True,
                      SERVE_USED={"layout": "mock", "variant": "mock", "ports": [cfg["port"]],
                                  "model_id": "mock-model", "servers": [], "template_ok": True})
            injected = True
        continue
    src = "".join(c["source"]).replace("/kaggle/working", cfg["work"]).replace("/kaggle/input", cfg["input"])
    if tag == "params":
        src += "\nSMOKE_N = %d\n" % cfg["smoke"]
    exec(compile(src, "<cell %s>" % tag, "exec"), ns)
'''


def _dry_run(nb, work, inp, port, smoke, tags=("unpack", "params", "run", "finalize")):
    work.mkdir(parents=True, exist_ok=True)
    (work / "driver.py").write_text(DRIVER, encoding="utf-8")
    cfg = json.dumps({"nb": str(nb), "work": str(work), "input": str(inp), "port": port,
                      "smoke": smoke, "tags": list(tags)})
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(work / "driver.py"), cfg], capture_output=True,
                          text=True, cwd=str(work), env=env, timeout=900)


def test_notebook_dry_run_with_mock_model_and_resume(tmp_path):
    with MockLLM() as mock:
        work = tmp_path / "s1"
        r = _dry_run(NB["granite"], work, tmp_path / "noinput", mock.port, smoke=3)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert mock.calls > 0
    out = r.stdout
    line = [l for l in out.splitlines() if l.startswith("MAIN_RESULT ")][-1]
    assert "run_id=granite_150 model=granite-4.2-8b shard=1/1 episodes_done=6/6 errors=0 missing=0" in line
    assert "per_condition=en:1/1,bn_mt:1/1,bn_llm:1/1,none:3/3" in line   # pair-major order
    assert "complete=True" in line and B.build_tarball()[1]["sha256"] in line
    jl = work / "runs" / "granite_150.jsonl"
    recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines()]
    eps = [x for x in recs if not x.get("_header")]
    hdr = [x for x in recs if x.get("_header")][0]
    assert hdr["embed_sha256"] and hdr["payload_sums_sha256"] and hdr["gguf_sha256"] == "TEST-ONLY-NO-GGUF"
    assert len(eps) == 6 and all(e["status"] == "ok" for e in eps)
    assert sorted({e["kind"] for e in eps}) == ["attack", "utility"]
    assert all(isinstance(e["normalised_security"], bool) for e in eps if e["kind"] == "attack")
    assert any(e["n_tool_calls"] > 0 for e in eps)
    traces = sorted((work / "runs" / "traces" / "granite-4.2-8b").rglob("*.json"))
    assert len(traces) == 6
    assert sum(1 for t in traces if t.parent.name == "none" and t.name == "none.json") == 3
    z = work / "granite_150_traces.zip"
    assert len(zipfile.ZipFile(z).namelist()) == 6
    man = json.loads((work / "runs_manifest_granite_150.json").read_text(encoding="utf-8"))
    assert man["summary"]["complete"] and man["embed"]["sha256"] == B.build_tarball()[1]["sha256"]
    assert man["payload_verify"]["canonical"] == 4 and man["smoke_n"] == 3

    # analysis/ingest reads the notebook's outputs (layer-1 trace + layer-2 JSONL join)
    from analysis.ingest import build_episodes, load_scores, load_traces
    df = build_episodes(load_traces(work / "runs" / "traces"), load_scores(jl))
    assert len(df) == 6 and set(df["condition"]) == {"en", "bn_mt", "bn_llm", "none"}

    # same session dir again: skip-if-done -> nothing re-run
    with MockLLM() as mock2:
        r2 = _dry_run(NB["granite"], work, tmp_path / "noinput", mock2.port, smoke=3)
        assert r2.returncode == 0, r2.stderr[-3000:]
        assert mock2.calls == 0
    assert "ran_this_session=0" in r2.stdout and "episodes_done=6/6" in r2.stdout

    # new Kaggle version with the previous Output attached as input -> seeded, nothing re-run
    inp = tmp_path / "input" / "prev-version"
    shutil.copytree(work / "runs", inp / "runs", ignore=shutil.ignore_patterns("traces"))
    shutil.copy2(z, inp / z.name)
    with MockLLM() as mock3:
        r3 = _dry_run(NB["granite"], tmp_path / "s2", tmp_path / "input", mock3.port, smoke=3)
        assert r3.returncode == 0, r3.stderr[-3000:]
        assert mock3.calls == 0
    assert "ran_this_session=0" in r3.stdout and "complete=True" in r3.stdout
    assert len(list((tmp_path / "s2" / "runs" / "traces").rglob("*.json"))) == 6


def test_run_grid_deadline_and_unreachable_server(tmp_path):
    pairs = json.loads((ROOT / "subset/pairs_150.json").read_text())["pairs"][:2]
    tasks = G.build_tasks(pairs, ["en", "bn_mt"])
    st = G.run_grid(tasks, [9], label="t", out_jsonl=tmp_path / "a.jsonl", trace_dir=tmp_path / "tr",
                    deadline_ts=0, log=lambda *a: None)
    assert st["ran"] == 0 and st["remaining"] == 4 and st["stopped_by_deadline"]
    st2 = G.run_grid(tasks, [1], label="t", out_jsonl=tmp_path / "b.jsonl", trace_dir=tmp_path / "tr",
                     pipeline_factory=lambda port: (_ for _ in ()).throw(ConnectionError("down")),
                     log=lambda *a: None)
    assert st2["ran"] == 0 and st2["worker_errors"] and st2["remaining"] == 4
    s = G.summarize(tmp_path / "b.jsonl", tasks)
    assert s["missing"] == 4 and not s["complete"]
    assert "episodes_done=0/4" in G.main_result_line("r", "t", "1/1", s, 1.0, st2, "abc")


def test_done_keys_retry_errors(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"_header": True},
            {"suite": "s", "user_task": "u", "injection_task": "i", "condition": "en", "status": "error"},
            {"suite": "s", "user_task": "u2", "injection_task": "i", "condition": "en", "status": "ok"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert G.done_keys(p) == {("s", "u2", "i", "en")}
    assert len(G.done_keys(p, retry_errors=False)) == 2
