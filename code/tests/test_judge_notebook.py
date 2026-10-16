"""Layer-3 judge notebooks (DECISIONS D39): builder, embedded inputs, routing, and a CPU dry run of
the notebook's own cells against the test-only mock endpoint (tests/_mock_llm.py). Nothing here
launches Kaggle or loads a model; mock verdicts are meaningless and never recorded as data."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_judge_notebook as BJ  # noqa: E402
from analysis.ingest import KEY, build_episodes, load_judge  # noqa: E402
from scoring import run_judge  # noqa: E402
from scoring.judge import (check_cross_vendor, load_judge_config, render_trace,  # noqa: E402
                           select_judge_id)
from tests._mock_llm import MockLLM  # noqa: E402

NB = {k: ROOT / "kaggle" / v["nb"] for k, v in BJ.JUDGES.items()}
HAVE_TRACES = all(any((ROOT / "results/main" / m / "runs/traces").rglob("*.json")) if
                  (ROOT / "results/main" / m / "runs/traces").is_dir() else
                  (ROOT / "results/main" / m / f"{m}_150_traces.zip").is_file() for m in BJ.MAIN_MODELS)


def _embedded(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][1]["source"])
    b64 = src.split("EMBED_B64 = '''")[1].split("'''")[0]
    blob = base64.b64decode("".join(b64.split()))
    files = {}
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(blob)), mode="r:") as tf:
        for m in tf.getmembers():
            files[m.name] = tf.extractfile(m).read()
    return nb, blob, files


# ----------------------------------------------------------------------------- routing
def test_fasttrack_routing_is_cross_vendor():
    cfg = load_judge_config(BJ.CONFIG)
    want = {"gemma-4-26b-a4b": "qwen_kaggle", "gpt-oss-20b": "qwen_kaggle", "ministral-3-14b": "qwen_kaggle",
            "granite-4.2-8b": "qwen_kaggle", "qwen3.6-35b-a3b": "gemma_kaggle"}
    for agent, jid in want.items():
        assert select_judge_id(agent, cfg) == jid
    with pytest.raises(ValueError):
        check_cross_vendor("qwen_kaggle", "qwen3.6-35b-a3b", cfg)
    with pytest.raises(ValueError):
        check_cross_vendor("gemma_kaggle", "gemma-4-26b-a4b", cfg)


# ----------------------------------------------------------------------------- committed notebooks
@pytest.mark.parametrize("key", list(BJ.JUDGES))
def test_committed_notebook_embeds_verified_inputs(key):
    nb, blob, files = _embedded(NB[key])
    assert len(NB[key].read_bytes()) <= BJ.KAGGLE_LIMIT_BYTES
    src1 = "".join(nb["cells"][1]["source"])
    assert f"EMBED_SHA256 = {hashlib.sha256(blob).hexdigest()!r}" in src1
    man = json.loads(files["EMBED_MANIFEST.json"])
    for rel, meta in man["files"].items():
        assert hashlib.sha256(files[rel]).hexdigest() == meta["sha256"], rel
    # embedded code = repo code (except the generated package marker and inputs)
    for rel in BJ.CODE_FILES:
        assert files[rel] == BJ.code_source(rel).read_bytes(), f"{rel} differs from repo: rebuild the judge notebooks"
    for rel, snap in BJ.AS_RUN_CODE.items():   # repo file may only have grown by additive later blocks
        assert (ROOT / rel).read_bytes().startswith(snap.read_bytes()), f"{rel}: as-run prefix changed"
    assert files["judge/judge_config.json"] == BJ.CONFIG.read_bytes()
    recs = [json.loads(l) for l in files["judge/judge_inputs.jsonl"].decode().splitlines()]
    jid = BJ.JUDGES[key]["judge_id"]
    assert recs and {r["judge_id"] for r in recs} == {jid}
    cfg = json.loads(BJ.CONFIG.read_text())
    for m in {r["model"] for r in recs}:
        check_cross_vendor(jid, m, cfg)
    assert all(r["condition"] != "none" and "FINAL ANSWER: " in r["trace_text"] for r in recs)
    assert all(set(KEY) <= set(r) and len(r["trace_sha256"]) == 64 for r in recs)
    assert f"N_INPUTS, N_GATED = {len(recs)}, {sum(r['gated'] for r in recs)}" in src1
    assert BJ.JUDGES[key]["title"] in "".join(nb["cells"][0]["source"])


def test_committed_counts_match_main_grid():
    """2,992 attacked status-ok episodes; gate counts = in-run needs_judge of the scoring JSONL."""
    _, _, fq = _embedded(NB["qwen"])
    _, _, fg = _embedded(NB["gemma"])
    rq = [json.loads(l) for l in fq["judge/judge_inputs.jsonl"].decode().splitlines()]
    rg = [json.loads(l) for l in fg["judge/judge_inputs.jsonl"].decode().splitlines()]
    assert len(rq) + len(rg) == 2992
    assert {r["model"] for r in rg} == {"qwen3.6-35b-a3b"}
    from analysis.ingest import load_scores
    s = load_scores(sorted(str(p) for p in ROOT.glob(BJ.SCORE_GLOB)))
    s = s[(s.status == "ok") & (s.condition != "none")]
    assert len(s) == 2992
    assert sum(r["gated"] for r in rq + rg) == int(s["needs_judge"].fillna(False).astype(bool).sum())


@pytest.mark.skipif(not HAVE_TRACES, reason="main-grid traces not on this machine")
def test_rebuild_is_identical_and_text_equals_render_trace():
    recs = BJ.build_inputs()
    for key in BJ.JUDGES:
        nb, _, _ = BJ.build_notebook(key, BJ.records_for(recs, key))
        txt = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        # compare digests: a plain == on multi-MB strings makes pytest's assertion diff run for minutes
        same = hashlib.sha256(txt.encode()).digest() == hashlib.sha256(NB[key].read_bytes()).digest()
        assert same, f"{NB[key].name} is stale: rerun the builder"
    # spot-check: embedded text == a fresh render_trace of the source trace file (sha-tied)
    from analysis.ingest import load_traces
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:   # clean clones ship traces only as <m>_150_traces.zip
        tr = load_traces(BJ._trace_roots(Path(tmp)), keep_messages=True)
        by = {tuple(r[k] for k in KEY): r for _, r in tr.iterrows()}
        for r in recs[::97]:
            t = by[tuple(r[k] for k in KEY)]
            assert render_trace(t["messages"], **r["render"]) == r["trace_text"]
            assert hashlib.sha256(Path(t["trace_path"]).read_bytes()).hexdigest() == r["trace_sha256"]


def test_estimate_fits_budget():
    for key in BJ.JUDGES:
        _, _, f = _embedded(NB[key])
        recs = [json.loads(l) for l in f["judge/judge_inputs.jsonl"].decode().splitlines()]
        e = BJ.estimate(recs, key)
        assert e["session_h_all_worst"] < 11.0
        assert e["session_h_gated"] <= e["session_h_all"] <= e["session_h_all_worst"]


# ----------------------------------------------------------------------------- run_judge --inputs
def _mini_inputs(tmp_path, n=4):
    _, _, f = _embedded(NB["gemma"])
    recs = [json.loads(l) for l in f["judge/judge_inputs.jsonl"].decode().splitlines()]
    pick = [r for r in recs if not r["gated"]][:n] + [r for r in recs if r["gated"]][:2]
    p = tmp_path / "inputs.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pick), encoding="utf-8")
    return p, pick


def test_run_judge_inputs_gated_first_limit_deadline(tmp_path):
    p, pick = _mini_inputs(tmp_path)
    cfg = json.loads(BJ.CONFIG.read_text())
    with MockLLM() as mock:
        cfg["judges"]["gemma_kaggle"]["base_url"] = f"http://127.0.0.1:{mock.port}/v1"
        cp = tmp_path / "cfg.json"
        cp.write_text(json.dumps(cfg))
        out = tmp_path / "j.jsonl"
        base = ["--config", str(cp), "--inputs", str(p), "--out", str(out), "--only-judge-id", "gemma_kaggle"]
        s = run_judge.main(base + ["--ungated", "--gated-first", "--limit", "2"])
        got = [json.loads(l) for l in out.read_text().splitlines()]
        assert s["judged"] == 2 and all(g["gated"] for g in got)      # gated first
        assert all(g["verdict"] == "full" and g["judge_id"] == "gemma_kaggle" for g in got)
        assert all(g["trace_sha256"] and g["finish_reason"] == "stop" for g in got)
        s2 = run_judge.main(base + ["--ungated", "--deadline-ts", "1"])   # past deadline: no call
        assert s2["judged"] == 0 and s2["stopped_at_deadline"]
        assert run_judge.main(base + ["--ungated", "--gated-first", "--limit", "2"])["to_judge"] == 0
        s3 = run_judge.main(base + ["--ungated", "--gated-first"])
        assert s3["judged"] == len(pick) - 2 and s3["already_done"] == 2
        assert run_judge.main(base + ["--ungated"])["to_judge"] == 0     # resumable
        # another judge's session would skip these (routing guard)
        s4 = run_judge.main(["--config", str(cp), "--inputs", str(p), "--out", str(tmp_path / "x.jsonl"),
                             "--only-judge-id", "qwen_kaggle", "--ungated", "--dry-run"])
        assert s4["to_judge"] == 0 and s4["skipped_other_judge"] == len(pick)
    # ingest: only gated "full" verdicts enter judge-augmented ASR (D39)
    j = load_judge(out)
    import pandas as pd
    sc = pd.DataFrame([{**{k: r[k] for k in KEY}, "status": "ok", "normalised_security": False,
                        "raw_security": False, "utility": True} for r in pick])
    df = build_episodes(None, sc, judge=j)
    g = df.set_index(KEY)
    for r in pick:
        k = tuple(r[k] for k in KEY)
        assert g.loc[k, "judge_verdict"] == "full"
        assert bool(g.loc[k, "judge_aug_security"]) == bool(r["gated"])


# ----------------------------------------------------------------------------- notebook dry run
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


def _dry_run(nb, work, inp, port, smoke):
    work.mkdir(parents=True, exist_ok=True)
    (work / "driver.py").write_text(DRIVER, encoding="utf-8")
    cfg = json.dumps({"nb": str(nb), "work": str(work), "input": str(inp), "port": port, "smoke": smoke})
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(work / "driver.py"), cfg], capture_output=True, text=True,
                          cwd=str(work), env=env, timeout=600)


@pytest.mark.parametrize("key", list(BJ.JUDGES))
def test_notebook_dry_run_with_mock_and_resume(tmp_path, key):
    smoke = 25 if key == "gemma" else 160    # gemma: 21 gated + 4 ungated; qwen: 155 gated + 5
    with MockLLM() as mock:
        r = _dry_run(NB[key], tmp_path / "s1", tmp_path / "noinput", mock.port, smoke)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert mock.calls == smoke
    line = [l for l in r.stdout.splitlines() if l.startswith("JUDGE_RESULT ")][-1]
    ng = {"gemma": 21, "qwen": 155}[key]
    assert f"judge_{key} " in line and f"gated={ng}/{ng}" in line and f"full_on_gated={ng}" in line
    assert "unparseable=0" in line
    out = tmp_path / "s1" / "judge" / f"judge_{key}.jsonl"
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == smoke and sum(r["gated"] for r in recs) == ng
    man = json.loads((tmp_path / "s1" / "judge" / f"judge_manifest_{key}.json").read_text())
    assert man["summary"]["complete_gated"] and not man["summary"]["complete_all"]
    assert man["gguf"]["sha256"] == "TEST-ONLY-NO-GGUF" and man["rubric_sha256"].startswith("0e76965e")
    assert (tmp_path / "s1" / f"judge_{key}_outputs.zip").is_file()
    # new version with the previous Output attached: seeded, the same smoke set is not re-judged
    inp = tmp_path / "input" / "prev"
    inp.mkdir(parents=True)
    (inp / out.name).write_bytes(out.read_bytes())
    with MockLLM() as mock2:
        r2 = _dry_run(NB[key], tmp_path / "s2", tmp_path / "input", mock2.port, smoke)
        assert r2.returncode == 0, r2.stderr[-3000:]
        assert mock2.calls == 0
    assert "resume seed:" in r2.stdout and "prev" in r2.stdout
