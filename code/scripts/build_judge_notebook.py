"""Build the SELF-CONTAINED layer-3 judge Kaggle notebooks (research-plan §4.4, DECISIONS D39).

Two notebooks, one judge model each (no self-judging; cross-vendor rule enforced in code):

  kaggle/judge_qwen.ipynb   Qwen3.6-35B-A3B (alibaba) judges Gemma, gpt-oss, Ministral, Granite episodes
  kaggle/judge_gemma.ipynb  Gemma-4-26B-A4B (google)  judges the Qwen episodes

Each embeds a deterministic xz tarball (base64) with exactly what the judge pass needs:

  scoring/{__init__,judge,run_judge}.py, analysis/{__init__,config,ingest}.py,
  runners/{__init__,main_grid}.py (pinned GGUF + proven serve flags + template check),
  data/goals_en.json, LICENSE, judge/judge_config.json (= scoring/judge_config.fasttrack.json),
  judge/judge_inputs.jsonl (pre-rendered judge inputs for THIS judge's episodes, built by
  scoring/judge_inputs.py from the main-grid scores + traces), EMBED_MANIFEST.json.

The traces themselves are NOT embedded (~60 MB of JSON): the judge only ever sees
``render_trace(messages)`` (tool calls + args + final answer; tool results excluded, D23), so the
inputs carry that text plus the source trace's SHA-256. All 2,992 attacked status-ok episodes are
included; the notebook judges the §4.4-gated ones FIRST (pre-registered, feed judge-augmented ASR),
then, if SCOPE == 'all', the rest (calibration: L2-vs-judge agreement; never enters judge-aug ASR).

`scoring/__init__.py` imports scoring.scoring, which is not needed by the judge, so the notebook
embeds an empty package marker instead (recorded in EMBED_MANIFEST as generated).

Usage (from code/; does NOT launch anything):
  python3 scripts/build_judge_notebook.py              # both notebooks
  python3 scripts/build_judge_notebook.py --estimates  # print the GPU-hour estimate only
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_main_grid_notebook import (DOWNLOAD_CELL, GPU_CELL, SERVE_CELL, code,  # noqa: E402
                                      md)
from runners.main_grid import LLAMA_BUILD, MAX_SESSION_H, MODELS, SEED  # noqa: E402

KAGGLE = ROOT / "kaggle"
KAGGLE_LIMIT_BYTES = 900_000
CONFIG = ROOT / "scoring" / "judge_config.fasttrack.json"
SCORE_GLOB = "results/main/*/runs/*_150.jsonl"
MAIN_MODELS = ["gemma", "gptoss", "granite", "ministral", "qwen"]
# judge notebook key -> (judge id in config, MODELS key of the judge model, launch title)
JUDGES = {
    "qwen": dict(judge_id="qwen_kaggle", model_key="qwen", title="bangla-pi-judge-qwen",
                 nb="judge_qwen.ipynb"),
    "gemma": dict(judge_id="gemma_kaggle", model_key="gemma", title="bangla-pi-judge-gemma",
                  nb="judge_gemma.ipynb"),
}
CODE_FILES = ["LICENSE", "analysis/__init__.py", "analysis/config.py", "analysis/ingest.py",
              "scoring/judge.py", "scoring/run_judge.py", "runners/__init__.py",
              "runners/main_grid.py", "data/goals_en.json"]

# ---- GPU-time model (measured on the main grid's own llama-server logs, same build/flags) ----
# Cold-prompt prefill (prompts >= 1000 tokens), 10th percentile = conservative; decode median.
# Qwen: prefill p10 665 tok/s (median 886), decode 59.5 tok/s; Gemma: p10 1006 (median 1253), 55.1.
RATES = {"qwen": dict(prefill=665.0, decode=59.5), "gemma": dict(prefill=1006.0, decode=55.1)}
RUBRIC_TOKENS = 300          # rubric + goal, ~1.2 kB English
OUT_TOKENS_TYPICAL = 60      # {"verdict": ..., "reason": "<one short sentence>"}
OUT_TOKENS_CAP = 256         # max_tokens: worst case
PER_CALL_OVERHEAD_S = 0.3
SETUP_H = 0.35               # llama.cpp download + GGUF download/SHA (14-20 GB) + server start


def prompt_tokens(text: str) -> int:
    """Conservative token count: ~3 ASCII chars/token, 1 token per non-ASCII char (Bengali)."""
    a = sum(1 for c in text if ord(c) < 128)
    return RUBRIC_TOKENS + a // 3 + (len(text) - a)


def estimate(records, judge_key: str) -> dict:
    r = RATES[JUDGES[judge_key]["model_key"]]

    def secs(recs, out_tok):
        return sum(prompt_tokens(x["trace_text"]) / r["prefill"] + out_tok / r["decode"]
                   + PER_CALL_OVERHEAD_S for x in recs)
    gated = [x for x in records if x["gated"]]
    out = {"n_all": len(records), "n_gated": len(gated),
           "gated_h": round(secs(gated, OUT_TOKENS_TYPICAL) / 3600, 3),
           "all_h": round(secs(records, OUT_TOKENS_TYPICAL) / 3600, 3),
           "all_h_worst": round(secs(records, OUT_TOKENS_CAP) / 3600, 3),
           "setup_h": SETUP_H, "rates": r}
    out["session_h_gated"] = round(out["gated_h"] + SETUP_H, 2)
    out["session_h_all"] = round(out["all_h"] + SETUP_H, 2)
    out["session_h_all_worst"] = round(out["all_h_worst"] + SETUP_H, 2)
    return out


# ----------------------------------------------------------------------------- inputs
def _trace_roots(tmp: Path) -> list[str]:
    """Unzipped runs/traces per model; extract <m>_150_traces.zip into tmp if not unzipped."""
    roots = []
    for m in MAIN_MODELS:
        d = ROOT / "results" / "main" / m / "runs" / "traces"
        if d.is_dir() and any(d.rglob("*.json")):
            roots.append(str(d))
            continue
        z = ROOT / "results" / "main" / m / f"{m}_150_traces.zip"
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp / m)
        roots.append(str(tmp / m))
    return roots


def build_inputs() -> list[dict]:
    from analysis.ingest import load_scores, load_traces
    from scoring.judge import load_judge_config
    from scoring.judge_inputs import build_records
    cfg = load_judge_config(CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        roots = _trace_roots(Path(tmp))
        recs = build_records(load_scores(sorted(str(p) for p in ROOT.glob(SCORE_GLOB))),
                             load_traces(roots, keep_messages=True), cfg)
    return recs


def records_for(recs, judge_key):
    jid = JUDGES[judge_key]["judge_id"]
    return [r for r in recs if r["judge_id"] == jid]


# ----------------------------------------------------------------------------- tarball
def build_tarball(recs_j: list[dict]) -> tuple[bytes, dict]:
    from scoring.judge_inputs import write_jsonl
    blobs = {rel: (ROOT / rel).read_bytes() for rel in CODE_FILES}
    blobs["scoring/__init__.py"] = b'"""Judge-notebook package marker (scoring.scoring is not needed here)."""\n'
    blobs["judge/judge_config.json"] = CONFIG.read_bytes()
    blobs["judge/judge_inputs.jsonl"] = write_jsonl(recs_j, None)
    manifest = {"files": {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)}
                          for k, v in sorted(blobs.items())},
                "generated": ["scoring/__init__.py", "judge/judge_inputs.jsonl"],
                "note": "generated by scripts/build_judge_notebook.py; sha256 of each embedded file"}
    manifest["n_files"] = len(manifest["files"])
    blobs["EMBED_MANIFEST.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for rel in sorted(blobs):
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(blobs[rel]), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(blobs[rel]))
    blob = lzma.compress(raw.getvalue(), format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    info = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
            "raw_tar_bytes": len(raw.getvalue()), "n_files": len(blobs),
            "inputs_sha256": manifest["files"]["judge/judge_inputs.jsonl"]["sha256"],
            "inputs_bytes": manifest["files"]["judge/judge_inputs.jsonl"]["bytes"]}
    return blob, info


# ----------------------------------------------------------------------------- cells
UNPACK_BODY = r'''
_blob = base64.b64decode(''.join(EMBED_B64.split()))
EMBED_SHA256_GOT = hashlib.sha256(_blob).hexdigest()
print(f'EMBED tarball: {len(_blob)} B, sha256 {EMBED_SHA256_GOT}', flush=True)
if len(_blob) != EMBED_BYTES or EMBED_SHA256_GOT != EMBED_SHA256:
    raise RuntimeError(f'EMBEDDED TARBALL CORRUPT: got {len(_blob)} B sha256 {EMBED_SHA256_GOT}, '
                       f'expected {EMBED_BYTES} B sha256 {EMBED_SHA256}. Re-import the notebook.')
if os.path.isdir(CODE_DIR):
    shutil.rmtree(CODE_DIR)
os.makedirs(CODE_DIR)
with tarfile.open(fileobj=io.BytesIO(_blob), mode='r:xz') as _tf:
    _members = _tf.getmembers()
    for _m in _members:
        if not _m.isfile() or _m.name.startswith('/') or '..' in _m.name.split('/'):
            raise RuntimeError(f'unsafe tar member {_m.name!r}')
    if hasattr(tarfile, 'data_filter'):
        _tf.extractall(CODE_DIR, members=_members, filter='data')
    else:
        _tf.extractall(CODE_DIR, members=_members)
_root = Path(CODE_DIR)
EMBED_MANIFEST = json.loads((_root / 'EMBED_MANIFEST.json').read_text(encoding='utf-8'))
_bad = [r for r, m in EMBED_MANIFEST['files'].items()
        if not (_root / r).is_file() or hashlib.sha256((_root / r).read_bytes()).hexdigest() != m['sha256']]
_extra = [str(p.relative_to(_root)) for p in _root.rglob('*') if p.is_file()
          and str(p.relative_to(_root)) not in EMBED_MANIFEST['files'] and p.name != 'EMBED_MANIFEST.json']
if _bad or _extra:
    raise RuntimeError(f'EMBED VERIFY FAILED: bad={_bad} extra={_extra}')
INPUTS_PATH = str(_root / 'judge' / 'judge_inputs.jsonl')
CONFIG_PATH = str(_root / 'judge' / 'judge_config.json')
_inputs = [json.loads(l) for l in open(INPUTS_PATH, encoding='utf-8') if l.strip()]
assert len(_inputs) == N_INPUTS and sum(r['gated'] for r in _inputs) == N_GATED, 'input counts differ from header'
assert {r['judge_id'] for r in _inputs} == {JUDGE_ID}, 'inputs routed to another judge'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)
print(f"verified {len(EMBED_MANIFEST['files'])} files; {N_INPUTS} judge inputs ({N_GATED} gated) for {JUDGE_ID}", flush=True)
'''


def unpack_cell(blob, info, judge_key, n_inputs, n_gated):
    b64 = base64.b64encode(blob).decode()
    lines = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
    return ("# ===== 1. Unpack the embedded judge harness + inputs, verify every sha256 (fails loudly) =====\n"
            "# Self-contained: no Kaggle dataset or git clone. Built by scripts/build_judge_notebook.py.\n"
            "import base64, hashlib, io, json, os, shutil, sys, tarfile, time\n"
            "from pathlib import Path\n"
            "SESSION_T0 = time.time()\n"
            "CODE_DIR = '/kaggle/working/code'\n"
            f"JUDGE_KEY = {judge_key!r}\n"
            f"JUDGE_ID = {JUDGES[judge_key]['judge_id']!r}\n"
            f"N_INPUTS, N_GATED = {n_inputs}, {n_gated}\n"
            f"EMBED_SHA256 = {info['sha256']!r}\n"
            f"EMBED_BYTES = {info['bytes']}\n"
            f"EMBED_B64 = '''\n{lines}\n'''\n" + UNPACK_BODY)


def params_cell(judge_key, est):
    return f'''# ===== 2. PARAMETERS =====
SCOPE      = 'all'     # 'all' = gated first, then every other attacked episode (calibration); 'gated' = §4.4 gate only
DEADLINE_H = {MAX_SESSION_H}      # no NEW judge call after this many hours since cell 1 (Kaggle kills at 12 h)
SMOKE_N    = None      # smoke test only: judge the first N (gated first); None = full pass
WORK       = '/kaggle/working'
OUT_DIR    = '/kaggle/working/judge'
INPUT_ROOT = '/kaggle/input'   # attach a previous version's Output -> its judge_*.jsonl is resumed (skip-if-done)
EST_SESSION_H = {{'gated': {est['session_h_gated']}, 'all': {est['session_h_all']}, 'all_worst': {est['session_h_all_worst']}}}

from runners.main_grid import MODELS, LLAMA_BUILD, SEED
from scoring.judge import RUBRIC_SHA256, load_judge_config, check_cross_vendor
MODEL_KEY = {JUDGES[judge_key]['model_key']!r}
M = MODELS[MODEL_KEY]
CFG = load_judge_config(CONFIG_PATH)
assert CFG['judges'][JUDGE_ID]['vendor'] != '' and JUDGE_ID in CFG['judges']
for _m in sorted({{r['model'] for r in _inputs}}):
    check_cross_vendor(JUDGE_ID, _m, CFG)      # raises on any same-vendor pairing
RUN_ID = f'judge_{{JUDGE_KEY}}'
OUT_JSONL = f'{{OUT_DIR}}/{{RUN_ID}}.jsonl'
print(f"{{RUN_ID}}: judge {{M['label']}} ({{CFG['judges'][JUDGE_ID]['vendor']}}) on agents "
      f"{{sorted({{r['model'] for r in _inputs}})}}; scope={{SCOPE}} rubric={{RUBRIC_SHA256[:12]}}", flush=True)
'''


INSTALL_CELL = '''# ===== 4. Pinned llama.cpp (CUDA 12.8 build, same as the main grid) =====
import glob
LLAMA_DIR = '/kaggle/tmp/llama' if os.path.isdir('/kaggle/tmp') else '/tmp/llama'
os.makedirs(LLAMA_DIR, exist_ok=True)
_rel = f'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}'
for _url in (f'{_rel}/cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz',
             f'{_rel}/llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz'):
    _fn = f"{LLAMA_DIR}/{_url.split('/')[-1]}"
    subprocess.run(['wget', '-q', '-O', _fn, _url], check=True)
    subprocess.run(['tar', 'xzf', _fn, '-C', LLAMA_DIR], check=True)
SERVER = glob.glob(f'{LLAMA_DIR}/**/llama-server', recursive=True)[0]
print('llama-server at', SERVER, flush=True)
print(subprocess.run([SERVER, '--version'], capture_output=True, text=True).stderr[:300], flush=True)
import httpx, pandas  # noqa: F401  (preinstalled on Kaggle; the judge client needs httpx)
'''

RUN_CELL = '''# ===== 7. Judge: gated episodes first, then (SCOPE='all') the rest; resumable; deadline guard =====
import glob
from scoring import run_judge
os.makedirs(OUT_DIR, exist_ok=True)
SEEDED = None
if not os.path.exists(OUT_JSONL):
    for _p in sorted(glob.glob(f'{INPUT_ROOT}/**/{RUN_ID}.jsonl', recursive=True)):
        shutil.copy(_p, OUT_JSONL)
        SEEDED = _p
        break
print('resume seed:', SEEDED, flush=True)
CFG['judges'][JUDGE_ID]['base_url'] = f'http://127.0.0.1:{PORTS[0]}/v1'
_cfg_used = f'{OUT_DIR}/judge_config_used.json'
Path(_cfg_used).write_text(json.dumps(CFG, indent=1), encoding='utf-8')
_args = ['--config', _cfg_used, '--inputs', INPUTS_PATH, '--out', OUT_JSONL, '--only-judge-id', JUDGE_ID,
         '--gated-first', '--deadline-ts', str(SESSION_T0 + DEADLINE_H * 3600)]
if SCOPE == 'all':
    _args.append('--ungated')
if SMOKE_N:
    _args += ['--limit', str(SMOKE_N)]
_t0 = time.time()
JUDGE_SUMMARY = run_judge.main(_args)
JUDGE_WALL_S = time.time() - _t0
print('JUDGE_SUMMARY', json.dumps(JUDGE_SUMMARY), f'wall {JUDGE_WALL_S:.0f}s', flush=True)
'''

FINAL_CELL = '''# ===== 8. Stop server, tally, manifest, JUDGE_RESULT line =====
import collections, datetime as _dt, zipfile
for _p in PROCS:
    try:
        _p.terminate()
    except Exception:
        pass
_keys = ['model', 'defense', 'suite', 'user_task', 'injection_task', 'condition', 'rep']
_done = {}
for _l in open(OUT_JSONL, encoding='utf-8'):
    if _l.strip():
        _r = json.loads(_l)
        _done[tuple(_r[k] for k in _keys)] = _r
_want_g = {tuple(r[k] for k in _keys) for r in _inputs if r['gated']}
_want_a = {tuple(r[k] for k in _keys) for r in _inputs}
TALLY = collections.Counter((r['model'], r['condition'], 'gated' if r['gated'] else 'ungated', r['verdict'])
                            for r in _done.values())
_unparse = sum(1 for r in _done.values() if str(r.get('reason', '')).startswith(('unparseable', 'judge parse error')))
_trunc = sum(1 for r in _done.values() if r.get('finish_reason') == 'length')
_lat = sorted(r.get('latency_s') or 0 for r in _done.values())
SUMMARY = {'judged': len(_done), 'gated_done': len(_want_g & set(_done)), 'gated_total': len(_want_g),
           'all_done': len(_want_a & set(_done)), 'all_total': len(_want_a),
           'complete_gated': _want_g <= set(_done), 'complete_all': _want_a <= set(_done),
           'verdicts': dict(collections.Counter(r['verdict'] for r in _done.values())),
           'full_on_gated': sum(1 for r in _done.values() if r['gated'] and r['verdict'] == 'full'),
           'unparseable': _unparse, 'finish_length': _trunc,
           'latency_s_median': _lat[len(_lat)//2] if _lat else None}
WALL_S = time.time() - SESSION_T0
MANIFEST = {'run_id': RUN_ID, 'judge_id': JUDGE_ID, 'judge_model': M['label'], 'scope': SCOPE, 'smoke_n': SMOKE_N,
            'gguf': {k: M[k] for k in ('repo', 'rev', 'file', 'size')} | {'sha256': GGUF_SHA},
            'serve': SERVE_USED, 'llama_build': LLAMA_BUILD, 'seed': SEED, 'rubric_sha256': RUBRIC_SHA256,
            'embed': {'sha256': EMBED_SHA256, 'bytes': EMBED_BYTES},
            'inputs_sha256': EMBED_MANIFEST['files']['judge/judge_inputs.jsonl']['sha256'],
            'resume_seeded': SEEDED, 'run_judge_summary': JUDGE_SUMMARY, 'summary': SUMMARY,
            'tally': [list(k) + [v] for k, v in sorted(TALLY.items())], 'est_session_h': EST_SESSION_H,
            'session_started': _dt.datetime.fromtimestamp(SESSION_T0).isoformat(timespec='seconds'),
            'session_ended': _dt.datetime.now().isoformat(timespec='seconds'), 'wall_s': round(WALL_S, 1)}
MANIFEST_PATH = f'{OUT_DIR}/judge_manifest_{JUDGE_KEY}.json'
Path(MANIFEST_PATH).write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
with zipfile.ZipFile(f'{WORK}/{RUN_ID}_outputs.zip', 'w', zipfile.ZIP_DEFLATED) as _z:
    for _f in (OUT_JSONL, MANIFEST_PATH, f'{OUT_DIR}/judge_config_used.json') + tuple(s['log'] for s in SERVE_USED['servers']):
        if os.path.exists(_f):
            _z.write(_f, os.path.basename(_f))
print(json.dumps(SUMMARY, indent=1), flush=True)
if not SUMMARY['complete_gated'] or (SCOPE == 'all' and not SUMMARY['complete_all']):
    print('INCOMPLETE: attach this version\\'s Output as an input and Save & Run All again (skip-if-done).', flush=True)
print(f"JUDGE_RESULT {RUN_ID} judge={M['label']} gated={SUMMARY['gated_done']}/{SUMMARY['gated_total']} "
      f"all={SUMMARY['all_done']}/{SUMMARY['all_total']} full_on_gated={SUMMARY['full_on_gated']} "
      f"verdicts={json.dumps(SUMMARY['verdicts'], sort_keys=True)} unparseable={_unparse} "
      f"wall_h={WALL_S/3600:.2f} embed={EMBED_SHA256[:12]}", flush=True)
'''


def header_md(judge_key, info, est, recs_j):
    j = JUDGES[judge_key]
    m = MODELS[j["model_key"]]
    agents = sorted({r["model"] for r in recs_j})
    by = {}
    for r in recs_j:
        d = by.setdefault(r["model"], [0, 0])
        d[0] += 1
        d[1] += r["gated"]
    rows = "\n".join(f"| {a} | {by[a][0]} | {by[a][1]} |" for a in agents)
    return md(f"""# Layer-3 judge — {m['label']} judges {', '.join(agents)}
Launch title: **`{j['title']}`**. Research plan §4.4 (judge gate) / §10.3 (cross-vendor), DECISIONS D23, D38, **D39**.
Rubric `scoring/judge.py::JUDGE_PROMPT` (unchanged; every record stores its SHA-256). No self-judging: the judge's
vendor (`{json.loads(CONFIG.read_text())['judges'][j['judge_id']]['vendor']}`) differs from every agent's; `check_cross_vendor` raises otherwise.

| agent model | attacked status-ok episodes | §4.4-gated |
|---|---|---|
{rows}

**Order:** all gated episodes first (pre-registered; they feed judge-augmented ASR), then — with `SCOPE='all'`
(default) — every other attacked episode as a calibration set (L2-vs-judge agreement; never added to ASR).

**Serve:** `{m['repo']}` → `{m['file']}` @ `{m['rev'][:12]}` (SHA-256 `{m['sha256'][:16]}…` asserted), exactly the
main-grid flags ({m['note']}), llama.cpp `{LLAMA_BUILD}`, `--temp 0 --seed {SEED}`, thinking off (template checked).

**GPU estimate** (from the main grid's measured cold-prefill p10 {est['rates']['prefill']:.0f} tok/s and decode
{est['rates']['decode']} tok/s; conservative token count): gated only ≈ **{est['session_h_gated']} h**; all ≈
**{est['session_h_all']} h** (worst case, every reply hits 256 tokens: {est['session_h_all_worst']} h), incl.
{est['setup_h']} h setup. One session, well under the 11 h deadline guard.

**Self-contained:** cell 1 unpacks the embedded harness + pre-rendered judge inputs (xz tarball {info['bytes']:,} B,
{info['n_files']} files, sha256 `{info['sha256']}`; inputs sha256 `{info['inputs_sha256'][:16]}…`) and verifies every file.
Inputs are `render_trace()` text of each trace (tool calls + final answer, tool results excluded, D23) + the trace's SHA-256.

**Outputs (/kaggle/working):** `judge/judge_{judge_key}.jsonl` (one verdict per episode; skip-if-done),
`judge/judge_manifest_{judge_key}.json`, `judge_{judge_key}_outputs.zip`, `server_8099.log`, final `JUDGE_RESULT …` line.
**Resume:** attach this version's Output as an input and Save & Run All again.

Settings: Accelerator **GPU T4 x2**, Internet **on**, notebook **private**, then **Save Version → Save & Run All (Commit)**.
Generated by `scripts/build_judge_notebook.py`; prepared 2026-10-07, **not run**.""")


def build_notebook(judge_key, recs_j):
    blob, info = build_tarball(recs_j)
    est = estimate(recs_j, judge_key)
    n_g = sum(r["gated"] for r in recs_j)
    cells = [
        header_md(judge_key, info, est, recs_j),
        code(unpack_cell(blob, info, judge_key, len(recs_j), n_g), "unpack"),
        code(params_cell(judge_key, est), "params"),
        code(GPU_CELL, "gpu"),
        code(INSTALL_CELL, "gpu"),
        code(DOWNLOAD_CELL, "gpu"),
        code(SERVE_CELL, "gpu"),
        code(RUN_CELL, "run"),
        code(FINAL_CELL, "finalize"),
        md(f"""## After the run (on the box)
Download `judge_{judge_key}_outputs.zip`, check the `JUDGE_RESULT` line and that `embed` matches this header, put
`judge_{judge_key}.jsonl` under `code/results/judge/`, then
`python -m analysis.run_analysis ... --judge 'results/judge/judge_*.jsonl'` (judge-augmented ASR uses gated verdicts
only, D39). Record slug/version, wall time and `JUDGE_RESULT` in `DECISIONS.md`."""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb, info, est


def write_notebooks(out_dir: Path = KAGGLE, recs=None):
    recs = recs if recs is not None else build_inputs()
    written = []
    for k in JUDGES:
        rj = records_for(recs, k)
        nb, info, est = build_notebook(k, rj)
        txt = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        size = len(txt.encode("utf-8"))
        if size > KAGGLE_LIMIT_BYTES:
            raise SystemExit(f"{JUDGES[k]['nb']} is {size} B > {KAGGLE_LIMIT_BYTES}: shard it or use a private dataset")
        path = out_dir / JUDGES[k]["nb"]
        path.write_text(txt, encoding="utf-8")
        written.append({"path": str(path), "bytes": size, "judge": k, "title": JUDGES[k]["title"],
                        "embed": info, "estimate": est})
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estimates", action="store_true")
    ap.add_argument("--out-dir", default=str(KAGGLE))
    a = ap.parse_args(argv)
    recs = build_inputs()
    if a.estimates:
        print(json.dumps({k: estimate(records_for(recs, k), k) for k in JUDGES}, indent=1))
        return 0
    for w in write_notebooks(Path(a.out_dir), recs):
        e = w["estimate"]
        print(f"wrote {w['path']} ({w['bytes']} B) title={w['title']} embed={w['embed']['bytes']} B "
              f"sha256={w['embed']['sha256'][:16]} n_all={e['n_all']} n_gated={e['n_gated']} "
              f"est_h gated={e['session_h_gated']} all={e['session_h_all']} worst={e['session_h_all_worst']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
