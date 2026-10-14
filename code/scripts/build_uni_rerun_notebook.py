"""Build the SELF-CONTAINED Unicode-preserving rerun notebooks (DECISIONS D41), one per model:
``kaggle/uni_rerun_<model>.ipynb``, Kaggle title ``bangla-pi-uni-<model>``.

Same builder approach and verification as the main grid (scripts/build_main_grid_notebook.py, D30):
a deterministic gzip tarball embedded as base64, unpacked and sha-verified by cell 1 (EMBED_MANIFEST,
PAYLOAD_SHA256SUMS, payloads.json vs the frozen files). The GPU / llama.cpp / GGUF / serve cells are
the main-grid cells verbatim, so every model is served exactly as in its main-grid run (D30/D32:
pinned GGUF + sha256, ``-c 24576 -np 1``, 1 agent stream per server, ``--temp 0 --seed 20261002``,
same template check).

The tarball = the 162 stock files of the main-grid embed (sha256 ee7b4195…, unchanged; cell 1 checks
every one against the stock EMBED_MANIFEST beb44a7d… that actually ran) + ``runners/unicode_patch.py``
+ ``runners/uni_rerun.py`` + ``data/uni_rerun/SELECTION.json`` + this model's
``data/uni_rerun/episodes_<model>.json`` + a copy of the stock manifest.

Runs (cell 7): the tier-A episodes of this model with the unicode patch ON (``<model>_150_uni``);
tier B only if ``TIERS`` is edited; Ministral additionally retries its D35 API-timeout episode with the
patch OFF (``ministral_150_stockretry``, 25-min budget), kept in separate files.

Usage (from code/; does NOT launch anything):
  python3 scripts/build_uni_rerun_notebook.py              # all five
  python3 scripts/build_uni_rerun_notebook.py --estimates  # GPU-h estimates only
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_main_grid_notebook as B  # noqa: E402
from runners.main_grid import LLAMA_BUILD, MODELS, SEED  # noqa: E402
from runners import uni_rerun as R  # noqa: E402

STOCK_EMBED_SHA256 = "ee7b4195e694977b3f4e060155d66ca373027cdc2dee98ee88750b5305169334"   # D32, ran D33-D37
STOCK_MANIFEST_SHA256 = "beb44a7dba688ffa8d1a2030c6451cd6c515d8d8c3caec8d26d657d06f898ba9"
STOCK_MANIFEST_REL = "data/uni_rerun/STOCK_EMBED_MANIFEST.json"
# Primary-pool models first (D38.3), the longest early so two slots finish together; Gemma (self-
# generated payload cells, outside pooled primary inference) last.
LAUNCH_ORDER = ["gptoss", "granite", "ministral", "qwen", "gemma"]
NEW_FILES = ["runners/unicode_patch.py", "runners/uni_rerun.py", "data/uni_rerun/SELECTION.json",
             STOCK_MANIFEST_REL]
RESULTS = ROOT / "results" / "main"


def title(mk):
    return f"bangla-pi-uni-{mk}"


def nb_filename(mk):
    return f"uni_rerun_{mk}.ipynb"


def stock_manifest_bytes() -> bytes:
    """EMBED_MANIFEST.json of the stock main-grid tarball, rebuilt from the repo; refuses unless the
    stock tarball is still byte-identical to the one that ran (ee7b4195…)."""
    blob, info = B.build_tarball()
    if info["sha256"] != STOCK_EMBED_SHA256:
        raise SystemExit(f"stock harness changed: main-grid tarball {info['sha256']} != {STOCK_EMBED_SHA256}")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        b = tf.extractfile("EMBED_MANIFEST.json").read()
    assert B.sha256_bytes(b) == STOCK_MANIFEST_SHA256 == info["manifest_sha256"]
    return b


def write_stock_manifest() -> Path:
    p = ROOT / STOCK_MANIFEST_REL
    b = stock_manifest_bytes()
    if not p.exists() or p.read_bytes() != b:
        p.write_bytes(b)
    return p


def embed_files(mk) -> list[str]:
    return sorted(set(B.embed_file_list()) | set(NEW_FILES) | {f"data/uni_rerun/episodes_{mk}.json"})


def episodes(mk) -> dict:
    return R.load_episodes(ROOT / "data" / "uni_rerun" / f"episodes_{mk}.json")


# ----------------------------------------------------------------------------- estimates
def observed(mk) -> dict:
    """Observed main-grid session numbers (runs_manifest_<m>_150.json = the MAIN_RESULT line)."""
    man = json.loads((RESULTS / mk / f"runs_manifest_{mk}_150.json").read_text(encoding="utf-8"))
    st = man["stats"]
    return {"wall_s": man["wall_s"], "grid_s": st["grid_s"], "eps_per_h": st["eps_per_h"],
            "streams": len(man["serve"]["servers"]) * man.get("streams_per_server", 1),
            "setup_h": (man["wall_s"] - st["grid_s"]) / 3600}


def en_durations(mk) -> dict:
    out = {}
    for line in (RESULTS / mk / "runs" / f"{mk}_150.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("condition") == "en" and r.get("status") == "ok":
            out[(r["suite"], r["user_task"], r["injection_task"])] = r["duration_s"]
    return out


def estimates(tiers=("tier_A",)) -> dict:
    out = {"tiers": list(tiers), "method": "sum of the OBSERVED stock durations of the same episodes / agent "
           "streams x1.10 + observed in-session setup (wall_s - grid_s) + 0.05 h Kaggle start/save; high = "
           "per episode max(stock, same-pair en duration); Ministral + 25-min stock-retry budget", "models": {}}
    for mk in LAUNCH_ORDER:
        ob = observed(mk)
        ep = episodes(mk)
        e = R.estimate_session_h(ep, ob["streams"], ob["setup_h"], tiers,
                                 retry_budget_s=R.STOCK_RETRY_BUDGET_S if ep["stock_retry"] else 0,
                                 en_durations=en_durations(mk))
        e["observed_main_grid"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ob.items()}
        e["main_grid_rate_h"] = round(e["n_episodes"] / ob["eps_per_h"] * 1.10 + e["setup_h"] + e["boot_h"]
                                      + e["retry_budget_h"], 2)
        e["stock_retry"] = len(ep["stock_retry"])
        out["models"][mk] = e
    out["total_est_session_h"] = round(sum(v["est_session_h"] for v in out["models"].values()), 2)
    out["total_est_session_h_high"] = round(sum(v["est_session_h_high"] for v in out["models"].values()), 2)
    return out


# ----------------------------------------------------------------------------- cells
STOCK_CHECK = r'''
# (d) D41: the stock harness inside this tarball is byte-identical to the main-grid embed that ran
_stock = (_root / 'data/uni_rerun/STOCK_EMBED_MANIFEST.json').read_bytes()
if _sha(_stock) != STOCK_MANIFEST_SHA256:
    raise RuntimeError(f'STOCK_EMBED_MANIFEST sha256 {_sha(_stock)} != {STOCK_MANIFEST_SHA256}')
_stock_bad = [r for r, m in json.loads(_stock)['files'].items()
              if not (_root / r).is_file() or _sha((_root / r).read_bytes()) != m['sha256']]
if _stock_bad:
    raise RuntimeError(f'stock harness files differ from the main-grid embed: {_stock_bad[:5]}')
_new = sorted(set(EMBED_MANIFEST['files']) - set(json.loads(_stock)['files']))
print(f'STOCK HARNESS OK: {len(json.loads(_stock)["files"])} files identical to main-grid embed '
      f'{STOCK_EMBED_SHA256[:12]}…; added: {_new}', flush=True)
'''


def unpack_cell(blob, info):
    src = B.unpack_cell(blob, info)
    head = (f"STOCK_EMBED_SHA256 = {STOCK_EMBED_SHA256!r}   # main-grid tarball that ran (D32-D37)\n"
            f"STOCK_MANIFEST_SHA256 = {STOCK_MANIFEST_SHA256!r}\n")
    return src.replace("SESSION_T0 = time.time()\n", "SESSION_T0 = time.time()\n" + head, 1) + STOCK_CHECK


def params_cell(mk, est):
    ep = episodes(mk)
    retry = bool(ep["stock_retry"])
    return f'''# ===== 2. PARAMETERS (Unicode-preserving rerun arm, D41) =====
MODEL_KEY  = {mk!r}
TIERS      = ['tier_A']          # tier_A = escaped-injection episodes (the arm). Add 'tier_B' only if decided (D41)
RUN_STOCK_RETRY = {retry!r}      # D35: retry the infrastructure-error episode(s) with the patch OFF
STOCK_RETRY_BUDGET_S = {R.STOCK_RETRY_BUDGET_S}
STREAMS_PER_SERVER = 1           # = main grid (D30/D32): 1 agent stream per llama-server
DEADLINE_H = 11.0                # no NEW episodes after this many hours since cell 1
SMOKE_N    = None                # smoke test only: first N episodes of the arm; None = all
WORK       = '/kaggle/working'
OUTDIR     = '/kaggle/working/runs'
INPUT_ROOT = '/kaggle/input'     # attach a previous version's Output here -> skip-if-done resume
EST_SESSION_H = {est['est_session_h']}   # builder estimate from observed main-grid durations (high {est['est_session_h_high']})

from runners.main_grid import MODELS, resolve_model, LLAMA_BUILD, SEED, BENCHMARK_VERSION
from runners import uni_rerun, unicode_patch
MODEL_KEY = resolve_model(MODEL_KEY)
M = MODELS[MODEL_KEY]
LABEL = M['label']               # same label as the stock run; the arm lives in separate files/dirs
DEFENSE = None
RUN_ID_UNI = f'{{MODEL_KEY}}_150_uni'
RUN_ID_RETRY = f'{{MODEL_KEY}}_150_stockretry'
_sel = json.loads(Path(f'{{CODE_DIR}}/data/uni_rerun/SELECTION.json').read_text(encoding='utf-8'))
_ep_path = Path(f'{{CODE_DIR}}/data/uni_rerun/episodes_{{MODEL_KEY}}.json')
if _sha(_ep_path.read_bytes()) != _sel['files'][MODEL_KEY]['sha256']:
    raise RuntimeError('episode list sha256 does not match SELECTION.json')
EPISODES = uni_rerun.load_episodes(_ep_path)
SELECTION_SHA256 = _sha(_ep_path.read_bytes())
UNI_TASKS = uni_rerun.arm_tasks(EPISODES, TIERS)
RETRY_TASKS = uni_rerun.retry_tasks(EPISODES) if RUN_STOCK_RETRY else []
_pay = json.loads(Path(f'{{CODE_DIR}}/data/payloads.json').read_text(encoding='utf-8'))
_filled = lambda v: isinstance(v, str) and v.strip() != '' and not v.strip().startswith('TODO_')
_miss = [t for t in UNI_TASKS + RETRY_TASKS
         if not _filled(_pay['conditions'][t['condition']]['goals'].get(f"{{t['suite']}}/{{t['injection_task']}}"))
         or not _filled(_pay['wrappers']['W1'].get(t['condition']))]
if _miss:
    raise RuntimeError(f'{{len(_miss)}} episodes without payload, e.g. {{_miss[:3]}}')
assert {{t['condition'] for t in UNI_TASKS + RETRY_TASKS}} <= set(PAYLOAD_VERIFY['conditions'])
from collections import Counter as _C
print(f'{{LABEL}}: unicode arm {{len(UNI_TASKS)}} episodes {{dict(_C(t["condition"] for t in UNI_TASKS))}} '
      f'(tiers {{TIERS}}); stock retry {{len(RETRY_TASKS)}}; selection sha256 {{SELECTION_SHA256[:16]}}…', flush=True)
print(json.dumps(M, indent=1), flush=True)
'''


RUN_CELL = '''# ===== 7. Run: Unicode-preserving arm (patch ON), then the stock retry (patch OFF) =====
unicode_patch.self_check(expect_enabled=False)    # stock behaviour until the arm starts (needs agentdojo, installed in cell 4)
from runners.main_grid import seed_from_inputs
from runners.run_benchmark import PINNED
os.makedirs(OUTDIR, exist_ok=True)
if SMOKE_N:   # smoke test only (never for the real run)
    UNI_TASKS = UNI_TASKS[:SMOKE_N]
    print(f'SMOKE_N={SMOKE_N}: unicode arm limited to {len(UNI_TASKS)} episodes', flush=True)
OUT_UNI, TRACE_UNI = f'{OUTDIR}/{RUN_ID_UNI}.jsonl', f'{OUTDIR}/traces_uni'
OUT_RETRY, TRACE_RETRY = f'{OUTDIR}/{RUN_ID_RETRY}.jsonl', f'{OUTDIR}/traces_stockretry'
SEEDED = {}
for _rid, _out, _tr in ((RUN_ID_UNI, OUT_UNI, TRACE_UNI), (RUN_ID_RETRY, OUT_RETRY, TRACE_RETRY)):
    if not os.path.exists(_out):
        SEEDED[_rid] = {k: (v if not isinstance(v, list) else len(v))
                        for k, v in seed_from_inputs(INPUT_ROOT, _out, _tr, _rid).items()}
print('resume seed from inputs:', SEEDED, flush=True)
HEADER = {'model_key': MODEL_KEY, 'subset': '150', 'serve': SERVE_USED, 'gguf_sha256': GGUF_SHA,
          'llama_build': LLAMA_BUILD, 'seed': SEED, 'pinned': PINNED, 'embed_sha256': EMBED_SHA256,
          'stock_embed_sha256': STOCK_EMBED_SHA256, 'payload_sums_sha256': PAYLOAD_SUMS_SHA256,
          'selection_sha256': SELECTION_SHA256, 'tiers': TIERS, 'decision': 'D41'}
DEADLINE_TS = SESSION_T0 + DEADLINE_H * 3600
STATS_UNI = uni_rerun.run_arm(UNI_TASKS, PORTS, unicode=True, label=LABEL, out_jsonl=OUT_UNI, trace_dir=TRACE_UNI,
                              header=dict(HEADER, run_id=RUN_ID_UNI), deadline_ts=DEADLINE_TS,
                              streams_per_port=STREAMS_PER_SERVER)
print('STATS_UNI', json.dumps({k: v for k, v in STATS_UNI.items() if k != 'worker_errors'}), flush=True)
STATS_RETRY = {}
if RETRY_TASKS and time.time() < DEADLINE_TS:
    STATS_RETRY = uni_rerun.run_stock_retry(RETRY_TASKS, PORTS, label=LABEL, out_jsonl=OUT_RETRY,
                                            trace_dir=TRACE_RETRY, header=dict(HEADER, run_id=RUN_ID_RETRY),
                                            budget_s=STOCK_RETRY_BUDGET_S)
    print('STATS_RETRY', json.dumps({k: v for k, v in STATS_RETRY.items() if k != 'worker_errors'}, default=str), flush=True)
for _s in (STATS_UNI, STATS_RETRY):
    if _s.get('worker_errors'):
        print('WORKER ERRORS:', _s['worker_errors'], flush=True)
'''

FINAL_CELL = '''# ===== 8. Stop servers, manifests, zipped traces, UNI_RESULT / STOCK_RETRY_RESULT lines =====
import datetime as _dt
from runners.main_grid import summarize, zip_traces, main_result_line
for _p in PROCS:
    try:
        _p.terminate()
    except Exception:
        pass
WALL_S = time.time() - SESSION_T0
RESULT_LINES = []
for _rid, _arm, _tasks, _out, _tr, _st, _tag in (
        (RUN_ID_UNI, 'unicode_preserving', UNI_TASKS, OUT_UNI, TRACE_UNI, STATS_UNI, 'UNI_RESULT'),
        (RUN_ID_RETRY, 'stock_retry', RETRY_TASKS, OUT_RETRY, TRACE_RETRY, STATS_RETRY, 'STOCK_RETRY_RESULT')):
    if not _tasks:
        continue
    _sum = summarize(_out, _tasks)
    _zip = zip_traces(_tr, f'{WORK}/{_rid}_traces.zip')
    _man = {'run_id': _rid, 'arm': _arm, 'decision': 'D41', 'model_key': MODEL_KEY, 'label': LABEL,
            'tiers': TIERS if _arm == 'unicode_preserving' else None, 'n_tasks': len(_tasks),
            'streams_per_server': STREAMS_PER_SERVER, 'smoke_n': SMOKE_N,
            'gguf': {k: M[k] for k in ('repo', 'rev', 'file', 'size')} | {'sha256': GGUF_SHA},
            'serve': SERVE_USED, 'llama_build': LLAMA_BUILD, 'seed': SEED, 'pinned': PINNED,
            'embed': {'sha256': EMBED_SHA256, 'bytes': EMBED_BYTES, 'n_files': EMBED_N_FILES},
            'stock_embed_sha256': STOCK_EMBED_SHA256, 'selection_sha256': SELECTION_SHA256,
            'unicode_patch': unicode_patch.describe() | {'enabled_for_arm': _arm == 'unicode_preserving'},
            'payload_verify': PAYLOAD_VERIFY, 'resume_seeded': SEEDED.get(_rid),
            'stats': _st, 'summary': _sum, 'traces_zip': _zip, 'est_session_h': EST_SESSION_H,
            'session_started': _dt.datetime.fromtimestamp(SESSION_T0).isoformat(timespec='seconds'),
            'session_ended': _dt.datetime.now().isoformat(timespec='seconds'), 'wall_s': round(WALL_S, 1),
            'outputs': {'jsonl': _out, 'traces_dir': _tr, 'traces_zip': _zip['zip'],
                        'server_logs': [s['log'] for s in SERVE_USED['servers']]}}
    Path(f'{WORK}/runs_manifest_{_rid}.json').write_text(json.dumps(_man, indent=2, ensure_ascii=False, default=str),
                                                         encoding='utf-8')
    print(f'wrote runs_manifest_{_rid}.json, {_out}, {_zip["zip"]} ({_zip["n_traces"]} traces)', flush=True)
    if not _sum['complete']:
        print(f"INCOMPLETE {_rid}: {_sum['missing']} missing + {_sum['errors']} errored. To resume, attach this "
              f"version's Output as an input and Save & Run All again (skip-if-done).", flush=True)
    RESULT_LINES.append(main_result_line(_rid, LABEL, '1/1', _sum, WALL_S, _st, EMBED_SHA256)
                        .replace('MAIN_RESULT ', _tag + ' ', 1))
for _l in RESULT_LINES:
    print(_l, flush=True)
'''


def header_md(mk, info, est):
    m = MODELS[mk]
    ep = episodes(mk)
    from collections import Counter
    ca = dict(Counter(e["condition"] for e in ep["tier_A"]))
    cb = dict(Counter(e["condition"] for e in ep["tier_B"]))
    retry = ep["stock_retry"]
    rline = ("" if not retry else
             f"\n**Stock retry (patch OFF, separate outputs `{mk}_150_stockretry*`):** "
             + ", ".join(f"{e['suite']}/{e['user_task']} × {e['injection_task']} `{e['condition']}`" for e in retry)
             + f" (D35 APITimeoutError; the stock episode hung 9 × 600 s, so the retry is capped at "
               f"{R.STOCK_RETRY_BUDGET_S // 60} min and recorded as missing if it hangs again).\n")
    return B.md(f"""# {title(mk)} — Unicode-preserving rerun arm — {m['label']}
DECISIONS D40/D41. **Secondary, post hoc arm**, not part of the pre-registered primary analysis.
AgentDojo 0.1.35 formats structured tool output with `yaml.safe_dump(allow_unicode=False)`, so in the stock
main grid the Bengali injection reached the agent as `\\uXXXX` escapes in these episodes. This notebook reruns
**exactly those episodes** with `runners/unicode_patch.py` ON (`allow_unicode=True` for tool output +
`ensure_ascii=False` when replaying the agent's own tool-call arguments) and **nothing else changed**.

**Episodes (tier A, run):** {len(ep['tier_A'])} = {ca}. Tier B (escapes only outside the injected block or
non-ASCII tool args; {len(ep['tier_B'])} = {cb}) is listed but **not run** unless `TIERS` is edited.
`en`/`banglish_llm` injected text is ASCII, so those cells are reused from the stock run.
{rline}
**Model / serve (identical to the main grid):** `{m['repo']}` → `{m['file']}` @ `{m['rev'][:12]}` (SHA-256 asserted);
{m['note']}; `-c {m['ctx']} -np {m['nslots']} --jinja --temp 0 --seed {SEED}`, llama.cpp `{LLAMA_BUILD}`,
1 agent stream per server.

**Estimate:** {est['est_session_h']:.2f} GPU-h (high {est['est_session_h_high']:.2f}), from the observed main-grid
durations of these very episodes ({est['observed_main_grid']['streams']} stream(s), observed setup
{est['setup_h']:.2f} h). Deadline guard: no new episodes after 11 h.

**Self-contained:** cell 1 unpacks the embedded harness (tarball {info['bytes']:,} B gz, {info['n_files']} files,
sha256 `{info['sha256']}`), verifies EMBED_MANIFEST, PAYLOAD_SHA256SUMS, and that the 162 stock harness files are
byte-identical to the main-grid embed `{STOCK_EMBED_SHA256[:12]}…` that produced the stock results.

**Outputs (/kaggle/working):** `runs/{mk}_150_uni.jsonl`, `runs/traces_uni/…`, `{mk}_150_uni_traces.zip`,
`runs_manifest_{mk}_150_uni.json`{', and the same set for `' + mk + '_150_stockretry`' if retry else ''}, `server_*.log`,
final `UNI_RESULT …`{' / `STOCK_RETRY_RESULT …`' if retry else ''} line(s). Resume: attach this Output as an input and
Save & Run All again (skip-if-done).

Settings: Accelerator **GPU T4 x2**, Internet **on**, notebook **private**, title **`{title(mk)}`**, then
**Save Version → Save & Run All (Commit)**. Generated by `scripts/build_uni_rerun_notebook.py`; **not run**.""")


def build_notebook(mk, est) -> tuple[dict, dict]:
    blob, info = B.build_tarball(ROOT, embed_files(mk))
    cells = [
        header_md(mk, info, est),
        B.code(unpack_cell(blob, info), "unpack"),
        B.code(params_cell(mk, est), "params"),
        B.code(B.GPU_CELL, "gpu"),
        B.code(B.INSTALL_CELL, "gpu"),
        B.code(B.DOWNLOAD_CELL, "gpu"),
        B.code(B.SERVE_CELL, "gpu"),
        B.code(RUN_CELL, "run"),
        B.code(FINAL_CELL, "finalize"),
        B.md("""## After the run (on the box)
Download the Output into `code/results/uni/<model>/`, check the `UNI_RESULT` line and that `embed_sha256` matches
the header, then `python -m analysis.unicode_arm …` (see `analysis/README.md`, D41). The stock retry
(`*_stockretry.jsonl` + traces) is folded into the primary set with `run_analysis --retry-scores/--retry-traces`.
Record slug/version, wall time and the result lines in `DECISIONS.md`."""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb, info


def write_notebooks(model_keys=LAUNCH_ORDER, out_dir: Path = B.KAGGLE) -> tuple[list, dict]:
    write_stock_manifest()
    est = estimates()
    written = []
    for mk in model_keys:
        nb, info = build_notebook(mk, est["models"][mk])
        txt = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        size = len(txt.encode("utf-8"))
        if size > B.KAGGLE_LIMIT_BYTES:
            raise SystemExit(f"{nb_filename(mk)} is {size} B > {B.KAGGLE_LIMIT_BYTES}")
        path = Path(out_dir) / nb_filename(mk)
        path.write_text(txt, encoding="utf-8")
        written.append({"path": str(path), "bytes": size, "model": mk, "title": title(mk),
                        "embed_sha256": info["sha256"], "n_files": info["n_files"]})
    return written, est


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="all")
    ap.add_argument("--estimates", action="store_true")
    ap.add_argument("--tier-b", action="store_true", help="with --estimates: also estimate tier A+B")
    ap.add_argument("--out-dir", default=str(B.KAGGLE))
    a = ap.parse_args(argv)
    if a.estimates:
        print(json.dumps(estimates(), indent=1))
        if a.tier_b:
            print(json.dumps(estimates(("tier_A", "tier_B")), indent=1))
        return 0
    keys = LAUNCH_ORDER if a.model == "all" else [a.model]
    written, est = write_notebooks(keys, Path(a.out_dir))
    for w in written:
        e = est["models"][w["model"]]
        print(f"wrote {w['path']} ({w['bytes']} B) title {w['title']} embed {w['embed_sha256'][:12]}… "
              f"{e['n_episodes']} eps est {e['est_session_h']} h (high {e['est_session_h_high']})")
    print(f"total {est['total_est_session_h']} GPU-h (high {est['total_est_session_h_high']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
