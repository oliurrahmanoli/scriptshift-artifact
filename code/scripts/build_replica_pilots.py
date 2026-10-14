"""Generate the 2-replica throughput pilots (D21): Granite-4.2-8B, Ministral-3-14B, gpt-oss-20b.

Same protocol as pilot Version #1 (D17) and the Gemma pilot (D20): llama.cpp b11344,
AgentDojo 0.1.35 no-attack utility on banking (16) + slack (21) = 37 episodes,
`-c 24576 -np 4 --jinja --temp 0`, Path logdir + OutputLogger, flat runs/{label},
strict parse check (D18), runs inventory/manifest (D19), heartbeat (D16), GGUF in scratch
(D13), size + SHA-256 assertion, machine-readable result marker.

New here: plan §11.3 layout "one replica per T4" -> 2 llama-server processes pinned with
CUDA_VISIBLE_DEVICES=0/1 on ports 8099/8100; the 37 tasks are split round-robin over
per-replica worker threads (each thread gets its own AgentDojo logger stack, because
agentdojo.logging.LOGGER_STACK is a ContextVar whose default list is shared across threads).

GGUFs verified via HF API + GGUF header range-read on 2026-10-06 (Sydney); see DECISIONS D21.
Run from code/:  python3 scripts/build_replica_pilots.py
"""
import json
from pathlib import Path

import nbformat as nbf

import build_notebooks as bn

OUT = Path(__file__).resolve().parent.parent / "kaggle"

MODELS = {
    "granite": dict(
        nb="pilot_granite_2rep_throughput.ipynb",
        title="Granite-4.2-8B Q4_K_M, 2 replicas (one per T4)",
        plan="§11.3 O5 / §11.6 (Granite 2 replicas) / §11.8",
        label="granite-4.2-8b-2rep",
        repo="ibm-granite/granite-4.2-8b-GGUF",
        rev="93f3f6a8938ee922b784cf4e5b4203cd3428df8f",
        file="granite-4.2-8b-Q4_K_M.gguf",
        size=5347917952,
        sha="16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311",
        min_free=8.0,
        settings_md=(
            "**Thinking OFF** (plan §11.3 O5 'use non-thinking mode'). The GGUF's embedded chat "
            "template defaults `enable_thinking=True`, and pilot Version #1 (D17, 31.5 eps/h) passed "
            "no template kwargs, so **Version #1 ran with thinking ON** and without GPU pinning "
            "(llama.cpp default layer split over both visible T4s). This pilot changes two things at "
            "once (2 pinned replicas + thinking off); the optional `RUN_THINKING_ON_ABLATION` phase "
            "separates them if you want the attribution. Temperature 0 vs card 1.0 / top_p 0.95 is "
            "the D17 deviation, unchanged."),
        variants=[
            ("no_think_kwargs", ["--chat-template-kwargs", '{"enable_thinking":false}'], 24576, 4),
            ("reasoning_off", ["--reasoning", "off"], 24576, 4),
            ("plain_THINKING_ON", [], 24576, 4),
        ],
        template_expect="<think></think>",
        concurrency_probe=True,
        thinking_ablation=("thinking_on", ["--chat-template-kwargs", '{"enable_thinking":true}']),
        assumed=[],
    ),
    "ministral": dict(
        nb="pilot_ministral_throughput.ipynb",
        title="Ministral-3-14B-Instruct-2512 Q4_K_M, 2 replicas (one per T4)",
        plan="§11.3 O4 / §11.6 / §11.8",
        label="ministral-3-14b",
        repo="mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        rev="74fac473c43357d7fb2671713608183cc72496d0",
        file="Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
        size=8239593024,
        sha="824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613",
        min_free=10.0,
        settings_md=(
            "Instruct model, **no reasoning mode** (plan §11.3 O4); native function calling via the "
            "GGUF's embedded Mistral template (`[AVAILABLE_TOOLS]` / `[TOOL_CALLS]`) parsed by "
            "llama.cpp `--jinja`. Temperature 0 (Mistral suggests ~0.1; logged deviation). The "
            "`*-mmproj.gguf` vision projector is **not** downloaded. Est. VRAM per T4: 7.7 GiB "
            "weights + ~3.8 GiB KV at 24,576 ctx ≈ 11.5 GiB. Fallback variant halves ctx if OOM."),
        variants=[
            ("plain", [], 24576, 4),
            ("ctx16k_np2", [], 16384, 2),
        ],
        template_expect="[SYSTEM_PROMPT]",
        concurrency_probe=False,
        thinking_ablation=None,
        assumed=[],
    ),
    "gptoss": dict(
        nb="pilot_gptoss_throughput.ipynb",
        title="gpt-oss-20b MXFP4, 2 replicas (one per T4), reasoning effort low",
        plan="§11.3 O3 / §11.6 / §11.8",
        label="gpt-oss-20b",
        repo="ggml-org/gpt-oss-20b-GGUF",
        rev="ef9b12f2ff56c69cf32153a02784e7a3c88bf524",
        file="gpt-oss-20b-MXFP4.gguf",
        size=12109566624,
        sha="27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
        min_free=14.0,
        settings_md=(
            "**Reasoning effort low** (plan §11.3 O3; it cannot be switched fully off) via the "
            "harmony template's `reasoning_effort` kwarg; llama.cpp parses harmony tool calls with "
            "`--jinja` and returns reasoning separately. Temperature 0 (card suggests 1.0; logged "
            "deviation). **ASSUMED repo:** OpenAI publishes no GGUF itself; `ggml-org/gpt-oss-20b-GGUF` "
            "is the llama.cpp maintainers' native-MXFP4 conversion (Apache-2.0, not gated) — the "
            "plan's 'native MXFP4 (~13 GB)'. Est. VRAM per T4: 11.3 GiB weights + ~1.2 GiB KV — tight "
            "on 15 GB; fallbacks shrink ctx / slots, last resort is ONE server split 1,1 (a logged "
            "layout deviation, verdict forced to REVIEW)."),
        variants=[
            ("low_effort_kwargs", ["--chat-template-kwargs", '{"reasoning_effort":"low"}'], 24576, 4),
            ("low_effort_flag", ["--reasoning-effort", "low"], 24576, 4),
            ("low_effort_ctx16k_np2", ["--chat-template-kwargs", '{"reasoning_effort":"low"}'], 16384, 2),
        ],
        split_fallback=("low_effort_1server_split", ["--chat-template-kwargs", '{"reasoning_effort":"low"}'], 24576, 4),
        template_expect="Reasoning: low",
        concurrency_probe=False,
        thinking_ablation=None,
        assumed=["HF repo ggml-org/gpt-oss-20b-GGUF (OpenAI ships no official GGUF)"],
    ),
}


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(s):
    return nbf.v4.new_code_cell(s)


def config_cell(m):
    variants = repr(m["variants"])
    split_fb = repr(m.get("split_fallback"))
    abl = repr(m.get("thinking_ablation"))
    return f'''# ===== MODEL CONFIG (the only cell that differs between the three replica pilots) =====
MODEL_KEY   = {json.dumps(m["label"])}      # pipeline name -> traces in /kaggle/working/runs/<LABEL>/
LABEL       = MODEL_KEY
HF_REPO     = {json.dumps(m["repo"])}
HF_REV      = {json.dumps(m["rev"])}   # pinned commit (HF API, 2026-10-06)
HF_FILE     = {json.dumps(m["file"])}
HF_SIZE     = {m["size"]}            # bytes (HF API lfs.size)
HF_SHA256   = {json.dumps(m["sha"])}
MIN_FREE_GIB = {m["min_free"]}

N_REPLICAS  = 2          # plan §11.3: one llama-server per T4, CUDA_VISIBLE_DEVICES pinned
BASE_PORT   = 8099       # replica r listens on 127.0.0.1:(BASE_PORT + r)
WORKERS_PER_REPLICA = 1  # primary measurement: 1 agent stream per replica (D17/D20 used 1 stream total)
# serve variants tried in order: (name, extra llama-server flags, ctx, n_slots). First that starts
# on ALL replicas wins; the name is printed as SERVE_VARIANT= and must be recorded in DECISIONS.
SERVE_VARIANTS = {variants}
# last resort (gpt-oss only): one server over both GPUs, tensor-split 1,1 -> layout deviation
SPLIT_FALLBACK = {split_fb}
TEMPLATE_EXPECT = {json.dumps(m["template_expect"])}   # substring the rendered prompt must contain

RUN_CONCURRENCY_PROBE = {m["concurrency_probe"]}   # optional phase B: 2 streams per replica (-np slots), label <LABEL>-w2
THINKING_ABLATION = {abl}   # (name, flags) for optional phase C
RUN_THINKING_ON_ABLATION = False   # Granite only: phase C = 2 replicas, thinking ON (attributes the Version-#1 gap)

SUITES = ['banking', 'slack']    # n = 16 + 21 = 37 episodes
BENCHMARK_VERSION = 'v1.2.2'
KEEP_EPH, UTIL_MIN, PARSE_MAX = 64.0, 0.20, 0.05   # D20 rule, reused for every new model
MAIN_GRID_EPISODES, LOAD_OVERHEAD = 1897, 1.10     # 300x6 + 97 utility per model (§11.6)
ASSUMED = {m["assumed"]!r}
print(f'{{LABEL}}: {{HF_REPO}}@{{HF_REV[:12]}} / {{HF_FILE}}  replicas={{N_REPLICAS}}', flush=True)
'''


GPU_CELL = '''import subprocess
print(subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],capture_output=True,text=True).stdout, flush=True)
_n = subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout.count('GPU ')
print(f'GPUs visible: {_n}', flush=True)
assert _n >= 2, 'Need Accelerator = GPU T4 x2 (one replica per GPU)'
'''

CLEAN_CELL = '''# Free /kaggle/working of any stale GGUFs / partial downloads (D13).
import os, shutil, subprocess
WORKING_MODELS = '/kaggle/working/models'
removed = []
if os.path.isdir(WORKING_MODELS):
    for root, dirs, files in os.walk(WORKING_MODELS, topdown=False):
        for name in files:
            if name.endswith(('.gguf', '.incomplete', '.lock')):
                fp = os.path.join(root, name)
                try: os.remove(fp); removed.append(fp)
                except FileNotFoundError: pass
print('removed from /kaggle/working/models:', removed or '(nothing)', flush=True)
print(subprocess.run(['df','-h','/kaggle/working','/tmp'], capture_output=True, text=True).stdout, flush=True)
'''

DOWNLOAD_CELL = '''import os, hashlib, shutil, time
from huggingface_hub import hf_hub_download

for candidate in ('/kaggle/tmp/models', '/tmp/models'):
    parent = os.path.dirname(candidate)
    if os.path.isdir(parent) or parent == '/tmp':
        MODELS_DIR = candidate
        break
else:
    MODELS_DIR = '/tmp/models'
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ['HF_HUB_CACHE'] = MODELS_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = MODELS_DIR
free = shutil.disk_usage(MODELS_DIR).free / (1024**3)
print(f'MODELS_DIR={MODELS_DIR}  free={free:.1f} GiB', flush=True)
assert free > MIN_FREE_GIB, f'scratch too small for {HF_FILE}'
assert not MODELS_DIR.startswith('/kaggle/working'), 'GGUF must not go to /kaggle/working (D13)'

t0 = time.time()
print(f'downloading {HF_REPO}/{HF_FILE} @ {HF_REV[:12]} ...', flush=True)
MODEL_PATH = hf_hub_download(HF_REPO, HF_FILE, revision=HF_REV, local_dir=MODELS_DIR)
sz = os.path.getsize(MODEL_PATH)
print(f'{LABEL} -> {MODEL_PATH}  {sz/(1024**3):.2f} GiB  ({time.time()-t0:.0f}s)', flush=True)
assert sz == HF_SIZE, f'size mismatch {sz} != {HF_SIZE}'

def sha256(p, n=1<<22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while (b := f.read(n)): h.update(b)
    return h.hexdigest()
MODEL_SHA = sha256(MODEL_PATH)
print(f'sha256 {MODEL_SHA}  match={MODEL_SHA == HF_SHA256}', flush=True)
assert MODEL_SHA == HF_SHA256, 'GGUF sha256 mismatch'
'''

SERVE_CELL = '''import subprocess, time, os, requests, json
SERVE_USED = None
PROCS = []

def _start(model_path, port, gpu, extra, ctx, nslots, split=None):
    cmd = [SERVER,'-m',model_path,'--host','127.0.0.1','--port',str(port),
           '-ngl','99','-c',str(ctx),'-np',str(nslots),'--jinja','--temp','0','--seed','20261002']
    env = dict(os.environ)
    if split:
        cmd += ['--tensor-split', split]
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)   # pin replica r to GPU r
    cmd += list(extra)
    return cmd, env

def _wait(p, port, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return False
        try:
            if requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5).ok and p.poll() is None:
                return True
        except Exception: pass
        time.sleep(3)
    return False

def _port_busy(port):
    try:
        requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=2)
        return True
    except Exception:
        return False

def stop_all():
    for p in PROCS:
        p.terminate()
    for p in PROCS:
        try: p.wait(timeout=60)
        except Exception: p.kill()
    PROCS.clear()
    time.sleep(5)

def check_template(port):
    """Render a 1-turn chat through the server's own template; prove kwargs took effect."""
    try:
        r = requests.post(f'http://127.0.0.1:{port}/apply-template', timeout=30,
                          json={'messages': [{'role': 'user', 'content': 'ping'}]})
        prompt = r.json().get('prompt', '')
    except Exception as e:
        prompt = f'<apply-template failed: {e}>'
    ok = TEMPLATE_EXPECT in prompt
    print(f'template[{port}] tail: {prompt[-200:]!r}  expect={TEMPLATE_EXPECT!r} ok={ok}', flush=True)
    return ok, prompt[-400:]

def serve_replicas(model_path, variants, n_replicas=N_REPLICAS, timeout_s=600):
    """Start n_replicas llama-servers (GPU r -> port BASE_PORT+r). Returns list of ports."""
    global SERVE_USED
    for name, extra, ctx, nslots in variants:
        stop_all()
        ok_all, cmds = True, []
        for r in range(n_replicas):
            port = BASE_PORT + r
            assert not _port_busy(port), f'port {port} already serving (stale llama-server?)'
            cmd, env = _start(model_path, port, r, extra, ctx, nslots)
            logf = f'/kaggle/working/server_{port}_{name}.log'
            print(f'serve[{name}] replica {r} GPU {r}: ' + ' '.join(cmd[1:]), flush=True)
            p = subprocess.Popen(cmd, env=env, stdout=open(logf,'w'), stderr=subprocess.STDOUT)
            PROCS.append(p); cmds.append({'replica': r, 'gpu': r, 'port': port, 'cmd': cmd[1:], 'log': logf})
            t0 = time.time()
            if not _wait(p, port, timeout_s):
                print(f'serve[{name}] replica {r} FAILED (rc={p.poll()}); log tail:\\n{open(logf).read()[-1500:]}', flush=True)
                ok_all = False
                break
            print(f'  replica {r} up on :{port} ({time.time()-t0:.0f}s)', flush=True)
        if ok_all:
            tmpl_ok, tmpl_tail = check_template(BASE_PORT)
            SERVE_USED = {'variant': name, 'layout': f'{n_replicas}x replica (1 per T4)', 'ctx': ctx,
                          'n_slots': nslots, 'replicas': cmds, 'template_ok': tmpl_ok,
                          'template_tail': tmpl_tail}
            print(f'server up  SERVE_VARIANT={name}  replicas={n_replicas}', flush=True)
            try:
                print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv'],
                                     capture_output=True, text=True).stdout, flush=True)
            except Exception as e:
                print(f'nvidia-smi unavailable: {e}', flush=True)
            return [BASE_PORT + r for r in range(n_replicas)]
    if SPLIT_FALLBACK:
        name, extra, ctx, nslots = SPLIT_FALLBACK
        stop_all()
        cmd, env = _start(model_path, BASE_PORT, None, extra, ctx, nslots, split='1,1')
        logf = f'/kaggle/working/server_{BASE_PORT}_{name}.log'
        print(f'serve[{name}] (LAYOUT DEVIATION: 1 server, tensor-split 1,1): ' + ' '.join(cmd[1:]), flush=True)
        p = subprocess.Popen(cmd, env=env, stdout=open(logf,'w'), stderr=subprocess.STDOUT)
        PROCS.append(p)
        if _wait(p, BASE_PORT, timeout_s):
            tmpl_ok, tmpl_tail = check_template(BASE_PORT)
            SERVE_USED = {'variant': name, 'layout': '1 server tensor-split 1,1 (DEVIATION)', 'ctx': ctx,
                          'n_slots': nslots, 'replicas': [{'replica': 0, 'gpu': '0,1', 'port': BASE_PORT,
                          'cmd': cmd[1:], 'log': logf}], 'template_ok': tmpl_ok, 'template_tail': tmpl_tail}
            print(f'server up  SERVE_VARIANT={name}  replicas=1 (split)', flush=True)
            return [BASE_PORT]
        print(f'serve[{name}] FAILED; log tail:\\n{open(logf).read()[-1500:]}', flush=True)
    stop_all()
    raise RuntimeError('llama-server did not start with any variant; see server_*.log')
'''

IMPORTS_CELL = '''import sys, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import openai
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.benchmark import benchmark_suite_without_injections
from agentdojo.logging import OutputLogger, LOGGER_STACK
'''

RUN_CELL = '''def _heartbeat(label, t0, stop, every=300):
    d = Path('/kaggle/working/runs') / label
    while not stop.wait(every):
        n = len(list(d.rglob('*.json'))) if d.is_dir() else 0
        print(f'[heartbeat] {label}: {time.time()-t0:.0f}s elapsed, {n} json traces so far', flush=True)

def _make_pipe(port, label):
    client = openai.OpenAI(api_key='EMPTY', base_url=f'http://127.0.0.1:{port}/v1')
    mid = client.models.list().data[0].id
    llm = OpenAILLM(client, mid)
    pipe = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=mid,
           defense=None, system_message_name=None, system_message=None))
    pipe.name = label   # same name for every replica -> one flat runs/<label>/ tree (D19)
    return pipe, mid

def _worker(wid, port, label, suite, task_ids, logdir):
    # agentdojo LOGGER_STACK is a ContextVar whose *default list is shared between threads*;
    # give each worker its own stack so TraceLoggers of concurrent episodes never interleave.
    LOGGER_STACK.set([])
    pipe, _ = _make_pipe(port, label)
    util, errs = {}, []
    with OutputLogger(str(logdir)):
        for tid in task_ids:
            try:
                r = benchmark_suite_without_injections(pipe, suite, logdir=logdir, force_rerun=False,
                        user_tasks=[tid], benchmark_version=BENCHMARK_VERSION)
                util.update({k[0]: bool(v) for k, v in r['utility_results'].items()})
            except Exception as e:   # harness/API failure: count it, keep the commit alive
                util[tid] = False
                errs.append(f'{suite.name}/{tid}: {type(e).__name__}: {str(e)[:300]}')
                print(f'[worker {wid}] HARNESS ERROR {errs[-1]}', flush=True)
    return util, errs

def run_utility_replicas(label, ports, workers_per_replica=WORKERS_PER_REPLICA):
    stop = threading.Event()
    logdir = Path('/kaggle/working/runs'); logdir.mkdir(parents=True, exist_ok=True)
    _, mid = _make_pipe(ports[0], label)
    print(f'{label}: model id={mid}  ports={ports}  workers/replica={workers_per_replica}', flush=True)
    suites = get_suites(BENCHMARK_VERSION)
    slots = [p for p in ports for _ in range(workers_per_replica)]
    results, ns, secs, harness_errors, per_task = {}, {}, {}, [], {}
    t0 = time.time(); n_eps = 0
    threading.Thread(target=_heartbeat, args=(label, t0, stop), daemon=True).start()
    try:
        for s in SUITES:
            suite = suites[s]
            ids = list(suite.user_tasks.keys())
            chunks = [ids[i::len(slots)] for i in range(len(slots))]   # round-robin split
            print(f'{label}: starting {s} utility (no attack), {len(ids)} tasks over {len(slots)} streams...', flush=True)
            t_suite = time.time()
            with ThreadPoolExecutor(max_workers=len(slots)) as ex:
                futs = [ex.submit(_worker, w, slots[w], label, suite, chunks[w], logdir)
                        for w in range(len(slots)) if chunks[w]]
                u = {}
                for f in futs:
                    uu, ee = f.result(); u.update(uu); harness_errors += ee
            per_task[s] = u
            n_eps += len(u)
            results[s] = sum(u.values())/len(u) if u else float('nan')
            ns[s] = len(u); secs[s] = time.time()-t_suite
            print(f'{label}/{s}: utility={results[s]:.3f}, n={len(u)}, {secs[s]:.0f}s', flush=True)
    finally:
        stop.set()
    dt = sum(secs.values())   # suite-time-sum, same method as D17 / D20 anchors
    eph = n_eps/(dt/3600) if dt > 0 else 0.0
    print(f'{label}: utility {results}, episodes {n_eps}, {dt:.0f}s, {eph:.1f} episodes/h', flush=True)
    print(f'{label}: harness errors {len(harness_errors)}/{n_eps}', flush=True)
    model_dir = logdir / label
    n_json = sum(1 for _ in model_dir.rglob('*.json')) if model_dir.is_dir() else 0
    print(f'{label}: traces under {model_dir}: {n_json} json files', flush=True)
    return {'label': label, 'model_id': mid, 'utility': results, 'n': ns, 'suite_seconds': secs,
            'episodes': n_eps, 'wall_s': dt, 'eps_per_h': eph, 'streams': len(slots),
            'workers_per_replica': workers_per_replica, 'replicas': len(ports),
            'harness_errors': harness_errors, 'per_task_utility': per_task}
'''

PHASE_A_CELL = '''# Phase A (primary, decides the verdict): 2 replicas x 1 stream
PORTS = serve_replicas(MODEL_PATH, SERVE_VARIANTS)
PRIMARY_SERVE = dict(SERVE_USED)
RES = run_utility_replicas(LABEL, PORTS, WORKERS_PER_REPLICA)
'''

PHASE_BC_CELL = '''# Phase B (optional): same servers, 2 streams per replica (uses the -np slots). Not used for the verdict.
PROBE = None
if RUN_CONCURRENCY_PROBE and PRIMARY_SERVE['layout'].startswith(f'{N_REPLICAS}x'):
    PROBE = run_utility_replicas(f'{LABEL}-w2', PORTS, 2)
stop_all()
print(f'{LABEL}: servers stopped', flush=True)

# Phase C (optional, Granite): 2 replicas with thinking ON -> attributes the Version-#1 31.5 eps/h gap
ABLATION = None
if RUN_THINKING_ON_ABLATION and THINKING_ABLATION:
    name, flags = THINKING_ABLATION
    _ports = serve_replicas(MODEL_PATH, [(name, flags, PRIMARY_SERVE['ctx'], PRIMARY_SERVE['n_slots'])])
    ABLATION = run_utility_replicas(f'{LABEL}-{name}', _ports, WORKERS_PER_REPLICA)
    ABLATION['serve'] = dict(SERVE_USED)
    stop_all()
'''

TOOLSTATS_CELL = '''# Tool-use spot check (D20 caveat b): real tool calls per episode, messages per episode
import json, glob, statistics
def tool_stats(label):
    files = sorted(glob.glob(f'/kaggle/working/runs/{label}/**/*.json', recursive=True))
    tc, nm, dur = [], [], []
    for fp in files:
        try: d = json.load(open(fp))
        except Exception: continue
        if not isinstance(d, dict) or 'messages' not in d: continue
        msgs = d.get('messages') or []
        tc.append(sum(len(m.get('tool_calls') or []) for m in msgs if isinstance(m, dict) and m.get('role') == 'assistant'))
        nm.append(len(msgs))
        if isinstance(d.get('duration'), (int, float)): dur.append(d['duration'])
    out = {'episodes': len(tc), 'zero_tool_episodes': sum(1 for x in tc if x == 0),
           'tool_calls_min_median_max': [min(tc), statistics.median(tc), max(tc)] if tc else None,
           'messages_min_median_max': [min(nm), statistics.median(nm), max(nm)] if nm else None,
           'episode_seconds_median': statistics.median(dur) if dur else None}
    print(f'{label}: tool stats {out}', flush=True)
    return out
TOOL_STATS = tool_stats(LABEL)
'''

VERDICT_CELL = '''import json, math
def verdict_for(res, parse_rate, serve):
    u = res['utility']
    util_ok = all((v >= UTIL_MIN) for v in u.values()) and len(u) == len(SUITES)
    parse_ok = (not math.isnan(parse_rate)) and parse_rate <= PARSE_MAX
    harness_ok = len(res['harness_errors']) <= PARSE_MAX * max(res['episodes'], 1)
    eph_ok = res['eps_per_h'] >= KEEP_EPH
    layout_ok = serve['layout'].startswith(f'{N_REPLICAS}x')
    template_ok = bool(serve.get('template_ok'))
    # D20 caveat (b): utility must come from real tool use, not trivially short answers
    ts = TOOL_STATS or {}
    tools_ok = bool(ts.get('episodes')) and ts['zero_tool_episodes'] <= 0.25 * ts['episodes']
    if eph_ok and util_ok and parse_ok and harness_ok and layout_ok and template_ok and tools_ok:
        v = 'KEEP'
    elif util_ok and parse_ok and harness_ok and tools_ok and not eph_ok:
        v = 'REVIEW_SLOW'      # usable but below 64 eps/h -> shrink / drop / overflow decision (§11.6 levers)
    else:
        v = 'REVIEW'
    return v, dict(eph_ok=eph_ok, util_ok=util_ok, parse_ok=parse_ok, harness_ok=harness_ok,
                   layout_ok=layout_ok, template_ok=template_ok, tools_ok=tools_ok)

VERDICT, CHECKS = verdict_for(RES, PARSE_RATE, PRIMARY_SERVE)
grid_h = MAIN_GRID_EPISODES / RES['eps_per_h'] * LOAD_OVERHEAD if RES['eps_per_h'] > 0 else float('inf')
out = dict(RES, parse_error_rate=PARSE_RATE, tool_stats=TOOL_STATS, serve=PRIMARY_SERVE,
           gguf={'repo': HF_REPO, 'rev': HF_REV, 'file': HF_FILE, 'size': HF_SIZE, 'sha256': MODEL_SHA},
           llama_build=LLAMA_BUILD, rule={'keep_eps_per_h': KEEP_EPH, 'util_min': UTIL_MIN, 'parse_max': PARSE_MAX},
           checks=CHECKS, verdict=VERDICT, main_grid_hours_x1_10=grid_h, assumed=ASSUMED,
           concurrency_probe=PROBE, thinking_on_ablation=ABLATION)
fn = f'/kaggle/working/{LABEL}_pilot_result.json'
open(fn, 'w').write(json.dumps(out, indent=2, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'per_task_utility'}, indent=2, default=str))
u = RES['utility']
print(f"PILOT_RESULT model={LABEL} eps_per_h={RES['eps_per_h']:.1f} "
      f"banking={u.get('banking', float('nan')):.3f} slack={u.get('slack', float('nan')):.3f} "
      f"parse={PARSE_RATE:.1%} harness_err={len(RES['harness_errors'])} variant={PRIMARY_SERVE['variant']} "
      f"layout={PRIMARY_SERVE['layout'].replace(' ', '_')} template_ok={PRIMARY_SERVE.get('template_ok')} "
      f"grid_h={grid_h:.1f} verdict={VERDICT}", flush=True)
if PROBE:
    print(f"PILOT_PROBE model={PROBE['label']} streams={PROBE['streams']} eps_per_h={PROBE['eps_per_h']:.1f} "
          f"banking={PROBE['utility'].get('banking', float('nan')):.3f} slack={PROBE['utility'].get('slack', float('nan')):.3f}", flush=True)
if ABLATION:
    print(f"PILOT_ABLATION model={ABLATION['label']} eps_per_h={ABLATION['eps_per_h']:.1f} "
          f"banking={ABLATION['utility'].get('banking', float('nan')):.3f} slack={ABLATION['utility'].get('slack', float('nan')):.3f}", flush=True)
print(f'wrote {fn}', flush=True)
'''


def replica_nb(key):
    m = MODELS[key]
    base = bn.pilot_nb()
    install_llama, pip_cell = base.cells[3], base.cells[4]
    parse_cell = base.cells[14]
    inv_cell = base.cells[16]
    assert "LLAMA_BUILD" in install_llama.source and "agentdojo==0.1.35" in pip_cell.source
    assert "PARSE_MARKERS" in parse_cell.source and "runs_manifest.json" in inv_cell.source
    # parse cell from pilot_nb ends by looping over Granite/Qwen labels; replace that tail
    psrc = parse_cell.source
    cut = psrc.index("for lab in ")
    psrc = psrc[:cut].rstrip() + "\n\nPARSE_RATE = parse_error_rate(LABEL)\n"
    psrc += ("PARSE_RATE_PROBE = parse_error_rate(f'{LABEL}-w2') if RUN_CONCURRENCY_PROBE and PROBE else None\n")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    nb.cells = [
        md(f"# Pilot: {m['title']}\n"
           f"Research plan {m['plan']}. Same protocol as pilot Version #1 and the Gemma pilot "
           "(DECISIONS D17/D20): llama.cpp **b11344** `llama-server`, AgentDojo 0.1.35 **no-attack "
           "utility** on `banking` (n=16) + `slack` (n=21) = 37 episodes, `-c 24576 -np 4 --jinja "
           "--temp 0 --seed 20261002`.\n\n"
           f"**Layout:** 2 replicas — one `llama-server` per T4 (`CUDA_VISIBLE_DEVICES=0` → :8099, "
           "`=1` → :8100); the 37 tasks are split round-robin over one agent stream per replica.\n\n"
           f"**Model:** `{m['repo']}` → `{m['file']}` ({m['size']/1e9:.2f} GB, Apache-2.0, not gated), "
           f"pinned revision `{m['rev'][:12]}`, SHA-256 `{m['sha'][:16]}…` asserted.\n\n"
           f"{m['settings_md']}\n\n"
           "**Decision rule (D20, reused):** KEEP if episodes/h ≥ 64 **and** utility ≥ 0.20 on each "
           "suite **and** strict parse-error rate ≤ 5% (and both replicas up, template kwargs verified). "
           "`REVIEW_SLOW` = usable but < 64 eps/h → shrink/drop decision, not automatic drop.\n\n"
           "**Terms:** self-hosted Apache-2.0 weights; fictional AgentDojo data; servers on 127.0.0.1 "
           "only; no tunnels. Keep this notebook **private**. One Kaggle account.\n\n"
           "Settings: Accelerator **GPU T4 x2**, Internet **on**. Then **Save Version → Save & Run All (Commit)**."),
        md("## 0. Model config\nEverything model-specific is in this one cell."),
        code(config_cell(m)),
        code(GPU_CELL),
        md("## 1. Install pinned llama.cpp (CUDA 12.8) and agentdojo"),
        install_llama,
        pip_cell,
        md("## 2. Download the GGUF to scratch (D13)\n"
           "Weights go to `/kaggle/tmp/models` if it exists, else `/tmp/models` — **never** "
           "`/kaggle/working`. Size and SHA-256 are asserted against the HF API values."),
        code(CLEAN_CELL),
        code(DOWNLOAD_CELL),
        md("## 3. Serve 2 replicas (one per T4)\n"
           "`serve_replicas()` tries `SERVE_VARIANTS` in order and prints `SERVE_VARIANT=`; any "
           "fallback must be logged in DECISIONS.md. After start it renders a test prompt through "
           "`/apply-template` to prove the template kwargs (thinking / reasoning effort) took effect."),
        code(SERVE_CELL),
        code(IMPORTS_CELL),
        md("## 4. No-attack utility across replicas\n"
           "Each worker thread gets its own AgentDojo logger stack + `OutputLogger` (Path logdir, D14/D15); "
           "traces land flat in `/kaggle/working/runs/<LABEL>/<suite>/…` (D19). Heartbeat every 5 min "
           "with `flush=True` (D16). eps/h uses the suite-time-sum method of D17/D20."),
        code(RUN_CELL),
        code(PHASE_A_CELL),
        code(PHASE_BC_CELL),
        md("## 5. Strict tool-call parse-error rate (D18)\nDoes **not** flag the bare substring `\"error\"`."),
        code(psrc),
        code(TOOLSTATS_CELL),
        md("## 6. Trace inventory (keep JSON in Kaggle Output, D19)"),
        inv_cell,
        md("## 7. Result + verdict\nWrites `/kaggle/working/<LABEL>_pilot_result.json` and one "
           "`PILOT_RESULT … verdict=` line to copy into DECISIONS.md."),
        code(VERDICT_CELL),
    ]
    return nb


def main():
    for key, m in MODELS.items():
        nb = replica_nb(key)
        nbf.validate(nb)
        nbf.write(nb, OUT / m["nb"])
        print("wrote", OUT / m["nb"], f"({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
