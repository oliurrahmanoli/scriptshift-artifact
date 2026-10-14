"""Generate the two ready-to-run Kaggle notebooks into kaggle/ as .ipynb.

Repo names / files were verified against the Hugging Face API on 2026-10-02:
  * ibm-granite/granite-4.2-8b-GGUF   -> granite-4.2-8b-Q4_K_M.gguf (5.35 GB), Apache-2.0
  * ggml-org/Qwen3.6-35B-A3B-GGUF     -> Qwen3.6-35B-A3B-Q4_K_M.gguf (20.42 GB), Apache-2.0
Both repos and files exist. (Alt Qwen GGUF: unsloth/Qwen3.6-35B-A3B-GGUF
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf, 22.13 GB.) See DECISIONS.md D8.
"""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "kaggle"
LLAMA_BUILD = "b11344"
LLAMA_CUDA = ("https://github.com/ggml-org/llama.cpp/releases/download/"
              f"{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz")
CUDART = ("https://github.com/ggml-org/llama.cpp/releases/download/"
          f"{LLAMA_BUILD}/cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def pilot_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Pilot: throughput + tool-call error rate on Kaggle 2xT4\n"
           "Research plan section 11.6 / 11.10 / 11.14. Measures episodes per session-hour "
           "and the tool-call parse-error rate for two open models served with llama.cpp, "
           "running AgentDojo's **no-attack utility** on `banking` and `slack`.\n\n"
           "**Terms**: self-hosted Apache-2.0 weights; benchmark on fictional AgentDojo data; "
           "server stays on localhost; no tunnels (Kaggle AUP, section 11.2). Keep the "
           "notebook and any dataset **private**. One Kaggle account only.\n\n"
           "Enable GPU **T4 x2** in the sidebar before running."),
        code("import subprocess\n"
             "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',"
             "'--format=csv'],capture_output=True,text=True).stdout)"),
        md("## 1. Install pinned llama.cpp (CUDA 12.8) and agentdojo"),
        code(f"LLAMA_BUILD = '{LLAMA_BUILD}'\n"
             f"CUDA_TARBALL = '{LLAMA_CUDA}'\n"
             f"CUDART_TARBALL = '{CUDART}'\n"
             "import os, subprocess\n"
             "os.makedirs('/kaggle/working/llama', exist_ok=True)\n"
             "for url in (CUDART_TARBALL, CUDA_TARBALL):\n"
             "    fn = url.split('/')[-1]\n"
             "    subprocess.run(['wget','-q','-O',f'/kaggle/working/llama/{fn}',url], check=True)\n"
             "    subprocess.run(['tar','xzf',f'/kaggle/working/llama/{fn}','-C','/kaggle/working/llama'], check=True)\n"
             "SERVER = '/kaggle/working/llama/build/bin/llama-server'\n"
             "import glob\n"
             "cands = glob.glob('/kaggle/working/llama/**/llama-server', recursive=True)\n"
             "SERVER = cands[0]; print('llama-server at', SERVER)\n"
             "print(subprocess.run([SERVER,'--version'],capture_output=True,text=True).stderr[:200])"),
        code("# pin agentdojo and deps\n"
             "!pip -q install agentdojo==0.1.35 openai==3.23.0"),
        md("## 2. Download the GGUFs (verified real repos, Apache-2.0)\n"
           "* `ibm-granite/granite-4.2-8b-GGUF` -> `granite-4.2-8b-Q4_K_M.gguf` (~5.35 GB)\n"
           "* `ggml-org/Qwen3.6-35B-A3B-GGUF` -> `Qwen3.6-35B-A3B-Q4_K_M.gguf` (~20.4 GB)\n"
           "\n"
           "**Disk layout (2026-10-06 fix):** `/kaggle/working` is capped at ~20 GB and is "
           "reserved for `llama-server` binaries, run logs, and small outputs. Large GGUFs go "
           "under **scratch** `MODELS_DIR` = `/kaggle/tmp/models` if that exists, else "
           "`/tmp/models` (session-only). Do **not** download weights into "
           "`/kaggle/working/models` — a prior Run All filled working with Granite then died at "
           "~67% of Qwen with ~13.7 GB free. Prefer Q4_K_M; only fall back to a smaller Qwen "
           "quant if scratch still cannot hold it (log any substitution in DECISIONS.md). "
           "If an exact file name has changed, list the repo and pick the nearest Q4_K_M."),
        code("# Free /kaggle/working space left by a prior failed Qwen (or other) GGUF download.\n"
             "import os, shutil, glob, subprocess\n"
             "WORKING_MODELS = '/kaggle/working/models'\n"
             "removed = []\n"
             "if os.path.isdir(WORKING_MODELS):\n"
             "    for pat in ['Qwen*', 'qwen*', '*.incomplete', '*.lock', '**/*Qwen*', '**/*.incomplete']:\n"
             "        for fp in glob.glob(os.path.join(WORKING_MODELS, pat), recursive=True):\n"
             "            try:\n"
             "                if os.path.isdir(fp):\n"
             "                    shutil.rmtree(fp); removed.append(fp + '/')\n"
             "                elif os.path.isfile(fp) or os.path.islink(fp):\n"
             "                    os.remove(fp); removed.append(fp)\n"
             "            except FileNotFoundError:\n"
             "                pass\n"
             "    for root, dirs, files in os.walk(WORKING_MODELS, topdown=False):\n"
             "        for name in files + dirs:\n"
             "            if 'qwen' in name.lower() or name.endswith('.incomplete'):\n"
             "                fp = os.path.join(root, name)\n"
             "                try:\n"
             "                    if os.path.isdir(fp): shutil.rmtree(fp)\n"
             "                    else: os.remove(fp)\n"
             "                    if fp not in removed: removed.append(fp)\n"
             "                except FileNotFoundError:\n"
             "                    pass\n"
             "print('removed from /kaggle/working/models:', removed or '(nothing)')\n"
             "print(subprocess.run(['df','-h','/kaggle/working','/tmp','/kaggle/tmp'],\n"
             "                     capture_output=True, text=True).stdout)\n"),
        code("import os, hashlib, shutil, subprocess\n"
             "from huggingface_hub import hf_hub_download\n"
             "\n"
             "# Scratch for large weights (NOT /kaggle/working — ~20 GB cap caused prior ENOSPC)\n"
             "for candidate in ('/kaggle/tmp/models', '/tmp/models'):\n"
             "    parent = os.path.dirname(candidate)\n"
             "    if os.path.isdir(parent) or parent == '/tmp':\n"
             "        MODELS_DIR = candidate\n"
             "        break\n"
             "else:\n"
             "    MODELS_DIR = '/tmp/models'\n"
             "os.makedirs(MODELS_DIR, exist_ok=True)\n"
             "os.environ['HF_HUB_CACHE'] = MODELS_DIR\n"
             "os.environ['HUGGINGFACE_HUB_CACHE'] = MODELS_DIR\n"
             "\n"
             "def free_gb(path):\n"
             "    return shutil.disk_usage(path).free / (1024**3)\n"
             "\n"
             "print(f'MODELS_DIR={MODELS_DIR}  free={free_gb(MODELS_DIR):.1f} GiB')\n"
             "print(subprocess.run(['df','-h',MODELS_DIR,'/kaggle/working'],\n"
             "                     capture_output=True, text=True).stdout)\n"
             "\n"
             "QWEN_REPO = 'ggml-org/Qwen3.6-35B-A3B-GGUF'\n"
             "QWEN_FILE = 'Qwen3.6-35B-A3B-Q4_K_M.gguf'  # prefer Q4_K_M (~20.4 GB)\n"
             "if free_gb(MODELS_DIR) < 22.0:\n"
             "    QWEN_FILE = 'Qwen3.6-35B-A3B-Q3_K_M.gguf'\n"
             "    print(f'WARN: low scratch space; trying smaller quant {QWEN_FILE}')\n"
             "\n"
             "GRANITE = hf_hub_download('ibm-granite/granite-4.2-8b-GGUF',\n"
             "                          'granite-4.2-8b-Q4_K_M.gguf', local_dir=MODELS_DIR)\n"
             "print('Granite ->', GRANITE)\n"
             "QWEN = hf_hub_download(QWEN_REPO, QWEN_FILE, local_dir=MODELS_DIR)\n"
             "print('Qwen ->', QWEN)\n"
             "\n"
             "def sha256(p, n=1<<20):\n"
             "    h=hashlib.sha256()\n"
             "    with open(p,'rb') as f:\n"
             "        while (b:=f.read(n)): h.update(b)\n"
             "    return h.hexdigest()\n"
             "for p in (GRANITE, QWEN):\n"
             "    print(p, f'{os.path.getsize(p)/(1024**3):.2f} GiB', sha256(p)[:16])"),
        md("## 3. Serve a model and run no-attack utility\n"
           "Granite 4.2-8B fits on one T4; Qwen3.6-35B-A3B (Q4_K_M ~20 GB) is split over both."),
        code("import subprocess, time, os, requests, signal\n"
             "def serve(model_path, ngl=99, ctx=24576, nslots=4, split=None, port=8099):\n"
             "    env = dict(os.environ)\n"
             "    cmd = [SERVER,'-m',model_path,'--host','127.0.0.1','--port',str(port),\n"
             "           '-ngl',str(ngl),'-c',str(ctx),'-np',str(nslots),'--jinja','--temp','0']\n"
             "    if split: cmd += ['--tensor-split', split]\n"
             "    log = open(f'/kaggle/working/server_{port}.log','w')\n"
             "    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)\n"
             "    for _ in range(180):\n"
             "        try:\n"
             "            if requests.get(f'http://127.0.0.1:{port}/v1/models').ok:\n"
             "                print('server up'); return p\n"
             "        except Exception: pass\n"
             "        time.sleep(2)\n"
             "    raise RuntimeError('server did not start; check the log')\n"),
        code("# attach this repo (add it as a Kaggle dataset or git clone) so attacks/scoring import\n"
             "import sys; sys.path.insert(0, '/kaggle/working/code')  # adjust to where you put code/\n"
             "import os; os.environ['LOCAL_LLM_PORT']='8099'\n"
             "from agentdojo.task_suite.load_suites import get_suites\n"
             "from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig\n"
             "from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM\n"
             "from agentdojo.benchmark import benchmark_suite_without_injections\n"
             "import openai, time\n"),
        code("def run_utility(model_path, label, split=None):\n"
             "    from pathlib import Path\n"
             "    from agentdojo.logging import OutputLogger\n"
             "    print(f'{label}: loading weights / starting llama-server...', flush=True)\n"
             "    proc = serve(model_path, split=split)\n"
             "    try:\n"
             "        client = openai.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8099/v1')\n"
             "        mid = client.models.list().data[0].id\n"
             "        print(f'{label}: model id={mid}', flush=True)\n"
             "        llm = OpenAILLM(client, mid)\n"
             "        pipe = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=mid,\n"
             "               defense=None, system_message_name=None, system_message=None))\n"
             "        pipe.name = label\n"
             "        suites = get_suites('v1.2.2')\n"
             "        results = {}\n"
             "        t0 = time.time(); n_eps = 0\n"
             "        # Official layout: runs/{pipeline_name}/suite/... (D19). Do not nest label in logdir.\n"
             "        logdir = Path('/kaggle/working/runs')\n"
             "        logdir.mkdir(parents=True, exist_ok=True)\n"
             "        with OutputLogger(str(logdir)):\n"
             "            for s in ['banking','slack']:\n"
             "                print(f'{label}: starting {s} utility (no attack)...', flush=True)\n"
             "                t_suite = time.time()\n"
             "                r = benchmark_suite_without_injections(pipe, suites[s],\n"
             "                        logdir=logdir, force_rerun=False,\n"
             "                        benchmark_version='v1.2.2')\n"
             "                u = list(r['utility_results'].values())\n"
             "                n_eps += len(u)\n"
             "                results[s] = sum(u)/len(u) if u else float('nan')\n"
             "                print(f'{label}/{s}: utility={results[s]:.3f}, n={len(u)}, {time.time()-t_suite:.0f}s', flush=True)\n"
             "        dt = time.time()-t0\n"
             "        eph = n_eps/(dt/3600) if dt > 0 else 0.0\n"
             "        print(f'{label}: utility {results}, episodes {n_eps}, {dt:.0f}s, {eph:.1f} episodes/h', flush=True)\n"
             "        model_dir = logdir / label\n"
             "        n_json = nbytes = 0\n"
             "        if model_dir.is_dir():\n"
             "            for fp in model_dir.rglob('*.json'):\n"
             "                n_json += 1\n"
             "                try: nbytes += fp.stat().st_size\n"
             "                except OSError: pass\n"
             "        print(f'{label}: traces under {model_dir}: {n_json} json files, {nbytes/1024:.1f} KiB', flush=True)\n"
             "        return results, eph\n"
             "    finally:\n"
             "        proc.terminate()\n"
             "        print(f'{label}: server stopped', flush=True)\n"),
        code("# Granite (1 GPU) then Qwen (split across 2 GPUs)\n"
             "run_utility(GRANITE, 'granite-4.2-8b')\n"
             "run_utility(QWEN, 'qwen3.6-35b-a3b', split='1,1')"),
        md("## 4. Tool-call parse-error rate\n"
           "Strict check aligned with AgentDojo EmptyFunctionCall / tool-call parse failure "
           "markers (research-plan §11.8: replace a model if parse errors >5%). Does **not** "
           "flag the bare substring `\"error\"` (that false-positive flagged 37/37 in pilot "
           "Version #1 — D18). Scans JSON under `/kaggle/working/runs/{label}/`."),
        code("import json, glob, re\n"
             "from pathlib import Path\n"
             "\n"
             "# Precise markers only. Do NOT match bare 'error' (D18).\n"
             "PARSE_MARKERS = (\n"
             "    'EmptyFunctionCall',\n"
             "    'could not parse function call',\n"
             "    'Invalid function call',\n"
             "    'Invalid function calling output',\n"
             "    'ASTParsingError',\n"
             "    'Error parsing the model output',\n"
             "    '<empty-function-name>',\n"
             "    'could not parse',\n"
             ")\n"
             "_MARKER_RE = re.compile('|'.join(re.escape(m) for m in PARSE_MARKERS), re.IGNORECASE)\n"
             "\n"
             "def _episode_has_parse_error(d):\n"
             "    top = d.get('error')\n"
             "    if isinstance(top, str) and top and _MARKER_RE.search(top):\n"
             "        return True\n"
             "    for msg in d.get('messages') or []:\n"
             "        if not isinstance(msg, dict):\n"
             "            continue\n"
             "        err = msg.get('error')\n"
             "        if isinstance(err, str) and err and _MARKER_RE.search(err):\n"
             "            return True\n"
             "        content = msg.get('content')\n"
             "        chunks = []\n"
             "        if isinstance(content, str):\n"
             "            chunks.append(content)\n"
             "        elif isinstance(content, list):\n"
             "            for b in content:\n"
             "                if isinstance(b, dict) and isinstance(b.get('content'), str):\n"
             "                    chunks.append(b['content'])\n"
             "                elif isinstance(b, str):\n"
             "                    chunks.append(b)\n"
             "        for c in chunks:\n"
             "            if _MARKER_RE.search(c):\n"
             "                return True\n"
             "    return False\n"
             "\n"
             "def parse_error_rate(label, threshold=0.05):\n"
             "    files = sorted(glob.glob(f'/kaggle/working/runs/{label}/**/*.json', recursive=True))\n"
             "    tot = err = 0\n"
             "    flagged = []\n"
             "    for fp in files:\n"
             "        try:\n"
             "            d = json.load(open(fp))\n"
             "        except Exception:\n"
             "            continue\n"
             "        if not isinstance(d, dict) or 'messages' not in d:\n"
             "            continue\n"
             "        tot += 1\n"
             "        if _episode_has_parse_error(d):\n"
             "            err += 1\n"
             "            flagged.append(fp)\n"
             "    rate = err / tot if tot else float('nan')\n"
             "    over = (rate > threshold) if tot else False\n"
             "    print(f\"{label}: parse-errors {err}/{tot} = {rate:.1%}  \"\n"
             "          f\"({'EXCEEDS' if over else 'within'} §11.8 {threshold:.0%} replacement rule)\")\n"
             "    if flagged[:5]:\n"
             "        print(f'  examples: {flagged[:5]}')\n"
             "    return rate\n"
             "\n"
             "for lab in ['granite-4.2-8b', 'qwen3.6-35b-a3b']:\n"
             "    parse_error_rate(lab)\n"),
        md("## 5. Trace inventory (keep JSON in Kaggle Output)\n"
           "List `/kaggle/working/runs` so Save & Run All commits per-episode AgentDojo JSON (D19)."),
        code("import json, os\n"
             "from pathlib import Path\n"
             "\n"
             "runs = Path('/kaggle/working/runs')\n"
             "manifest = {'runs_root': str(runs), 'models': {}}\n"
             "print(f'runs root exists={runs.is_dir()}')\n"
             "if runs.is_dir():\n"
             "    for model_dir in sorted(p for p in runs.iterdir() if p.is_dir()):\n"
             "        files = list(model_dir.rglob('*.json'))\n"
             "        nbytes = sum(f.stat().st_size for f in files if f.is_file())\n"
             "        manifest['models'][model_dir.name] = {\n"
             "            'n_json': len(files), 'bytes': nbytes,\n"
             "            'sample': [str(f.relative_to(runs)) for f in files[:5]],\n"
             "        }\n"
             "        print(f'  {model_dir.name}: {len(files)} json, {nbytes/1024:.1f} KiB')\n"
             "        for f in files[:3]:\n"
             "            print(f'    e.g. {f.relative_to(runs)}')\n"
             "else:\n"
             "    print('  WARNING: /kaggle/working/runs missing — traces will not appear in Output')\n"
             "\n"
             "man_path = Path('/kaggle/working/runs_manifest.json')\n"
             "man_path.write_text(json.dumps(manifest, indent=2))\n"
             "print(f'wrote {man_path} ({man_path.stat().st_size} bytes)')\n"
             "print('/kaggle/working top-level:')\n"
             "for p in sorted(Path('/kaggle/working').iterdir()):\n"
             "    if p.name.startswith('.'): continue\n"
             "    print(f\"  {'dir' if p.is_dir() else 'file'} {p.name}\")\n"),
        md("### Record in DECISIONS.md\n"
           "Measured episodes/session-hour per model, no-attack utility on banking+slack, "
           "**strict** tool-call parse-error rate (err/tot and whether >5%), and the Gemma "
           "go/no-go draft. Re-run `scoop/cost_noapproval.py` with the measured rates "
           "(section 11.6). See D17–D19."),
    ]
    return nb


# --- Gemma-only pilot (D20). GGUF verified via HF API on 2026-10-06 (Sydney):
#   google/gemma-4-26B-A4B-it-qat-q4_0-gguf @ d1c082be9cf3c8a514acf63b8761f4b41935842e
#   -> gemma-4-26B_q4_0-it.gguf  14,439,363,584 B (14.44 GB / 13.45 GiB)
#   sha256 3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d
#   licence apache-2.0 (GGUF general.license), NOT gated (no HF click-through needed).
#   GGUF header: general.architecture=gemma4, 30 layers, 128 experts / 8 active,
#   sliding_window=1024, embedded chat template with tool calls, enable_thinking default false.
GEMMA_REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
GEMMA_REV = "d1c082be9cf3c8a514acf63b8761f4b41935842e"
GEMMA_FILE = "gemma-4-26B_q4_0-it.gguf"
GEMMA_SIZE = 14439363584
GEMMA_SHA256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"
GEMMA_LABEL = "gemma-4-26b-a4b"
GEMMA_KEEP_EPH = 64.0


def gemma_pilot_nb():
    """Gemma-only clone of pilot_nb(): same protocol (banking+slack no-attack utility,
    llama.cpp b11344, -c 24576 -np 4 --jinja --temp 0, tensor-split 1,1) with the
    D13-D19 fixes, plus a background heartbeat and a machine-readable result marker."""
    base = pilot_nb()
    # reuse the exact install / agentdojo-import / strict-parse / inventory cells from pilot_nb
    install_llama, pip_cell = base.cells[3], base.cells[4]
    imports_cell = base.cells[10]
    parse_md, parse_cell = base.cells[13], base.cells[14]
    inv_md, inv_cell = base.cells[15], base.cells[16]
    assert "LLAMA_BUILD" in install_llama.source and "agentdojo==0.1.35" in pip_cell.source
    assert "benchmark_suite_without_injections" in imports_cell.source
    assert "PARSE_MARKERS" in parse_cell.source and "runs_manifest.json" in inv_cell.source
    parse_src = parse_cell.source.replace(
        "for lab in ['granite-4.2-8b', 'qwen3.6-35b-a3b']:\n    parse_error_rate(lab)\n",
        f"PARSE_RATE = parse_error_rate('{GEMMA_LABEL}')\n")
    assert parse_src != parse_cell.source

    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Pilot (Gemma only): throughput + tool-call error rate on Kaggle 2xT4\n"
           "Research plan §11.3 (O1) / §11.6 / §11.8. Same protocol as pilot Version #1 "
           "(Granite 31.5 eps/h, Qwen 171.4 eps/h — DECISIONS D17): llama.cpp **b11344** "
           "`llama-server`, AgentDojo 0.1.35 **no-attack utility** on `banking` (n=16) + "
           "`slack` (n=21) = 37 episodes, `-c 24576 -np 4 --jinja --temp 0`, "
           "tensor-split `1,1` over both T4s.\n\n"
           f"**Model:** `{GEMMA_REPO}` → `{GEMMA_FILE}` (Google official QAT Q4_0, "
           "14.44 GB, Apache-2.0, not gated). Gemma-only flags: `--swa-full` (plan §11.8, "
           "prefix reuse on sliding-window layers) and `--chat-template-kwargs "
           "{\"enable_thinking\":false}` (thinking off, plan §11.3).\n\n"
           f"**Decision rule (D20):** keep Gemma in the five if episodes/h ≥ {GEMMA_KEEP_EPH:.0f}, "
           "utility ≥ 20% and strict parse-error rate ≤ 5%.\n\n"
           "**Terms**: self-hosted Apache-2.0 weights; fictional AgentDojo data; server on "
           "localhost only; no tunnels. Keep this notebook **private**. One Kaggle account.\n\n"
           "Settings: Accelerator **GPU T4 x2**, Internet **on**. Then **Save Version → "
           "Save & Run All (Commit)**."),
        code("import subprocess\n"
             "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',"
             "'--format=csv'],capture_output=True,text=True).stdout, flush=True)\n"
             "_n = subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout.count('GPU ')\n"
             "print(f'GPUs visible: {_n}', flush=True)\n"
             "assert _n >= 2, 'Need Accelerator = GPU T4 x2 (tensor-split 1,1)'\n"),
        md("## 1. Install pinned llama.cpp (CUDA 12.8) and agentdojo\n"
           "b11344 (built 2 Oct 2026) post-dates llama.cpp's Gemma-4 support and the "
           "Gemma-4 tool-grammar fix (PR #29115, 19 Sep 2026)."),
        install_llama,
        pip_cell,
        md("## 2. Download the Gemma GGUF to scratch (D13)\n"
           f"* `{GEMMA_REPO}` → `{GEMMA_FILE}` (14,439,363,584 B ≈ 13.45 GiB), pinned revision "
           f"`{GEMMA_REV[:12]}`, SHA-256 `{GEMMA_SHA256[:16]}…` checked below.\n"
           "* Weights go to `/kaggle/tmp/models` if it exists, else `/tmp/models` — **never** "
           "`/kaggle/working/models` (~20 GB cap; ENOSPC in the first pilot). Any stale GGUF "
           "under `/kaggle/working/models` is removed first.\n"
           "* Do **not** download the `*-mmproj.gguf` (vision projector; not needed for text agents)."),
        code("# Free /kaggle/working of any stale GGUFs / partial downloads (D13).\n"
             "import os, shutil, subprocess\n"
             "WORKING_MODELS = '/kaggle/working/models'\n"
             "removed = []\n"
             "if os.path.isdir(WORKING_MODELS):\n"
             "    for root, dirs, files in os.walk(WORKING_MODELS, topdown=False):\n"
             "        for name in files:\n"
             "            if name.endswith(('.gguf', '.incomplete', '.lock')):\n"
             "                fp = os.path.join(root, name)\n"
             "                try: os.remove(fp); removed.append(fp)\n"
             "                except FileNotFoundError: pass\n"
             "print('removed from /kaggle/working/models:', removed or '(nothing)', flush=True)\n"
             "print(subprocess.run(['df','-h','/kaggle/working','/tmp'],\n"
             "                     capture_output=True, text=True).stdout, flush=True)\n"),
        code("import os, hashlib, shutil, subprocess, time\n"
             "from huggingface_hub import hf_hub_download\n"
             "\n"
             "for candidate in ('/kaggle/tmp/models', '/tmp/models'):\n"
             "    parent = os.path.dirname(candidate)\n"
             "    if os.path.isdir(parent) or parent == '/tmp':\n"
             "        MODELS_DIR = candidate\n"
             "        break\n"
             "else:\n"
             "    MODELS_DIR = '/tmp/models'\n"
             "os.makedirs(MODELS_DIR, exist_ok=True)\n"
             "os.environ['HF_HUB_CACHE'] = MODELS_DIR\n"
             "os.environ['HUGGINGFACE_HUB_CACHE'] = MODELS_DIR\n"
             "free = shutil.disk_usage(MODELS_DIR).free / (1024**3)\n"
             "print(f'MODELS_DIR={MODELS_DIR}  free={free:.1f} GiB', flush=True)\n"
             "assert free > 15.0, 'scratch too small for the 13.45 GiB Gemma GGUF'\n"
             "\n"
             f"GEMMA_REPO = '{GEMMA_REPO}'\n"
             f"GEMMA_REV = '{GEMMA_REV}'\n"
             f"GEMMA_FILE = '{GEMMA_FILE}'\n"
             f"GEMMA_SIZE = {GEMMA_SIZE}\n"
             f"GEMMA_SHA256 = '{GEMMA_SHA256}'\n"
             "t0 = time.time()\n"
             "print(f'downloading {GEMMA_REPO}/{GEMMA_FILE} ...', flush=True)\n"
             "GEMMA = hf_hub_download(GEMMA_REPO, GEMMA_FILE, revision=GEMMA_REV, local_dir=MODELS_DIR)\n"
             "sz = os.path.getsize(GEMMA)\n"
             "print(f'Gemma -> {GEMMA}  {sz/(1024**3):.2f} GiB  ({time.time()-t0:.0f}s)', flush=True)\n"
             "assert sz == GEMMA_SIZE, f'size mismatch {sz} != {GEMMA_SIZE}'\n"
             "\n"
             "def sha256(p, n=1<<22):\n"
             "    h = hashlib.sha256()\n"
             "    with open(p, 'rb') as f:\n"
             "        while (b := f.read(n)): h.update(b)\n"
             "    return h.hexdigest()\n"
             "GEMMA_SHA = sha256(GEMMA)\n"
             "print(f'sha256 {GEMMA_SHA}  match={GEMMA_SHA == GEMMA_SHA256}', flush=True)\n"
             "assert GEMMA_SHA == GEMMA_SHA256, 'GGUF sha256 mismatch'\n"),
        md("## 3. Serve Gemma (2×T4, tensor-split 1,1) and run no-attack utility\n"
           "`serve()` tries the full plan flags first and logs which variant came up: "
           "(a) `--swa-full` + thinking-off kwargs, (b) without `--swa-full` (if the "
           "full SWA KV cache does not fit), (c) neither. Any fallback is printed as "
           "`SERVE_VARIANT=` and must be logged in DECISIONS.md. A background heartbeat "
           "prints every 5 min (elapsed, JSON traces written so far) so a long suite never "
           "looks hung (D16)."),
        code("import subprocess, time, os, requests, json\n"
             "SERVE_VARIANTS = [\n"
             "    ('swa_full+no_think', ['--swa-full', '--chat-template-kwargs', '{\"enable_thinking\":false}']),\n"
             "    ('no_think',          ['--chat-template-kwargs', '{\"enable_thinking\":false}']),\n"
             "    ('plain',             []),\n"
             "]\n"
             "SERVE_USED = None\n"
             "def serve(model_path, ngl=99, ctx=24576, nslots=4, split='1,1', port=8099, timeout_s=600):\n"
             "    global SERVE_USED\n"
             "    for name, extra in SERVE_VARIANTS:\n"
             "        cmd = [SERVER,'-m',model_path,'--host','127.0.0.1','--port',str(port),\n"
             "               '-ngl',str(ngl),'-c',str(ctx),'-np',str(nslots),'--jinja','--temp','0']\n"
             "        if split: cmd += ['--tensor-split', split]\n"
             "        cmd += extra\n"
             "        logf = f'/kaggle/working/server_{port}_{name}.log'\n"
             "        print(f'serve[{name}]: ' + ' '.join(cmd[1:]), flush=True)\n"
             "        p = subprocess.Popen(cmd, stdout=open(logf,'w'), stderr=subprocess.STDOUT)\n"
             "        t0 = time.time()\n"
             "        while time.time() - t0 < timeout_s:\n"
             "            if p.poll() is not None:\n"
             "                break\n"
             "            try:\n"
             "                if requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5).ok:\n"
             "                    SERVE_USED = {'variant': name, 'cmd': cmd[1:], 'log': logf}\n"
             "                    print(f'server up  SERVE_VARIANT={name}  ({time.time()-t0:.0f}s)', flush=True)\n"
             "                    return p\n"
             "            except Exception: pass\n"
             "            time.sleep(3)\n"
             "        p.kill(); p.wait()\n"
             "        tail = open(logf).read()[-1500:]\n"
             "        print(f'serve[{name}] FAILED (rc={p.returncode}); log tail:\\n{tail}', flush=True)\n"
             "    raise RuntimeError('llama-server did not start with any variant; see server_*.log')\n"),
        imports_cell,
        code("import threading\n"
             "from pathlib import Path\n"
             "\n"
             "def _heartbeat(label, t0, stop, every=300):\n"
             "    d = Path('/kaggle/working/runs') / label\n"
             "    while not stop.wait(every):\n"
             "        n = len(list(d.rglob('*.json'))) if d.is_dir() else 0\n"
             "        print(f'[heartbeat] {label}: {time.time()-t0:.0f}s elapsed, {n} json traces so far', flush=True)\n"
             "\n"
             "def run_utility(model_path, label, split='1,1'):\n"
             "    from agentdojo.logging import OutputLogger\n"
             "    print(f'{label}: loading weights / starting llama-server...', flush=True)\n"
             "    proc = serve(model_path, split=split)\n"
             "    stop = threading.Event()\n"
             "    try:\n"
             "        client = openai.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8099/v1')\n"
             "        mid = client.models.list().data[0].id\n"
             "        print(f'{label}: model id={mid}', flush=True)\n"
             "        llm = OpenAILLM(client, mid)\n"
             "        pipe = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=mid,\n"
             "               defense=None, system_message_name=None, system_message=None))\n"
             "        pipe.name = label\n"
             "        suites = get_suites('v1.2.2')\n"
             "        results, ns, secs = {}, {}, {}\n"
             "        t0 = time.time(); n_eps = 0\n"
             "        threading.Thread(target=_heartbeat, args=(label, t0, stop), daemon=True).start()\n"
             "        # Official layout: runs/{pipeline_name}/suite/... (D19). Do not nest label in logdir.\n"
             "        logdir = Path('/kaggle/working/runs')\n"
             "        logdir.mkdir(parents=True, exist_ok=True)\n"
             "        with OutputLogger(str(logdir)):\n"
             "            for s in ['banking','slack']:\n"
             "                print(f'{label}: starting {s} utility (no attack)...', flush=True)\n"
             "                t_suite = time.time()\n"
             "                r = benchmark_suite_without_injections(pipe, suites[s],\n"
             "                        logdir=logdir, force_rerun=False,\n"
             "                        benchmark_version='v1.2.2')\n"
             "                u = list(r['utility_results'].values())\n"
             "                n_eps += len(u)\n"
             "                results[s] = sum(u)/len(u) if u else float('nan')\n"
             "                ns[s] = len(u); secs[s] = time.time()-t_suite\n"
             "                print(f'{label}/{s}: utility={results[s]:.3f}, n={len(u)}, {secs[s]:.0f}s', flush=True)\n"
             "        dt = time.time()-t0\n"
             "        eph = n_eps/(dt/3600) if dt > 0 else 0.0\n"
             "        print(f'{label}: utility {results}, episodes {n_eps}, {dt:.0f}s, {eph:.1f} episodes/h', flush=True)\n"
             "        model_dir = logdir / label\n"
             "        n_json = sum(1 for _ in model_dir.rglob('*.json')) if model_dir.is_dir() else 0\n"
             "        print(f'{label}: traces under {model_dir}: {n_json} json files', flush=True)\n"
             "        return {'model_id': mid, 'utility': results, 'n': ns, 'suite_seconds': secs,\n"
             "                'episodes': n_eps, 'wall_s': dt, 'eps_per_h': eph}\n"
             "    finally:\n"
             "        stop.set()\n"
             "        proc.terminate()\n"
             "        print(f'{label}: server stopped', flush=True)\n"),
        code(f"# Gemma-4-26B-A4B QAT Q4_0, split across 2 GPUs\n"
             f"GEMMA_RES = run_utility(GEMMA, '{GEMMA_LABEL}', split='1,1')\n"),
        parse_md,
        nbf.v4.new_code_cell(parse_src),
        inv_md,
        inv_cell,
        md("## 6. Result + keep/drop marker (D20)\n"
           "Writes `/kaggle/working/gemma_pilot_result.json` and prints one "
           "`GEMMA_PILOT_RESULT` line to copy into DECISIONS.md."),
        code("import json, math\n"
             f"KEEP_EPH = {GEMMA_KEEP_EPH}\n"
             "u = GEMMA_RES['utility']\n"
             "util_ok = all((v >= 0.20) for v in u.values())\n"
             "parse_ok = (not math.isnan(PARSE_RATE)) and PARSE_RATE <= 0.05\n"
             "eph_ok = GEMMA_RES['eps_per_h'] >= KEEP_EPH\n"
             "verdict = 'KEEP' if (eph_ok and util_ok and parse_ok) else 'REVIEW'\n"
             "out = dict(GEMMA_RES, parse_error_rate=PARSE_RATE, serve=SERVE_USED,\n"
             "           gguf={'repo': GEMMA_REPO, 'rev': GEMMA_REV, 'file': GEMMA_FILE, 'sha256': GEMMA_SHA},\n"
             "           llama_build=LLAMA_BUILD, keep_threshold_eps_per_h=KEEP_EPH,\n"
             "           eph_ok=eph_ok, util_ok=util_ok, parse_ok=parse_ok, verdict=verdict)\n"
             "open('/kaggle/working/gemma_pilot_result.json','w').write(json.dumps(out, indent=2))\n"
             "print(json.dumps(out, indent=2))\n"
             "print(f\"GEMMA_PILOT_RESULT eps_per_h={GEMMA_RES['eps_per_h']:.1f} \"\n"
             "      f\"banking={u.get('banking', float('nan')):.3f} slack={u.get('slack', float('nan')):.3f} \"\n"
             "      f\"parse={PARSE_RATE:.1%} variant={SERVE_USED['variant']} verdict={verdict}\", flush=True)\n"),
    ]
    return nb


def main_grid_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Main grid (parameterised by MODEL and CONDITION)\n"
           "Research plan sections 11.6-11.10. Serves one open model with llama.cpp and runs "
           "the 300-pair subset for one condition, with JSONL logging and **resume** after a "
           "12-h Kaggle session ends (AgentDojo-style skip-if-done).\n\n"
           "Set `MODEL_KEY` and `CONDITION` in the parameters cell (Kaggle 'Save & Run All' "
           "runs it in the background). One model per contiguous block; download the JSONL "
           "after each block. Keep the notebook private."),
        code("# --- PARAMETERS (edit these) ---\n"
             "MODEL_KEY = 'granite-4.2-8b'   # one of MODELS below\n"
             "CONDITION = 'en'               # en|bn_mt|bn_native|banglish_native|banglish_translit|codemix_native\n"
             "CODE_DIR  = '/kaggle/working/code'   # where this repo's code/ lives\n"
             "PAIRS     = f'{CODE_DIR}/subset/pairs.json'\n"
             "OUTDIR    = '/kaggle/working/runs'\n"),
        code("# Verified GGUFs (HF, 2026-10-02). See DECISIONS.md D8 for substitution rules.\n"
             "MODELS = {\n"
             "  'granite-4.2-8b': ('ibm-granite/granite-4.2-8b-GGUF','granite-4.2-8b-Q4_K_M.gguf', None),\n"
             "  'qwen3.6-35b-a3b': ('ggml-org/Qwen3.6-35B-A3B-GGUF','Qwen3.6-35B-A3B-Q4_K_M.gguf','1,1'),\n"
             "  # gpt-oss-20b, Ministral-3-14B, Gemma-4 QAT: add their verified GGUF repos here.\n"
             "}\n"),
        md("## Install (same pinned llama.cpp + agentdojo as the pilot)"),
        code(f"LLAMA_BUILD='{LLAMA_BUILD}'\n"
             f"CUDA_TARBALL='{LLAMA_CUDA}'\nCUDART_TARBALL='{CUDART}'\n"
             "import subprocess, glob, os\n"
             "os.makedirs('/kaggle/working/llama', exist_ok=True)\n"
             "for url in (CUDART_TARBALL, CUDA_TARBALL):\n"
             "    fn=url.split('/')[-1]\n"
             "    subprocess.run(['wget','-q','-O',f'/kaggle/working/llama/{fn}',url],check=True)\n"
             "    subprocess.run(['tar','xzf',f'/kaggle/working/llama/{fn}','-C','/kaggle/working/llama'],check=True)\n"
             "SERVER=glob.glob('/kaggle/working/llama/**/llama-server',recursive=True)[0]\n"
             "!pip -q install agentdojo==0.1.35 openai==3.23.0"),
        code("from huggingface_hub import hf_hub_download\n"
             "repo, fname, split = MODELS[MODEL_KEY]\n"
             "MODEL_PATH = hf_hub_download(repo, fname, local_dir='/kaggle/working/models')\n"
             "print('model', MODEL_PATH, 'split', split)"),
        md("## Serve and run the condition over the 300-pair subset (resumable)"),
        code("import subprocess, time, os, requests\n"
             "def serve(model_path, split=None, port=8099):\n"
             "    cmd=[SERVER,'-m',model_path,'--host','127.0.0.1','--port',str(port),\n"
             "         '-ngl','99','-c','24576','-np','4','--jinja','--temp','0']\n"
             "    if split: cmd+=['--tensor-split',split]\n"
             "    log=open(f'/kaggle/working/server_{port}.log','w')\n"
             "    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)\n"
             "    for _ in range(240):\n"
             "        try:\n"
             "            if requests.get(f'http://127.0.0.1:{port}/v1/models').ok: return p\n"
             "        except Exception: pass\n"
             "        time.sleep(2)\n"
             "    raise RuntimeError('server did not start')\n"
             "proc = serve(MODEL_PATH, split=split)\n"
             "os.environ['LOCAL_LLM_PORT']='8099'"),
        code("import sys; sys.path.insert(0, CODE_DIR)\n"
             "out = f'{OUTDIR}/{MODEL_KEY}_{CONDITION}.jsonl'\n"
             "# run_benchmark resumes from an existing JSONL automatically\n"
             "cmd = [sys.executable, f'{CODE_DIR}/runners/run_benchmark.py',\n"
             "       '--provider','local','--condition',CONDITION,'--pairs',PAIRS,'--out',out]\n"
             "import subprocess\n"
             "print('running', ' '.join(cmd))\n"
             "subprocess.run(cmd, check=True)\n"
             "proc.terminate()"),
        md("## Download the JSONL, then start the next (MODEL, CONDITION) block.\n"
           "Analyse locally with `analysis/stats.py`. Pooled n per contrast = 949 (Gemini) "
           "+ 5 x 300 (open models). Keep each model's block contiguous; log the served model "
           "version and GGUF SHA-256 in DECISIONS.md."),
    ]
    return nb


def main():
    OUT.mkdir(exist_ok=True)
    nbf.write(pilot_nb(), OUT / "pilot_throughput.ipynb")
    nbf.write(main_grid_nb(), OUT / "main_grid.ipynb")
    nbf.write(gemma_pilot_nb(), OUT / "pilot_gemma_throughput.ipynb")
    print("wrote", OUT / "pilot_gemma_throughput.ipynb")
    print("wrote", OUT / "pilot_throughput.ipynb", "and", OUT / "main_grid.ipynb")


if __name__ == "__main__":
    main()


def nllb_nb():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# bn_mt via NLLB-200-3.3B (Kaggle GPU)\n"
            "Research plan 4.3. The box produced bn_mt with the distilled 600M model for "
            "feasibility; this notebook regenerates it with the plan's engine "
            "`facebook/nllb-200-3.3B` on a Kaggle GPU, reusing the same literal-protection "
            "pipeline in `scripts/make_mt.py`. NLLB is CC-BY-NC, so the generated subset "
            "ships separately under CC BY-NC 4.0."),
        nbf.v4.new_code_cell(
            "!pip -q install 'transformers<5' sentencepiece sacremoses\n"
            "import sys; sys.path.insert(0,'/kaggle/working/code')  # this repo's code/\n"
            "import subprocess\n"
            "subprocess.run([sys.executable,'/kaggle/working/code/scripts/build_payloads.py'],check=True)\n"
            "subprocess.run([sys.executable,'/kaggle/working/code/scripts/make_mt.py',\n"
            "  '--model','facebook/nllb-200-3.3B'],check=True)\n"
            "print('bn_mt regenerated with NLLB-200-3.3B; see data/mt_report.json')"),
    ]
    return nb


if __name__ == "__main__":  # re-emit including NLLB nb
    import nbformat as nbf
    from pathlib import Path as _P
    _o = _P(__file__).resolve().parent.parent / "kaggle"
    nbf.write(nllb_nb(), _o / "make_bn_mt_nllb.ipynb")
    print("wrote", _o / "make_bn_mt_nllb.ipynb")
