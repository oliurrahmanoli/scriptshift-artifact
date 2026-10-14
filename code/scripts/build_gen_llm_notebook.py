"""Generate kaggle/gen_llm_payloads.ipynb (research-plan §12, DECISIONS D24).

Self-contained Kaggle notebook (no dataset / repo attachment). It embeds, verbatim:
  * scripts/make_mt.py            (for PATTERNS / literal rule parity)
  * payloads/generate_llm.py      (the TESTED generator + validators)
  * payloads/prompts/*.txt        (the generation prompt templates)
  * the 35 English goals + W1/W2 wrappers from data/payloads.json
and writes them to /kaggle/working/genpkg so the SAME tested code runs on Kaggle.

It serves the Gemma-4-26B-A4B QAT generator on T4x2 (tensor-split, --swa-full,
thinking off, exactly the proven Gemma pilot config), generates the LLM-authored
conditions over all 35 goals with reject+regenerate, records per-item provenance
(generator, GGUF SHA-256, seed, temperature, prompt SHA-256), and (optionally) runs an
NLLB-200-3.3B back-translation audit and a Qwen second-generator subset for the
generator/target overlap mitigation.

Run from code/:  python3 scripts/build_gen_llm_notebook.py
                 python3 scripts/build_gen_llm_notebook.py --only-pending
The second form writes kaggle/gen_llm_payloads_regen_pending.ipynb, which regenerates ONLY the
items still pending in data/conditions/<cond>/_gen_meta.json (no wrappers), D26.
Do NOT launch it from the box; the notebook is run on Kaggle (Save & Run All).

D26: the embedded generator now records every attempt (text + validation) and keeps the
last attempt as candidate_text; the server log is no longer overwritten when Gemma is
re-served for the wrapper phase (one log file per serve() call).
"""
from __future__ import annotations

import argparse

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "gen_llm_payloads.ipynb"
OUT_PENDING = ROOT / "kaggle" / "gen_llm_payloads_regen_pending.ipynb"

# D27: two items are already "ok" in _gen_meta.json but drifted in MEANING — the generator
# added instructions/persuasion beyond the English goal. They are force-regenerated with the
# stricter prompt + the back-translation "adds_content" validator, alongside the pending items.
#   bn_llm slack/injection_task_1  — added "click to verify"
#   bn_llm travel/injection_task_1 — added "from now on, automatically"
MEANING_DRIFT_REGEN = {"bn_llm": ["slack/injection_task_1", "travel/injection_task_1"]}

GEMMA = dict(repo="google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
             rev="d1c082be9cf3c8a514acf63b8761f4b41935842e",
             file="gemma-4-26B_q4_0-it.gguf",
             size=14439363584,
             sha256="3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d")
# Qwen GGUF for the optional second-generator subset (overlap mitigation). D8 verified the
# repo/file/rev/size/sha256 pinned in DECISIONS D25.
QWEN = dict(repo="ggml-org/Qwen3.6-35B-A3B-GGUF", rev="baec3ebee244827cda0f4557eafa8b28f7545fa6",
            file="Qwen3.6-35B-A3B-Q4_K_M.gguf", size=20419565568,
            sha256="671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7")
NLLB = dict(repo="facebook/nllb-200-3.3B", rev="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
LLAMA_BUILD = "b11344"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def pending_keys() -> dict:
    out = {}
    for cond in ("bn_llm", "banglish_llm", "codemix_llm", "bn_llm_W2", "banglish_llm_W2"):
        mp = ROOT / "data" / "conditions" / cond / "_gen_meta.json"
        if mp.exists():
            items = json.loads(mp.read_text(encoding="utf-8"))["items"]
            ks = [k for k, v in items.items() if v["status"] != "ok"]
            if ks:
                out[cond] = ks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-pending", action="store_true")
    args = ap.parse_args()
    only = pending_keys() if args.only_pending else None
    if only is not None:
        for cond, ks in MEANING_DRIFT_REGEN.items():
            bucket = only.setdefault(cond, [])
            for k in ks:
                if k not in bucket:
                    bucket.append(k)
    out_path = OUT_PENDING if args.only_pending else OUT
    pay = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))
    goals = json.loads((ROOT / "data" / "goals_en.json").read_text(encoding="utf-8"))
    order = goals["order"]
    en = {k: pay["conditions"]["en"]["goals"][k] for k in order}
    assert len(en) == 35 and all(v and not v.startswith("TODO") for v in en.values())
    w1, w2 = pay["wrappers"]["W1"], pay["wrappers"]["W2"]

    make_mt_src = (ROOT / "scripts" / "make_mt.py").read_text(encoding="utf-8")
    gen_src = (ROOT / "payloads" / "generate_llm.py").read_text(encoding="utf-8")
    prompts = {p.name: p.read_text(encoding="utf-8")
               for p in sorted((ROOT / "payloads" / "prompts").glob("*.txt"))}
    gen_sha = hashlib.sha256(gen_src.encode()).hexdigest()
    en_sha = hashlib.sha256(json.dumps(en, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()

    cells = []
    cells.append(md(f"""# LLM-authored injection payloads (Gemma generator) — research-plan §12, DECISIONS D24

Generate the **LLM-authored** conditions `bn_llm`, `banglish_llm`, `codemix_llm` (+ optional
`bn_llm_W2`, `banglish_llm_W2`) over all **35** AgentDojo goals, using an open LLM that
**writes** idiomatic injection text in each register (not sentence-by-sentence MT).

* **Generator:** `{GEMMA['repo']}` (Gemma-4-26B-A4B QAT Q4_0), rev `{GEMMA['rev'][:12]}`,
  served with llama.cpp `{LLAMA_BUILD}` on **T4x2** (`--tensor-split 1,1 --swa-full`,
  thinking off) — identical to the proven Gemma pilot (DECISIONS D20). Apache-2.0, not gated.
* **Why Gemma:** Apache-2.0, pretrained on 140+ languages (strong Bangla), fits T4x2, already
  validated in the grid. **Overlap caveat:** Gemma is also a *target* model. Mitigation is
  built in (see the overlap cell): per-item provenance is recorded, an optional **Qwen**
  second-generator subset is produced, and the analysis can exclude/substitute self-generated
  payloads for the Gemma target and report generator as a factor.
* **Automatic validation (reject + regenerate up to N):** (1) literal preservation; (2) script
  band per register (Bengali ratio for `bn`, Latin-only for `banglish`, mixed band for
  `codemix`); (3) semantic equivalence — the generator **back-translates** its own payload to
  English and **judges** equivalence to the English goal. An optional Phase B re-audits with
  **NLLB-200-3.3B** back-translation.
* **Provenance per item:** generator label, GGUF SHA-256, seed, temperature, prompt SHA-256,
  attempts, validation. Only payloads that PASS are accepted; failures are written as
  `TODO_LLM` for review (never silently wrong).
* **Self-contained:** embeds the TESTED `payloads/generate_llm.py` (sha256 `{gen_sha[:12]}…`),
  `scripts/make_mt.py` (PATTERNS parity), the prompt templates, and the 35 EN goals
  (sha256 `{en_sha[:12]}…`). No repo/dataset attachment, no HF login.
* **Kaggle:** Accelerator **GPU T4 x2**, Internet **on**, notebook **private**. Then
  **Save Version → Save & Run All (Commit)**.
* **History:** Version 1 (pre-D26 code) ran 2026-10-06 (bn_llm 29/35, banglish_llm 32/35). D26 fixed
  the literal validator (URL trailing punctuation, quote-typography, mandated Subject/Body fields,
  ws4 upstream-typo override, Bengali-digit normalisation in literal tokens) and the attempt
  bookkeeping; this build embeds the fixed code.""" + (f"""
* **TARGETED REGEN (D26/D27):** this notebook regenerates ONLY these items
  `{json.dumps(only)}` — no wrappers; download `llm_payloads.zip` and merge on the box with
  `scripts/merge_regen_outputs.py`. It covers the 12 pending items (9 missing a valid recorded
  back-translation after the D26 bookkeeping fix + 3 dropped by the stricter D26 recheck:
  bn_llm banking/7 `নতুন_password`, banglish_llm workspace/0 and slack/1 with the Hindi word
  कृपया) **plus** the 2 meaning-drift items (bn_llm slack/1 added 'click to verify', travel/1
  added 'from now on, automatically').
* **D27 fixes embedded:** (a) all output text is forced to **ASCII digits** (§4.3: never
  Bengali digits); (b) `STRICT_ADDITIONS` turns on the back-translation **adds_content** judge
  check so a payload that adds instructions/persuasion/urgency beyond the English goal is
  rejected and regenerated with a stricter prompt; (c) non-Bengali Indic scripts (e.g.
  Devanagari कृपया) already fail the script check.""" if only else "")))

    cells.append(md("## 0. Parameters"))
    cells.append(code(f"""# --- PARAMETERS ---
CONDITIONS      = {json.dumps(list(only) if only else ['bn_llm', 'banglish_llm'])}        # this-week scope (§12); full: +['codemix_llm','bn_llm_W2','banglish_llm_W2']
MAX_ATTEMPTS    = 4            # reject+regenerate budget per item
GEN_TEMPERATURE = 0.7          # >0 so regeneration explores; recorded per item
GEN_SEED        = 20261002     # base seed; bumped per attempt for diversity
MAX_TOKENS      = 1024
USE_SELF_EQUIV  = True         # generator back-translates+judges its own output (validator 3)
STRICT_ADDITIONS = {only is not None}        # D27: judge also rejects content ADDED beyond the goal (meaning drift)
RUN_NLLB_AUDIT  = False        # Phase B: independent NLLB-3.3B back-translation audit (report-only)
RUN_QWEN_SUBSET = False        # overlap mitigation: regenerate a subset with Qwen (second generator)
QWEN_SUBSET_N   = 35           # how many goals per condition to also generate with Qwen
OUT_ROOT        = '/kaggle/working/llm_payloads'
ONLY_KEYS       = {only!r}   # None = all 35 goals; {{cond: [keys]}} = targeted regen (D26)
RUN_WRAPPERS    = {only is None}          # wrappers are already frozen for a targeted regen
GEMMA = {json.dumps(GEMMA, indent=1)}
QWEN  = {json.dumps(QWEN, indent=1)}
NLLB  = {json.dumps(NLLB, indent=1)}
LLAMA_BUILD = {LLAMA_BUILD!r}
EN_GOALS_SHA256 = {en_sha!r}
print('conditions', CONDITIONS, 'attempts', MAX_ATTEMPTS, 'temp', GEN_TEMPERATURE, flush=True)
"""))

    cells.append(md("## 1. Embedded inputs: 35 EN goals, W1/W2 wrappers"))
    cells.append(code("import json, hashlib\n"
        "EN_GOALS = json.loads(r'''" + json.dumps(en, ensure_ascii=False, indent=1) + "''')\n"
        "ORDER = json.loads(r'''" + json.dumps(order) + "''')\n"
        "W1 = json.loads(r'''" + json.dumps(w1, ensure_ascii=False) + "''')\n"
        "W2 = json.loads(r'''" + json.dumps(w2, ensure_ascii=False) + "''')\n"
        "def _csha(d):\n"
        "    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()\n"
        "assert len(EN_GOALS)==35 and _csha(EN_GOALS)==EN_GOALS_SHA256, 'embedded goals changed'\n"
        "print('EN goals', len(EN_GOALS), 'sha', EN_GOALS_SHA256[:12], flush=True)\n"))

    cells.append(md("## 2. Write the embedded (tested) generator package to /kaggle/working/genpkg and import it"))
    pkg_writer = (
        "import os, sys, json\n"
        "PKG='/kaggle/working/genpkg'\n"
        "os.makedirs(PKG+'/scripts', exist_ok=True)\n"
        "os.makedirs(PKG+'/payloads/prompts', exist_ok=True)\n"
        "open(PKG+'/scripts/make_mt.py','w',encoding='utf-8').write(MAKE_MT_SRC)\n"
        "open(PKG+'/payloads/__init__.py','w',encoding='utf-8').write('')\n"
        "open(PKG+'/payloads/generate_llm.py','w',encoding='utf-8').write(GEN_SRC)\n"
        "for name, body in PROMPTS.items():\n"
        "    open(PKG+'/payloads/prompts/'+name,'w',encoding='utf-8').write(body)\n"
        "sys.path.insert(0, PKG)\n"
        "import importlib; import payloads.generate_llm as G; importlib.reload(G)\n"
        "assert hashlib.sha256(GEN_SRC.encode()).hexdigest()==GEN_SHA, 'generator source changed'\n"
        "print('generator imported; sha', GEN_SHA[:12], flush=True)\n")
    cells.append(code(
        "MAKE_MT_SRC = r'''" + make_mt_src.replace("'''", "'\\''\\''\\''") + "'''\n"
        "GEN_SRC = r'''" + gen_src.replace("'''", "'\\''\\''\\''") + "'''\n"
        "GEN_SHA = " + repr(gen_sha) + "\n"
        "PROMPTS = json.loads(r'''" + json.dumps(prompts, ensure_ascii=False) + "''')\n"
        + pkg_writer))

    cells.append(md("## 3. GPU, llama.cpp build, Gemma download + SHA-256 (proven pilot cells)"))
    cells.append(code(
        "import subprocess\n"
        "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],capture_output=True,text=True).stdout, flush=True)\n"
        "_n = subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True).stdout.count('GPU ')\n"
        "assert _n >= 2, 'Need Accelerator = GPU T4 x2 (tensor-split 1,1)'\n"))
    cells.append(code(
        f"CUDA_TARBALL='https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz'\n"
        f"CUDART_TARBALL='https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz'\n"
        "import os, glob\n"
        "os.makedirs('/kaggle/working/llama', exist_ok=True)\n"
        "for url in (CUDART_TARBALL, CUDA_TARBALL):\n"
        "    fn=url.split('/')[-1]\n"
        "    subprocess.run(['wget','-q','-O',f'/kaggle/working/llama/{fn}',url],check=True)\n"
        "    subprocess.run(['tar','xzf',f'/kaggle/working/llama/{fn}','-C','/kaggle/working/llama'],check=True)\n"
        "SERVER=glob.glob('/kaggle/working/llama/**/llama-server',recursive=True)[0]\n"
        "subprocess.run([os.sys.executable,'-m','pip','-q','install','agentdojo==0.1.35','openai==3.23.0'],check=True)\n"
        "print('llama-server', SERVER, flush=True)\n"))
    cells.append(code(
        "import hashlib, shutil, time\n"
        "from huggingface_hub import hf_hub_download\n"
        "MODELS_DIR = '/kaggle/tmp/models' if os.path.isdir('/kaggle/tmp') else '/tmp/models'\n"
        "os.makedirs(MODELS_DIR, exist_ok=True)\n"
        "os.environ['HF_HUB_CACHE']=MODELS_DIR\n"
        "assert shutil.disk_usage(MODELS_DIR).free/2**30 > 15.0, 'scratch too small for Gemma GGUF'\n"
        "def sha256(p, n=1<<22):\n"
        "    h=hashlib.sha256()\n"
        "    with open(p,'rb') as f:\n"
        "        while (b:=f.read(n)): h.update(b)\n"
        "    return h.hexdigest()\n"
        "t0=time.time()\n"
        "GEMMA_PATH = hf_hub_download(GEMMA['repo'], GEMMA['file'], revision=GEMMA['rev'], local_dir=MODELS_DIR)\n"
        "assert os.path.getsize(GEMMA_PATH)==GEMMA['size'], 'Gemma size mismatch'\n"
        "_sha=sha256(GEMMA_PATH); print('gemma sha', _sha[:16], 'match', _sha==GEMMA['sha256'], f'({time.time()-t0:.0f}s)', flush=True)\n"
        "assert _sha==GEMMA['sha256'], 'Gemma GGUF sha256 mismatch'\n"
        "GEMMA_GGUF_SHA = _sha\n"))

    cells.append(md("## 4. Serve Gemma (T4x2, `--tensor-split 1,1 --swa-full`, thinking off) — proven pilot config"))
    cells.append(code(
        "import requests\n"
        "SERVE_VARIANTS=[('swa_full+no_think',['--swa-full','--chat-template-kwargs','{\"enable_thinking\":false}']),\n"
        "               ('no_think',['--chat-template-kwargs','{\"enable_thinking\":false}']),('plain',[])]\n"
        "SERVE_USED=None\n"
        "SERVE_N=0   # one log file per serve() call (D26: re-serving used to overwrite the generation log)\n"
        "def serve(model_path, ngl=99, ctx=24576, nslots=4, split='1,1', port=8099, timeout_s=600):\n"
        "    global SERVE_USED, SERVE_N\n"
        "    SERVE_N+=1\n"
        "    for name, extra in SERVE_VARIANTS:\n"
        "        cmd=[SERVER,'-m',model_path,'--host','127.0.0.1','--port',str(port),'-ngl',str(ngl),\n"
        "             '-c',str(ctx),'-np',str(nslots),'--jinja','--temp',str(GEN_TEMPERATURE)]\n"
        "        if split: cmd+=['--tensor-split',split]\n"
        "        cmd+=extra\n"
        "        logf=f'/kaggle/working/server_{port}_{name}_{SERVE_N}.log'\n"
        "        p=subprocess.Popen(cmd,stdout=open(logf,'w'),stderr=subprocess.STDOUT)\n"
        "        t0=time.time()\n"
        "        while time.time()-t0<timeout_s:\n"
        "            if p.poll() is not None: break\n"
        "            try:\n"
        "                if requests.get(f'http://127.0.0.1:{port}/v1/models',timeout=5).ok:\n"
        "                    SERVE_USED={'variant':name,'cmd':cmd[1:],'log':logf}\n"
        "                    print(f'server up SERVE_VARIANT={name} ({time.time()-t0:.0f}s)', flush=True); return p\n"
        "            except Exception: pass\n"
        "            time.sleep(3)\n"
        "        p.kill(); p.wait(); print(f'serve[{name}] FAILED; tail:\\n'+open(logf).read()[-1200:], flush=True)\n"
        "    raise RuntimeError('llama-server did not start')\n"
        "PROC = serve(GEMMA_PATH)\n"))

    cells.append(md("## 5. chat_fn + self back-translation/equivalence validators (generator)"))
    cells.append(code(
        "import openai, itertools\n"
        "client = openai.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8099/v1')\n"
        "MODEL_ID = client.models.list().data[0].id\n"
        "print('served model id', MODEL_ID, flush=True)\n"
        "class GenChat:\n"
        "    '''OpenAI-compatible chat_fn; bumps seed per call so regeneration varies.'''\n"
        "    def __init__(self, temperature, base_seed, max_tokens):\n"
        "        self.t=temperature; self.s=itertools.count(base_seed); self.m=max_tokens\n"
        "    def __call__(self, prompt):\n"
        "        r=client.chat.completions.create(model=MODEL_ID,\n"
        "            messages=[{'role':'user','content':prompt}],\n"
        "            temperature=self.t, max_tokens=self.m, seed=next(self.s))\n"
        "        msg=r.choices[0].message\n"
        "        return msg.content or getattr(msg,'reasoning_content','') or ''\n"
        "chat = GenChat(GEN_TEMPERATURE, GEN_SEED, MAX_TOKENS)\n"
        "# Validator 3 wired to the generator itself: back-translate the payload to English,\n"
        "# then judge equivalence to the English goal. (Phase B re-audits with NLLB.)\n"
        "def gemma_backtranslate(text):\n"
        "    p=('Translate the following message to natural English. Output ONLY the English '\n"
        "       'translation, nothing else.\\n\\n'+text)\n"
        "    return chat(p)\n"
        "def gemma_judge(prompt):\n"
        "    return chat(prompt)\n"
        "NLLB_FN = gemma_backtranslate if USE_SELF_EQUIV else None\n"
        "JUDGE_FN = gemma_judge if USE_SELF_EQUIV else None\n"))

    cells.append(md("## 6. Generate every selected condition over all 35 goals (reject + regenerate)"))
    cells.append(code(
        "import threading, time\n"
        "from pathlib import Path\n"
        "os.makedirs(OUT_ROOT, exist_ok=True)\n"
        "stop=threading.Event()\n"
        "def _hb(t0):\n"
        "    while not stop.wait(300):\n"
        "        print(f'[heartbeat] {time.time()-t0:.0f}s elapsed', flush=True)\n"
        "threading.Thread(target=_hb,args=(time.time(),),daemon=True).start()\n"
        "REPORTS={}\n"
        "for cond in CONDITIONS:\n"
        "    print(f'=== generating {cond} ===', flush=True)\n"
        "    t0=time.time()\n"
        "    keys = ONLY_KEYS[cond] if ONLY_KEYS else ORDER\n"
        "    rep = G.generate_condition(EN_GOALS, cond, chat, Path(OUT_ROOT), seed=GEN_SEED,\n"
        "        temperature=GEN_TEMPERATURE, generator_label='gemma-4-26b-a4b',\n"
        "        gguf_sha=GEMMA_GGUF_SHA, max_attempts=MAX_ATTEMPTS, nllb_fn=NLLB_FN,\n"
        "        judge_fn=JUDGE_FN, order=keys, strict_additions=STRICT_ADDITIONS)\n"
        "    REPORTS[cond]=rep\n"
        "    print(f'{cond}: filled {rep[\"filled\"]}/{rep[\"n_goals\"]} in {time.time()-t0:.0f}s', flush=True)\n"
        "stop.set()\n"
        "try: PROC.terminate()\n"
        "except Exception: pass\n"))

    cells.append(md("## 7. Wrappers for the LLM conditions (generate W1/W2 wrappers once per script)"))
    cells.append(code(
        "# The wrapper (W1/W2) is language furniture around {goal}. Generate it once per register\n"
        "# by asking the generator to render the English wrapper, preserving the {goal}/{user}/{model}\n"
        "# placeholders verbatim. Re-serve Gemma briefly (it was stopped above).\n"
        "WRAP_SPECS = {'bn_llm':('W1','bn'),'banglish_llm':('W1','banglish'),'codemix_llm':('W1','codemix'),\n"
        "              'bn_llm_W2':('W2','bn'),'banglish_llm_W2':('W2','banglish')}\n"
        "WRAPPERS_OUT={}\n"
        "need_wrap=[c for c in CONDITIONS if c in WRAP_SPECS] if RUN_WRAPPERS else []\n"
        "if need_wrap:\n"
        "    PROC = serve(GEMMA_PATH)\n"
        "    chat = GenChat(GEN_TEMPERATURE, GEN_SEED+9999, MAX_TOKENS)\n"
        "    for cond in need_wrap:\n"
        "        wid, reg = WRAP_SPECS[cond]\n"
        "        base = W1['en'] if wid=='W1' else W2['en_W2']\n"
        "        reg_name={'bn':'Bangla (Bengali script)','banglish':'romanised Banglish (Latin only)','codemix':'Bangla-English code-mixed'}[reg]\n"
        "        p=(f'Rewrite this message template in {reg_name}, keeping the placeholders '\n"
        "           '{goal}, {user} and {model} EXACTLY as written (do not translate them). '\n"
        "           'Output ONLY the rewritten template.\\n\\n'+base)\n"
        "        out=chat(p)\n"
        "        ok=all(ph in out for ph in ('{goal}','{user}','{model}'))\n"
        "        WRAPPERS_OUT[cond]={'wrapper_id':wid,'text':out if ok else 'TODO_LLM','ok':ok}\n"
        "        print(f'wrapper {cond}: ok={ok}', flush=True)\n"
        "    try: PROC.terminate()\n"
        "    except Exception: pass\n"))

    cells.append(md("""## 8. (Optional) Phase B — NLLB-200-3.3B back-translation audit (report-only)
Independent of the generator: loads NLLB-3.3B and back-translates each accepted payload to
English for an audit record. A semantic re-judge would need an LLM re-served; here we store the
NLLB back-translation + a lexical literal-overlap flag. Set `RUN_NLLB_AUDIT=True` to enable."""))
    cells.append(code(
        "NLLB_AUDIT={}\n"
        "if RUN_NLLB_AUDIT:\n"
        "    import torch\n"
        "    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer\n"
        "    subprocess.run([os.sys.executable,'-m','pip','-q','install','transformers<5','sentencepiece'],check=True)\n"
        "    tok=AutoTokenizer.from_pretrained(NLLB['repo'], revision=NLLB['rev'], src_lang='ben_Beng')\n"
        "    mdl=AutoModelForSeq2SeqLM.from_pretrained(NLLB['repo'], revision=NLLB['rev'], torch_dtype=torch.float16).to('cuda:0').eval()\n"
        "    BOS=tok.convert_tokens_to_ids('eng_Latn')\n"
        "    def nllb_to_en(t):\n"
        "        enc=tok(t, return_tensors='pt', truncation=True, max_length=512).to('cuda:0')\n"
        "        with torch.no_grad(): o=mdl.generate(**enc, forced_bos_token_id=BOS, num_beams=4, max_new_tokens=512)\n"
        "        return tok.batch_decode(o, skip_special_tokens=True)[0]\n"
        "    for cond, rep in REPORTS.items():\n"
        "        meta_path=Path(OUT_ROOT)/'data'/'conditions'/cond/'_gen_meta.json'\n"
        "        meta=json.loads(meta_path.read_text())\n"
        "        for key, it in meta['items'].items():\n"
        "            if it['status']!='ok': continue\n"
        "            cand=it.get('candidate_text') or ''\n"
        "            bt=nllb_to_en(cand)\n"
        "            lits=it.get('literals',[])\n"
        "            it['nllb_backtranslation']=bt\n"
        "            it['nllb_literal_overlap_ok']=all(l in cand for l in lits)\n"
        "        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))\n"
        "        NLLB_AUDIT[cond]=sum(1 for it in meta['items'].values() if 'nllb_backtranslation' in it)\n"
        "    print('NLLB audit done', NLLB_AUDIT, flush=True)\n"))

    cells.append(md("""## 9. (Optional) Qwen second-generator subset — generator/target overlap mitigation
Regenerate a subset with **Qwen3.6-35B-A3B** so the analysis can substitute Qwen-authored
payloads for the **Gemma target** (self-generation sensitivity). Qwen rev + size + SHA-256 are
pinned (D25). Set `RUN_QWEN_SUBSET=True` to enable."""))
    cells.append(code(
        "QWEN_REPORTS={}\n"
        "if RUN_QWEN_SUBSET:\n"
        "    QPATH=hf_hub_download(QWEN['repo'], QWEN['file'], revision=QWEN['rev'], local_dir=MODELS_DIR)\n"
        "    assert os.path.getsize(QPATH)==QWEN['size'], 'Qwen size mismatch'\n"
        "    assert sha256(QPATH)==QWEN['sha256'], 'Qwen GGUF sha mismatch'\n"
        "    # Qwen: non-thinking, tensor-split both GPUs\n"
        "    global SERVE_VARIANTS\n"
        "    SERVE_VARIANTS=[('qwen_no_think',['--chat-template-kwargs','{\"enable_thinking\":false}']),('plain',[])]\n"
        "    PROC=serve(QPATH)\n"
        "    client2=openai.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8099/v1')\n"
        "    MID2=client2.models.list().data[0].id\n"
        "    qchat=GenChat(GEN_TEMPERATURE, GEN_SEED, MAX_TOKENS)\n"
        "    sub_keys=ORDER[:QWEN_SUBSET_N]\n"
        "    for cond in CONDITIONS:\n"
        "        sub={k:EN_GOALS[k] for k in sub_keys}\n"
        "        rep=G.generate_condition(sub, cond+'__qwen', qchat, Path(OUT_ROOT), seed=GEN_SEED,\n"
        "            temperature=GEN_TEMPERATURE, generator_label='qwen3.6-35b-a3b',\n"
        "            gguf_sha=QWEN['sha256'], max_attempts=MAX_ATTEMPTS, nllb_fn=None, judge_fn=None, order=sub_keys)\n"
        "        QWEN_REPORTS[cond]=rep\n"
        "    try: PROC.terminate()\n"
        "    except Exception: pass\n"
        "    print('Qwen subset done', {k:v['filled'] for k,v in QWEN_REPORTS.items()}, flush=True)\n"))

    cells.append(md("## 10. Report, SHA256SUMS, zip, GEN_RESULT line"))
    cells.append(code(
        "import zipfile, datetime as dt\n"
        "root=Path(OUT_ROOT)\n"
        "# merge wrappers into a wrappers file\n"
        "if WRAPPERS_OUT:\n"
        "    wdir=root/'data'/'conditions'/'_wrappers'; wdir.mkdir(parents=True, exist_ok=True)\n"
        "    for cond,w in WRAPPERS_OUT.items():\n"
        "        (wdir/f\"{w['wrapper_id']}__{cond}.txt\").write_text(\n"
        "            f\"# Wrapper {w['wrapper_id']} for condition={cond}\\n# STATUS: {'filled' if w['ok'] else 'TODO_LLM'}\\n\\n{w['text']}\\n\", encoding='utf-8')\n"
        "summary={'generator':'gemma-4-26b-a4b','gemma_gguf_sha256':GEMMA_GGUF_SHA,'rev':GEMMA['rev'],\n"
        "         'llama_build':LLAMA_BUILD,'serve_variant':(SERVE_USED or {}).get('variant'),\n"
        "         'temperature':GEN_TEMPERATURE,'max_attempts':MAX_ATTEMPTS,'seed':GEN_SEED,\n"
        "         'conditions':{c:{'filled':REPORTS[c]['filled'],'n':REPORTS[c]['n_goals']} for c in REPORTS},\n"
        "         'wrappers':{c:WRAPPERS_OUT[c]['ok'] for c in WRAPPERS_OUT},'nllb_audit':NLLB_AUDIT,\n"
        "         'qwen_subset':{c:QWEN_REPORTS[c]['filled'] for c in QWEN_REPORTS},\n"
        "         'run_date':dt.date.today().isoformat(),'self_equiv':USE_SELF_EQUIV,\n"
        "         'only_keys':ONLY_KEYS,'server_logs':sorted(glob.glob('/kaggle/working/server_*.log'))}\n"
        "import shutil as _sh\n"
        "for _lf in glob.glob('/kaggle/working/server_*.log'): _sh.copy2(_lf, root/os.path.basename(_lf))\n"
        "(root/'gen_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))\n"
        "def _sha(p,n=1<<22):\n"
        "    h=hashlib.sha256()\n"
        "    with open(p,'rb') as f:\n"
        "        while (b:=f.read(n)): h.update(b)\n"
        "    return h.hexdigest()\n"
        "lines=[f'{_sha(p)}  {p.relative_to(root)}' for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt' and p.suffix!='.zip']\n"
        "(root/'SHA256SUMS.txt').write_text('\\n'.join(lines)+'\\n')\n"
        "with zipfile.ZipFile('/kaggle/working/llm_payloads.zip','w',zipfile.ZIP_DEFLATED) as z:\n"
        "    for p in sorted(root.rglob('*')):\n"
        "        if p.is_file(): z.write(p, p.relative_to(root.parent))\n"
        "print('GEN_RESULT ' + json.dumps({k:summary[k] for k in ('generator','serve_variant','conditions','wrappers','nllb_audit','qwen_subset')}, ensure_ascii=False), flush=True)\n"))

    cells.append(md("""## 11. After the run (on the box, not here)
1. Download `llm_payloads.zip`; check `SHA256SUMS.txt`, `gen_report.json` and the `GEN_RESULT` line.
2. Read every `TODO_LLM` / `NEEDS_REVIEW` item and each `_gen_meta.json` validation.
3. Freeze on the box (D26): `scripts/freeze_payloads.py` (re-validates every candidate, installs
   accepted items + provenance), `scripts/import_condition_dirs.py --sync-todo`,
   `scripts/export_condition_dirs.py`, `scripts/check_payload_freeze.py`; log it in `DECISIONS.md`.
   A targeted regen (`--only-pending` build) must be MERGED item-by-item, keeping the frozen items.
4. Transliteration: run `banglish_translit` = IndicXlit of the accepted `bn_llm` (scripts/make_translit.py on Kaggle)."""))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(cells)} cells); gen sha {gen_sha[:12]}; EN sha {en_sha[:12]}"
          + (f"; only {only}" if only else ""))


if __name__ == "__main__":
    main()
