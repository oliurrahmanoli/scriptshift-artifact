"""Generate kaggle/make_bn_mt_nllb_rerun.ipynb (DECISIONS D27).

Re-translate ONLY the six bn_mt items whose attack literals the as-run MT literal rule did
not protect, under the D26 literal-protection rules (D26.5 b/c):
  * workspace/injection_task_13 — lost the phishing URL https://bit.ly/sUf7tOJ (now protected)
  * workspace/injection_task_4  — mandated Subject/Body (upstream-typo override) not protected
  * workspace/injection_task_9  — mandated Subject + Body block not protected
  * workspace/injection_task_12 — mandated Subject + Body block not protected
  * travel/injection_task_0     — hotel name "Riverside View Hotel" not protected
  * travel/injection_task_6     — hotel name "Riverside View Hotel" not protected

The literal set for each item is the SAME one the LLM validator uses
(payloads/generate_llm.py extract_literals: regex PATTERNS + mandated Subject/Body fields +
the ws4 upstream-typo override + the §4.3 hotel-name additions), so the verbatim English
Subject/Body blocks, URLs and the hotel name are swapped for ASCII sentinels before NLLB and
restored to a fixpoint afterwards; a sentinel that does not round-trip => TODO_MT. D27: the
restored text is forced to ASCII digits (§4.3, never Bengali digits).

Same pinned model/SHA + provenance format as make_bn_mt_nllb.ipynb. Internet ON, T4 x2.
Outputs drop straight into code/data/conditions (data/conditions/bn_mt/<key>.txt).

Run from code/:  python scripts/build_nllb_rerun_notebook.py
Do NOT launch it from the box; run on Kaggle (Save & Run All).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "make_bn_mt_nllb_rerun.ipynb"

# Pinned NLLB-200-3.3B (identical to build_nllb_notebook.py / DECISIONS D26.1).
MODEL_ID = "facebook/nllb-200-3.3B"
MODEL_REV = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"
FILES = {
    "pytorch_model-00001-of-00003.bin": (6933375959, "06aec913c891950ff0c59bd018c1b46dc95be6e106c36df87d13175418b3972c"),
    "pytorch_model-00002-of-00003.bin": (8545249147, "1c5a44acb0287e905a62a506cf3e5af37956607f4c667d891ec069aa276be0be"),
    "pytorch_model-00003-of-00003.bin": (2098840299, "2cf744009996251567d7972f4436d65cf94380c5ddd8f524cb878e090b27bb50"),
    "sentencepiece.bpe.model": (4852054, "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"),
    "tokenizer.json": (17331176, "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"),
}
SMALL = ["config.json", "generation_config.json", "pytorch_model.bin.index.json",
         "special_tokens_map.json", "tokenizer_config.json"]

RERUN_KEYS = ["workspace/injection_task_13", "workspace/injection_task_4",
              "workspace/injection_task_9", "workspace/injection_task_12",
              "travel/injection_task_0", "travel/injection_task_6"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def canonical_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    pay = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))
    goals = json.loads((ROOT / "data" / "goals_en.json").read_text(encoding="utf-8"))
    order = goals["order"]
    for k in RERUN_KEYS:
        assert k in order, k
    en = {k: pay["conditions"]["en"]["goals"][k] for k in RERUN_KEYS}
    en_sha = canonical_sha(en)

    make_mt_src = (ROOT / "scripts" / "make_mt.py").read_text(encoding="utf-8")
    gen_src = (ROOT / "payloads" / "generate_llm.py").read_text(encoding="utf-8")
    prompts = {p.name: p.read_text(encoding="utf-8")
               for p in sorted((ROOT / "payloads" / "prompts").glob("*.txt"))}
    gen_sha = hashlib.sha256(gen_src.encode()).hexdigest()

    cells = []
    cells.append(md(f"""# bn_mt NLLB-200-3.3B RE-RUN (6 items, D27)

Re-translate **only** the six `bn_mt` items whose attack literals the as-run MT rule did not
protect, under the **D26 literal-protection rules**, with the **same pinned NLLB-200-3.3B**
model and provenance format as `make_bn_mt_nllb.ipynb`.

* **Items:** `{json.dumps(RERUN_KEYS)}`
  — ws13 lost `https://bit.ly/sUf7tOJ`; ws4/9/12 have mandated verbatim **Subject/Body** blocks;
  travel/0 and travel/6 must keep the hotel name **`Riverside View Hotel`** verbatim.
* **Literal set = the LLM validator's** (`payloads/generate_llm.py` `extract_literals`, sha256
  `{gen_sha[:12]}…`): regex PATTERNS (emails, URLs, IBANs, quoted strings, money, dates, times)
  **plus** mandated Subject/Body fields, the ws4 upstream-typo override, and the §4.3 hotel-name
  additions. Each literal is swapped for an ASCII sentinel `LLZ<n>ZLL` before NLLB and restored
  to a fixpoint afterwards (dropped sentinels are caught); any item whose literal does not
  round-trip is written `TODO_MT` (never silently wrong).
* **D27:** the restored text is forced to **ASCII digits** (§4.3: never Bengali digits).
* **Settings:** engine `{MODEL_ID}` @ `{MODEL_REV[:12]}`, beam 4, max_len 512, **sentence-split**,
  fp16 on one T4. Same as the full bn_mt run.
* **Outputs** (`/kaggle/working/bn_mt_nllb_rerun/`): `data/conditions/bn_mt/<suite>__<task>.txt`
  (drops straight into `code/data/conditions/`), `bn_mt_nllb_rerun.json`,
  `mt_report_nllb_rerun.json`, `SHA256SUMS.txt`, and a zip.
* **Licence:** NLLB-200 output ships under CC BY-NC 4.0.
* **Self-contained:** embeds the tested `scripts/make_mt.py` + `payloads/generate_llm.py`; no
  dataset/repo attachment, no HF login (repo not gated). Weights download to scratch (D13).

**Kaggle:** Accelerator **GPU T4 x2** (one GPU used), Internet **ON**, notebook **private**;
then **Save Version → Save & Run All (Commit)**. Expected: ~17.6 GB download + SHA check
(~10–20 min on the Kaggle network), then <1 min for six short translations."""))

    cells.append(md("## 0. Config"))
    cells.append(code(f"""MODEL_ID  = {MODEL_ID!r}
MODEL_REV = {MODEL_REV!r}
FILES = {json.dumps(FILES, indent=1)}
SMALL_FILES = {SMALL!r}
SRC_LANG, TGT_LANG = 'eng_Latn', 'ben_Beng'
BEAM, MAX_LEN = 4, 512
SENTENCE_SPLIT = True
DTYPE = 'float16'
SEED = 20261002
OUTPUT_CONDITION = 'bn_mt'
RERUN_KEYS = {json.dumps(RERUN_KEYS)}
OUT_ROOT = '/kaggle/working/bn_mt_nllb_rerun'
MIN_FREE_GIB = 20.0
EN_GOALS_SHA256 = {en_sha!r}
print(MODEL_ID, MODEL_REV[:12], 'items', len(RERUN_KEYS), 'beam', BEAM, DTYPE, flush=True)
"""))

    cells.append(md("## 1. Embedded inputs (the 6 English goals)"))
    cells.append(code("import json, hashlib\n"
        "EN_GOALS = json.loads(r'''" + json.dumps(en, ensure_ascii=False, indent=1) + "''')\n"
        "def canonical_sha(d):\n"
        "    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()\n"
        "assert sorted(EN_GOALS) == sorted(RERUN_KEYS), 'embedded goals != RERUN_KEYS'\n"
        "assert canonical_sha(EN_GOALS) == EN_GOALS_SHA256, 'embedded EN goals changed'\n"
        "print('EN goals', len(EN_GOALS), 'sha', EN_GOALS_SHA256[:12], flush=True)\n"))

    cells.append(md("## 2. Write the embedded (tested) make_mt + generate_llm package and import it"))
    pkg_writer = (
        "import os, sys\n"
        "PKG='/kaggle/working/genpkg'\n"
        "os.makedirs(PKG+'/scripts', exist_ok=True)\n"
        "os.makedirs(PKG+'/payloads/prompts', exist_ok=True)\n"
        "open(PKG+'/scripts/make_mt.py','w',encoding='utf-8').write(MAKE_MT_SRC)\n"
        "open(PKG+'/payloads/__init__.py','w',encoding='utf-8').write('')\n"
        "open(PKG+'/payloads/generate_llm.py','w',encoding='utf-8').write(GEN_SRC)\n"
        "for name, body in PROMPTS.items():\n"
        "    open(PKG+'/payloads/prompts/'+name,'w',encoding='utf-8').write(body)\n"
        "sys.path.insert(0, PKG); sys.path.insert(0, PKG+'/scripts')\n"
        "import importlib\n"
        "import make_mt as MT; importlib.reload(MT)\n"
        "import payloads.generate_llm as G; importlib.reload(G)\n"
        "assert hashlib.sha256(GEN_SRC.encode()).hexdigest()==GEN_SHA, 'generator source changed'\n"
        "from make_mt import PATTERNS, protect, restore, force_ascii_digits\n"
        "print('make_mt + generate_llm imported; gen sha', GEN_SHA[:12], flush=True)\n")
    cells.append(code(
        "MAKE_MT_SRC = r'''" + make_mt_src.replace("'''", "'\\''\\''\\''") + "'''\n"
        "GEN_SRC = r'''" + gen_src.replace("'''", "'\\''\\''\\''") + "'''\n"
        "GEN_SHA = " + repr(gen_sha) + "\n"
        "PROMPTS = json.loads(r'''" + json.dumps(prompts, ensure_ascii=False) + "''')\n"
        + pkg_writer))

    cells.append(md("""## 3. Full-literal protection (D26 rules) — protect the mandated Subject/Body, URLs, hotel name
`protect_all` uses the SAME literal list as the LLM validator (`G.extract_literals(goal, key)`),
so the verbatim English Subject/Body blocks, the phishing URL and the hotel name are each swapped
for an ASCII sentinel and restored afterwards (longest literal first; a quoted literal matches by
its inner content, exactly as `G.literal_forms`)."""))
    cells.append(code(r'''import re as _re
_SENT = _re.compile(r'(?<=[.!?])\s+(?=[A-Z\'"(\[]|LLZ)')
def split_sentences(prot_text):
    parts = [p.strip() for p in _SENT.split(prot_text) if p.strip()]
    return parts or [prot_text]

def protect_all(text, key):
    """Swap every D26 literal (regex + mandated Subject/Body + ws4 override + §4.3 additions)
    for an ASCII sentinel. Returns (protected_text, {sentinel: literal})."""
    lits = G.extract_literals(text, key)
    literals, prot, i = {}, text, 0
    for lit in sorted(lits, key=len, reverse=True):
        for form in G.literal_forms(lit):
            if form and form in prot:
                i += 1
                tok = f' LLZ{i}ZLL '
                prot = prot.replace(form, tok)
                literals[tok.strip()] = form
                break
    return prot, literals

# self-check on ws13 (URL + mandated subject/body must all be protected)
_t = EN_GOALS['workspace/injection_task_13']
_p, _l = protect_all(_t, 'workspace/injection_task_13')
assert any('bit.ly/sUf7tOJ' in v for v in _l.values()), 'URL not protected'
assert 'LLZ' in _p
print('protect_all ok; ws13 literals:', len(_l), flush=True)
'''))

    cells.append(md("## 4. GPU, packages, scratch space"))
    cells.append(code(r'''import subprocess, shutil
print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'], capture_output=True, text=True).stdout, flush=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'transformers<5', 'sentencepiece', 'sacremoses'], check=True)
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), 'Need a GPU accelerator (T4 x2)'
SCRATCH = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp'
HF_DIR = f'{SCRATCH}/hf_nllb33b'
os.makedirs(HF_DIR, exist_ok=True)
free = shutil.disk_usage(SCRATCH).free / 2**30
print(f'HF_DIR={HF_DIR} free={free:.1f} GiB', flush=True)
assert free >= MIN_FREE_GIB, f'need >= {MIN_FREE_GIB} GiB scratch'
'''))

    cells.append(md("## 5. Download the pinned revision and verify size + SHA-256 of every weight/tokenizer file"))
    cells.append(code(r'''from huggingface_hub import hf_hub_download
import time
t0 = time.time()
local = {}
for fn in list(FILES) + SMALL_FILES:
    local[fn] = hf_hub_download(MODEL_ID, fn, revision=MODEL_REV, local_dir=HF_DIR)
def sha256_file(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()
VERIFIED = {}
for fn, (size, sha) in FILES.items():
    got_size, got_sha = os.path.getsize(local[fn]), sha256_file(local[fn])
    VERIFIED[fn] = {'bytes': got_size, 'sha256': got_sha, 'match': got_size == size and got_sha == sha}
    print(f'{fn}: {got_size} B sha256 {got_sha[:16]}…  match={VERIFIED[fn]["match"]}', flush=True)
assert all(v['match'] for v in VERIFIED.values()), 'pinned file mismatch'
print(f'download+verify {time.time()-t0:.0f}s', flush=True)
'''))

    cells.append(md("## 6. Load NLLB-200-3.3B (fp16) and translate the 6 items with full-literal protection"))
    cells.append(code(r'''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
torch.manual_seed(SEED)
dtype = getattr(torch, DTYPE)
tok = AutoTokenizer.from_pretrained(HF_DIR, src_lang=SRC_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_DIR, torch_dtype=dtype, low_cpu_mem_usage=True).to('cuda:0').eval()
BOS = tok.convert_tokens_to_ids(TGT_LANG)
print('loaded', type(model).__name__, dtype, f'{torch.cuda.memory_allocated(0)/2**30:.1f} GiB', flush=True)

def mt(text):
    enc = tok(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to('cuda:0')
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=BOS, num_beams=BEAM, max_new_tokens=MAX_LEN)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

def strict_literal_audit(en_text, out_text, key):
    lits = G.extract_literals(en_text, key)
    return sorted({l for l in lits if G.literal_found(out_text, l) is None})

def translate(text, key):
    prot, literals = protect_all(text, key)
    pieces = split_sentences(prot) if SENTENCE_SPLIT else [prot]
    raw = ' '.join(mt(p) for p in pieces)
    restored, missing = restore(raw, literals)
    n_digits = 0
    if not missing:
        restored, n_digits = force_ascii_digits(restored)   # D27 §4.3
    return restored, raw, literals, missing, len(pieces), n_digits

t0 = time.time()
RESULTS, REPORT_ITEMS, filled = {}, {}, 0
for key in RERUN_KEYS:
    en = EN_GOALS[key]
    bn, raw, literals, missing, n_sent, n_dig = translate(en, key)
    status = 'NEEDS_FIX_literal_lost' if missing else 'ok'
    RESULTS[key] = 'TODO_MT' if missing else bn
    filled += (not missing)
    REPORT_ITEMS[key] = {'status': status, 'protected_literals': list(literals.values()),
                         'missing': list(missing), 'n_sentences': n_sent, 'digit_normalizations': n_dig,
                         'bn_mt': bn, 'raw_mt': raw,
                         'strict_audit_not_verbatim': strict_literal_audit(en, bn, key) if not missing else None}
    print(f'{key}: {status} ({n_sent} sent., {n_dig} digit fix)  {bn[:90]}', flush=True)
print(f'rerun filled {filled}/{len(RERUN_KEYS)} in {time.time()-t0:.0f}s', flush=True)
'''))

    cells.append(md("## 7. Write outputs (drop-in `data/conditions/bn_mt` layout) + report + SHA-256 + zip"))
    cells.append(code(r'''import datetime as dt, zipfile
from pathlib import Path
root = Path(OUT_ROOT)
cdir = root / 'data' / 'conditions' / OUTPUT_CONDITION
cdir.mkdir(parents=True, exist_ok=True)
for old in cdir.glob('*.txt'):
    old.unlink()
engine = f'{MODEL_ID}@{MODEL_REV}'
run_date = dt.date.today().isoformat()
for key, val in RESULTS.items():
    safe = key.replace('/', '__')
    todo = val == 'TODO_MT'
    p = cdir / (f'{safe}.TODO_MT.txt' if todo else f'{safe}.txt')
    p.write_text(f'# condition: {OUTPUT_CONDITION}\n# key: {key}\n# STATUS: {"TODO_MT" if todo else "filled"}\n'
                 f'# ENGINE: {engine} beam={BEAM} sentence_split={SENTENCE_SPLIT} dtype={DTYPE} rerun=D27 date={run_date}\n'
                 f'# EN: {EN_GOALS[key]}\n\n{val}\n', encoding='utf-8')
payload = {'meta': {'condition': OUTPUT_CONDITION, 'engine': engine, 'beam': BEAM, 'max_len': MAX_LEN,
                    'sentence_split': SENTENCE_SPLIT, 'dtype': DTYPE, 'run_date': run_date, 'rerun': 'D27',
                    'rerun_keys': RERUN_KEYS, 'en_goals_sha256': EN_GOALS_SHA256,
                    'licence': 'CC-BY-NC-4.0 (NLLB-200 output)'},
           'goals': RESULTS}
(root / 'bn_mt_nllb_rerun.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
report = {'model': MODEL_ID, 'revision': MODEL_REV, 'verified_files': VERIFIED, 'src': SRC_LANG, 'tgt': TGT_LANG,
          'run_date': run_date, 'beam': BEAM, 'max_len': MAX_LEN, 'sentence_split': SENTENCE_SPLIT, 'dtype': DTYPE,
          'torch': torch.__version__, 'transformers': transformers.__version__,
          'gpu': torch.cuda.get_device_name(0), 'filled': filled, 'n_goals': len(RERUN_KEYS),
          'rerun': 'D27', 'items': REPORT_ITEMS}
(root / 'mt_report_nllb_rerun.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
def _csha(d):
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
lines = []
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS.txt' and p.suffix != '.zip':
        lines.append(f'{sha256_file(p)}  {p.relative_to(root)}')
PAYLOAD_SET_SHA256 = _csha(RESULTS)
lines.append(f'{PAYLOAD_SET_SHA256}  CANONICAL_GOALS_JSON(bn_mt_rerun, sort_keys, compact, utf-8)')
(root / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with zipfile.ZipFile('/kaggle/working/bn_mt_nllb_rerun.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(root.parent))
print('\n'.join(lines), flush=True)
print(f'NLLB_RERUN_RESULT engine={MODEL_ID}@{MODEL_REV[:12]} filled={filled}/{len(RERUN_KEYS)} '
      f'strict_audit_flags={sum(1 for v in REPORT_ITEMS.values() if v["strict_audit_not_verbatim"])} '
      f'payload_sha256={PAYLOAD_SET_SHA256}', flush=True)
'''))

    cells.append(md("""## 8. After the run (on the box, not here)
1. Download `bn_mt_nllb_rerun.zip` (Output tab) → unzip to a scratch dir; check `SHA256SUMS.txt`
   and the `NLLB_RERUN_RESULT` line; read every `TODO_MT`/`strict_audit_not_verbatim` entry.
2. Merge with `scripts/merge_regen_outputs.py --nllb-rerun-dir <unzipped>/bn_mt_nllb_rerun ...`
   (together with the gen-llm regen output). It re-validates literals, applies the §4.3 ASCII-digit
   rule, merges only passing items into `data/conditions/`, regenerates `PAYLOAD_SHA256SUMS.txt`,
   and reports per-condition counts and 150-pair coverage."""))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells); EN sha {en_sha[:16]}; gen sha {gen_sha[:12]}")


if __name__ == "__main__":
    main()
