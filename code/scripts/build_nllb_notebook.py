"""Generate kaggle/make_bn_mt_nllb.ipynb (checklist D8; DECISIONS D23).

Self-contained Kaggle notebook (no dataset / repo attachment): the 35 English goals and the W1
wrapper are embedded from data/payloads.json, and the literal-protection code (PATTERNS, protect,
restore) is copied VERBATIM from scripts/make_mt.py so the rule is identical to the box run (D5/D6).

Run from code/:  python scripts/build_nllb_notebook.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "make_bn_mt_nllb.ipynb"

MODEL_ID = "facebook/nllb-200-3.3B"
MODEL_REV = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"   # HF API main @ 2026-10-06 (lastModified 2023-02-11)
# LFS sha256 + sizes from https://huggingface.co/api/models/facebook/nllb-200-3.3B/revision/<rev>?blobs=true
FILES = {
    "pytorch_model-00001-of-00003.bin": (6933375959, "06aec913c891950ff0c59bd018c1b46dc95be6e106c36df87d13175418b3972c"),
    "pytorch_model-00002-of-00003.bin": (8545249147, "1c5a44acb0287e905a62a506cf3e5af37956607f4c667d891ec069aa276be0be"),
    "pytorch_model-00003-of-00003.bin": (2098840299, "2cf744009996251567d7972f4436d65cf94380c5ddd8f524cb878e090b27bb50"),
    "sentencepiece.bpe.model": (4852054, "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"),
    "tokenizer.json": (17331176, "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"),
}
SMALL = ["config.json", "generation_config.json", "pytorch_model.bin.index.json",
         "special_tokens_map.json", "tokenizer_config.json"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


def literal_block() -> str:
    src = (ROOT / "scripts" / "make_mt.py").read_text(encoding="utf-8")
    start = src.index("# Literal patterns to protect")
    end = src.index("def main():")
    return src[start:end].rstrip() + "\n"


def canonical_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    pay = json.loads((ROOT / "data" / "payloads.json").read_text(encoding="utf-8"))
    goals = json.loads((ROOT / "data" / "goals_en.json").read_text(encoding="utf-8"))
    order = goals["order"]
    en = {k: pay["conditions"]["en"]["goals"][k] for k in order}
    assert len(en) == 35 and all(v and not v.startswith("TODO") for v in en.values())
    w1 = pay["wrappers"]["W1"]["en"]
    en_sha = canonical_sha(en)
    lit = literal_block()
    lit_sha = hashlib.sha256(lit.encode("utf-8")).hexdigest()

    cells = [
        md(f"""# bn_mt with NLLB-200-3.3B (Kaggle GPU) — checklist D8
Research plan §4.3 / §10.3 / §11.6: regenerate the machine-translated Bangla condition for **all 35
goals** (+ the W1 wrapper) with the plan's engine **`{MODEL_ID}`**, pinned revision
`{MODEL_REV[:12]}`, replacing the box's distilled-600M feasibility run (DECISIONS D5).

* **Literal preservation:** identical rule to `scripts/make_mt.py` (D6) — emails, URLs, IBANs, quoted
  strings, money, dates and times are swapped for ASCII sentinels `LLZ<n>ZLL` before MT and restored to
  a fixpoint afterwards; any item whose sentinel does not come back is written as `TODO_MT` (never
  silently wrong). The code block below is copied verbatim from `make_mt.py` (sha256 `{lit_sha[:12]}…`).
  A stricter *report-only* audit also lists any original literal that is not present verbatim. **D27:** Bengali digits in the restored text are mapped to ASCII (§4.3: never Bengali digits).
* **Settings (§4.3):** beam 4, max length 512 tokens, **sentence-split input** (plan §4.3; the box
  600M run did not split — set `SENTENCE_SPLIT=False` to reproduce that), fp16 on one T4.
* **Outputs** (`/kaggle/working/bn_mt_nllb-3.3b/`): `data/conditions/bn_mt/<suite>__<task>.txt` (+
  `_wrappers/W1__bn_mt.txt`) in the same layout as `code/data/conditions/` (importable with
  `scripts/import_condition_dirs.py`), `bn_mt_nllb-3.3b.json`, `mt_report_nllb-3.3b.json`,
  `SHA256SUMS.txt` (every file + a canonical payload-set hash), and a zip of the lot.
* **Licence:** NLLB-200 is **CC-BY-NC-4.0** → the generated text ships separately under CC BY-NC 4.0.
* **Self-contained:** no dataset or repo attachment; no HF login (repo not gated). Weights download to
  scratch (`/kaggle/tmp` or `/tmp`, D13), not `/kaggle/working`.

**Kaggle settings:** Accelerator **GPU T4 x2** (one GPU is used; 3.3B fp16 ≈ 6.7 GB), Internet **on**,
keep the notebook **private**. Then **Save Version → Save & Run All (Commit)**. Expected: ~17.6 GB
download + SHA check, then a few minutes of translation. Prepared 2026-10-06, **not yet run**."""),
        md("## 0. Config"),
        code(f"""MODEL_ID  = {MODEL_ID!r}
MODEL_REV = {MODEL_REV!r}   # pinned HF commit (main @ 2026-10-06)
FILES = {json.dumps(FILES, indent=1)}   # name -> (bytes, LFS sha256), asserted after download
SMALL_FILES = {SMALL!r}
SRC_LANG, TGT_LANG = 'eng_Latn', 'ben_Beng'
BEAM, MAX_LEN = 4, 512            # plan §4.3
SENTENCE_SPLIT = True             # plan §4.3 'sentence-split input' (box 600M run: False)
DTYPE = 'float16'                 # T4 has no bf16; 'float32' (~13.2 GB) also fits one T4 if fp16 misbehaves
SEED = 20261002
OUTPUT_CONDITION = 'bn_mt'        # directory name in data/conditions/ (see open question in D23)
OUT_ROOT = '/kaggle/working/bn_mt_nllb-3.3b'
MIN_FREE_GIB = 20.0
EN_GOALS_SHA256 = {en_sha!r}   # canonical JSON of the 35 embedded EN goals
print(MODEL_ID, MODEL_REV[:12], 'beam', BEAM, 'split', SENTENCE_SPLIT, DTYPE, flush=True)
"""),
        md("## 1. Embedded inputs (35 English goals from `data/payloads.json` condition `en`, W1 wrapper)"),
        code("import json, hashlib\nEN_GOALS = json.loads(r'''" + json.dumps(en, ensure_ascii=False, indent=1) + "''')\n"
             "W1_EN = json.loads(r'''" + json.dumps(w1, ensure_ascii=False) + "''')\n"
             "def canonical_sha(d):\n"
             "    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()\n"
             "assert len(EN_GOALS) == 35, len(EN_GOALS)\n"
             "assert canonical_sha(EN_GOALS) == EN_GOALS_SHA256, 'embedded EN goals changed'\n"
             "print('EN goals:', len(EN_GOALS), 'sha256', EN_GOALS_SHA256[:16], flush=True)\n"),
        md("## 2. Literal protection — verbatim copy of `scripts/make_mt.py` (D6)"),
        code("import re\n" + lit),
        code(r'''# Report-only strict audit (does NOT change status): which original literals are not verbatim?
def strict_literal_audit(en_text, out_text):
    lits = []
    for kind, pat in PATTERNS:
        lits += [m.group(0) for m in pat.finditer(en_text)]
    return sorted({l for l in lits if l not in out_text})

_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\'"(\[]|LLZ)')
def split_sentences(prot_text):
    """Split AFTER protection, so dots inside emails/URLs/amounts are already sentinels."""
    parts = [p.strip() for p in _SENT.split(prot_text) if p.strip()]
    return parts or [prot_text]
print('literal block ok; sentinel example:', protect("Send 'hi' to a@b.com on 2024-05-30.")[0], flush=True)
'''),
        md("## 3. GPU, packages, scratch space"),
        code(r'''import subprocess, os, shutil, sys
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
assert free >= MIN_FREE_GIB, f'need >= {MIN_FREE_GIB} GiB scratch for the fp32 .bin shards'
'''),
        md("## 4. Download the pinned revision and verify size + SHA-256 of every weight/tokenizer file"),
        code(r'''from huggingface_hub import hf_hub_download
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
'''),
        md("## 5. Load NLLB-200-3.3B (fp16, cuda:0) and translate with literal protection"),
        code(r'''from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
torch.manual_seed(SEED)
dtype = getattr(torch, DTYPE)
tok = AutoTokenizer.from_pretrained(HF_DIR, src_lang=SRC_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_DIR, torch_dtype=dtype, low_cpu_mem_usage=True).to('cuda:0').eval()
BOS = tok.convert_tokens_to_ids(TGT_LANG)
print('loaded', type(model).__name__, dtype, f'{torch.cuda.memory_allocated(0)/2**30:.1f} GiB on cuda:0', flush=True)

def mt(text):
    enc = tok(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to('cuda:0')
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=BOS, num_beams=BEAM, max_new_tokens=MAX_LEN)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

def translate(text):
    prot, literals = protect(text)
    pieces = split_sentences(prot) if SENTENCE_SPLIT else [prot]
    raw = ' '.join(mt(p) for p in pieces)
    restored, missing = restore(raw, literals)
    if not missing:
        restored, _ndig = force_ascii_digits(restored)  # D27: §4.3 ASCII digits, never Bengali
    return restored, raw, literals, missing, len(pieces)

t0 = time.time()
RESULTS, REPORT_ITEMS, filled = {}, {}, 0
for key, en in EN_GOALS.items():
    bn, raw, literals, missing, n_sent = translate(en)
    status = 'NEEDS_FIX_literal_lost' if missing else 'ok'
    RESULTS[key] = 'TODO_MT' if missing else bn
    filled += (not missing)
    REPORT_ITEMS[key] = {'status': status, 'protected_literals': list(literals.values()),
                         'missing': list(missing), 'n_sentences': n_sent, 'bn_mt': bn, 'raw_mt': raw,
                         'strict_audit_not_verbatim': strict_literal_audit(en, bn) if not missing else None}
    print(f'{key}: {status} ({n_sent} sent.)  {bn[:90]}', flush=True)
print(f'bn_mt filled {filled}/{len(EN_GOALS)} in {time.time()-t0:.0f}s', flush=True)
'''),
        md("## 6. W1 wrapper (placeholders protected exactly as in `make_mt.py`)"),
        code(r'''w1_prot = W1_EN.replace('{goal}', ' LLGOALLL ').replace('{user}', ' LLUSERLL ').replace('{model}', ' LLMODELLL ')
w1_bn = mt(w1_prot)   # make_mt.py translates the wrapper as one segment (no sentence split)
w1_bn = w1_bn.replace('LLGOALLL', '{goal}').replace('LLUSERLL', '{user}').replace('LLMODELLL', '{model}')
W1_STATUS = 'ok' if all(p in w1_bn for p in ('{goal}', '{user}', '{model}')) else f'NEEDS_FIX placeholders lost: {w1_bn!r}'
print('W1:', W1_STATUS, '|', w1_bn[:160], flush=True)
'''),
        md("## 7. Write outputs in the `code/data/conditions/bn_mt` layout + report + SHA-256"),
        code(r'''import datetime as dt, zipfile
from pathlib import Path
root = Path(OUT_ROOT)
cdir = root / 'data' / 'conditions' / OUTPUT_CONDITION
wdir = root / 'data' / 'conditions' / '_wrappers'
for d in (cdir, wdir):
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob('*.txt'):
        old.unlink()
engine = f'{MODEL_ID}@{MODEL_REV}'
run_date = dt.date.today().isoformat()
for key, val in RESULTS.items():
    safe = key.replace('/', '__')
    todo = val == 'TODO_MT'
    p = cdir / (f'{safe}.TODO_MT.txt' if todo else f'{safe}.txt')
    p.write_text(f'# condition: {OUTPUT_CONDITION}\n# key: {key}\n# STATUS: {"TODO_MT" if todo else "filled"}\n'
                 f'# ENGINE: {engine} beam={BEAM} sentence_split={SENTENCE_SPLIT} dtype={DTYPE} date={run_date}\n'
                 f'# EN: {EN_GOALS[key]}\n\n{val}\n', encoding='utf-8')
if W1_STATUS == 'ok':
    (wdir / f'W1__{OUTPUT_CONDITION}.txt').write_text(
        f'# Wrapper W1 for condition={OUTPUT_CONDITION}\n# Placeholders: {{goal}}, {{user}}, {{model}}\n'
        f'# STATUS: filled\n# ENGINE: {engine}\n\n{w1_bn}\n', encoding='utf-8')
payload = {'meta': {'condition': OUTPUT_CONDITION, 'engine': engine, 'beam': BEAM, 'max_len': MAX_LEN,
                    'sentence_split': SENTENCE_SPLIT, 'dtype': DTYPE, 'run_date': run_date,
                    'en_goals_sha256': EN_GOALS_SHA256, 'licence': 'CC-BY-NC-4.0 (NLLB-200 output)'},
           'goals': RESULTS, 'wrapper_W1': w1_bn if W1_STATUS == 'ok' else 'TODO_MT'}
(root / 'bn_mt_nllb-3.3b.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
report = {'model': MODEL_ID, 'revision': MODEL_REV, 'verified_files': VERIFIED, 'src': SRC_LANG, 'tgt': TGT_LANG,
          'run_date': run_date, 'beam': BEAM, 'max_len': MAX_LEN, 'sentence_split': SENTENCE_SPLIT, 'dtype': DTYPE,
          'torch': torch.__version__, 'transformers': transformers.__version__,
          'gpu': torch.cuda.get_device_name(0), 'filled': filled, 'n_goals': len(EN_GOALS),
          'wrapper_W1_bn_mt': W1_STATUS, 'items': REPORT_ITEMS}
(root / 'mt_report_nllb-3.3b.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
# SHA-256: every output file + the canonical payload-set hash (sorted-key compact JSON of the 35 goals)
lines = []
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS.txt' and p.suffix != '.zip':
        lines.append(f'{sha256_file(p)}  {p.relative_to(root)}')
PAYLOAD_SET_SHA256 = canonical_sha(RESULTS)
lines.append(f'{PAYLOAD_SET_SHA256}  CANONICAL_GOALS_JSON({OUTPUT_CONDITION}, sort_keys, compact, utf-8)')
(root / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with zipfile.ZipFile('/kaggle/working/bn_mt_nllb-3.3b.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(root.parent))
print('\n'.join(lines[-3:]), flush=True)
print(f'NLLB_RESULT engine={MODEL_ID}@{MODEL_REV[:12]} filled={filled}/{len(EN_GOALS)} wrapper={W1_STATUS[:12]} '
      f'strict_audit_flags={sum(1 for v in REPORT_ITEMS.values() if v["strict_audit_not_verbatim"])} '
      f'payload_sha256={PAYLOAD_SET_SHA256}', flush=True)
'''),
        md("""## 8. After the run (on the box, not here)
1. Download `bn_mt_nllb-3.3b.zip` (Output tab) → unzip to a scratch dir; check `SHA256SUMS.txt` and the
   `NLLB_RESULT` log line; read every `TODO_MT` / `strict_audit_not_verbatim` entry in the report.
2. Decide (authors, D23 open question): does NLLB-3.3B replace the 600M `bn_mt` until Google NMT exists, or
   go to a separate `bn_mt_nllb` condition (plan: Google NMT = `bn_mt`, NLLB = robustness subset)?
3. Copy `data/conditions/<cond>/` into `code/data/conditions/` and run `scripts/import_condition_dirs.py`;
   log engine, revision, date and the payload SHA-256 in `DECISIONS.md` before any main-grid use."""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells); EN sha {en_sha[:16]}; literal block sha {lit_sha[:16]}")


if __name__ == "__main__":
    main()
