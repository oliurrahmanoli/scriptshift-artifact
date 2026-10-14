# Attack condition directories (authoring scaffold)

Canonical runtime source of truth remains `../payloads.json` (loaded by
`attacks/multilingual_attack.py`). These directories mirror that file for
human authoring and review.

## Conditions (§4.3 / §11)

| Dir | Meaning | Who fills |
|---|---|---|
| `en` | English W1 (AgentDojo verbatim) | filled |
| `bn_mt` | Machine-translated Bangla (NLLB-200-3.3B) | `make_bn_mt_nllb.ipynb` |
| `bn_llm` | **LLM-authored** idiomatic Bangla | `generate_llm.py` (Gemma) |
| `banglish_llm` | **LLM-authored** romanised Banglish | `generate_llm.py` (Gemma) |
| `banglish_translit` | IndicXlit of `bn_llm` | scripts after `bn_llm` |
| `codemix_llm` | **LLM-authored** code-mix | `generate_llm.py` (Gemma) |
| `en_W2` / `bn_llm_W2` / `banglish_llm_W2` | Register ablation | en filled; `_llm` by generator |

Wrappers: `_wrappers/`. `TODO_*` sentinels must not run (attack class refuses).

Regenerate: `python scripts/export_condition_dirs.py`
Import: `python scripts/import_condition_dirs.py`
