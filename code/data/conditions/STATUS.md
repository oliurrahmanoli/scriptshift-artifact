# Payload scaffold status

Generated: 2026-10-06 16:53 Australia/Sydney.
Source: `data/payloads.json` · goals: **35** / 35 injection tasks.

| Condition | Wrapper | Filled | Total | TODO counts |
|---|---|---:|---:|---|
| `en` | W1 | 35 | 35 | — |
| `bn_mt` | W1 | 35 | 35 | — |
| `bn_llm` | W1 | 35 | 35 | — |
| `banglish_llm` | W1 | 35 | 35 | — |
| `banglish_translit` | W1 | 0 | 35 | TODO_TRANSLIT×35 |
| `codemix_llm` | W1 | 0 | 35 | TODO_LLM×35 |
| `en_W2` | W2 | 35 | 35 | — |
| `bn_llm_W2` | W2 | 0 | 35 | TODO_LLM×35 |
| `banglish_llm_W2` | W2 | 0 | 35 | TODO_LLM×35 |

## Summary vs 35 injection goals

- `en` + `en_W2`: filled.
- `bn_mt`: **35/35** (NLLB-200-3.3B via make_bn_mt_nllb.ipynb).
- LLM-authored `TODO_LLM` goal cells: **105** + wrappers (generate_llm.py on Kaggle).
- `banglish_translit`: blocked on `bn_llm`.
- Frozen fast-track sets (D26): see `PAYLOAD_SHA256SUMS.txt`, `_provenance/` and `_gen_meta.json` in each LLM condition dir; DECISIONS.md D26 lists every fix and every item left TODO.
