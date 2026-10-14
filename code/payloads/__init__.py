"""LLM-authored injection payload generation (research-plan.md §12, DECISIONS D24).

The former human-native conditions (bn_native/banglish_native/codemix_native, _W2)
are reframed as LLM-AUTHORED conditions (bn_llm/banglish_llm/codemix_llm, _W2):
idiomatic Bangla-script, romanised Banglish and Bangla-English code-mixed injection
payloads written directly by an open LLM (not translated sentence-by-sentence),
compared against MT (NLLB-200-3.3B bn_mt), English (en) and transliteration
(banglish_translit = IndicXlit of bn_llm). The assistant does all the work; no
native authoring/validation is required. See generate_llm.py.
"""
from payloads.generate_llm import (  # noqa: F401
    REGISTER_OF,
    extract_literals,
    build_prompt,
    validate_literals,
    validate_script,
    validate_item,
    parse_generation,
    generate_item,
    generate_condition,
)
