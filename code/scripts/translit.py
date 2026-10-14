"""Bengali -> Roman (Banglish) transliteration.

Primary engine: AI4Bharat IndicXlit (Aksharantar, arXiv 2205.03018, MIT), used in
IndicJR for its romanised condition (research-plan.md 4.3). IndicXlit depends on
fairseq, which needs a C++ toolchain; where it cannot be installed (e.g. the box
has no g++), this module falls back to a deterministic rule-based phonetic
transliterator. The fallback is clearly labelled in every output so it is never
mistaken for IndicXlit output or for native Banglish.

The *real* banglish_translit condition in the paper must be produced from the
human-authored bn_native text with IndicXlit on Kaggle (where g++ and a GPU are
available). This module's fallback exists only to (a) keep the pipeline runnable
on the box and (b) generate clearly-labelled machine DRAFTS for native review.
"""
from __future__ import annotations

ENGINE = None


def _try_indicxlit():
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    try:
        from ai4bharat.transliteration import XlitEngine  # type: ignore
        ENGINE = ("indicxlit", XlitEngine("bn", beam_width=4, src_script_type="indic"))
    except Exception:
        ENGINE = ("fallback", None)
    return ENGINE


# Deterministic Bengali -> Roman phonetic map (fallback only).
_VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
_MATRA = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}
_CONS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t",
    "ং": "ng", "ঃ": "h", "ঁ": "",
}
_HASANTA = "্"
_DIGITS = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
           "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}


def _fallback_word(word: str) -> str:
    import unicodedata
    word = unicodedata.normalize("NFC", word)
    out = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        if ch == "\u09bc":  # stray nukta
            i += 1
            continue
        if ch in _CONS:
            out.append(_CONS[ch])
            if nxt == _HASANTA:            # conjunct: suppress inherent vowel
                i += 2
                continue
            if nxt in _MATRA:              # explicit vowel sign
                out.append(_MATRA[nxt])
                i += 2
                continue
            # inherent vowel: add 'o' unless the next char is itself a vowel sign
            # start or at end of a long word (common Banglish spelling drops final 'o')
            if nxt == "" or (nxt not in _CONS and nxt not in _VOWELS and nxt not in _DIGITS):
                pass  # at end or before punctuation: drop inherent vowel
            else:
                out.append("o")
            i += 1
        elif ch in _VOWELS:
            out.append(_VOWELS[ch]); i += 1
        elif ch in _MATRA:
            out.append(_MATRA[ch]); i += 1
        elif ch in _DIGITS:
            out.append(_DIGITS[ch]); i += 1
        else:
            out.append(ch); i += 1
    return "".join(x for x in out if x)


def transliterate(text: str) -> tuple[str, str]:
    """Return (roman_text, engine_name). Non-Bengali runs (ASCII literals) pass through."""
    engine, obj = _try_indicxlit()
    if engine == "indicxlit":
        try:
            return obj.translit_sentence(text, "bn"), "indicxlit"
        except Exception:
            pass
    # fallback: transliterate each whitespace token; keep ASCII tokens verbatim
    import re
    out = []
    for tok in re.split(r"(\s+)", text):
        if tok.strip() == "" or not any("\u0980" <= c <= "\u09ff" for c in tok):
            out.append(tok)
        else:
            out.append(_fallback_word(tok))
    return "".join(out), "fallback_rulebased"


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "ব্যবহারকারীর পাসওয়ার্ড পরিবর্তন করে new_password করুন।"
    r, eng = transliterate(s)
    print(f"[{eng}] {r}")
