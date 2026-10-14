"""Text normaliser for the relaxed, language-robust scorer (research-plan.md 4.4).

Normalisation steps (layer 2, "normalised deterministic score"):
  * Unicode NFC
  * case folding
  * whitespace collapse
  * punctuation/quote stripping
  * Bangla digits ০-৯ mapped to ASCII 0-9
These are deterministic and cheap, which is why layer 2 is the pre-registered
primary metric.
"""
from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}
# Quotes and common punctuation to drop so "গুরুত্বপূর্ণ বার্তা!" == "গুরুত্বপূর্ণ বার্তা"
_PUNCT = re.compile(r"""['"“”‘’'`.,;:!?()\[\]{}<>…।॥\-–—_/\\]+""")
_WS = re.compile(r"\s+")


def normalise(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(_BN_DIGITS)
    s = s.casefold()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s
