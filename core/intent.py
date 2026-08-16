"""
web_demo/core/intent.py
-----------------------
Deterministic question-intent detection for the demo chat.

"What is this document about?" and "لخص المستند" are not lookups — nearest
neighbour search on such a query returns whichever chunk happens to sit closest
to a vague sentence, which is a poor basis for a summary. Those questions get a
document-level context instead (ordered, bounded chunks).

Pure string matching: no model call, no network, no per-request cost, and the
result is reproducible in tests.
"""

from __future__ import annotations

import re
import unicodedata

OVERVIEW = "overview"
FACTUAL = "factual"

_TASHKEEL = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = re.compile(r"\u0640+")
_NON_WORD = re.compile(r"[^\w\s\u0600-\u06ff]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold case, strip Arabic diacritics, and unify letter variants."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    for src, dst in (("أإآٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        for ch in src:
            text = text.replace(ch, dst)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# Patterns are written in NORMALIZED form: lower-cased, punctuation already
# turned into spaces ("what's" -> "what s", "TL;DR" -> "tl dr"), and Arabic
# folded (ى->ي so "محتوى" -> "محتوي", ة->ه so "نبذة" -> "نبذه").
_OVERVIEW_PATTERNS = [
    # --- English ---------------------------------------------------------
    r"\b(summarize|summarise|summary|overview|recap)\b",
    r"\btl\s*dr\b",
    r"\bwhat(?:\s+s|\s+is|\s+are)?\s+(?:in|inside|within)\b",
    r"\bwhat\s+(?:is|are)\s+(?:this|the)\s+(?:document|doc|pdf|file|paper)\b",
    r"\bwhat\s+(?:does|do)\s+(?:this|the)\s+\w+\s+(?:say|contain|cover|talk|discuss)\b",
    r"\b(?:tell|give)\s+me\s+(?:about|an?\s+overview|an?\s+summary)\b",
    r"\bmain\s+(?:points|ideas|topics)\b",
    r"\bwhat\s+is\s+it\s+about\b",
    # --- Arabic (incl. Gulf dialect) --------------------------------------
    r"\b(لخص|تلخيص|ملخص|نبذه|خلاصه)\b",
    r"\b(شنو|وش|ايش|ايه|ما|ماذا)\s+(داخل|في|محتوي|يحتوي)\b",
    r"\bمحتوي\s+(هذا\s+)?(المستند|الملف|الوثيقه|المرفق)\b",
    r"\b(عن\s+ماذا|عما|عن\s+ايش|عن\s+شنو)\s+(يتحدث|يتكلم|يدور)\b",
    r"\b(يتحدث|يتكلم|يدور)\s+(عن\s+ماذا|عن\s+ايش)\b",
    r"\b(ما|شنو|وش|ايش)\s+(هو\s+)?(موضوع|فكره)\b",
    r"\bنظره\s+عامه\b",
]

_COMPILED = [re.compile(p) for p in _OVERVIEW_PATTERNS]


def classify(question: str) -> str:
    """Return ``OVERVIEW`` for whole-document questions, else ``FACTUAL``."""
    normalized = normalize(question)
    if not normalized:
        return FACTUAL
    for pattern in _COMPILED:
        if pattern.search(normalized):
            return OVERVIEW
    return FACTUAL


def is_overview(question: str) -> bool:
    return classify(question) == OVERVIEW
