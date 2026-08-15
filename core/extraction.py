"""
web_demo/core/extraction.py
---------------------------
Generic, document-agnostic PDF text extraction for the public demo.

This is a self-contained reimplementation of the desktop project's *generic*
extraction path (the destructive `aaoifi_legacy` profile is intentionally NOT
included — the demo must work with arbitrary Arabic/English PDFs).

Security posture:
- pdfplumber only, pure-Python parsing. No shell commands, no OCR.
- Order-preserving Unicode normalization. Never reorders/reverses Arabic text.
- Hard caps on total extracted characters to bound memory.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from config import MAX_EXTRACTED_CHARS, OCR_CHAR_THRESHOLD

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


class PDFValidationError(Exception):
    """Raised when a file is not a usable, unencrypted PDF."""


@dataclass
class PageResult:
    page_number: int
    text: str
    character_count: int
    has_text: bool
    needs_ocr: bool


@dataclass
class ExtractionResult:
    pages: list[PageResult] = field(default_factory=list)
    truncated: bool = False

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.character_count for p in self.pages)

    def non_empty_pages(self) -> list[PageResult]:
        return [p for p in self.pages if p.has_text]

    def has_usable_text(self) -> bool:
        return any(p.has_text for p in self.pages)


def generic_clean(text: str) -> str:
    """Safe normalization applied to every document. Order-preserving.

    Removes control characters (except newline/tab) and zero-width joiners,
    collapses runs of spaces/newlines, and applies NFC normalization so that
    combined Arabic forms compare/search consistently.
    """
    if not text:
        return ""
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )
    text = text.translate(_ZERO_WIDTH)
    # NFC keeps Arabic in its correct, composed order — never reorders glyphs.
    text = unicodedata.normalize("NFC", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: str | Path, progress=None) -> ExtractionResult:
    """Extract + clean every page using the generic profile.

    Stops early (``truncated=True``) once MAX_EXTRACTED_CHARS is reached to
    bound memory on a public server. ``progress`` is an optional callback
    ``(fraction: float, message: str)``.
    """
    path = Path(path)
    result = ExtractionResult()
    running_chars = 0

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            text = generic_clean(raw)
            char_count = len(text.strip())
            has_text = char_count > 0
            needs_ocr = char_count < OCR_CHAR_THRESHOLD
            result.pages.append(
                PageResult(
                    page_number=i + 1,
                    text=text,
                    character_count=char_count,
                    has_text=has_text,
                    needs_ocr=needs_ocr,
                )
            )
            running_chars += char_count
            if running_chars >= MAX_EXTRACTED_CHARS:
                result.truncated = True
                break
            if progress and (i % 10 == 0 or i == total - 1):
                progress((i + 1) / total, f"جاري استخراج الصفحات... ({i + 1}/{total})")

    return result
