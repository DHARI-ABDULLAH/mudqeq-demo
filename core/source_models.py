"""
web_demo/core/source_models.py
------------------------------
The vocabulary shared by the two kinds of source the demo can index.

A "source" is anything the user added to their session and can select, search,
chat about, or analyse a case against. Today that is an uploaded PDF or a
fetched web page. Both flow through the SAME pipeline (chunk -> embed -> FAISS
-> retrieve -> cite); only their provenance labels differ, and this module is
where those labels live so services and UI never re-derive them differently.

Backward compatibility is the point of the defaults here: anything that does
not say otherwise is a PDF, exactly as before URL sources existed.
"""

from __future__ import annotations

from urllib.parse import urlsplit

SOURCE_TYPE_PDF = "pdf"
SOURCE_TYPE_URL = "url"

SOURCE_TYPES = (SOURCE_TYPE_PDF, SOURCE_TYPE_URL)

SOURCE_ICONS = {
    SOURCE_TYPE_PDF: "📄",
    SOURCE_TYPE_URL: "🔗",
}

SOURCE_TYPE_LABELS_AR = {
    SOURCE_TYPE_PDF: "ملف",
    SOURCE_TYPE_URL: "رابط",
}


def normalize_source_type(value) -> str:
    """Coerce anything to a known source type, defaulting to PDF."""
    text = str(value or "").strip().lower()
    return text if text in SOURCE_TYPES else SOURCE_TYPE_PDF


def is_url_source(value) -> bool:
    return normalize_source_type(value) == SOURCE_TYPE_URL


def source_icon(source_type) -> str:
    return SOURCE_ICONS.get(normalize_source_type(source_type), SOURCE_ICONS[SOURCE_TYPE_PDF])


def source_type_label_ar(source_type) -> str:
    return SOURCE_TYPE_LABELS_AR.get(
        normalize_source_type(source_type), SOURCE_TYPE_LABELS_AR[SOURCE_TYPE_PDF]
    )


def domain_of(url: str) -> str:
    """Human-readable host for a citation ("example.com").

    Never raises: a citation label must not be able to break rendering.
    """
    if not url:
        return ""
    try:
        host = urlsplit(str(url)).hostname or ""
    except (ValueError, AttributeError):
        return ""
    host = host.strip().lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def pages_label_ar(page_start, page_end) -> str:
    """Arabic page locator for a PDF chunk."""
    if page_start and page_end and page_start != page_end:
        return f"الصفحات {page_start}–{page_end}"
    if page_start:
        return f"صفحة {page_start}"
    return "صفحة غير معروفة"


def url_locator_ar(section_title: str) -> str:
    """Arabic in-page locator for a web chunk."""
    section = (section_title or "").strip()
    return f"قسم: {section}" if section else "الصفحة كاملة"


def url_citation_ar(page_title: str, url: str, fallback_name: str = "") -> str:
    """Plain-text citation for a web source: "العنوان — example.com".

    A page with no usable title falls back to its domain, and the domain is not
    then repeated on both sides of the dash.
    """
    title = (page_title or "").strip() or (fallback_name or "").strip() or "صفحة ويب"
    domain = domain_of(url)
    if not domain or title.lower() == domain:
        return title
    return f"{title} — {domain}"
