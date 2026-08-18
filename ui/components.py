"""
web_demo/ui/components.py
-------------------------
Reusable Streamlit UI fragments for the demo.

All user-derived strings (display filename, retrieved snippets) are HTML-escaped
before being placed inside ``unsafe_allow_html`` blocks to prevent HTML/JS
injection from a malicious PDF or filename.
"""

from __future__ import annotations

import html
import time

import streamlit as st

from config import APP_NAME_AR, APP_TAGLINE_AR, DEMO_VERSION
from core.source_models import (
    SOURCE_TYPE_URL,
    domain_of,
    normalize_source_type,
    source_icon,
    source_type_label_ar,
)


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def brand_sidebar() -> None:
    st.markdown(
        f"""
        <div style="display:flex; gap:0.7rem; align-items:center;">
          <div class="brand-logo">م</div>
          <div>
            <p class="brand-title">{_esc(APP_NAME_AR)}</p>
            <p class="brand-tag">{_esc(APP_TAGLINE_AR)}</p>
          </div>
        </div>
        <div class="demo-badge">نسخة تجريبية عامة · {_esc(DEMO_VERSION)}</div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_privacy() -> None:
    st.markdown(
        """
        <div class="privacy-box side">
          <b>تنبيه:</b> هذه نسخة تجريبية <b>تعمل عبر الإنترنت</b>. عند استخدام
          المحادثة، يتم إرسال السؤال والمقاطع ذات الصلة اللازمة لإنتاج الإجابة
          إلى مزوّد نموذج الذكاء الاصطناعي. <b>لا تستخدم هذه النسخة للمستندات
          السرية أو الحساسة.</b> للاستخدام المحلي والخاص استخدم نسخة Desktop.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="page-title">{_esc(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{_esc(subtitle)}</div>', unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{_esc(APP_NAME_AR)} — Mudqeq AI</h1>
          <p>ارفع مستند PDF واسأل عنه باللغة العربية أو الإنجليزية، واحصل على
          إجابة مدعومة باقتباسات من صفحات المستند. هذه نسخة تجريبية عامة
          للعرض؛ النسخة المكتبية الكاملة تعمل محلياً بالكامل وبخصوصية تامة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


_STATUS_LABELS = {
    "ready": "جاهز",
    "processing": "قيد المعالجة",
    "needs_ocr": "يحتاج OCR",
    "error": "خطأ",
}

_STATUS_BADGE_CLASS = {
    "ready": "badge-ready",
    "processing": "badge-ocr",
    "needs_ocr": "badge-ocr",
    "error": "badge-error",
}


def status_badge(status: str) -> str:
    """Return an HTML badge span for a document status."""
    label = _STATUS_LABELS.get(status, status)
    css = _STATUS_BADGE_CLASS.get(status, "badge-ocr")
    return f'<span class="badge {css}">{_esc(label)}</span>'


def dashboard(stats: dict, remaining_questions: int) -> None:
    """Render SESSION-ONLY stats. No global/user statistics are shown."""
    num_docs = stats.get("num_documents", 0)
    num_sources = stats.get("num_sources", num_docs)
    num_urls = stats.get("num_urls", 0)
    total_pages = stats.get("total_pages", 0)
    total_chunks = stats.get("total_chunks", 0)
    breakdown = f"{num_docs} ملف · {num_urls} رابط"
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-value">{_esc(num_sources)}</div>
            <div class="stat-label">عدد المصادر ({_esc(breakdown)})</div></div>
          <div class="stat-card"><div class="stat-value">{total_pages:,}</div>
            <div class="stat-label">إجمالي الصفحات</div></div>
          <div class="stat-card"><div class="stat-value">{total_chunks:,}</div>
            <div class="stat-label">إجمالي المقاطع</div></div>
          <div class="stat-card"><div class="stat-value">{_esc(remaining_questions)}</div>
            <div class="stat-label">الأسئلة المتبقية</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _retrieved_label(retrieved_at: float) -> str:
    if not retrieved_at:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(retrieved_at)))
    except (TypeError, ValueError, OSError):
        return ""


def source_label(source) -> str:
    """Icon + name for one source, as shown in the selection list."""
    icon = source_icon(getattr(source, "source_type", None))
    name = getattr(source, "display_name", "") or "مصدر"
    if getattr(source, "is_url", False):
        domain = getattr(source, "domain", "") or domain_of(getattr(source, "url", ""))
        if domain:
            return f"{icon} {name} — {domain}"
    return f"{icon} {name}"


def document_card(doc) -> None:
    """Render one source card — an uploaded file or a fetched web page.

    Both kinds show name, type, status, and chunk count. A web source also
    shows its domain and when it was fetched, because unlike a file its content
    can change after it was indexed.
    """
    is_url = bool(getattr(doc, "is_url", False))
    icon = source_icon(getattr(doc, "source_type", None))
    name = _esc(getattr(doc, "display_name", "") or "مصدر")
    badge = status_badge(getattr(doc, "status", ""))
    kind = _esc(source_type_label_ar(getattr(doc, "source_type", None)))

    if is_url:
        url = getattr(doc, "url", "") or ""
        domain = _esc(getattr(doc, "domain", "") or domain_of(url))
        parts = [kind, domain, f"{_esc(getattr(doc, 'num_chunks', 0))} مقطع"]
        fetched = _retrieved_label(getattr(doc, "retrieved_at", 0.0))
        if fetched:
            parts.append(f"جُلبت في {_esc(fetched)}")
        meta = " · ".join(p for p in parts if p)
        link = (
            f'<a class="doc-link" href="{_esc(url)}" target="_blank" '
            f'rel="noopener noreferrer nofollow">فتح الصفحة الأصلية</a>'
            if url
            else ""
        )
    else:
        meta = (
            f"{kind} · {_esc(getattr(doc, 'num_pages', 0))} صفحة · "
            f"{_esc(getattr(doc, 'num_chunks', 0))} مقطع"
        )
        link = ""

    st.markdown(
        f"""
        <div class="doc-card">
          <div class="doc-title">{icon} {name} &nbsp; {badge}</div>
          <div class="doc-meta">{meta}</div>
          {link}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Unified name — a card renders either source kind.
source_card_record = document_card


def session_dashboard(doc) -> None:
    """Backward-compatible alias for older app versions (single-document view)."""
    if doc is None:
        dashboard({"num_documents": 0, "total_pages": 0, "total_chunks": 0}, 0)
    else:
        dashboard(
            {
                "num_documents": 1,
                "total_pages": doc.num_pages,
                "total_chunks": doc.num_chunks,
            },
            0,
        )


def source_card(result: dict) -> None:
    """Render one retrieved chunk's provenance line.

    A document chunk cites a file and a page range. A web chunk cites the page
    title, its section, and its domain, and links to the address the server
    fetched — taken from the chunk's stored metadata, never from model output.
    """
    score = result.get("score", 0.0)
    try:
        score_text = f"{float(score):.2f}"
    except (TypeError, ValueError):
        score_text = "—"

    if normalize_source_type(result.get("source_type")) == SOURCE_TYPE_URL:
        url = str(result.get("url") or "")
        title = result.get("page_title") or result.get("document_name") or "صفحة ويب"
        domain = domain_of(url)
        section = str(result.get("section_title") or "").strip()
        bits = [_esc(title)]
        if section:
            bits.append(_esc(section))
        if domain:
            bits.append(_esc(domain))
        head = "🔗 " + " · ".join(bits)
        if url:
            head += (
                f' &nbsp;<a class="src-link" href="{_esc(url)}" target="_blank" '
                f'rel="noopener noreferrer nofollow">فتح الرابط</a>'
            )
    else:
        ps, pe = result.get("page_start"), result.get("page_end")
        if ps and pe and ps != pe:
            pages = f"صفحات {ps}-{pe}"
        elif ps:
            pages = f"صفحة {ps}"
        else:
            pages = "صفحة غير معروفة"
        head = f"📄 {_esc(result.get('document_name', 'مستند'))} · {_esc(pages)}"

    st.markdown(
        f"""
        <div class="src-card">
          <span class="src-score">{score_text}</span>
          <div class="src-head">{head}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def excerpt(text: str, limit: int = 400) -> None:
    snippet = (text or "")[:limit]
    if len(text or "") > limit:
        snippet += "…"
    st.text(snippet)
