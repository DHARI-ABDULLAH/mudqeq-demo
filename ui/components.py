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

import streamlit as st

from config import APP_NAME_AR, APP_TAGLINE_AR, DEMO_VERSION


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
    total_pages = stats.get("total_pages", 0)
    total_chunks = stats.get("total_chunks", 0)
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-value">{_esc(num_docs)}</div>
            <div class="stat-label">عدد المستندات</div></div>
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


def document_card(doc) -> None:
    """Render a single document card (accepts a DocumentRecord)."""
    name = _esc(doc.display_name)
    badge = status_badge(doc.status)
    meta = f"{_esc(doc.num_pages)} صفحة · {_esc(doc.num_chunks)} مقطع"
    st.markdown(
        f"""
        <div class="doc-card">
          <div class="doc-title">{name} &nbsp; {badge}</div>
          <div class="doc-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_card(result: dict) -> None:
    ps, pe = result.get("page_start"), result.get("page_end")
    if ps and pe and ps != pe:
        pages = f"صفحات {ps}-{pe}"
    elif ps:
        pages = f"صفحة {ps}"
    else:
        pages = "صفحة غير معروفة"
    score = result.get("score", 0.0)
    name = _esc(result.get("document_name", "مستند"))
    st.markdown(
        f"""
        <div class="src-card">
          <span class="src-score">{score:.2f}</span>
          <div class="src-head">{name} · {_esc(pages)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def excerpt(text: str, limit: int = 400) -> None:
    snippet = (text or "")[:limit]
    if len(text or "") > limit:
        snippet += "…"
    st.text(snippet)
