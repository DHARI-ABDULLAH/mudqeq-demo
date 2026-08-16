"""
web_demo/compat.py
------------------
Backward-compatible wrappers for mixed Streamlit Cloud deploys.

Streamlit Community Cloud has repeatedly served mismatched file versions
(newer app.py with older ui/components.py or services/*.py). These helpers
let the app start and render even when optional APIs are missing.
"""

from __future__ import annotations

import streamlit as st

from services import document_service, session_service
from ui import components


def live_document_count(session_id: str) -> int:
    if hasattr(session_service, "live_document_count"):
        return int(session_service.live_document_count(session_id))
    doc = _current_document(session_id)
    return 1 if doc is not None else 0


def list_documents(session_id: str) -> list:
    if hasattr(session_service, "list_documents"):
        return session_service.list_documents(session_id)
    doc = _current_document(session_id)
    return [doc] if doc is not None else []


def ready_documents(session_id: str) -> list:
    if hasattr(session_service, "ready_documents"):
        return session_service.ready_documents(session_id)
    doc = _current_document(session_id)
    if doc is not None and getattr(doc, "status", "") == "ready":
        return [doc]
    return []


def session_stats(session_id: str) -> dict:
    if hasattr(session_service, "stats"):
        return session_service.stats(session_id)
    doc = _current_document(session_id)
    if doc is None:
        return {"num_documents": 0, "total_pages": 0, "total_chunks": 0}
    return {
        "num_documents": 1,
        "total_pages": doc.num_pages,
        "total_chunks": doc.num_chunks,
    }


def render_dashboard(session_id: str) -> None:
    """Render stats row — works with old and new ui/components.py."""
    remaining = session_service.remaining_questions(session_id)
    stats = session_stats(session_id)
    if hasattr(components, "dashboard"):
        components.dashboard(stats, remaining)
        return
    if hasattr(components, "session_dashboard"):
        components.session_dashboard(_current_document(session_id))
        return
    # Last-resort inline fallback (no dependency on components CSS classes).
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد المستندات", stats.get("num_documents", 0))
    c2.metric("إجمالي الصفحات", stats.get("total_pages", 0))
    c3.metric("إجمالي المقاطع", stats.get("total_chunks", 0))
    c4.metric("الأسئلة المتبقية", remaining)


def render_document_card(doc) -> None:
    if hasattr(components, "document_card"):
        components.document_card(doc)
        return
    status = getattr(doc, "status", "ready")
    st.markdown(f"**{doc.display_name}** — {doc.num_pages} صفحة · {doc.num_chunks} مقطع · {status}")


def delete_document(session_id: str, document_id: str) -> None:
    if hasattr(document_service, "delete_document"):
        document_service.delete_document(session_id, document_id)
        return
    if hasattr(document_service, "delete_current"):
        document_service.delete_current(session_id)
        return
    raise RuntimeError("No document delete API available")


def _current_document(session_id: str):
    if hasattr(session_service, "current_document"):
        return session_service.current_document(session_id)
    return None
