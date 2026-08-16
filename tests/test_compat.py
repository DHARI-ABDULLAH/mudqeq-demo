"""Tests for compat fallbacks on stale Streamlit Cloud deploys."""

from __future__ import annotations

import types

import compat


def test_render_dashboard_uses_session_dashboard_when_dashboard_missing(monkeypatch):
    fake_components = types.SimpleNamespace(
        session_dashboard=lambda doc: setattr(fake_components, "_called", doc)
    )
    monkeypatch.setattr(compat, "components", fake_components)
    monkeypatch.setattr(
        compat.session_service,
        "remaining_questions",
        lambda _sid: 3,
    )
    monkeypatch.setattr(compat, "session_stats", lambda _sid: {"num_documents": 0})
    monkeypatch.setattr(compat, "_current_document", lambda _sid: None)

    compat.render_dashboard("sid")
    assert hasattr(fake_components, "_called")


def test_list_documents_falls_back_to_current_document(monkeypatch):
    doc = types.SimpleNamespace(
        document_id="abc",
        display_name="t.pdf",
        status="ready",
        num_pages=1,
        num_chunks=1,
    )
    monkeypatch.delattr(compat.session_service, "list_documents", raising=False)
    monkeypatch.setattr(compat, "_current_document", lambda _sid: doc)
    assert compat.list_documents("sid") == [doc]


def test_retrieve_legacy_single_doc_api(monkeypatch):
    """Old retrieval_service.retrieve(session_id, document_id: str, ...)."""
    doc_id = "a" * 32
    seen: list[str] = []

    def legacy_retrieve(session_id, document_id, query, top_k=4):
        seen.append(document_id)
        return [{"score": 0.9, "text": "hit", "page_start": 1, "page_end": 1}]

    monkeypatch.setattr(compat.retrieval_service, "retrieve", legacy_retrieve)
    hits = compat.retrieve("s" * 32, [doc_id], "test query", top_k=4)
    assert seen == [doc_id]
    assert len(hits) == 1
    assert hits[0]["text"] == "hit"


def test_retrieve_filters_invalid_ids():
    assert compat._normalize_document_ids(["__all__", "not-valid", ""]) == []
