"""FAISS retrieval + bounded RAG context tests."""

from __future__ import annotations

from services import document_service, llm_service, retrieval_service, security
from tests.pdf_util import make_pdf


def test_faiss_retrieval_returns_relevant_chunk(new_session):
    data = make_pdf(
        [
            "Murabaha is a cost-plus-profit sale contract in Islamic finance.",
            "Ijarah is a leasing arrangement. Sukuk are asset-backed certificates.",
        ]
    )
    res = document_service.ingest(new_session, data, "finance.pdf")
    hits = retrieval_service.retrieve(new_session, res.document_id, "leasing", top_k=2)
    assert len(hits) >= 1
    top = hits[0]
    assert set(["score", "page_start", "page_end", "text", "document_name"]).issubset(top)
    # Excerpt is present; internal ids are not exposed in results.
    assert "chunk_id" not in top
    assert "document_id" not in top


def test_retrieve_unknown_document_returns_empty(new_session):
    unknown = security.new_id()
    assert retrieval_service.retrieve(new_session, unknown, "anything") == []


def test_rag_context_length_bounded():
    huge = "x" * 100_000
    results = [{"text": huge, "page_start": 1, "page_end": 1}]
    ctx = llm_service.build_context(results, max_chars=200)
    assert len(ctx) <= 200


def test_rag_context_multiple_chunks_bounded():
    results = [
        {"text": "a" * 400, "page_start": 1, "page_end": 1},
        {"text": "b" * 400, "page_start": 2, "page_end": 2},
        {"text": "c" * 400, "page_start": 3, "page_end": 3},
    ]
    max_chars = 500
    ctx = llm_service.build_context(results, max_chars=max_chars)
    # Allow for join separators between retained parts.
    assert len(ctx) <= max_chars + 4 * len("\n\n---\n\n")
