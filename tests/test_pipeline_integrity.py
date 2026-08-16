"""End-to-end integrity of the demo pipeline.

Covers the failure that shipped to production: a document reported "ready"
while its index was unreadable, and retrieval reported that as "no relevant
information" instead of an infrastructure fault.
"""

from __future__ import annotations

import json

import pytest

from services import (
    chat_service,
    document_service,
    llm_service,
    retrieval_service,
    security,
    session_service,
)
from tests.pdf_util import make_pdf

EC2_PAGES = [
    "To launch an EC2 instance, open the AWS console and choose an AMI.",
    "Configure the security group to allow inbound SSH traffic on port 22.",
]
FINANCE_PAGES = [
    "Ijarah is a leasing arrangement where the lessor retains ownership.",
]


def _ingest(session_id, pages=None, name="guide.pdf"):
    return document_service.ingest(session_id, make_pdf(pages or EC2_PAGES), name)


def _spy_on_retrieval(monkeypatch) -> dict:
    """Record which retrieval path ran, while still running the real one."""
    used = {"ctx": False, "faiss": False}
    real_ctx = retrieval_service.document_context
    real_retrieve = retrieval_service.retrieve

    def spy_ctx(*args, **kwargs):
        used["ctx"] = True
        return real_ctx(*args, **kwargs)

    def spy_retrieve(*args, **kwargs):
        used["faiss"] = True
        return real_retrieve(*args, **kwargs)

    monkeypatch.setattr(retrieval_service, "document_context", spy_ctx)
    monkeypatch.setattr(retrieval_service, "retrieve", spy_retrieve)
    return used


# --- 1. Canonical persistence --------------------------------------------
def test_ingest_writes_canonical_index_files(new_session):
    res = _ingest(new_session)
    index_file, chunks_file = retrieval_service.canonical_paths(
        new_session, res.document_id
    )
    assert index_file.exists(), "FAISS index missing at the canonical path"
    assert chunks_file.exists(), "chunks JSON missing at the canonical path"
    assert index_file.name == f"{res.document_id}.faiss"
    assert chunks_file.name == f"{res.document_id}.chunks.json"


def test_writer_and_reader_agree_on_paths(new_session):
    """The exact files the writer produced are the ones the reader loads."""
    res = _ingest(new_session)
    index_file, chunks_file = retrieval_service.canonical_paths(
        new_session, res.document_id
    )
    assert (index_file, chunks_file) == (
        session_service.index_path(new_session, res.document_id),
        session_service.chunks_path(new_session, res.document_id),
    )


# --- 2. Cold reload ------------------------------------------------------
def test_retrieval_works_after_cache_drop(new_session):
    """Simulates a fresh process: nothing cached, everything read from disk."""
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)  # drop all in-memory state

    hits = retrieval_service.retrieve(new_session, [res.document_id], "security group", top_k=3)
    assert hits, "retrieval failed after a cold cache"
    assert hits[0]["document_name"] == "guide.pdf"


# --- 3. Multi-document ---------------------------------------------------
def test_multi_document_retrieval_and_selection(new_session):
    r1 = _ingest(new_session, EC2_PAGES, "aws.pdf")
    r2 = _ingest(new_session, FINANCE_PAGES, "ijarah.pdf")

    both = retrieval_service.retrieve(
        new_session, [r1.document_id, r2.document_id], "leasing ownership", top_k=4
    )
    assert {h["document_name"] for h in both} <= {"aws.pdf", "ijarah.pdf"}
    assert both[0]["document_name"] == "ijarah.pdf"

    # Selecting one document must exclude the other entirely.
    only_aws = retrieval_service.retrieve(
        new_session, [r1.document_id], "leasing ownership", top_k=4
    )
    assert only_aws and all(h["document_name"] == "aws.pdf" for h in only_aws)


# --- 4/5/6. Index faults are faults, not empty knowledge -----------------
def test_missing_index_file_raises_index_unavailable(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_service.index_path(new_session, res.document_id).unlink()

    with pytest.raises(retrieval_service.IndexUnavailable) as exc:
        retrieval_service.retrieve(new_session, [res.document_id], "ssh", top_k=3)
    assert exc.value.reason == "index_missing"


def test_missing_chunks_file_raises_index_unavailable(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_service.chunks_path(new_session, res.document_id).unlink()

    with pytest.raises(retrieval_service.IndexUnavailable):
        retrieval_service.retrieve(new_session, [res.document_id], "ssh", top_k=3)


def test_corrupt_index_raises_index_unavailable(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_service.index_path(new_session, res.document_id).write_bytes(b"not-faiss")

    with pytest.raises(retrieval_service.IndexUnavailable) as exc:
        retrieval_service.retrieve(new_session, [res.document_id], "ssh", top_k=3)
    assert exc.value.reason == "index_unreadable"


def test_index_chunks_mismatch_detected(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    path = session_service.chunks_path(new_session, res.document_id)
    chunks = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(chunks * 3, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(retrieval_service.IndexUnavailable) as exc:
        retrieval_service.retrieve(new_session, [res.document_id], "ssh", top_k=3)
    assert exc.value.reason == "index_chunks_mismatch"


# --- Post-ingest verification -------------------------------------------
def test_ingest_rejects_document_when_verification_fails(new_session, monkeypatch):
    """A document must never reach 'ready' if its index is not loadable."""
    monkeypatch.setattr(
        retrieval_service,
        "verify_document_index",
        lambda *a, **k: (_ for _ in ()).throw(
            retrieval_service.IndexUnavailable("index_missing")
        ),
    )
    with pytest.raises(security.UploadRejected) as exc:
        _ingest(new_session)
    assert "التحقق من فهرس" in str(exc.value)
    # No misleading metadata is left behind.
    assert session_service.list_documents(new_session) == []


def test_verification_runs_through_the_real_read_path(new_session):
    res = _ingest(new_session)
    info = retrieval_service.verify_document_index(new_session, res.document_id)
    assert info["num_vectors"] == info["num_chunks"] > 0


# --- Legacy layout (read-only compatibility) -----------------------------
def test_legacy_single_document_layout_is_readable(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_dir = session_service.session_dir(new_session)

    # Rename canonical files to the original single-document layout.
    session_service.index_path(new_session, res.document_id).rename(
        session_dir / retrieval_service.LEGACY_INDEX_NAME
    )
    session_service.chunks_path(new_session, res.document_id).rename(
        session_dir / retrieval_service.LEGACY_CHUNKS_NAME
    )

    hits = retrieval_service.retrieve(new_session, [res.document_id], "ssh", top_k=3)
    assert hits, "legacy layout should still be readable"


# --- Diagnostics ---------------------------------------------------------
def test_document_diagnostics_are_healthy_and_content_free(new_session):
    res = _ingest(new_session)
    diags = document_service.diagnostics(new_session)
    assert len(diags) == 1
    d = diags[0]

    assert d["document_id"] == res.document_id
    assert d["status"] == "ready"
    assert d["num_pages"] == 2
    assert d["index_exists"] and d["chunks_file_exists"]
    assert d["index_loadable"] and d["chunks_loadable"]
    assert d["num_vectors"] == d["num_indexed_chunks"] > 0

    blob = json.dumps(diags, ensure_ascii=False)
    for secret in ["EC2", "security group", "AMI", "SSH"]:
        assert secret not in blob, "diagnostics must not contain document text"


def test_diagnostics_report_broken_index(new_session):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_service.index_path(new_session, res.document_id).unlink()

    d = document_service.diagnostics(new_session)[0]
    assert d["index_loadable"] is False
    assert d["reason"] == "index_missing"


# --- Citations -----------------------------------------------------------
def test_results_carry_correct_page_citations(new_session):
    res = _ingest(new_session)
    hits = retrieval_service.retrieve(new_session, [res.document_id], "port 22", top_k=4)
    assert hits
    for h in hits:
        assert h["document_name"] == "guide.pdf"
        assert 1 <= h["page_start"] <= h["page_end"] <= res.num_pages
    # Internal identifiers never leak into results.
    assert "document_id" not in hits[0] and "chunk_id" not in hits[0]


def test_overview_context_is_page_ordered_with_citations(new_session):
    res = _ingest(new_session)
    ctx = retrieval_service.document_context(new_session, [res.document_id])
    assert ctx
    pages = [c["page_start"] for c in ctx]
    assert pages == sorted(pages), "overview context must follow page order"
    assert all(c["document_name"] == "guide.pdf" for c in ctx)


def test_overview_context_respects_char_budget(new_session):
    res = _ingest(new_session)
    ctx = retrieval_service.document_context(new_session, [res.document_id], max_chars=120)
    assert sum(len(c["text"]) for c in ctx) <= 120


def test_overview_context_only_covers_selected_documents(new_session):
    r1 = _ingest(new_session, EC2_PAGES, "aws.pdf")
    _ingest(new_session, FINANCE_PAGES, "ijarah.pdf")
    ctx = retrieval_service.document_context(new_session, [r1.document_id])
    assert ctx and all(c["document_name"] == "aws.pdf" for c in ctx)


# --- Chat pipeline contract ---------------------------------------------
def test_chat_index_failure_does_not_call_llm_or_spend_quota(new_session, monkeypatch):
    res = _ingest(new_session)
    retrieval_service.invalidate(new_session)
    session_service.index_path(new_session, res.document_id).unlink()

    called = []
    monkeypatch.setattr(
        llm_service, "answer", lambda *a, **k: called.append(1)
    )
    before = session_service.remaining_questions(new_session)

    outcome = chat_service.respond(new_session, "what is the ssh port?", [res.document_id])

    assert outcome.kind == chat_service.KIND_INDEX_ERROR
    assert outcome.text == chat_service.INDEX_ERROR_MESSAGE
    assert outcome.text != chat_service.NO_CONTENT_MESSAGE
    assert outcome.llm_called is False
    assert called == [], "Groq must not be called when the index is unreadable"
    assert session_service.remaining_questions(new_session) == before


def test_chat_success_calls_llm_and_spends_one_question(new_session, monkeypatch):
    res = _ingest(new_session)
    seen = {}

    def fake_answer(session_id, question, results, mode="factual"):
        seen["mode"] = mode
        seen["results"] = results
        return llm_service.LLMResult(ok=True, text="إجابة تجريبية (صفحة 2)")

    monkeypatch.setattr(llm_service, "answer", fake_answer)
    before = session_service.remaining_questions(new_session)

    outcome = chat_service.respond(new_session, "which port is used for ssh?", [res.document_id])

    assert outcome.kind == chat_service.KIND_ANSWER
    assert outcome.llm_called is True
    assert outcome.mode == "factual"
    assert seen["results"], "context must be handed to the model"
    assert session_service.remaining_questions(new_session) == before - 1


def test_chat_overview_question_uses_document_context(new_session, monkeypatch):
    res = _ingest(new_session)
    monkeypatch.setattr(
        llm_service,
        "answer",
        lambda *a, **k: llm_service.LLMResult(ok=True, text="ملخص"),
    )

    used = _spy_on_retrieval(monkeypatch)

    outcome = chat_service.respond(new_session, "what is inside the pdf", [res.document_id])
    assert outcome.mode == "overview"
    assert used["ctx"] and not used["faiss"]
    assert outcome.sources, "overview must still provide cited context"


def test_chat_factual_question_uses_faiss(new_session, monkeypatch):
    res = _ingest(new_session)
    monkeypatch.setattr(
        llm_service,
        "answer",
        lambda *a, **k: llm_service.LLMResult(ok=True, text="جواب"),
    )
    used = _spy_on_retrieval(monkeypatch)

    outcome = chat_service.respond(new_session, "which port allows inbound SSH?", [res.document_id])
    assert outcome.mode == "factual"
    assert used["faiss"] and not used["ctx"]


def test_chat_respects_document_selection(new_session, monkeypatch):
    r1 = _ingest(new_session, EC2_PAGES, "aws.pdf")
    _ingest(new_session, FINANCE_PAGES, "ijarah.pdf")
    monkeypatch.setattr(
        llm_service, "answer", lambda *a, **k: llm_service.LLMResult(ok=True, text="ok")
    )

    outcome = chat_service.respond(new_session, "leasing ownership", [r1.document_id])
    assert outcome.sources
    assert all(s["document_name"] == "aws.pdf" for s in outcome.sources)


def test_chat_without_selection_never_calls_llm(new_session, monkeypatch):
    called = []
    monkeypatch.setattr(llm_service, "answer", lambda *a, **k: called.append(1))
    outcome = chat_service.respond(new_session, "anything", [])
    assert outcome.kind == chat_service.KIND_NO_SELECTION
    assert called == []


def test_chat_diagnostics_are_counts_only(new_session, monkeypatch):
    res = _ingest(new_session)
    monkeypatch.setattr(
        llm_service, "answer", lambda *a, **k: llm_service.LLMResult(ok=True, text="ok")
    )
    outcome = chat_service.respond(new_session, "ssh port", [res.document_id])
    assert set(outcome.diagnostics) == {
        "selected_document_count",
        "valid_document_count",
        "retrieved_results_count",
        "error",
    }
    assert outcome.diagnostics["valid_document_count"] == 1
    assert outcome.diagnostics["retrieved_results_count"] >= 1


def test_groq_auth_failure_is_distinct_from_no_content(new_session, monkeypatch):
    res = _ingest(new_session)
    monkeypatch.setattr(
        llm_service,
        "answer",
        lambda *a, **k: llm_service.LLMResult(
            ok=False, error_category=llm_service.ERR_AUTH
        ),
    )
    outcome = chat_service.respond(new_session, "ssh port", [res.document_id])

    assert outcome.kind == chat_service.KIND_ANSWER  # retrieval succeeded
    assert outcome.text != chat_service.NO_CONTENT_MESSAGE
    assert outcome.text != chat_service.INDEX_ERROR_MESSAGE
    assert "مفتاح الخدمة" in outcome.text
