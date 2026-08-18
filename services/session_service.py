"""
web_demo/services/session_service.py
------------------------------------
Per-session isolation for the public demo (MULTI-DOCUMENT).

CRITICAL: every document/index/chat operation is keyed by a secret, random
``session_id``. A session's working files live only under::

    <DEMO_STORAGE_ROOT>/<session_id>/
        <document_id>.pdf
        <document_id>.faiss
        <document_id>.chunks.json

Because ``session_id`` is a cryptographically-random UUID that is kept only in
the requesting browser's Streamlit session state, one visitor can never name,
reach, or query another visitor's directory or in-memory record.

Like the desktop app, a session can hold MULTIPLE documents (bounded by
``MAX_FILES_PER_SESSION``). Each document has its own FAISS index and chunk
file, addressed strictly by a validated ``document_id`` — never by filename.

This module is pure Python (no Streamlit import) so isolation can be unit
tested directly.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    DEMO_STORAGE_ROOT,
    MAX_CASES_PER_SESSION,
    MAX_FILES_PER_SESSION,
    MAX_QUESTIONS_PER_SESSION,
    MAX_UPLOADS_PER_SESSION,
    MAX_URL_SOURCES_PER_SESSION,
    MAX_URLS_PER_SESSION,
    SESSION_TTL_MINUTES,
)
from core.source_models import (
    SOURCE_TYPE_PDF,
    SOURCE_TYPE_URL,
    domain_of,
    normalize_source_type,
)
from services import security

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_NEEDS_OCR = "needs_ocr"
STATUS_ERROR = "error"


@dataclass
class DocumentRecord:
    """One source in a session: an uploaded PDF or a fetched web page.

    The name is historical — PDFs were the only source when it was written, and
    the whole application addresses sources by ``document_id``. Rather than
    rename that concept everywhere (and break every caller), URL sources reuse
    this record with ``source_type="url"`` and the web fields filled in. A
    record that says nothing about its type is a PDF, exactly as before, so
    every existing caller keeps its meaning.
    """

    document_id: str
    display_name: str
    num_pages: int = 0
    num_chunks: int = 0
    status: str = STATUS_PROCESSING  # processing | ready | needs_ocr | error
    file_hash: str = ""
    created_at: float = field(default_factory=time.time)
    # --- Source kind + web provenance (empty/default for PDFs) -------------
    source_type: str = SOURCE_TYPE_PDF
    original_url: str = ""
    final_url: str = ""
    page_title: str = ""
    content_type: str = ""
    retrieved_at: float = 0.0

    @property
    def source_id(self) -> str:
        """Unified name for ``document_id`` (they are the same identifier)."""
        return self.document_id

    @property
    def is_url(self) -> bool:
        return normalize_source_type(self.source_type) == SOURCE_TYPE_URL

    @property
    def url(self) -> str:
        """The address a citation should link to (final beats original)."""
        return self.final_url or self.original_url

    @property
    def domain(self) -> str:
        return domain_of(self.url)


@dataclass
class SessionRecord:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    uploads: int = 0
    questions: int = 0
    # Case analyses are a separate, much more expensive operation, so they are
    # metered on their own counter instead of draining the question quota.
    cases: int = 0
    # Outbound page fetches (adds + refreshes), metered separately from uploads
    # because each one costs the SERVER a network request.
    url_fetches: int = 0
    documents: dict[str, DocumentRecord] = field(default_factory=dict)


_sessions: dict[str, SessionRecord] = {}
_lock = threading.RLock()


# --- Lifecycle ------------------------------------------------------------
def get_or_create(session_id: str) -> SessionRecord:
    """Return the record for ``session_id``, creating dir + record if needed."""
    security.require_valid_id(session_id)
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            rec = SessionRecord(session_id=session_id)
            _sessions[session_id] = rec
        rec.last_active = time.time()
    session_dir(session_id).mkdir(parents=True, exist_ok=True)
    return rec


def touch(session_id: str) -> None:
    with _lock:
        rec = _sessions.get(session_id)
        if rec is not None:
            rec.last_active = time.time()


def get(session_id: str) -> Optional[SessionRecord]:
    with _lock:
        return _sessions.get(session_id)


# --- Paths (always scoped + containment-checked) --------------------------
def session_dir(session_id: str) -> Path:
    security.require_valid_id(session_id)
    return security.safe_child_path(DEMO_STORAGE_ROOT, session_id)


def pdf_path(session_id: str, document_id: str) -> Path:
    security.require_valid_id(document_id)
    return security.safe_child_path(session_dir(session_id), f"{document_id}.pdf")


def index_path(session_id: str, document_id: str) -> Path:
    security.require_valid_id(document_id)
    return security.safe_child_path(session_dir(session_id), f"{document_id}.faiss")


def chunks_path(session_id: str, document_id: str) -> Path:
    security.require_valid_id(document_id)
    return security.safe_child_path(
        session_dir(session_id), f"{document_id}.chunks.json"
    )


# --- Document records -----------------------------------------------------
def add_document(session_id: str, record: DocumentRecord) -> None:
    """Register (or replace) a document record within this session."""
    with _lock:
        rec = get_or_create(session_id)
        rec.documents[record.document_id] = record
        rec.last_active = time.time()


# Backwards-compatible alias.
set_document = add_document


def get_document(session_id: str, document_id: str) -> Optional[DocumentRecord]:
    """Return the document ONLY if it belongs to this exact session."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return None
        return rec.documents.get(document_id)


def has_document(session_id: str, document_id: str) -> bool:
    return get_document(session_id, document_id) is not None


def list_sources(session_id: str) -> list[DocumentRecord]:
    """Every source in the session — PDFs and URLs — newest first."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return []
        sources = list(rec.documents.values())
    sources.sort(key=lambda d: d.created_at, reverse=True)
    return sources


def list_documents(session_id: str) -> list[DocumentRecord]:
    """Uploaded PDF documents only, newest first (any status).

    Kept PDF-only on purpose: the upload cap, the page counters, and the
    upload-side callers all mean "files", and URL sources must not silently
    consume a document slot. Use :func:`list_sources` for the unified view.
    """
    return [d for d in list_sources(session_id) if not d.is_url]


def list_url_sources(session_id: str) -> list[DocumentRecord]:
    """Web page sources only, newest first (any status)."""
    return [d for d in list_sources(session_id) if d.is_url]


def ready_sources(session_id: str) -> list[DocumentRecord]:
    """Every source that finished indexing (usable in chat/search/case)."""
    return [d for d in list_sources(session_id) if d.status == STATUS_READY]


def ready_documents(session_id: str) -> list[DocumentRecord]:
    """Only PDF documents that finished indexing."""
    return [d for d in ready_sources(session_id) if not d.is_url]


def ready_url_sources(session_id: str) -> list[DocumentRecord]:
    """Only URL sources that finished indexing."""
    return [d for d in ready_sources(session_id) if d.is_url]


def get_source(session_id: str, source_id: str) -> Optional[DocumentRecord]:
    """Unified alias for :func:`get_document` (same ownership guarantee)."""
    return get_document(session_id, source_id)


def find_by_hash(
    session_id: str, file_hash: str, source_type: Optional[str] = None
) -> Optional[DocumentRecord]:
    """Find a source by its content/URL hash, optionally within one kind."""
    if not file_hash:
        return None
    wanted = normalize_source_type(source_type) if source_type else None
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return None
        for doc in rec.documents.values():
            if not doc.file_hash or doc.file_hash != file_hash:
                continue
            if wanted is not None and normalize_source_type(doc.source_type) != wanted:
                continue
            return doc
    return None


def find_url_source(session_id: str, url_hash: str) -> Optional[DocumentRecord]:
    """Find an existing URL source by its canonical-URL hash."""
    return find_by_hash(session_id, url_hash, source_type=SOURCE_TYPE_URL)


def remove_document(session_id: str, document_id: str) -> None:
    """Drop a source's record and delete its files from disk."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is not None:
            rec.documents.pop(document_id, None)
    _delete_document_files(session_id, document_id)


# Unified alias — deletion is identical for both source kinds.
remove_source = remove_document


def _delete_document_files(session_id: str, document_id: str) -> None:
    if not security.is_valid_id(document_id):
        return
    for path in (
        pdf_path(session_id, document_id),
        index_path(session_id, document_id),
        chunks_path(session_id, document_id),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# --- Stats (session-only; never global) -----------------------------------
def stats(session_id: str) -> dict:
    ready = ready_sources(session_id)
    pdfs = [d for d in ready if not d.is_url]
    urls = [d for d in ready if d.is_url]
    return {
        # Unchanged meaning: uploaded files only.
        "num_documents": len(pdfs),
        "num_urls": len(urls),
        "num_sources": len(ready),
        "total_pages": sum(d.num_pages for d in pdfs),
        "total_chunks": sum(d.num_chunks for d in ready),
    }


# --- Rate / quota gates ---------------------------------------------------
def can_upload(session_id: str) -> bool:
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return True
        return rec.uploads < MAX_UPLOADS_PER_SESSION


def record_upload(session_id: str) -> None:
    with _lock:
        rec = get_or_create(session_id)
        rec.uploads += 1


def live_document_count(session_id: str) -> int:
    """Concurrent uploaded PDFs. URL sources have their own, separate cap."""
    return len(list_documents(session_id))


def has_document_slot(session_id: str) -> bool:
    """Enforce MAX_FILES_PER_SESSION (concurrent live documents)."""
    return live_document_count(session_id) < MAX_FILES_PER_SESSION


# --- URL source quotas (separate from the upload quotas) ------------------
def live_url_count(session_id: str) -> int:
    return len(list_url_sources(session_id))


def has_url_slot(session_id: str) -> bool:
    """Enforce MAX_URL_SOURCES_PER_SESSION (concurrent live URL sources)."""
    return live_url_count(session_id) < MAX_URL_SOURCES_PER_SESSION


def can_fetch_url(session_id: str) -> bool:
    """Whether another outbound page fetch (add or refresh) is allowed."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return True
        return rec.url_fetches < MAX_URLS_PER_SESSION


def record_url_fetch(session_id: str) -> None:
    """Charge one outbound fetch. Called before the request goes out."""
    with _lock:
        rec = get_or_create(session_id)
        rec.url_fetches += 1


def remaining_url_fetches(session_id: str) -> int:
    with _lock:
        rec = _sessions.get(session_id)
        used = rec.url_fetches if rec else 0
        return max(0, MAX_URLS_PER_SESSION - used)


def remaining_url_slots(session_id: str) -> int:
    return max(0, MAX_URL_SOURCES_PER_SESSION - live_url_count(session_id))


def can_ask(session_id: str) -> bool:
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return True
        return rec.questions < MAX_QUESTIONS_PER_SESSION


def record_question(session_id: str) -> None:
    with _lock:
        rec = get_or_create(session_id)
        rec.questions += 1


def remaining_questions(session_id: str) -> int:
    with _lock:
        rec = _sessions.get(session_id)
        used = rec.questions if rec else 0
        return max(0, MAX_QUESTIONS_PER_SESSION - used)


# --- Case analysis quota (separate from the question quota) ---------------
def can_analyze_case(session_id: str) -> bool:
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return True
        return rec.cases < MAX_CASES_PER_SESSION


def record_case(session_id: str) -> None:
    """Charge one case operation. Called ONLY after a full report succeeded."""
    with _lock:
        rec = get_or_create(session_id)
        rec.cases += 1


def remaining_cases(session_id: str) -> int:
    with _lock:
        rec = _sessions.get(session_id)
        used = rec.cases if rec else 0
        return max(0, MAX_CASES_PER_SESSION - used)


# --- Teardown -------------------------------------------------------------
def destroy(session_id: str) -> None:
    """Remove a session's in-memory record and its entire directory."""
    with _lock:
        _sessions.pop(session_id, None)
    d = DEMO_STORAGE_ROOT / session_id
    if security.is_valid_id(session_id) and d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _drop_record(session_id: str) -> None:
    """Drop only the in-memory record (used by cleanup after files removed)."""
    with _lock:
        _sessions.pop(session_id, None)


def active_session_ids() -> list[str]:
    with _lock:
        return list(_sessions.keys())


def ttl_seconds() -> int:
    return SESSION_TTL_MINUTES * 60
