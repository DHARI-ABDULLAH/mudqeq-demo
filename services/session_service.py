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
    SESSION_TTL_MINUTES,
)
from services import security

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_NEEDS_OCR = "needs_ocr"
STATUS_ERROR = "error"


@dataclass
class DocumentRecord:
    document_id: str
    display_name: str
    num_pages: int = 0
    num_chunks: int = 0
    status: str = STATUS_PROCESSING  # processing | ready | needs_ocr | error
    file_hash: str = ""
    created_at: float = field(default_factory=time.time)


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


def list_documents(session_id: str) -> list[DocumentRecord]:
    """All documents in the session, newest first (any status)."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return []
        docs = list(rec.documents.values())
    docs.sort(key=lambda d: d.created_at, reverse=True)
    return docs


def ready_documents(session_id: str) -> list[DocumentRecord]:
    """Only documents that finished indexing (usable in chat/search)."""
    return [d for d in list_documents(session_id) if d.status == STATUS_READY]


def find_by_hash(session_id: str, file_hash: str) -> Optional[DocumentRecord]:
    if not file_hash:
        return None
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None:
            return None
        for doc in rec.documents.values():
            if doc.file_hash and doc.file_hash == file_hash:
                return doc
    return None


def remove_document(session_id: str, document_id: str) -> None:
    """Drop a document's record and delete its files from disk."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is not None:
            rec.documents.pop(document_id, None)
    _delete_document_files(session_id, document_id)


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
    ready = ready_documents(session_id)
    return {
        "num_documents": len(ready),
        "total_pages": sum(d.num_pages for d in ready),
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
    with _lock:
        rec = _sessions.get(session_id)
        return len(rec.documents) if rec else 0


def has_document_slot(session_id: str) -> bool:
    """Enforce MAX_FILES_PER_SESSION (concurrent live documents)."""
    return live_document_count(session_id) < MAX_FILES_PER_SESSION


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
