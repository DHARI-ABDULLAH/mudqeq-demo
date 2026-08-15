"""
web_demo/services/session_service.py
------------------------------------
Per-session isolation for the public demo.

CRITICAL: every document/index/chat operation is keyed by a secret, random
``session_id``. A session's working files live only under::

    <DEMO_STORAGE_ROOT>/<session_id>/

Because ``session_id`` is a cryptographically-random UUID that is kept only in
the requesting browser's Streamlit session state, one visitor can never name,
reach, or query another visitor's directory or in-memory record.

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
    MAX_FILES_PER_SESSION,
    MAX_QUESTIONS_PER_SESSION,
    MAX_UPLOADS_PER_SESSION,
    SESSION_TTL_MINUTES,
)
from services import security

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


@dataclass
class DocumentRecord:
    document_id: str
    display_name: str
    num_pages: int = 0
    num_chunks: int = 0
    status: str = "processing"  # processing | ready | needs_ocr | error


@dataclass
class SessionRecord:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    uploads: int = 0
    questions: int = 0
    document: Optional[DocumentRecord] = None


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


def index_path(session_id: str) -> Path:
    return security.safe_child_path(session_dir(session_id), INDEX_FILE)


def chunks_path(session_id: str) -> Path:
    return security.safe_child_path(session_dir(session_id), CHUNKS_FILE)


def pdf_path(session_id: str, document_id: str) -> Path:
    security.require_valid_id(document_id)
    return security.safe_child_path(session_dir(session_id), f"{document_id}.pdf")


# --- Document record ------------------------------------------------------
def set_document(session_id: str, record: DocumentRecord) -> None:
    with _lock:
        rec = get_or_create(session_id)
        rec.document = record
        rec.last_active = time.time()


def get_document(session_id: str, document_id: str) -> Optional[DocumentRecord]:
    """Return the document ONLY if it belongs to this exact session."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is None or rec.document is None:
            return None
        if rec.document.document_id != document_id:
            return None
        return rec.document


def current_document(session_id: str) -> Optional[DocumentRecord]:
    with _lock:
        rec = _sessions.get(session_id)
        return rec.document if rec else None


def clear_document(session_id: str) -> None:
    """Delete the current document's files from disk and drop the record."""
    with _lock:
        rec = _sessions.get(session_id)
        if rec is not None:
            rec.document = None
    _delete_document_files(session_id)


def _delete_document_files(session_id: str) -> None:
    d = session_dir(session_id)
    if not d.exists():
        return
    for child in d.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass


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


def has_document_slot(session_id: str) -> bool:
    """Enforce MAX_FILES_PER_SESSION (concurrent live documents)."""
    with _lock:
        rec = _sessions.get(session_id)
        live = 1 if (rec and rec.document is not None) else 0
        return live < MAX_FILES_PER_SESSION


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
