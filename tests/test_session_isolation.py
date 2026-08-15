"""Cross-session isolation and cleanup tests (the most security-critical set)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from config import DEMO_STORAGE_ROOT
from services import (
    document_service,
    retrieval_service,
    security,
    session_service,
)
from tests.pdf_util import make_pdf


def _ingest(session_id: str):
    data = make_pdf(
        [
            "Murabaha is a cost plus profit sale contract used in Islamic finance.",
            "Sukuk are asset backed certificates. Ijarah is a leasing arrangement.",
        ]
    )
    return document_service.ingest(session_id, data, "doc.pdf")


def test_session_a_cannot_read_session_b_document():
    sid_a = security.new_id()
    sid_b = security.new_id()
    session_service.get_or_create(sid_a)
    session_service.get_or_create(sid_b)
    try:
        res_a = _ingest(sid_a)

        # B knows A's document_id (worst case) but must still be denied.
        assert session_service.get_document(sid_b, res_a.document_id) is None
        assert session_service.get_document(sid_a, res_a.document_id) is not None
    finally:
        session_service.destroy(sid_a)
        session_service.destroy(sid_b)


def test_session_b_cannot_retrieve_from_session_a_index():
    sid_a = security.new_id()
    sid_b = security.new_id()
    session_service.get_or_create(sid_a)
    session_service.get_or_create(sid_b)
    try:
        res_a = _ingest(sid_a)

        # A can retrieve from its own document.
        own = retrieval_service.retrieve(sid_a, res_a.document_id, "Murabaha", top_k=3)
        assert len(own) > 0

        # B attempts to query A's document_id — must get nothing.
        cross = retrieval_service.retrieve(sid_b, res_a.document_id, "Murabaha", top_k=3)
        assert cross == []
    finally:
        session_service.destroy(sid_a)
        session_service.destroy(sid_b)


def test_session_directories_are_separate():
    sid_a = security.new_id()
    sid_b = security.new_id()
    da = session_service.session_dir(sid_a)
    db = session_service.session_dir(sid_b)
    assert da != db
    assert security.is_within(DEMO_STORAGE_ROOT, da)
    assert security.is_within(DEMO_STORAGE_ROOT, db)


def test_replacing_document_deletes_previous(new_session):
    r1 = _ingest(new_session)
    p1 = session_service.pdf_path(new_session, r1.document_id)
    assert p1.exists()

    r2 = _ingest(new_session)  # replace
    assert r2.document_id != r1.document_id
    # Old PDF file must be gone.
    assert not p1.exists()
    # Only the new document is the current one.
    cur = session_service.current_document(new_session)
    assert cur is not None and cur.document_id == r2.document_id


def test_explicit_delete_removes_files(new_session):
    r = _ingest(new_session)
    sdir = session_service.session_dir(new_session)
    assert any(sdir.iterdir())
    document_service.delete_current(new_session)
    assert session_service.current_document(new_session) is None
    # Files removed from the session directory.
    assert not any(sdir.iterdir())


def test_ttl_cleanup_removes_expired_session():
    from services import cleanup_service

    sid = security.new_id()
    session_service.get_or_create(sid)
    _ingest(sid)
    sdir = session_service.session_dir(sid)
    assert sdir.exists()

    # Age the directory beyond the TTL (1 minute in tests).
    old = time.time() - (session_service.ttl_seconds() + 120)
    os.utime(sdir, (old, old))

    removed = cleanup_service.sweep(force=True)
    assert removed >= 1
    assert not sdir.exists()
