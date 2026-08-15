"""Upload validation, filename safety, and path-containment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import DEMO_STORAGE_ROOT
from services import security, session_service
from tests.pdf_util import make_pdf


def test_valid_pdf_accepted():
    data = make_pdf(["Hello world", "Second page"])
    # Should not raise.
    security.validate_upload(data, "sample.pdf")


def test_non_pdf_rejected_by_extension():
    with pytest.raises(security.UploadRejected):
        security.validate_upload(b"not a pdf", "malware.exe")


def test_non_pdf_rejected_by_signature():
    with pytest.raises(security.UploadRejected):
        security.validate_upload(b"%ZIP fake content", "fake.pdf")


def test_zero_byte_rejected():
    with pytest.raises(security.UploadRejected):
        security.validate_upload(b"", "empty.pdf")


def test_oversized_rejected():
    # Limit is 2 MB in tests; build a >2 MB payload with a valid header.
    big = b"%PDF-1.4\n" + b"0" * (3 * 1024 * 1024)
    with pytest.raises(security.UploadRejected):
        security.validate_upload(big, "big.pdf")


def test_encrypted_pdf_rejected(monkeypatch):
    import io

    class _FakeEncrypted:
        def __enter__(self):
            raise Exception("file has not been decrypted (password required)")

        def __exit__(self, *a):
            return False

    def fake_open(_stream):
        return _FakeEncrypted()

    monkeypatch.setattr(security.pdfplumber, "open", lambda s: fake_open(s))
    data = b"%PDF-1.4\n" + b"encrypted body"
    with pytest.raises(security.UploadRejected) as exc:
        security.validate_upload(data, "locked.pdf")
    assert "كلمة مرور" in str(exc.value)


def test_max_pages_enforced():
    # Test limit is 5 pages; 6 pages must be rejected.
    data = make_pdf([f"page {i}" for i in range(6)])
    with pytest.raises(security.UploadRejected) as exc:
        security.validate_upload(data, "long.pdf")
    assert "صفحات" in str(exc.value) or "صفحة" in str(exc.value)


def test_unsafe_filename_cannot_escape_session_dir(new_session):
    base = session_service.session_dir(new_session)
    # A traversal attempt must never resolve outside the session directory.
    with pytest.raises(security.UploadRejected):
        security.safe_child_path(base, "..", "..", "etc", "passwd")


def test_safe_display_filename_strips_paths():
    assert security.safe_display_filename("../../etc/passwd") == "passwd.pdf"
    assert security.safe_display_filename("a/b/c/report.pdf") == "report.pdf"
    assert security.safe_display_filename(None) == "document.pdf"
    # Forbidden characters are replaced, extension enforced.
    out = security.safe_display_filename('weird:name*?.pdf')
    assert "/" not in out and ":" not in out and out.endswith(".pdf")


def test_document_id_addressing_not_filename(new_session):
    # pdf_path uses a validated document_id, never the user filename.
    doc_id = security.new_id()
    p = session_service.pdf_path(new_session, doc_id)
    assert p.name == f"{doc_id}.pdf"
    assert security.is_within(DEMO_STORAGE_ROOT, p)


def test_invalid_ids_rejected():
    assert not security.is_valid_id("../evil")
    assert not security.is_valid_id("")
    assert not security.is_valid_id("g" * 32)
    assert security.is_valid_id(security.new_id())
