"""
web_demo/services/document_service.py
-------------------------------------
Orchestrates the demo ingest pipeline for a single session:

    validate (untrusted bytes)
      -> store in session dir (id-addressed, never by filename)
      -> extract (generic, pdfplumber)
      -> chunk (page-tagged)
      -> embed + FAISS (per session)
      -> ready

Enforces per-session upload quotas and the single-document slot. Replacing a
document deletes the previous one first. Never logs document content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from config import MAX_PAGES
from core import chunking, extraction
from core.logging_utils import log_event
from services import retrieval_service, security, session_service

STATUS_READY = "ready"
STATUS_NEEDS_OCR = "needs_ocr"
STATUS_ERROR = "error"


@dataclass
class IngestResult:
    document_id: str
    display_name: str
    status: str
    num_pages: int
    num_chunks: int
    message: str


def ingest(session_id: str, file_bytes: bytes, original_name: str | None) -> IngestResult:
    """Validate + process an uploaded PDF for the given session.

    Raises security.UploadRejected (Arabic message) for user-facing rejects.
    """
    security.require_valid_id(session_id)
    session_service.get_or_create(session_id)
    started = time.time()

    if not session_service.can_upload(session_id):
        raise security.UploadRejected(
            "تم الوصول إلى حد عدد مرات الرفع في هذه الجلسة التجريبية."
        )

    # Strict validation of untrusted input BEFORE writing anything to disk.
    security.validate_upload(file_bytes, original_name)

    display_name = security.safe_display_filename(original_name)

    # Replacing a document: remove the previous one first (single slot).
    session_service.clear_document(session_id)
    retrieval_service.invalidate(session_id)

    document_id = security.new_id()
    dest = session_service.pdf_path(session_id, document_id)
    with open(dest, "wb") as f:
        f.write(file_bytes)

    session_service.record_upload(session_id)

    try:
        extract_result = extraction.extract_pdf(dest)
    except Exception as exc:  # noqa: BLE001
        _fail(session_id, document_id, display_name)
        log_event("ingest", session_id, status="extract_error",
                  error_category=type(exc).__name__)
        raise security.UploadRejected("تعذّر معالجة المستند.") from exc

    num_pages = extract_result.num_pages

    if not extract_result.has_usable_text():
        _fail(session_id, document_id, display_name, status=STATUS_NEEDS_OCR)
        log_event("ingest", session_id, status="needs_ocr", pages=num_pages)
        raise security.UploadRejected(
            "لم يتم العثور على نص قابل للاستخراج من المستند "
            "(قد يكون ممسوحاً ضوئياً)."
        )

    chunks = chunking.build_chunks(
        extract_result.non_empty_pages(), document_id, display_name
    )
    if not chunks:
        _fail(session_id, document_id, display_name, status=STATUS_NEEDS_OCR)
        log_event("ingest", session_id, status="no_chunks", pages=num_pages)
        raise security.UploadRejected("تعذّر إنشاء مقاطع قابلة للفهرسة من المستند.")

    try:
        retrieval_service.build_and_store(session_id, document_id, chunks)
    except Exception as exc:  # noqa: BLE001
        _fail(session_id, document_id, display_name)
        log_event("ingest", session_id, status="index_error",
                  error_category=type(exc).__name__)
        raise security.UploadRejected("تعذّر فهرسة المستند.") from exc

    record = session_service.DocumentRecord(
        document_id=document_id,
        display_name=display_name,
        num_pages=num_pages,
        num_chunks=len(chunks),
        status=STATUS_READY,
    )
    session_service.set_document(session_id, record)

    log_event(
        "ingest",
        session_id,
        status="ready",
        pages=num_pages,
        chunks=len(chunks),
        duration_ms=int((time.time() - started) * 1000),
    )
    return IngestResult(
        document_id=document_id,
        display_name=display_name,
        status=STATUS_READY,
        num_pages=num_pages,
        num_chunks=len(chunks),
        message="تم تجهيز المستند بنجاح.",
    )


def _fail(session_id: str, document_id: str, display_name: str,
          status: str = STATUS_ERROR) -> None:
    """Best-effort cleanup of a failed ingest; keeps status for the UI."""
    try:
        session_service.pdf_path(session_id, document_id).unlink(missing_ok=True)
    except OSError:
        pass
    retrieval_service.invalidate(session_id, document_id)


def delete_current(session_id: str) -> None:
    """Explicit user delete of the active document."""
    security.require_valid_id(session_id)
    session_service.clear_document(session_id)
    retrieval_service.invalidate(session_id)
    log_event("delete_document", session_id, status="ok")


def max_pages() -> int:
    return MAX_PAGES
