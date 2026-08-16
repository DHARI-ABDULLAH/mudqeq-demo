"""
web_demo/services/document_service.py
-------------------------------------
Orchestrates the demo ingest pipeline for a session (MULTI-DOCUMENT):

    validate (untrusted bytes)
      -> dedup within session (by content hash)
      -> store in session dir (id-addressed, never by filename)
      -> extract (generic, pdfplumber)
      -> chunk (page-tagged)
      -> embed + FAISS (per document)
      -> ready

Enforces per-session upload quotas and the concurrent-document cap. Each
document is independent; deleting one never affects the others. Never logs
document content.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from config import MAX_FILES_PER_SESSION, MAX_PAGES
from core import chunking, extraction
from core.logging_utils import log_event
from services import retrieval_service, security, session_service

STATUS_READY = session_service.STATUS_READY
STATUS_NEEDS_OCR = session_service.STATUS_NEEDS_OCR
STATUS_ERROR = session_service.STATUS_ERROR


@dataclass
class IngestResult:
    document_id: str
    display_name: str
    status: str
    num_pages: int
    num_chunks: int
    message: str


def ingest(session_id: str, file_bytes: bytes, original_name: str | None) -> IngestResult:
    """Validate + process an uploaded PDF, ADDING it to the session's set.

    Raises security.UploadRejected (Arabic message) for user-facing rejects.
    """
    security.require_valid_id(session_id)
    session_service.get_or_create(session_id)
    started = time.time()

    if not session_service.has_document_slot(session_id):
        raise security.UploadRejected(
            f"تم الوصول إلى الحد الأقصى لعدد المستندات في الجلسة "
            f"({MAX_FILES_PER_SESSION}). احذف مستنداً لإضافة آخر."
        )

    if not session_service.can_upload(session_id):
        raise security.UploadRejected(
            "تم الوصول إلى حد عدد مرات الرفع في هذه الجلسة التجريبية."
        )

    # Strict validation of untrusted input BEFORE writing anything to disk.
    security.validate_upload(file_bytes, original_name)

    display_name = security.safe_display_filename(original_name)

    # Per-session dedup by content hash (mirrors the desktop behavior).
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = session_service.find_by_hash(session_id, file_hash)
    if existing is not None:
        raise security.UploadRejected(
            f"هذا المستند مضاف مسبقاً باسم: {existing.display_name}"
        )

    document_id = security.new_id()
    dest = session_service.pdf_path(session_id, document_id)
    with open(dest, "wb") as f:
        f.write(file_bytes)

    session_service.record_upload(session_id)

    try:
        extract_result = extraction.extract_pdf(dest)
    except Exception as exc:  # noqa: BLE001
        _fail(session_id, document_id)
        log_event("ingest", session_id, status="extract_error",
                  error_category=type(exc).__name__)
        raise security.UploadRejected("تعذّر معالجة المستند.") from exc

    num_pages = extract_result.num_pages

    if not extract_result.has_usable_text():
        _fail(session_id, document_id)
        log_event("ingest", session_id, status="needs_ocr", pages=num_pages)
        raise security.UploadRejected(
            "لم يتم العثور على نص قابل للاستخراج من المستند "
            "(قد يكون ممسوحاً ضوئياً)."
        )

    chunks = chunking.build_chunks(
        extract_result.non_empty_pages(), document_id, display_name
    )
    if not chunks:
        _fail(session_id, document_id)
        log_event("ingest", session_id, status="no_chunks", pages=num_pages)
        raise security.UploadRejected("تعذّر إنشاء مقاطع قابلة للفهرسة من المستند.")

    try:
        retrieval_service.build_and_store(session_id, document_id, chunks)
    except Exception as exc:  # noqa: BLE001
        _fail(session_id, document_id)
        log_event("ingest", session_id, status="index_error",
                  error_category=type(exc).__name__)
        raise security.UploadRejected("تعذّر فهرسة المستند.") from exc

    record = session_service.DocumentRecord(
        document_id=document_id,
        display_name=display_name,
        num_pages=num_pages,
        num_chunks=len(chunks),
        status=STATUS_READY,
        file_hash=file_hash,
    )
    session_service.add_document(session_id, record)

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


def _fail(session_id: str, document_id: str) -> None:
    """Best-effort cleanup of a failed ingest (files + cache)."""
    retrieval_service.invalidate(session_id, document_id)
    session_service._delete_document_files(session_id, document_id)


def delete_document(session_id: str, document_id: str) -> None:
    """Explicit user delete of a single document."""
    security.require_valid_id(session_id)
    security.require_valid_id(document_id)
    session_service.remove_document(session_id, document_id)
    retrieval_service.invalidate(session_id, document_id)
    log_event("delete_document", session_id, status="ok")


def max_pages() -> int:
    return MAX_PAGES
