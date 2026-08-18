"""
web_demo/services/url_source_service.py
---------------------------------------
Ingest a web page as a SOURCE, using the same pipeline PDFs already use:

    url
      -> validate + SSRF checks   (url_security_service)
      -> dedupe within session    (canonical-URL hash)
      -> fetch, bounded           (url_fetch_service)
      -> extract readable content (core.html_extract)
      -> chunk, section-tagged    (core.chunking.build_url_chunks)
      -> embed + FAISS            (retrieval_service — the SAME index format)
      -> verify through the real read path
      -> ready

There is deliberately no second retrieval stack. A URL source is stored as
``<session>/<source_id>.faiss`` + ``<source_id>.chunks.json`` exactly like a
PDF, which is why search, chat, overview, and case analysis pick it up without
knowing it came from the web.

Privacy: the page is fetched, extracted, chunked, and embedded on this server.
Only bounded retrieved chunks ever reach the model — never the whole page. The
page's text is never logged; only counters and coarse status codes are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from config import (
    MAX_URL_EXTRACTED_CHARS,
    MAX_URL_SOURCES_PER_SESSION,
    MAX_URLS_PER_SESSION,
    MIN_URL_EXTRACTED_CHARS,
)
from core import chunking, html_extract
from core.logging_utils import log_event
from core.source_models import SOURCE_TYPE_URL, domain_of
from services import (
    retrieval_service,
    security,
    session_service,
    url_fetch_service,
    url_security_service,
)
from services.url_fetch_service import UrlFetchFailed
from services.url_security_service import UrlRejected

STATUS_READY = session_service.STATUS_READY

# --- Progress stages (shown in the UI while the page is processed) ---------
STAGE_FETCH = "fetch"
STAGE_EXTRACT = "extract"
STAGE_CHUNK = "chunk"
STAGE_INDEX = "index"
STAGE_READY = "ready"

STAGE_ORDER = (STAGE_FETCH, STAGE_EXTRACT, STAGE_CHUNK, STAGE_INDEX, STAGE_READY)

STAGE_LABELS_AR = {
    STAGE_FETCH: "جاري جلب الصفحة...",
    STAGE_EXTRACT: "جاري استخراج المحتوى...",
    STAGE_CHUNK: "جاري إنشاء المقاطع...",
    STAGE_INDEX: "جاري الفهرسة...",
    STAGE_READY: "جاهز",
}

# --- Reason codes owned by this stage --------------------------------------
REASON_SLOT_LIMIT = "url_slot_limit"
REASON_FETCH_QUOTA = "url_fetch_quota"
REASON_DUPLICATE = "duplicate_url"
REASON_NO_READABLE = "no_readable_content"
REASON_NO_CHUNKS = "no_chunks"
REASON_INDEX_FAILED = "index_failed"
REASON_VERIFY_FAILED = "verify_failed"
REASON_NOT_FOUND = "source_not_found"
REASON_NOT_URL = "not_a_url_source"

DUPLICATE_MESSAGE = "هذا الرابط مضاف مسبقاً."
NO_READABLE_MESSAGE = "تعذر استخراج محتوى قابل للقراءة من هذا الرابط."
NO_CHUNKS_MESSAGE = "تعذّر إنشاء مقاطع قابلة للفهرسة من هذه الصفحة."
INDEX_FAILED_MESSAGE = "تعذّرت فهرسة محتوى الصفحة."
VERIFY_FAILED_MESSAGE = (
    "تعذّر التحقق من فهرس الصفحة بعد المعالجة. يرجى إعادة إضافة الرابط."
)
NOT_FOUND_MESSAGE = "هذا المصدر غير موجود في جلستك."
NOT_URL_MESSAGE = "هذا المصدر ليس رابطاً."


class UrlSourceError(Exception):
    """Any failure of the add/refresh pipeline, with an Arabic message."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def user_message(self) -> str:
        return str(self)


@dataclass
class UrlIngestResult:
    source_id: str
    display_name: str
    page_title: str
    original_url: str
    final_url: str
    domain: str
    status: str
    num_chunks: int
    retrieved_at: float
    content_type: str
    truncated: bool = False
    message: str = ""


def _notify(progress, stage: str) -> None:
    """Report a stage to the UI. A UI callback must never break ingestion."""
    if progress is None:
        return
    try:
        progress(stage, STAGE_LABELS_AR.get(stage, stage))
    except Exception:  # noqa: BLE001
        pass


def _display_name(page_title: str, url: str) -> str:
    """Short, safe label for the sources list and citations.

    Derived from the page's own title when it has one, else from its address.
    Never used to build a filesystem path — sources are addressed by id.
    """
    title = " ".join((page_title or "").split()).strip()
    if title:
        return title[:120]
    domain = domain_of(url)
    return (domain or "صفحة ويب")[:120]


def _cleanup(session_id: str, source_id: str) -> None:
    """Best-effort removal of a half-built source."""
    retrieval_service.invalidate(session_id, source_id)
    session_service._delete_document_files(session_id, source_id)


def _build_source(
    session_id: str,
    source_id: str,
    raw_url: str,
    *,
    progress=None,
) -> tuple[UrlIngestResult, str]:
    """Fetch -> extract -> chunk -> index one URL into ``source_id``.

    Returns ``(result, url_hash)``. Raises :class:`UrlSourceError`,
    :class:`UrlRejected`, or :class:`UrlFetchFailed` — all of which carry an
    Arabic message. Nothing is written until extraction succeeded, so a failed
    refresh cannot destroy a working index.
    """
    _notify(progress, STAGE_FETCH)
    session_service.record_url_fetch(session_id)
    fetched = url_fetch_service.fetch(raw_url, session_id=session_id)

    _notify(progress, STAGE_EXTRACT)
    if fetched.content_type in url_fetch_service.TEXT_CONTENT_TYPES:
        extracted = html_extract.extract_plain_text(
            fetched.text, max_chars=MAX_URL_EXTRACTED_CHARS
        )
    else:
        extracted = html_extract.extract(
            fetched.text, max_chars=MAX_URL_EXTRACTED_CHARS
        )

    if not extracted.has_usable_text() or extracted.total_chars < MIN_URL_EXTRACTED_CHARS:
        log_event(
            "url_ingest",
            session_id,
            status="no_readable_content",
            http_status=fetched.status_code,
        )
        raise UrlSourceError(REASON_NO_READABLE, NO_READABLE_MESSAGE)

    page_title = extracted.title or _display_name("", fetched.final_url)
    display_name = _display_name(extracted.title, fetched.final_url)

    _notify(progress, STAGE_CHUNK)
    chunks = chunking.build_url_chunks(
        extracted.non_empty_sections(),
        source_id,
        display_name,
        fetched.final_url,
        page_title,
    )
    if not chunks:
        raise UrlSourceError(REASON_NO_CHUNKS, NO_CHUNKS_MESSAGE)

    _notify(progress, STAGE_INDEX)
    try:
        retrieval_service.build_and_store(session_id, source_id, chunks)
    except Exception as exc:  # noqa: BLE001
        _cleanup(session_id, source_id)
        log_event(
            "url_ingest",
            session_id,
            status="index_error",
            error_category=type(exc).__name__,
        )
        raise UrlSourceError(REASON_INDEX_FAILED, INDEX_FAILED_MESSAGE) from exc

    # Same rule as PDFs: "ready" must mean genuinely queryable, proved through
    # the exact read path Search and Chat will use.
    try:
        retrieval_service.verify_document_index(session_id, source_id)
    except retrieval_service.IndexUnavailable as exc:
        _cleanup(session_id, source_id)
        log_event(
            "url_ingest", session_id, status="verify_failed", error_category=exc.reason
        )
        raise UrlSourceError(REASON_VERIFY_FAILED, VERIFY_FAILED_MESSAGE) from exc

    _notify(progress, STAGE_READY)
    result = UrlIngestResult(
        source_id=source_id,
        display_name=display_name,
        page_title=page_title,
        original_url=fetched.original_url,
        final_url=fetched.final_url,
        domain=domain_of(fetched.final_url),
        status=STATUS_READY,
        num_chunks=len(chunks),
        retrieved_at=time.time(),
        content_type=fetched.content_type,
        truncated=extracted.truncated,
        message="تم تجهيز الرابط بنجاح.",
    )
    return result, url_security_service.url_hash(fetched.original_url)


def _record_for(result: UrlIngestResult, url_hash: str, created_at: float) -> session_service.DocumentRecord:
    return session_service.DocumentRecord(
        document_id=result.source_id,
        display_name=result.display_name,
        num_pages=0,
        num_chunks=result.num_chunks,
        status=result.status,
        file_hash=url_hash,
        created_at=created_at,
        source_type=SOURCE_TYPE_URL,
        original_url=result.original_url,
        final_url=result.final_url,
        page_title=result.page_title,
        content_type=result.content_type,
        retrieved_at=result.retrieved_at,
    )


# --- Public API ------------------------------------------------------------
def add_url(session_id: str, raw_url: str, *, progress=None) -> UrlIngestResult:
    """Add a web page to the session as a selectable, searchable source.

    Raises :class:`UrlRejected` (bad/unsafe address), :class:`UrlFetchFailed`
    (the request failed), or :class:`UrlSourceError` (quota, duplicate, or the
    page had nothing readable). Every one carries an Arabic message that is
    distinct from "no information found".
    """
    security.require_valid_id(session_id)
    session_service.get_or_create(session_id)
    started = time.time()

    if not session_service.has_url_slot(session_id):
        raise UrlSourceError(
            REASON_SLOT_LIMIT,
            f"تم الوصول إلى الحد الأقصى لعدد الروابط في الجلسة "
            f"({MAX_URL_SOURCES_PER_SESSION}). احذف رابطاً لإضافة آخر.",
        )
    if not session_service.can_fetch_url(session_id):
        raise UrlSourceError(
            REASON_FETCH_QUOTA,
            f"تم الوصول إلى حد جلب الروابط في هذه الجلسة التجريبية "
            f"({MAX_URLS_PER_SESSION}).",
        )

    # Validate the address before anything else so an unsafe target never even
    # reaches the duplicate check or the fetch counter.
    validated = url_security_service.validate_url(raw_url)
    incoming_hash = url_security_service.url_hash(validated.url)

    existing = session_service.find_url_source(session_id, incoming_hash)
    if existing is not None:
        raise UrlSourceError(REASON_DUPLICATE, DUPLICATE_MESSAGE)

    source_id = security.new_id()
    try:
        result, final_hash = _build_source(
            session_id, source_id, validated.url, progress=progress
        )
    except (UrlSourceError, UrlRejected, UrlFetchFailed):
        _cleanup(session_id, source_id)
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internals
        _cleanup(session_id, source_id)
        log_event(
            "url_ingest", session_id, status="error", error_category=type(exc).__name__
        )
        raise UrlSourceError(REASON_INDEX_FAILED, INDEX_FAILED_MESSAGE) from exc

    # A redirect can land two different entry URLs on the same page; keep the
    # entry hash as identity so re-adding the same link is still a duplicate.
    session_service.add_document(
        session_id, _record_for(result, incoming_hash, time.time())
    )
    log_event(
        "url_ingest",
        session_id,
        status="ready",
        chunks=result.num_chunks,
        http_status=200,
        duration_ms=int((time.time() - started) * 1000),
    )
    return result


def refresh_url(session_id: str, source_id: str, *, progress=None) -> UrlIngestResult:
    """Re-fetch a URL source and replace its chunks and index in place.

    The record's id never changes, so any selection the user already made stays
    valid. The old index is only overwritten once the new content has been
    fetched, extracted, and chunked successfully.
    """
    security.require_valid_id(session_id)
    security.require_valid_id(source_id)

    record = session_service.get_document(session_id, source_id)
    if record is None:
        raise UrlSourceError(REASON_NOT_FOUND, NOT_FOUND_MESSAGE)
    if not record.is_url:
        raise UrlSourceError(REASON_NOT_URL, NOT_URL_MESSAGE)

    if not session_service.can_fetch_url(session_id):
        raise UrlSourceError(
            REASON_FETCH_QUOTA,
            f"تم الوصول إلى حد جلب الروابط في هذه الجلسة التجريبية "
            f"({MAX_URLS_PER_SESSION}).",
        )

    target = record.original_url or record.final_url
    result, _ = _build_source(session_id, source_id, target, progress=progress)

    session_service.add_document(
        session_id, _record_for(result, record.file_hash, record.created_at)
    )
    log_event(
        "url_refresh", session_id, status="ready", chunks=result.num_chunks
    )
    return result


def delete_url_source(session_id: str, source_id: str) -> None:
    """Delete a URL source: record, chunks, and FAISS index.

    Identical to deleting a PDF — the stored artefacts are the same — so other
    sources in the session are untouched.
    """
    security.require_valid_id(session_id)
    security.require_valid_id(source_id)
    session_service.remove_document(session_id, source_id)
    retrieval_service.invalidate(session_id, source_id)
    log_event("url_delete", session_id, status="ok")


def describe_error(exc: BaseException) -> tuple[str, str]:
    """Return ``(reason_code, arabic_message)`` for any add/refresh failure."""
    if isinstance(exc, (UrlSourceError, UrlRejected, UrlFetchFailed)):
        return exc.reason, str(exc)
    return REASON_INDEX_FAILED, INDEX_FAILED_MESSAGE
