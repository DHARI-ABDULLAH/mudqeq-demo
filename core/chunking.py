"""
web_demo/core/chunking.py
-------------------------
Overlapping, page-tagged chunking for the public demo. Self-contained
reimplementation of the desktop generic chunker.

Each chunk carries page provenance and the safe *display* filename only —
never a filesystem path.
"""

from __future__ import annotations

from config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    MAX_CHUNKS,
    MAX_URL_CHUNKS,
)
from core.extraction import PageResult
from core.source_models import SOURCE_TYPE_URL


def build_chunks(
    pages: list[PageResult],
    document_id: str,
    document_name: str,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> list[dict]:
    """Split pages into overlapping chunks. Stops at ``max_chunks``.

    Returns a list of plain dicts (JSON-serializable — no pickle) so the demo
    never deserializes arbitrary objects.
    """
    chunks: list[dict] = []
    buffer = ""
    start_page: int | None = None
    last_page: int | None = None

    def emit(page_start: int, page_end: int, text: str) -> bool:
        idx = len(chunks)
        chunks.append(
            {
                "chunk_id": f"{document_id}:{idx}",
                "document_id": document_id,
                "document_name": document_name,
                "text": text.strip(),
                "page_start": page_start,
                "page_end": page_end,
            }
        )
        return len(chunks) < max_chunks

    for p in pages:
        if not p.text.strip():
            continue
        if start_page is None:
            start_page = p.page_number
        last_page = p.page_number

        buffer = (buffer + "\n" + p.text) if buffer else p.text

        if len(buffer) >= target_chars:
            if not emit(start_page, p.page_number, buffer):
                return chunks
            buffer = buffer[-overlap_chars:] if overlap_chars > 0 else ""
            start_page = p.page_number

    if buffer.strip() and start_page is not None and last_page is not None:
        emit(start_page, last_page, buffer)

    return chunks


def build_url_chunks(
    sections,
    source_id: str,
    source_name: str,
    url: str,
    page_title: str = "",
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    max_chunks: int = MAX_URL_CHUNKS,
) -> list[dict]:
    """Split extracted web sections into chunks that mirror the PDF contract.

    A web page has no pages, so ``page_start``/``page_end`` stay ``None`` and
    the section heading becomes the in-page locator instead. Every other key
    matches :func:`build_chunks`, which is what lets retrieval, evidence
    collection, and the existing renderers treat both source kinds alike.

    Chunks never span two headings: a citation that pointed at one section
    while quoting another would not be a real citation.
    """
    chunks: list[dict] = []
    max_chunks = max(1, int(max_chunks))
    target_chars = max(1, int(target_chars))
    overlap_chars = max(0, int(overlap_chars))

    def emit(section_title: str, text: str) -> bool:
        body = text.strip()
        if not body:
            return True
        index = len(chunks)
        chunks.append(
            {
                "chunk_id": f"{source_id}:{index}",
                "document_id": source_id,
                "document_name": source_name,
                "source_type": SOURCE_TYPE_URL,
                "url": url,
                "page_title": page_title,
                "section_title": section_title,
                "text": body,
                "page_start": None,
                "page_end": None,
            }
        )
        return len(chunks) < max_chunks

    # A heading with no text of its own (a section that only introduces
    # sub-sections) is carried forward as breadcrumb context instead of being
    # indexed alone. A chunk that is nothing but a heading has no information
    # in it, yet still competes for retrieval slots against real content.
    pending: list[str] = []

    for section in sections or []:
        heading = (getattr(section, "heading", "") or "").strip()
        body = (getattr(section, "text", "") or "").strip()
        if not body and not heading:
            continue

        if not body:
            pending.append(heading)
            continue

        # The heading trail is repeated into every chunk of its section so a
        # chunk retrieved alone still carries the context it was written under.
        trail = [h for h in (*pending, heading) if h]
        pending = []
        prefix = ("\n".join(trail) + "\n") if trail else ""
        locator = heading or (trail[-1] if trail else "")

        buffer = ""
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            buffer = f"{buffer}\n{line}" if buffer else line
            if len(buffer) >= target_chars:
                if not emit(locator, prefix + buffer):
                    return chunks
                buffer = buffer[-overlap_chars:] if overlap_chars > 0 else ""

        if buffer.strip():
            if not emit(locator, prefix + buffer):
                return chunks

    # An outline with no prose under any heading is thin, but indexing it is
    # still better than telling the user the page could not be read.
    if not chunks and pending:
        emit(pending[0], "\n".join(pending))

    return chunks
