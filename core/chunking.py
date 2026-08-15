"""
web_demo/core/chunking.py
-------------------------
Overlapping, page-tagged chunking for the public demo. Self-contained
reimplementation of the desktop generic chunker.

Each chunk carries page provenance and the safe *display* filename only —
never a filesystem path.
"""

from __future__ import annotations

from config import CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS, MAX_CHUNKS
from core.extraction import PageResult


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
