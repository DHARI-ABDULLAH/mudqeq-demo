"""
web_demo/services/retrieval_service.py
--------------------------------------
Per-session, per-document FAISS retrieval for the public demo (MULTI-DOCUMENT).

Canonical storage contract (ONE contract, used by writer and reader alike)
-------------------------------------------------------------------------
::

    <DEMO_STORAGE_ROOT>/<session_id>/<document_id>.faiss
    <DEMO_STORAGE_ROOT>/<session_id>/<document_id>.chunks.json

``build_and_store`` writes exactly these paths, and ``_load_from_disk`` reads
exactly these paths, both via ``session_service.index_path`` /
``session_service.chunks_path``. New writes NEVER use any other layout.

A very early build of the demo used a single-document layout
(``index.faiss`` / ``chunks.json``). Those names are still accepted for
*reading only*, so a session created before an upgrade keeps working. They are
never written again.

Isolation model
---------------
- Retrieval verifies EACH requested document belongs to the calling session
  before touching its index — a session can never read another's document.
- The in-memory cache is keyed by ``(session_id, document_id)``.
- Chunks are stored/loaded as JSON — the demo never unpickles user data.

Failure model
-------------
A missing/corrupt index is an INFRASTRUCTURE failure, not "no results". It
raises :class:`IndexUnavailable` so the UI can say so plainly instead of
misreporting it as "the document has no relevant information".
"""

from __future__ import annotations

import json
import threading
from typing import Iterator, Optional, Union

import faiss
import numpy as np

from config import MAX_RAG_CONTEXT_CHARS, TOP_K
from core.source_models import SOURCE_TYPE_PDF, normalize_source_type
from services import embedding_service, security, session_service
from services.security import require_valid_id

# Legacy single-document filenames — accepted for READING only.
LEGACY_INDEX_NAME = "index.faiss"
LEGACY_CHUNKS_NAME = "chunks.json"

# Cache key: (session_id, document_id) -> (faiss_index, chunks)
_cache: dict[tuple[str, str], tuple] = {}
_cache_lock = threading.Lock()


class IndexUnavailable(Exception):
    """The document's index/chunks are missing, unreadable, or inconsistent.

    Carries only a short, non-sensitive reason code — never document content.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --- Path contract --------------------------------------------------------
def canonical_paths(session_id: str, document_id: str) -> tuple:
    """The ONE pair of paths new writes use (index, chunks)."""
    return (
        session_service.index_path(session_id, document_id),
        session_service.chunks_path(session_id, document_id),
    )


def _legacy_paths(session_id: str) -> tuple:
    """Read-only compatibility with the original single-document layout."""
    base = session_service.session_dir(session_id)
    return (
        security.safe_child_path(base, LEGACY_INDEX_NAME),
        security.safe_child_path(base, LEGACY_CHUNKS_NAME),
    )


def _candidate_paths(session_id: str, document_id: str) -> Iterator[tuple]:
    yield canonical_paths(session_id, document_id)
    try:
        yield _legacy_paths(session_id)
    except security.UploadRejected:  # pragma: no cover - containment guard
        return


# --- Build ----------------------------------------------------------------
def build_and_store(session_id: str, document_id: str, chunks: list[dict]) -> int:
    """Embed chunks, build a FAISS index, and persist it under the session dir.

    Always writes the canonical per-document paths. Returns vectors indexed.
    """
    require_valid_id(session_id)
    require_valid_id(document_id)

    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.embed_passages(texts)
    dim = int(embeddings.shape[1]) if len(embeddings) else embedding_service.EMBED_DIM

    index = faiss.IndexFlatIP(dim)
    if len(embeddings):
        index.add(embeddings)

    index_file, chunks_file = canonical_paths(session_id, document_id)
    faiss.write_index(index, str(index_file))
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)

    return int(index.ntotal)


# --- Load -----------------------------------------------------------------
def _load_from_disk(session_id: str, document_id: str) -> tuple:
    """Read index + chunks from disk. Raises IndexUnavailable on any problem."""
    for index_file, chunks_file in _candidate_paths(session_id, document_id):
        if not (index_file.exists() and chunks_file.exists()):
            continue
        try:
            index = faiss.read_index(str(index_file))
        except Exception as exc:  # noqa: BLE001 - corrupt/unreadable index
            raise IndexUnavailable("index_unreadable") from exc
        try:
            with open(chunks_file, encoding="utf-8") as f:
                chunks = json.load(f)
        except (OSError, ValueError) as exc:
            raise IndexUnavailable("chunks_unreadable") from exc
        if not isinstance(chunks, list):
            raise IndexUnavailable("chunks_malformed")
        if index.ntotal != len(chunks):
            raise IndexUnavailable("index_chunks_mismatch")
        return index, chunks
    raise IndexUnavailable("index_missing")


def _load(session_id: str, document_id: str) -> tuple:
    with _cache_lock:
        hit = _cache.get((session_id, document_id))
    if hit is not None:
        return hit

    loaded = _load_from_disk(session_id, document_id)
    with _cache_lock:
        _cache[(session_id, document_id)] = loaded
    return loaded


# --- Verification + diagnostics -------------------------------------------
def verify_document_index(session_id: str, document_id: str) -> dict:
    """Prove the freshly-written index is usable by the REAL read path.

    Drops the in-memory cache first so this exercises exactly what Search and
    Chat will do later (disk -> faiss.read_index -> json -> index.search).
    Raises :class:`IndexUnavailable` if anything is wrong.
    """
    require_valid_id(session_id)
    require_valid_id(document_id)
    invalidate(session_id, document_id)

    index, chunks = _load_from_disk(session_id, document_id)
    if index.ntotal == 0 or not chunks:
        raise IndexUnavailable("index_empty")

    # Smoke search through the same FAISS call used at query time.
    probe = np.zeros((1, index.d), dtype="float32")
    probe[0][0] = 1.0
    try:
        scores, idx = index.search(probe, min(1, index.ntotal))
    except Exception as exc:  # noqa: BLE001
        raise IndexUnavailable("search_failed") from exc
    if idx[0][0] < 0:
        raise IndexUnavailable("search_returned_nothing")

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)

    return {
        "num_vectors": int(index.ntotal),
        "num_chunks": len(chunks),
        "dim": int(index.d),
    }


def index_diagnostics(session_id: str, document_id: str) -> dict:
    """Safe, content-free health report for one document. Never raises."""
    diag = {
        "index_exists": False,
        "chunks_file_exists": False,
        "index_loadable": False,
        "chunks_loadable": False,
        "num_vectors": 0,
        "num_indexed_chunks": 0,
        "layout": "none",
        "reason": "",
    }
    try:
        require_valid_id(session_id)
        require_valid_id(document_id)
    except security.UploadRejected:
        diag["reason"] = "invalid_id"
        return diag

    for layout, (index_file, chunks_file) in zip(
        ("canonical", "legacy"), _candidate_paths(session_id, document_id)
    ):
        exists_index = index_file.exists()
        exists_chunks = chunks_file.exists()
        if not (exists_index or exists_chunks):
            continue
        diag["layout"] = layout
        diag["index_exists"] = exists_index
        diag["chunks_file_exists"] = exists_chunks
        break

    try:
        index, chunks = _load_from_disk(session_id, document_id)
    except IndexUnavailable as exc:
        diag["reason"] = exc.reason
        return diag

    diag["index_loadable"] = True
    diag["chunks_loadable"] = True
    diag["num_vectors"] = int(index.ntotal)
    diag["num_indexed_chunks"] = len(chunks)
    return diag


# --- Ownership ------------------------------------------------------------
def _owned_ids(session_id: str, document_ids: Union[str, list[str]]) -> list[str]:
    """Filter to ids this session actually owns (foreign ids vanish silently)."""
    if isinstance(document_ids, str):
        document_ids = [document_ids]
    elif not isinstance(document_ids, (list, tuple)):
        return []
    return [
        d
        for d in document_ids
        if security.is_valid_id(d)
        and session_service.get_document(session_id, d) is not None
    ]


# --- Search ---------------------------------------------------------------
def _result_from_chunk(chunk: dict, score: float) -> dict:
    """Shape one stored chunk as a retrieval result.

    The PDF keys are unchanged. The provenance keys a web chunk needs are added
    alongside them and default to the PDF values, so a caller that only knows
    about documents keeps working and a caller that renders citations can tell
    a page number from a page title. Internal ids stay out of results.
    """
    source_type = normalize_source_type(chunk.get("source_type"))
    result = {
        "score": float(score),
        "document_name": chunk.get("document_name", ""),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "text": chunk.get("text", ""),
        "source_type": source_type,
    }
    if source_type != SOURCE_TYPE_PDF:
        result["url"] = chunk.get("url", "") or ""
        result["page_title"] = chunk.get("page_title", "") or ""
        result["section_title"] = chunk.get("section_title", "") or ""
    return result


def _search_one(session_id: str, document_id: str, q_emb, top_k: int) -> list[dict]:
    """Search a single owned document. Assumes ownership already verified."""
    index, chunks = _load(session_id, document_id)
    if index.ntotal == 0:
        return []

    k = min(max(1, top_k), index.ntotal)
    scores, idx = index.search(q_emb, k)

    out: list[dict] = []
    for score, i in zip(scores[0], idx[0]):
        if i == -1 or i >= len(chunks):
            continue
        out.append(_result_from_chunk(chunks[i], score))
    return out


def retrieve(
    session_id: str,
    document_ids: Union[str, list[str]],
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Return up to ``top_k`` results merged across the given owned documents.

    ``document_ids`` may be a single id (str) or a list of ids. Only documents
    that belong to ``session_id`` are searched; unknown/foreign ids are ignored.

    Raises :class:`IndexUnavailable` when an OWNED document's index cannot be
    loaded — that is an infrastructure failure, not an empty result set.
    """
    require_valid_id(session_id)
    valid_ids = _owned_ids(session_id, document_ids)
    if not valid_ids:
        return []

    q_emb = embedding_service.embed_query(query)

    merged: list[dict] = []
    for doc_id in valid_ids:
        merged.extend(_search_one(session_id, doc_id, q_emb, top_k))

    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:top_k]


# --- Document-level (overview) context ------------------------------------
def select_overview_chunks(
    chunks: list[dict], max_chars: int = MAX_RAG_CONTEXT_CHARS
) -> list[dict]:
    """Pick an evenly-spread, order-preserving subset that fits ``max_chars``.

    Taking only the first N chunks would summarize just the opening pages, so
    when a document exceeds the budget we stride across it instead. Reading
    order (and therefore citation order) is always preserved.
    """
    if not chunks:
        return []
    total = sum(len(c.get("text") or "") for c in chunks)
    if total <= max_chars:
        return list(chunks)

    picked: list[dict] = []
    used = 0
    stride = max(1, round(total / max_chars))
    for c in chunks[::stride]:
        size = len(c.get("text") or "")
        if used + size > max_chars:
            break
        picked.append(c)
        used += size

    if not picked:
        # Even a single chunk overflows the budget — keep its head so the
        # citation stays accurate while the context stays bounded.
        head = dict(chunks[0])
        head["text"] = (head.get("text") or "")[:max_chars]
        picked = [head]
    return picked


def document_context(
    session_id: str,
    document_ids: Union[str, list[str]],
    max_chars: int = MAX_RAG_CONTEXT_CHARS,
) -> list[dict]:
    """Ordered, bounded chunks for whole-document questions (summaries).

    Chunks keep document order, then page order, then their stored order, so
    the model reads the document the way a person would. Raises
    :class:`IndexUnavailable` if an owned document's chunks cannot be read.
    """
    require_valid_id(session_id)
    valid_ids = _owned_ids(session_id, document_ids)
    if not valid_ids:
        return []

    per_doc_budget = max(1, max_chars // len(valid_ids))
    out: list[dict] = []
    for doc_id in valid_ids:
        _, chunks = _load(session_id, doc_id)
        ordered = sorted(
            enumerate(chunks),
            key=lambda pair: (pair[1].get("page_start") or 0, pair[0]),
        )
        selected = select_overview_chunks(
            [c for _, c in ordered], max_chars=per_doc_budget
        )
        for c in selected:
            out.append(_result_from_chunk(c, 1.0))
    return out


def invalidate(session_id: str, document_id: Optional[str] = None) -> None:
    with _cache_lock:
        for key in list(_cache.keys()):
            if key[0] == session_id and (document_id is None or key[1] == document_id):
                _cache.pop(key, None)
