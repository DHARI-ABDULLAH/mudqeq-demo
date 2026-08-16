"""
web_demo/services/retrieval_service.py
--------------------------------------
Per-session, per-document FAISS retrieval for the public demo (MULTI-DOCUMENT).

Isolation model
---------------
- Each document's index (``<document_id>.faiss``) and chunk metadata
  (``<document_id>.chunks.json``) live only under that session's directory.
- The in-memory cache is keyed by ``(session_id, document_id)``. A lookup can
  only ever return data for the exact session that owns it.
- Retrieval verifies EACH requested document belongs to the calling session
  before touching its index — a session can never read another's document.
- Chunks are stored/loaded as JSON — the demo never unpickles user data.

Multi-document search: each selected document is queried independently, then
results are merged and globally ranked by score (cosine similarity via
IndexFlatIP over L2-normalized embeddings).
"""

from __future__ import annotations

import json
import threading
from typing import Optional, Union

import faiss

from config import TOP_K
from services import embedding_service, session_service
from services.security import require_valid_id

# Cache key: (session_id, document_id) -> (faiss_index, chunks)
_cache: dict[tuple[str, str], tuple] = {}
_cache_lock = threading.Lock()


def build_and_store(session_id: str, document_id: str, chunks: list[dict]) -> int:
    """Embed chunks, build a FAISS index, and persist it under the session dir.

    Returns the number of vectors indexed.
    """
    require_valid_id(session_id)
    require_valid_id(document_id)

    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.embed_passages(texts)
    dim = int(embeddings.shape[1]) if len(embeddings) else embedding_service.EMBED_DIM

    index = faiss.IndexFlatIP(dim)
    if len(embeddings):
        index.add(embeddings)

    faiss.write_index(index, str(session_service.index_path(session_id, document_id)))
    with open(
        session_service.chunks_path(session_id, document_id), "w", encoding="utf-8"
    ) as f:
        json.dump(chunks, f, ensure_ascii=False)

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)

    return int(index.ntotal)


def _load(session_id: str, document_id: str) -> Optional[tuple]:
    with _cache_lock:
        hit = _cache.get((session_id, document_id))
    if hit is not None:
        return hit

    idx_path = session_service.index_path(session_id, document_id)
    chk_path = session_service.chunks_path(session_id, document_id)
    if not (idx_path.exists() and chk_path.exists()):
        return None

    index = faiss.read_index(str(idx_path))
    with open(chk_path, encoding="utf-8") as f:
        chunks = json.load(f)

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)
    return index, chunks


def _search_one(session_id: str, document_id: str, q_emb, top_k: int) -> list[dict]:
    """Search a single owned document. Assumes ownership already verified."""
    loaded = _load(session_id, document_id)
    if loaded is None:
        return []
    index, chunks = loaded
    if index.ntotal == 0:
        return []

    k = min(max(1, top_k), index.ntotal)
    scores, idx = index.search(q_emb, k)

    out: list[dict] = []
    for score, i in zip(scores[0], idx[0]):
        if i == -1 or i >= len(chunks):
            continue
        c = chunks[i]
        out.append(
            {
                "score": float(score),
                "document_name": c.get("document_name", ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "text": c.get("text", ""),
            }
        )
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
    """
    require_valid_id(session_id)
    if isinstance(document_ids, str):
        document_ids = [document_ids]

    # A session may only retrieve against documents it currently owns.
    valid_ids = [
        d
        for d in document_ids
        if session_service.get_document(session_id, d) is not None
    ]
    if not valid_ids:
        return []

    q_emb = embedding_service.embed_query(query)

    merged: list[dict] = []
    for doc_id in valid_ids:
        merged.extend(_search_one(session_id, doc_id, q_emb, top_k))

    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:top_k]


def invalidate(session_id: str, document_id: str | None = None) -> None:
    with _cache_lock:
        for key in list(_cache.keys()):
            if key[0] == session_id and (document_id is None or key[1] == document_id):
                _cache.pop(key, None)
