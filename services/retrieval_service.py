"""
web_demo/services/retrieval_service.py
--------------------------------------
Per-session FAISS retrieval for the public demo.

Isolation model
---------------
- Each session's index (``index.faiss``) and chunk metadata (``chunks.json``)
  live only under that session's directory.
- The in-memory cache is keyed by ``session_id`` AND ``document_id``. A lookup
  can only ever return data for the exact session that owns it.
- Chunks are stored/loaded as JSON — the demo never unpickles user data.

FAISS: IndexFlatIP over L2-normalized embeddings == cosine similarity.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import faiss
import numpy as np

from config import TOP_K
from services import embedding_service, session_service
from services.security import require_valid_id

# Cache key: (session_id, document_id) -> (faiss_index, chunks)
_cache: dict[tuple[str, str], tuple] = {}
_cache_lock = threading.Lock()


def build_and_store(session_id: str, document_id: str, chunks: list[dict]) -> int:
    """Embed chunks, build a FAISS index, and persist it to the session dir.

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

    faiss.write_index(index, str(session_service.index_path(session_id)))
    with open(session_service.chunks_path(session_id), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)

    return int(index.ntotal)


def _load(session_id: str, document_id: str) -> Optional[tuple]:
    with _cache_lock:
        hit = _cache.get((session_id, document_id))
    if hit is not None:
        return hit

    idx_path = session_service.index_path(session_id)
    chk_path = session_service.chunks_path(session_id)
    if not (idx_path.exists() and chk_path.exists()):
        return None

    index = faiss.read_index(str(idx_path))
    with open(chk_path, encoding="utf-8") as f:
        chunks = json.load(f)

    with _cache_lock:
        _cache[(session_id, document_id)] = (index, chunks)
    return index, chunks


def retrieve(
    session_id: str, document_id: str, query: str, top_k: int = TOP_K
) -> list[dict]:
    """Return up to ``top_k`` results for the given session+document only."""
    require_valid_id(session_id)
    require_valid_id(document_id)

    # A session may only retrieve against the document it currently owns.
    if session_service.get_document(session_id, document_id) is None:
        return []

    loaded = _load(session_id, document_id)
    if loaded is None:
        return []
    index, chunks = loaded
    if index.ntotal == 0:
        return []

    q_emb = embedding_service.embed_query(query)
    k = min(max(1, top_k), index.ntotal)
    scores, idx = index.search(q_emb, k)

    results: list[dict] = []
    for score, i in zip(scores[0], idx[0]):
        if i == -1 or i >= len(chunks):
            continue
        c = chunks[i]
        results.append(
            {
                "score": float(score),
                "document_name": c.get("document_name", ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "text": c.get("text", ""),
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def invalidate(session_id: str, document_id: str | None = None) -> None:
    with _cache_lock:
        for key in list(_cache.keys()):
            if key[0] == session_id and (document_id is None or key[1] == document_id):
                _cache.pop(key, None)
