"""
web_demo/services/embedding_service.py
--------------------------------------
Loads intfloat/multilingual-e5-small once (process-wide singleton) and embeds
passages/queries with the e5 prefix convention and L2 normalization.

The MODEL is read-only and expensive, so it is cached globally. User data
(documents, chunks, indexes) is NEVER cached here — that lives per-session in
``retrieval_service`` keyed strictly by session id.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from config import EMBED_MODEL_NAME

EMBED_DIM = 384  # multilingual-e5-small

_model = None
_model_lock = threading.Lock()


def get_embedder():
    """Return the cached SentenceTransformer, loading it on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                # Prefer offline if the model was baked into the image.
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_passages(
    texts: list[str], batch_size: int = 32, show_progress: bool = False
) -> np.ndarray:
    model = get_embedder()
    prefixed = ["passage: " + t for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    model = get_embedder()
    emb = model.encode(["query: " + text], normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")
