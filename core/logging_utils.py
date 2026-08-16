"""
web_demo/core/logging_utils.py
------------------------------
Privacy-safe structured logging for the public demo.

Design rule: it must be *impossible* to accidentally log private content.
The only public entry point, :func:`log_event`, accepts a fixed set of
non-sensitive fields. It does NOT accept free-form message text, so document
text, chunks, questions, answers, RAG context, filenames, or API keys can
never flow through it.

Session identifiers are hashed before logging so raw session IDs never appear
in logs either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

_LOGGER_NAME = "mudqeq.demo"
_SALT = os.environ.get("LOG_HASH_SALT", "mudqeq-demo-static-salt")

# Fields that are allowed to appear in a log record. Anything else is dropped.
_ALLOWED_FIELDS = {
    "event",
    "session",  # already-hashed session id
    "status",  # coarse status/category string
    "duration_ms",
    "pages",
    "chunks",
    "top_k",
    "size_bytes",
    "http_status",
    "error_category",
    "attempts",
}


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_logger = _build_logger()


def hash_session(session_id: str | None) -> str:
    """Return a short, non-reversible hash of a session id for logs."""
    if not session_id:
        return "anon"
    digest = hashlib.sha256((_SALT + session_id).encode("utf-8")).hexdigest()
    return digest[:12]


def log_event(event: str, session_id: str | None = None, **fields) -> None:
    """Emit a single privacy-safe JSON log line.

    Only whitelisted, non-sensitive fields are recorded. Unknown fields are
    silently dropped so callers cannot leak content by mistake.
    """
    record: dict[str, object] = {"event": str(event)[:64]}
    if session_id is not None:
        record["session"] = hash_session(session_id)
    for key, value in fields.items():
        if key not in _ALLOWED_FIELDS:
            continue
        if isinstance(value, str):
            value = value[:64]
        record[key] = value
    try:
        _logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))
    except Exception:  # noqa: BLE001 - logging must never crash the app
        pass
