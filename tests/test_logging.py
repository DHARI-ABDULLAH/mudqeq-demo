"""Privacy-safe logging tests: content must never reach the logs."""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager

from core import logging_utils
from services import llm_service


@contextmanager
def capture_logs():
    """Attach a temporary handler to the demo logger and capture its output.

    (capsys can't see it: the logger binds the real stdout at import time.)
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("mudqeq.demo")
    logger.addHandler(handler)
    try:
        yield buf
    finally:
        logger.removeHandler(handler)


def test_document_text_not_logged():
    with capture_logs() as buf:
        logging_utils.log_event(
            "ingest",
            "sid-123",
            pages=3,
            chunks=5,
            # Attempt to smuggle content via non-whitelisted kwargs:
            text="TOP-SECRET-DOCUMENT-BODY",
            content="SECRET-CONTENT",
        )
    out = buf.getvalue()
    assert "TOP-SECRET-DOCUMENT-BODY" not in out
    assert "SECRET-CONTENT" not in out
    assert '"pages": 3' in out and '"chunks": 5' in out


def test_question_not_logged():
    with capture_logs() as buf:
        logging_utils.log_event("llm", "sid-123", status="ok", question="MY-PRIVATE-QUESTION")
    out = buf.getvalue()
    assert "MY-PRIVATE-QUESTION" not in out
    assert '"status": "ok"' in out


def test_session_id_is_hashed():
    raw = "abcdef0123456789abcdef0123456789"
    with capture_logs() as buf:
        logging_utils.log_event("event", raw, status="ok")
    out = buf.getvalue()
    assert raw not in out  # raw session id never appears
    assert logging_utils.hash_session(raw) in out


def test_groq_key_never_logged_or_returned(monkeypatch):
    monkeypatch.setattr(llm_service, "groq_is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "get_groq_api_key", lambda: "sk-super-secret-key")
    monkeypatch.setattr(llm_service, "GROQ_MAX_RETRIES", 0)

    class _Resp:
        status_code = 500

        def json(self):
            return {}

    monkeypatch.setattr(llm_service.requests, "post", lambda *a, **k: _Resp())
    with capture_logs() as buf:
        res = llm_service.answer("sid", "question", [{"text": "c", "page_start": 1, "page_end": 1}])
    out = buf.getvalue()
    assert "sk-super-secret-key" not in out
    assert "sk-super-secret-key" not in res.user_message
    assert res.ok is False
