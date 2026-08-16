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


def test_openai_key_never_logged_or_returned(monkeypatch):
    import httpx
    from openai import InternalServerError

    monkeypatch.setattr(llm_service, "openai_is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "get_openai_api_key", lambda: "sk-super-secret-key")
    monkeypatch.setattr(llm_service, "get_openai_model", lambda: "gpt-4o-mini")
    monkeypatch.setattr(llm_service, "OPENAI_MAX_RETRIES", 0)

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = InternalServerError(
        "Internal server error",
        response=httpx.Response(500, request=request, json={"error": {}}),
        body={},
    )

    class _Responses:
        def create(self, **kwargs):
            raise error

    class _Client:
        responses = _Responses()

    monkeypatch.setattr(llm_service, "get_client", lambda: _Client())
    with capture_logs() as buf:
        res = llm_service.answer("sid", "question", [{"text": "c", "page_start": 1, "page_end": 1}])
    out = buf.getvalue()
    assert "sk-super-secret-key" not in out
    assert "sk-super-secret-key" not in res.user_message
    assert res.ok is False


def test_llm_answer_logs_status_without_content(monkeypatch):
    monkeypatch.setattr(llm_service, "openai_is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "get_openai_model", lambda: "gpt-4o-mini")

    class _Response:
        output_text = "SECRET-ANSWER-TEXT"

    class _Client:
        class responses:  # noqa: N801 - test double
            @staticmethod
            def create(**kwargs):
                return _Response()

    monkeypatch.setattr(llm_service, "get_client", lambda: _Client())
    with capture_logs() as buf:
        res = llm_service.answer(
            "sid", "MY-PRIVATE-QUESTION", [{"text": "c", "page_start": 1, "page_end": 1}]
        )
    out = buf.getvalue()
    assert res.ok is True
    assert "SECRET-ANSWER-TEXT" not in out
    assert "MY-PRIVATE-QUESTION" not in out
    assert '"status": "ok"' in out
