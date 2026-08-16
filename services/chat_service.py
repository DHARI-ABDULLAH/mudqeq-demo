"""
web_demo/services/chat_service.py
---------------------------------
The chat pipeline's decision logic, kept out of the Streamlit layer so it can
be tested directly:

    question -> intent -> (retrieval | document context) -> OpenAI -> quota

Three rules this module exists to guarantee:

1. An unreadable index NEVER masquerades as "the document has no information".
   It returns ``KIND_INDEX_ERROR`` instead.
2. The hosted LLM is only ever called after context was genuinely gathered.
3. The session's question quota is spent only once the provider has actually
   returned a usable answer. A 401/429/5xx/timeout/empty response costs the
   user nothing, and is reported as ``KIND_PROVIDER_ERROR`` — never as an
   answer.

Contains no Streamlit import and no document text in its diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import TOP_K
from core import intent
from core.logging_utils import log_event
from services import llm_service, retrieval_service, security, session_service

KIND_ANSWER = "answer"
KIND_NO_CONTENT = "no_content"
KIND_INDEX_ERROR = "index_error"
KIND_NO_SELECTION = "no_selection"
KIND_QUOTA = "quota_exceeded"
KIND_PROVIDER_ERROR = "provider_error"

INDEX_ERROR_MESSAGE = (
    "تعذر الوصول إلى فهرس المستند. أعد رفع المستند أو أعد المحاولة."
)
NO_CONTENT_MESSAGE = "لم أجد في المستندات المحددة معلومات كافية للإجابة."
NO_SELECTION_MESSAGE = "يرجى اختيار مستند واحد على الأقل قبل إرسال السؤال."
QUOTA_MESSAGE = "تم الوصول إلى حد عدد الأسئلة في هذه الجلسة التجريبية."


@dataclass
class ChatOutcome:
    kind: str
    mode: str = intent.FACTUAL
    text: str = ""
    sources: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    llm_called: bool = False
    error_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == KIND_ANSWER


def _diag(selected: int, valid: int, found: int, error: str = "") -> dict:
    return {
        "selected_document_count": selected,
        "valid_document_count": valid,
        "retrieved_results_count": found,
        "error": error,
    }


def gather_context(
    session_id: str, question: str, document_ids: list[str], top_k: int = TOP_K
) -> tuple[str, list[dict]]:
    """Return ``(mode, results)``. Raises ``IndexUnavailable`` on index faults."""
    mode = intent.classify(question)
    if mode == intent.OVERVIEW:
        return mode, retrieval_service.document_context(session_id, document_ids)
    return mode, retrieval_service.retrieve(
        session_id, document_ids, question, top_k=top_k
    )


def respond(
    session_id: str,
    question: str,
    document_ids: list[str],
    top_k: int = TOP_K,
) -> ChatOutcome:
    """Run one chat turn end-to-end and describe exactly what happened."""
    security.require_valid_id(session_id)

    selected = list(document_ids or [])
    valid = [d for d in selected if security.is_valid_id(d)]
    if not valid:
        return ChatOutcome(
            kind=KIND_NO_SELECTION,
            text=NO_SELECTION_MESSAGE,
            diagnostics=_diag(len(selected), 0, 0),
        )

    if not session_service.can_ask(session_id):
        return ChatOutcome(
            kind=KIND_QUOTA,
            text=QUOTA_MESSAGE,
            diagnostics=_diag(len(selected), len(valid), 0, error="quota"),
        )

    try:
        mode, results = gather_context(session_id, question, valid, top_k=top_k)
    except retrieval_service.IndexUnavailable as exc:
        log_event("chat", session_id, status="index_error", error_category=exc.reason)
        return ChatOutcome(
            kind=KIND_INDEX_ERROR,
            text=INDEX_ERROR_MESSAGE,
            diagnostics=_diag(len(selected), len(valid), 0, error=exc.reason),
            error_reason=exc.reason,
        )

    if not results:
        return ChatOutcome(
            kind=KIND_NO_CONTENT,
            mode=intent.classify(question),
            text=NO_CONTENT_MESSAGE,
            diagnostics=_diag(len(selected), len(valid), 0),
        )

    result = llm_service.answer(session_id, question, results, mode=mode)

    if not result.ok:
        # The provider failed us, not the user — their quota stays untouched.
        log_event(
            "chat", session_id, status="provider_error", error_category=result.error_category
        )
        return ChatOutcome(
            kind=KIND_PROVIDER_ERROR,
            mode=mode,
            text=result.user_message,
            sources=results,
            diagnostics=_diag(
                len(selected), len(valid), len(results), error=result.error_category
            ),
            llm_called=True,
            error_reason=result.error_category,
        )

    # Only a turn that produced a real answer costs the user a question.
    session_service.record_question(session_id)
    return ChatOutcome(
        kind=KIND_ANSWER,
        mode=mode,
        text=result.text,
        sources=results,
        diagnostics=_diag(len(selected), len(valid), len(results)),
        llm_called=True,
    )
