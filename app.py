"""
web_demo/app.py
---------------
Public web demo of "المدقق الشامل" (Mudqeq AI).

Streamlit entry point. Wires the isolated demo services together and mirrors
the desktop app's UX (multi-document management, selection, search, chat):

    upload (validated, temporary, per-session, MANY docs)
      -> extract -> chunk -> embed -> FAISS (per document)
      -> select documents -> search (local) / chat (OpenAI hosted LLM)

Resilience note
---------------
Streamlit re-executes THIS file from disk on every rerun, but already-imported
modules stay cached in ``sys.modules`` for the life of the process. On hosted
deploys that can leave a fresh app.py running against older service/UI modules.
Every cross-module call below therefore goes through a small defensive helper
that falls back to the previous API (or an inline implementation) instead of
raising AttributeError/TypeError.

The desktop application is NOT imported or affected by this module.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import sys
import threading
from pathlib import Path

import streamlit as st

_DEMO_PACKAGES = {"config", "core", "services", "ui"}
_STAMP_ATTR = "_mudqeq_demo_source_stamp"
_LOCK_ATTR = "_mudqeq_demo_reload_lock"


def _reload_demo_modules_if_updated() -> None:
    """Drop cached demo modules whenever this file's source changes.

    Streamlit re-executes app.py from disk on every rerun, but modules already
    in ``sys.modules`` survive for the life of the process. After a hosted
    redeploy that leaves a fresh app.py running against stale service/UI
    modules, which surfaces as AttributeError or silently empty results.
    Stamping the process with this file's hash forces exactly one clean
    re-import per deployed version.
    """
    lock = getattr(sys, _LOCK_ATTR, None)
    if lock is None:
        lock = threading.Lock()
        setattr(sys, _LOCK_ATTR, lock)

    with lock:
        try:
            stamp = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
        except OSError:
            return
        if getattr(sys, _STAMP_ATTR, None) == stamp:
            return
        setattr(sys, _STAMP_ATTR, stamp)
        for name in list(sys.modules):
            if name.split(".")[0] in _DEMO_PACKAGES:
                sys.modules.pop(name, None)


_reload_demo_modules_if_updated()

import config  # noqa: E402

_bootstrap = getattr(config, "bootstrap_streamlit_secrets", None)
if _bootstrap is not None:
    try:
        _bootstrap()
    except Exception:  # noqa: BLE001
        pass

from core.logging_utils import log_event  # noqa: E402

try:  # ``core.intent`` is newer than the first deploy; never break start-up.
    from core import intent as _intent  # noqa: E402
except Exception:  # noqa: BLE001
    _intent = None

from services import (  # noqa: E402
    cleanup_service,
    document_service,
    llm_service,
    retrieval_service,
    security,
    session_service,
)

try:  # newer than the first deploy; never break start-up if absent
    from services import chat_service  # noqa: E402
except Exception:  # noqa: BLE001
    chat_service = None

try:  # "تحليل حالة" ships after the first deploy; keep start-up resilient.
    from core import case_models  # noqa: E402
    from services import case_analysis_service  # noqa: E402
except Exception as exc:  # noqa: BLE001
    case_analysis_service = None
    case_models = None
    _case_import_error = f"{type(exc).__name__}: {exc}"
else:
    _case_import_error = ""

from ui import components, styles  # noqa: E402

st.set_page_config(
    page_title="المدقق الشامل — نسخة تجريبية",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS as early as possible so Cloud toolbar icons never flash in.
try:
    styles.inject()
except Exception:  # noqa: BLE001
    pass

NAV = [
    ("chat", "المحادثة"),
    ("case", "تحليل حالة"),
    ("documents", "المستندات"),
    ("search", "البحث"),
    ("about", "حول النسخة التجريبية"),
]

ALL_DOCS = "__all__"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")

# Shown when the index itself is unreadable. Deliberately different from the
# "no relevant content" answer — conflating the two sends users hunting through
# a document that was never actually searched.
INDEX_ERROR_MESSAGE = (
    "تعذر الوصول إلى فهرس المستند. أعد رفع المستند أو أعد المحاولة."
)
NO_CONTENT_MESSAGE = "لم أجد في المستندات المحددة معلومات كافية للإجابة."


class RetrievalUnavailable(Exception):
    """The document index could not be read — not an empty result set."""


# --- Defensive config accessors -------------------------------------------
def _cfg_int(name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def _top_k_min() -> int:
    return max(1, _cfg_int("TOP_K_MIN", 2))


def _top_k_max() -> int:
    return max(_cfg_int("TOP_K_MAX", 10), _top_k_min())


def _top_k_default() -> int:
    raw = getattr(config, "TOP_K_DEFAULT", None)
    if raw is None:
        raw = _cfg_int("TOP_K", 4)
    return min(max(int(raw), _top_k_min()), _top_k_max())


def _clamp_top_k(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = _top_k_default()
    return min(max(value, _top_k_min()), _top_k_max())


def _search_max_results() -> int:
    return max(1, _cfg_int("SEARCH_MAX_RESULTS", 20))


def _upload_limits_caption() -> str:
    fn = getattr(config, "upload_limits_caption_ar", None)
    if fn is not None:
        try:
            return fn()
        except Exception:  # noqa: BLE001
            pass
    return (
        f"PDF فقط، بحد أقصى {_cfg_int('MAX_FILE_SIZE_MB', 50)} ميغابايت و"
        f"{_cfg_int('MAX_PAGES', 200)} صفحة لكل ملف."
    )


def _search_default_results() -> int:
    return min(max(1, _cfg_int("SEARCH_DEFAULT_RESULTS", 8)), _search_max_results())


def _max_files_per_session() -> int:
    return max(1, _cfg_int("MAX_FILES_PER_SESSION", 5))


# --- Defensive service accessors ------------------------------------------
def _is_valid_id(value) -> bool:
    fn = getattr(security, "is_valid_id", None)
    if fn is not None:
        try:
            return bool(fn(value))
        except Exception:  # noqa: BLE001
            pass
    return bool(isinstance(value, str) and _HEX32.match(value))


def _current_document(session_id: str):
    fn = getattr(session_service, "current_document", None)
    if fn is None:
        return None
    try:
        return fn(session_id)
    except Exception:  # noqa: BLE001
        return None


def _list_documents(session_id: str) -> list:
    fn = getattr(session_service, "list_documents", None)
    if fn is not None:
        try:
            return list(fn(session_id))
        except Exception:  # noqa: BLE001
            pass
    doc = _current_document(session_id)
    return [doc] if doc is not None else []


def _ready_documents(session_id: str) -> list:
    fn = getattr(session_service, "ready_documents", None)
    if fn is not None:
        try:
            return list(fn(session_id))
        except Exception:  # noqa: BLE001
            pass
    return [d for d in _list_documents(session_id) if getattr(d, "status", "") == "ready"]


def _live_document_count(session_id: str) -> int:
    fn = getattr(session_service, "live_document_count", None)
    if fn is not None:
        try:
            return int(fn(session_id))
        except Exception:  # noqa: BLE001
            pass
    return len(_list_documents(session_id))


def _has_document_slot(session_id: str) -> bool:
    fn = getattr(session_service, "has_document_slot", None)
    if fn is not None:
        try:
            return bool(fn(session_id))
        except Exception:  # noqa: BLE001
            pass
    return _live_document_count(session_id) < _max_files_per_session()


def _session_stats(session_id: str) -> dict:
    fn = getattr(session_service, "stats", None)
    if fn is not None:
        try:
            return dict(fn(session_id))
        except Exception:  # noqa: BLE001
            pass
    ready = _ready_documents(session_id)
    return {
        "num_documents": len(ready),
        "total_pages": sum(getattr(d, "num_pages", 0) for d in ready),
        "total_chunks": sum(getattr(d, "num_chunks", 0) for d in ready),
    }


def _remaining_questions(session_id: str) -> int:
    fn = getattr(session_service, "remaining_questions", None)
    if fn is None:
        return 0
    try:
        return int(fn(session_id))
    except Exception:  # noqa: BLE001
        return 0


def _remaining_cases(session_id: str) -> int:
    fn = getattr(session_service, "remaining_cases", None)
    if fn is None:
        return 0
    try:
        return int(fn(session_id))
    except Exception:  # noqa: BLE001
        return 0


def _delete_document(session_id: str, document_id: str) -> None:
    fn = getattr(document_service, "delete_document", None)
    if fn is not None:
        fn(session_id, document_id)
        return
    fn = getattr(document_service, "delete_current", None)
    if fn is not None:
        fn(session_id)
        return
    raise RuntimeError("no delete API available")


def _note_failure(where: str, exc: BaseException) -> None:
    """Record the last internal failure so the diagnostics panel can show it.

    Stores the exception type and its short reason code only — never document
    text, questions, or credentials.
    """
    reason = getattr(exc, "reason", None) or str(exc)[:120]
    st.session_state["last_error"] = f"{where}: {type(exc).__name__}: {reason}"


def _record_retrieval(selected: int, valid: int, found: int, error: str = "") -> None:
    st.session_state["last_retrieval"] = {
        "selected_document_count": selected,
        "valid_document_count": valid,
        "retrieved_results_count": found,
        "error": error,
    }


def _selected_ids(document_ids) -> list[str]:
    return [d for d in (document_ids or []) if d != ALL_DOCS and _is_valid_id(d)]


def _retrieve(session_id: str, document_ids, query: str, top_k: int) -> list[dict]:
    """Top-K semantic retrieval across the selected documents.

    Raises :class:`RetrievalUnavailable` when the index cannot be read, so the
    caller can report an infrastructure problem instead of "no information".
    """
    ids = _selected_ids(document_ids)
    if not ids:
        _record_retrieval(len(document_ids or []), 0, 0)
        return []

    try:
        results = list(retrieval_service.retrieve(session_id, ids, query, top_k=top_k) or [])
    except Exception as exc:  # noqa: BLE001 - surfaced, never silently emptied
        _note_failure("retrieve", exc)
        _record_retrieval(len(ids), len(ids), 0, error=type(exc).__name__)
        raise RetrievalUnavailable(str(exc)) from exc

    _record_retrieval(len(ids), len(ids), len(results))
    return results


def _document_context(session_id: str, document_ids) -> list[dict]:
    """Ordered, bounded whole-document context for overview questions."""
    ids = _selected_ids(document_ids)
    if not ids:
        _record_retrieval(len(document_ids or []), 0, 0)
        return []

    fn = getattr(retrieval_service, "document_context", None)
    if fn is None:  # older module in memory — fall back to semantic retrieval
        return _retrieve(session_id, ids, "ملخص المستند", top_k=_top_k_max())

    try:
        results = list(fn(session_id, ids) or [])
    except Exception as exc:  # noqa: BLE001
        _note_failure("document_context", exc)
        _record_retrieval(len(ids), len(ids), 0, error=type(exc).__name__)
        raise RetrievalUnavailable(str(exc)) from exc

    _record_retrieval(len(ids), len(ids), len(results))
    return results


def _document_diagnostics(session_id: str) -> list[dict]:
    """Per-document index health. Content-free and safe to render."""
    fn = getattr(document_service, "diagnostics", None)
    if fn is None:
        return [
            {"document_id": d.document_id, "status": d.status,
             "num_pages": d.num_pages, "num_chunks": d.num_chunks}
            for d in _list_documents(session_id)
        ]
    try:
        return fn(session_id)
    except Exception as exc:  # noqa: BLE001
        _note_failure("diagnostics", exc)
        return []


def _classify(question: str) -> str:
    if _intent is None:
        return "factual"
    try:
        return _intent.classify(question)
    except Exception:  # noqa: BLE001
        return "factual"


def _llm_answer(session_id: str, question: str, results: list[dict], mode: str):
    """Call the LLM adapter, tolerating builds without the ``mode`` parameter."""
    try:
        return llm_service.answer(session_id, question, results, mode=mode)
    except TypeError:
        return llm_service.answer(session_id, question, results)


def _capabilities() -> dict[str, bool]:
    """Which module APIs are actually loaded in this process."""
    try:
        params = inspect.signature(retrieval_service.retrieve).parameters
        multi_doc = "document_ids" in params
    except (TypeError, ValueError, AttributeError):
        multi_doc = False
    return {
        "config.TOP_K_MIN": hasattr(config, "TOP_K_MIN"),
        "config.MAX_FILES_PER_SESSION": hasattr(config, "MAX_FILES_PER_SESSION"),
        "session_service.list_documents": hasattr(session_service, "list_documents"),
        "session_service.get_document": hasattr(session_service, "get_document"),
        "document_service.delete_document": hasattr(
            document_service, "delete_document"
        ),
        "retrieval_service.retrieve(multi-doc)": multi_doc,
        "retrieval_service.document_context": hasattr(
            retrieval_service, "document_context"
        ),
        "retrieval_service.verify_document_index": hasattr(
            retrieval_service, "verify_document_index"
        ),
        "core.intent": _intent is not None,
        "services.chat_service": chat_service is not None,
        "chat_service.provider_error_kind": hasattr(
            chat_service, "KIND_PROVIDER_ERROR"
        ),
        "config.openai_is_configured": hasattr(config, "openai_is_configured"),
        "services.case_analysis_service": case_analysis_service is not None,
        "llm_service.complete_json": hasattr(llm_service, "complete_json"),
        "session_service.remaining_cases": hasattr(
            session_service, "remaining_cases"
        ),
        "llm_service(OpenAI)": getattr(llm_service, "PROVIDER_NAME", "") == "OpenAI",
        "components.dashboard": hasattr(components, "dashboard"),
    }


def _llm_is_configured() -> bool:
    """True when the hosted provider has a key available on the server."""
    fn = getattr(config, "openai_is_configured", None)
    if fn is not None:
        try:
            return bool(fn())
        except Exception as exc:  # noqa: BLE001
            _note_failure("openai_is_configured", exc)

    getter = getattr(config, "get_openai_api_key", None)
    if getter is not None:
        try:
            return bool(getter())
        except Exception as exc:  # noqa: BLE001
            _note_failure("get_openai_api_key", exc)
    return False


def _llm_diagnostics() -> dict:
    """Provider status for the technical panel. NEVER includes the API key."""
    base: dict = {}
    diag_fn = getattr(config, "openai_config_diagnostics", None)
    if diag_fn is not None:
        try:
            base = diag_fn()
        except Exception as exc:  # noqa: BLE001
            _note_failure("openai_config_diagnostics", exc)

    model = base.get("openai_model")
    if not model:
        getter = getattr(config, "get_openai_model", None)
        try:
            model = getter() if getter else ""
        except Exception as exc:  # noqa: BLE001
            _note_failure("get_openai_model", exc)
            model = ""

    configured = base.get("openai_configured") or ("yes" if _llm_is_configured() else "no")
    return {
        "OpenAI configured": configured,
        "OpenAI model": model or getattr(config, "DEFAULT_OPENAI_MODEL", "غير معروف"),
        "API key detected": base.get("api_key_detected", configured),
        "API key source": base.get("api_key_source", "missing"),
        "Streamlit secrets loaded": base.get("streamlit_secrets_loaded", "unknown"),
        "Secret key names": base.get("secret_key_names", ""),
        "Configuration hint": base.get("configuration_hint", ""),
        "LLM provider": getattr(config, "LLM_PROVIDER", "OpenAI"),
        "Max output tokens": getattr(config, "OPENAI_MAX_OUTPUT_TOKENS", None),
        "Max RAG context chars": _cfg_int("MAX_RAG_CONTEXT_CHARS", 6000),
        "Retries (transient faults only)": getattr(config, "OPENAI_MAX_RETRIES", None),
    }


# --- Defensive UI helpers -------------------------------------------------
def _ui(name: str):
    return getattr(components, name, None)


def _render_dashboard(session_id: str) -> None:
    stats = _session_stats(session_id)
    remaining = _remaining_questions(session_id)

    fn = _ui("dashboard")
    if fn is not None:
        try:
            fn(stats, remaining)
            return
        except Exception:  # noqa: BLE001
            pass

    fn = _ui("session_dashboard")
    if fn is not None:
        try:
            fn(_current_document(session_id))
            return
        except Exception:  # noqa: BLE001
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد المستندات", stats.get("num_documents", 0))
    c2.metric("إجمالي الصفحات", stats.get("total_pages", 0))
    c3.metric("إجمالي المقاطع", stats.get("total_chunks", 0))
    c4.metric("الأسئلة المتبقية", remaining)


def _render_document_card(doc) -> None:
    fn = _ui("document_card")
    if fn is not None:
        try:
            fn(doc)
            return
        except Exception:  # noqa: BLE001
            pass
    st.markdown(
        f"**{doc.display_name}** — {getattr(doc, 'num_pages', 0)} صفحة · "
        f"{getattr(doc, 'num_chunks', 0)} مقطع · {getattr(doc, 'status', '')}"
    )


def _render_source(result: dict, limit: int = 400) -> None:
    fn = _ui("source_card")
    if fn is not None:
        try:
            fn(result)
        except Exception:  # noqa: BLE001
            st.markdown(f"**{result.get('document_name', 'مستند')}**")
    fn = _ui("excerpt")
    if fn is not None:
        try:
            fn(result.get("text", ""), limit=limit)
            return
        except Exception:  # noqa: BLE001
            pass
    st.text((result.get("text") or "")[:limit])


def _page_header(title: str, subtitle: str = "") -> None:
    fn = _ui("page_header")
    if fn is not None:
        try:
            fn(title, subtitle)
            return
        except Exception:  # noqa: BLE001
            pass
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


# --- State ----------------------------------------------------------------
def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = security.new_id()
        log_event("session_start", st.session_state["session_id"], status="new")
    st.session_state.setdefault("page", "documents")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("top_k", _top_k_default())
    st.session_state.setdefault("confirm_delete", None)
    # Case analysis ("تحليل حالة") state — kept separate from chat history.
    st.session_state.setdefault("case_outcome", None)
    st.session_state.setdefault("case_state", None)
    st.session_state.setdefault("case_followups", [])
    st.session_state.setdefault("case_extra_answers", "")
    session_service.get_or_create(st.session_state["session_id"])


def _sid() -> str:
    return st.session_state["session_id"]


# --- Sidebar --------------------------------------------------------------
def _sidebar() -> None:
    with st.sidebar:
        fn = _ui("brand_sidebar")
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                st.markdown("### المدقق الشامل")

        st.markdown("---")
        active = st.session_state["page"]
        for key, label in NAV:
            css = "nav-active" if key == active else ""
            st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{key}"):
                st.session_state["page"] = key
                # Keep the chat/case toggle in step with sidebar navigation so
                # it never shows a mode the user is not actually on.
                if key in {k for k, _ in _MODE_PAGES}:
                    st.session_state["interaction_mode"] = key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("إعدادات الاسترجاع"):
            # Clamp before the slider renders so an out-of-range stored value
            # can never make Streamlit raise.
            st.session_state["top_k"] = _clamp_top_k(
                st.session_state.get("top_k", _top_k_default())
            )
            st.slider(
                "عدد المقاطع المسترجعة لكل سؤال (Top-K)",
                min_value=_top_k_min(),
                max_value=_top_k_max(),
                key="top_k",
            )

        st.caption(f"الأسئلة المتبقية في الجلسة: {_remaining_questions(_sid())}")
        if case_analysis_service is not None:
            st.caption(f"تحليلات الحالة المتبقية: {_remaining_cases(_sid())}")
        fn = _ui("sidebar_privacy")
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


# --- Document selector (shared by chat + search) --------------------------
def _document_selector(ready_docs: list, key: str) -> list[str]:
    """Multiselect with a 'جميع المستندات' sentinel. Returns resolved doc ids."""
    id_to_name = {d.document_id: d.display_name for d in ready_docs}
    options = [ALL_DOCS] + list(id_to_name.keys())

    # Initialize once, then prune stale ids so the widget state stays valid.
    if key not in st.session_state:
        st.session_state[key] = [ALL_DOCS]
    pruned = [s for s in st.session_state[key] if s == ALL_DOCS or s in id_to_name]
    st.session_state[key] = pruned or [ALL_DOCS]

    def _fmt(opt: str) -> str:
        return "جميع المستندات" if opt == ALL_DOCS else id_to_name.get(opt, opt)

    selection = st.multiselect("البحث في:", options, format_func=_fmt, key=key)
    if ALL_DOCS in selection or not selection:
        return list(id_to_name.keys())
    return [s for s in selection if s in id_to_name]


_MODE_PAGES = [("chat", "سؤال عن المستند"), ("case", "تحليل حالة")]


def _apply_mode_choice() -> None:
    st.session_state["page"] = st.session_state.get("interaction_mode", "chat")


def _mode_selector(current: str) -> None:
    """Explicit choice between a direct question and a full case analysis.

    Deliberately a user-facing switch rather than automatic intent detection:
    a case analysis costs far more than a chat turn, so the user decides.

    Navigation happens in the widget's ``on_change`` callback. Comparing the
    returned value against ``current`` instead would fight Streamlit's widget
    state — the key already holds the previous page's mode on the first render
    of the new page, which would bounce the user back and forth forever.
    """
    if case_analysis_service is None:
        return
    keys = [key for key, _ in _MODE_PAGES]
    labels = dict(_MODE_PAGES)
    st.radio(
        "نوع الطلب",
        keys,
        index=keys.index(current) if current in keys else 0,
        format_func=lambda k: labels.get(k, k),
        horizontal=True,
        key="interaction_mode",
        on_change=_apply_mode_choice,
    )


# --- Pages ----------------------------------------------------------------
def page_documents() -> None:
    _page_header("المستندات", "أضف مستنداتك وأدرها مؤقتاً ضمن جلستك التجريبية.")

    live = _live_document_count(_sid())
    max_files = _max_files_per_session()
    can_add = live < max_files

    with st.expander("＋ إضافة مستند جديد", expanded=(live == 0)):
        if not can_add:
            st.info(
                f"تم الوصول إلى الحد الأقصى ({max_files} مستندات). "
                "احذف مستنداً لإضافة آخر."
            )
        st.caption(_upload_limits_caption())
        uploaded = st.file_uploader(
            "اختر ملف PDF (يمكن اختيار أكثر من ملف)",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not can_add,
            key=f"uploader_{live}",
        )
        if uploaded and st.button("رفع وفهرسة", type="primary", disabled=not can_add):
            _ingest_uploads(uploaded)

    docs = _list_documents(_sid())
    if not docs:
        st.info("لا توجد مستندات بعد. أضف مستنداً للبدء.")
        return

    st.markdown("### مستنداتك")
    for doc in docs:
        _render_document_card(doc)
        if st.session_state.get("confirm_delete") == doc.document_id:
            st.warning(f"هل أنت متأكد من حذف: {doc.display_name}؟ لا يمكن التراجع.")
            c1, c2 = st.columns(2)
            if c1.button("نعم، احذف", key=f"yes_{doc.document_id}", type="primary"):
                _delete_document(_sid(), doc.document_id)
                st.session_state["confirm_delete"] = None
                st.success("تم حذف المستند.")
                st.rerun()
            if c2.button("إلغاء", key=f"no_{doc.document_id}"):
                st.session_state["confirm_delete"] = None
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            if c1.button("فتح في المحادثة", key=f"open_{doc.document_id}"):
                st.session_state["chat_doc_selector"] = [doc.document_id]
                st.session_state["page"] = "chat"
                st.rerun()
            if c2.button("حذف", key=f"del_{doc.document_id}"):
                st.session_state["confirm_delete"] = doc.document_id
                st.rerun()


def _ingest_uploads(uploaded) -> None:
    added = 0
    for uf in uploaded:
        if not _has_document_slot(_sid()):
            st.warning(
                f"تم بلوغ الحد الأقصى ({_max_files_per_session()}). "
                "لم تتم إضافة باقي الملفات."
            )
            break
        with st.spinner(f"جاري التحقق والمعالجة: {uf.name}"):
            try:
                result = document_service.ingest(_sid(), uf.getvalue(), uf.name)
            except security.UploadRejected as exc:
                st.error(f"{uf.name}: {exc}")
                continue
            except Exception:  # noqa: BLE001 - never leak internals
                st.error(f"{uf.name}: حدث خطأ غير متوقع أثناء المعالجة.")
                continue
        st.success(
            f"تم تجهيز المستند: {result.display_name} "
            f"({result.num_pages} صفحة، {result.num_chunks} مقطع)."
        )
        added += 1
    if added:
        st.session_state["messages"] = []
        st.rerun()


def _no_documents_notice() -> None:
    st.info("لا توجد مستندات جاهزة. أضف مستنداً من قسم «المستندات» أولاً.")


def page_search() -> None:
    _page_header("البحث", "بحث دلالي مباشر داخل مستنداتك — بدون نموذج ذكاء اصطناعي خارجي.")
    ready = _ready_documents(_sid())
    if not ready:
        _no_documents_notice()
        return

    resolved = _document_selector(ready, key="search_doc_selector")

    col_q, col_n = st.columns([4, 1])
    query = col_q.text_input("كلمة أو جملة للبحث", key="search_query")
    n_results = col_n.number_input(
        "عدد النتائج",
        min_value=1,
        max_value=_search_max_results(),
        value=_search_default_results(),
        step=1,
        key="search_n",
    )

    if not query:
        return
    if not resolved:
        st.warning("يرجى اختيار مستند واحد على الأقل.")
        return

    try:
        with st.spinner("جاري البحث..."):
            results = _retrieve(_sid(), resolved, query.strip(), top_k=int(n_results))
    except RetrievalUnavailable:
        st.error(INDEX_ERROR_MESSAGE)
        return

    if not results:
        st.warning("لا توجد نتائج مطابقة.")
        return

    st.caption(f"عدد النتائج: {len(results)}")
    for r in results:
        _render_source(r, limit=500)


def page_chat() -> None:
    _page_header("المحادثة", "اسأل عن محتوى مستنداتك واحصل على إجابة بمصادرها.")
    ready = _ready_documents(_sid())
    if not ready:
        _no_documents_notice()
        return

    _mode_selector("chat")

    resolved = _document_selector(ready, key="chat_doc_selector")

    col_new, _ = st.columns([1, 5])
    if col_new.button("محادثة جديدة"):
        st.session_state["messages"] = []
        st.rerun()

    configured = _llm_is_configured()
    if not configured:
        st.warning(
            "المحادثة غير مُهيأة حالياً في هذه النسخة التجريبية. "
            "يمكنك استخدام **البحث** الذي يعمل محلياً على الخادم."
        )
        with st.expander(
            "تشخيص إعداد OpenAI (آمن — بدون عرض المفتاح)",
            expanded=True,
        ):
            diag = _llm_diagnostics()
            st.json(diag)
            hint = diag.get("Configuration hint")
            if hint:
                st.info(hint)
        st.markdown(
            "**لتفعيل المحادثة على Streamlit Cloud:**\n"
            "1. App settings → **Secrets**\n"
            "2. انسخ محتوى `.streamlit/secrets.toml.example` والصق مفتاحك\n"
            "3. **Save** ثم **Reboot app**"
        )

    _render_history()

    remaining = _remaining_questions(_sid())
    if remaining <= 0:
        st.error("تم الوصول إلى حد عدد الأسئلة في هذه الجلسة التجريبية.")

    question = st.chat_input(
        "اكتب سؤالك هنا...", disabled=(remaining <= 0 or not configured)
    )
    if not question:
        return

    if not resolved:
        st.warning("يرجى اختيار مستند واحد على الأقل قبل إرسال السؤال.")
        return

    question = question.strip()[: _cfg_int("MAX_QUESTION_CHARS", 2000)]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    mode = _classify(question)
    # Overview questions ("لخص المستند", "what is inside the pdf") read the
    # document in page order; specific questions use Top-K retrieval.
    spinner_text = (
        "جاري قراءة المستندات..." if mode == "overview" else "جاري البحث في المستندات..."
    )

    with st.chat_message("assistant"):
        with st.spinner(spinner_text):
            outcome = _chat_turn(question, resolved)

        st.session_state["last_retrieval"] = outcome["diagnostics"]
        answer_text = outcome["text"]
        results = outcome["sources"]

        if outcome["kind"] in ("index_error", "provider_error"):
            st.error(answer_text)
        else:
            st.markdown(answer_text)

        if results:
            with st.expander("عرض المقاطع المستخدمة"):
                for s in results:
                    _render_source(s)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text, "sources": results}
    )


def _chat_turn(question: str, resolved: list[str]) -> dict:
    """Run one turn via chat_service, falling back to inline logic if absent."""
    fn = getattr(chat_service, "respond", None) if chat_service else None
    if fn is not None:
        try:
            outcome = fn(
                _sid(),
                question,
                resolved,
                top_k=_clamp_top_k(st.session_state["top_k"]),
            )
            if outcome.error_reason:
                st.session_state["last_error"] = f"chat: {outcome.error_reason}"
            return {
                "kind": outcome.kind,
                "text": outcome.text,
                "sources": outcome.sources,
                "diagnostics": outcome.diagnostics,
            }
        except Exception as exc:  # noqa: BLE001
            _note_failure("chat_service", exc)

    mode = _classify(question)
    try:
        if mode == "overview":
            results = _document_context(_sid(), resolved)
        else:
            results = _retrieve(
                _sid(), resolved, question, top_k=_clamp_top_k(st.session_state["top_k"])
            )
    except RetrievalUnavailable:
        return {
            "kind": "index_error",
            "text": INDEX_ERROR_MESSAGE,
            "sources": [],
            "diagnostics": st.session_state.get("last_retrieval", {}),
        }

    if not results:
        return {
            "kind": "no_content",
            "text": NO_CONTENT_MESSAGE,
            "sources": [],
            "diagnostics": st.session_state.get("last_retrieval", {}),
        }

    try:
        result = _llm_answer(_sid(), question, results, mode)
    except Exception as exc:  # noqa: BLE001 - never leak internals
        _note_failure("llm_service", exc)
        return {
            "kind": "provider_error",
            "text": "تعذّر إنتاج الإجابة حالياً. يرجى المحاولة مرة أخرى. "
            "لم يُستهلك أي سؤال من رصيدك.",
            "sources": results,
            "diagnostics": st.session_state.get("last_retrieval", {}),
        }

    if not getattr(result, "ok", False):
        return {
            "kind": "provider_error",
            "text": result.user_message,
            "sources": results,
            "diagnostics": st.session_state.get("last_retrieval", {}),
        }

    # Quota is spent only once the provider actually returned an answer.
    session_service.record_question(_sid())
    return {
        "kind": "answer",
        "text": result.text,
        "sources": results,
        "diagnostics": st.session_state.get("last_retrieval", {}),
    }


def _render_history() -> None:
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("عرض المقاطع المستخدمة"):
                    for s in msg["sources"]:
                        _render_source(s)


# --- Case analysis ("تحليل حالة") -----------------------------------------
CASE_PLACEHOLDER_AR = (
    "اكتب تفاصيل الحالة كاملة، بما في ذلك الأطراف والوقائع والشروط وأي معلومات "
    "تعتقد أنها مهمة..."
)


def _case_progress(placeholder):
    """High-level stage progress only — never the model's internal reasoning."""
    stages = list(getattr(case_analysis_service, "STAGE_ORDER", ()))
    labels = dict(getattr(case_analysis_service, "STAGE_LABELS_AR", {}))

    def render(stage: str, label: str) -> None:
        if not stages:
            placeholder.info(f"⏳ {label}")
            return
        current = stages.index(stage) if stage in stages else 0
        lines = []
        for index, key in enumerate(stages):
            mark = "✅" if index < current else ("⏳" if index == current else "◻️")
            lines.append(f"{mark} {index + 1}. {labels.get(key, key)}")
        placeholder.markdown("  \n".join(lines))

    return render


def _render_case_evidence(evidence: list) -> None:
    """Show the chunks the analysis actually used, with their provenance."""
    labels = dict(getattr(case_models, "STRENGTH_LABELS_AR", {}))
    for item in evidence:
        strength = labels.get(getattr(item, "strength", ""), "")
        header = f"`{item.ref}` · {item.citation_ar()}"
        if strength:
            header += f" · {strength}"
        with st.expander(header):
            queries = getattr(item, "queries", [])
            if queries:
                st.caption("عُثر عليه عبر: " + " ، ".join(queries))
            st.text((item.text or "")[:900])


def _render_case_outcome(outcome) -> None:
    kind = outcome.kind
    service = case_analysis_service

    if kind == service.KIND_REPORT:
        st.markdown(outcome.report_markdown or outcome.text)

        if outcome.citations:
            st.caption(f"عدد المصادر المستشهد بها: {len(outcome.citations)}")
        if outcome.evidence:
            with st.expander(f"الأدلة المستخدمة ({len(outcome.evidence)} مقطع)"):
                _render_case_evidence(outcome.evidence)
        if outcome.queries:
            with st.expander("نقاط البحث التي استُخدمت"):
                for index, query in enumerate(outcome.queries, start=1):
                    purpose = f" — {query.purpose}" if query.purpose else ""
                    st.markdown(f"{index}. {query.text}{purpose}")
        return

    if kind == service.KIND_NEEDS_INFO:
        st.warning(service.NEEDS_INFO_MESSAGE)
        for index, missing in enumerate(outcome.missing_information, start=1):
            reason = f" — {missing.reason}" if missing.reason else ""
            st.markdown(f"{index}. {missing.question}{reason}")
        return

    if kind in (service.KIND_INDEX_ERROR, service.KIND_PROVIDER_ERROR):
        st.error(outcome.text)
        return

    st.warning(outcome.text)


def _run_case_analysis(case_text: str, resolved: list, *, force: bool) -> None:
    placeholder = st.empty()
    try:
        outcome = case_analysis_service.analyze(
            _sid(),
            case_text,
            resolved,
            additional_answers=st.session_state.get("case_extra_answers", ""),
            force_incomplete=force,
            progress=_case_progress(placeholder),
        )
    except Exception as exc:  # noqa: BLE001 - never leak internals
        placeholder.empty()
        _note_failure("case_analysis", exc)
        st.error(
            "تعذّر إكمال تحليل الحالة حالياً. يرجى المحاولة مرة أخرى. "
            "لم يُستهلك أي تحليل من رصيدك."
        )
        return

    placeholder.empty()
    st.session_state["case_outcome"] = outcome
    st.session_state["case_state"] = outcome.state
    st.session_state["case_followups"] = []
    if outcome.diagnostics:
        st.session_state["last_case_diagnostics"] = outcome.diagnostics
    if outcome.error_reason:
        st.session_state["last_error"] = f"case: {outcome.error_reason}"
    st.rerun()


def page_case() -> None:
    _page_header(
        "تحليل حالة",
        "اكتب مشكلة واقعية كاملة، ويقوم النظام بتحليلها والبحث عنها في مستنداتك.",
    )
    if case_analysis_service is None:
        st.warning(
            "ميزة تحليل الحالة غير متاحة في هذه النسخة المحمّلة على الخادم. "
            "أعد تشغيل التطبيق (Reboot app)."
        )
        err = globals().get("_case_import_error") or st.session_state.get(
            "case_import_error"
        )
        if err:
            st.session_state["case_import_error"] = err
            with st.expander("تشخيص (آمن — بدون محتوى مستندات)"):
                st.code(str(err))
        return

    _mode_selector("case")

    ready = _ready_documents(_sid())
    if not ready:
        _no_documents_notice()
        return

    resolved = _document_selector(ready, key="case_doc_selector")

    configured = _llm_is_configured()
    if not configured:
        st.warning(
            "تحليل الحالة يحتاج إلى نموذج الذكاء الاصطناعي وهو غير مُهيأ حالياً. "
            "يمكنك استخدام **البحث** الذي يعمل محلياً على الخادم."
        )

    remaining = _remaining_cases(_sid())
    st.caption(f"تحليلات الحالة المتبقية في هذه الجلسة: {remaining}")
    if remaining <= 0:
        st.error(case_analysis_service.QUOTA_MESSAGE)

    case_text = st.text_area(
        "تفاصيل الحالة أو المشكلة",
        placeholder=CASE_PLACEHOLDER_AR,
        height=220,
        max_chars=_cfg_int("MAX_CASE_CHARS", 6000),
        key="case_text",
    )

    disabled = remaining <= 0 or not configured
    col_run, col_clear = st.columns([1, 1])
    run = col_run.button("تحليل الحالة", type="primary", disabled=disabled)
    if col_clear.button("حالة جديدة"):
        st.session_state["case_outcome"] = None
        st.session_state["case_state"] = None
        st.session_state["case_followups"] = []
        st.session_state["case_extra_answers"] = ""
        st.rerun()

    if run:
        if not resolved:
            st.warning("يرجى اختيار مستند واحد على الأقل قبل تحليل الحالة.")
        elif not (case_text or "").strip():
            st.warning(case_analysis_service.NO_CASE_TEXT_MESSAGE)
        else:
            _run_case_analysis(case_text, resolved, force=False)

    outcome = st.session_state.get("case_outcome")
    if outcome is None:
        return

    st.markdown("---")
    _render_case_outcome(outcome)

    if outcome.kind == case_analysis_service.KIND_NEEDS_INFO:
        _render_missing_info_form(case_text, resolved)
        return

    if outcome.kind == case_analysis_service.KIND_REPORT:
        _render_case_followup()


def _render_missing_info_form(case_text: str, resolved: list) -> None:
    st.text_area(
        "إجاباتك على الأسئلة أعلاه (اختياري)",
        key="case_extra_answers",
        height=140,
    )
    col_again, col_force = st.columns([1, 1])
    if col_again.button("متابعة التحليل بالمعلومات الجديدة", type="primary"):
        if resolved and (case_text or "").strip():
            _run_case_analysis(case_text, resolved, force=False)
    if col_force.button("أكمل التحليل رغم نقص المعلومات"):
        if resolved and (case_text or "").strip():
            _run_case_analysis(case_text, resolved, force=True)


def _render_case_followup() -> None:
    st.markdown("### أسئلة متابعة على هذه الحالة")
    for entry in st.session_state.get("case_followups", []):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

    max_followups = _cfg_int("MAX_CASE_FOLLOWUPS_PER_CASE", 5)
    used = len(st.session_state.get("case_followups", []))
    if used >= max_followups:
        st.caption("تم الوصول إلى حد أسئلة المتابعة لهذه الحالة.")
        return

    question = st.chat_input("اسأل عن هذا التحليل...", key="case_followup_input")
    if not question:
        return

    state = st.session_state.get("case_state")
    with st.spinner("جاري إعداد الإجابة..."):
        try:
            result = case_analysis_service.follow_up(_sid(), state, question.strip())
        except Exception as exc:  # noqa: BLE001
            _note_failure("case_followup", exc)
            st.error("تعذّر إنتاج الإجابة حالياً. يرجى المحاولة مرة أخرى.")
            return

    if not result.ok:
        st.error(result.text)
        return

    st.session_state["case_followups"].append(
        {"question": question.strip(), "answer": result.text}
    )
    st.rerun()


def page_about() -> None:
    fn = _ui("hero")
    if fn is not None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    _page_header("حول النسخة التجريبية", "ماذا تُرسل ولماذا.")
    st.markdown(
        f"""
### النسخة التجريبية مقابل النسخة المكتبية

- **النسخة المكتبية (Desktop):** تعمل محلياً بالكامل على جهازك عبر محرك
  Ollama، ولا تُرسَل مستنداتك أو محادثاتك إلى أي خادم خارجي.
- **النسخة التجريبية (هذه):** مستضافة على الإنترنت لأغراض العرض. تُعالَج
  المستندات مؤقتاً على خادم العرض وتُحذف تلقائياً بعد انتهاء مدة الجلسة.

### ما الذي يُرسَل إلى مزوّد النموذج المستضاف؟

عند استخدام **المحادثة** فقط، يُرسَل إلى المزوّد المُهيّأ:
- نص سؤالك.
- الحد الأدنى من المقاطع المسترجعة ذات الصلة (Top-K) مع أرقام صفحاتها.

**لا يُرسَل** الملف الكامل، ولا بقية المستند. **البحث** يعمل محلياً بالكامل على
الخادم دون أي خدمة خارجية.

### الحدود الحالية للنسخة التجريبية

- الحجم الأقصى للملف: {_cfg_int('MAX_FILE_SIZE_MB', 50)} ميغابايت.
- الحد الأقصى للصفحات: {_cfg_int('MAX_PAGES', 200)} صفحة لكل ملف.
- عدد المستندات في الجلسة: {_max_files_per_session()}.
- الحد الأقصى للأسئلة في الجلسة: {_cfg_int('MAX_QUESTIONS_PER_SESSION', 20)}.
- الحد الأقصى لتحليلات الحالة في الجلسة: {_cfg_int('MAX_CASES_PER_SESSION', 3)}.
- مدة الجلسة قبل الحذف التلقائي: {_cfg_int('SESSION_TTL_MINUTES', 30)} دقيقة.

### مزوّد النموذج

تتم صياغة الإجابات عبر واجهة **OpenAI** البرمجية. لا يُرفع ملف PDF كاملاً إلى
المزوّد؛ يُرسَل فقط سؤالك مع مقاطع نصية محدودة الحجم مستخرجة من المستندات التي
اخترتها. الاستخراج والتقطيع والتمثيل المتجهي والبحث (FAISS) تتم كلها محلياً على
خادم النسخة التجريبية. لا تُستخدم النسخة التجريبية للمستندات الحساسة.
        """
    )
    st.caption(f"الإصدار: {getattr(config, 'DEMO_VERSION', '')}")

    with st.expander("معلومات تقنية"):
        st.caption("مزوّد النموذج اللغوي:")
        st.json(_llm_diagnostics())

        caps = _capabilities()
        st.caption("الوحدات المُحمّلة على الخادم:")
        for name, ok in caps.items():
            st.markdown(f"- {'✅' if ok else '⚠️'} `{name}`")
        if not all(caps.values()):
            st.warning(
                "بعض الوحدات قديمة في ذاكرة الخادم. أعد تشغيل التطبيق "
                "(Reboot app) لتحميل أحدث نسخة."
            )

        st.caption("حالة فهارس مستندات هذه الجلسة (بدون أي محتوى):")
        st.json(_document_diagnostics(_sid()))

        last_retrieval = st.session_state.get("last_retrieval")
        if last_retrieval:
            st.caption("آخر عملية استرجاع:")
            st.json(last_retrieval)

        last_error = st.session_state.get("last_error")
        if last_error:
            st.caption("آخر خطأ داخلي:")
            st.code(last_error)


PAGES = {
    "documents": page_documents,
    "chat": page_chat,
    "case": page_case,
    "search": page_search,
    "about": page_about,
}


def _warm_streamlit_secrets() -> None:
    """Force ``st.secrets`` to parse so Cloud secrets promote to the runtime."""
    fn = getattr(config, "bootstrap_streamlit_secrets", None)
    if fn is not None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass
    status_fn = getattr(config, "streamlit_secrets_status", None)
    if status_fn is not None:
        try:
            status_fn()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    _warm_streamlit_secrets()

    fn = getattr(config, "ensure_storage_root", None)
    if fn is not None:
        fn()

    try:
        cleanup_service.start_background_sweeper()
        cleanup_service.sweep()  # opportunistic, throttled
    except Exception:  # noqa: BLE001
        pass

    _init_state()
    _sidebar()
    session_service.touch(_sid())

    page = st.session_state["page"]
    if page != "about":
        _render_dashboard(_sid())
    PAGES.get(page, page_documents)()


main()
