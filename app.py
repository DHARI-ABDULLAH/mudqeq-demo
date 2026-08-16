"""
web_demo/app.py
---------------
Public web demo of "المدقق الشامل" (Mudqeq AI).

Streamlit entry point. Wires the isolated demo services together and mirrors
the desktop app's UX (multi-document management, selection, search, chat):

    upload (validated, temporary, per-session, MANY docs)
      -> extract -> chunk -> embed -> FAISS (per document)
      -> select documents -> search (local) / chat (Groq hosted LLM)

The desktop application is NOT imported or affected by this module.
"""

from __future__ import annotations

import streamlit as st

import config
import app_config
from core.logging_utils import log_event
from services import (
    cleanup_service,
    document_service,
    llm_service,
    retrieval_service,
    security,
    session_service,
)
from ui import components, styles

st.set_page_config(
    page_title="المدقق الشامل — نسخة تجريبية",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV = [
    ("chat", "المحادثة"),
    ("documents", "المستندات"),
    ("search", "البحث"),
    ("about", "حول النسخة التجريبية"),
]

ALL_DOCS = "__all__"


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = security.new_id()
        log_event("session_start", st.session_state["session_id"], status="new")
    st.session_state.setdefault("page", "documents")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("top_k", app_config.top_k_default())
    st.session_state.setdefault("confirm_delete", None)
    session_service.get_or_create(st.session_state["session_id"])


def _sid() -> str:
    return st.session_state["session_id"]


# --- Sidebar --------------------------------------------------------------
def _sidebar() -> None:
    with st.sidebar:
        components.brand_sidebar()
        st.markdown("---")
        active = st.session_state["page"]
        for key, label in NAV:
            css = "nav-active" if key == active else ""
            st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{key}"):
                st.session_state["page"] = key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("إعدادات الاسترجاع"):
            # Clamp any pre-existing value into the valid range before the
            # slider renders (guards against env/config changes across reruns).
            current = st.session_state.get("top_k", app_config.top_k_default())
            st.session_state["top_k"] = app_config.clamp_top_k(current)
            st.slider(
                "عدد المقاطع المسترجعة لكل سؤال (Top-K)",
                min_value=app_config.top_k_min(),
                max_value=app_config.top_k_max(),
                key="top_k",
            )

        remaining = session_service.remaining_questions(_sid())
        st.caption(f"الأسئلة المتبقية في الجلسة: {remaining}")
        components.sidebar_privacy()


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


# --- Pages ----------------------------------------------------------------
def page_documents() -> None:
    components.page_header(
        "المستندات", "أضف مستنداتك وأدرها مؤقتاً ضمن جلستك التجريبية."
    )

    live = session_service.live_document_count(_sid())
    can_add = live < app_config.max_files_per_session()

    with st.expander("＋ إضافة مستند جديد", expanded=(live == 0)):
        if not can_add:
            st.info(
                f"تم الوصول إلى الحد الأقصى ({app_config.max_files_per_session()} مستندات). "
                "احذف مستنداً لإضافة آخر."
            )
        st.caption(
            f"PDF فقط · بحد أقصى {config.MAX_FILE_SIZE_MB} ميغابايت و"
            f"{config.MAX_PAGES} صفحة لكل ملف."
        )
        uploaded = st.file_uploader(
            "اختر ملف PDF (يمكن اختيار أكثر من ملف)",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not can_add,
            key=f"uploader_{live}",
        )
        if uploaded and st.button("رفع وفهرسة", type="primary", disabled=not can_add):
            _ingest_uploads(uploaded)

    docs = session_service.list_documents(_sid())
    if not docs:
        st.info("لا توجد مستندات بعد. أضف مستنداً للبدء.")
        return

    st.markdown("### مستنداتك")
    for doc in docs:
        components.document_card(doc)
        if st.session_state.get("confirm_delete") == doc.document_id:
            st.warning(f"هل أنت متأكد من حذف: {doc.display_name}؟ لا يمكن التراجع.")
            c1, c2 = st.columns(2)
            if c1.button("نعم، احذف", key=f"yes_{doc.document_id}", type="primary"):
                document_service.delete_document(_sid(), doc.document_id)
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
    added, skipped = 0, 0
    for uf in uploaded:
        if not session_service.has_document_slot(_sid()):
            st.warning(
                f"تم بلوغ الحد الأقصى ({app_config.max_files_per_session()}). "
                "لم تتم إضافة باقي الملفات."
            )
            break
        with st.spinner(f"جاري التحقق والمعالجة: {uf.name}"):
            try:
                result = document_service.ingest(_sid(), uf.getvalue(), uf.name)
            except security.UploadRejected as exc:
                st.error(f"{uf.name}: {exc}")
                skipped += 1
                continue
            except Exception:  # noqa: BLE001 - never leak internals
                st.error(f"{uf.name}: حدث خطأ غير متوقع أثناء المعالجة.")
                skipped += 1
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
    components.page_header(
        "البحث", "بحث دلالي مباشر داخل مستنداتك — بدون نموذج ذكاء اصطناعي خارجي."
    )
    ready = session_service.ready_documents(_sid())
    if not ready:
        _no_documents_notice()
        return

    resolved = _document_selector(ready, key="search_doc_selector")

    col_q, col_n = st.columns([4, 1])
    query = col_q.text_input("كلمة أو جملة للبحث", key="search_query")
    n_results = col_n.number_input(
        "عدد النتائج",
        min_value=1,
        max_value=app_config.search_max_results(),
        value=app_config.search_default_results(),
        step=1,
        key="search_n",
    )

    if not query:
        return
    if not resolved:
        st.warning("يرجى اختيار مستند واحد على الأقل.")
        return

    with st.spinner("جاري البحث..."):
        results = retrieval_service.retrieve(
            _sid(), resolved, query.strip(), top_k=int(n_results)
        )
    if not results:
        st.warning("لا توجد نتائج مطابقة.")
        return

    st.caption(f"عدد النتائج: {len(results)}")
    for r in results:
        components.source_card(r)
        components.excerpt(r.get("text", ""), limit=500)


def page_chat() -> None:
    components.page_header(
        "المحادثة", "اسأل عن محتوى مستنداتك واحصل على إجابة بمصادرها."
    )
    ready = session_service.ready_documents(_sid())
    if not ready:
        _no_documents_notice()
        return

    resolved = _document_selector(ready, key="chat_doc_selector")

    col_new, _ = st.columns([1, 5])
    if col_new.button("محادثة جديدة"):
        st.session_state["messages"] = []
        st.rerun()

    if not config.groq_is_configured():
        st.warning(
            "المحادثة غير مُهيأة حالياً في هذه النسخة التجريبية. "
            "يمكنك استخدام **البحث** الذي يعمل محلياً على الخادم."
        )

    _render_history()

    remaining = session_service.remaining_questions(_sid())
    if remaining <= 0:
        st.error("تم الوصول إلى حد عدد الأسئلة في هذه الجلسة التجريبية.")

    disabled = remaining <= 0 or not config.groq_is_configured()
    question = st.chat_input("اكتب سؤالك هنا...", disabled=disabled)
    if not question:
        return

    if not resolved:
        st.warning("يرجى اختيار مستند واحد على الأقل قبل إرسال السؤال.")
        return

    question = question.strip()[: config.MAX_QUESTION_CHARS]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في المستندات..."):
            results = retrieval_service.retrieve(
                _sid(), resolved, question, top_k=st.session_state["top_k"]
            )
        if not results:
            answer_text = "لم أجد في المستندات المحددة معلومات كافية للإجابة."
            st.markdown(answer_text)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer_text, "sources": []}
            )
            return

        session_service.record_question(_sid())
        with st.spinner("جاري توليد الإجابة..."):
            result = llm_service.answer(_sid(), question, results)

        answer_text = result.user_message
        st.markdown(answer_text)
        with st.expander("عرض المقاطع المستخدمة"):
            for s in results:
                components.source_card(s)
                components.excerpt(s.get("text", ""))

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text, "sources": results}
    )


def _render_history() -> None:
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("عرض المقاطع المستخدمة"):
                    for s in msg["sources"]:
                        components.source_card(s)
                        components.excerpt(s.get("text", ""))


def page_about() -> None:
    components.hero()
    components.page_header("حول النسخة التجريبية", "ماذا تُرسل ولماذا.")
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

- الحجم الأقصى للملف: {config.MAX_FILE_SIZE_MB} ميغابايت.
- الحد الأقصى للصفحات: {config.MAX_PAGES} صفحة لكل ملف.
- عدد المستندات في الجلسة: {app_config.max_files_per_session()}.
- الحد الأقصى للأسئلة في الجلسة: {config.MAX_QUESTIONS_PER_SESSION}.
- مدة الجلسة قبل الحذف التلقائي: {config.SESSION_TTL_MINUTES} دقيقة.

### مزوّد النموذج

يستخدم هذا العرض نموذجاً مستضافاً عبر واجهة برمجية. سياسات الاحتفاظ بالبيانات
تخضع لإعدادات حساب المزوّد المُستخدَم في النشر؛ لا تُستخدم النسخة التجريبية
للمستندات الحساسة.
        """
    )
    st.caption(f"الإصدار: {config.DEMO_VERSION}")


PAGES = {
    "documents": page_documents,
    "chat": page_chat,
    "search": page_search,
    "about": page_about,
}


def main() -> None:
    styles.inject()
    config.ensure_storage_root()
    cleanup_service.start_background_sweeper()
    cleanup_service.sweep()  # opportunistic, throttled
    _init_state()
    _sidebar()
    session_service.touch(_sid())

    page = st.session_state["page"]
    if page != "about":
        components.dashboard(
            session_service.stats(_sid()),
            session_service.remaining_questions(_sid()),
        )
    PAGES.get(page, page_documents)()


main()
