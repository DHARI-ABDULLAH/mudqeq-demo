"""
web_demo/app.py
---------------
Public web demo of "المدقق الشامل" (Mudqeq AI).

Streamlit entry point. Wires the isolated demo services together:

    consent -> upload (validated, temporary, per-session)
            -> extract -> chunk -> embed -> FAISS (per session)
            -> search (local, no LLM) / chat (Groq hosted LLM)

The desktop application is NOT imported or affected by this module.
"""

from __future__ import annotations

import streamlit as st

import config
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
    ("home", "الرئيسية"),
    ("upload", "رفع المستند"),
    ("chat", "المحادثة"),
    ("search", "البحث"),
    ("about", "حول النسخة التجريبية"),
]


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = security.new_id()
        log_event("session_start", st.session_state["session_id"], status="new")
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("consent", False)
    st.session_state.setdefault("messages", [])
    session_service.get_or_create(st.session_state["session_id"])


def _sid() -> str:
    return st.session_state["session_id"]


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
        remaining = session_service.remaining_questions(_sid())
        st.caption(f"الأسئلة المتبقية في الجلسة: {remaining}")
        components.sidebar_privacy()


# --- Pages ----------------------------------------------------------------
def page_home() -> None:
    components.hero()
    doc = session_service.current_document(_sid())
    components.page_header("لوحة الجلسة", "معلومات خاصة بجلستك الحالية فقط.")
    components.session_dashboard(doc)

    st.markdown("### كيف تعمل النسخة التجريبية؟")
    st.markdown(
        "- ارفع ملف PDF واحداً (بحد أقصى "
        f"{config.MAX_FILE_SIZE_MB} ميغابايت و{config.MAX_PAGES} صفحة).\n"
        "- يُستخرج النص ويُقسَّم ويُفهرس مؤقتاً على خادم العرض.\n"
        "- في **البحث** يبقى كل شيء على الخادم (بدون نموذج ذكاء اصطناعي خارجي).\n"
        "- في **المحادثة** يُرسَل سؤالك والمقاطع ذات الصلة فقط إلى مزوّد النموذج "
        "المستضاف لإنتاج الإجابة.\n"
        "- يُحذف المستند تلقائياً بعد انتهاء مدة الجلسة أو عند الضغط على «حذف المستند»."
    )
    if not config.groq_is_configured():
        st.info(
            "المحادثة غير مُهيأة حالياً (لا يوجد مفتاح للنموذج المستضاف). "
            "يمكنك تجربة **البحث** الذي يعمل محلياً على الخادم دون أي خدمة خارجية."
        )


def _consent_gate() -> bool:
    st.markdown(
        f"""
        <div class="privacy-box">
          <b>تنبيه الخصوصية:</b> هذه نسخة تجريبية <b>تعمل عبر الإنترنت</b>.
          عند استخدام المحادثة، يتم إرسال السؤال والمقاطع ذات الصلة اللازمة
          لإنتاج الإجابة إلى مزوّد نموذج الذكاء الاصطناعي.
          <b>لا تستخدم هذه النسخة للمستندات السرية أو الحساسة.</b>
          للاستخدام المحلي والخاص، استخدم نسخة Desktop.
        </div>
        """,
        unsafe_allow_html=True,
    )
    agreed = st.checkbox(
        "قرأت وأوافق على استخدام النسخة التجريبية للمستندات غير الحساسة.",
        value=st.session_state.get("consent", False),
        key="consent_checkbox",
    )
    st.session_state["consent"] = agreed
    return agreed


def page_upload() -> None:
    components.page_header("رفع المستند", "ملف PDF واحد، معالجة مؤقتة على الخادم.")
    agreed = _consent_gate()

    doc = session_service.current_document(_sid())
    if doc is not None:
        st.success(f"المستند الحالي: {doc.display_name} — {doc.num_pages} صفحة، {doc.num_chunks} مقطع.")
        if st.button("حذف المستند", type="secondary"):
            document_service.delete_current(_sid())
            st.session_state["messages"] = []
            st.rerun()
        st.info("لرفع مستند آخر، احذف المستند الحالي أولاً (النسخة التجريبية تدعم مستنداً واحداً).")
        return

    uploaded = st.file_uploader(
        "اختر ملف PDF",
        type=["pdf"],
        accept_multiple_files=False,
        disabled=not agreed,
        key="uploader",
    )
    if not agreed:
        st.warning("يرجى الموافقة على تنبيه الخصوصية أولاً لتفعيل الرفع.")
        return

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        st.caption(f"الحجم: {size_mb:.1f} ميغابايت")
        if st.button("رفع وفهرسة", type="primary"):
            if not session_service.has_document_slot(_sid()):
                st.error("النسخة التجريبية تدعم مستنداً واحداً في الجلسة.")
                return
            with st.spinner("جاري التحقق من الملف ومعالجته..."):
                try:
                    result = document_service.ingest(
                        _sid(), uploaded.getvalue(), uploaded.name
                    )
                except security.UploadRejected as exc:
                    st.error(str(exc))
                    return
                except Exception:  # noqa: BLE001 - never leak internals
                    st.error("حدث خطأ غير متوقع أثناء المعالجة.")
                    return
            st.session_state["messages"] = []
            st.success(
                f"{result.message} ({result.num_pages} صفحة، {result.num_chunks} مقطع)."
            )
            st.rerun()


def _require_document() -> bool:
    if session_service.current_document(_sid()) is None:
        st.info("لا يوجد مستند بعد. انتقل إلى «رفع المستند» لإضافة ملف PDF.")
        return False
    return True


def page_search() -> None:
    components.page_header("البحث", "بحث دلالي محلي داخل مستندك — بدون نموذج خارجي.")
    if not _require_document():
        return
    doc = session_service.current_document(_sid())
    query = st.text_input("اكتب عبارة البحث:", key="search_query")
    if query:
        with st.spinner("جاري البحث..."):
            results = retrieval_service.retrieve(
                _sid(), doc.document_id, query.strip(), top_k=config.TOP_K
            )
        if not results:
            st.warning("لا توجد نتائج مطابقة.")
            return
        for r in results:
            components.source_card(r)
            components.excerpt(r.get("text", ""))


def page_chat() -> None:
    components.page_header("المحادثة", "اسأل عن محتوى مستندك واحصل على إجابة باقتباساتها.")
    if not _require_document():
        return
    doc = session_service.current_document(_sid())

    if not config.groq_is_configured():
        st.warning(
            "المحادثة غير مُهيأة حالياً في هذه النسخة التجريبية. "
            "يمكنك استخدام **البحث** الذي يعمل محلياً على الخادم."
        )

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("عرض المقاطع المستخدمة"):
                    for s in msg["sources"]:
                        components.source_card(s)
                        components.excerpt(s.get("text", ""))

    remaining = session_service.remaining_questions(_sid())
    disabled = remaining <= 0 or not config.groq_is_configured()
    if remaining <= 0:
        st.error("تم الوصول إلى حد عدد الأسئلة في هذه الجلسة التجريبية.")

    question = st.chat_input("اكتب سؤالك هنا...", disabled=disabled)
    if not question:
        return

    question = question.strip()[: config.MAX_QUESTION_CHARS]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في المستند..."):
            results = retrieval_service.retrieve(
                _sid(), doc.document_id, question, top_k=config.TOP_K
            )
        if not results:
            answer_text = "لم أجد في المستند معلومات كافية للإجابة."
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


def page_about() -> None:
    components.page_header("حول النسخة التجريبية", "ماذا ترسل ولماذا.")
    st.markdown(
        f"""
### النسخة التجريبية مقابل النسخة المكتبية

- **النسخة المكتبية (Desktop):** تعمل محلياً بالكامل على جهازك عبر محرك
  Ollama، ولا تُرسَل مستنداتك أو محادثاتك إلى أي خادم خارجي.
- **النسخة التجريبية (هذه):** مستضافة على الإنترنت لأغراض العرض. تُعالَج
  المستندات مؤقتاً على خادم العرض وتُحذف تلقائياً.

### ما الذي يُرسَل إلى مزوّد النموذج المستضاف؟

عند استخدام **المحادثة** فقط، يُرسَل إلى المزوّد المُهيّأ:
- نص سؤالك.
- الحد الأدنى من المقاطع المسترجعة ذات الصلة (Top-K) مع أرقام صفحاتها.

**لا يُرسَل** الملف الكامل، ولا بقية المستند.

### الحدود الحالية للنسخة التجريبية

- الحجم الأقصى للملف: {config.MAX_FILE_SIZE_MB} ميغابايت.
- الحد الأقصى للصفحات: {config.MAX_PAGES} صفحة.
- عدد المستندات في الجلسة: {config.MAX_FILES_PER_SESSION}.
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
    "home": page_home,
    "upload": page_upload,
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
    PAGES.get(st.session_state["page"], page_home)()


main()
