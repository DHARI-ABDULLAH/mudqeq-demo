"""UI smoke tests for adding a link as a source.

These render `app.py` headlessly and drive the real widgets: type a URL, press
the button, read the rendered page, refresh, delete. Only the socket is faked,
so no page is ever really fetched; validation, extraction, indexing, listing,
and session state are the same code a user would hit.

`app.py` re-imports the demo packages on its first run, so the module objects
this file imported at collection time can go stale. Everything here therefore
reaches for the modules the *running app* is using (`_live`), and the fakes are
installed after the app has booted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests import url_fakes

APP = str(Path(__file__).resolve().parent.parent / "app.py")

ARTICLE_URL = "https://example.com/financing"
ARTICLE_TITLE = "ضوابط التمويل"
ARTICLE_BODY = """
<h1>ضوابط التمويل</h1>
<p>يشترط لصحة عقد التمويل أن يكون مكتوباً وأن تكون مدة السداد محددة بوضوح تام.</p>
<h2>الاستثناءات</h2>
<p>يجوز تمديد مدة السداد عند وجود ظرف طارئ خارج عن إرادة الطرفين بإشعار كتابي
مسبق يوضح السبب والمدة الجديدة.</p>
"""


def _live(name: str):
    """The service module the running app is actually using."""
    return sys.modules[f"services.{name}"]


def _boot() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _serve(monkeypatch, routes: dict) -> url_fakes.Router:
    """Route the app's outbound requests. Unrouted URLs fail loudly."""
    return url_fakes.install(monkeypatch, routes, live=True)


def _article_page() -> url_fakes.FakeResponse:
    return url_fakes.FakeResponse(url_fakes.html_page(ARTICLE_TITLE, ARTICLE_BODY))


def _goto(at: AppTest, label: str) -> AppTest:
    next(b for b in at.sidebar.button if b.label == label).click().run()
    return at


def _add_link(at: AppTest, url: str) -> AppTest:
    at.text_input(key="add_url_input").set_value(url).run()
    next(b for b in at.button if b.key == "add_url_btn").click().run()
    return at


def _texts(at: AppTest) -> str:
    parts = [str(m.value) for m in at.markdown]
    parts += [str(t.value) for t in at.text]
    parts += [str(i.value) for i in at.info]
    parts += [str(s.value) for s in at.success]
    parts += [str(w.value) for w in at.warning]
    parts += [str(e.value) for e in at.error]
    parts += [str(c.value) for c in at.caption]
    return " ".join(parts)


def _url_sources(at: AppTest) -> list:
    return _live("session_service").list_url_sources(at.session_state["session_id"])


@pytest.fixture
def sources_page(monkeypatch):
    """The sources page, with one article served to the app."""
    at = _goto(_boot(), "المصادر")
    router = _serve(monkeypatch, {ARTICLE_URL: _article_page()})
    return at, router


# --- The form exists and is discoverable ---------------------------------
def test_sources_page_offers_both_ways_to_add_a_source():
    at = _goto(_boot(), "المصادر")
    assert not at.exception

    assert at.text_input(key="add_url_input") is not None
    assert any(b.key == "add_url_btn" for b in at.button)
    assert any(b.label == "رفع وفهرسة" for b in at.button) or at.file_uploader


def test_the_link_form_states_its_limits():
    at = _goto(_boot(), "المصادر")
    caption = " ".join(str(c.value) for c in at.caption)
    assert "روابط" in caption
    assert "http" in caption


# --- Happy path -----------------------------------------------------------
def test_adding_a_link_indexes_it_and_lists_it(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    assert not at.exception

    sources = _url_sources(at)
    assert len(sources) == 1
    record = sources[0]
    assert record.page_title == ARTICLE_TITLE
    assert record.status == _live("session_service").STATUS_READY
    assert record.num_chunks > 0
    assert record.original_url == ARTICLE_URL

    page = _texts(at)
    assert "🔗" in page
    assert ARTICLE_TITLE in page
    assert "example.com" in page


def test_the_page_is_fetched_exactly_once(sources_page):
    at, router = sources_page
    _add_link(at, ARTICLE_URL)
    assert router.urls == [ARTICLE_URL]


def test_a_pdf_and_a_link_appear_in_the_same_list(sources_page):
    at, _ = sources_page
    sid = at.session_state["session_id"]
    from tests.pdf_util import make_pdf

    _live("document_service").ingest(
        sid, make_pdf(["Murabaha is a cost plus profit sale."]), "rules.pdf"
    )
    at = _add_link(at, ARTICLE_URL)

    page = _texts(at)
    assert "📄" in page and "🔗" in page
    assert "rules.pdf" in page and ARTICLE_TITLE in page
    assert len(_live("session_service").list_sources(sid)) == 2


def test_both_kinds_are_selectable_together_in_chat(sources_page):
    at, _ = sources_page
    sid = at.session_state["session_id"]
    from tests.pdf_util import make_pdf

    pdf = _live("document_service").ingest(
        sid, make_pdf(["Murabaha is a cost plus sale."]), "rules.pdf"
    )
    at = _add_link(at, ARTICLE_URL)
    web = _url_sources(at)[0]

    at = _goto(at, "المحادثة")
    selector = at.multiselect(key="chat_doc_selector")
    # Both kinds are offered side by side, each with its own icon.
    assert "📄 rules.pdf" in selector.options
    assert f"🔗 {ARTICLE_TITLE} — example.com" in selector.options

    selector.set_value([pdf.document_id, web.document_id]).run()
    assert not at.exception
    assert set(at.session_state["chat_doc_selector"]) == {pdf.document_id, web.document_id}


# --- Failures are reported as failures ------------------------------------
def test_an_empty_link_is_refused_gently(sources_page):
    at, _ = sources_page
    next(b for b in at.button if b.key == "add_url_btn").click().run()

    assert not at.exception
    assert "يرجى إدخال رابط" in " ".join(str(w.value) for w in at.warning)


def test_a_blocked_address_shows_a_security_message(sources_page):
    at, _ = sources_page
    at = _add_link(at, "http://127.0.0.1/admin")

    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert errors.strip()
    assert "Traceback" not in errors
    assert "لا توجد معلومات" not in errors
    assert _url_sources(at) == []


def test_a_dead_link_is_not_reported_as_an_empty_result(monkeypatch):
    at = _goto(_boot(), "المصادر")
    _serve(monkeypatch, {ARTICLE_URL: url_fakes.FakeResponse(b"not found", status_code=404)})
    at = _add_link(at, ARTICLE_URL)

    errors = " ".join(str(e.value) for e in at.error)
    assert errors.strip()
    assert "لا توجد معلومات" not in errors
    assert _url_sources(at) == []


def test_an_unreadable_page_says_so(monkeypatch):
    at = _goto(_boot(), "المصادر")
    empty = "<html><head><title>تطبيق</title></head><body><div id=root></div></body></html>"
    _serve(monkeypatch, {ARTICLE_URL: url_fakes.FakeResponse(empty)})
    at = _add_link(at, ARTICLE_URL)

    errors = " ".join(str(e.value) for e in at.error)
    assert "تعذر استخراج محتوى قابل للقراءة من هذا الرابط." in errors


def test_adding_the_same_link_twice_is_refused(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    at = _add_link(at, ARTICLE_URL)

    errors = " ".join(str(e.value) for e in at.error)
    assert "مضاف مسبقاً" in errors
    assert len(_url_sources(at)) == 1


# --- Managing an existing link --------------------------------------------
def test_a_link_source_offers_refresh_and_delete(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    source_id = _url_sources(at)[0].document_id

    labels = {b.label for b in at.button}
    assert "تحديث المحتوى" in labels
    assert "حذف" in labels
    assert any(b.key == f"refresh_{source_id}" for b in at.button)


def test_refreshing_a_link_refetches_it(sources_page, monkeypatch):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    sid = at.session_state["session_id"]
    record = _url_sources(at)[0]
    source_id = record.document_id
    fetched_at = record.retrieved_at

    updated = ARTICLE_BODY.replace(
        "مدة السداد محددة بوضوح تام", "مدة السداد الجديدة ستة وثلاثين شهراً"
    )
    router = _serve(
        monkeypatch,
        {ARTICLE_URL: url_fakes.FakeResponse(url_fakes.html_page(ARTICLE_TITLE, updated))},
    )
    next(b for b in at.button if b.key == f"refresh_{source_id}").click().run()

    assert not at.exception
    assert router.urls == [ARTICLE_URL], "refresh must re-fetch the page"

    refreshed = _live("session_service").get_document(sid, source_id)
    assert refreshed is not None
    assert refreshed.retrieved_at >= fetched_at

    hits = _live("retrieval_service").retrieve(sid, [source_id], "مدة السداد الجديدة", top_k=3)
    assert any("ستة وثلاثين شهراً" in h["text"] for h in hits)


def test_deleting_a_link_removes_it_from_the_list(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    source_id = _url_sources(at)[0].document_id

    next(b for b in at.button if b.key == f"del_{source_id}").click().run()
    next(b for b in at.button if b.key == f"yes_{source_id}").click().run()

    assert not at.exception
    assert _url_sources(at) == []
    assert ARTICLE_TITLE not in _texts(at)


# --- Other pages see the new source --------------------------------------
def test_search_page_can_target_a_link_source(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)
    web = _url_sources(at)[0]

    at = _goto(at, "البحث")
    assert not at.exception
    selector = at.multiselect(key="search_doc_selector")
    assert f"🔗 {ARTICLE_TITLE} — example.com" in selector.options

    selector.set_value([web.document_id]).run()
    at.text_input(key="search_query").set_value("تمديد مدة السداد").run()
    assert not at.exception
    page = _texts(at)
    assert "ظرف طارئ" in page, "a web result should be searchable and shown"
    assert ARTICLE_URL in page, "the result links back to the page it came from"


def test_case_page_can_target_a_link_source(sources_page):
    at, _ = sources_page
    at = _add_link(at, ARTICLE_URL)

    at = _goto(at, "تحليل حالة")
    assert not at.exception
    assert f"🔗 {ARTICLE_TITLE} — example.com" in at.multiselect(key="case_doc_selector").options
