"""Readable-content extraction and bounded fetching.

Two separate promises are pinned here:

1. What comes OUT of a page is what a reader would read — headings, paragraphs,
   lists, tables — and never scripts, styles, navigation, cookie banners, or
   hidden elements. Anything indexed is eligible to be cited, so boilerplate in
   the index is boilerplate in someone's citation.
2. What comes IN is bounded — size, timeout, redirects, content type — and each
   failure is reported as the infrastructure failure it is, never as "no
   information found".
"""

from __future__ import annotations

import pytest
import requests

from core import html_extract
from core.chunking import build_url_chunks
from services import url_fetch_service
from tests import url_fakes

ARABIC_BODY = """
<h1>ضوابط التمويل</h1>
<p>يشترط لصحة التمويل أن يكون العقد مكتوباً وأن تكون المدة محددة بوضوح تام.</p>
<h2>الشروط الأساسية</h2>
<ul>
  <li>أن يكون المبلغ معلوماً عند التعاقد</li>
  <li>أن تكون المدة محددة في العقد</li>
</ul>
<h2>الاستثناءات</h2>
<p>يجوز تمديد المدة عند وجود ظرف طارئ خارج عن إرادة الطرفين مع إشعار كتابي.</p>
<table>
  <tr><th>النوع</th><th>المدة القصوى</th></tr>
  <tr><td>تمويل قصير</td><td>سنة واحدة</td></tr>
</table>
"""

ENGLISH_BODY = """
<h1>Financing Rules</h1>
<p>A financing contract must be written and must state a clearly defined term.</p>
<h2>Core Conditions</h2>
<ul>
  <li>The amount must be known at contract time</li>
  <li>The term must be stated in the contract</li>
</ul>
"""


# --- Title ----------------------------------------------------------------
def test_title_is_extracted_from_the_title_tag():
    page = url_fakes.html_page("ضوابط التمويل — الموقع", ARABIC_BODY)
    result = html_extract.extract(page)
    assert result.title == "ضوابط التمويل — الموقع"


def test_open_graph_title_wins_over_the_title_tag():
    page = """<html><head><title>Site name</title>
    <meta property="og:title" content="The Real Article Title"></head>
    <body><main><p>Some body text that is long enough to matter here.</p></main></body></html>"""
    assert html_extract.extract(page).title == "The Real Article Title"


def test_h1_is_used_when_no_title_tag_exists():
    page = "<html><body><main><h1>عنوان من الترويسة</h1><p>نص كافٍ للاختبار هنا.</p></main></body></html>"
    assert html_extract.extract(page).title == "عنوان من الترويسة"


# --- Arabic + English content ---------------------------------------------
def test_arabic_html_content_is_extracted():
    result = html_extract.extract(url_fakes.html_page("ضوابط التمويل", ARABIC_BODY))
    text = result.text
    assert "يشترط لصحة التمويل أن يكون العقد مكتوباً" in text
    assert "أن يكون المبلغ معلوماً عند التعاقد" in text
    assert "يجوز تمديد المدة عند وجود ظرف طارئ" in text


def test_english_html_content_is_extracted():
    result = html_extract.extract(url_fakes.html_page("Financing Rules", ENGLISH_BODY, lang="en"))
    text = result.text
    assert "A financing contract must be written" in text
    assert "The amount must be known at contract time" in text


def test_headings_become_sections_with_their_paragraphs():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    headings = [s.heading for s in result.sections]
    assert "الشروط الأساسية" in headings
    assert "الاستثناءات" in headings

    exceptions = next(s for s in result.sections if s.heading == "الاستثناءات")
    assert "ظرف طارئ" in exceptions.text
    assert "المبلغ معلوماً" not in exceptions.text, "sections must not bleed into each other"


def test_list_items_are_retained_individually():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    blocks = [b for s in result.sections for b in s.blocks]
    assert "أن يكون المبلغ معلوماً عند التعاقد" in blocks
    assert "أن تكون المدة محددة في العقد" in blocks


def test_tables_are_flattened_into_readable_rows():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    text = result.text
    assert "النوع | المدة القصوى" in text
    assert "تمويل قصير | سنة واحدة" in text


# --- Boilerplate removal --------------------------------------------------
def test_scripts_and_styles_are_never_indexed():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    text = result.text
    assert "should never be indexed" not in text
    assert "window.tracker" not in text
    assert "font-family" not in text


def test_navigation_cookie_banner_ads_and_footer_are_dropped():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    text = result.text
    for noise in (
        "ملفات تعريف الارتباط",
        "إعلان ممول",
        "جميع الحقوق محفوظة",
        "مقالات ذات صلة",
        "يرجى تفعيل الجافاسكربت",
    ):
        assert noise not in text, f"boilerplate leaked into the index: {noise}"


def test_hidden_elements_are_dropped():
    result = html_extract.extract(url_fakes.html_page("ضوابط", ARABIC_BODY))
    assert "نص مخفي" not in result.text

    aria = """<html><body><main>
      <p>نص ظاهر ومقروء بشكل طبيعي تماماً.</p>
      <p aria-hidden="true">نص مخفي عن قارئ الشاشة</p>
      <p hidden>نص محجوب بالسمة hidden</p>
    </main></body></html>"""
    text = html_extract.extract(aria).text
    assert "نص ظاهر" in text
    assert "قارئ الشاشة" not in text
    assert "محجوب بالسمة" not in text


def test_boilerplate_nested_inside_boilerplate_is_handled():
    """Removing a wrapper also removes its children — they must not be revisited.

    Real pages nest chrome inside chrome constantly; visiting an already
    removed descendant afterwards used to abort the whole ingest.
    """
    page = """<html><body>
      <div class="sidebar">
        <div class="menu"><span hidden>مخفي</span><p>عناصر تنقل</p></div>
        <div class="advert"><div class="promo">إعلان داخل إعلان</div></div>
      </div>
      <main><p>النص الحقيقي للمقالة، وهو ما نريد فهرسته دون سواه.</p></main>
    </body></html>"""
    result = html_extract.extract(page)
    assert "النص الحقيقي للمقالة" in result.text
    assert "عناصر تنقل" not in result.text
    assert "إعلان داخل إعلان" not in result.text


def test_a_chrome_class_on_the_document_root_does_not_erase_the_page():
    """<html>/<body> carry site-wide feature flags; they are not chrome.

    Wikipedia's <html> advertises "…-main-menu-disabled", which used to match
    the "menu" hint and delete the entire document.
    """
    page = """<html class="client-nojs vector-feature-main-menu-disabled">
      <body class="mediawiki page-Finance">
        <main><h1>تمويل</h1>
        <p>التمويل هو عملية توفير الموارد اللازمة لتمويل مشروع أو احتياج معيّن.</p>
        </main>
      </body></html>"""
    result = html_extract.extract(page)
    assert "التمويل هو عملية توفير الموارد" in result.text
    assert result.title == "تمويل"


def test_a_text_rich_page_never_extracts_to_nothing():
    """An over-eager strip falls back rather than claiming the page is empty."""
    body = "".join(
        f"<p>فقرة رقم {i} تحمل نصاً حقيقياً كافياً للقراءة والفهرسة معاً.</p>"
        for i in range(8)
    )
    # The article sits inside a wrapper whose class matches a chrome hint.
    page = f"<html><body><div class='sidebar-content'>{body}</div></body></html>"
    result = html_extract.extract(page)
    assert result.has_usable_text()
    assert "فقرة رقم 3" in result.text


def test_a_genuinely_empty_page_still_reports_empty():
    """The fallback must not resurrect pages that really have no text."""
    page = "<html><body><div class='sidebar'><span>×</span></div></body></html>"
    assert html_extract.extract(page).has_usable_text() is False


def test_repeated_boilerplate_lines_are_collapsed():
    page = "<html><body><main>" + ("<p>اقرأ المزيد عن هذا الموضوع الآن</p>" * 5) + (
        "<p>نص فريد يجب أن يبقى موجوداً في الفهرس.</p>"
    ) + "</main></body></html>"
    text = html_extract.extract(page).text
    assert text.count("اقرأ المزيد عن هذا الموضوع الآن") == 1
    assert "نص فريد" in text


# --- Pages with nothing readable ------------------------------------------
def test_javascript_only_page_yields_no_readable_text():
    page = """<html><head><title>تطبيق</title></head>
    <body><div id="root"></div>
    <script>document.getElementById('root').innerHTML = 'محتوى يظهر بالجافاسكربت';</script>
    </body></html>"""
    result = html_extract.extract(page)
    assert result.has_usable_text() is False
    assert "الجافاسكربت" not in result.text


def test_empty_html_is_handled():
    assert html_extract.extract("").has_usable_text() is False
    assert html_extract.extract("   ").has_usable_text() is False


# --- Bounds ---------------------------------------------------------------
def test_extraction_is_bounded_by_max_chars():
    body = url_fakes.long_body("فقرة طويلة جداً تتكرر كثيراً لاختبار الحد الأقصى.", times=200)
    result = html_extract.extract(url_fakes.html_page("طويلة", body), max_chars=500)
    assert result.total_chars <= 500 + 200  # headings + join separators
    assert result.truncated is True


# --- Chunking web sections ------------------------------------------------
def _sections_of(page: str):
    return html_extract.extract(page).sections


def test_a_heading_with_no_text_never_becomes_a_chunk():
    """An outline heading carries no information but competes for retrieval."""
    page = """<html><body><main>
      <h2>أنواع التمويل</h2>
      <h3>التمويل الشخصي</h3>
      <p>يتضمن التمويل الشخصي استخدام الموارد المالية الشخصية لتمويل مشروع معين.</p>
    </main></body></html>"""
    chunks = build_url_chunks(_sections_of(page), "src-1", "example.com", "https://example.com/x")

    assert chunks
    assert all(c["text"].strip() != "أنواع التمويل" for c in chunks)
    # The parent heading survives as breadcrumb context on the real chunk.
    body_chunk = chunks[0]
    assert "أنواع التمويل" in body_chunk["text"]
    assert "التمويل الشخصي" in body_chunk["text"]
    assert body_chunk["section_title"] == "التمويل الشخصي"


def test_an_outline_only_page_is_still_indexed():
    page = "<html><body><main><h2>الفصل الأول</h2><h2>الفصل الثاني</h2></main></body></html>"
    chunks = build_url_chunks(_sections_of(page), "src-1", "example.com", "https://example.com/x")
    assert len(chunks) == 1
    assert "الفصل الأول" in chunks[0]["text"]


def test_url_chunks_never_span_two_sections():
    page = """<html><body><main>
      <h2>الشروط</h2><p>الشرط الأول هو أن يكون العقد مكتوباً وواضحاً تماماً.</p>
      <h2>الاستثناءات</h2><p>يستثنى من ذلك حالة الظرف الطارئ الخارج عن الإرادة.</p>
    </main></body></html>"""
    chunks = build_url_chunks(_sections_of(page), "src-1", "example.com", "https://example.com/x")
    for chunk in chunks:
        assert not ("الشرط الأول" in chunk["text"] and "الظرف الطارئ" in chunk["text"])


def test_plain_text_extraction_keeps_lines():
    result = html_extract.extract_plain_text("سطر أول\n\nسطر ثانٍ\nسطر ثالث")
    assert result.sections
    blocks = result.sections[0].blocks
    assert blocks == ["سطر أول", "سطر ثانٍ", "سطر ثالث"]


# --- Fetch: content types -------------------------------------------------
def test_html_content_type_is_accepted(monkeypatch):
    page = url_fakes.html_page("عنوان", "<p>" + "نص كافٍ للاختبار. " * 20 + "</p>")
    url_fakes.install(monkeypatch, {"https://example.com/a": url_fakes.FakeResponse(page)})
    result = url_fetch_service.fetch("https://example.com/a")
    assert result.content_type == "text/html"
    assert result.is_html is True


def test_plain_text_content_type_is_accepted(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {
            "https://example.com/a.txt": url_fakes.FakeResponse(
                "نص عادي طويل بما يكفي. " * 20, content_type="text/plain; charset=utf-8"
            )
        },
    )
    assert url_fetch_service.fetch("https://example.com/a.txt").content_type == "text/plain"


@pytest.mark.parametrize(
    "content_type",
    ["image/png", "video/mp4", "audio/mpeg", "application/zip", "application/octet-stream"],
)
def test_binary_content_types_are_rejected(content_type, monkeypatch):
    url_fakes.install(
        monkeypatch,
        {"https://example.com/f": url_fakes.FakeResponse(b"\x00\x01\x02", content_type=content_type)},
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/f")
    assert exc.value.reason == url_fetch_service.REASON_CONTENT_TYPE


def test_pdf_url_gets_its_own_guidance(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {"https://example.com/doc.pdf": url_fakes.FakeResponse(b"%PDF-1.4", content_type="application/pdf")},
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/doc.pdf")
    assert exc.value.reason == url_fetch_service.REASON_PDF_URL
    assert "رفع ملف" in exc.value.user_message


# --- Fetch: size ----------------------------------------------------------
def test_oversized_response_is_rejected_while_streaming(monkeypatch):
    """The cap is enforced on bytes read, not on a header the server controls."""
    huge = b"x" * 200_000
    url_fakes.install(
        monkeypatch,
        {"https://example.com/big": url_fakes.FakeResponse(huge, headers={"Content-Length": "10"})},
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/big", max_bytes=50_000)
    assert exc.value.reason == url_fetch_service.REASON_TOO_LARGE


def test_declared_oversize_is_rejected_before_reading(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {
            "https://example.com/big": url_fakes.FakeResponse(
                b"small", headers={"Content-Length": str(10_000_000)}
            )
        },
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/big", max_bytes=50_000)
    assert exc.value.reason == url_fetch_service.REASON_TOO_LARGE


def test_empty_body_is_reported(monkeypatch):
    url_fakes.install(monkeypatch, {"https://example.com/e": url_fakes.FakeResponse(b"")})
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/e")
    assert exc.value.reason == url_fetch_service.REASON_EMPTY_BODY


# --- Fetch: transport failures --------------------------------------------
def test_timeout_is_reported_as_a_timeout(monkeypatch):
    url_fakes.install(
        monkeypatch, {"https://example.com/slow": requests.exceptions.ConnectTimeout("slow")}
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/slow")
    assert exc.value.reason == url_fetch_service.REASON_TIMEOUT
    assert "مهلة" in exc.value.user_message


def test_read_timeout_is_reported_as_a_timeout(monkeypatch):
    url_fakes.install(
        monkeypatch, {"https://example.com/slow": requests.exceptions.ReadTimeout("slow")}
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/slow")
    assert exc.value.reason == url_fetch_service.REASON_TIMEOUT


def test_connection_error_is_reported_as_a_network_failure(monkeypatch):
    url_fakes.install(
        monkeypatch, {"https://example.com/x": requests.exceptions.ConnectionError("refused")}
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/x")
    assert exc.value.reason == url_fetch_service.REASON_NETWORK
    assert "refused" not in exc.value.user_message, "internals must not leak"


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, url_fetch_service.REASON_HTTP_UNAUTHORIZED),
        (403, url_fetch_service.REASON_HTTP_UNAUTHORIZED),
        (404, url_fetch_service.REASON_HTTP_NOT_FOUND),
        (410, url_fetch_service.REASON_HTTP_CLIENT),
        (429, url_fetch_service.REASON_HTTP_RATE_LIMIT),
        (500, url_fetch_service.REASON_HTTP_SERVER),
        (502, url_fetch_service.REASON_HTTP_SERVER),
        (503, url_fetch_service.REASON_HTTP_SERVER),
    ],
)
def test_http_error_statuses_map_to_distinct_reasons(status, expected, monkeypatch):
    url_fakes.install(
        monkeypatch,
        {"https://example.com/x": url_fakes.FakeResponse(b"error page", status_code=status)},
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/x")
    assert exc.value.reason == expected
    assert exc.value.user_message.strip()


def test_every_fetch_failure_message_is_arabic_and_user_safe():
    for reason in [
        url_fetch_service.REASON_TIMEOUT,
        url_fetch_service.REASON_NETWORK,
        url_fetch_service.REASON_TOO_MANY_REDIRECTS,
        url_fetch_service.REASON_TOO_LARGE,
        url_fetch_service.REASON_CONTENT_TYPE,
        url_fetch_service.REASON_PDF_URL,
        url_fetch_service.REASON_HTTP_UNAUTHORIZED,
        url_fetch_service.REASON_HTTP_NOT_FOUND,
        url_fetch_service.REASON_HTTP_RATE_LIMIT,
        url_fetch_service.REASON_HTTP_SERVER,
    ]:
        message = url_fetch_service.message_for(reason)
        assert message.strip()
        assert "Traceback" not in message
        assert "لا توجد معلومات" not in message, "a fetch failure is not an empty result"


# --- Request shape --------------------------------------------------------
def test_redirects_are_never_delegated_to_the_http_client(monkeypatch):
    page = url_fakes.html_page("عنوان", "<p>" + "نص كافٍ. " * 30 + "</p>")
    router = url_fakes.install(monkeypatch, {"https://example.com/a": url_fakes.FakeResponse(page)})
    url_fetch_service.fetch("https://example.com/a")
    assert router.requests[0]["allow_redirects"] is False
    assert router.requests[0]["stream"] is True


def test_both_timeouts_are_supplied(monkeypatch):
    page = url_fakes.html_page("عنوان", "<p>" + "نص كافٍ. " * 30 + "</p>")
    router = url_fakes.install(monkeypatch, {"https://example.com/a": url_fakes.FakeResponse(page)})
    url_fetch_service.fetch("https://example.com/a")
    connect, read = router.requests[0]["timeout"]
    assert connect > 0 and read > 0
