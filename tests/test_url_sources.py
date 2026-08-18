"""URL sources end to end: ingest, index, retrieve, refresh, delete, isolate.

Only the socket is faked. Extraction, chunking, real multilingual-e5-small
embeddings, real FAISS indexes, retrieval, and session isolation all run for
real — a URL source is stored in exactly the same index format as a PDF, and
these tests exist to prove that claim rather than assume it.
"""

from __future__ import annotations

import pytest

from core.source_models import SOURCE_TYPE_PDF, SOURCE_TYPE_URL
from services import (
    document_service,
    retrieval_service,
    security,
    session_service,
    url_fetch_service,
    url_source_service,
)
from tests import url_fakes
from tests.pdf_util import make_pdf

FINANCE_URL = "https://example.com/financing-rules"
FAQ_URL = "https://example.com/faq"

FINANCE_BODY = """
<h1>ضوابط التمويل</h1>
<p>يشترط لصحة عقد التمويل أن يكون مكتوباً وأن تكون مدة السداد محددة بوضوح.</p>
<h2>الاستثناءات</h2>
<p>يجوز تمديد مدة السداد عند وجود ظرف طارئ خارج عن إرادة الطرفين بإشعار كتابي.</p>
<h2>الرسوم</h2>
<p>لا يجوز فرض رسوم إضافية على التأخير تتجاوز التكلفة الفعلية للمعالجة.</p>
"""

FAQ_BODY = """
<h1>الأسئلة الشائعة</h1>
<p>سؤال: كيف يمكن تقديم طلب اعتراض على قرار التمويل خلال المدة النظامية؟</p>
<p>الجواب: يُقدَّم الاعتراض كتابياً خلال ثلاثين يوماً من تاريخ التبليغ بالقرار،
ويرفق معه ما يثبت الصفة والمصلحة في الاعتراض على القرار الصادر.</p>
<p>سؤال: هل يمكن سحب الاعتراض بعد تقديمه؟ الجواب: نعم، يجوز سحب الاعتراض كتابياً
قبل صدور القرار النهائي فيه دون أن يمنع ذلك من تقديم اعتراض جديد لاحقاً.</p>
"""

PDF_PAGES = [
    "Ijarah is a leasing arrangement where the lessor retains ownership of the asset.",
    "Murabaha is a cost plus profit sale contract used widely in Islamic finance.",
]


def _serve(monkeypatch, pages: dict) -> url_fakes.Router:
    """Route each URL to an HTML page built from ``{url: (title, body)}``."""
    routes = {
        url: url_fakes.FakeResponse(url_fakes.html_page(title, body))
        for url, (title, body) in pages.items()
    }
    return url_fakes.install(monkeypatch, routes)


def _add_finance(session_id, monkeypatch) -> object:
    _serve(monkeypatch, {FINANCE_URL: ("ضوابط التمويل", FINANCE_BODY)})
    return url_source_service.add_url(session_id, FINANCE_URL)


# --- Ingest ---------------------------------------------------------------
def test_url_is_ingested_and_becomes_ready(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)

    assert result.status == session_service.STATUS_READY
    assert result.num_chunks > 0
    assert result.page_title == "ضوابط التمويل"
    assert result.final_url == FINANCE_URL
    assert result.domain == "example.com"
    assert result.retrieved_at > 0


def test_url_source_is_recorded_with_its_metadata(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)
    record = session_service.get_document(new_session, result.source_id)

    assert record is not None
    assert record.source_type == SOURCE_TYPE_URL
    assert record.is_url is True
    assert record.original_url == FINANCE_URL
    assert record.final_url == FINANCE_URL
    assert record.page_title == "ضوابط التمويل"
    assert record.domain == "example.com"
    assert record.content_type == "text/html"
    assert record.num_pages == 0
    assert record.num_chunks == result.num_chunks
    assert record.retrieved_at > 0


def test_url_source_writes_the_same_index_layout_as_a_pdf(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)
    index_file, chunks_file = retrieval_service.canonical_paths(new_session, result.source_id)
    assert index_file.exists() and chunks_file.exists()
    assert index_file.name == f"{result.source_id}.faiss"


def test_url_chunks_carry_web_provenance(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)
    _, chunks = retrieval_service._load_from_disk(new_session, result.source_id)

    assert chunks
    for chunk in chunks:
        assert chunk["source_type"] == SOURCE_TYPE_URL
        assert chunk["url"] == FINANCE_URL
        assert chunk["page_title"] == "ضوابط التمويل"
        assert chunk["page_start"] is None and chunk["page_end"] is None
        assert chunk["document_id"] == result.source_id
    assert any(c["section_title"] == "الاستثناءات" for c in chunks)


def test_no_readable_content_is_its_own_error(new_session, monkeypatch):
    empty = """<html><head><title>تطبيق</title></head><body><div id="root"></div>
    <script>render();</script></body></html>"""
    url_fakes.install(monkeypatch, {FINANCE_URL: url_fakes.FakeResponse(empty)})

    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.add_url(new_session, FINANCE_URL)

    assert exc.value.reason == url_source_service.REASON_NO_READABLE
    assert exc.value.user_message == url_source_service.NO_READABLE_MESSAGE
    assert session_service.list_url_sources(new_session) == []


def test_failed_ingest_leaves_no_index_behind(new_session, monkeypatch):
    url_fakes.install(
        monkeypatch, {FINANCE_URL: url_fakes.FakeResponse(b"", status_code=500)}
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed):
        url_source_service.add_url(new_session, FINANCE_URL)
    assert session_service.list_sources(new_session) == []


def test_progress_reports_every_stage(new_session, monkeypatch):
    seen: list[str] = []
    _serve(monkeypatch, {FINANCE_URL: ("ضوابط التمويل", FINANCE_BODY)})
    url_source_service.add_url(
        new_session, FINANCE_URL, progress=lambda stage, label: seen.append(stage)
    )
    assert seen == list(url_source_service.STAGE_ORDER)


# --- Duplicates -----------------------------------------------------------
def test_duplicate_url_is_rejected(new_session, monkeypatch):
    _add_finance(new_session, monkeypatch)
    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.add_url(new_session, FINANCE_URL)
    assert exc.value.reason == url_source_service.REASON_DUPLICATE
    assert exc.value.user_message == url_source_service.DUPLICATE_MESSAGE
    assert len(session_service.list_url_sources(new_session)) == 1


@pytest.mark.parametrize(
    "variant",
    [
        FINANCE_URL + "/",
        FINANCE_URL + "#section",
        "https://EXAMPLE.com/financing-rules",
        "  https://example.com/financing-rules  ",
    ],
)
def test_equivalent_url_forms_are_treated_as_the_same_source(new_session, monkeypatch, variant):
    _add_finance(new_session, monkeypatch)
    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.add_url(new_session, variant)
    assert exc.value.reason == url_source_service.REASON_DUPLICATE


def test_a_different_page_on_the_same_domain_is_allowed(new_session, monkeypatch):
    _serve(
        monkeypatch,
        {FINANCE_URL: ("ضوابط التمويل", FINANCE_BODY), FAQ_URL: ("الأسئلة الشائعة", FAQ_BODY)},
    )
    url_source_service.add_url(new_session, FINANCE_URL)
    url_source_service.add_url(new_session, FAQ_URL)
    assert len(session_service.list_url_sources(new_session)) == 2


# --- Retrieval ------------------------------------------------------------
def test_url_source_is_retrievable(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)
    hits = retrieval_service.retrieve(new_session, [result.source_id], "ظرف طارئ تمديد المدة", top_k=3)

    assert hits
    top = hits[0]
    assert top["source_type"] == SOURCE_TYPE_URL
    assert top["url"] == FINANCE_URL
    assert top["page_title"] == "ضوابط التمويل"
    assert top["page_start"] is None
    assert "chunk_id" not in top and "document_id" not in top


def test_url_retrieval_survives_a_cold_cache(new_session, monkeypatch):
    result = _add_finance(new_session, monkeypatch)
    retrieval_service.invalidate(new_session)
    hits = retrieval_service.retrieve(new_session, [result.source_id], "الرسوم على التأخير", top_k=3)
    assert hits and hits[0]["url"] == FINANCE_URL


def test_pdf_retrieval_is_unchanged_by_the_presence_of_url_sources(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    _add_finance(new_session, monkeypatch)

    hits = retrieval_service.retrieve(new_session, [pdf.document_id], "leasing ownership", top_k=3)
    assert hits
    top = hits[0]
    assert top["document_name"] == "finance.pdf"
    assert top["source_type"] == SOURCE_TYPE_PDF
    assert isinstance(top["page_start"], int) and top["page_start"] >= 1
    assert "url" not in top, "a PDF result must not grow web fields"


def test_pdf_and_url_are_searched_together(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    url = _add_finance(new_session, monkeypatch)

    hits = retrieval_service.retrieve(
        new_session, [pdf.document_id, url.source_id], "عقد التمويل والإجارة leasing", top_k=8
    )
    kinds = {h["source_type"] for h in hits}
    assert kinds == {SOURCE_TYPE_PDF, SOURCE_TYPE_URL}, "both source kinds must be reachable"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_selecting_only_the_url_excludes_the_pdf(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    url = _add_finance(new_session, monkeypatch)

    only_url = retrieval_service.retrieve(new_session, [url.source_id], "leasing ownership", top_k=5)
    assert only_url and all(h["source_type"] == SOURCE_TYPE_URL for h in only_url)

    only_pdf = retrieval_service.retrieve(new_session, [pdf.document_id], "ظرف طارئ", top_k=5)
    assert only_pdf and all(h["source_type"] == SOURCE_TYPE_PDF for h in only_pdf)


def test_overview_context_covers_both_source_kinds(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    url = _add_finance(new_session, monkeypatch)

    context = retrieval_service.document_context(new_session, [pdf.document_id, url.source_id])
    assert context
    kinds = {c["source_type"] for c in context}
    assert kinds == {SOURCE_TYPE_PDF, SOURCE_TYPE_URL}
    web = [c for c in context if c["source_type"] == SOURCE_TYPE_URL]
    assert all(c["url"] == FINANCE_URL for c in web)


def test_overview_of_a_single_url_only_reads_that_url(new_session, monkeypatch):
    _serve(
        monkeypatch,
        {FINANCE_URL: ("ضوابط التمويل", FINANCE_BODY), FAQ_URL: ("الأسئلة الشائعة", FAQ_BODY)},
    )
    finance = url_source_service.add_url(new_session, FINANCE_URL)
    url_source_service.add_url(new_session, FAQ_URL)

    context = retrieval_service.document_context(new_session, [finance.source_id])
    assert context and all(c["url"] == FINANCE_URL for c in context)
    assert "الاعتراض" not in " ".join(c["text"] for c in context)


# --- Refresh --------------------------------------------------------------
def test_refresh_replaces_the_content_and_keeps_the_id(new_session, monkeypatch):
    original = _add_finance(new_session, monkeypatch)
    before = session_service.get_document(new_session, original.source_id).retrieved_at

    updated_body = """
    <h1>ضوابط التمويل</h1>
    <p>تم تعديل النص: أصبحت مدة السداد القصوى ستة وثلاثين شهراً بدلاً من المدة
    السابقة، ويسري التعديل على العقود الجديدة فقط دون أثر رجعي على ما سبقها.</p>
    <p>ويجب على الجهة الممولة إشعار العميل بالمدة الجديدة قبل توقيع العقد بمدة
    كافية تتيح له مراجعة الشروط والاعتراض عليها إن رغب في ذلك.</p>
    """
    _serve(monkeypatch, {FINANCE_URL: ("ضوابط التمويل", updated_body)})
    refreshed = url_source_service.refresh_url(new_session, original.source_id)

    assert refreshed.source_id == original.source_id
    record = session_service.get_document(new_session, original.source_id)
    assert record.retrieved_at >= before
    assert record.num_chunks == refreshed.num_chunks

    retrieval_service.invalidate(new_session)
    hits = retrieval_service.retrieve(new_session, [original.source_id], "مدة السداد القصوى", top_k=3)
    text = " ".join(h["text"] for h in hits)
    assert "ستة وثلاثين شهراً" in text
    assert "ظرف طارئ" not in text, "stale chunks must be replaced, not merged"


def test_failed_refresh_leaves_the_existing_index_usable(new_session, monkeypatch):
    original = _add_finance(new_session, monkeypatch)

    url_fakes.install(monkeypatch, {FINANCE_URL: url_fakes.FakeResponse(b"", status_code=503)})
    with pytest.raises(url_fetch_service.UrlFetchFailed):
        url_source_service.refresh_url(new_session, original.source_id)

    retrieval_service.invalidate(new_session)
    hits = retrieval_service.retrieve(new_session, [original.source_id], "ظرف طارئ", top_k=3)
    assert hits, "a failed refresh must not destroy the working index"


def test_refresh_rejects_a_pdf_source(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.refresh_url(new_session, pdf.document_id)
    assert exc.value.reason == url_source_service.REASON_NOT_URL


def test_refresh_rejects_an_unknown_source(new_session):
    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.refresh_url(new_session, security.new_id())
    assert exc.value.reason == url_source_service.REASON_NOT_FOUND


# --- Delete ---------------------------------------------------------------
def test_deleting_a_url_source_removes_record_chunks_and_index(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    url = _add_finance(new_session, monkeypatch)
    index_file, chunks_file = retrieval_service.canonical_paths(new_session, url.source_id)
    assert index_file.exists() and chunks_file.exists()

    url_source_service.delete_url_source(new_session, url.source_id)

    assert session_service.get_document(new_session, url.source_id) is None
    assert not index_file.exists() and not chunks_file.exists()
    assert retrieval_service.retrieve(new_session, [url.source_id], "ظرف طارئ", top_k=3) == []

    # The PDF alongside it is untouched.
    assert session_service.get_document(new_session, pdf.document_id) is not None
    assert retrieval_service.retrieve(new_session, [pdf.document_id], "leasing", top_k=3)


def test_deleting_one_url_leaves_the_other(new_session, monkeypatch):
    _serve(
        monkeypatch,
        {FINANCE_URL: ("ضوابط التمويل", FINANCE_BODY), FAQ_URL: ("الأسئلة الشائعة", FAQ_BODY)},
    )
    finance = url_source_service.add_url(new_session, FINANCE_URL)
    faq = url_source_service.add_url(new_session, FAQ_URL)

    url_source_service.delete_url_source(new_session, finance.source_id)

    assert session_service.get_document(new_session, faq.source_id) is not None
    assert retrieval_service.retrieve(new_session, [faq.source_id], "الاعتراض", top_k=3)


def test_deleting_and_re_adding_the_same_url_is_allowed(new_session, monkeypatch):
    first = _add_finance(new_session, monkeypatch)
    url_source_service.delete_url_source(new_session, first.source_id)
    second = url_source_service.add_url(new_session, FINANCE_URL)
    assert second.source_id != first.source_id


# --- Session isolation ----------------------------------------------------
def test_session_b_cannot_read_session_a_url_source(monkeypatch):
    sid_a, sid_b = security.new_id(), security.new_id()
    session_service.get_or_create(sid_a)
    session_service.get_or_create(sid_b)
    try:
        url = _add_finance(sid_a, monkeypatch)

        # B knows A's source id (worst case) and must still be denied.
        assert session_service.get_document(sid_b, url.source_id) is None
        assert retrieval_service.retrieve(sid_b, [url.source_id], "ظرف طارئ", top_k=3) == []
        assert session_service.list_url_sources(sid_b) == []

        # A still has it.
        assert retrieval_service.retrieve(sid_a, [url.source_id], "ظرف طارئ", top_k=3)
    finally:
        session_service.destroy(sid_a)
        session_service.destroy(sid_b)


def test_session_b_cannot_delete_or_refresh_session_a_url_source(monkeypatch):
    sid_a, sid_b = security.new_id(), security.new_id()
    session_service.get_or_create(sid_a)
    session_service.get_or_create(sid_b)
    try:
        url = _add_finance(sid_a, monkeypatch)

        with pytest.raises(url_source_service.UrlSourceError) as exc:
            url_source_service.refresh_url(sid_b, url.source_id)
        assert exc.value.reason == url_source_service.REASON_NOT_FOUND

        url_source_service.delete_url_source(sid_b, url.source_id)
        assert session_service.get_document(sid_a, url.source_id) is not None
        assert retrieval_service.retrieve(sid_a, [url.source_id], "ظرف طارئ", top_k=3)
    finally:
        session_service.destroy(sid_a)
        session_service.destroy(sid_b)


# --- Quotas + unified listings --------------------------------------------
def test_url_sources_do_not_consume_pdf_slots(new_session, monkeypatch):
    before = session_service.live_document_count(new_session)
    _add_finance(new_session, monkeypatch)
    assert session_service.live_document_count(new_session) == before
    assert session_service.live_url_count(new_session) == 1
    assert session_service.has_document_slot(new_session) is True


def test_url_slot_limit_is_enforced(new_session, monkeypatch):
    limit = session_service.MAX_URL_SOURCES_PER_SESSION
    routes = {
        f"https://example.com/page-{i}": url_fakes.FakeResponse(
            url_fakes.html_page(f"صفحة {i}", FINANCE_BODY)
        )
        for i in range(limit + 1)
    }
    url_fakes.install(monkeypatch, routes)

    for i in range(limit):
        url_source_service.add_url(new_session, f"https://example.com/page-{i}")

    with pytest.raises(url_source_service.UrlSourceError) as exc:
        url_source_service.add_url(new_session, f"https://example.com/page-{limit}")
    assert exc.value.reason == url_source_service.REASON_SLOT_LIMIT
    assert session_service.live_url_count(new_session) == limit


def test_unified_listings_separate_the_two_kinds(new_session, monkeypatch):
    pdf = document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    url = _add_finance(new_session, monkeypatch)

    ids = {s.document_id for s in session_service.list_sources(new_session)}
    assert ids == {pdf.document_id, url.source_id}
    assert [d.document_id for d in session_service.list_documents(new_session)] == [pdf.document_id]
    assert [u.document_id for u in session_service.list_url_sources(new_session)] == [url.source_id]
    assert {s.document_id for s in session_service.ready_sources(new_session)} == ids


def test_stats_count_both_kinds(new_session, monkeypatch):
    document_service.ingest(new_session, make_pdf(PDF_PAGES), "finance.pdf")
    _add_finance(new_session, monkeypatch)

    stats = session_service.stats(new_session)
    assert stats["num_documents"] == 1
    assert stats["num_urls"] == 1
    assert stats["num_sources"] == 2
    assert stats["total_pages"] == 2, "a web page contributes no PDF pages"
    assert stats["total_chunks"] > 0


def test_diagnostics_cover_url_sources_without_leaking_content(new_session, monkeypatch):
    url = _add_finance(new_session, monkeypatch)
    diags = document_service.diagnostics(new_session)
    entry = next(d for d in diags if d["document_id"] == url.source_id)

    assert entry["source_type"] == SOURCE_TYPE_URL
    assert entry["domain"] == "example.com"
    assert entry["index_loadable"] and entry["chunks_loadable"]
    assert entry["num_vectors"] == entry["num_indexed_chunks"] > 0

    blob = str(diags)
    assert "ظرف طارئ" not in blob
    assert "يشترط لصحة عقد التمويل" not in blob
