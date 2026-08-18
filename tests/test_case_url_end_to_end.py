"""End-to-end case analysis across a PDF **and** a web page.

The PDF carries the general rule and its conditions. The web page carries the
exception and the procedure that resolves the case. Neither source answers the
case alone, so this run proves the pipeline actually reaches, cites, and
compares evidence of both kinds rather than answering from whichever source
happened to rank first.

Only two things are faked: the socket (no real page is fetched) and the model
provider. Extraction, chunking, embeddings, FAISS retrieval, evidence curation,
citation resolution, verification, and quota accounting all run for real.
"""

from __future__ import annotations

import pytest

from core import case_models
from core.source_models import SOURCE_TYPE_PDF, SOURCE_TYPE_URL
from services import (
    case_analysis_service as cas,
)
from services import (
    document_service,
    security,
    session_service,
    url_source_service,
)
from tests import case_fakes, url_fakes
from tests.pdf_util import make_pdf

# --- The scenario ---------------------------------------------------------
RULES_PDF_PAGES = [
    "القاعدة العامة: يلتزم المموّل بصرف مبلغ التمويل خلال المدة المنصوص عليها "
    "في العقد، ويُعد التأخير عن ذلك مخالفة نظامية.",
    "شروط تطبيق القاعدة: يشترط أن يكون العقد مكتوباً، وأن تكون مدة الصرف "
    "محددة بوضوح، وأن يكون العميل قد استوفى جميع المستندات المطلوبة.",
]

GUIDE_URL = "https://rules.example.com/financing-guide"
GUIDE_TITLE = "دليل إجراءات التمويل"
GUIDE_BODY = """
<h1>دليل إجراءات التمويل</h1>
<p>يوضح هذا الدليل الإجراءات النظامية المتبعة عند التأخر في صرف مبلغ التمويل.</p>
<h2>الاستثناء</h2>
<p>لا يُعد التأخير مخالفة إذا كان ناتجاً عن نقص في مستندات العميل، بشرط أن تكون
الجهة الممولة قد أشعرت العميل كتابياً بالنقص خلال خمسة أيام عمل من تاريخ تقديم
الطلب، وبيّنت له المستندات الناقصة بشكل محدد.</p>
<h2>إجراء التظلم</h2>
<p>يجوز للعميل المتضرر من التأخير تقديم تظلم كتابي خلال ثلاثين يوماً من تاريخ
انتهاء مدة الصرف، ويُرفق بالتظلم ما يثبت استيفاء المستندات في وقتها.</p>
"""

CASE_TEXT = (
    "تقدم عميل بطلب تمويل بعقد مكتوب ينص على صرف المبلغ خلال عشرين يوماً، "
    "وسلّم جميع المستندات المطلوبة عند التقديم. تأخرت الجهة الممولة عن الصرف "
    "أربعين يوماً، ولم ترسل له أي إشعار كتابي بنقص المستندات. مضى على انتهاء "
    "مدة الصرف خمسة عشر يوماً. ما الحل الأنسب؟"
)

# Retrieval decides which chunk becomes E1, E2… so the script cites the whole
# collected set rather than guessing an order. Between them the two candidate
# solutions rest on every piece of evidence the run produced.
MIXED_SOLUTIONS = {
    "solutions": [
        {
            "title": "اعتبار التأخير مخالفة والمطالبة بأثرها",
            "description": "الاستناد إلى القاعدة العامة وشروطها المستوفاة.",
            "supporting_evidence": ["E1", "E2"],
            "conflicting_evidence": [],
            "advantages": ["الشروط مستوفاة بحسب الوقائع"],
            "limitations": ["يحتاج إثبات عدم وجود نقص في المستندات"],
            "required_conditions": ["عقد مكتوب ومدة محددة"],
            "missing_information_affecting_it": [],
        },
        {
            "title": "تقديم تظلم كتابي خلال المدة النظامية",
            "description": "سلوك الإجراء المنصوص عليه في دليل الإجراءات.",
            "supporting_evidence": ["E3", "E4"],
            "conflicting_evidence": [],
            "advantages": ["إجراء منصوص عليه ومحدد المدة"],
            "limitations": ["مقيّد بثلاثين يوماً من انتهاء مدة الصرف"],
            "required_conditions": ["إرفاق ما يثبت استيفاء المستندات"],
            "missing_information_affecting_it": [],
        },
    ],
    "conflicts": [],
    "undecidable": False,
    "undecidable_reason": "",
}

MIXED_REPORT = """# تحليل الحالة

## 1. فهم الحالة
تأخرت الجهة الممولة أربعين يوماً عن مدة صرف منصوص عليها في عقد مكتوب.

## 2. النقاط الرئيسية
- ينص المصدر على التزام المموّل بمدة الصرف وشروط تطبيق القاعدة (E1) (E2).
- ينص المصدر على استثناء نقص المستندات وعلى إجراء التظلم ومدته (E3) (E4).

## 3. النصوص والضوابط ذات العلاقة
القاعدة وشروطها واردة في الملف (E1) (E2)، والاستثناء وإجراء التظلم واردان في
دليل الإجراءات المنشور (E3) (E4).

## 4. التحليل
وبناءً على ذلك، لا ينطبق الاستثناء لعدم وجود إشعار كتابي بالنقص (E3)، وتكون
شروط القاعدة مستوفاة (E1)، ولا تزال مدة التظلم سارية (E4).

## 5. الحلول الممكنة
### الحل الأول
اعتبار التأخير مخالفة (E1) (E2).
### الحل الثاني
تقديم تظلم كتابي (E3) (E4).

## 6. الحل الأنسب بحسب المستندات
تقديم تظلم كتابي خلال المدة النظامية (E3).

## 7. سبب الترجيح
لأن الإجراء منصوص عليه ومدته لم تنقضِ بعد (E4).

## 8. المعلومات الناقصة
تاريخ تقديم الطلب بدقة غير محدد.

## 9. مستوى قوة الاستناد
متوسطة."""

MIXED_VERIFY = case_fakes.verify_pass(
    claims=[
        {
            "claim": "ينص المصدر على التزام المموّل بمدة الصرف",
            "type": "document_fact",
            "evidence_ids": ["E1", "E2"],
            "support_level": "strong",
            "conflicting_evidence_ids": [],
            "is_recommendation": False,
        },
        {
            "claim": "إجراء التظلم متاح خلال ثلاثين يوماً",
            "type": "document_fact",
            "evidence_ids": ["E3", "E4"],
            "support_level": "strong",
            "conflicting_evidence_ids": [],
            "is_recommendation": True,
        },
    ],
    recommendation_title="تقديم تظلم كتابي",
    recommendation_reason="الإجراء منصوص عليه ومدته سارية",
    recommendation_evidence_ids=["E3", "E4"],
    conflicts=[],
)


@pytest.fixture
def mixed_session(monkeypatch):
    """A session holding one PDF and one indexed web page."""
    monkeypatch.setattr(session_service, "MAX_CASES_PER_SESSION", 20)
    url_fakes.install(
        monkeypatch,
        {GUIDE_URL: url_fakes.FakeResponse(url_fakes.html_page(GUIDE_TITLE, GUIDE_BODY))},
    )
    sid = security.new_id()
    session_service.get_or_create(sid)
    try:
        pdf = document_service.ingest(sid, make_pdf(RULES_PDF_PAGES), "قواعد التمويل.pdf")
        web = url_source_service.add_url(sid, GUIDE_URL)
        yield sid, pdf, web
    finally:
        session_service.destroy(sid)


@pytest.fixture
def mixed_recorder(monkeypatch):
    return case_fakes.use_script(
        monkeypatch,
        case_fakes.full_script(
            **{
                case_fakes.PLAN: case_fakes.plan(6),
                case_fakes.SOLUTIONS: MIXED_SOLUTIONS,
                case_fakes.REPORT: MIXED_REPORT,
                case_fakes.VERIFY: MIXED_VERIFY,
                case_fakes.FOLLOWUP: "رجّحت التظلم لأن مدته لم تنقضِ (E2).",
            }
        ),
    )


@pytest.fixture
def mixed_outcome(mixed_session, mixed_recorder):
    sid, pdf, web = mixed_session
    outcome = cas.analyze(sid, CASE_TEXT, [pdf.document_id, web.source_id])
    assert outcome.ok, f"analysis failed: {outcome.kind} / {outcome.text}"
    return outcome, sid, pdf, web, mixed_recorder


# --- The pipeline runs the same way for a mixed selection ----------------
def test_the_full_pipeline_runs_over_mixed_sources(mixed_outcome):
    outcome, _, _, _, recorder = mixed_outcome

    assert recorder.stages == [
        case_fakes.UNDERSTAND,
        case_fakes.PLAN,
        case_fakes.SOLUTIONS,
        case_fakes.REPORT,
        case_fakes.VERIFY,
    ]
    assert outcome.structured_case.summary
    assert len(outcome.queries) >= 3
    assert outcome.llm_calls <= cas.max_llm_calls_per_case()


def test_evidence_is_collected_from_both_the_pdf_and_the_web_page(mixed_outcome):
    outcome, _, pdf, web, _ = mixed_outcome

    source_ids = {e.document_id for e in outcome.evidence}
    assert pdf.document_id in source_ids, "the PDF was never reached"
    assert web.source_id in source_ids, "the web page was never reached"

    kinds = {e.source_type for e in outcome.evidence}
    assert kinds == {SOURCE_TYPE_PDF, SOURCE_TYPE_URL}


def test_each_evidence_item_knows_its_real_origin(mixed_outcome):
    outcome, _, pdf, web, _ = mixed_outcome

    for item in outcome.evidence:
        if item.source_type == SOURCE_TYPE_PDF:
            assert item.document_id == pdf.document_id
            assert item.document_name == "قواعد التمويل.pdf"
            assert isinstance(item.page_start, int) and item.page_start >= 1
            assert not item.url
        else:
            assert item.document_id == web.source_id
            assert item.url == GUIDE_URL
            assert item.page_title == GUIDE_TITLE
            assert item.page_start is None
            assert item.section_title in {"", "الاستثناء", "إجراء التظلم", GUIDE_TITLE}


def test_solution_comparison_weighs_evidence_from_both_kinds(mixed_outcome):
    outcome, _, _, _, _ = mixed_outcome

    solutions = outcome.solution_set.solutions
    assert len(solutions) >= 2
    assert all(s.supporting_evidence for s in solutions)

    by_ref = {e.ref: e for e in outcome.evidence}
    cited_kinds = {
        by_ref[ref].source_type
        for s in solutions
        for ref in s.supporting_evidence
        if ref in by_ref
    }
    assert cited_kinds == {SOURCE_TYPE_PDF, SOURCE_TYPE_URL}


# --- Citations ------------------------------------------------------------
def test_the_report_cites_a_page_number_and_a_real_link(mixed_outcome):
    outcome, _, _, _, _ = mixed_outcome
    report = outcome.report_markdown

    assert "## 10. المصادر" in report
    assert outcome.citations

    pdf_cited = [c for c in outcome.citations if c.source_type == SOURCE_TYPE_PDF]
    url_cited = [c for c in outcome.citations if c.source_type == SOURCE_TYPE_URL]
    assert pdf_cited and url_cited, "both kinds must survive into the citation list"

    for cited in pdf_cited:
        assert "صفحة" in cited.citation_ar() or "الصفحات" in cited.citation_ar()
        assert f"📄 {cited.citation_ar()}" in report

    for cited in url_cited:
        assert cited.citation_ar() == f"{GUIDE_TITLE} — rules.example.com"
        assert f"🔗 [{cited.citation_ar()}]({GUIDE_URL})" in report


def test_every_citation_traces_back_to_collected_evidence(mixed_outcome):
    outcome, _, _, _, _ = mixed_outcome

    known = {e.ref for e in outcome.evidence}
    assert all(c.ref in known for c in outcome.citations)
    assert all(c.chunk_id for c in outcome.citations)


def test_no_link_in_the_report_was_written_by_the_model(mixed_outcome):
    """Every URL rendered must be one the server actually fetched."""
    outcome, _, _, _, _ = mixed_outcome

    stored = {e.url for e in outcome.evidence if e.url}
    assert stored == {GUIDE_URL}

    import re

    for link in re.findall(r"\]\((https?://[^)]+)\)", outcome.report_markdown):
        assert link in stored, f"report contains a link that no source provides: {link}"


def test_the_model_is_never_shown_a_url_it_could_paraphrase(mixed_outcome):
    outcome, _, _, _, recorder = mixed_outcome

    for stage in (case_fakes.SOLUTIONS, case_fakes.REPORT, case_fakes.VERIFY):
        sent = recorder.call_for(stage)["input"]
        assert GUIDE_URL not in sent
        assert "https://" not in sent
        # The web evidence is still identifiable to the model, by title.
        assert GUIDE_TITLE in sent


def test_a_model_invented_reference_never_becomes_a_citation(mixed_session, monkeypatch):
    sid, pdf, web = mixed_session
    bogus_report = MIXED_REPORT + "\n\nوورد أيضاً نص إضافي لا وجود له (E77)."
    case_fakes.use_script(
        monkeypatch,
        case_fakes.full_script(
            **{
                case_fakes.PLAN: case_fakes.plan(4),
                case_fakes.SOLUTIONS: MIXED_SOLUTIONS,
                case_fakes.REPORT: bogus_report,
                case_fakes.VERIFY: MIXED_VERIFY,
            }
        ),
    )

    outcome = cas.analyze(sid, CASE_TEXT, [pdf.document_id, web.source_id])

    assert outcome.ok
    known = {e.ref for e in outcome.evidence}
    assert all(c.ref in known for c in outcome.citations)
    assert "E77" not in {c.ref for c in outcome.citations}


# --- Grounding + verification --------------------------------------------
def test_verification_runs_over_web_evidence_too(mixed_outcome):
    outcome, _, _, _, recorder = mixed_outcome

    verify_input = recorder.call_for(case_fakes.VERIFY)["input"]
    assert GUIDE_TITLE in verify_input
    assert "rules.example.com" in verify_input
    assert outcome.grounding in {
        case_models.GROUNDING_STRONG,
        case_models.GROUNDING_MEDIUM,
        case_models.GROUNDING_LIMITED,
    }


def test_page_text_only_reaches_the_model_through_retrieval(mixed_outcome):
    outcome, _, _, _, recorder = mixed_outcome

    evidence_texts = [e.text for e in outcome.evidence]
    reasoning_input = recorder.call_for(case_fakes.SOLUTIONS)["input"]

    assert "<html" not in reasoning_input and "<script" not in reasoning_input
    for sentence in (
        "لا يُعد التأخير مخالفة إذا كان ناتجاً عن نقص",
        "يجوز للعميل المتضرر من التأخير تقديم تظلم",
        "القاعدة العامة: يلتزم المموّل بصرف مبلغ التمويل",
    ):
        head = sentence[:40]
        if head in reasoning_input:
            assert any(head in text for text in evidence_texts), (
                "source text reached the model without passing through retrieval"
            )


def test_follow_up_reuses_the_same_mixed_evidence(mixed_outcome):
    outcome, sid, _, _, recorder = mixed_outcome
    before = len(recorder.calls)

    answer = cas.follow_up(sid, outcome.state, "ليش اخترت التظلم؟")

    assert answer.ok
    assert len(recorder.calls) == before + 1
    assert answer.evidence == outcome.evidence
    assert {e.source_type for e in answer.evidence} == {SOURCE_TYPE_PDF, SOURCE_TYPE_URL}


# --- Single-kind selections still behave ---------------------------------
def test_a_case_can_be_analysed_against_the_web_page_alone(mixed_session, mixed_recorder):
    sid, _, web = mixed_session

    outcome = cas.analyze(sid, CASE_TEXT, [web.source_id])

    assert outcome.ok
    assert outcome.evidence
    assert {e.source_type for e in outcome.evidence} == {SOURCE_TYPE_URL}
    assert all(e.url == GUIDE_URL for e in outcome.evidence)
    assert "صفحة " not in " ".join(c.citation_ar() for c in outcome.citations)


def test_a_case_can_still_be_analysed_against_the_pdf_alone(mixed_session, mixed_recorder):
    sid, pdf, _ = mixed_session

    outcome = cas.analyze(sid, CASE_TEXT, [pdf.document_id])

    assert outcome.ok
    assert outcome.evidence
    assert {e.source_type for e in outcome.evidence} == {SOURCE_TYPE_PDF}
    assert all(isinstance(e.page_start, int) for e in outcome.evidence)
    assert all("](" not in c.citation_markdown_ar() for c in outcome.citations)


def test_a_deleted_web_source_is_not_analysable(mixed_session, mixed_recorder):
    sid, _, web = mixed_session
    url_source_service.delete_url_source(sid, web.source_id)

    outcome = cas.analyze(sid, CASE_TEXT, [web.source_id])

    assert not outcome.ok
    assert outcome.text.strip()
