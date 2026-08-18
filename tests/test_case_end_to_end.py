"""End-to-end case analysis across two documents.

Document A carries a general rule, its conditions, and an exception.
Document B carries an alternative procedure and a limitation on it.
The case can only be resolved by reading both, so this exercise proves the
pipeline actually reaches, cites, and compares evidence from each — instead of
answering from whichever document happened to rank first.

Only the provider is faked. Extraction, chunking, embeddings, FAISS retrieval,
evidence curation, citation resolution, and quota accounting all run for real.
"""

from __future__ import annotations

import pytest

from core import case_models
from services import (
    case_analysis_service as cas,
)
from services import (
    document_service,
    llm_service,
    security,
    session_service,
)
from tests import case_fakes
from tests.pdf_util import make_pdf

# --- The scenario ----------------------------------------------------------
RULES_PDF_PAGES = [
    "القاعدة العامة: يلتزم البائع بتسليم البضاعة إلى المشتري خلال المدة "
    "المنصوص عليها في العقد، ويترتب على مخالفتها أثر نظامي.",
    "شروط تطبيق القاعدة: يشترط أن يكون العقد مكتوباً، وأن تكون مدة التسليم "
    "محددة بوضوح، وأن يكون المشتري قد أوفى بالتزاماته المالية.",
    "الاستثناء: لا يُعد التأخير مخالفة إذا كان ناتجاً عن قوة قاهرة أو ظرف "
    "طارئ خارج عن إرادة البائع وأخطر به المشتري كتابياً.",
]

PROCEDURE_PDF_PAGES = [
    "الإجراء البديل: يجوز للطرف المتضرر من التأخير أن يطلب تعويضاً عن الضرر "
    "الفعلي بدلاً من طلب فسخ العقد، ويقدم الطلب خلال ستين يوماً.",
    "القيد على الفسخ: لا يجوز فسخ العقد بسبب التأخير قبل توجيه إعذار كتابي "
    "إلى البائع ومنحه مهلة معقولة لتصحيح الوضع.",
]

CASE_TEXT = (
    "تعاقد مشتري مع بائع بعقد مكتوب على تسليم بضاعة خلال ثلاثين يوماً، وسدد "
    "المشتري كامل المبلغ. تأخر البائع خمسة عشر يوماً، ولم يذكر أي ظرف طارئ، "
    "ولم يوجه المشتري أي إعذار كتابي حتى الآن. المشتري يريد فسخ العقد فوراً. "
    "ما الحل الأنسب؟"
)

# Two solutions, each grounded in a different document, so the comparison stage
# has something real to weigh.
E2E_SOLUTIONS = {
    "solutions": [
        {
            "title": "فسخ العقد",
            "description": "فسخ العقد استناداً إلى مخالفة مدة التسليم.",
            "supporting_evidence": ["E1"],
            "conflicting_evidence": ["E2"],
            "advantages": ["إنهاء العلاقة التعاقدية"],
            "limitations": ["مقيّد بالإعذار الكتابي المسبق"],
            "required_conditions": ["توجيه إعذار كتابي ومنح مهلة"],
            "missing_information_affecting_it": [],
        },
        {
            "title": "طلب التعويض",
            "description": "المطالبة بتعويض عن الضرر الفعلي مع بقاء العقد.",
            "supporting_evidence": ["E2"],
            "conflicting_evidence": [],
            "advantages": ["متاح دون إعذار مسبق"],
            "limitations": ["مقيّد بمهلة ستين يوماً"],
            "required_conditions": ["إثبات الضرر الفعلي"],
            "missing_information_affecting_it": [],
        },
    ],
    "conflicts": [
        "نص يجيز الفسخ عند مخالفة المدة، ونص آخر يمنع الفسخ قبل الإعذار الكتابي."
    ],
    "undecidable": False,
    "undecidable_reason": "",
}

E2E_REPORT = """# تحليل الحالة

## 1. فهم الحالة
تأخر البائع خمسة عشر يوماً عن مدة تسليم متفق عليها في عقد مكتوب.

## 2. النقاط الرئيسية
- ينص المستند على التزام البائع بمدة التسليم (E1).
- ينص المستند على قيد يمنع الفسخ قبل الإعذار الكتابي (E2).

## 3. النصوص والضوابط ذات العلاقة
ينص المستند الأول على القاعدة وشروطها (E1)، وينص المستند الثاني على الإجراء
البديل والقيد الوارد عليه (E2).

## 4. التحليل
وبناءً على ذلك، ينطبق على الحالة شرط الإعذار الكتابي، ولم يقم به المشتري بعد.

## 5. الحلول الممكنة
### الحل الأول
فسخ العقد (E1).
### الحل الثاني
طلب التعويض (E2).

## 6. الحل الأنسب بحسب المستندات
طلب التعويض هو الأنسب حالياً (E2).

## 7. سبب الترجيح
لأن الفسخ مقيّد بإعذار كتابي لم يقع بعد (E2).

## 8. المعلومات الناقصة
مقدار الضرر الفعلي غير محدد.

## 9. مستوى قوة الاستناد
متوسطة."""


@pytest.fixture
def two_document_session(monkeypatch):
    monkeypatch.setattr(session_service, "MAX_CASES_PER_SESSION", 20)
    sid = security.new_id()
    session_service.get_or_create(sid)
    try:
        rules = document_service.ingest(
            sid, make_pdf(RULES_PDF_PAGES), "القواعد.pdf"
        )
        procedure = document_service.ingest(
            sid, make_pdf(PROCEDURE_PDF_PAGES), "الإجراءات.pdf"
        )
        yield sid, rules, procedure
    finally:
        session_service.destroy(sid)


@pytest.fixture
def e2e_recorder(monkeypatch):
    return case_fakes.use_script(
        monkeypatch,
        case_fakes.full_script(
            **{
                case_fakes.PLAN: case_fakes.plan(6),
                case_fakes.SOLUTIONS: E2E_SOLUTIONS,
                case_fakes.REPORT: E2E_REPORT,
                case_fakes.FOLLOWUP: "رجّحت التعويض لأن الفسخ مقيّد بالإعذار (E2).",
            }
        ),
    )


@pytest.fixture
def e2e_outcome(two_document_session, e2e_recorder):
    sid, rules, procedure = two_document_session
    outcome = cas.analyze(
        sid, CASE_TEXT, [rules.document_id, procedure.document_id]
    )
    assert outcome.ok, f"analysis failed: {outcome.kind} / {outcome.text}"
    return outcome, sid, rules, procedure, e2e_recorder


# --- 1. The case is understood --------------------------------------------
def test_case_is_understood_before_any_retrieval(e2e_outcome):
    outcome, _, _, _, recorder = e2e_outcome

    assert recorder.stages[0] == case_fakes.UNDERSTAND
    case = outcome.structured_case
    assert case.summary
    assert case.facts
    assert case.core_issues


# --- 2. Several focused queries are generated ------------------------------
def test_multiple_research_queries_are_generated(e2e_outcome):
    outcome, _, _, _, _ = e2e_outcome

    assert len(outcome.queries) >= 3
    texts = [q.text for q in outcome.queries]
    assert len(set(texts)) == len(texts)


# --- 3. Evidence comes from BOTH documents --------------------------------
def test_evidence_is_found_in_both_documents(e2e_outcome):
    outcome, _, rules, procedure, _ = e2e_outcome

    document_ids = {e.document_id for e in outcome.evidence}
    assert rules.document_id in document_ids, "rule document was never reached"
    assert procedure.document_id in document_ids, "procedure document was never reached"


# --- 4. Page provenance survives the whole pipeline -----------------------
def test_page_numbers_survive_to_the_report(e2e_outcome):
    outcome, _, _, _, _ = e2e_outcome

    for item in outcome.evidence:
        assert item.page_start is not None
        assert item.page_end is not None
        assert item.document_name

    for cited in outcome.citations:
        assert "صفحة" in cited.citation_ar() or "الصفحات" in cited.citation_ar()


# --- 5. At least two solutions are compared -------------------------------
def test_at_least_two_solutions_are_compared(e2e_outcome):
    outcome, _, _, _, _ = e2e_outcome

    assert len(outcome.solution_set.solutions) >= 2
    titles = [s.title for s in outcome.solution_set.solutions]
    assert len(set(titles)) == len(titles)
    # Each candidate is tied to evidence rather than asserted bare.
    assert all(s.supporting_evidence for s in outcome.solution_set.solutions)


# --- 6. A supported solution wins, or the tie is declared -----------------
def test_the_better_supported_solution_is_selected(e2e_outcome):
    outcome, _, _, _, _ = e2e_outcome

    assert not outcome.solution_set.undecidable
    assert "## 6. الحل الأنسب بحسب المستندات" in outcome.report_markdown
    assert "## 7. سبب الترجيح" in outcome.report_markdown


def test_equal_evidence_refuses_to_pick_a_winner(two_document_session, monkeypatch):
    sid, rules, procedure = two_document_session
    recorder = case_fakes.use_script(
        monkeypatch,
        case_fakes.full_script(
            **{case_fakes.SOLUTIONS: case_fakes.solutions(undecidable=True)}
        ),
    )

    outcome = cas.analyze(
        sid, CASE_TEXT, [rules.document_id, procedure.document_id]
    )

    assert outcome.ok
    assert outcome.solution_set.undecidable
    assert outcome.grounding == case_models.GROUNDING_LIMITED
    report_prompt = recorder.call_for(case_fakes.REPORT)["input"]
    assert "لا ترجّح أي حل" in report_prompt


# --- 7. Citations are real ------------------------------------------------
def test_report_ends_with_traceable_sources(e2e_outcome):
    outcome, _, _, _, _ = e2e_outcome

    assert "## 10. المصادر" in outcome.report_markdown
    assert outcome.citations
    known_refs = {e.ref for e in outcome.evidence}
    assert all(c.ref in known_refs for c in outcome.citations)
    for cited in outcome.citations:
        assert cited.citation_ar() in outcome.report_markdown


# --- 8. No rule is invented ------------------------------------------------
def test_the_model_only_ever_sees_retrieved_text(e2e_outcome):
    outcome, _, _, _, recorder = e2e_outcome

    evidence_texts = [e.text for e in outcome.evidence]
    reasoning_input = recorder.call_for(case_fakes.SOLUTIONS)["input"]

    # Every document sentence in the prompt came from a retrieved chunk.
    for page in RULES_PDF_PAGES + PROCEDURE_PDF_PAGES:
        head = page[:40]
        if head in reasoning_input:
            assert any(head in text for text in evidence_texts), (
                "document text reached the model without passing through retrieval"
            )


def test_grounding_rules_forbid_inventing_a_rule(e2e_outcome):
    _, _, _, _, recorder = e2e_outcome

    for stage in (case_fakes.SOLUTIONS, case_fakes.REPORT):
        instructions = recorder.call_for(stage)["instructions"]
        assert "يُمنع منعاً باتاً اختلاق" in instructions
        assert llm_service.UNTRUSTED_RULES in instructions


# --- Follow-up on the same case -------------------------------------------
def test_follow_up_answers_from_the_same_evidence(e2e_outcome):
    outcome, sid, _, _, recorder = e2e_outcome
    before = len(recorder.calls)

    answer = cas.follow_up(sid, outcome.state, "ليش اخترت الحل الثاني؟")

    assert answer.ok
    assert len(recorder.calls) == before + 1
    assert answer.evidence == outcome.evidence
    followup_input = recorder.call_for(case_fakes.FOLLOWUP)["input"]
    assert "سؤال المتابعة" in followup_input
    assert "%PDF" not in followup_input


def test_full_run_stays_within_the_call_budget(e2e_outcome):
    outcome, _, _, _, recorder = e2e_outcome

    assert outcome.llm_calls == 5
    assert outcome.llm_calls <= cas.max_llm_calls_per_case()
    assert recorder.stages == [
        case_fakes.UNDERSTAND,
        case_fakes.PLAN,
        case_fakes.SOLUTIONS,
        case_fakes.REPORT,
        case_fakes.VERIFY,
    ]
