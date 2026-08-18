"""
web_demo/services/case_analysis_service.py
------------------------------------------
The "تحليل حالة" (case analysis) pipeline.

    case text
      -> understand           (LLM, structured — describes, never rules)
      -> missing-info gate    (stops and asks when something critical is absent)
      -> plan research        (LLM, query_planner_service)
      -> multi-step retrieval (FAISS, evidence_service — selected docs only)
      -> evidence curation    (dedupe, rank, strength labels)
      -> candidate solutions  (LLM, structured, each tied to evidence refs)
      -> final Arabic report  (LLM, grounded, cited)

This layer sits ON TOP of the existing RAG stack: chat, overview, and search are
untouched and keep their own code paths.

Guarantees this module exists to enforce
----------------------------------------
1. Only documents the session owns are ever searched.
2. The full PDF is never sent to the provider — only bounded, retrieved chunks.
3. Every substantive conclusion must cite evidence refs that resolve to real
   chunks; unresolvable references are dropped, not rendered.
4. A stage failure is reported as a failure. A partial pipeline never renders
   as a finished report.
5. The case quota is charged only after a complete report is produced.
6. Document text and case text never reach the logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import (
    MAX_CASE_CHARS,
    MAX_CASE_CONTEXT_CHARS,
    MAX_CASE_LLM_CALLS,
    MAX_CASE_RESEARCH_QUERIES,
    MAX_RESULTS_PER_QUERY,
    MAX_TOTAL_EVIDENCE_CHUNKS,
)
from core.case_models import (
    GROUNDING_LABELS_AR,
    GROUNDING_LIMITED,
    GROUNDING_MEDIUM,
    GROUNDING_STRONG,
    STRENGTH_STRONG,
    STRENGTH_SUPPORTING,
    MissingInfo,
    Solution,
    SolutionSet,
    StructuredCase,
)
from core.logging_utils import log_event
from services import (
    case_verifier_service,
    evidence_service,
    llm_service,
    query_planner_service,
    security,
    session_service,
)

# --- Outcome kinds --------------------------------------------------------
KIND_REPORT = "report"
KIND_NEEDS_INFO = "needs_info"
KIND_NO_SELECTION = "no_selection"
KIND_NO_CASE_TEXT = "no_case_text"
KIND_QUOTA = "quota_exceeded"
KIND_INDEX_ERROR = "index_error"
KIND_NO_EVIDENCE = "no_evidence"
KIND_PROVIDER_ERROR = "provider_error"
KIND_VERIFY_FAILED = "verify_failed"

# --- Pipeline stages (used for progress + failure reporting) --------------
STAGE_UNDERSTAND = "understand"
STAGE_PLAN = "plan"
STAGE_RETRIEVE = "retrieve"
STAGE_EVIDENCE = "evidence"
STAGE_SOLUTIONS = "solutions"
STAGE_REPORT = "report"
STAGE_VERIFY = "verify"

STAGE_LABELS_AR = {
    STAGE_UNDERSTAND: "فهم الحالة",
    STAGE_PLAN: "تحديد نقاط البحث",
    STAGE_RETRIEVE: "البحث في المستندات",
    STAGE_EVIDENCE: "جمع الأدلة",
    STAGE_SOLUTIONS: "مقارنة الحلول",
    STAGE_REPORT: "إعداد النتيجة",
    STAGE_VERIFY: "التحقق من الاستناد",
}

STAGE_ORDER = (
    STAGE_UNDERSTAND,
    STAGE_PLAN,
    STAGE_RETRIEVE,
    STAGE_EVIDENCE,
    STAGE_SOLUTIONS,
    STAGE_REPORT,
    STAGE_VERIFY,
)

# --- Arabic user-facing messages ------------------------------------------
NO_SELECTION_MESSAGE = "يرجى اختيار مستند واحد على الأقل قبل تحليل الحالة."
NO_CASE_TEXT_MESSAGE = "يرجى كتابة تفاصيل الحالة قبل بدء التحليل."
QUOTA_MESSAGE = (
    "تم الوصول إلى حد عدد تحليلات الحالة في هذه الجلسة التجريبية."
)
INDEX_ERROR_MESSAGE = (
    "تعذر الوصول إلى فهرس أحد المستندات المختارة. أعد رفع المستند أو أعد المحاولة."
)
NO_EVIDENCE_MESSAGE = (
    "لم أجد في المستندات المختارة نصوصاً ذات صلة كافية بهذه الحالة. "
    "جرّب اختيار مستندات أخرى أو إعادة صياغة تفاصيل الحالة."
)
NEEDS_INFO_MESSAGE = "أحتاج معلومات إضافية قبل إكمال التحليل"
VERIFY_FAILURE_MESSAGE = (
    "تم إعداد التحليل، لكن تعذر التحقق النهائي من مدى استناده إلى المستندات. "
    "لم يتم عرض توصية نهائية غير متحققة."
)

_INSUFFICIENT_AR = "لا تكفي المستندات الحالية للوصول إلى نتيجة موثوقة."


# --- Prompts ---------------------------------------------------------------
_GROUNDING_RULES = f"""# قاعدة الاستناد (الأهم على الإطلاق)
- يجوز لك التحليل والمقارنة والاستنتاج وربط وقائع الحالة بالنصوص.
- لكن كل نتيجة جوهرية يجب أن تستند إلى نص مسترجَع من المستندات المختارة.
- يُمنع منعاً باتاً اختلاق: قواعد، أنظمة، معايير، بنود، أحكام، سياسات، أو
  استثناءات غير موجودة في الأدلة المعروضة عليك.
- إذا كانت النتيجة تحتاج معرفة غير موجودة في الأدلة فقل صراحة:
  "{_INSUFFICIENT_AR}"
- ميّز دائماً بين ما ينص عليه المستند وبين استنتاجك أنت."""

UNDERSTAND_INSTRUCTIONS = f"""أنت محلل حالات. مهمتك في هذه المرحلة هي **فهم**
الحالة وتفكيكها فقط.

{llm_service.UNTRUSTED_RULES}

# قواعد المهمة
- لا تُصدر أي حكم أو ترجيح أو حل في هذه المرحلة إطلاقاً.
- لا تضف وقائع غير مذكورة في وصف الحالة.
- في "missing_information" اذكر فقط المعلومات التي قد يتغير بها ناتج التحليل.
- ضع "critical": true فقط إذا كان غياب المعلومة يمنع الوصول إلى نتيجة موثوقة.
- اكتب كل الحقول النصية بلغة وصف الحالة نفسها.

# صيغة الإخراج
أعد JSON فقط، بدون أي نص خارج JSON:
{{
  "summary": "...",
  "parties": ["..."],
  "facts": ["..."],
  "core_issues": ["..."],
  "conditions": ["..."],
  "constraints": ["..."],
  "questions_to_resolve": ["..."],
  "missing_information": [
    {{"question": "...", "reason": "...", "critical": true}}
  ]
}}"""

SOLUTIONS_INSTRUCTIONS = f"""أنت محلل حالات يعمل حصراً على الأدلة المسترجَعة من
مستندات المستخدم.

{llm_service.UNTRUSTED_RULES}

{_GROUNDING_RULES}

# المهمة
استخرج الحلول أو المعالجات الممكنة للحالة كما تدعمها الأدلة. لا تختر الأفضل في
هذه المرحلة؛ اعرض البدائل فقط.

# قواعد إضافية
- استشهد بالأدلة عبر معرّفاتها فقط مثل "E1" و"E3". لا تخترع معرّفات.
- إذا وجدت نصين متعارضين أو أحدهما يقيّد الآخر فسجّلهما في "conflicts"، ولا
  تتجاهل أحدهما.
- إذا لم تدعم الأدلة أي حل واضح فاجعل "solutions" فارغة واضبط
  "undecidable": true مع سبب مختصر.
- اضبط "undecidable": true أيضاً إذا كانت الأدلة متكافئة ولا تسمح بالترجيح.

# صيغة الإخراج
أعد JSON فقط:
{{
  "solutions": [
    {{
      "title": "...",
      "description": "...",
      "supporting_evidence": ["E1"],
      "conflicting_evidence": ["E4"],
      "advantages": ["..."],
      "limitations": ["..."],
      "required_conditions": ["..."],
      "missing_information_affecting_it": ["..."]
    }}
  ],
  "conflicts": ["..."],
  "undecidable": false,
  "undecidable_reason": ""
}}"""

REPORT_INSTRUCTIONS = f"""أنت "المدقق الشامل"، محلل حالات يعتمد حصراً على
المستندات المختارة.

{llm_service.UNTRUSTED_RULES}

{_GROUNDING_RULES}

# التمييز بين النص والاستنتاج (إلزامي)
- عند نقل ما في المستند ابدأ بصيغة مثل: "ينص المستند على..."
- عند تحليلك أنت ابدأ بصيغة مثل: "وبناءً على ذلك، ينطبق على الحالة..."
- لا تقدّم استنتاجك وكأنه اقتباس مباشر من المستند.

# الاستشهاد
- استشهد بعد كل نقطة جوهرية بمعرّف الدليل بين قوسين هكذا: (E2) أو (E1، E3).
- لا تستخدم معرّفاً غير موجود في قائمة الأدلة المعروضة عليك.
- الأدلة تأتي من ملفات (لها أرقام صفحات) ومن صفحات ويب (لها عناوين وأقسام).
- لا تخترع أسماء مستندات أو أرقام صفحات أو عناوين صفحات، ولا تكتب أي رابط
  (URL) في التقرير؛ أسماء المصادر وأرقام صفحاتها وروابطها تُضاف تلقائياً
  لاحقاً من بيانات المصادر الفعلية.

# مستوى قوة الاستناد
قدّره بكلمة واحدة: "قوية" أو "متوسطة" أو "محدودة"، بناءً على كمية الأدلة
ووضوحها واتفاقها ووجود معلومات ناقصة. لا تذكر نسبة مئوية ولا رقم ثقة.

# التنسيق المطلوب (Markdown بالعربية، بهذه العناوين وبهذا الترتيب)
# تحليل الحالة

## 1. فهم الحالة
## 2. النقاط الرئيسية
## 3. النصوص والضوابط ذات العلاقة
## 4. التحليل
## 5. الحلول الممكنة
## 6. الحل الأنسب بحسب المستندات
## 7. سبب الترجيح
## 8. المعلومات الناقصة
## 9. مستوى قوة الاستناد

لا تكتب قسم المصادر؛ يُضاف تلقائياً من الأدلة الفعلية.
إذا تعذّر الترجيح فاكتب في القسم السادس بوضوح أنه لا يوجد حل يمكن ترجيحه بشكل
كافٍ من المستندات الحالية، ولا تختر حلاً بلا سند."""

FOLLOWUP_INSTRUCTIONS = f"""أنت "المدقق الشامل". أجب عن سؤال متابعة يتعلق بتحليل
حالة سبق أن أعددته، اعتماداً على الأدلة المعروضة أدناه فقط.

{llm_service.UNTRUSTED_RULES}

{_GROUNDING_RULES}

# قواعد
- استشهد بمعرّفات الأدلة مثل (E2) عند الاستناد إليها.
- ميّز بين ما ينص عليه المستند وبين استنتاجك.
- إذا كان السؤال يحتاج معلومات غير موجودة في الأدلة فقل ذلك صراحة.
- أجب بإيجاز ووضوح وبالعربية."""


# --- Result types ----------------------------------------------------------
@dataclass
class CaseState:
    """Everything a follow-up question needs. Holds no chain-of-thought."""

    case_id: str = ""
    case_text: str = ""
    document_ids: list = field(default_factory=list)
    structured_case: StructuredCase = None
    evidence: list = field(default_factory=list)
    report_markdown: str = ""
    grounding: str = GROUNDING_MEDIUM
    followups: int = 0


@dataclass
class CaseOutcome:
    kind: str
    stage: str = ""
    text: str = ""
    report_markdown: str = ""
    structured_case: StructuredCase = None
    solution_set: SolutionSet = None
    queries: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    missing_information: list = field(default_factory=list)
    grounding: str = ""
    verification: object = None
    diagnostics: dict = field(default_factory=dict)
    llm_calls: int = 0
    error_reason: str = ""
    state: CaseState = None

    @property
    def ok(self) -> bool:
        return self.kind == KIND_REPORT

    @property
    def sources(self) -> list:
        """Evidence in the shape the existing source renderer expects."""
        return [e.as_source() for e in self.evidence]


# --- Parsing helpers -------------------------------------------------------
def _str_list(value, limit: int = 20) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(" ".join(item.split()))
        elif isinstance(item, (int, float)):
            out.append(str(item))
        if len(out) >= limit:
            break
    return out


def parse_structured_case(data: object) -> StructuredCase:
    """Read the understanding stage's JSON, tolerating missing keys."""
    if not isinstance(data, dict):
        return StructuredCase()

    missing: list = []
    raw_missing = data.get("missing_information")
    if isinstance(raw_missing, (list, tuple)):
        for item in raw_missing[:12]:
            if isinstance(item, str) and item.strip():
                missing.append(MissingInfo(question=" ".join(item.split())))
            elif isinstance(item, dict):
                question = item.get("question") or item.get("info") or ""
                if not isinstance(question, str) or not question.strip():
                    continue
                missing.append(
                    MissingInfo(
                        question=" ".join(question.split()),
                        reason=" ".join(str(item.get("reason") or "").split()),
                        critical=bool(item.get("critical")),
                    )
                )

    summary = data.get("summary")
    return StructuredCase(
        summary=" ".join(summary.split()) if isinstance(summary, str) else "",
        parties=_str_list(data.get("parties")),
        facts=_str_list(data.get("facts")),
        core_issues=_str_list(data.get("core_issues")),
        conditions=_str_list(data.get("conditions")),
        constraints=_str_list(data.get("constraints")),
        questions_to_resolve=_str_list(data.get("questions_to_resolve")),
        missing_information=missing,
    )


def parse_solution_set(data: object) -> SolutionSet:
    """Read the solution stage's JSON into typed candidates."""
    if not isinstance(data, dict):
        return SolutionSet()

    raw = data.get("solutions")
    solutions: list = []
    if isinstance(raw, (list, tuple)):
        for item in raw[:6]:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or ""
            description = item.get("description") or item.get("summary") or ""
            if not (isinstance(title, str) and title.strip()) and not (
                isinstance(description, str) and description.strip()
            ):
                continue
            solutions.append(
                Solution(
                    title=" ".join(str(title).split()),
                    description=" ".join(str(description).split()),
                    supporting_evidence=_str_list(item.get("supporting_evidence")),
                    conflicting_evidence=_str_list(item.get("conflicting_evidence")),
                    advantages=_str_list(item.get("advantages")),
                    limitations=_str_list(item.get("limitations")),
                    required_conditions=_str_list(item.get("required_conditions")),
                    missing_information_affecting_it=_str_list(
                        item.get("missing_information_affecting_it")
                    ),
                )
            )

    return SolutionSet(
        solutions=solutions,
        conflicts=_str_list(data.get("conflicts")),
        undecidable=bool(data.get("undecidable")) or not solutions,
        undecidable_reason=" ".join(
            str(data.get("undecidable_reason") or "").split()
        ),
    )


# --- Grounding assessment --------------------------------------------------
def assess_grounding(evidence: list, solution_set: SolutionSet, case) -> str:
    """Qualitative grounding level, computed from real counters.

    Based on how much strong evidence was found, whether the evidence conflicts
    or is undecidable, and whether critical information is still missing —
    never on a number the model asserted about itself.
    """
    strong = sum(1 for e in evidence if e.strength == STRENGTH_STRONG)
    supporting = sum(1 for e in evidence if e.strength == STRENGTH_SUPPORTING)
    critical_missing = bool(getattr(case, "critical_missing", []))
    undecidable = bool(solution_set and solution_set.undecidable)
    conflicts = bool(solution_set and solution_set.conflicts)

    if undecidable or critical_missing or strong == 0:
        return GROUNDING_LIMITED
    if strong >= 3 and supporting >= 1 and not conflicts:
        return GROUNDING_STRONG
    if strong >= 2:
        return GROUNDING_MEDIUM
    return GROUNDING_LIMITED


# --- Citations -------------------------------------------------------------
def collect_citations(evidence: list, report_markdown: str) -> list:
    """Evidence actually referenced by the report, in first-mention order.

    Only refs that appear in the text AND resolve to a real chunk survive, so a
    citation can never be decorative or invented. If the model cited nothing
    resolvable, fall back to the strongest evidence that was in its context so
    the reader still has traceable sources.
    """
    table = evidence_service.by_ref(evidence)
    cited: list = []
    for item in evidence:
        if not item.ref:
            continue
        # Match "(E2)", "(E1، E3)", "[E2]" and bare mentions alike.
        if item.ref in (report_markdown or "") and item.ref in table:
            cited.append(item)
    if cited:
        return cited
    return [e for e in evidence if e.strength == STRENGTH_STRONG][:5]


def render_sources_section(citations: list) -> str:
    """The report's "## 10. المصادر" block, built from real chunks only.

    Each line is rendered from the evidence's own stored metadata: a document
    gets its filename and page range, a web page gets its title, domain, and a
    link to the address the server actually fetched. Nothing here comes from
    the model's text, so a citation cannot name a source that does not exist.
    """
    if not citations:
        return ""
    lines = ["", "## 10. المصادر", ""]
    seen: set = set()
    for index, item in enumerate(citations, start=1):
        rendered = item.citation_markdown_ar()
        if rendered in seen:
            continue
        seen.add(rendered)
        lines.append(f"{index}. {rendered}")
    return "\n".join(lines)


# --- Progress --------------------------------------------------------------
def _notify(progress, stage: str) -> None:
    if progress is None:
        return
    try:
        progress(stage, STAGE_LABELS_AR.get(stage, stage))
    except Exception:  # noqa: BLE001 - UI callback must never break analysis
        pass


def _fail(
    stage: str, category: str, diagnostics: dict, llm_calls: int
) -> CaseOutcome:
    message = llm_service.LLMResult(
        ok=False, error_category=category
    ).user_message
    return CaseOutcome(
        kind=KIND_PROVIDER_ERROR,
        stage=stage,
        text=(
            f"تعذّر إكمال التحليل في مرحلة «{STAGE_LABELS_AR.get(stage, stage)}».\n\n"
            f"{message}"
        ),
        diagnostics={**diagnostics, "failed_stage": stage, "error": category},
        llm_calls=llm_calls,
        error_reason=category,
    )


# --- Pipeline --------------------------------------------------------------
def analyze(
    session_id: str,
    case_text: str,
    document_ids,
    *,
    additional_answers: str = "",
    force_incomplete: bool = False,
    progress=None,
    max_queries: int = MAX_CASE_RESEARCH_QUERIES,
    results_per_query: int = MAX_RESULTS_PER_QUERY,
    max_evidence: int = MAX_TOTAL_EVIDENCE_CHUNKS,
) -> CaseOutcome:
    """Run the full case-analysis pipeline once.

    ``additional_answers`` carries the user's replies to a previous
    ``KIND_NEEDS_INFO`` outcome. ``force_incomplete=True`` proceeds despite
    critical gaps, and the report is told to state its assumptions.
    """
    security.require_valid_id(session_id)
    started = time.time()
    llm_calls = 0

    case_text = (case_text or "").strip()[:MAX_CASE_CHARS]
    if not case_text:
        return CaseOutcome(kind=KIND_NO_CASE_TEXT, text=NO_CASE_TEXT_MESSAGE)

    owned = evidence_service.owned_document_ids(session_id, document_ids)
    diagnostics = {
        "selected_document_count": len(list(document_ids or [])),
        "valid_document_count": len(owned),
        "num_queries": 0,
        "num_evidence": 0,
    }
    if not owned:
        return CaseOutcome(
            kind=KIND_NO_SELECTION,
            text=NO_SELECTION_MESSAGE,
            diagnostics=diagnostics,
        )

    if not session_service.can_analyze_case(session_id):
        return CaseOutcome(
            kind=KIND_QUOTA,
            text=QUOTA_MESSAGE,
            diagnostics={**diagnostics, "error": "quota"},
        )

    log_event(
        "case_analysis",
        session_id,
        status="started",
        num_documents=len(owned),
    )

    full_case_text = case_text
    if additional_answers.strip():
        full_case_text = (
            f"{case_text}\n\n"
            f"معلومات إضافية قدّمها المستخدم:\n{additional_answers.strip()}"
        )[:MAX_CASE_CHARS]

    # --- 1. Understand ----------------------------------------------------
    _notify(progress, STAGE_UNDERSTAND)
    understand = llm_service.complete_json(
        session_id,
        UNDERSTAND_INSTRUCTIONS,
        llm_service.wrap_untrusted("وصف الحالة كما كتبه المستخدم", full_case_text),
        event="case_understand",
    )
    llm_calls += 1
    if not understand.ok:
        return _fail(
            STAGE_UNDERSTAND, understand.error_category, diagnostics, llm_calls
        )

    case = parse_structured_case(understand.data)

    # --- 2. Missing-information gate --------------------------------------
    critical = case.critical_missing
    if critical and not force_incomplete:
        log_event(
            "case_analysis",
            session_id,
            status="needs_info",
            stage=STAGE_UNDERSTAND,
            llm_calls=llm_calls,
        )
        return CaseOutcome(
            kind=KIND_NEEDS_INFO,
            stage=STAGE_UNDERSTAND,
            text=NEEDS_INFO_MESSAGE,
            structured_case=case,
            missing_information=critical,
            diagnostics=diagnostics,
            llm_calls=llm_calls,
        )

    # --- 3. Plan research queries -----------------------------------------
    _notify(progress, STAGE_PLAN)
    plan = query_planner_service.plan(
        session_id, case, full_case_text, max_queries=max_queries
    )
    llm_calls += 1
    if not plan.ok:
        return _fail(STAGE_PLAN, plan.error_category, diagnostics, llm_calls)

    queries = plan.queries[:max_queries]
    diagnostics["num_queries"] = len(queries)

    # --- 4/5. Multi-step retrieval + evidence curation --------------------
    _notify(progress, STAGE_RETRIEVE)
    try:
        evidence = evidence_service.collect(
            session_id,
            owned,
            queries,
            results_per_query=results_per_query,
            max_total=max_evidence,
        )
    except evidence_service.EvidenceUnavailable as exc:
        log_event(
            "case_analysis",
            session_id,
            status="index_error",
            stage=STAGE_RETRIEVE,
            error_category=exc.reason,
        )
        return CaseOutcome(
            kind=KIND_INDEX_ERROR,
            stage=STAGE_RETRIEVE,
            text=INDEX_ERROR_MESSAGE,
            queries=queries,
            diagnostics={**diagnostics, "error": exc.reason},
            llm_calls=llm_calls,
            error_reason=exc.reason,
        )

    _notify(progress, STAGE_EVIDENCE)
    diagnostics.update(evidence_service.summarize(evidence))
    if not evidence:
        log_event(
            "case_analysis",
            session_id,
            status="no_evidence",
            stage=STAGE_EVIDENCE,
            num_queries=len(queries),
        )
        return CaseOutcome(
            kind=KIND_NO_EVIDENCE,
            stage=STAGE_EVIDENCE,
            text=NO_EVIDENCE_MESSAGE,
            structured_case=case,
            queries=queries,
            diagnostics=diagnostics,
            llm_calls=llm_calls,
        )

    context = evidence_service.build_context(
        evidence, max_chars=MAX_CASE_CONTEXT_CHARS
    )
    case_block = _case_block(case, full_case_text)

    # --- 6. Candidate solutions -------------------------------------------
    _notify(progress, STAGE_SOLUTIONS)
    solutions_result = llm_service.complete_json(
        session_id,
        SOLUTIONS_INSTRUCTIONS,
        f"{case_block}\n\n"
        + llm_service.wrap_untrusted("الأدلة المسترجَعة من المستندات", context),
        event="case_solutions",
    )
    llm_calls += 1
    if not solutions_result.ok:
        return _fail(
            STAGE_SOLUTIONS, solutions_result.error_category, diagnostics, llm_calls
        )

    solution_set = parse_solution_set(solutions_result.data)

    # --- 7. Final grounded report -----------------------------------------
    _notify(progress, STAGE_REPORT)
    grounding = assess_grounding(evidence, solution_set, case)
    report_result = llm_service.complete(
        session_id,
        REPORT_INSTRUCTIONS,
        _report_payload(
            case_block, context, solution_set, grounding, force_incomplete, critical
        ),
        event="case_report",
    )
    llm_calls += 1
    if not report_result.ok:
        return _fail(
            STAGE_REPORT, report_result.error_category, diagnostics, llm_calls
        )

    report_markdown = report_result.text.strip()

    # --- 8. Groundedness verification (one extra LLM call, bounded) ---------
    _notify(progress, STAGE_VERIFY)
    verification = case_verifier_service.verify(
        session_id,
        report_markdown,
        case,
        evidence,
        solution_set,
        case_text=full_case_text,
    )
    llm_calls += 1
    if not verification.ok:
        log_event(
            "case_analysis",
            session_id,
            status="verify_failed",
            stage=STAGE_VERIFY,
            error_category=verification.error_category,
            llm_calls=llm_calls,
        )
        return CaseOutcome(
            kind=KIND_VERIFY_FAILED,
            stage=STAGE_VERIFY,
            text=VERIFY_FAILURE_MESSAGE,
            structured_case=case,
            solution_set=solution_set,
            queries=queries,
            evidence=evidence,
            diagnostics={
                **diagnostics,
                "failed_stage": STAGE_VERIFY,
                "error": verification.error_category,
                "llm_calls": llm_calls,
            },
            llm_calls=llm_calls,
            error_reason=verification.error_category,
        )

    report_markdown, citations, grounding = case_verifier_service.apply(
        report_markdown, verification, evidence, solution_set
    )

    # Only a verified, complete report costs the user a case operation.
    session_service.record_case(session_id)

    diagnostics["llm_calls"] = llm_calls
    diagnostics["grounding"] = grounding
    diagnostics["evidence_coverage"] = verification.evidence_coverage
    log_event(
        "case_analysis",
        session_id,
        status="ok",
        stage=STAGE_VERIFY,
        num_queries=len(queries),
        num_evidence=len(evidence),
        num_documents=len(owned),
        llm_calls=llm_calls,
        duration_ms=int((time.time() - started) * 1000),
    )

    state = CaseState(
        case_id=security.new_id(),
        case_text=full_case_text,
        document_ids=list(owned),
        structured_case=case,
        evidence=evidence,
        report_markdown=report_markdown,
        grounding=grounding,
    )

    return CaseOutcome(
        kind=KIND_REPORT,
        stage=STAGE_VERIFY,
        text=report_markdown,
        report_markdown=report_markdown,
        structured_case=case,
        solution_set=solution_set,
        queries=queries,
        evidence=evidence,
        citations=citations,
        missing_information=list(
            verification.missing_information or case.missing_information
        ),
        grounding=grounding,
        verification=verification,
        diagnostics=diagnostics,
        llm_calls=llm_calls,
        state=state,
    )


def _case_block(case: StructuredCase, case_text: str) -> str:
    """Untrusted-framed case description + its structured form."""
    lines = []
    if case.summary:
        lines.append(f"الملخص: {case.summary}")
    for label, values in (
        ("الأطراف", case.parties),
        ("الوقائع", case.facts),
        ("المسائل الأساسية", case.core_issues),
        ("الشروط", case.conditions),
        ("القيود", case.constraints),
        ("ما يجب حسمه", case.questions_to_resolve),
    ):
        if values:
            lines.append(f"{label}: " + " | ".join(values))
    if case.missing_information:
        lines.append(
            "معلومات ناقصة: "
            + " | ".join(m.question for m in case.missing_information)
        )

    return "\n\n".join(
        [
            llm_service.wrap_untrusted("وصف الحالة كما كتبه المستخدم", case_text),
            llm_service.wrap_untrusted("التمثيل المنظّم للحالة", "\n".join(lines)),
        ]
    )


def _report_payload(
    case_block: str,
    context: str,
    solution_set: SolutionSet,
    grounding: str,
    force_incomplete: bool,
    critical_missing: list,
) -> str:
    """Assemble the report prompt input (all untrusted content stays fenced)."""
    solution_lines: list = []
    for index, sol in enumerate(solution_set.solutions, start=1):
        solution_lines.append(f"الحل {index}: {sol.title}")
        if sol.description:
            solution_lines.append(f"  الوصف: {sol.description}")
        if sol.supporting_evidence:
            solution_lines.append(
                "  أدلة داعمة: " + "، ".join(sol.supporting_evidence)
            )
        if sol.conflicting_evidence:
            solution_lines.append(
                "  أدلة معارضة: " + "، ".join(sol.conflicting_evidence)
            )
        if sol.required_conditions:
            solution_lines.append(
                "  شروط لازمة: " + "، ".join(sol.required_conditions)
            )
        if sol.limitations:
            solution_lines.append("  قيود: " + "، ".join(sol.limitations))
    if solution_set.conflicts:
        solution_lines.append("تعارضات: " + " | ".join(solution_set.conflicts))
    if solution_set.undecidable:
        solution_lines.append(
            "تنبيه: الأدلة لا تسمح بالترجيح. "
            + (solution_set.undecidable_reason or "")
        )

    directives = [
        f"مستوى قوة الاستناد المحسوب من الأدلة هو: "
        f"«{GROUNDING_LABELS_AR.get(grounding, grounding)}». استخدمه في القسم التاسع.",
    ]
    if solution_set.undecidable:
        directives.append(
            "لا ترجّح أي حل في القسم السادس؛ اذكر بوضوح أنه لا يوجد حل يمكن "
            "ترجيحه بشكل كافٍ من المستندات الحالية."
        )
    if force_incomplete and critical_missing:
        directives.append(
            "المستخدم اختار إكمال التحليل رغم نقص معلومات جوهرية. اذكر في القسم "
            "الثامن الافتراضات التي بنيت عليها وأثر النقص على النتيجة."
        )

    return "\n\n".join(
        [
            case_block,
            llm_service.wrap_untrusted("الأدلة المسترجَعة من المستندات", context),
            llm_service.wrap_untrusted(
                "الحلول المرشّحة من مرحلة سابقة", "\n".join(solution_lines)
            ),
            "# توجيهات\n" + "\n".join(f"- {d}" for d in directives),
            "اكتب الآن التقرير النهائي بالتنسيق المطلوب.",
        ]
    )


# --- Follow-up -------------------------------------------------------------
def follow_up(session_id: str, state: CaseState, question: str) -> CaseOutcome:
    """Answer a question about an existing analysis using its stored evidence.

    Reuses the case's evidence set, so no new retrieval or planning happens and
    exactly one provider call is made. The follow-up quota is per case and does
    not consume another case operation.
    """
    security.require_valid_id(session_id)
    question = (question or "").strip()[:MAX_CASE_CHARS]

    if state is None or not state.evidence:
        return CaseOutcome(
            kind=KIND_NO_EVIDENCE,
            stage=STAGE_EVIDENCE,
            text=NO_EVIDENCE_MESSAGE,
        )
    if not question:
        return CaseOutcome(kind=KIND_NO_CASE_TEXT, text=NO_CASE_TEXT_MESSAGE)

    context = evidence_service.build_context(
        state.evidence, max_chars=MAX_CASE_CONTEXT_CHARS
    )
    payload = "\n\n".join(
        [
            llm_service.wrap_untrusted("وصف الحالة", state.case_text),
            llm_service.wrap_untrusted(
                "خلاصة التحليل السابق", (state.report_markdown or "")[:4000]
            ),
            llm_service.wrap_untrusted("الأدلة المسترجَعة من المستندات", context),
            llm_service.wrap_untrusted("سؤال المتابعة", question),
        ]
    )

    result = llm_service.complete(
        session_id, FOLLOWUP_INSTRUCTIONS, payload, event="case_followup"
    )
    if not result.ok:
        return CaseOutcome(
            kind=KIND_PROVIDER_ERROR,
            stage=STAGE_REPORT,
            text=result.user_message,
            llm_calls=1,
            error_reason=result.error_category,
        )

    citations = collect_citations(state.evidence, result.text)
    log_event(
        "case_followup",
        session_id,
        status="ok",
        num_evidence=len(state.evidence),
        llm_calls=1,
    )
    return CaseOutcome(
        kind=KIND_REPORT,
        stage=STAGE_REPORT,
        text=result.text.strip(),
        evidence=state.evidence,
        citations=citations,
        grounding=state.grounding,
        llm_calls=1,
        state=state,
    )


def max_llm_calls_per_case() -> int:
    """Worst-case provider calls for one successful analysis."""
    return MAX_CASE_LLM_CALLS
