"""
web_demo/services/case_verifier_service.py
------------------------------------------
Final groundedness verification for Case Analysis.

Runs once, after the draft Arabic report is produced and before it is shown.
One bounded OpenAI JSON call maps major conclusions to evidence refs; program
logic then validates refs, rejects unsupported claims, and rewrites sections
6–9 so an unverified recommendation never reaches the user.

Normal Chat / Overview / Search never call this module.
"""

from __future__ import annotations

import re

from config import MAX_CASE_CONTEXT_CHARS
from core.case_models import (
    CLAIM_DOCUMENT_FACT,
    CLAIM_INFERENCE,
    CLAIM_UNSUPPORTED,
    COVERAGE_INSUFFICIENT,
    COVERAGE_LABELS_AR,
    COVERAGE_LIMITED,
    COVERAGE_MODERATE,
    COVERAGE_STRONG,
    COVERAGE_TO_GROUNDING,
    SUPPORT_LIMITED,
    SUPPORT_MODERATE,
    SUPPORT_STRONG,
    SUPPORT_UNSUPPORTED,
    Claim,
    SolutionSet,
    StructuredCase,
    VerificationResult,
)
from core.source_models import SOURCE_TYPE_URL
from services import evidence_service, llm_service

# --- User-facing status messages (Arabic) ---------------------------------
MSG_NO_DOCS = (
    "لا توجد معلومات كافية في المستندات المحددة للوصول إلى نتيجة موثوقة."
)
MSG_MISSING_CASE = (
    "المستندات تحتوي على معلومات مرتبطة بالحالة، لكن بعض المعلومات الأساسية "
    "عن الحالة ناقصة ولا يمكن ترجيح حل بشكل موثوق."
)
MSG_TIE = (
    "توجد عدة حلول محتملة تدعمها المستندات، لكن الأدلة الحالية لا تكفي "
    "لترجيح أحدها بشكل موثوق."
)
MSG_NO_RELIABLE = (
    "لا يوجد حل يمكن ترجيحه بشكل كافٍ من المستندات الحالية بناءً على "
    "التحقق من الأدلة المتاحة."
)

_REF_PATTERN = re.compile(r"\bE\d+\b", re.IGNORECASE)
_SECTION_PATTERN = re.compile(r"^##\s+(\d+)\.\s+", re.MULTILINE)

_COVERAGE_RANK = {
    COVERAGE_INSUFFICIENT: 0,
    COVERAGE_LIMITED: 1,
    COVERAGE_MODERATE: 2,
    COVERAGE_STRONG: 3,
}

VERIFY_INSTRUCTIONS = f"""أنت مدقق استناد نهائي لتحليل حالة. مهمتك مراجعة مسودة
التقرير والأدلة المسترجَعة فقط — لا تُصدر تحليلاً جديداً من معرفتك العامة.

{llm_service.UNTRUSTED_RULES}

# المطلوب
1. استخرج الاستنتاجات الجوهرية من مسودة التقرير (5–12 بحد أقصى).
2. لكل استنتاج حدّد:
   - type: document_fact | inference | unsupported
   - evidence_ids: معرّفات E# الموجودة في قائمة الأدلة فقط
   - support_level: strong | moderate | limited | unsupported
   - conflicting_evidence_ids: معرّفات E# للأدلة المتعارضة إن وُجدت
   - is_recommendation: true فقط للترجيح/الحل المقترح في القسم السادس
3. قرّر هل يمكن ترجيح حل موثوق (recommendation_determinable).
4. إذا لا: اضبط insufficient_documents أو insufficient_case_facts أو
   tie_between_solutions حسب السبب.
5. evidence_coverage: strong | moderate | limited | insufficient
   — نوعي فقط، بدون أي نسبة مئوية.
6. missing_information: أسئلة محددة عن الحالة إن ظهر نقص بعد قراءة الأدلة.
7. conflicts: وصف موجز لكل تعارض بين الأدلة (بدون chain-of-thought).

# قواعد صارمة
- unsupported = لا يوجد دليل E# يدعمه في القائمة المعروضة.
- document_fact = نقل صريح لما في المستند.
- inference = تطبيق/استنتاج من نص مستندي على الحالة.
- لا تخترع معرّفات E# ولا قواعد غير موجودة في الأدلة.
- إذا الأدلة لا تغطي السؤال: insufficient_documents=true.
- إذا ينقص وقائع جوهرية عن الحالة: insufficient_case_facts=true.

# الإخراج — JSON فقط:
{{
  "claims": [
    {{
      "claim": "...",
      "type": "document_fact|inference|unsupported",
      "evidence_ids": ["E1"],
      "support_level": "strong|moderate|limited|unsupported",
      "conflicting_evidence_ids": ["E2"],
      "is_recommendation": false
    }}
  ],
  "recommendation_determinable": false,
  "recommendation_title": "",
  "recommendation_reason": "",
  "recommendation_evidence_ids": [],
  "missing_information": ["..."],
  "conflicts": ["..."],
  "evidence_coverage": "moderate",
  "insufficient_documents": false,
  "insufficient_case_facts": false,
  "tie_between_solutions": false
}}"""


def _metadata_line(meta: dict) -> str:
    """One line of the evidence index shown to the verifier.

    Web evidence is identified by title, domain, and section rather than a page
    number, so the verifier can tell the two source kinds apart when it maps
    conclusions onto them.
    """
    ref = meta.get("ref", "")
    if meta.get("source_type") == SOURCE_TYPE_URL:
        title = meta.get("page_title") or meta.get("document_name") or "صفحة ويب"
        domain = meta.get("domain") or ""
        section = meta.get("section_title") or ""
        tail = f" · قسم: {section}" if section else ""
        return f"- {ref}: (رابط) {title} — {domain}{tail}"
    return f"- {ref}: (ملف) {meta.get('document_name', '')} — صفحة {meta.get('page_start')}"


def verify(
    session_id: str,
    draft_report: str,
    case: StructuredCase,
    evidence: list,
    solution_set: SolutionSet,
    *,
    case_text: str = "",
) -> VerificationResult:
    """One LLM call + program validation. Never raises for provider errors."""
    if not evidence:
        return VerificationResult(
            ok=True,
            insufficient_documents=True,
            evidence_coverage=COVERAGE_INSUFFICIENT,
            conflicts=list(solution_set.conflicts if solution_set else []),
        )

    context = evidence_service.build_context(
        evidence, max_chars=MAX_CASE_CONTEXT_CHARS
    )
    metadata = [item.as_metadata() for item in evidence]
    payload = "\n\n".join(
        [
            llm_service.wrap_untrusted("وصف الحالة", case_text[:4000]),
            llm_service.wrap_untrusted(
                "ملخص الحالة المنظّم", (case.summary if case else "")[:2000]
            ),
            llm_service.wrap_untrusted("مسودة التقرير", draft_report[:8000]),
            llm_service.wrap_untrusted("الأدلة المسترجَعة", context),
            "فهرس الأدلة (metadata فقط):\n"
            + "\n".join(_metadata_line(m) for m in metadata),
            "تعارضات مسجّلة سابقاً: "
            + (" | ".join(solution_set.conflicts) if solution_set.conflicts else "لا يوجد"),
            "undecidable من مرحلة الحلول: "
            + ("نعم" if solution_set.undecidable else "لا"),
        ]
    )

    result = llm_service.complete_json(
        session_id, VERIFY_INSTRUCTIONS, payload, event="case_verify"
    )
    if not result.ok:
        return VerificationResult(ok=False, error_category=result.error_category)

    parsed = _parse_verification(result.data, evidence)
    parsed.evidence_coverage = _conservative_coverage(
        parsed, evidence, solution_set, case
    )
    return parsed


def apply(
    draft_report: str,
    verification: VerificationResult,
    evidence: list,
    solution_set: SolutionSet | None = None,
) -> tuple[str, list, str]:
    """Return (final_markdown, validated_citations, grounding_key).

    Rewrites sections 6–9 from verified conclusions; strips fake E# refs from
    the rest of the draft; builds sources only from validated evidence chunks.
    """
    solution_set = solution_set or SolutionSet()
    valid_refs = _valid_ref_set(verification, evidence)
    body = _sanitize_refs(draft_report, valid_refs)
    body = _strip_sources_section(body)

    section6 = _section_recommendation(verification, solution_set)
    section7 = _section_reason(verification)
    section8 = _section_missing_and_conflicts(verification)
    section9 = _section_coverage(verification)

    body = _replace_section(body, 6, section6)
    body = _replace_section(body, 7, section7)
    body = _replace_section(body, 8, section8)
    body = _replace_section(body, 9, section9)

    citations = _validated_citations(verification, evidence, body)
    sources = _render_sources(citations)
    grounding = COVERAGE_TO_GROUNDING.get(
        verification.evidence_coverage, COVERAGE_TO_GROUNDING[COVERAGE_LIMITED]
    )
    return f"{body.rstrip()}\n{sources}".rstrip(), citations, grounding


# --- Parsing ---------------------------------------------------------------
def _parse_verification(data: object, evidence: list) -> VerificationResult:
    if not isinstance(data, dict):
        return VerificationResult(ok=True, evidence_coverage=COVERAGE_INSUFFICIENT)

    claims: list[Claim] = []
    for raw in (data.get("claims") or [])[:15]:
        if not isinstance(raw, dict):
            continue
        claim_text = str(raw.get("claim") or "").strip()
        if not claim_text:
            continue
        claim_type = str(raw.get("type") or CLAIM_INFERENCE).strip().lower()
        if claim_type not in (CLAIM_DOCUMENT_FACT, CLAIM_INFERENCE, CLAIM_UNSUPPORTED):
            claim_type = CLAIM_UNSUPPORTED if claim_type == "unsupported" else CLAIM_INFERENCE
        support = str(raw.get("support_level") or SUPPORT_LIMITED).strip().lower()
        if support not in (
            SUPPORT_STRONG,
            SUPPORT_MODERATE,
            SUPPORT_LIMITED,
            SUPPORT_UNSUPPORTED,
        ):
            support = SUPPORT_LIMITED

        ev_ids = _normalize_refs(raw.get("evidence_ids"))
        conflict_ids = _normalize_refs(raw.get("conflicting_evidence_ids"))
        ev_ids = _filter_existing_refs(ev_ids, evidence)
        conflict_ids = _filter_existing_refs(conflict_ids, evidence)

        if claim_type == CLAIM_UNSUPPORTED or not ev_ids:
            claim_type = CLAIM_UNSUPPORTED
            support = SUPPORT_UNSUPPORTED
            ev_ids = []

        claims.append(
            Claim(
                claim=claim_text,
                claim_type=claim_type,
                evidence_ids=ev_ids,
                support_level=support,
                conflicting_evidence_ids=conflict_ids,
                is_recommendation=bool(raw.get("is_recommendation")),
            )
        )

    rec_ids = _filter_existing_refs(
        _normalize_refs(data.get("recommendation_evidence_ids")), evidence
    )
    rec_det = bool(data.get("recommendation_determinable")) and bool(rec_ids)
    if data.get("insufficient_documents") or data.get("tie_between_solutions"):
        rec_det = False
    if data.get("insufficient_case_facts"):
        rec_det = False

    rec_claims = [c for c in claims if c.is_recommendation]
    if rec_claims and all(c.support_level == SUPPORT_UNSUPPORTED for c in rec_claims):
        rec_det = False

    coverage = str(data.get("evidence_coverage") or COVERAGE_LIMITED).strip().lower()
    if coverage not in _COVERAGE_RANK:
        coverage = COVERAGE_LIMITED

    return VerificationResult(
        ok=True,
        claims=claims,
        recommendation_determinable=rec_det,
        recommendation_title=str(data.get("recommendation_title") or "").strip(),
        recommendation_reason=str(data.get("recommendation_reason") or "").strip(),
        recommendation_evidence_ids=rec_ids,
        missing_information=_str_list(data.get("missing_information")),
        conflicts=_str_list(data.get("conflicts")),
        evidence_coverage=coverage,
        insufficient_documents=bool(data.get("insufficient_documents")),
        insufficient_case_facts=bool(data.get("insufficient_case_facts")),
        tie_between_solutions=bool(data.get("tie_between_solutions")),
    )


def _str_list(value) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:12]


def _normalize_refs(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        ref = str(item or "").strip().upper()
        if ref.startswith("E") and ref[1:].isdigit():
            out.append(ref)
    return out


def _filter_existing_refs(refs: list[str], evidence: list) -> list[str]:
    table = evidence_service.by_ref(evidence)
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref in table and ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


# --- Coverage (programmatic, conservative) ---------------------------------
def _conservative_coverage(
    verification: VerificationResult,
    evidence: list,
    solution_set: SolutionSet,
    case: StructuredCase,
) -> str:
    computed = _compute_coverage(
        evidence,
        verification.claims,
        solution_set,
        case,
        conflicts=verification.conflicts,
    )
    llm_cov = verification.evidence_coverage
    if _COVERAGE_RANK.get(computed, 0) < _COVERAGE_RANK.get(llm_cov, 2):
        return computed
    return llm_cov


def _compute_coverage(
    evidence: list,
    claims: list[Claim],
    solution_set: SolutionSet,
    case: StructuredCase,
    *,
    conflicts: list | None = None,
) -> str:
    if not evidence:
        return COVERAGE_INSUFFICIENT

    supported = [c for c in claims if c.support_level != SUPPORT_UNSUPPORTED]
    strong = sum(1 for c in supported if c.support_level == SUPPORT_STRONG)
    moderate = sum(1 for c in supported if c.support_level == SUPPORT_MODERATE)
    conflict_list = conflicts if conflicts is not None else list(solution_set.conflicts)

    if case and case.critical_missing:
        return COVERAGE_INSUFFICIENT

    if solution_set.undecidable or conflict_list:
        return COVERAGE_LIMITED if len(evidence) >= 2 else COVERAGE_INSUFFICIENT

    if strong >= 2 and moderate >= 1 and not conflict_list:
        return COVERAGE_STRONG
    if strong >= 1 or moderate >= 2:
        return COVERAGE_MODERATE
    if supported:
        return COVERAGE_LIMITED
    return COVERAGE_INSUFFICIENT


# --- Section builders ------------------------------------------------------
def _section_recommendation(
    verification: VerificationResult, solution_set: SolutionSet
) -> str:
    lines = ["## 6. الحل الأنسب بحسب المستندات", ""]

    if verification.insufficient_documents:
        lines.append(MSG_NO_DOCS)
        return "\n".join(lines)

    if verification.insufficient_case_facts:
        lines.append(MSG_MISSING_CASE)
        return "\n".join(lines)

    if verification.tie_between_solutions or solution_set.undecidable:
        lines.append(MSG_TIE)
        return "\n".join(lines)

    if not verification.recommendation_determinable:
        lines.append(MSG_NO_RELIABLE)
        return "\n".join(lines)

    title = verification.recommendation_title
    if not title:
        rec_claims = [c for c in verification.claims if c.is_recommendation]
        if rec_claims:
            title = rec_claims[0].claim
    refs = verification.recommendation_evidence_ids
    ref_tag = f" ({'، '.join(refs)})" if refs else ""
    lines.append(f"{title or '—'}{ref_tag}".strip())
    return "\n".join(lines)


def _section_reason(verification: VerificationResult) -> str:
    lines = ["## 7. سبب الترجيح", ""]
    if verification.recommendation_determinable and verification.recommendation_reason:
        refs = verification.recommendation_evidence_ids
        tag = f" ({'، '.join(refs)})" if refs else ""
        lines.append(f"{verification.recommendation_reason}{tag}")
    elif verification.insufficient_documents:
        lines.append("لا توجد نصوص كافية في المستندات المختارة لدعم ترجيح محدد.")
    elif verification.insufficient_case_facts:
        lines.append(
            "لا يمكن ترجيح حل نهائي قبل توضيح المعلومات الناقصة عن الحالة."
        )
    elif verification.tie_between_solutions:
        lines.append("الأدلة تدعم أكثر من مسار ولا تكفي لحسم أيهما أرجح.")
    else:
        lines.append(
            "لم يُثبت ترجيح موثوق بعد التحقق من ربط الاستنتاجات بالأدلة."
        )
    return "\n".join(lines)


def _section_missing_and_conflicts(verification: VerificationResult) -> str:
    lines = ["## 8. المعلومات الناقصة", ""]
    missing = list(verification.missing_information)
    if missing:
        for index, item in enumerate(missing, start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("لا توجد معلومات جوهرية ناقصة مُحددة بعد التحقق.")

    if verification.conflicts:
        lines.extend(["", "**أدلة أو قيود متعارضة:**"])
        for index, conflict in enumerate(verification.conflicts, start=1):
            lines.append(f"- {conflict}")

    return "\n".join(lines)


def _section_coverage(verification: VerificationResult) -> str:
    label = COVERAGE_LABELS_AR.get(
        verification.evidence_coverage, verification.evidence_coverage
    )
    lines = [
        "## 9. مستوى قوة الاستناد",
        "",
        label + ".",
    ]
    if verification.insufficient_documents:
        lines.append("السبب: المستندات لا تغطي السؤال بشكل كافٍ.")
    elif verification.insufficient_case_facts:
        lines.append("السبب: معلومات أساسية عن الحالة غير مكتملة.")
    elif verification.tie_between_solutions:
        lines.append("السبب: أدلة متكافئة لعدة حلول.")
    elif verification.conflicts:
        lines.append("السبب: وجود نصوص أو قيود متعارضة.")
    else:
        supported = len(verification.supported_claims)
        lines.append(
            f"السبب: {supported} استنتاج/ات مدعوم/ة بالأدلة المسترجَعة."
        )
    return "\n".join(lines)


# --- Citation validation ---------------------------------------------------
def _valid_ref_set(verification: VerificationResult, evidence: list) -> set[str]:
    refs: set[str] = set()
    for claim in verification.supported_claims:
        refs.update(claim.evidence_ids)
    refs.update(
        _filter_existing_refs(verification.recommendation_evidence_ids, evidence)
    )
    table = evidence_service.by_ref(evidence)
    return {r for r in refs if r in table}


def _validated_citations(
    verification: VerificationResult, evidence: list, report_body: str
) -> list:
    valid = _valid_ref_set(verification, evidence)
    table = evidence_service.by_ref(evidence)
    cited: list = []
    for ref in sorted(valid, key=lambda r: int(r[1:])):
        item = table.get(ref)
        if item is not None and ref in (report_body or ""):
            cited.append(item)
    if cited:
        return cited
    # Fall back to recommendation evidence, then strongest in the set.
    for ref in verification.recommendation_evidence_ids:
        item = table.get(ref)
        if item is not None and item not in cited:
            cited.append(item)
    if cited:
        return cited
    from core.case_models import STRENGTH_STRONG

    return [e for e in evidence if e.strength == STRENGTH_STRONG][:5]


def _render_sources(citations: list) -> str:
    """Sources block built purely from validated evidence metadata.

    URL sources are rendered as Markdown links to the address the server
    fetched, so the reader can open the exact page the claim came from and the
    model has no way to substitute a different one.
    """
    if not citations:
        return ""
    lines = ["", "## 10. المصادر", ""]
    seen: set[str] = set()
    for index, item in enumerate(citations, start=1):
        rendered = item.citation_markdown_ar()
        if rendered in seen:
            continue
        seen.add(rendered)
        lines.append(f"{index}. {rendered}")
    return "\n".join(lines)


# --- Draft sanitization ----------------------------------------------------
def _sanitize_refs(text: str, valid_refs: set[str]) -> str:
    if not text:
        return ""

    def _replace(match: re.Match) -> str:
        ref = match.group(0).upper()
        return ref if ref in valid_refs else ""

    cleaned = _REF_PATTERN.sub(_replace, text)
    cleaned = re.sub(r"\(\s*[،,\s]*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _strip_sources_section(text: str) -> str:
    marker = "## 10."
    index = text.find(marker)
    if index == -1:
        return text.rstrip()
    return text[:index].rstrip()


def _replace_section(body: str, number: int, new_section: str) -> str:
    pattern = re.compile(
        rf"^##\s+{number}\.\s+.*?(?=^##\s+\d+\.\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(body):
        return pattern.sub(new_section + "\n", body, count=1)
    return f"{body.rstrip()}\n\n{new_section}"

