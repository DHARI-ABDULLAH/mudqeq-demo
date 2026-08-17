"""
web_demo/services/query_planner_service.py
------------------------------------------
Turns an understood case into a small set of focused *research queries*.

Why this exists
---------------
Embedding a whole case narrative and running one nearest-neighbour search is a
poor way to find the texts that decide it: the narrative is dominated by names,
dates, and incidental facts, so the closest chunks are rarely the governing
rule, its conditions, or its exceptions. Splitting the case into a few targeted
queries (rule / conditions / exceptions / procedure / consequences / limits)
surfaces each of those separately.

Bounds
------
The planner never emits more than ``MAX_CASE_RESEARCH_QUERIES`` queries and
never loops. If the model returns fewer than ``MIN_CASE_RESEARCH_QUERIES``
usable queries, deterministic fallbacks derived from the structured case top it
up — that is a completion of a successful stage, not a way to mask a failure.
A provider error is reported as a failure and stops the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    MAX_CASE_RESEARCH_QUERIES,
    MIN_CASE_RESEARCH_QUERIES,
)
from core.case_models import ResearchQuery, StructuredCase
from services import llm_service

# Each planned query must stay short — it is embedded, not read by a human.
MAX_QUERY_CHARS = 220

_ASPECTS_AR = (
    ("القاعدة الأساسية", "القاعدة أو الحكم الأساسي المتعلق بموضوع الحالة"),
    ("شروط التطبيق", "شروط تطبيق القاعدة والضوابط المطلوبة"),
    ("الاستثناءات", "الاستثناءات أو الحالات التي لا تنطبق فيها القاعدة"),
    ("الإجراءات", "الإجراءات أو المعالجات أو الحلول المتاحة"),
    ("الآثار والنتائج", "الآثار والنتائج المترتبة على الحالة"),
    ("النصوص المقيِّدة", "أي نص مقيِّد أو متعارض يؤثر على الحكم"),
)

PLANNER_INSTRUCTIONS = f"""أنت مخطّط بحث داخل مستندات. مهمتك تحويل وصف حالة إلى
عبارات بحث قصيرة تُستخدم في بحث دلالي داخل مستندات المستخدم.

{llm_service.UNTRUSTED_RULES}

# قواعد المهمة
- لا تُصدر أي حكم أو رأي، ولا تقترح حلاً. مهمتك صياغة عبارات بحث فقط.
- أنتج بين {MIN_CASE_RESEARCH_QUERIES} و{MAX_CASE_RESEARCH_QUERIES} عبارة بحث.
- كل عبارة تغطي زاوية مختلفة: القاعدة الأساسية، شروط التطبيق، الاستثناءات،
  الإجراءات، الآثار، والنصوص المقيِّدة أو المتعارضة.
- اجعل كل عبارة قصيرة (أقل من 25 كلمة) وبصيغة موضوعية عامة، لا تتضمن أسماء
  الأطراف ولا التواريخ الخاصة بالحالة.
- استخدم لغة وصف الحالة نفسها.

# صيغة الإخراج
أعد JSON فقط، بدون أي نص خارج JSON، بالشكل:
{{"queries": [{{"text": "...", "purpose": "..."}}]}}"""


@dataclass
class PlanResult:
    ok: bool
    queries: list = None  # list[ResearchQuery]
    error_category: str = ""

    def __post_init__(self) -> None:
        if self.queries is None:
            self.queries = []


def _clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:MAX_QUERY_CHARS].strip()


def _dedupe(queries: list) -> list:
    """Drop repeats (case/whitespace-insensitive) while keeping planner order."""
    seen: set[str] = set()
    out: list = []
    for q in queries:
        key = q.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def fallback_queries(case: StructuredCase, case_text: str = "") -> list:
    """Deterministic queries derived from the structured case.

    Used to top up a short plan. Built from the case's own issues first, then
    from generic aspect templates, so retrieval always covers rule/conditions/
    exceptions even when the planner returned a thin list.
    """
    topic = _clean(case.summary) or _clean(case_text)
    topic_tail = " ".join(topic.split()[:12])

    out: list = []
    for issue in case.core_issues:
        cleaned = _clean(issue)
        if cleaned:
            out.append(ResearchQuery(text=cleaned, purpose="مسألة أساسية في الحالة"))

    for label, purpose in _ASPECTS_AR:
        text = f"{label} {topic_tail}".strip() if topic_tail else label
        out.append(ResearchQuery(text=_clean(text), purpose=purpose))

    return _dedupe(out)


def _parse(data: object) -> list:
    """Read the planner's JSON into ResearchQuery objects (tolerant of shape)."""
    if isinstance(data, dict):
        raw = data.get("queries") or data.get("research_queries") or []
    elif isinstance(data, list):
        raw = data
    else:
        return []

    out: list = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            text, purpose = item, ""
        elif isinstance(item, dict):
            text = item.get("text") or item.get("query") or ""
            purpose = item.get("purpose") or item.get("goal") or ""
        else:
            continue
        cleaned = _clean(text)
        if cleaned:
            out.append(ResearchQuery(text=cleaned, purpose=_clean(purpose)))
    return out


def plan(
    session_id: str,
    case: StructuredCase,
    case_text: str = "",
    *,
    max_queries: int = MAX_CASE_RESEARCH_QUERIES,
    min_queries: int = MIN_CASE_RESEARCH_QUERIES,
) -> PlanResult:
    """Produce a bounded list of research queries for ``case``.

    Returns ``PlanResult(ok=False, ...)`` when the provider fails — the caller
    must stop rather than analysing on a guessed plan.
    """
    max_queries = max(1, int(max_queries))
    min_queries = max(1, min(int(min_queries), max_queries))

    payload = "\n\n".join(
        [
            llm_service.wrap_untrusted("وصف الحالة كما كتبه المستخدم", case_text),
            llm_service.wrap_untrusted(
                "التمثيل المنظّم للحالة",
                "\n".join(
                    filter(
                        None,
                        [
                            f"الملخص: {case.summary}" if case.summary else "",
                            "المسائل: " + " | ".join(case.core_issues)
                            if case.core_issues
                            else "",
                            "الشروط: " + " | ".join(case.conditions)
                            if case.conditions
                            else "",
                            "القيود: " + " | ".join(case.constraints)
                            if case.constraints
                            else "",
                        ],
                    )
                ),
            ),
            f"أنتج الآن عبارات البحث (بحد أقصى {max_queries}).",
        ]
    )

    result = llm_service.complete_json(
        session_id, PLANNER_INSTRUCTIONS, payload, event="case_plan"
    )
    if not result.ok:
        return PlanResult(ok=False, error_category=result.error_category)

    queries = _dedupe(_parse(result.data))[:max_queries]

    if len(queries) < min_queries:
        for extra in fallback_queries(case, case_text):
            if len(queries) >= min_queries:
                break
            queries.append(extra)
        queries = _dedupe(queries)[:max_queries]

    if not queries:
        return PlanResult(ok=False, error_category=llm_service.ERR_EMPTY)

    return PlanResult(ok=True, queries=queries)
