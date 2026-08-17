"""
web_demo/tests/case_fakes.py
----------------------------
A scripted stand-in for the OpenAI Responses API, routed by pipeline stage.

The case pipeline makes several provider calls with different instructions.
Tests script one canned reply per stage and get back a recorder holding every
request that was actually sent, which is what the token/PDF-boundary and
prompt-injection assertions inspect.

Stages are identified by comparing against the exact instruction constants, so
editing prompt wording can never silently mis-route a test.
"""

from __future__ import annotations

import json

from services import case_analysis_service as cas
from services import llm_service
from services import query_planner_service as qps

UNDERSTAND = "understand"
PLAN = "plan"
SOLUTIONS = "solutions"
REPORT = "report"
FOLLOWUP = "followup"

_STAGE_BY_INSTRUCTIONS = {
    cas.UNDERSTAND_INSTRUCTIONS: UNDERSTAND,
    qps.PLANNER_INSTRUCTIONS: PLAN,
    cas.SOLUTIONS_INSTRUCTIONS: SOLUTIONS,
    cas.REPORT_INSTRUCTIONS: REPORT,
    cas.FOLLOWUP_INSTRUCTIONS: FOLLOWUP,
}


def stage_of(instructions: str) -> str:
    return _STAGE_BY_INSTRUCTIONS.get(instructions, "unknown")


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class Recorder:
    """Captures every request and replies with the script for its stage."""

    def __init__(self, script: dict) -> None:
        self.script = dict(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        stage = stage_of(kwargs.get("instructions", ""))
        self.calls.append({**kwargs, "stage": stage})

        reply = self.script.get(stage)
        if reply is None:
            raise AssertionError(f"unscripted stage: {stage}")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (dict, list)):
            reply = json.dumps(reply, ensure_ascii=False)
        return FakeResponse(reply)

    # --- Assertions helpers ------------------------------------------------
    @property
    def stages(self) -> list:
        return [c["stage"] for c in self.calls]

    def inputs(self) -> list:
        return [c.get("input", "") for c in self.calls]

    def instructions(self) -> list:
        return [c.get("instructions", "") for c in self.calls]

    def call_for(self, stage: str) -> dict:
        for call in self.calls:
            if call["stage"] == stage:
                return call
        raise AssertionError(f"stage not called: {stage}")


class FakeClient:
    def __init__(self, script: dict) -> None:
        self.responses = Recorder(script)


def use_script(monkeypatch, script: dict) -> Recorder:
    """Point llm_service at a scripted client and return its recorder."""
    client = FakeClient(script)
    monkeypatch.setattr(llm_service, "openai_is_configured", lambda: True)
    monkeypatch.setattr(llm_service, "get_openai_model", lambda: "gpt-4o-mini")
    monkeypatch.setattr(llm_service, "get_client", lambda: client)
    monkeypatch.setattr(llm_service, "OPENAI_MAX_RETRIES", 0)
    monkeypatch.setattr(llm_service, "_backoff", lambda attempt: None)
    return client.responses


# --- Reusable canned payloads ---------------------------------------------
def understanding(
    *,
    summary: str = "نزاع حول تنفيذ عقد بيع بين طرفين.",
    critical_missing: bool = False,
    language: str = "ar",
) -> dict:
    if language == "en":
        payload = {
            "summary": "A dispute over performance of a sale contract.",
            "parties": ["Buyer", "Seller"],
            "facts": ["Goods were delivered late."],
            "core_issues": ["Is late delivery a breach?"],
            "conditions": ["Delivery within 30 days"],
            "constraints": ["Written contract only"],
            "questions_to_resolve": ["Which remedy applies?"],
            "missing_information": [],
        }
    else:
        payload = {
            "summary": summary,
            "parties": ["المشتري", "البائع"],
            "facts": ["تم التسليم بعد الموعد المتفق عليه."],
            "core_issues": ["هل يُعد التأخير إخلالاً بالعقد؟"],
            "conditions": ["التسليم خلال ثلاثين يوماً"],
            "constraints": ["الاعتماد على العقد المكتوب فقط"],
            "questions_to_resolve": ["ما المعالجة المناسبة؟"],
            "missing_information": [],
        }

    if critical_missing:
        payload["missing_information"] = [
            {
                "question": "هل يوجد نص صريح في العقد بخصوص التأخير؟",
                "reason": "يحدد المعالجة المطبقة",
                "critical": True,
            },
            {
                "question": "ما تاريخ التوقيع؟",
                "reason": "معلومة مساعدة",
                "critical": False,
            },
        ]
    return payload


def plan(count: int = 4) -> dict:
    base = [
        {"text": "القاعدة الأساسية في تأخير التسليم", "purpose": "القاعدة"},
        {"text": "شروط اعتبار التأخير إخلالاً", "purpose": "الشروط"},
        {"text": "استثناءات التأخير المعذور", "purpose": "الاستثناءات"},
        {"text": "الإجراء البديل عند التأخير", "purpose": "الإجراءات"},
        {"text": "الآثار المترتبة على الإخلال", "purpose": "الآثار"},
        {"text": "النصوص المقيدة للفسخ", "purpose": "القيود"},
    ]
    return {"queries": base[:count]}


def solutions(*, undecidable: bool = False, conflicts=None) -> dict:
    if undecidable:
        return {
            "solutions": [],
            "conflicts": list(conflicts or []),
            "undecidable": True,
            "undecidable_reason": "الأدلة متكافئة ولا تسمح بالترجيح.",
        }
    return {
        "solutions": [
            {
                "title": "الفسخ",
                "description": "فسخ العقد لتأخر التسليم.",
                "supporting_evidence": ["E1"],
                "conflicting_evidence": ["E2"],
                "advantages": ["إنهاء سريع للنزاع"],
                "limitations": ["يحتاج إعذاراً مسبقاً"],
                "required_conditions": ["تجاوز المدة المتفق عليها"],
                "missing_information_affecting_it": [],
            },
            {
                "title": "التعويض مع الإبقاء على العقد",
                "description": "المطالبة بتعويض دون فسخ.",
                "supporting_evidence": ["E2"],
                "conflicting_evidence": [],
                "advantages": ["يحافظ على العلاقة التعاقدية"],
                "limitations": ["يحتاج إثبات الضرر"],
                "required_conditions": ["وجود ضرر فعلي"],
                "missing_information_affecting_it": [],
            },
        ],
        "conflicts": list(conflicts or ["نص يقيّد الفسخ ونص يجيزه"]),
        "undecidable": False,
        "undecidable_reason": "",
    }


REPORT_MARKDOWN = """# تحليل الحالة

## 1. فهم الحالة
نزاع حول تأخر التسليم.

## 2. النقاط الرئيسية
- ينص المستند على مدة تسليم محددة (E1).

## 3. النصوص والضوابط ذات العلاقة
ينص المستند على القاعدة (E1) وعلى قيد عليها (E2).

## 4. التحليل
وبناءً على ذلك، ينطبق على الحالة الشرط الوارد في (E1).

## 5. الحلول الممكنة
### الحل الأول
الفسخ (E1).
### الحل الثاني
التعويض (E2).

## 6. الحل الأنسب بحسب المستندات
التعويض (E2).

## 7. سبب الترجيح
لوجود نص مقيّد للفسخ (E2).

## 8. المعلومات الناقصة
لا يوجد نقص جوهري.

## 9. مستوى قوة الاستناد
متوسطة."""


def full_script(**overrides) -> dict:
    script = {
        UNDERSTAND: understanding(),
        PLAN: plan(),
        SOLUTIONS: solutions(),
        REPORT: REPORT_MARKDOWN,
        FOLLOWUP: "اعتمدت على النص الوارد في (E2).",
    }
    script.update(overrides)
    return script
