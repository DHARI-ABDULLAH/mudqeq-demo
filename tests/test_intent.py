"""Deterministic intent classification for overview vs factual questions."""

from __future__ import annotations

import pytest

from core import intent

ARABIC_OVERVIEW = [
    "لخص المستند",
    "لخص الملف",
    "لخّص المستند من فضلك",
    "شنو داخل المستند",
    "شنو داخل الملف؟",
    "وش داخل الملف",
    "ايش داخل المستند",
    "عن ماذا يتحدث المستند",
    "عن ماذا يتحدث هذا الملف؟",
    "ما محتوى هذا المستند",
    "محتوى المستند",
    "أعطني نبذة",
    "اعطني ملخص",
    "نظرة عامة",
    "ما هو موضوع المستند",
]

ENGLISH_OVERVIEW = [
    "summarize the document",
    "summarize the pdf",
    "Summarise this file",
    "give me a summary",
    "what is inside the pdf",
    "what inside the pdf",
    "what's inside the document",
    "what is this document about",
    "what is this pdf about",
    "give me an overview",
    "tell me about this document",
    "what are the main points",
    "TL;DR",
]

FACTUAL = [
    "which port allows inbound SSH?",
    "ما هو رقم المنفذ المستخدم للاتصال؟",
    "what is the price of an m5.large instance",
    "من هو المؤلف المذكور في الصفحة الثالثة؟",
    "list the security group rules for port 443",
    "متى تم إصدار النسخة الثانية؟",
    "how do I attach an EBS volume",
    "ما هي شروط عقد الإجارة المذكورة؟",
]


@pytest.mark.parametrize("question", ARABIC_OVERVIEW)
def test_arabic_overview_questions(question):
    assert intent.classify(question) == intent.OVERVIEW, question


@pytest.mark.parametrize("question", ENGLISH_OVERVIEW)
def test_english_overview_questions(question):
    assert intent.classify(question) == intent.OVERVIEW, question


@pytest.mark.parametrize("question", FACTUAL)
def test_factual_questions_stay_factual(question):
    assert intent.classify(question) == intent.FACTUAL, question


def test_empty_question_is_factual():
    assert intent.classify("") == intent.FACTUAL
    assert intent.classify("   ") == intent.FACTUAL


def test_classification_is_deterministic():
    for _ in range(5):
        assert intent.classify("لخص المستند") == intent.OVERVIEW
        assert intent.classify("which port?") == intent.FACTUAL


def test_normalization_handles_diacritics_and_variants():
    assert intent.normalize("لَخِّصْ") == intent.normalize("لخص")
    assert intent.normalize("الأمر") == intent.normalize("الامر")
