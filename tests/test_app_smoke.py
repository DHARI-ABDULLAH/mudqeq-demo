"""End-to-end UI smoke tests for the Streamlit demo.

These render app.py headlessly (no browser) and assert it starts cleanly and
survives stale-module scenarios like the ones seen on hosted deploys.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exception():
    at = _run()
    assert not at.exception, f"app raised: {at.exception}"


def test_sidebar_navigation_and_top_k_slider():
    at = _run()
    # Nav buttons for every page are present.
    labels = {b.label for b in at.sidebar.button}
    for expected in [
        "المحادثة",
        "تحليل حالة",
        "المستندات",
        "البحث",
        "حول النسخة التجريبية",
    ]:
        assert expected in labels

    # Retrieval control renders and is adjustable.
    assert len(at.sidebar.slider) == 1
    slider = at.sidebar.slider[0]
    assert slider.min <= slider.value <= slider.max
    slider.set_value(int(slider.max)).run()
    assert not at.exception


def test_empty_state_shows_upload_hint():
    at = _run()
    text = " ".join(i.value for i in at.info)
    assert "لا توجد مستندات" in text


def test_pages_render_without_exception():
    at = _run()
    for label in [
        "المحادثة",
        "تحليل حالة",
        "البحث",
        "حول النسخة التجريبية",
        "المستندات",
    ]:
        button = next(b for b in at.sidebar.button if b.label == label)
        button.click().run()
        assert not at.exception, f"page '{label}' raised: {at.exception}"


def _goto(at: AppTest, label: str) -> AppTest:
    next(b for b in at.sidebar.button if b.label == label).click().run()
    return at


def test_case_page_shows_its_input_and_button():
    at = _goto(_run(), "تحليل حالة")
    assert not at.exception

    # No documents yet, so the page stops at the upload hint.
    assert "لا توجد مستندات" in " ".join(i.value for i in at.info)

    headings = " ".join(str(m.value) for m in at.markdown)
    assert "تحليل حالة" in headings


def test_switching_between_chat_and_case_does_not_loop():
    """The mode toggle must settle on the chosen page, not ping-pong."""
    at = _goto(_run(), "تحليل حالة")
    assert at.session_state["page"] == "case"

    at = _goto(at, "المحادثة")
    assert at.session_state["page"] == "chat"
    assert at.session_state["interaction_mode"] == "chat"

    at = _goto(at, "تحليل حالة")
    assert at.session_state["page"] == "case"
    assert at.session_state["interaction_mode"] == "case"
    assert not at.exception


def test_case_state_starts_empty():
    at = _run()
    assert at.session_state["case_outcome"] is None
    assert at.session_state["case_state"] is None
    assert at.session_state["case_followups"] == []


def test_app_survives_stale_ui_components(monkeypatch):
    """Simulate a hosted deploy where ui.components predates dashboard()."""
    import ui.components as real

    stale = types.ModuleType("ui.components")
    # Old builds only had these; no dashboard()/document_card().
    for name in [
        "brand_sidebar",
        "sidebar_privacy",
        "page_header",
        "hero",
        "source_card",
        "excerpt",
    ]:
        setattr(stale, name, getattr(real, name))

    monkeypatch.setitem(sys.modules, "ui.components", stale)
    at = _run()
    assert not at.exception, f"stale-components run raised: {at.exception}"


def test_app_survives_stale_config(monkeypatch):
    """Simulate a config module missing newer retrieval attributes."""
    import config as real

    stale = types.ModuleType("config")
    for name in [
        "APP_NAME_AR",
        "APP_TAGLINE_AR",
        "DEMO_VERSION",
        "TOP_K",
        "MAX_FILE_SIZE_MB",
        "MAX_PAGES",
        "MAX_QUESTION_CHARS",
        "MAX_QUESTIONS_PER_SESSION",
        "SESSION_TTL_MINUTES",
        "openai_is_configured",
        "ensure_storage_root",
    ]:
        setattr(stale, name, getattr(real, name))
    # Deliberately no TOP_K_MIN / TOP_K_MAX / TOP_K_DEFAULT / SEARCH_* .

    monkeypatch.setitem(sys.modules, "config", stale)
    at = _run()
    assert not at.exception, f"stale-config run raised: {at.exception}"
    assert len(at.sidebar.slider) == 1
