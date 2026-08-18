"""The demo must stand alone: no desktop backend, no Ollama, no path escapes.

These guard the deployment contract for Streamlit Community Cloud, where only
``web_demo/`` is published and nothing else exists on the host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB_DEMO_ROOT = Path(__file__).resolve().parent.parent

# Only the shipped application is scanned. Test files are excluded because they
# must be able to name the very things they forbid.
SOURCE_FILES = [
    p
    for p in WEB_DEMO_ROOT.rglob("*.py")
    if not {"__pycache__", ".git", "tests"} & set(p.parts)
]


def _strip_comments_and_docstrings(text: str) -> str:
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return re.sub(r"#.*", "", text)


@pytest.mark.parametrize("forbidden", ["8765", "11434", "127.0.0.1:", "localhost:"])
def test_no_local_backend_endpoints(forbidden):
    offenders = []
    for path in SOURCE_FILES:
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        if forbidden in code:
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"{forbidden!r} referenced in: {offenders}"


def test_no_ollama_runtime_dependency():
    offenders = []
    for path in SOURCE_FILES:
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        if "ollama" in code.lower():
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"Ollama referenced in code: {offenders}"


def test_no_fastapi_or_uvicorn_dependency():
    requirements = (WEB_DEMO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "fastapi" not in requirements
    assert "uvicorn" not in requirements

    offenders = []
    for path in SOURCE_FILES:
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8")).lower()
        if "import fastapi" in code or "from fastapi" in code:
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders


def test_no_imports_from_the_desktop_application():
    """web_demo must never import the desktop project's packages."""
    banned = re.compile(
        r"^\s*(?:from|import)\s+(?:\.\.|backend|desktop|api|src_tauri)\b", re.M
    )
    offenders = []
    for path in SOURCE_FILES:
        if banned.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"desktop imports found in: {offenders}"


def test_storage_root_is_ephemeral_and_outside_the_repo():
    from config import DEMO_STORAGE_ROOT

    resolved = DEMO_STORAGE_ROOT.resolve()
    assert WEB_DEMO_ROOT.resolve() not in resolved.parents
    assert resolved != WEB_DEMO_ROOT.resolve()


def test_session_paths_cannot_escape_the_storage_root():
    from config import DEMO_STORAGE_ROOT
    from services import security, session_service

    sid = security.new_id()
    doc = security.new_id()
    session_service.get_or_create(sid)
    try:
        for path in (
            session_service.session_dir(sid),
            session_service.pdf_path(sid, doc),
            session_service.index_path(sid, doc),
            session_service.chunks_path(sid, doc),
        ):
            assert security.is_within(DEMO_STORAGE_ROOT, path)

        for evil in ["../escape", "..", "/etc/passwd", "a/../../b"]:
            with pytest.raises(security.UploadRejected):
                session_service.index_path(sid, evil)
    finally:
        session_service.destroy(sid)


def test_llm_adapter_targets_openai_only():
    from services import llm_service

    assert llm_service.PROVIDER_NAME == "OpenAI"
    source = (WEB_DEMO_ROOT / "services" / "llm_service.py").read_text(encoding="utf-8")
    assert "api.openai.com" not in source, "endpoint must come from the SDK/config"


def test_no_groq_runtime_dependency_remains():
    requirements = (WEB_DEMO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "groq" not in requirements.lower()
    assert "openai==" in requirements.lower()

    offenders = []
    for path in SOURCE_FILES:
        if path.name == "config.py":
            # config may mention the legacy Groq secret name in operator hints only.
            continue
        if "groq" in path.read_text(encoding="utf-8").lower():
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"Groq still referenced in: {offenders}"


def test_no_api_key_literals_in_source():
    # Groq (gsk_) and OpenAI (sk-...) shapes, plus any *_API_KEY assignment.
    patterns = [
        re.compile(r"gsk_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
        re.compile(r"API_KEY\s*=\s*[\"'][^\"']+[\"']"),
    ]
    offenders = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        if any(p.search(text) for p in patterns):
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"hardcoded API key in: {offenders}"


def test_api_key_is_only_read_through_the_config_accessor():
    """No module may reach for the raw environment variable on its own."""
    offenders = []
    for path in SOURCE_FILES:
        if path.name == "config.py":
            continue
        if "OPENAI_API_KEY" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(WEB_DEMO_ROOT)))
    assert not offenders, f"OPENAI_API_KEY read outside config.py: {offenders}"


def test_logging_whitelist_cannot_carry_a_key_or_content():
    from core import logging_utils

    forbidden = {"api_key", "key", "token", "text", "content", "question", "answer"}
    assert not forbidden & logging_utils._ALLOWED_FIELDS


def test_host_toolbar_actions_hidden_via_css():
    """Community Cloud GitHub/Edit/Star/Share render in stToolbarActions."""
    from ui import styles

    css = styles._CSS
    for selector in (
        '[data-testid="stToolbarActions"]',
        '[data-testid="stToolbarActionButton"]',
        '[data-testid="stToolbar"]',
    ):
        assert selector in css
    assert "display: none !important" in css


def test_streamlit_toolbar_mode_viewer():
    cfg = WEB_DEMO_ROOT / ".streamlit" / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    assert 'toolbarMode = "viewer"' in text
