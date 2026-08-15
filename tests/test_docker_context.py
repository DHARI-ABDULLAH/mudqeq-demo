"""Deployment-hygiene tests: private/production files must not ship, and no
secrets may be embedded in the web_demo source tree."""

from __future__ import annotations

from pathlib import Path

WEB_DEMO = Path(__file__).resolve().parent.parent

# Files/dirs that must NEVER exist inside the web_demo build context.
FORBIDDEN_NAMES = {
    "app.db",
    ".env",
    ".backup_baseline",
}
FORBIDDEN_DIRS = {"storage", "reports", "index"}
FORBIDDEN_SUFFIXES = {".pdf", ".faiss", ".pkl", ".dmg", ".app"}


def _iter_files():
    for p in WEB_DEMO.rglob("*"):
        # Ignore local dev secrets (gitignored, never deployed to HF).
        if p.name == ".env":
            continue
        parts = set(p.parts)
        if "__pycache__" in parts or ".venv" in parts or ".pytest_cache" in parts:
            continue
        yield p


def test_dockerignore_present_and_aggressive():
    di = WEB_DEMO / ".dockerignore"
    assert di.exists()
    text = di.read_text(encoding="utf-8")
    for needle in ["storage/", "data/private/", "*.faiss", "app.db", ".env", "*.pdf"]:
        assert needle in text, f"missing .dockerignore rule: {needle}"


def test_no_private_or_production_files_in_context():
    for p in _iter_files():
        name = p.name
        assert name not in FORBIDDEN_NAMES, f"forbidden file present: {p}"
        if p.is_dir():
            assert name not in FORBIDDEN_DIRS, f"forbidden dir present: {p}"
        if p.is_file():
            assert p.suffix.lower() not in FORBIDDEN_SUFFIXES, f"forbidden file type: {p}"


def _requirement_lines() -> list[str]:
    raw = (WEB_DEMO / "requirements.txt").read_text(encoding="utf-8").lower().splitlines()
    # Only actual dependency lines (ignore comments / blank lines).
    return [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]


def test_requirements_are_demo_scoped():
    lines = _requirement_lines()
    joined = "\n".join(lines)
    # Desktop-only server/packaging deps must not be pulled into the demo.
    for banned in ["uvicorn", "fastapi", "python-multipart", "pyinstaller"]:
        assert banned not in joined, f"desktop dependency leaked into demo: {banned}"
    for needed in ["streamlit", "pdfplumber", "sentence-transformers", "faiss-cpu"]:
        assert any(needed in ln for ln in lines), f"missing demo dependency: {needed}"


def test_no_ollama_dependency():
    """Web demo must not depend on Ollama or desktop stack."""
    lines = _requirement_lines()
    joined = "\n".join(lines)
    assert "ollama" not in joined


def test_no_hardcoded_secret_in_source():
    # Scan shipped source only (tests are excluded from the runtime image).
    for p in _iter_files():
        if "tests" in p.parts:
            continue
        if p.is_file() and p.suffix == ".py":
            text = p.read_text(encoding="utf-8", errors="ignore").replace("'", '"')
            # A real key literal would look like GROQ_API_KEY = "gsk_..."; keys
            # must always come from the environment.
            if 'GROQ_API_KEY = "' in text:
                assert "os.environ" in text
            assert "gsk_" not in text  # Groq key prefix must never be committed
