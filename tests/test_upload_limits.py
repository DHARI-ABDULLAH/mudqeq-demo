"""Production upload limits: 50 MB per file, 200 pages per file."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from config import upload_limits_caption_ar
from services import security
from tests.pdf_util import make_pdf

PROD_SIZE_MB = 50
PROD_PAGES = 200


@pytest.fixture
def prod_limits(monkeypatch):
    """Apply production upload limits on imported config/security bindings."""
    size_bytes = PROD_SIZE_MB * 1024 * 1024
    monkeypatch.setattr(config, "MAX_FILE_SIZE_MB", PROD_SIZE_MB)
    monkeypatch.setattr(config, "MAX_FILE_SIZE_BYTES", size_bytes)
    monkeypatch.setattr(config, "MAX_PAGES", PROD_PAGES)
    monkeypatch.setattr(security, "MAX_FILE_SIZE_MB", PROD_SIZE_MB)
    monkeypatch.setattr(security, "MAX_FILE_SIZE_BYTES", size_bytes)
    monkeypatch.setattr(security, "MAX_PAGES", PROD_PAGES)


def test_pdf_at_max_size_accepted(prod_limits):
    pdf = make_pdf(["size check"])
    target = PROD_SIZE_MB * 1024 * 1024
    padding = target - len(pdf)
    assert padding > 0
    data = pdf + b"0" * padding
    assert len(data) == target
    security.validate_upload(data, "at-limit.pdf")


def test_pdf_over_max_size_rejected(prod_limits):
    over = PROD_SIZE_MB * 1024 * 1024 + 1
    data = b"%PDF-1.4\n" + b"0" * (over - 9)
    assert len(data) == over
    with pytest.raises(security.UploadRejected) as exc:
        security.validate_upload(data, "too-big.pdf")
    assert "50" in str(exc.value)


def test_pdf_at_max_pages_accepted(prod_limits):
    data = make_pdf([f"page {i}" for i in range(PROD_PAGES)])
    security.validate_upload(data, "max-pages.pdf")


def test_pdf_over_max_pages_rejected(prod_limits):
    data = make_pdf([f"page {i}" for i in range(PROD_PAGES + 1)])
    with pytest.raises(security.UploadRejected) as exc:
        security.validate_upload(data, "too-long.pdf")
    assert "200" in str(exc.value)


def test_ui_caption_shows_prod_limits(prod_limits):
    caption = upload_limits_caption_ar()
    assert caption == "PDF فقط، بحد أقصى 50 ميغابايت و200 صفحة لكل ملف."


def test_app_upload_caption_helper(prod_limits, monkeypatch):
    import app as demo_app

    # app.py may re-import config on first load; keep both bindings in sync.
    size_bytes = PROD_SIZE_MB * 1024 * 1024
    for mod in (config, demo_app.config):
        monkeypatch.setattr(mod, "MAX_FILE_SIZE_MB", PROD_SIZE_MB)
        monkeypatch.setattr(mod, "MAX_FILE_SIZE_BYTES", size_bytes)
        monkeypatch.setattr(mod, "MAX_PAGES", PROD_PAGES)

    assert demo_app._upload_limits_caption() == upload_limits_caption_ar()


def test_streamlit_max_upload_size_at_least_50():
    cfg = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("maxUploadSize"):
            _, _, value = stripped.partition("=")
            assert int(value.strip()) >= 50
            return
    pytest.fail("maxUploadSize not found in .streamlit/config.toml")
