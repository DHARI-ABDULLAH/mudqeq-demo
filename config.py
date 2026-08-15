"""
web_demo/config.py
------------------
Single source of truth for the PUBLIC WEB DEMO of "المدقق الشامل" (Mudqeq AI).

This module is completely independent from the desktop application's
`core/config.py`. It never touches the desktop storage tree, Application
Support, the production SQLite DB, or local Ollama.

Every tunable is read from an environment variable with a conservative,
demo-safe default. The ONLY value without a default is GROQ_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

_WEB_DEMO_ROOT = Path(__file__).resolve().parent


def _load_local_dotenv() -> None:
    """Load ``web_demo/.env`` for local development (optional).

    - Only reads ``web_demo/.env`` next to this file — never repo-root ``.env``.
    - Does not override variables already set in the shell (export / HF Secret).
    - ``python-dotenv`` is optional; without it, use ``export GROQ_API_KEY=...``.
    """
    env_file = _WEB_DEMO_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:
        pass


_load_local_dotenv()

# --- Product identity -----------------------------------------------------
APP_NAME_AR = "المدقق الشامل"
APP_NAME_EN = "Mudqeq AI"
APP_TAGLINE_AR = "نسخة تجريبية عامة — تحليل المستندات بالذكاء الاصطناعي"
DEMO_VERSION = "1.0.0-demo"


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an int env var and clamp it to [minimum, maximum]."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
    return max(minimum, min(maximum, value))


# --- Temporary, ephemeral storage (NEVER the desktop storage tree) --------
# Root for all per-session working directories. Defaults to the system temp
# area so nothing is ever persisted alongside the repository.
DEMO_STORAGE_ROOT = Path(
    os.environ.get("DEMO_STORAGE_ROOT", "/tmp/mudqeq_demo")
).resolve()

# --- Upload safety limits -------------------------------------------------
MAX_FILE_SIZE_MB = _int_env("MAX_FILE_SIZE_MB", 10, minimum=1, maximum=50)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PAGES = _int_env("MAX_PAGES", 50, minimum=1, maximum=500)
MAX_FILES_PER_SESSION = _int_env("MAX_FILES_PER_SESSION", 1, minimum=1, maximum=10)

# --- Abuse / rate control -------------------------------------------------
MAX_QUESTIONS_PER_SESSION = _int_env(
    "MAX_QUESTIONS_PER_SESSION", 20, minimum=1, maximum=1000
)
MAX_UPLOADS_PER_SESSION = _int_env(
    "MAX_UPLOADS_PER_SESSION", 5, minimum=1, maximum=1000
)

# --- Session lifetime -----------------------------------------------------
SESSION_TTL_MINUTES = _int_env("SESSION_TTL_MINUTES", 30, minimum=1, maximum=1440)

# --- RAG / retrieval ------------------------------------------------------
TOP_K = _int_env("TOP_K", 4, minimum=1, maximum=10)
MAX_RAG_CONTEXT_CHARS = _int_env(
    "MAX_RAG_CONTEXT_CHARS", 6000, minimum=500, maximum=30000
)

# --- Chunking -------------------------------------------------------------
CHUNK_TARGET_CHARS = _int_env("CHUNK_TARGET_CHARS", 1200, minimum=300, maximum=4000)
CHUNK_OVERLAP_CHARS = _int_env("CHUNK_OVERLAP_CHARS", 200, minimum=0, maximum=1000)

# A page with fewer than this many extractable characters is treated as
# "no usable text" (likely scanned). We never run OCR in the demo.
OCR_CHAR_THRESHOLD = _int_env("OCR_CHAR_THRESHOLD", 50, minimum=1, maximum=1000)

# Hard cap on total extracted characters to protect server memory.
MAX_EXTRACTED_CHARS = _int_env(
    "MAX_EXTRACTED_CHARS", 2_000_000, minimum=10_000, maximum=20_000_000
)
MAX_CHUNKS = _int_env("MAX_CHUNKS", 4000, minimum=10, maximum=50_000)

# --- Embeddings -----------------------------------------------------------
# Same model as the desktop product. Runs locally on the demo server.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")

# --- Groq (hosted LLM) ----------------------------------------------------
# NEVER hardcode the key. Sources (first match wins):
#   1. OS environment variable (export / platform env)
#   2. web_demo/.env via python-dotenv (local dev only)
#   3. Streamlit Community Cloud Secrets (st.secrets["GROQ_API_KEY"])
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = os.environ.get(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")


def _read_secret(name: str) -> str:
    """Read a secret from env or Streamlit Cloud secrets (never log the value)."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001 — st.secrets unavailable outside Streamlit
        pass
    return ""


def get_groq_api_key() -> str:
    return _read_secret("GROQ_API_KEY")


def get_groq_model() -> str:
    return _read_secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL


# Backward-compatible module-level name (resolved at import; prefer getters).
GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = get_groq_model()

# LLM generation controls.
GROQ_MAX_OUTPUT_TOKENS = _int_env(
    "GROQ_MAX_OUTPUT_TOKENS", 1024, minimum=64, maximum=8192
)
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.2") or "0.2")
GROQ_CONNECT_TIMEOUT = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "10") or "10")
GROQ_READ_TIMEOUT = float(os.environ.get("GROQ_READ_TIMEOUT", "60") or "60")
GROQ_MAX_RETRIES = _int_env("GROQ_MAX_RETRIES", 2, minimum=0, maximum=5)

# --- Question length guard ------------------------------------------------
MAX_QUESTION_CHARS = _int_env("MAX_QUESTION_CHARS", 2000, minimum=10, maximum=10000)


def groq_is_configured() -> bool:
    """True only if an API key is present (model always has a default)."""
    return bool(get_groq_api_key())


def ensure_storage_root() -> Path:
    """Create the ephemeral storage root (safe, idempotent)."""
    DEMO_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return DEMO_STORAGE_ROOT
