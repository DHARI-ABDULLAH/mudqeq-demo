"""
web_demo/config.py
------------------
Single source of truth for the PUBLIC WEB DEMO of "المدقق الشامل" (Mudqeq AI).

This module is completely independent from the desktop application's
`core/config.py`. It never touches the desktop storage tree, Application
Support, the production SQLite DB, or local Ollama.

Every tunable is read from an environment variable with a conservative,
demo-safe default. The ONLY value without a default is OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_WEB_DEMO_ROOT = Path(__file__).resolve().parent


def _load_local_dotenv() -> None:
    """Load ``web_demo/.env`` for local development (optional).

    - Only reads ``web_demo/.env`` next to this file — never repo-root ``.env``.
    - Does not override variables already set in the shell (export / HF Secret).
    - ``python-dotenv`` is optional; without it, use ``export OPENAI_API_KEY=...``.
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
MAX_FILE_SIZE_MB = _int_env("MAX_FILE_SIZE_MB", 50, minimum=1, maximum=100)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PAGES = _int_env("MAX_PAGES", 200, minimum=1, maximum=500)
# The demo supports MULTIPLE documents per session (like the desktop app),
# bounded to protect the shared public server.
MAX_FILES_PER_SESSION = _int_env("MAX_FILES_PER_SESSION", 5, minimum=1, maximum=20)

# --- Abuse / rate control -------------------------------------------------
MAX_QUESTIONS_PER_SESSION = _int_env(
    "MAX_QUESTIONS_PER_SESSION", 20, minimum=1, maximum=1000
)
MAX_UPLOADS_PER_SESSION = _int_env(
    "MAX_UPLOADS_PER_SESSION", 15, minimum=1, maximum=1000
)

# --- Session lifetime -----------------------------------------------------
SESSION_TTL_MINUTES = _int_env("SESSION_TTL_MINUTES", 30, minimum=1, maximum=1440)

# --- RAG / retrieval ------------------------------------------------------
# Bounds for the chat Top-K control (mirrors the desktop settings slider).
TOP_K_MIN = _int_env("TOP_K_MIN", 2, minimum=1, maximum=10)
TOP_K_MAX = _int_env("TOP_K_MAX", 10, minimum=TOP_K_MIN, maximum=20)
# Canonical Top-K used as the retrieval default across services and the UI.
TOP_K = _int_env("TOP_K", 4, minimum=1, maximum=10)
# Slider-safe default: always guaranteed to fall within [TOP_K_MIN, TOP_K_MAX]
# so the Streamlit slider can never receive an out-of-range initial value.
TOP_K_DEFAULT = min(max(TOP_K, TOP_K_MIN), TOP_K_MAX)
# Search page has its own results control (independent of chat Top-K).
SEARCH_DEFAULT_RESULTS = _int_env("SEARCH_DEFAULT_RESULTS", 8, minimum=1, maximum=50)
SEARCH_MAX_RESULTS = _int_env("SEARCH_MAX_RESULTS", 20, minimum=1, maximum=100)
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

# --- Case Analysis ("تحليل حالة") ----------------------------------------
# A case analysis is a multi-step pipeline (understand -> plan -> retrieve ->
# evidence -> solutions -> report). Every stage is hard-bounded so a single
# case can never fan out into unbounded retrieval or provider spend.
MAX_CASE_CHARS = _int_env("MAX_CASE_CHARS", 6000, minimum=100, maximum=20_000)
MAX_CASE_RESEARCH_QUERIES = _int_env(
    "MAX_CASE_RESEARCH_QUERIES", 6, minimum=1, maximum=10
)
MIN_CASE_RESEARCH_QUERIES = _int_env(
    "MIN_CASE_RESEARCH_QUERIES", 3, minimum=1, maximum=MAX_CASE_RESEARCH_QUERIES
)
MAX_RESULTS_PER_QUERY = _int_env("MAX_RESULTS_PER_QUERY", 5, minimum=1, maximum=20)
MAX_TOTAL_EVIDENCE_CHUNKS = _int_env(
    "MAX_TOTAL_EVIDENCE_CHUNKS", 18, minimum=1, maximum=60
)
MAX_CASE_CONTEXT_CHARS = _int_env(
    "MAX_CASE_CONTEXT_CHARS", 14_000, minimum=1_000, maximum=60_000
)
# Worst case per successful analysis: understand + plan + solutions + report + verify.
MAX_CASE_LLM_CALLS = _int_env("MAX_CASE_LLM_CALLS", 5, minimum=1, maximum=8)
# Case analysis costs far more than one chat question, so it has its own quota
# instead of silently draining MAX_QUESTIONS_PER_SESSION.
MAX_CASES_PER_SESSION = _int_env("MAX_CASES_PER_SESSION", 3, minimum=1, maximum=50)
MAX_CASE_FOLLOWUPS_PER_CASE = _int_env(
    "MAX_CASE_FOLLOWUPS_PER_CASE", 5, minimum=1, maximum=20
)


def upload_limits_caption_ar() -> str:
    """Arabic upload hint for the UI — always matches server-side validation."""
    return (
        f"PDF فقط، بحد أقصى {MAX_FILE_SIZE_MB} ميغابايت و"
        f"{MAX_PAGES} صفحة لكل ملف."
    )

# --- Embeddings -----------------------------------------------------------
# Same model as the desktop product. Runs locally on the demo server.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")

# --- OpenAI (hosted LLM) --------------------------------------------------
# NEVER hardcode the key. Sources (first match wins):
#   1. OS environment variable (export / platform env)
#   2. web_demo/.env via python-dotenv (local dev only)
#   3. Streamlit Community Cloud Secrets (st.secrets["OPENAI_API_KEY"])
LLM_PROVIDER = "OpenAI"

# Non-reasoning, low-cost, 128K context — it honours `temperature` and follows
# the strict "answer only from context" instructions this demo depends on.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Optional override for Azure/proxy deployments. Empty means the SDK default.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")

# Fallback locations when operators nest secrets under a TOML section instead of
# placing OPENAI_* keys at the top level (a common Streamlit Cloud mistake).
_OPENAI_KEY_NESTED_PATHS: tuple[tuple[str, ...], ...] = (
    ("secrets", "OPENAI_API_KEY"),
    ("secrets", "OPENAI_KEY"),
    ("openai", "api_key"),
    ("openai", "API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("openai", "OPENAI_KEY"),
    ("OPENAI", "API_KEY"),
    ("OPENAI", "OPENAI_API_KEY"),
)
_OPENAI_MODEL_NESTED_PATHS: tuple[tuple[str, ...], ...] = (
    ("secrets", "OPENAI_MODEL"),
    ("openai", "model"),
    ("openai", "OPENAI_MODEL"),
    ("OPENAI", "MODEL"),
    ("OPENAI", "OPENAI_MODEL"),
)
_OPENAI_KEY_ALIASES: frozenset[str] = frozenset(
    {"OPENAI_API_KEY", "OPENAI_KEY", "openai_api_key", "api_key", "API_KEY"}
)
_OPENAI_KEY_ENV_ALIASES: tuple[str, ...] = ("OPENAI_API_KEY", "OPENAI_KEY")


def _env_non_empty(name: str) -> str:
    """Return a stripped env var, treating blank/whitespace as missing."""
    raw = os.environ.get(name)
    if raw is None:
        return ""
    return raw.strip()


def _is_secret_mapping(node: object) -> bool:
    return isinstance(node, Mapping) and not isinstance(node, (str, bytes))


def _streamlit_secrets_status() -> dict[str, str]:
    """Safe metadata about ``st.secrets`` — key names only, never values."""
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return {
            "loaded": "no",
            "key_names": "",
            "hint": "streamlit unavailable",
        }

    try:
        secrets = st.secrets
        names: list[str] = []
        for top in secrets.keys():
            names.append(str(top))
            try:
                child = secrets[top]
            except KeyError:
                continue
            if _is_secret_mapping(child):
                for sub in child.keys():
                    names.append(f"{top}.{sub}")

        hint = ""
        flat = set(names)
        nested_flat = {n.split(".", 1)[-1] for n in names if "." in n}
        has_openai = any(n in _OPENAI_KEY_ALIASES for n in (*flat, *nested_flat))
        if "GROQ_API_KEY" in flat and not has_openai:
            hint = (
                "وُجد GROQ_API_KEY لكن لا يوجد OPENAI_API_KEY — "
                "حدّث Secrets في Streamlit Cloud."
            )
        elif not names:
            hint = (
                "لا توجد Secrets محمّلة — أضف OPENAI_API_KEY في "
                "App Settings → Secrets ثم Reboot."
            )
        elif not has_openai:
            hint = (
                "Secrets موجودة لكن OPENAI_API_KEY غير موجود. "
                f"المفاتيح الحالية: {', '.join(names)}"
            )

        return {
            "loaded": "yes",
            "key_names": ", ".join(names) if names else "(none)",
            "hint": hint,
        }
    except StreamlitSecretNotFoundError:
        return {
            "loaded": "no",
            "key_names": "",
            "hint": (
                "ملف Secrets غير موجود — الصق OPENAI_API_KEY في "
                "App Settings → Secrets ثم Reboot."
            ),
        }
    except Exception:  # noqa: BLE001
        return {
            "loaded": "error",
            "key_names": "",
            "hint": "تعذّر قراءة st.secrets.",
        }


def _scan_secrets_for_openai_key() -> str:
    """Scan top-level and one-level nested ``st.secrets`` for OpenAI key aliases."""
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return ""

    try:
        secrets = st.secrets
        for alias in _OPENAI_KEY_ALIASES:
            try:
                value = secrets[alias]
            except KeyError:
                value = None
            if value is not None and not _is_secret_mapping(value):
                text = str(value).strip()
                if text:
                    return text

        for top in secrets.keys():
            try:
                child = secrets[top]
            except KeyError:
                continue
            if not _is_secret_mapping(child):
                continue
            for alias in _OPENAI_KEY_ALIASES:
                try:
                    value = child[alias]
                except KeyError:
                    continue
                if not _is_secret_mapping(value):
                    text = str(value).strip()
                    if text:
                        return text
    except StreamlitSecretNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass
    return ""


def _read_openai_api_key_from_streamlit() -> str:
    for alias in ("OPENAI_API_KEY", "OPENAI_KEY"):
        text = _read_from_streamlit_secrets(alias, nested_paths=_OPENAI_KEY_NESTED_PATHS)
        if text:
            return text
    return _scan_secrets_for_openai_key()


def _walk_secrets(node: object, path: tuple[str, ...]) -> str:
    """Follow a nested ``st.secrets`` path; return a scalar string or ``""``."""
    for part in path:
        try:
            if isinstance(node, dict):
                node = node[part]
            elif hasattr(node, "__getitem__"):
                node = node[part]  # type: ignore[index]
            else:
                node = getattr(node, part)
        except (KeyError, AttributeError, TypeError):
            return ""
    if isinstance(node, dict):
        return ""
    return str(node).strip()


def _read_from_streamlit_secrets(
    name: str, *, nested_paths: tuple[tuple[str, ...], ...] = ()
) -> str:
    """Read a scalar secret from ``st.secrets`` (top-level or nested).

    Never logs or raises. Returns ``""`` when Streamlit is unavailable, secrets
    are missing, or the key cannot be resolved.
    """
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return ""

    try:
        secrets = st.secrets
        try:
            value = secrets[name]
            if not _is_secret_mapping(value):
                text = str(value).strip()
                if text:
                    return text
        except KeyError:
            pass

        for path in nested_paths:
            text = _walk_secrets(secrets, path)
            if text:
                return text
    except StreamlitSecretNotFoundError:
        pass
    except Exception:  # noqa: BLE001 — runtime not ready, malformed secrets, etc.
        pass
    return ""


def _read_secret(name: str) -> str:
    """Read a secret from env or Streamlit Cloud secrets (never log the value)."""
    if name == "OPENAI_API_KEY":
        for alias in _OPENAI_KEY_ENV_ALIASES:
            if val := _env_non_empty(alias):
                return val
        return _read_openai_api_key_from_streamlit()

    val = _env_non_empty(name)
    if val:
        return val

    nested: tuple[tuple[str, ...], ...] = ()
    if name == "OPENAI_MODEL":
        nested = _OPENAI_MODEL_NESTED_PATHS

    return _read_from_streamlit_secrets(name, nested_paths=nested)


def _detect_api_key_source() -> str:
    """Return where the OpenAI key was found — never the key itself."""
    for alias in _OPENAI_KEY_ENV_ALIASES:
        if _env_non_empty(alias):
            return "environment"
    if _read_openai_api_key_from_streamlit():
        return "st.secrets"
    return "missing"


def openai_config_diagnostics() -> dict[str, str]:
    """Safe OpenAI configuration metadata for the UI/diagnostics panel."""
    model = get_openai_model()
    source = _detect_api_key_source()
    detected = source != "missing"
    secrets_status = streamlit_secrets_status()
    return {
        "openai_configured": "yes" if detected else "no",
        "openai_model": model,
        "api_key_detected": "yes" if detected else "no",
        "api_key_source": source,
        "streamlit_secrets_loaded": secrets_status["loaded"],
        "secret_key_names": secrets_status["key_names"],
        "configuration_hint": secrets_status["hint"],
    }


def bootstrap_streamlit_secrets() -> None:
    """Promote ``st.secrets`` into ``os.environ`` early in the Streamlit runtime.

    Streamlit Community Cloud injects secrets through ``st.secrets``. Promotion
    to ``os.environ`` normally happens on first parse, but importing ``config``
    inside cached modules can race that on hosted deploys. Calling this once
    per script run (before any OpenAI check) makes Cloud secrets visible to
    ``get_openai_api_key()`` reliably.
    """
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return

    def _promote(name: str, raw: object) -> None:
        if isinstance(raw, (str, int, float)):
            text = str(raw).strip()
            if text and not _env_non_empty(name):
                os.environ[name] = text

    try:
        secrets = st.secrets
        for top in secrets.keys():
            try:
                value = secrets[top]
            except KeyError:
                continue
            if _is_secret_mapping(value):
                for sub in value.keys():
                    try:
                        _promote(str(sub), value[sub])
                    except KeyError:
                        continue
            else:
                _promote(str(top), value)
    except StreamlitSecretNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass


def streamlit_secrets_status() -> dict[str, str]:
    """Public wrapper for safe Streamlit secrets metadata."""
    return _streamlit_secrets_status()


def get_openai_api_key() -> str:
    return _read_secret("OPENAI_API_KEY")


def get_openai_model() -> str:
    return _read_secret("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


# LLM generation controls.
OPENAI_MAX_OUTPUT_TOKENS = _int_env(
    "OPENAI_MAX_OUTPUT_TOKENS", 1024, minimum=64, maximum=8192
)


def _temperature_env() -> float | None:
    """Read OPENAI_TEMPERATURE; an empty value omits the parameter entirely.

    Reasoning-family models reject `temperature`, so operators must be able to
    switch it off without editing code.
    """
    raw = os.environ.get("OPENAI_TEMPERATURE", "0.2")
    if raw is None or raw.strip() == "":
        return None
    try:
        return max(0.0, min(2.0, float(raw)))
    except (TypeError, ValueError):
        return 0.2


OPENAI_TEMPERATURE = _temperature_env()
OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60") or "60")
# Retries apply ONLY to transient 5xx/timeout/network failures. Rate limits are
# never retried: on a shared demo key that would just multiply the throttling.
OPENAI_MAX_RETRIES = _int_env("OPENAI_MAX_RETRIES", 1, minimum=0, maximum=3)

# --- Question length guard ------------------------------------------------
MAX_QUESTION_CHARS = _int_env("MAX_QUESTION_CHARS", 2000, minimum=10, maximum=10000)


def openai_is_configured() -> bool:
    """True only if an API key is present (model always has a default)."""
    return bool(get_openai_api_key())


def ensure_storage_root() -> Path:
    """Create the ephemeral storage root (safe, idempotent)."""
    DEMO_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return DEMO_STORAGE_ROOT
