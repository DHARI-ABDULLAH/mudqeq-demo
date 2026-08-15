"""
web_demo/tests/conftest.py
--------------------------
Test configuration for the web demo. Sets a temporary ephemeral storage root
and conservative limits BEFORE any demo module is imported, so config picks
them up. Never touches the desktop storage tree.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make ``web_demo/`` importable as the package root (import config, services...).
WEB_DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_DEMO_ROOT))

# Ephemeral, isolated storage for the whole test session.
_TMP = tempfile.mkdtemp(prefix="mudqeq_demo_test_")
os.environ["DEMO_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MAX_FILE_SIZE_MB", "2")
os.environ.setdefault("MAX_PAGES", "5")
os.environ.setdefault("MAX_QUESTIONS_PER_SESSION", "3")
os.environ.setdefault("SESSION_TTL_MINUTES", "1")
# Ensure no key leaks in from the developer environment during tests.
os.environ.pop("GROQ_API_KEY", None)


@pytest.fixture(scope="session")
def storage_root() -> Path:
    return Path(_TMP)


@pytest.fixture()
def new_session():
    """Create a fresh, isolated session and tear it down afterwards."""
    from services import security, session_service

    sid = security.new_id()
    session_service.get_or_create(sid)
    yield sid
    session_service.destroy(sid)
