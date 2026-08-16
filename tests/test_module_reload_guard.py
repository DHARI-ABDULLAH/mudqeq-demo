"""Tests for the stale-module reload guard in app.py.

Hosted Streamlit re-executes app.py from disk each rerun while keeping imported
modules cached in the process. The guard must drop those cached demo modules
exactly once per deployed version of app.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
STAMP_ATTR = "_mudqeq_demo_source_stamp"


def _load_guard():
    """Exec only the guard block from app.py (importing app.py would run the UI)."""
    src = APP_PATH.read_text(encoding="utf-8")
    start = src.index("_DEMO_PACKAGES =")
    # Stop at the call site, not the `def` line that contains the same name.
    end = src.index("\n_reload_demo_modules_if_updated()")
    preamble = "import hashlib, sys, threading\nfrom pathlib import Path\n"
    ns: dict = {"__file__": str(APP_PATH)}
    exec(preamble + src[start:end], ns)  # noqa: S102 - test-only, trusted source
    return ns["_reload_demo_modules_if_updated"]


@pytest.fixture
def guard(monkeypatch):
    fn = _load_guard()
    monkeypatch.delattr(sys, STAMP_ATTR, raising=False)
    yield fn
    if hasattr(sys, STAMP_ATTR):
        delattr(sys, STAMP_ATTR)


def test_first_run_stamps_the_process(guard):
    guard()
    assert getattr(sys, STAMP_ATTR, None)


def test_unchanged_source_does_not_purge(guard, monkeypatch):
    guard()
    sentinel = types.ModuleType("config")
    sentinel.MARKER = "kept"
    monkeypatch.setitem(sys.modules, "config", sentinel)

    guard()

    assert sys.modules["config"].MARKER == "kept"


def test_changed_source_purges_demo_modules(guard, monkeypatch):
    guard()
    for name in ["config", "services.session_service", "ui.components", "core.x"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    unrelated = types.ModuleType("numpy_lookalike")
    monkeypatch.setitem(sys.modules, "numpy_lookalike", unrelated)

    # Simulate a new deploy: app.py hash differs from the stored stamp.
    monkeypatch.setattr(sys, STAMP_ATTR, "stamp-from-a-previous-deploy")
    guard()

    for name in ["config", "services.session_service", "ui.components", "core.x"]:
        assert name not in sys.modules, f"{name} should have been purged"
    assert "numpy_lookalike" in sys.modules, "third-party modules must be untouched"


def test_reimport_after_purge_yields_current_source(guard, monkeypatch):
    guard()
    stale = types.ModuleType("config")
    monkeypatch.setitem(sys.modules, "config", stale)
    assert not hasattr(sys.modules["config"], "TOP_K_MIN")

    monkeypatch.setattr(sys, STAMP_ATTR, "stamp-from-a-previous-deploy")
    guard()

    import config as fresh

    assert hasattr(fresh, "TOP_K_MIN")
    assert hasattr(fresh, "MAX_FILES_PER_SESSION")
