"""Tests for app_config fallbacks when config.py is stale on deploy."""

from __future__ import annotations

import types

import app_config


def test_top_k_fallbacks_without_new_config_attrs(monkeypatch):
    """Simulate Streamlit Cloud serving an old config.py (only TOP_K)."""
    stale = types.SimpleNamespace(TOP_K=4)
    monkeypatch.setattr(app_config, "_cfg", stale)

    assert app_config.top_k_min() == 2
    assert app_config.top_k_max() == 10
    assert app_config.top_k_default() == 4
    assert app_config.clamp_top_k(99) == 10
    assert app_config.search_default_results() == 8
    assert app_config.search_max_results() == 20
    assert app_config.max_files_per_session() == 5


def test_top_k_default_uses_config_when_present(monkeypatch):
    stale = types.SimpleNamespace(
        TOP_K=4, TOP_K_MIN=2, TOP_K_MAX=10, TOP_K_DEFAULT=6
    )
    monkeypatch.setattr(app_config, "_cfg", stale)
    assert app_config.top_k_default() == 6
