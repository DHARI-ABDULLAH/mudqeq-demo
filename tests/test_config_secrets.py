"""Secret-loading tests for OpenAI configuration (no real keys, no network)."""

from __future__ import annotations

import pytest

import config as cfg


class _FakeSecrets:
    """Minimal stand-in for ``st.secrets`` with nested TOML layouts."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):
        if key not in self._data:
            raise KeyError(key)
        return self._data[key]


def test_env_non_empty_treats_blank_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert cfg._env_non_empty("OPENAI_API_KEY") == ""


def test_read_secret_prefers_non_empty_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-not-printed")
    assert cfg.get_openai_api_key() == "env-key-not-printed"
    assert cfg._detect_api_key_source() == "environment"


def test_empty_environment_falls_through_to_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")

    fake = _FakeSecrets({"OPENAI_API_KEY": "secret-from-st"})
    monkeypatch.setattr(cfg, "_read_from_streamlit_secrets", lambda name, **kw: "secret-from-st")

    assert cfg.get_openai_api_key() == "secret-from-st"


def test_top_level_streamlit_secret_is_read(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import streamlit as st

    monkeypatch.setattr(st, "secrets", _FakeSecrets({"OPENAI_API_KEY": "  cloud-key  "}))
    assert cfg._read_from_streamlit_secrets("OPENAI_API_KEY") == "cloud-key"


def test_nested_openai_section_secret_is_read(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import streamlit as st

    monkeypatch.setattr(st, "secrets", _FakeSecrets({"openai": {"api_key": "nested-key"}}))
    assert cfg.get_openai_api_key() == "nested-key"
    assert cfg._detect_api_key_source() == "st.secrets"


def test_streamlit_secret_not_found_is_handled(monkeypatch):
    from streamlit.errors import StreamlitSecretNotFoundError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Secrets:
        def __getitem__(self, key):
            raise StreamlitSecretNotFoundError("no secrets")

    class _Module:
        secrets = _Secrets()

    monkeypatch.setitem(__import__("sys").modules, "streamlit", _Module())
    assert cfg.get_openai_api_key() == ""


def test_openai_is_configured_false_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "_read_from_streamlit_secrets", lambda *a, **k: "")
    assert cfg.openai_is_configured() is False


def test_openai_model_defaults_without_secret(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(cfg, "_read_from_streamlit_secrets", lambda *a, **k: "")
    assert cfg.get_openai_model() == cfg.DEFAULT_OPENAI_MODEL


def test_openai_model_does_not_affect_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr(cfg, "_read_from_streamlit_secrets", lambda *a, **k: "")
    assert cfg.openai_is_configured() is False
    assert cfg.get_openai_model() == "gpt-4.1-mini"


def test_openai_config_diagnostics_never_exposes_key_material(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    diag = cfg.openai_config_diagnostics()
    blob = repr(diag)
    assert diag["openai_configured"] == "yes"
    assert diag["api_key_detected"] == "yes"
    assert diag["api_key_source"] == "environment"
    assert diag["openai_model"] == "gpt-4o-mini"
    assert "sk-super-secret-value" not in blob
    assert "sk-" not in blob


@pytest.mark.parametrize(
    "path, expected",
    [
        (("openai", "API_KEY"), "upper-nested"),
        (("OPENAI", "OPENAI_API_KEY"), "section-key"),
    ],
)
def test_additional_nested_openai_paths(monkeypatch, path, expected):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import streamlit as st

    nested: dict = {}
    node = nested
    for part in path[:-1]:
        node[part] = {}
        node = node[part]
    node[path[-1]] = expected

    monkeypatch.setattr(st, "secrets", _FakeSecrets(nested))
    assert cfg.get_openai_api_key() == expected
