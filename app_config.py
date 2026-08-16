"""
web_demo/app_config.py
----------------------
Safe config accessors for app.py.

Streamlit Community Cloud has been observed serving a stale config.py
alongside a newer app.py (cached/partial deploy). These helpers always
return sensible values even when newer config attributes are missing.
"""

from __future__ import annotations

import config as _cfg


def _int(name: str, default: int) -> int:
    return int(getattr(_cfg, name, default))


def top_k_min() -> int:
    return _int("TOP_K_MIN", 2)


def top_k_max() -> int:
    return max(_int("TOP_K_MAX", 10), top_k_min())


def top_k_default() -> int:
    if hasattr(_cfg, "TOP_K_DEFAULT"):
        return int(_cfg.TOP_K_DEFAULT)
    return min(max(_int("TOP_K", 4), top_k_min()), top_k_max())


def clamp_top_k(value: int) -> int:
    return min(max(int(value), top_k_min()), top_k_max())


def search_default_results() -> int:
    return _int("SEARCH_DEFAULT_RESULTS", 8)


def search_max_results() -> int:
    return max(_int("SEARCH_MAX_RESULTS", 20), 1)


def max_files_per_session() -> int:
    return _int("MAX_FILES_PER_SESSION", 5)
