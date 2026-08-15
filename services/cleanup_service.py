"""
web_demo/services/cleanup_service.py
------------------------------------
TTL-based cleanup of expired session directories.

Resilience: expiry is derived from filesystem modification time, so cleanup
keeps working across process restarts (the in-memory session registry is not
required). A best-effort background thread runs periodically, and callers can
also trigger an opportunistic sweep on each request.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from config import DEMO_STORAGE_ROOT, SESSION_TTL_MINUTES
from core.logging_utils import log_event
from services import security, session_service

_sweeper_started = False
_sweeper_lock = threading.Lock()
_last_sweep = 0.0
_MIN_SWEEP_INTERVAL = 60.0  # seconds between opportunistic sweeps


def _dir_expired(path: Path, ttl_seconds: int, now: float) -> bool:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (now - mtime) > ttl_seconds


def sweep(force: bool = False) -> int:
    """Delete expired session directories. Returns count removed."""
    global _last_sweep
    now = time.time()
    if not force and (now - _last_sweep) < _MIN_SWEEP_INTERVAL:
        return 0
    _last_sweep = now

    root = DEMO_STORAGE_ROOT
    if not root.exists():
        return 0

    ttl = SESSION_TTL_MINUTES * 60
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # Only touch directories that look like our session ids.
        if not security.is_valid_id(child.name):
            continue
        if _dir_expired(child, ttl, now):
            shutil.rmtree(child, ignore_errors=True)
            session_service._drop_record(child.name)
            removed += 1
    if removed:
        log_event("cleanup_sweep", status="ok", chunks=removed)
    return removed


def _loop() -> None:
    interval = max(30, min(300, SESSION_TTL_MINUTES * 60 // 2 or 60))
    while True:
        time.sleep(interval)
        try:
            sweep(force=True)
        except Exception:  # noqa: BLE001 - never let the sweeper die loudly
            pass


def start_background_sweeper() -> None:
    """Start the periodic sweeper exactly once per process."""
    global _sweeper_started
    with _sweeper_lock:
        if _sweeper_started:
            return
        _sweeper_started = True
    t = threading.Thread(target=_loop, name="mudqeq-demo-cleanup", daemon=True)
    t.start()
