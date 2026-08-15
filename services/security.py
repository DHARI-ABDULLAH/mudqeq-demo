"""
web_demo/services/security.py
-----------------------------
Untrusted-input hardening for the public demo:

- secure id generation (session_id, document_id)
- safe display-filename derivation (never used as a filesystem path)
- path-containment checks (defeats ../, absolute paths, symlink escapes)
- strict PDF validation (signature, size, page count, encryption, corruption)

All user-facing messages are Arabic. Nothing here runs shell commands or
executes uploaded content.
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

import pdfplumber

from config import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, MAX_PAGES


class UploadRejected(Exception):
    """Raised with an Arabic, user-safe message when an upload is rejected."""


# --- Identifiers ----------------------------------------------------------
_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def new_id() -> str:
    """Cryptographically-random 32-char hex id (uuid4)."""
    return uuid.uuid4().hex


def is_valid_id(value: str) -> bool:
    """True only for our own generated id shape (defends path building)."""
    return bool(value) and isinstance(value, str) and bool(_HEX_RE.match(value))


def require_valid_id(value: str) -> str:
    if not is_valid_id(value):
        # Internal guard — not shown verbatim to users.
        raise UploadRejected("معرّف غير صالح.")
    return value


# --- Filenames ------------------------------------------------------------
def safe_display_filename(name: str | None) -> str:
    """Return a sanitized *display* filename.

    This value is for UI display and citations only. It is NEVER used to build
    a filesystem path (documents are addressed by generated ids).
    """
    if not name:
        return "document.pdf"
    # Strip any directory components an attacker may have embedded.
    name = Path(str(name)).name
    name = name.replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    if not name:
        name = "document.pdf"
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    return name[:120]


# --- Path containment -----------------------------------------------------
def is_within(base: Path, target: Path) -> bool:
    """True iff ``target`` resolves to a location inside ``base``.

    Resolves symlinks on both sides, so a symlink pointing outside ``base``
    is rejected.
    """
    try:
        base_r = base.resolve()
        target_r = target.resolve()
    except (OSError, RuntimeError):
        return False
    return base_r == target_r or base_r in target_r.parents


def safe_child_path(base: Path, *parts: str) -> Path:
    """Join ``parts`` under ``base`` and verify containment.

    Raises UploadRejected if the result would escape ``base``.
    """
    candidate = base.joinpath(*parts)
    if not is_within(base, candidate):
        raise UploadRejected("مسار غير صالح.")
    return candidate


# --- PDF validation -------------------------------------------------------
def validate_upload(file_bytes: bytes, original_name: str | None) -> None:
    """Validate an uploaded PDF given as bytes. Raises UploadRejected.

    Checks, in order: non-empty, extension, size, %PDF signature, parseable,
    not encrypted, page count within MAX_PAGES.
    """
    if not file_bytes:
        raise UploadRejected("الملف فارغ.")

    if original_name and not str(original_name).lower().strip().endswith(".pdf"):
        raise UploadRejected("الملف المرفوع ليس ملف PDF صالحاً.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise UploadRejected(
            f"حجم الملف أكبر من الحد المسموح في النسخة التجريبية "
            f"({MAX_FILE_SIZE_MB} ميغابايت)."
        )

    # Magic bytes: a real PDF starts with %PDF within the first bytes.
    if file_bytes[:4] != b"%PDF":
        raise UploadRejected("الملف المرفوع ليس ملف PDF صالحاً.")

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            num_pages = len(pdf.pages)
            if num_pages == 0:
                raise UploadRejected("ملف PDF لا يحتوي على أي صفحات.")
            if num_pages > MAX_PAGES:
                raise UploadRejected(
                    f"عدد صفحات المستند يتجاوز الحد المسموح في النسخة التجريبية "
                    f"({MAX_PAGES} صفحة)."
                )
    except UploadRejected:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            raise UploadRejected(
                "ملف PDF محمي بكلمة مرور ولا يمكن معالجته في النسخة التجريبية."
            ) from exc
        raise UploadRejected("تعذّر فتح ملف PDF؛ قد يكون تالفاً.") from exc
