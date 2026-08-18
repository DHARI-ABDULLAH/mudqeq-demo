"""
web_demo/services/url_fetch_service.py
--------------------------------------
The only place the demo server makes an outbound HTTP request on a visitor's
behalf.

Every hop is re-validated
-------------------------
Redirects are followed MANUALLY (``allow_redirects=False``). Letting the HTTP
client follow them would validate the first URL and then silently follow a
``Location:`` header to anywhere — which is precisely the SSRF hole the checks
in ``url_security_service`` exist to close. Each hop goes back through
:func:`url_security_service.validate_url` before a socket is opened.

Everything is bounded
---------------------
- connect timeout and read timeout, separately
- maximum number of redirects
- maximum response size, enforced while STREAMING (a ``Content-Length`` header
  is a hint from an untrusted server, never a guarantee)
- content-type allow-list: HTML and plain text only

Failures are categorised, never raw. The caller gets an Arabic message and a
short reason code — never a traceback, never a raw provider error, and never
"no information found", which is a different thing entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests

from config import (
    MAX_URL_REDIRECTS,
    MAX_URL_RESPONSE_BYTES,
    URL_CONNECT_TIMEOUT,
    URL_READ_TIMEOUT,
    URL_USER_AGENT,
)
from core.logging_utils import log_event
from services import url_security_service
from services.url_security_service import UrlRejected

# --- Reason codes ----------------------------------------------------------
REASON_TIMEOUT = "timeout"
REASON_NETWORK = "network_error"
REASON_TOO_MANY_REDIRECTS = "too_many_redirects"
REASON_TOO_LARGE = "response_too_large"
REASON_CONTENT_TYPE = "unsupported_content_type"
REASON_PDF_URL = "pdf_url"
REASON_EMPTY_BODY = "empty_body"
REASON_HTTP_UNAUTHORIZED = "http_unauthorized"
REASON_HTTP_NOT_FOUND = "http_not_found"
REASON_HTTP_RATE_LIMIT = "http_rate_limit"
REASON_HTTP_CLIENT = "http_client_error"
REASON_HTTP_SERVER = "http_server_error"
REASON_BAD_REDIRECT = "bad_redirect"

_ARABIC_MESSAGES = {
    REASON_TIMEOUT: "انتهت مهلة الاتصال بالصفحة. حاول مرة أخرى أو جرّب رابطاً آخر.",
    REASON_NETWORK: "تعذّر الاتصال بالموقع. تأكد من الرابط وحاول مرة أخرى.",
    REASON_TOO_MANY_REDIRECTS: "الرابط يعيد التوجيه أكثر من الحد المسموح.",
    REASON_TOO_LARGE: "حجم الصفحة أكبر من الحد المسموح في النسخة التجريبية.",
    REASON_CONTENT_TYPE: (
        "نوع محتوى هذه الصفحة غير مدعوم. تدعم النسخة التجريبية صفحات الويب "
        "النصية (HTML) والنصوص العادية فقط."
    ),
    REASON_PDF_URL: (
        "هذا الرابط يشير إلى ملف PDF. نزّل الملف ثم أضفه عبر «رفع ملف» "
        "للحصول على أرقام الصفحات في الاستشهادات."
    ),
    REASON_EMPTY_BODY: "الصفحة لم ترجع أي محتوى.",
    REASON_HTTP_UNAUTHORIZED: (
        "الصفحة تتطلب تسجيل دخول أو صلاحية وصول، ولا يمكن جلبها."
    ),
    REASON_HTTP_NOT_FOUND: "الصفحة غير موجودة على الموقع (404).",
    REASON_HTTP_RATE_LIMIT: "الموقع رفض الطلب مؤقتاً بسبب كثرة الطلبات. حاول لاحقاً.",
    REASON_HTTP_CLIENT: "رفض الموقع الطلب ولم تُجلب الصفحة.",
    REASON_HTTP_SERVER: "الموقع يواجه مشكلة حالياً ولم تُجلب الصفحة. حاول لاحقاً.",
    REASON_BAD_REDIRECT: "الرابط يعيد التوجيه إلى وجهة غير مسموح بها.",
}

# --- Content types ---------------------------------------------------------
HTML_CONTENT_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "application/xhtml"}
)
TEXT_CONTENT_TYPES = frozenset({"text/plain", "text/markdown"})
PDF_CONTENT_TYPES = frozenset({"application/pdf", "application/x-pdf"})

_STREAM_BLOCK = 32 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class UrlFetchFailed(Exception):
    """A fetch that failed for an infrastructure reason, with Arabic text."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(
            message or _ARABIC_MESSAGES.get(reason, _ARABIC_MESSAGES[REASON_NETWORK])
        )
        self.reason = reason

    @property
    def user_message(self) -> str:
        return str(self)


@dataclass
class FetchResult:
    original_url: str
    final_url: str
    status_code: int = 200
    content_type: str = ""
    charset: str = ""
    text: str = ""
    num_bytes: int = 0
    truncated: bool = False
    redirects: list[str] = field(default_factory=list)

    @property
    def is_html(self) -> bool:
        return self.content_type in HTML_CONTENT_TYPES


def message_for(reason: str) -> str:
    return _ARABIC_MESSAGES.get(reason, _ARABIC_MESSAGES[REASON_NETWORK])


# --- Helpers ---------------------------------------------------------------
def parse_content_type(raw: str) -> tuple[str, str]:
    """Split a Content-Type header into ``(mime, charset)``, both lower-case."""
    if not raw:
        return "", ""
    parts = [p.strip() for p in str(raw).split(";")]
    mime = parts[0].lower()
    charset = ""
    for param in parts[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip('"').lower()
    return mime, charset


def classify_status(status: int) -> str:
    """Map an HTTP status onto a reason code (only called for failures)."""
    if status in (401, 403):
        return REASON_HTTP_UNAUTHORIZED
    if status == 404:
        return REASON_HTTP_NOT_FOUND
    if status == 429:
        return REASON_HTTP_RATE_LIMIT
    if 500 <= status < 600:
        return REASON_HTTP_SERVER
    return REASON_HTTP_CLIENT


def require_supported_content_type(mime: str) -> str:
    """Return the mime type if the demo can read it, else raise.

    A PDF URL gets its own reason so the UI can point the user at the upload
    form (where page-numbered citations come from) instead of a generic error.
    """
    if mime in HTML_CONTENT_TYPES or mime in TEXT_CONTENT_TYPES:
        return mime
    if mime in PDF_CONTENT_TYPES:
        raise UrlFetchFailed(REASON_PDF_URL)
    # An absent header is common on small servers; HTML is the safe assumption
    # and extraction will reject it anyway if it turns out to be binary.
    if not mime:
        return "text/html"
    raise UrlFetchFailed(REASON_CONTENT_TYPE)


def _read_bounded(response, limit: int) -> tuple[bytes, bool]:
    """Stream at most ``limit`` bytes. Returns ``(body, hit_limit)``."""
    collected = bytearray()
    for block in response.iter_content(chunk_size=_STREAM_BLOCK):
        if not block:
            continue
        collected.extend(block)
        if len(collected) > limit:
            return bytes(collected[:limit]), True
    return bytes(collected), False


def _decode(body: bytes, charset: str, response) -> str:
    """Decode the body, preferring the declared charset then the sniffed one."""
    for candidate in (charset, getattr(response, "encoding", None), "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


# --- Fetch -----------------------------------------------------------------
def fetch(
    raw_url: str,
    *,
    session_id: str | None = None,
    max_bytes: int = MAX_URL_RESPONSE_BYTES,
    max_redirects: int = MAX_URL_REDIRECTS,
) -> FetchResult:
    """Fetch a user-supplied URL safely and return its decoded text.

    Raises :class:`url_security_service.UrlRejected` when the destination is
    not allowed, and :class:`UrlFetchFailed` when the request itself failed.
    Both carry an Arabic, user-safe message.
    """
    max_bytes = max(1024, int(max_bytes))
    max_redirects = max(0, int(max_redirects))

    original = url_security_service.normalize_url(raw_url)
    current = original
    redirects: list[str] = []

    headers = {
        "User-Agent": URL_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        "Accept-Language": "ar,en;q=0.8",
    }
    timeout = (float(URL_CONNECT_TIMEOUT), float(URL_READ_TIMEOUT))

    for _ in range(max_redirects + 1):
        # Re-validated on EVERY hop: the previous response chose this address.
        validated = url_security_service.validate_url(current)

        try:
            response = requests.get(
                validated.url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.Timeout as exc:
            raise UrlFetchFailed(REASON_TIMEOUT) from exc
        except requests.exceptions.TooManyRedirects as exc:
            raise UrlFetchFailed(REASON_TOO_MANY_REDIRECTS) from exc
        except requests.exceptions.RequestException as exc:
            raise UrlFetchFailed(REASON_NETWORK) from exc
        except OSError as exc:  # socket-level failure below the client
            raise UrlFetchFailed(REASON_NETWORK) from exc

        with response:
            status = int(getattr(response, "status_code", 0) or 0)

            if status in _REDIRECT_STATUSES:
                location = (response.headers or {}).get("Location") or ""
                if not location.strip():
                    raise UrlFetchFailed(REASON_BAD_REDIRECT)
                redirects.append(validated.url)
                current = urljoin(validated.url, location.strip())
                continue

            if status >= 400:
                raise UrlFetchFailed(classify_status(status))

            mime, charset = parse_content_type(
                (response.headers or {}).get("Content-Type", "")
            )
            mime = require_supported_content_type(mime)

            declared = (response.headers or {}).get("Content-Length")
            try:
                if declared is not None and int(declared) > max_bytes:
                    raise UrlFetchFailed(REASON_TOO_LARGE)
            except (TypeError, ValueError):
                pass  # a malformed header proves nothing; the stream cap rules

            body, hit_limit = _read_bounded(response, max_bytes)
            if hit_limit:
                raise UrlFetchFailed(REASON_TOO_LARGE)
            if not body:
                raise UrlFetchFailed(REASON_EMPTY_BODY)

            text = _decode(body, charset, response)
            log_event(
                "url_fetch",
                session_id,
                status="ok",
                http_status=status,
                size_bytes=len(body),
                attempts=len(redirects) + 1,
            )
            return FetchResult(
                original_url=original,
                final_url=validated.url,
                status_code=status,
                content_type=mime,
                charset=charset,
                text=text,
                num_bytes=len(body),
                redirects=redirects,
            )

    raise UrlFetchFailed(REASON_TOO_MANY_REDIRECTS)


def describe_failure(exc: BaseException) -> tuple[str, str]:
    """Return ``(reason_code, arabic_message)`` for any fetch-stage failure."""
    if isinstance(exc, (UrlFetchFailed, UrlRejected)):
        return exc.reason, str(exc)
    return REASON_NETWORK, _ARABIC_MESSAGES[REASON_NETWORK]
