"""
web_demo/services/url_security_service.py
-----------------------------------------
Server-side request forgery (SSRF) defence for user-supplied URLs.

Adding a link means the DEMO SERVER makes an outbound request to an address a
stranger chose. Without this module that is a request-forgery primitive: the
visitor cannot reach the cloud metadata endpoint or an internal service, but
the server can, and it would happily hand back the response as "page content".

What is enforced, in order
--------------------------
1. Scheme allow-list — only ``http`` and ``https``. ``file:``, ``ftp:``,
   ``data:``, ``javascript:``, ``gopher:`` and everything else are rejected by
   absence, not by blocklist.
2. Port allow-list — the standard web ports only.
3. Hostname blocklist — loopback names and known metadata hostnames.
4. DNS resolution BEFORE connecting, with every returned address checked. A
   public name that resolves to a private address is rejected.
5. IP class checks — loopback, private, link-local (which covers the cloud
   metadata address), reserved, multicast, and unspecified are all refused,
   for IPv4 and IPv6, including IPv4-mapped IPv6 forms.

The redirect chain is validated by re-running :func:`validate_url` on every hop
(see ``url_fetch_service``), so a public URL cannot bounce the server to an
internal one.

Known limitation: this module resolves DNS and the HTTP client resolves it
again, so a name whose record changes between the two lookups (DNS rebinding)
is not fully closed off. Closing it needs connection-level IP pinning, which
would break TLS hostname verification; the residual window is documented rather
than papered over.

This module performs DNS lookups only. It never opens an HTTP connection.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from config import URL_ALLOWED_PORTS

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Reason codes are internal; users see the Arabic message on the exception.
REASON_EMPTY = "empty_url"
REASON_MALFORMED = "malformed_url"
REASON_SCHEME = "unsupported_scheme"
REASON_PORT = "unsupported_port"
REASON_NO_HOST = "missing_host"
REASON_BLOCKED_HOST = "blocked_host"
REASON_DNS = "dns_failure"
REASON_PRIVATE_IP = "private_address"

# Hostnames that always denote the local machine or a cloud metadata service.
# Written without a port so this stays a hostname list, never an endpoint.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

# Suffixes that only ever name something inside a private network.
BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".home.arpa",
    ".localdomain",
)

_ARABIC_MESSAGES = {
    REASON_EMPTY: "يرجى إدخال رابط الصفحة.",
    REASON_MALFORMED: "الرابط غير صالح. تأكد من كتابته بشكل صحيح.",
    REASON_SCHEME: "يُسمح فقط بروابط تبدأ بـ http:// أو https://.",
    REASON_PORT: "يُسمح فقط بالمنافذ القياسية للويب (80 و443).",
    REASON_NO_HOST: "الرابط لا يحتوي على اسم نطاق صالح.",
    REASON_BLOCKED_HOST: "لا يمكن استخدام روابط تشير إلى الخادم نفسه أو إلى شبكة داخلية.",
    REASON_DNS: "تعذّر العثور على اسم النطاق. تأكد من صحة الرابط واتصال الشبكة.",
    REASON_PRIVATE_IP: "لا يمكن استخدام روابط تشير إلى عنوان داخلي أو خاص.",
}


class UrlRejected(Exception):
    """Raised with an Arabic, user-safe message when a URL cannot be used."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or _ARABIC_MESSAGES.get(reason, _ARABIC_MESSAGES[REASON_MALFORMED]))
        self.reason = reason

    @property
    def user_message(self) -> str:
        return str(self)


@dataclass
class ValidatedUrl:
    """A URL that passed every check, plus what it resolved to."""

    url: str
    scheme: str
    hostname: str
    port: int
    addresses: list[str] = field(default_factory=list)

    @property
    def is_https(self) -> bool:
        return self.scheme == "https"


# --- Normalization ---------------------------------------------------------
def normalize_url(raw: str) -> str:
    """Trim, add a default scheme, and drop the fragment.

    A bare ``example.com/page`` is what people paste, so it is upgraded to
    ``https://`` rather than rejected — the scheme check still runs afterwards
    and still refuses anything that is not http(s).
    """
    text = str(raw or "").strip()
    # Strip bidirectional/zero-width marks a copied RTL link often carries.
    text = text.translate(dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff"), None))
    text = "".join(ch for ch in text if ch.isprintable() or ch == " ").strip()
    if not text:
        raise UrlRejected(REASON_EMPTY)

    if "://" not in text:
        if ":" in text.split("/", 1)[0]:
            # Something like "javascript:alert(1)" or "file:/etc/passwd".
            raise UrlRejected(REASON_SCHEME)
        text = f"https://{text}"

    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise UrlRejected(REASON_MALFORMED) from exc

    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))


def canonical_url(raw: str) -> str:
    """A stable identity for duplicate detection.

    Lower-cases scheme and host, drops the default port, the fragment, and a
    trailing slash on an otherwise empty path. Query strings are preserved:
    two URLs differing only by query are genuinely different pages.
    """
    try:
        parts = urlsplit(normalize_url(raw))
    except UrlRejected:
        raise
    host = (parts.hostname or "").lower().rstrip(".")
    port = parts.port
    scheme = parts.scheme.lower()
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, parts.query, ""))


def url_hash(raw: str) -> str:
    """SHA-256 of the canonical URL — the per-session duplicate key."""
    return hashlib.sha256(canonical_url(raw).encode("utf-8")).hexdigest()


# --- Address checks --------------------------------------------------------
def is_blocked_hostname(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in BLOCKED_HOSTNAMES:
        return True
    return any(host.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES)


def is_blocked_ip(address) -> bool:
    """True for any address the demo server must never be pointed at.

    Covers loopback, private ranges (10/8, 172.16/12, 192.168/16), link-local
    (169.254/16, which is where the cloud metadata endpoint lives), reserved,
    multicast, and unspecified — for IPv4 and IPv6 alike. IPv4-mapped IPv6 is
    unwrapped first so ``::ffff:127.0.0.1`` cannot slip past the IPv4 rules.
    """
    try:
        ip = ipaddress.ip_address(str(address).strip())
    except ValueError:
        return True

    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif getattr(ip, "sixtofour", None) is not None:
            ip = ip.sixtofour

    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def require_public_ip(address) -> str:
    if is_blocked_ip(address):
        raise UrlRejected(REASON_PRIVATE_IP)
    return str(address)


def resolve_host(hostname: str) -> list[str]:
    """Resolve a hostname to every address it currently maps to.

    A literal IP is returned as-is (no lookup). Raises :class:`UrlRejected`
    with ``dns_failure`` when the name does not resolve.
    """
    host = (hostname or "").strip().rstrip(".")
    if not host:
        raise UrlRejected(REASON_NO_HOST)

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return [host]

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.herror, UnicodeError, OSError) as exc:
        raise UrlRejected(REASON_DNS) from exc

    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and str(sockaddr[0]) not in addresses:
            addresses.append(str(sockaddr[0]))
    if not addresses:
        raise UrlRejected(REASON_DNS)
    return addresses


# --- Full validation -------------------------------------------------------
def validate_url(raw: str) -> ValidatedUrl:
    """Run every check and return the URL the fetcher may request.

    Raises :class:`UrlRejected` (Arabic message + reason code) otherwise. Used
    for the URL the user typed AND for every redirect hop.
    """
    url = normalize_url(raw)
    parts = urlsplit(url)

    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UrlRejected(REASON_SCHEME)

    hostname = (parts.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise UrlRejected(REASON_NO_HOST)

    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:  # non-numeric port in the netloc
        raise UrlRejected(REASON_MALFORMED) from exc
    if port not in URL_ALLOWED_PORTS:
        raise UrlRejected(REASON_PORT)

    if is_blocked_hostname(hostname):
        raise UrlRejected(REASON_BLOCKED_HOST)

    addresses = resolve_host(hostname)
    for address in addresses:
        require_public_ip(address)

    return ValidatedUrl(
        url=url,
        scheme=scheme,
        hostname=hostname,
        port=port,
        addresses=addresses,
    )


def message_for(reason: str) -> str:
    """Arabic, user-safe text for a rejection reason code."""
    return _ARABIC_MESSAGES.get(reason, _ARABIC_MESSAGES[REASON_MALFORMED])
