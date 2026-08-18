"""
web_demo/tests/url_fakes.py
---------------------------
A scripted stand-in for the outbound HTTP call made by ``url_fetch_service``.

No test in this suite touches the real network. Two seams are faked:

1. ``requests.get`` — replaced by a router that serves canned responses per URL
   and records every request that was actually issued (which is what the
   redirect-validation assertions inspect).
2. ``url_security_service.resolve_host`` — replaced by a DNS table, so a test
   can decide exactly what a hostname resolves to. Tests that exercise the SSRF
   rules point a public-looking name at a private address on purpose.

Everything downstream of the socket — validation, size caps, decoding,
extraction, chunking, embeddings, FAISS, retrieval, citations — runs for real.
"""

from __future__ import annotations

import ipaddress
import sys

from services import url_fetch_service, url_security_service

PUBLIC_IP = "93.184.216.34"


class FakeHeaders(dict):
    """Case-insensitive header mapping, like the real client returns."""

    def __init__(self, mapping=None) -> None:
        super().__init__({str(k).lower(): v for k, v in (mapping or {}).items()})

    def get(self, key, default=None):  # noqa: D102
        return super().get(str(key).lower(), default)

    def __contains__(self, key) -> bool:  # noqa: D105
        return super().__contains__(str(key).lower())


class FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(
        self,
        body: bytes | str = b"",
        *,
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
        headers: dict | None = None,
        encoding: str | None = "utf-8",
        chunk_size: int = 4096,
    ) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.body = body
        self.status_code = status_code
        self.encoding = encoding
        self._chunk_size = chunk_size
        merged = {}
        if content_type:
            merged["Content-Type"] = content_type
        merged.update(headers or {})
        self.headers = FakeHeaders(merged)
        self.closed = False

    def iter_content(self, chunk_size: int = 8192):
        step = max(1, int(chunk_size or self._chunk_size))
        for start in range(0, len(self.body), step):
            yield self.body[start : start + step]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False


def redirect(location: str, status_code: int = 302) -> FakeResponse:
    """A redirect response pointing at ``location``."""
    return FakeResponse(
        b"",
        status_code=status_code,
        content_type="text/html",
        headers={"Location": location},
    )


class Router:
    """Serves canned responses by URL and records every issued request."""

    def __init__(self, routes: dict) -> None:
        self.routes = dict(routes)
        self.requests: list[dict] = []

    def __call__(self, url, **kwargs):
        self.requests.append({"url": url, **kwargs})
        outcome = self.routes.get(url)
        if outcome is None:
            outcome = self.routes.get("*")
        if outcome is None:
            raise AssertionError(f"unrouted URL: {url}")
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome) and not isinstance(outcome, FakeResponse):
            outcome = outcome(url, **kwargs)
        return outcome

    @property
    def urls(self) -> list[str]:
        return [r["url"] for r in self.requests]


def live_modules() -> tuple:
    """The demo modules currently in ``sys.modules``.

    ``app.py`` drops cached demo modules whenever its source changes, so a
    Streamlit run can leave the test process holding stale module objects.
    Tests that drive the UI must patch the modules the *app* is using, which is
    what this returns.
    """
    return (
        sys.modules.get("services.url_fetch_service", url_fetch_service),
        sys.modules.get("services.url_security_service", url_security_service),
    )


def install_http(monkeypatch, routes: dict, fetch_module=None) -> Router:
    """Point the URL fetcher at a scripted HTTP router."""
    router = Router(routes)
    monkeypatch.setattr((fetch_module or url_fetch_service).requests, "get", router)
    return router


def install_dns(
    monkeypatch,
    mapping: dict | None = None,
    default=PUBLIC_IP,
    security_module=None,
) -> None:
    """Replace DNS resolution with a fixed table.

    ``default=None`` makes every unlisted hostname fail to resolve, which is how
    the DNS-failure path is exercised.
    """
    target = security_module or url_security_service
    table = dict(mapping or {})

    def fake_resolve(hostname: str) -> list[str]:
        host = (hostname or "").strip().lower().rstrip(".")
        if not host:
            raise target.UrlRejected(target.REASON_NO_HOST)
        # A literal address is never looked up — mirror the real behaviour so
        # the IP-class rules still see exactly what the user typed.
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return [host]
        if host in table:
            value = table[host]
            return list(value) if isinstance(value, (list, tuple)) else [value]
        if default is None:
            raise target.UrlRejected(target.REASON_DNS)
        return [default]

    monkeypatch.setattr(target, "resolve_host", fake_resolve)


def install(
    monkeypatch,
    routes: dict,
    dns: dict | None = None,
    dns_default=PUBLIC_IP,
    *,
    live: bool = False,
) -> Router:
    """Install both seams at once. Returns the HTTP router.

    Pass ``live=True`` from tests that drive the Streamlit app, so the patch
    lands on the modules the running app imported.
    """
    fetch_module, security_module = live_modules() if live else (None, None)
    install_dns(monkeypatch, dns, default=dns_default, security_module=security_module)
    return install_http(monkeypatch, routes, fetch_module=fetch_module)


def html_page(
    title: str,
    body: str,
    *,
    with_chrome: bool = True,
    lang: str = "ar",
) -> str:
    """A realistic page: the article wrapped in the usual site furniture."""
    chrome_head = (
        """
        <nav class="site-nav"><a href="/">الرئيسية</a><a href="/about">من نحن</a></nav>
        <div class="cookie-banner">هذا الموقع يستخدم ملفات تعريف الارتباط. موافق؟</div>
        <div id="advert-top">إعلان ممول: اشترك الآن في خدمتنا المميزة اليوم</div>
        """
        if with_chrome
        else ""
    )
    chrome_foot = (
        """
        <aside class="sidebar"><h4>مقالات ذات صلة</h4><p>مقال آخر لا علاقة له</p></aside>
        <footer class="site-footer">جميع الحقوق محفوظة 2026</footer>
        <div style="display:none">نص مخفي يجب ألا يُفهرس إطلاقاً</div>
        """
        if with_chrome
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>.hidden {{ display:none }} body {{ font-family: sans-serif }}</style>
  <script>window.tracker = "should never be indexed";</script>
</head>
<body>
  {chrome_head}
  <main>
    <article>
      {body}
    </article>
  </main>
  {chrome_foot}
  <noscript>يرجى تفعيل الجافاسكربت لعرض الموقع</noscript>
</body>
</html>"""


def long_body(paragraph: str, times: int = 12) -> str:
    return "\n".join(f"<p>{paragraph} ({i + 1})</p>" for i in range(times))
