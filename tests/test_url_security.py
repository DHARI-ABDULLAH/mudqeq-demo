"""SSRF and URL-validation tests — the most security-critical URL set.

Adding a link makes the SERVER issue an outbound request to an address a
stranger chose. These tests pin down what it is allowed to reach: public
http(s) endpoints on standard ports, and nothing else — not the loopback
interface, not RFC1918 space, not link-local (which is where cloud metadata
lives), and not via a redirect that starts public and lands internal.
"""

from __future__ import annotations

import pytest

from services import url_fetch_service, url_security_service
from services.url_security_service import UrlRejected
from tests import url_fakes


# --- Scheme allow-list ----------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "file:///etc/passwd",
        "file://localhost/etc/passwd",
        "ftp://example.com/file.txt",
        "data:text/html,<script>alert(1)</script>",
        "javascript:alert(document.cookie)",
        "gopher://example.com/",
        "ws://example.com/socket",
    ],
)
def test_non_http_schemes_are_rejected(raw):
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url(raw)
    assert exc.value.reason in (
        url_security_service.REASON_SCHEME,
        url_security_service.REASON_BLOCKED_HOST,
    )
    assert "http" in exc.value.user_message


def test_empty_url_is_rejected():
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url("   ")
    assert exc.value.reason == url_security_service.REASON_EMPTY


def test_valid_https_url_passes(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    validated = url_security_service.validate_url("https://example.com/article")
    assert validated.url == "https://example.com/article"
    assert validated.hostname == "example.com"
    assert validated.port == 443
    assert validated.is_https is True


def test_bare_domain_is_upgraded_to_https(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    validated = url_security_service.validate_url("example.com/rules")
    assert validated.url.startswith("https://")


def test_plain_http_is_allowed_but_marked(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    validated = url_security_service.validate_url("http://example.com/page")
    assert validated.port == 80
    assert validated.is_https is False


# --- Loopback and internal hostnames --------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "http://localhost/admin",
        "https://localhost/secret",
        "http://LOCALHOST/admin",
        "http://ip6-localhost/",
        "http://service.internal/keys",
        "http://printer.local/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_internal_hostnames_are_rejected(raw, monkeypatch):
    url_fakes.install_dns(monkeypatch)
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url(raw)
    assert exc.value.reason == url_security_service.REASON_BLOCKED_HOST


# --- Literal IP destinations ----------------------------------------------
@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.1.2.3",
        "0.0.0.0",
        "10.0.0.5",
        "10.255.255.254",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "169.254.0.1",
        "169.254.169.254",  # cloud metadata endpoint
        "[::1]",
        "[fe80::1]",
        "[fc00::1]",
        "[::ffff:127.0.0.1]",
        "[::ffff:10.0.0.1]",
    ],
)
def test_private_and_loopback_addresses_are_rejected(host, monkeypatch):
    url_fakes.install_dns(monkeypatch)
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url(f"http://{host}/")
    assert exc.value.reason == url_security_service.REASON_PRIVATE_IP


def test_cloud_metadata_endpoint_is_rejected_by_ip_class(monkeypatch):
    """169.254.169.254 must be refused as link-local, not by name-matching."""
    url_fakes.install_dns(monkeypatch)
    assert url_security_service.is_blocked_ip("169.254.169.254") is True
    with pytest.raises(UrlRejected):
        url_security_service.validate_url("http://169.254.169.254/latest/meta-data/")


def test_public_ip_literal_is_allowed(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    validated = url_security_service.validate_url("https://93.184.216.34/page")
    assert validated.addresses == ["93.184.216.34"]


# --- DNS-level defence ----------------------------------------------------
def test_public_name_resolving_to_a_private_address_is_rejected(monkeypatch):
    """The classic bypass: a public hostname with an RFC1918 A record."""
    url_fakes.install_dns(monkeypatch, {"evil.example.com": "10.0.0.7"})
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url("https://evil.example.com/page")
    assert exc.value.reason == url_security_service.REASON_PRIVATE_IP


def test_any_private_address_in_the_record_set_rejects_the_host(monkeypatch):
    """One bad address is enough — the client may pick any of them."""
    url_fakes.install_dns(
        monkeypatch, {"mixed.example.com": ["93.184.216.34", "192.168.0.9"]}
    )
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url("https://mixed.example.com/")
    assert exc.value.reason == url_security_service.REASON_PRIVATE_IP


def test_dns_failure_is_its_own_reason(monkeypatch):
    url_fakes.install_dns(monkeypatch, {}, default=None)
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url("https://nonexistent.example/")
    assert exc.value.reason == url_security_service.REASON_DNS
    assert "النطاق" in exc.value.user_message


# --- Ports ----------------------------------------------------------------
@pytest.mark.parametrize("port", [22, 3306, 5432, 6379, 9200, 8080])
def test_non_web_ports_are_rejected(port, monkeypatch):
    url_fakes.install_dns(monkeypatch)
    with pytest.raises(UrlRejected) as exc:
        url_security_service.validate_url(f"http://example.com:{port}/")
    assert exc.value.reason == url_security_service.REASON_PORT


def test_explicit_standard_ports_are_allowed(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    assert url_security_service.validate_url("https://example.com:443/x").port == 443
    assert url_security_service.validate_url("http://example.com:80/x").port == 80


# --- Redirect chain -------------------------------------------------------
def test_public_url_redirecting_to_a_private_address_is_rejected(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {
            "https://example.com/start": url_fakes.redirect("http://10.0.0.5/internal"),
        },
    )
    with pytest.raises(UrlRejected) as exc:
        url_fetch_service.fetch("https://example.com/start")
    assert exc.value.reason == url_security_service.REASON_PRIVATE_IP


def test_redirect_to_the_metadata_endpoint_is_rejected(monkeypatch):
    router = url_fakes.install(
        monkeypatch,
        {
            "https://example.com/start": url_fakes.redirect(
                "http://169.254.169.254/latest/meta-data/iam/"
            ),
        },
    )
    with pytest.raises(UrlRejected):
        url_fetch_service.fetch("https://example.com/start")
    # The forbidden hop must never have been requested.
    assert not any("169.254.169.254" in u for u in router.urls)


def test_redirect_to_loopback_hostname_is_rejected(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {"https://example.com/start": url_fakes.redirect("http://localhost/admin")},
    )
    with pytest.raises(UrlRejected) as exc:
        url_fetch_service.fetch("https://example.com/start")
    assert exc.value.reason == url_security_service.REASON_BLOCKED_HOST


def test_redirect_to_a_non_http_scheme_is_rejected(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {"https://example.com/start": url_fakes.redirect("file:///etc/passwd")},
    )
    with pytest.raises(UrlRejected) as exc:
        url_fetch_service.fetch("https://example.com/start")
    assert exc.value.reason == url_security_service.REASON_SCHEME


def test_allowed_redirect_is_followed_and_final_url_recorded(monkeypatch):
    page = url_fakes.html_page("الصفحة النهائية", "<p>" + "محتوى مفيد وكافٍ. " * 20 + "</p>")
    url_fakes.install(
        monkeypatch,
        {
            "https://example.com/old": url_fakes.redirect("https://example.com/new"),
            "https://example.com/new": url_fakes.FakeResponse(page),
        },
    )
    result = url_fetch_service.fetch("https://example.com/old")
    assert result.original_url == "https://example.com/old"
    assert result.final_url == "https://example.com/new"
    assert result.redirects == ["https://example.com/old"]


def test_redirect_loop_is_bounded(monkeypatch):
    url_fakes.install(
        monkeypatch,
        {
            "https://example.com/a": url_fakes.redirect("https://example.com/b"),
            "https://example.com/b": url_fakes.redirect("https://example.com/a"),
        },
    )
    with pytest.raises(url_fetch_service.UrlFetchFailed) as exc:
        url_fetch_service.fetch("https://example.com/a")
    assert exc.value.reason == url_fetch_service.REASON_TOO_MANY_REDIRECTS


# --- Canonicalization + duplicate identity --------------------------------
@pytest.mark.parametrize(
    "left, right",
    [
        ("https://Example.com/page", "https://example.com/page"),
        ("https://example.com/page#section", "https://example.com/page"),
        ("https://example.com/page/", "https://example.com/page"),
        ("https://example.com:443/page", "https://example.com/page"),
        ("  https://example.com/page  ", "https://example.com/page"),
    ],
)
def test_canonical_urls_collapse_equivalent_forms(left, right):
    assert url_security_service.canonical_url(left) == url_security_service.canonical_url(right)
    assert url_security_service.url_hash(left) == url_security_service.url_hash(right)


def test_query_string_makes_a_different_page():
    a = url_security_service.url_hash("https://example.com/p?id=1")
    b = url_security_service.url_hash("https://example.com/p?id=2")
    assert a != b


def test_http_and_https_are_not_the_same_source():
    assert url_security_service.url_hash(
        "http://example.com/p"
    ) != url_security_service.url_hash("https://example.com/p")


# --- Messages -------------------------------------------------------------
def test_rejection_messages_are_arabic_and_contain_no_traceback(monkeypatch):
    url_fakes.install_dns(monkeypatch)
    for raw in ("http://localhost/x", "ftp://example.com/x", "http://10.0.0.1/x"):
        with pytest.raises(UrlRejected) as exc:
            url_security_service.validate_url(raw)
        message = exc.value.user_message
        assert message.strip()
        assert "Traceback" not in message
        assert "Error" not in message
