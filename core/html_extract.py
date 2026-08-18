"""
web_demo/core/html_extract.py
-----------------------------
Readable-content extraction from a fetched HTML page.

Indexing raw HTML would fill the vector store with navigation, cookie banners,
and script bodies, and every one of those would then be eligible to come back
as a "citation". So the page is reduced to what a reader would actually read:
title, headings, paragraphs, lists, and tables that survive being flattened to
text.

Security posture
----------------
- Parsing only. Scripts are never executed, and ``<script>``/``<style>`` bodies
  are discarded rather than indexed.
- Output is plain text that stays UNTRUSTED downstream: instructions embedded
  in a page are indexed as ordinary source text and framed as data in prompts,
  exactly like PDF text.
- Pure Python (BeautifulSoup + the stdlib ``html.parser`` backend). No browser,
  no JavaScript engine, no network access from this module.

Server-rendered pages are the V1 target. A page whose body only appears after
client-side JavaScript yields no readable text, and the caller reports that
plainly instead of indexing an empty shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from config import MAX_URL_EXTRACTED_CHARS
from core.extraction import generic_clean

# Elements whose text is never article content under any layout. These are the
# only removals the conservative fallback performs.
_HARD_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "object",
        "embed",
        "form",
        "input",
        "select",
        "textarea",
        "button",
        "map",
        "audio",
        "video",
        "picture",
        "source",
        "track",
    }
)

# Structural chrome. Dropped in the normal pass, kept by the fallback because a
# site that puts its article inside one of these is wrong but not unheard of.
_DROP_TAGS = _HARD_DROP_TAGS | frozenset(
    {"nav", "footer", "aside", "dialog", "menu"}
)

# Elements the class/role heuristics may never remove. The document root and
# body routinely carry site-wide feature flags in their class list (Wikipedia's
# <html> advertises "…-main-menu-disabled"), and matching a chrome hint there
# would delete the entire page. Semantic content containers are protected for
# the same reason: whatever their class says, they are the article.
_PROTECTED_TAGS = frozenset({"html", "head", "body", "main", "article"})

# Substrings in class/id/role that mark site chrome rather than article text.
_BOILERPLATE_HINTS = (
    "cookie",
    "consent",
    "gdpr",
    "banner",
    "navbar",
    "navigation",
    "menu",
    "sidebar",
    "side-bar",
    "breadcrumb",
    "pagination",
    "paginator",
    "advert",
    "adsense",
    "sponsor",
    "promo",
    "newsletter",
    "subscribe",
    "social",
    "share",
    "sharing",
    "follow-us",
    "comment",
    "disqus",
    "popup",
    "modal",
    "overlay",
    "lightbox",
    "skip-link",
    "screen-reader",
    "sr-only",
    "visually-hidden",
    "site-header",
    "site-footer",
    "page-footer",
    "masthead",
    "toolbar",
    "widget",
    "related-posts",
    "backtotop",
)

# Roles that describe chrome. "main"/"article" are deliberately absent.
_BOILERPLATE_ROLES = frozenset(
    {
        "navigation",
        "banner",
        "complementary",
        "contentinfo",
        "search",
        "menu",
        "menubar",
        "toolbar",
        "dialog",
        "alertdialog",
        "tablist",
    }
)

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = frozenset(
    {"p", "li", "dt", "dd", "pre", "blockquote", "figcaption", "caption", "summary"}
)

_HIDDEN_STYLE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:;|$)", re.I
)
_WHITESPACE = re.compile(r"\s+")

# Very short leaf blocks are almost always UI labels ("More", "×", "Menu").
_MIN_BLOCK_CHARS = 2

# How much visible text a page must have before an empty extraction is treated
# as over-stripping rather than a genuinely empty page.
_FALLBACK_MIN_CHARS = 200


@dataclass
class Section:
    """A heading plus the readable blocks that follow it."""

    heading: str = ""
    blocks: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.blocks).strip()

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class HtmlExtractResult:
    title: str = ""
    sections: list[Section] = field(default_factory=list)
    truncated: bool = False

    @property
    def text(self) -> str:
        parts: list[str] = []
        for section in self.sections:
            if section.heading:
                parts.append(section.heading)
            if section.text:
                parts.append(section.text)
        return "\n\n".join(p for p in parts if p).strip()

    @property
    def total_chars(self) -> int:
        return len(self.text)

    def non_empty_sections(self) -> list[Section]:
        return [s for s in self.sections if s.text or s.heading]

    def has_usable_text(self) -> bool:
        return bool(self.text)


# --- Noise removal ---------------------------------------------------------
def _is_live(node) -> bool:
    """Whether ``node`` is a tag that still exists in the tree.

    Removing an element also removes its descendants, but a list built by
    ``find_all`` before the removal still holds them. Touching one afterwards
    raises, so every removal loop below skips nodes that are already gone.
    Real pages nest chrome inside chrome constantly, so this is the common
    case, not an edge case.
    """
    if not isinstance(node, Tag) or getattr(node, "decomposed", False):
        return False
    return node.attrs is not None


def _attr_blob(tag: Tag) -> str:
    """Lower-cased class + id + role of a tag, for boilerplate matching."""
    if not _is_live(tag):
        return ""
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    parts = [str(c) for c in classes]
    for key in ("id", "role", "data-testid"):
        value = tag.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _is_hidden(tag: Tag) -> bool:
    if not _is_live(tag):
        return False
    if (tag.name or "").lower() in _PROTECTED_TAGS:
        return False
    if tag.has_attr("hidden"):
        return True
    if str(tag.get("aria-hidden") or "").strip().lower() == "true":
        return True
    style = str(tag.get("style") or "")
    return bool(style and _HIDDEN_STYLE.search(style))


def _is_boilerplate(tag: Tag) -> bool:
    if not _is_live(tag):
        return False
    if (tag.name or "").lower() in _PROTECTED_TAGS:
        return False
    role = str(tag.get("role") or "").strip().lower()
    if role in _BOILERPLATE_ROLES:
        return True
    blob = _attr_blob(tag)
    return any(hint in blob for hint in _BOILERPLATE_HINTS)


def _strip_noise(soup: BeautifulSoup, *, conservative: bool = False) -> None:
    """Remove everything that is not candidate reading material, in place.

    ``conservative`` keeps only the unarguable removals (scripts, styles, form
    controls, media) and skips the class/role heuristics. It is the fallback
    used when the full strip leaves a text-rich page empty, because returning
    "no readable content" for a page that plainly has content is worse than
    keeping a little chrome.
    """
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    drop = _HARD_DROP_TAGS if conservative else _DROP_TAGS
    for tag in soup.find_all(list(drop)):
        if _is_live(tag):
            tag.decompose()

    if conservative:
        return

    # <header> is dropped only when it is the page masthead, not an in-article
    # header wrapping the title of the piece.
    for tag in soup.find_all("header"):
        if _is_live(tag) and tag.find_parent(["article", "main"]) is None:
            tag.decompose()

    for tag in soup.find_all(True):
        if not _is_live(tag):
            continue
        if _is_hidden(tag) or _is_boilerplate(tag):
            tag.decompose()


# --- Main-content selection ------------------------------------------------
def _text_length(tag: Tag) -> int:
    return len(_WHITESPACE.sub(" ", tag.get_text(" ", strip=True)))


def _pick_main(soup: BeautifulSoup) -> Tag:
    """Choose the densest plausible article container.

    Explicit semantics win when the page provides them; otherwise the candidate
    holding the most text does, which is a good proxy for "the article" once
    chrome has already been stripped.
    """
    for finder in (
        lambda: soup.find("main"),
        lambda: soup.find(attrs={"role": "main"}),
        lambda: soup.find("article"),
    ):
        try:
            node = finder()
        except (AttributeError, TypeError):  # pragma: no cover - parser guard
            node = None
        if isinstance(node, Tag) and _text_length(node) >= _MIN_BLOCK_CHARS:
            return node

    body = soup.body if isinstance(soup.body, Tag) else soup
    best, best_len = body, _text_length(body) if isinstance(body, Tag) else 0
    for tag in body.find_all(["article", "section", "div"], recursive=True):
        blob = _attr_blob(tag)
        if not any(k in blob for k in ("content", "article", "post", "entry", "main", "body")):
            continue
        length = _text_length(tag)
        # Require a clear majority of the page text before narrowing the scope,
        # so a promising class name on a small box cannot hide the real article.
        if length > best_len * 0.6 and length >= _MIN_BLOCK_CHARS:
            best, best_len = tag, max(best_len, length)
    return best if isinstance(best, Tag) else body


# --- Title -----------------------------------------------------------------
def _clean_inline(value: str) -> str:
    return _WHITESPACE.sub(" ", generic_clean(value or "")).strip()


def extract_title(soup: BeautifulSoup) -> str:
    """Best available page title: og:title, then <title>, then first <h1>."""
    meta = soup.find("meta", attrs={"property": "og:title"})
    if isinstance(meta, Tag):
        text = _clean_inline(str(meta.get("content") or ""))
        if text:
            return text[:300]

    if isinstance(soup.title, Tag):
        text = _clean_inline(soup.title.get_text(" ", strip=True))
        if text:
            return text[:300]

    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        text = _clean_inline(h1.get_text(" ", strip=True))
        if text:
            return text[:300]
    return ""


# --- Table flattening ------------------------------------------------------
def _flatten_table(table: Tag) -> list[str]:
    """Render a table as one text line per row, cells separated by " | "."""
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False) or row.find_all(["th", "td"])
        values = [_clean_inline(c.get_text(" ", strip=True)) for c in cells]
        values = [v for v in values if v]
        if values:
            lines.append(" | ".join(values))
    return lines


# --- Walk ------------------------------------------------------------------
def _iter_blocks(root: Tag):
    """Yield ``(kind, text)`` in document order.

    ``kind`` is ``"heading"`` or ``"block"``. Tables are flattened whole so a
    row never gets split away from its neighbours.
    """
    handled: set[int] = set()

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        if id(node) in handled:
            continue

        name = node.name.lower() if node.name else ""

        if name == "table":
            for line in _flatten_table(node):
                yield "block", line
            for inner in node.find_all(True):
                handled.add(id(inner))
            continue

        if name in _HEADING_TAGS:
            text = _clean_inline(node.get_text(" ", strip=True))
            if text:
                yield "heading", text
            continue

        if name in _BLOCK_TAGS:
            # A list item that only wraps nested lists contributes nothing on
            # its own; its children are visited in order anyway.
            text = _clean_inline(node.get_text(" ", strip=True))
            if len(text) >= _MIN_BLOCK_CHARS:
                yield "block", text
                for inner in node.find_all(list(_BLOCK_TAGS) + list(_HEADING_TAGS)):
                    handled.add(id(inner))
            continue

        if name in {"br", "hr"}:
            continue

        # Loose text directly under a container (common in hand-written HTML).
        direct = " ".join(
            str(child) for child in node.children if isinstance(child, NavigableString)
        )
        text = _clean_inline(direct)
        if len(text) >= _MIN_BLOCK_CHARS and name in {"div", "span", "section", "article"}:
            yield "block", text


def extract(html: str, *, max_chars: int = MAX_URL_EXTRACTED_CHARS) -> HtmlExtractResult:
    """Reduce an HTML document to ordered, readable sections.

    Stops once ``max_chars`` of readable text has been collected so a hostile
    or merely enormous page cannot exhaust server memory.

    If the normal pass finds nothing on a page that visibly has text, the
    document is re-read with only the unarguable removals applied. "No readable
    content" must mean the page really has none, not that one class name on a
    wrapper matched a chrome hint.
    """
    if not html or not html.strip():
        return HtmlExtractResult()

    result = _extract_once(html, max_chars=max_chars, conservative=False)
    if result.has_usable_text():
        return result

    if _visible_text_length(html) >= _FALLBACK_MIN_CHARS:
        fallback = _extract_once(html, max_chars=max_chars, conservative=True)
        if fallback.has_usable_text():
            return fallback
    return result


def _visible_text_length(html: str) -> int:
    """Rough size of the page's text once scripts and styles are ignored."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001
        return 0
    for tag in soup.find_all(["script", "style", "template", "noscript"]):
        if _is_live(tag):
            tag.decompose()
    return len(_WHITESPACE.sub(" ", soup.get_text(" ", strip=True)))


def _extract_once(
    html: str, *, max_chars: int, conservative: bool
) -> HtmlExtractResult:
    result = HtmlExtractResult()
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001 - malformed markup is a caller-level error
        return result

    result.title = extract_title(soup)
    _strip_noise(soup, conservative=conservative)
    root = _pick_main(soup)

    max_chars = max(1, int(max_chars))
    current = Section()
    sections: list[Section] = []
    used = 0
    seen_blocks: set[str] = set()

    def close_current() -> None:
        if current.heading or current.blocks:
            sections.append(Section(heading=current.heading, blocks=list(current.blocks)))

    for kind, text in _iter_blocks(root):
        if used >= max_chars:
            result.truncated = True
            break

        if kind == "heading":
            close_current()
            current = Section(heading=text[: max(1, max_chars - used)])
            used += len(current.heading)
            continue

        # Repeated identical lines are leftover navigation or per-item chrome.
        key = text[:160]
        if key in seen_blocks:
            continue
        seen_blocks.add(key)

        remaining = max_chars - used
        if len(text) > remaining:
            text = text[:remaining]
            result.truncated = True
        if text:
            current.blocks.append(text)
            used += len(text)

    close_current()

    # A page with no headings still has one implicit section holding its body.
    result.sections = [s for s in sections if s.text or s.heading]
    return result


def extract_plain_text(
    text: str, *, max_chars: int = MAX_URL_EXTRACTED_CHARS
) -> HtmlExtractResult:
    """Wrap a ``text/plain`` response in the same section shape as HTML."""
    result = HtmlExtractResult()
    cleaned = generic_clean(text or "")
    if not cleaned:
        return result
    max_chars = max(1, int(max_chars))
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
        result.truncated = True
    blocks = [line.strip() for line in cleaned.split("\n") if line.strip()]
    result.sections = [Section(heading="", blocks=blocks)]
    return result
