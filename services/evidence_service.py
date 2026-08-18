"""
web_demo/services/evidence_service.py
-------------------------------------
Multi-step retrieval and evidence curation for case analysis.

Responsibilities
----------------
1. Run each research query independently against the SELECTED documents only.
2. Merge the per-query hits into one evidence set, de-duplicating chunks that
   several queries found while remembering *which* queries found them.
3. Rank the set and label each item's strength qualitatively.
4. Build a hard-bounded, reference-tagged context block for the model.

Isolation
---------
Retrieval goes through ``retrieval_service``, which verifies every document id
against the calling session before touching an index. Queries are issued one
document at a time so each hit keeps its ``document_id`` — the retrieval API
returns display names only, and two uploads can share a name.

Nothing here calls a provider or logs document text.
"""

from __future__ import annotations

from config import (
    MAX_CASE_CONTEXT_CHARS,
    MAX_RESULTS_PER_QUERY,
    MAX_TOTAL_EVIDENCE_CHUNKS,
)
from core.case_models import (
    STRENGTH_POSSIBLE,
    STRENGTH_STRONG,
    STRENGTH_SUPPORTING,
    Evidence,
    make_chunk_id,
)
from core.source_models import SOURCE_TYPE_PDF, SOURCE_TYPE_URL
from services import retrieval_service, security, session_service

# Score fractions (relative to the best hit in this run) that separate the
# qualitative buckets. Relative rather than absolute because inner-product
# scores are not comparable across documents or embedding runs.
STRONG_RATIO = 0.85
SUPPORTING_RATIO = 0.65


class EvidenceUnavailable(Exception):
    """A selected document's index could not be read during collection."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def owned_document_ids(session_id: str, document_ids) -> list[str]:
    """Filter a selection down to sources this session actually owns.

    Works for both PDF and URL sources — ownership is checked against the
    session's source table, which holds both kinds under the same ids.
    """
    out: list[str] = []
    for doc_id in document_ids or []:
        if not security.is_valid_id(doc_id):
            continue
        if session_service.get_document(session_id, doc_id) is None:
            continue
        if doc_id not in out:
            out.append(doc_id)
    return out


def _retrieve_one(session_id: str, document_id: str, query: str, top_k: int) -> list:
    try:
        return list(
            retrieval_service.retrieve(session_id, [document_id], query, top_k=top_k)
            or []
        )
    except retrieval_service.IndexUnavailable as exc:
        raise EvidenceUnavailable(exc.reason) from exc


def collect(
    session_id: str,
    document_ids,
    queries,
    *,
    results_per_query: int = MAX_RESULTS_PER_QUERY,
    max_total: int = MAX_TOTAL_EVIDENCE_CHUNKS,
) -> list:
    """Run every query over every selected document and curate the results.

    Returns a ranked ``list[Evidence]`` of at most ``max_total`` items, each
    tagged ``E1``, ``E2``… in final order. Raises :class:`EvidenceUnavailable`
    if an owned document's index is unreadable — that is an infrastructure
    failure and must not be reported as "no evidence found".
    """
    security.require_valid_id(session_id)
    owned = owned_document_ids(session_id, document_ids)
    if not owned or not queries:
        return []

    results_per_query = max(1, int(results_per_query))
    max_total = max(1, int(max_total))

    by_chunk: dict[str, Evidence] = {}

    for query in queries:
        query_text = getattr(query, "text", None) or str(query)
        query_text = query_text.strip()
        if not query_text:
            continue

        for document_id in owned:
            for hit in _retrieve_one(
                session_id, document_id, query_text, results_per_query
            ):
                text = (hit.get("text") or "").strip()
                if not text:
                    continue
                page_start, page_end = hit.get("page_start"), hit.get("page_end")
                chunk_id = make_chunk_id(document_id, page_start, page_end, text)
                score = float(hit.get("score") or 0.0)

                existing = by_chunk.get(chunk_id)
                if existing is None:
                    by_chunk[chunk_id] = Evidence(
                        document_id=document_id,
                        document_name=hit.get("document_name") or "",
                        page_start=page_start,
                        page_end=page_end,
                        chunk_id=chunk_id,
                        score=score,
                        text=text,
                        queries=[query_text],
                        # Provenance is copied straight from the stored chunk,
                        # so a web citation can only ever name a page the
                        # server itself fetched.
                        source_type=hit.get("source_type") or SOURCE_TYPE_PDF,
                        url=hit.get("url") or "",
                        page_title=hit.get("page_title") or "",
                        section_title=hit.get("section_title") or "",
                    )
                else:
                    # Same chunk, another query: keep the best score and record
                    # the extra query — agreement across queries is a ranking
                    # signal and is shown to the user.
                    existing.score = max(existing.score, score)
                    if query_text not in existing.queries:
                        existing.queries.append(query_text)

    return rank(list(by_chunk.values()), max_total=max_total)


def rank(evidence: list, *, max_total: int = MAX_TOTAL_EVIDENCE_CHUNKS) -> list:
    """Order by (queries that found it, score), label strength, assign refs."""
    if not evidence:
        return []

    ordered = sorted(
        evidence,
        key=lambda e: (len(e.queries), e.score),
        reverse=True,
    )[: max(1, int(max_total))]

    best = max((e.score for e in ordered), default=0.0)
    for index, item in enumerate(ordered, start=1):
        item.strength = classify_strength(item, best_score=best)
        item.ref = f"E{index}"
    return ordered


def classify_strength(item: Evidence, *, best_score: float) -> str:
    """Qualitative bucket for one piece of evidence.

    Deliberately not a probability. It combines how close the hit is to the
    best hit in this run with how many independent research queries surfaced
    it — a chunk found by several angles of the case is better support than one
    that squeaked into a single Top-K list.
    """
    if best_score <= 0:
        return STRENGTH_POSSIBLE

    ratio = item.score / best_score
    multi_query = len(item.queries) > 1

    if ratio >= STRONG_RATIO or (multi_query and ratio >= SUPPORTING_RATIO):
        return STRENGTH_STRONG
    if ratio >= SUPPORTING_RATIO or multi_query:
        return STRENGTH_SUPPORTING
    return STRENGTH_POSSIBLE


def evidence_header(item) -> str:
    """The ``[E#] source — locator`` line that heads one evidence block.

    A web source is described by its title, domain, and section — deliberately
    NOT by its full URL. The model never needs to reproduce a link: citations
    are rendered afterwards from this evidence's stored metadata, so keeping
    the raw URL out of the prompt removes the only way it could echo a wrong
    one back.
    """
    ref = f"[{item.ref}]" if item.ref else "[E?]"
    if getattr(item, "is_url", False):
        domain = item.domain
        head = f"{ref} {item.source_label_ar}"
        if domain:
            head += f" — {domain}"
        section = (item.section_title or "").strip()
        return f"{head} · {section}" if section else head
    return f"{ref} {item.document_name} — {item.pages_label_ar}"


def build_context(evidence: list, *, max_chars: int = MAX_CASE_CONTEXT_CHARS) -> str:
    """Reference-tagged, hard-bounded evidence block for the prompt.

    Each block is headed with its ``[E#]`` reference and its provenance (page
    range for a document, title/section for a web page) so the model can cite
    by reference and the citations can later be resolved back to real chunks.
    """
    max_chars = max(1, int(max_chars))
    parts: list[str] = []
    used = 0

    for item in evidence:
        header = evidence_header(item)
        body = (item.text or "").strip()
        block = f"{header}\n{body}"

        if used + len(block) > max_chars:
            remaining = max_chars - used
            # Only include a truncated block if the header plus some text fits;
            # a header on its own would be a citation with no content.
            if remaining > len(header) + 40:
                parts.append(f"{header}\n{body[: remaining - len(header) - 1]}")
            break

        parts.append(block)
        used += len(block)

    return "\n\n---\n\n".join(parts)


def by_ref(evidence: list) -> dict:
    """Map ``E1`` -> Evidence for resolving model citations back to chunks."""
    return {item.ref: item for item in evidence if item.ref}


def resolve_refs(evidence: list, refs) -> list:
    """Return the Evidence objects for ``refs``, silently dropping unknown ones.

    A model that invents ``E99`` must not produce a citation — an unresolvable
    reference is discarded rather than rendered.
    """
    table = by_ref(evidence)
    out: list = []
    for ref in refs or []:
        if not isinstance(ref, str):
            continue
        item = table.get(ref.strip().upper())
        if item is not None and item not in out:
            out.append(item)
    return out


def summarize(evidence: list) -> dict:
    """Content-free counters for diagnostics and logging."""
    documents = {e.document_id for e in evidence}
    pdf_sources = {
        e.document_id for e in evidence if getattr(e, "source_type", SOURCE_TYPE_PDF) != SOURCE_TYPE_URL
    }
    url_sources = {e.document_id for e in evidence if getattr(e, "is_url", False)}
    return {
        "num_evidence": len(evidence),
        "num_documents": len(documents),
        "num_pdf_sources": len(pdf_sources),
        "num_url_sources": len(url_sources),
        "num_strong": sum(1 for e in evidence if e.strength == STRENGTH_STRONG),
        "num_supporting": sum(
            1 for e in evidence if e.strength == STRENGTH_SUPPORTING
        ),
        "num_possible": sum(1 for e in evidence if e.strength == STRENGTH_POSSIBLE),
    }
