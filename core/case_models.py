"""
web_demo/core/case_models.py
----------------------------
Typed data carried between the stages of the "تحليل حالة" (case analysis)
pipeline.

These are plain dataclasses with no Streamlit, network, or provider imports so
each pipeline stage can be unit tested on its own. Every structure that leaves
retrieval keeps its provenance (document id/name, page range, chunk id, score,
and the research query that surfaced it) — a conclusion in the final report can
always be traced back to a real chunk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from core.source_models import (
    SOURCE_TYPE_PDF,
    SOURCE_TYPE_URL,
    domain_of,
    normalize_source_type,
    pages_label_ar,
    source_icon,
    url_citation_ar,
    url_locator_ar,
)

# --- Evidence strength ----------------------------------------------------
# Qualitative only. The demo never prints a fabricated probability such as
# "95% متأكد"; these buckets are derived from retrieval scores and agreement.
STRENGTH_STRONG = "strong"
STRENGTH_SUPPORTING = "supporting"
STRENGTH_POSSIBLE = "possible"

STRENGTH_LABELS_AR = {
    STRENGTH_STRONG: "دليل قوي",
    STRENGTH_SUPPORTING: "دليل مساند",
    STRENGTH_POSSIBLE: "قد يكون ذا صلة",
}

# --- Overall grounding of the final report --------------------------------
GROUNDING_STRONG = "strong"
GROUNDING_MEDIUM = "medium"
GROUNDING_LIMITED = "limited"

GROUNDING_LABELS_AR = {
    GROUNDING_STRONG: "قوية",
    GROUNDING_MEDIUM: "متوسطة",
    GROUNDING_LIMITED: "محدودة",
}

# --- Claim types (groundedness verification) ------------------------------
CLAIM_DOCUMENT_FACT = "document_fact"
CLAIM_INFERENCE = "inference"
CLAIM_UNSUPPORTED = "unsupported"

SUPPORT_STRONG = "strong"
SUPPORT_MODERATE = "moderate"
SUPPORT_LIMITED = "limited"
SUPPORT_UNSUPPORTED = "unsupported"

# Overall evidence coverage assessed before the report is shown.
COVERAGE_STRONG = "strong"
COVERAGE_MODERATE = "moderate"
COVERAGE_LIMITED = "limited"
COVERAGE_INSUFFICIENT = "insufficient"

COVERAGE_LABELS_AR = {
    COVERAGE_STRONG: "قوي",
    COVERAGE_MODERATE: "متوسط",
    COVERAGE_LIMITED: "محدود",
    COVERAGE_INSUFFICIENT: "غير كافٍ",
}

# Map coverage buckets onto the report's existing grounding labels.
COVERAGE_TO_GROUNDING = {
    COVERAGE_STRONG: GROUNDING_STRONG,
    COVERAGE_MODERATE: GROUNDING_MEDIUM,
    COVERAGE_LIMITED: GROUNDING_LIMITED,
    COVERAGE_INSUFFICIENT: GROUNDING_LIMITED,
}


@dataclass
class Claim:
    """One major conclusion mapped to evidence (internal — not shown to users)."""

    claim: str
    claim_type: str = CLAIM_INFERENCE
    evidence_ids: list[str] = field(default_factory=list)
    support_level: str = SUPPORT_LIMITED
    conflicting_evidence_ids: list[str] = field(default_factory=list)
    is_recommendation: bool = False

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "type": self.claim_type,
            "evidence_ids": list(self.evidence_ids),
            "support_level": self.support_level,
            "conflicting_evidence_ids": list(self.conflicting_evidence_ids),
            "is_recommendation": self.is_recommendation,
        }


@dataclass
class VerificationResult:
    """Structured output of the final groundedness check (no chain-of-thought)."""

    ok: bool
    error_category: str = ""
    claims: list[Claim] = field(default_factory=list)
    recommendation_determinable: bool = False
    recommendation_title: str = ""
    recommendation_reason: str = ""
    recommendation_evidence_ids: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    evidence_coverage: str = COVERAGE_LIMITED
    insufficient_documents: bool = False
    insufficient_case_facts: bool = False
    tie_between_solutions: bool = False

    @property
    def supported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.support_level != SUPPORT_UNSUPPORTED]


@dataclass
class MissingInfo:
    """One piece of information whose absence may change the outcome."""

    question: str
    reason: str = ""
    critical: bool = False


@dataclass
class StructuredCase:
    """The user's case after the understanding stage.

    This stage describes the problem only. It never contains a ruling — the
    verdict is produced later, and only from retrieved evidence.
    """

    summary: str = ""
    parties: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    core_issues: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    questions_to_resolve: list[str] = field(default_factory=list)
    missing_information: list[MissingInfo] = field(default_factory=list)

    @property
    def critical_missing(self) -> list[MissingInfo]:
        return [m for m in self.missing_information if m.critical]

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "parties": list(self.parties),
            "facts": list(self.facts),
            "core_issues": list(self.core_issues),
            "conditions": list(self.conditions),
            "constraints": list(self.constraints),
            "questions_to_resolve": list(self.questions_to_resolve),
            "missing_information": [
                {"question": m.question, "reason": m.reason, "critical": m.critical}
                for m in self.missing_information
            ],
        }


@dataclass
class ResearchQuery:
    """One focused retrieval query produced by the planner."""

    text: str
    purpose: str = ""

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        self.purpose = (self.purpose or "").strip()


def make_chunk_id(document_id: str, page_start, page_end, text: str) -> str:
    """Stable identity for a retrieved chunk.

    ``retrieval_service`` returns chunk *content*, not its index, so identity is
    derived from the document, its page range, and a digest of the text. The
    same chunk found by three different research queries collapses to one id.
    """
    digest = hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:{page_start}:{page_end}:{digest}"


@dataclass
class Evidence:
    """A retrieved chunk plus the provenance needed for a real citation.

    Both source kinds live in this one structure. A PDF chunk is located by
    page range; a web chunk by page title and section heading. The URL fields
    are populated ONLY from the stored chunk metadata, never from model output,
    which is what makes an invented link structurally impossible.
    """

    document_id: str
    document_name: str
    page_start: object = None
    page_end: object = None
    chunk_id: str = ""
    score: float = 0.0
    text: str = ""
    queries: list[str] = field(default_factory=list)
    strength: str = STRENGTH_POSSIBLE
    ref: str = ""  # e.g. "E1" — how the model is told to cite this chunk
    # --- Web provenance (empty for PDF evidence) --------------------------
    source_type: str = SOURCE_TYPE_PDF
    url: str = ""
    page_title: str = ""
    section_title: str = ""

    def __post_init__(self) -> None:
        self.source_type = normalize_source_type(self.source_type)
        if not self.chunk_id:
            self.chunk_id = make_chunk_id(
                self.document_id, self.page_start, self.page_end, self.text
            )

    @property
    def is_url(self) -> bool:
        return self.source_type == SOURCE_TYPE_URL

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    @property
    def pages_label_ar(self) -> str:
        return pages_label_ar(self.page_start, self.page_end)

    @property
    def locator_ar(self) -> str:
        """Where inside the source this chunk came from, in Arabic."""
        if self.is_url:
            return url_locator_ar(self.section_title)
        return self.pages_label_ar

    @property
    def source_label_ar(self) -> str:
        """The source's own name, without the in-source locator."""
        if self.is_url:
            return (self.page_title or self.document_name or "صفحة ويب").strip()
        return self.document_name

    def citation_ar(self) -> str:
        """Plain-text citation. PDFs cite a page; web pages cite a domain."""
        if self.is_url:
            return url_citation_ar(self.page_title, self.url, self.document_name)
        return f"{self.document_name} — {self.pages_label_ar}"

    def citation_markdown_ar(self) -> str:
        """Citation for Markdown output, with the icon and a real link.

        The URL comes from this evidence's stored metadata, so a rendered link
        always points at a page the server actually fetched and indexed.
        """
        icon = source_icon(self.source_type)
        label = self.citation_ar()
        if self.is_url and self.url:
            safe = self.url.replace("(", "%28").replace(")", "%29")
            return f"{icon} [{label}]({safe})"
        return f"{icon} {label}"

    def as_source(self) -> dict:
        """Shape understood by the existing source renderer in the UI."""
        payload = {
            "score": self.score,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "source_type": self.source_type,
        }
        if self.is_url:
            payload.update(
                {
                    "url": self.url,
                    "page_title": self.page_title,
                    "section_title": self.section_title,
                }
            )
        return payload

    def as_metadata(self) -> dict:
        """Provenance without the document text (safe for diagnostics)."""
        payload = {
            "ref": self.ref,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "strength": self.strength,
            "queries": list(self.queries),
            "source_type": self.source_type,
        }
        if self.is_url:
            payload.update(
                {
                    "url": self.url,
                    "page_title": self.page_title,
                    "section_title": self.section_title,
                    "domain": self.domain,
                }
            )
        return payload


@dataclass
class Solution:
    """One candidate resolution extracted from the evidence."""

    title: str = ""
    description: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    required_conditions: list[str] = field(default_factory=list)
    missing_information_affecting_it: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "supporting_evidence": list(self.supporting_evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
            "advantages": list(self.advantages),
            "limitations": list(self.limitations),
            "required_conditions": list(self.required_conditions),
            "missing_information_affecting_it": list(
                self.missing_information_affecting_it
            ),
        }


@dataclass
class SolutionSet:
    """Candidate solutions plus any conflict the model was asked to surface."""

    solutions: list[Solution] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    # True when the evidence does not let one solution be preferred.
    undecidable: bool = False
    undecidable_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "solutions": [s.as_dict() for s in self.solutions],
            "conflicts": list(self.conflicts),
            "undecidable": self.undecidable,
            "undecidable_reason": self.undecidable_reason,
        }
