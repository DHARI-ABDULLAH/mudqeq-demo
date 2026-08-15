"""Generic extraction + chunking tests, including scanned-PDF handling."""

from __future__ import annotations

from core import chunking, extraction
from services import document_service, security, session_service
from tests.pdf_util import make_empty_pdf, make_pdf


def test_generic_clean_preserves_arabic_order():
    # Arabic must never be reversed or reordered by generic cleaning.
    arabic = "المرابحة عقد بيع بالتكلفة زائد الربح."
    cleaned = extraction.generic_clean("  " + arabic + "  \u200b")
    assert cleaned == arabic


def test_generic_clean_removes_control_and_zero_width():
    dirty = "abc\u200b\u200c\x07def"
    assert extraction.generic_clean(dirty) == "abcdef"


def test_extract_and_chunk(tmp_path):
    data = make_pdf(["Alpha content here", "Beta content follows", "Gamma tail"])
    p = tmp_path / "t.pdf"
    p.write_bytes(data)
    result = extraction.extract_pdf(p)
    assert result.num_pages == 3
    assert result.has_usable_text()

    doc_id = security.new_id()
    chunks = chunking.build_chunks(
        result.non_empty_pages(), doc_id, "t.pdf", target_chars=10, overlap_chars=2
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c["document_id"] == doc_id
        assert c["document_name"] == "t.pdf"
        assert c["page_start"] is not None and c["page_end"] is not None
        # No filesystem paths leak into chunk metadata.
        assert "/" not in str(c.get("document_name", ""))


def test_scanned_pdf_flagged_and_rejected(new_session):
    # Empty/scanned-like PDF has no extractable text → user-facing reject.
    data = make_empty_pdf()
    import pytest

    with pytest.raises(security.UploadRejected) as exc:
        document_service.ingest(new_session, data, "scan.pdf")
    assert "نص قابل للاستخراج" in str(exc.value)


def test_full_ingest_arabic(new_session):
    arabic_pages = [
        "المرابحة هي عقد بيع يذكر فيه الثمن الأصلي مع هامش ربح متفق عليه.",
        "الصكوك أدوات مالية إسلامية مدعومة بأصول. الإجارة عقد تأجير.",
    ]
    data = make_pdf(arabic_pages)
    result = document_service.ingest(new_session, data, "arabic.pdf")
    assert result.status == document_service.STATUS_READY
    assert result.num_pages == 2
    assert result.num_chunks >= 1
