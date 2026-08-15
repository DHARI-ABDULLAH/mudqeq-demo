"""
web_demo/tests/pdf_util.py
--------------------------
Build minimal, valid PDFs in pure Python for tests (no external writer, no
private documents). Also provides a helper for a fake "encrypted" PDF.
"""

from __future__ import annotations


def _escape(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1", "replace")


def make_pdf(pages_text: list[str]) -> bytes:
    """Return bytes of a simple PDF, one text line per page."""
    objects: dict[int, bytes] = {}
    catalog_obj, pages_obj, font_obj = 1, 2, 3
    next_num = 4
    kids: list[int] = []

    for text in pages_text:
        content_num = next_num
        next_num += 1
        page_num = next_num
        next_num += 1

        stream = b"BT /F1 24 Tf 72 720 Td (" + _escape(text) + b") Tj ET"
        objects[content_num] = (
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
        objects[page_num] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>" % content_num
        )
        kids.append(page_num)

    objects[catalog_obj] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids_str = " ".join(f"{k} 0 R" for k in kids).encode()
    objects[pages_obj] = (
        b"<< /Type /Pages /Kids [" + kids_str + b"] /Count %d >>" % len(kids)
    )
    objects[font_obj] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"

    xref_pos = len(out)
    count = max(objects) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, count):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


def make_empty_pdf() -> bytes:
    """A PDF with a page but effectively no extractable text (scanned-like)."""
    return make_pdf([" "])
