"""PDF parsing via Docling into a page-annotated ``raw.md``."""

from __future__ import annotations

from pathlib import Path

SENTINEL = "<<<DEEPFIND_PAGE_BREAK>>>"


class PdfParseError(RuntimeError):
    pass


def _annotate_pages(markdown: str) -> str:
    parts = markdown.split(SENTINEL)
    blocks = []
    for i, part in enumerate(parts, start=1):
        body = part.strip()
        blocks.append(
            f"<!-- page: {i} -->\n\n{body}" if body else f"<!-- page: {i} -->"
        )
    return "\n\n".join(blocks).strip() + "\n"


def _fallback_from_items(doc) -> str:
    """Reconstruct page-annotated markdown from item provenance."""
    try:
        pages: dict[int, list[str]] = {}
        for item, _level in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text:
                continue
            page_no = 1
            prov = getattr(item, "prov", None)
            if prov:
                page_no = getattr(prov[0], "page_no", 1) or 1
            pages.setdefault(page_no, []).append(str(text).strip())
        if not pages:
            raise PdfParseError("Docling produced no extractable text")
        blocks = []
        for page_no in sorted(pages):
            joined = "\n\n".join(t for t in pages[page_no] if t)
            blocks.append(f"<!-- page: {page_no} -->\n\n{joined}")
        return "\n\n".join(blocks).strip() + "\n"
    except PdfParseError:
        raise
    except Exception as exc:  # pragma: no cover
        raise PdfParseError(f"page reconstruction failed: {exc}") from exc


def parse_pdf(source: Path, parsed_dir: Path) -> str:
    """Return page-annotated markdown for ``source``.

    Uses Docling's page-break placeholder when available, otherwise falls back
    to provenance-based reconstruction. Images are rendered as placeholders to
    keep parsing deterministic and offline-safe.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise PdfParseError("docling is not installed") from exc

    try:
        converter = DocumentConverter()
        result = converter.convert(str(source))
        doc = result.document
    except Exception as exc:
        raise PdfParseError(f"Docling failed to convert {source.name}: {exc}") from exc

    markdown = None
    for kwargs in (
        {"page_break_placeholder": SENTINEL, "image_placeholder": ""},
        {"page_break_placeholder": SENTINEL},
    ):
        try:
            markdown = doc.export_to_markdown(**kwargs)
            break
        except TypeError:
            continue
        except Exception as exc:
            raise PdfParseError(f"Docling markdown export failed: {exc}") from exc

    if markdown is not None and SENTINEL in markdown:
        return _annotate_pages(markdown)

    return _fallback_from_items(doc)
