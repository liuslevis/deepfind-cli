"""Convert plain text and Markdown files into Kindle-friendly PDFs.

Page geometry targets the Kindle 8th generation screen (600x800 px at
167 ppi), so pages fill the display without the reader having to zoom.
"""

from __future__ import annotations

import html
import re
import shutil
import tempfile
from pathlib import Path


class PdfConversionError(RuntimeError):
    """Raised when a text/Markdown file cannot be rendered as PDF."""


# Kindle (gen 8) screen: 600 x 800 px at 167 ppi.
KINDLE_SCREEN_PX = (600, 800)
KINDLE_SCREEN_PPI = 167
POINTS_PER_INCH = 72.0
KINDLE_PAGE_WIDTH = KINDLE_SCREEN_PX[0] / KINDLE_SCREEN_PPI * POINTS_PER_INCH
KINDLE_PAGE_HEIGHT = KINDLE_SCREEN_PX[1] / KINDLE_SCREEN_PPI * POINTS_PER_INCH
KINDLE_MARGIN = 10.0
BASE_FONT_SIZE = 9.0
BASE_LEADING = 13.0
CODE_FONT_SIZE = 7.5

CONVERTIBLE_EXTENSIONS = {".txt", ".md", ".markdown"}

_MARKDOWN_EXTENSIONS = {".md", ".markdown"}

# Preferred system CJK fonts, newest first, per platform.
_FONT_CANDIDATES: tuple[tuple[str, str | None], ...] = (
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"),
    ("C:/Windows/Fonts/simsun.ttc", None),
    ("/System/Library/Fonts/PingFang.ttc", None),
    ("/System/Library/Fonts/STHeiti Light.ttc", None),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", None),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", None),
    ("/usr/share/fonts/truetype/arphic/uming.ttc", None),
)

_FONT_FAMILY = "DeepFindKindle"
_CODE_FONT = "Courier"
_registered_family: str | None = None


def is_convertible(path: Path) -> bool:
    return path.suffix.lower() in CONVERTIBLE_EXTENSIONS


def _require_reportlab():
    try:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import (
            HRFlowable,
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PdfConversionError(
            "PDF conversion needs reportlab. Install it with: pip install reportlab"
        ) from exc
    return (
        ParagraphStyle,
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
        KeepTogether,
    )


def _register_fonts() -> tuple[str, str]:
    """Register a Unicode-capable font family and return (regular, bold) names."""

    global _registered_family
    if _registered_family:
        return _registered_family, _registered_family + "-Bold"

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    regular = f"{_FONT_FAMILY}"
    bold = f"{_FONT_FAMILY}-Bold"

    for regular_path, bold_path in _FONT_CANDIDATES:
        candidate = Path(regular_path)
        if not candidate.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont(regular, str(candidate), subfontIndex=0))
            bold_file = Path(bold_path) if bold_path else None
            if bold_file and bold_file.is_file():
                pdfmetrics.registerFont(TTFont(bold, str(bold_file), subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont(bold, str(candidate), subfontIndex=0))
        except Exception:  # noqa: BLE001 - try the next candidate font
            continue
        break
    else:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        except Exception as exc:  # noqa: BLE001 - no usable CJK font at all
            raise PdfConversionError(
                f"No Unicode font available for PDF conversion: {exc}"
            ) from exc
        regular = bold = "STSong-Light"

    pdfmetrics.registerFontFamily(
        regular, normal=regular, bold=bold, italic=regular, boldItalic=bold
    )
    _registered_family = regular
    return regular, bold


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _starts_with_heading(text: str) -> bool:
    """Return True when the first meaningful line is a markdown heading.

    Used to avoid printing a separate title above a document whose own first
    line is already a ``# Heading`` of equal intent.
    """
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        return bool(_HEADING_RE.match(stripped))
    return False


_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _IMAGE_RE.sub(lambda m: m.group(1) or "[image]", escaped)
    escaped = _LINK_RE.sub(lambda m: m.group(1) or m.group(2), escaped)
    escaped = _INLINE_CODE_RE.sub(
        lambda m: f'<font face="{_CODE_FONT}">{m.group(1)}</font>', escaped
    )
    escaped = _BOLD_RE.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", escaped)
    escaped = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    escaped = _STRIKE_RE.sub(lambda m: f"<strike>{m.group(1)}</strike>", escaped)
    return escaped


def _plain_markup(text: str) -> str:
    return html.escape(text, quote=False)


def _build_styles():
    (ParagraphStyle, *_rest) = _require_reportlab()
    regular, bold = _register_fonts()
    body = ParagraphStyle(
        "KindleBody",
        fontName=regular,
        fontSize=BASE_FONT_SIZE,
        leading=BASE_LEADING,
        spaceAfter=5,
        firstLineIndent=0,
        wordWrap="CJK",
    )
    styles = {
        "body": body,
        "code": ParagraphStyle(
            "KindleCode",
            parent=body,
            fontName=_CODE_FONT,
            fontSize=CODE_FONT_SIZE,
            leading=CODE_FONT_SIZE + 2.5,
            leftIndent=6,
            spaceBefore=3,
            spaceAfter=5,
            wordWrap=None,
        ),
        "quote": ParagraphStyle(
            "KindleQuote",
            parent=body,
            leftIndent=8,
            textColor="#444444",
        ),
        "bullet": ParagraphStyle(
            "KindleBullet",
            parent=body,
            leftIndent=10,
            bulletIndent=2,
            spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "KindleCell",
            parent=body,
            fontSize=7.5,
            leading=10,
            spaceAfter=0,
        ),
        "cell_head": ParagraphStyle(
            "KindleCellHead",
            parent=body,
            fontName=bold,
            fontSize=7.5,
            leading=10,
            spaceAfter=0,
        ),
        "title": ParagraphStyle(
            "KindleTitle",
            parent=body,
            fontName=bold,
            fontSize=14,
            leading=18,
            spaceAfter=8,
        ),
    }
    for level, size in ((1, 13.5), (2, 12), (3, 10.5), (4, 10), (5, 9.5), (6, 9)):
        styles[f"h{level}"] = ParagraphStyle(
            f"KindleH{level}",
            parent=body,
            fontName=bold,
            fontSize=size,
            leading=size + 4,
            spaceBefore=7 if level <= 2 else 5,
            spaceAfter=3,
        )
    return styles


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)(.*)$")
_RULE_RE = re.compile(r"^\s*(?:\*\s*\*\s*\*|-\s*-\s*-|_\s*_\s*_)[\s*\-_]*$")
_UL_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")


def _split_table_row(line: str) -> list[str]:
    inner = _TABLE_ROW_RE.match(line)
    raw = inner.group(1) if inner else line.strip().strip("|")
    return [cell.strip() for cell in raw.split("|")]


def _table_flowable(rows: list[list[str]], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    width = KINDLE_PAGE_WIDTH - 2 * KINDLE_MARGIN
    columns = max(len(row) for row in rows)
    col_width = width / columns
    data = [
        [
            Paragraph(_inline_markup(cell), styles["cell_head" if index == 0 else "cell"])
            for cell in row + [""] * (columns - len(row))
        ]
        for index, row in enumerate(rows)
    ]
    table = Table(data, colWidths=[col_width] * columns, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ]
        )
    )
    return table


def _markdown_flowables(text: str, styles) -> list:
    (
        _ParagraphStyle,
        _SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
        _KeepTogether,
    ) = _require_reportlab()

    flowables: list = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(line.strip() for line in paragraph).strip()
            if joined:
                flowables.append(Paragraph(_inline_markup(joined), styles["body"]))
            paragraph.clear()

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _FENCE_RE.match(line)
        if fence:
            flush_paragraph()
            marker = fence.group(1)
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith(marker):
                block.append(lines[index])
                index += 1
            index += 1
            if block:
                body = "<br/>".join(
                    _plain_markup(item).replace(" ", "&nbsp;") for item in block
                )
                flowables.append(Paragraph(body, styles["code"]))
            continue

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        if _RULE_RE.match(line):
            flush_paragraph()
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=0.5, color="#999999"))
            flowables.append(Spacer(1, 3))
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            flowables.append(
                Paragraph(_inline_markup(heading.group(2)), styles[f"h{level}"])
            )
            index += 1
            continue

        if (
            _TABLE_ROW_RE.match(line)
            and index + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[index + 1])
            and "-" in lines[index + 1]
        ):
            flush_paragraph()
            rows = [_split_table_row(line)]
            index += 2
            while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
                rows.append(_split_table_row(lines[index]))
                index += 1
            flowables.append(_table_flowable(rows, styles))
            flowables.append(Spacer(1, 5))
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            flush_paragraph()
            flowables.append(Paragraph(_inline_markup(quote.group(1)), styles["quote"]))
            index += 1
            continue

        bullet = _UL_RE.match(line)
        if bullet:
            flush_paragraph()
            indent = len(bullet.group(1).expandtabs(4)) // 2
            style = styles["bullet"]
            flowables.append(
                Paragraph(
                    _inline_markup(bullet.group(3)),
                    style,
                    bulletText="\u2022",
                )
                if indent == 0
                else Paragraph(
                    f"{'&nbsp;' * (indent * 2)}\u2013 {_inline_markup(bullet.group(3))}",
                    style,
                )
            )
            index += 1
            continue

        ordered = _OL_RE.match(line)
        if ordered:
            flush_paragraph()
            flowables.append(
                Paragraph(
                    _inline_markup(ordered.group(3)),
                    styles["bullet"],
                    bulletText=f"{ordered.group(2)}.",
                )
            )
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return flowables


def _plain_flowables(text: str, styles) -> list:
    (
        _ParagraphStyle,
        _SimpleDocTemplate,
        Paragraph,
        Spacer,
        _HRFlowable,
        _KeepTogether,
    ) = _require_reportlab()

    flowables: list = []
    block: list[str] = []

    def flush() -> None:
        if block:
            body = "<br/>".join(_plain_markup(item.rstrip()) for item in block)
            flowables.append(Paragraph(body, styles["body"]))
            block.clear()

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            block.append(line)
        else:
            flush()
    flush()
    return flowables


def convert_to_pdf(source: Path, output_path: Path, *, title: str | None = None) -> Path:
    """Render a .txt/.md file to a Kindle-sized PDF at output_path."""

    try:
        return _convert_to_pdf(source, output_path, title=title)
    except PdfConversionError as exc:
        if isinstance(exc.__cause__, PermissionError):
            # Some sandboxes deny writes inside the process temp directory, so a
            # document that builds fine elsewhere never lands. Render into a
            # sibling directory next to the source instead of failing the send.
            fallback_dir = Path(tempfile.mkdtemp(prefix=".deepfind-kindle-", dir=source.parent))
            try:
                rendered = _convert_to_pdf(source, fallback_dir / output_path.name, title=title)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(rendered, output_path)
                rendered.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - the original error explains the failure
                raise exc from exc.__cause__
            finally:
                shutil.rmtree(fallback_dir, ignore_errors=True)
            return output_path
        raise


def _convert_to_pdf(source: Path, output_path: Path, *, title: str | None = None) -> Path:
    (
        _ParagraphStyle,
        SimpleDocTemplate,
        Paragraph,
        _Spacer,
        _HRFlowable,
        _KeepTogether,
    ) = _require_reportlab()

    text = _read_text(source)
    styles = _build_styles()
    document_title = (title or "").strip() or source.stem

    flowables: list = []
    # Only inject a heading when the document does not already open with one.
    # Markdown summaries typically start with `# Title`, and prefixing another
    # title would render the same heading twice on page one.
    inject_title = bool(document_title) and not _starts_with_heading(text)
    if inject_title:
        flowables.append(Paragraph(_plain_markup(document_title), styles["title"]))
    if source.suffix.lower() in _MARKDOWN_EXTENSIONS:
        body = _markdown_flowables(text, styles)
    else:
        body = _plain_flowables(text, styles)

    if not body:
        body = [Paragraph("(empty document)", styles["body"])]
    flowables.extend(body)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=(KINDLE_PAGE_WIDTH, KINDLE_PAGE_HEIGHT),
        leftMargin=KINDLE_MARGIN,
        rightMargin=KINDLE_MARGIN,
        topMargin=KINDLE_MARGIN,
        bottomMargin=KINDLE_MARGIN,
        title=document_title,
        author="DeepFind",
    )
    try:
        document.build(flowables)
    except PdfConversionError:
        raise
    except Exception as exc:  # noqa: BLE001 - reportlab raises broad errors
        raise PdfConversionError(f"Failed to render {source.name} as PDF: {exc}") from exc
    return output_path


__all__ = [
    "CONVERTIBLE_EXTENSIONS",
    "KINDLE_MARGIN",
    "KINDLE_PAGE_HEIGHT",
    "KINDLE_PAGE_WIDTH",
    "KINDLE_SCREEN_PPI",
    "KINDLE_SCREEN_PX",
    "PdfConversionError",
    "convert_to_pdf",
    "is_convertible",
]
