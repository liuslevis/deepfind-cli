"""Split a template-compliant ``content.md`` into ``chunks.jsonl`` records."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .formatter import parse_header
from .models import Chunk, SourceType

NAMESPACE = uuid.UUID("6f9b2c1e-0e5a-5d3b-9c4a-1b2c3d4e5f60")

TARGET_MIN = 800
TARGET_MAX = 1200
HARD_MAX = 1600
OVERLAP = 150
MIN_EFFECTIVE = 80
CHUNK_CONFIG_VERSION = f"v1:{TARGET_MIN}-{TARGET_MAX}-{HARD_MAX}-{OVERLAP}"

PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
TIME_RE = re.compile(r"<!--\s*time:\s*(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
MARKER_LINE_RE = re.compile(r"^\s*<!--\s*(page|time):.*-->\s*$")


def _hms_to_seconds(value: str) -> int:
    h, m, s = (int(x) for x in value.split(":"))
    return h * 3600 + m * 60 + s


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _effective_len(text: str) -> int:
    stripped = re.sub(r"<!--.*?-->", "", text)
    stripped = re.sub(r"^#{1,6}\s+", "", stripped, flags=re.MULTILINE)
    return len(_normalize(stripped))


class _Unit:
    __slots__ = ("end", "is_heading", "page", "section", "start", "text")

    def __init__(self, text, section, page, start, end, is_heading):
        self.text = text
        self.section = section
        self.page = page
        self.start = start
        self.end = end
        self.is_heading = is_heading


def _content_body(content_md: str) -> str:
    if "## Content" not in content_md:
        return ""
    return content_md.split("## Content", 1)[1].lstrip("\n")


def _iter_units(body: str):
    section = ""
    page: int | None = None
    start: int | None = None
    end: int | None = None
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            text = "\n".join(para).strip()
            if text:
                yield_unit = _Unit(text, section, page, start, end, False)
                para = []
                return yield_unit
            para = []
        return None

    lines = body.splitlines()
    for line in lines:
        pm = PAGE_RE.search(line)
        tm = TIME_RE.search(line)
        if MARKER_LINE_RE.match(line):
            u = flush_para()
            if u:
                yield u
            if pm:
                page = int(pm.group(1))
            if tm:
                start = _hms_to_seconds(tm.group(1))
                end = _hms_to_seconds(tm.group(2))
            continue
        if pm:
            page = int(pm.group(1))
        if tm:
            start = _hms_to_seconds(tm.group(1))
            end = _hms_to_seconds(tm.group(2))
        hm = HEADING_RE.match(line.strip())
        if hm:
            u = flush_para()
            if u:
                yield u
            section = hm.group(2).strip()
            continue
        if not line.strip():
            u = flush_para()
            if u:
                yield u
        else:
            para.append(line)
    u = flush_para()
    if u:
        yield u


def _split_oversized(unit: _Unit) -> list[_Unit]:
    if len(unit.text) <= HARD_MAX:
        return [unit]
    pieces: list[_Unit] = []
    text = unit.text
    while len(text) > HARD_MAX:
        cut = text.rfind(" ", TARGET_MIN, HARD_MAX)
        if cut == -1:
            cut = HARD_MAX
        pieces.append(
            _Unit(
                text[:cut].strip(), unit.section, unit.page, unit.start, unit.end, False
            )
        )
        text = text[cut:].strip()
    if text:
        pieces.append(_Unit(text, unit.section, unit.page, unit.start, unit.end, False))
    return pieces


def _merge_range(values: list[int | None], use_min: bool) -> int | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return min(present) if use_min else max(present)


def chunk_content(
    content_md: str,
    *,
    source: str,
    source_type: SourceType,
    source_sha256: str,
) -> list[Chunk]:
    header = parse_header(content_md)
    title = header.get("title", "")
    tags = header.get("tags", [])

    body = _content_body(content_md)
    units: list[_Unit] = []
    for unit in _iter_units(body):
        units.extend(_split_oversized(unit))

    raw_chunks: list[list[_Unit]] = []
    cur: list[_Unit] = []
    cur_len = 0
    for unit in units:
        add_len = len(unit.text) + 2
        if cur and cur_len + add_len > TARGET_MAX and cur_len >= TARGET_MIN:
            raw_chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(unit)
        cur_len += add_len
        if cur_len >= TARGET_MAX:
            raw_chunks.append(cur)
            cur = []
            cur_len = 0
    if cur:
        raw_chunks.append(cur)

    chunks: list[Chunk] = []
    prev_tail = ""
    index = 0
    for group in raw_chunks:
        core = "\n\n".join(u.text for u in group).strip()
        if _effective_len(core) < MIN_EFFECTIVE:
            continue
        text = (prev_tail + "\n\n" + core).strip() if prev_tail else core
        prev_tail = core[-OVERLAP:]

        section = next((u.section for u in reversed(group) if u.section), "")
        page_start = _merge_range([u.page for u in group], use_min=True)
        page_end = _merge_range([u.page for u in group], use_min=False)
        start_seconds = _merge_range([u.start for u in group], use_min=True)
        end_seconds = _merge_range([u.end for u in group], use_min=False)

        chunk_id = str(
            uuid.uuid5(
                NAMESPACE, f"{source}\n{source_sha256}\n{index}\n{_normalize(core)}"
            )
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_index=index,
                source=source,
                source_type=source_type,
                title=title,
                tags=tags,
                section=section,
                page_start=page_start if source_type == "pdf" else None,
                page_end=page_end if source_type == "pdf" else None,
                start_seconds=start_seconds if source_type != "pdf" else None,
                end_seconds=end_seconds if source_type != "pdf" else None,
                text=text,
            )
        )
        index += 1
    return chunks


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    from .discovery import atomic_write_text

    lines = "\n".join(json.dumps(c.model_dump(), ensure_ascii=False) for c in chunks)
    atomic_write_text(path, lines + ("\n" if lines else ""))


def read_chunks(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Chunk.model_validate_json(line))
    return out
