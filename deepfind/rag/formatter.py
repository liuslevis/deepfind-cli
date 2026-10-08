"""Template-compliant ``content.md`` rendering, LLM metadata extraction, and validation.

Design choice for minimalism and fidelity: the LLM only extracts header fields
(title / summary / tags) from the head of the document. The body is carried over
from the raw artifact with light deterministic normalization, which guarantees
that traceability markers (``<!-- page: N -->`` / ``<!-- time: ... -->``) survive
and that numbers, dates and names are never rewritten by the model.
"""

from __future__ import annotations

import json
import re

from .config import Config
from .models import SourceType

TIME_HEADER_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\]\s*$")
PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*\d+\s*-->")
TIME_MARKER_RE = re.compile(r"<!--\s*time:\s*[0-9:]+-[0-9:]+\s*-->")


class FormatError(RuntimeError):
    """Raised when formatting fails or output fails template validation."""


def body_from_raw(raw_text: str, source_type: SourceType) -> str:
    """Prepare the ``## Content`` body from a raw artifact.

    - PDF raw.md already carries ``<!-- page: N -->`` markers; passed through.
    - Media raw.txt uses ``[HH:MM:SS - HH:MM:SS]`` block headers which are
      converted into ``<!-- time: start-end -->`` markers.
    """
    if source_type == "pdf":
        return raw_text.strip()

    lines_out: list[str] = []
    for line in raw_text.splitlines():
        m = TIME_HEADER_RE.match(line.strip())
        if m:
            lines_out.append(f"<!-- time: {m.group(1)}-{m.group(2)} -->")
        else:
            lines_out.append(line)
    return "\n".join(lines_out).strip()


def render_content_md(
    *,
    title: str,
    summary: str,
    tags: list[str],
    created: str,
    updated: str,
    source: str,
    body: str,
) -> str:
    tag_str = " ".join(f"#{t.lstrip('#')}" for t in tags)
    return (
        f"# {title}\n\n"
        f"**Summary**: {summary}\n"
        f"**Tags**: {tag_str}\n"
        f"**Created**: {created}\n"
        f"**Last Updated**: {updated}\n"
        f"**Source File**: {source}\n"
        f"---\n\n"
        f"## Content\n\n"
        f"{body.strip()}\n"
    )


_HEADER_FIELDS = [
    "**Summary**:",
    "**Tags**:",
    "**Created**:",
    "**Last Updated**:",
    "**Source File**:",
]


def validate_content(
    text: str, source_type: SourceType, expected_source: str | None = None
) -> None:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# ") or not lines[0][2:].strip():
        raise FormatError("missing or empty document title (`# ...`)")

    # header fields must appear in order before the divider
    pos = 0
    for field in _HEADER_FIELDS:
        idx = text.find(field, pos)
        if idx == -1:
            raise FormatError(f"missing header field {field}")
        pos = idx + len(field)

    if "## Content" not in text:
        raise FormatError("missing `## Content` section")

    if expected_source is not None:
        m = re.search(r"\*\*Source File\*\*:\s*(.+)", text)
        if not m or m.group(1).strip() != expected_source:
            raise FormatError("Source File does not match expected source path")

    body = text.split("## Content", 1)[1]
    if source_type == "pdf":
        if not PAGE_MARKER_RE.search(body):
            raise FormatError(
                "PDF content.md must retain at least one <!-- page: N --> marker"
            )
    else:
        if not TIME_MARKER_RE.search(body):
            raise FormatError(
                "media content.md must retain at least one <!-- time: ... --> marker"
            )


_EXTRACT_PROMPT = """You extract catalog metadata for an investment-research document.
Return ONLY a compact JSON object with keys:
  "title": string  (the document's real title; if unclear use the given fallback)
  "summary": string  (ONE factual sentence; no added conclusions)
  "tags": array of 2 to 8 short lowercase strings (companies, industries, asset classes, regions, themes; no '#')

Do not include any other text. Base everything strictly on the provided content.
Fallback title: {fallback}

DOCUMENT HEAD:
{head}
"""


def _fallback_fields(fallback_title: str) -> tuple[str, str, list[str]]:
    return fallback_title, fallback_title, ["uncategorized"]


def extract_header_fields(
    raw_text: str, fallback_title: str, config: Config
) -> tuple[str, str, list[str]]:
    """Call the configured OpenAI-compatible LLM to derive title/summary/tags."""
    if not config.formatter.enabled:
        raise FormatError("formatter LLM is not configured (missing API key/model)")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise FormatError("openai package is not installed") from exc

    client = OpenAI(
        base_url=config.formatter.base_url, api_key=config.formatter.api_key
    )
    head = raw_text[:12000]
    prompt = _EXTRACT_PROMPT.format(fallback=fallback_title, head=head)

    try:
        resp = client.chat.completions.create(
            model=config.formatter.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
    except Exception as exc:
        raise FormatError(f"formatter LLM call failed: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise FormatError("formatter LLM returned no JSON")
        data = json.loads(match.group(0))

    title = str(data.get("title") or fallback_title).strip() or fallback_title
    summary = str(data.get("summary") or "").strip()
    raw_tags = data.get("tags") or []
    tags = [str(t).lstrip("#").strip().lower() for t in raw_tags if str(t).strip()]
    tags = [t for t in tags if t][:8]
    if len(tags) < 2:
        tags = (tags + ["investment", "research"])[:2]
    if not summary:
        summary = title
    return title, summary, tags


def format_document(
    *,
    raw_text: str,
    source: str,
    source_type: SourceType,
    fallback_title: str,
    created: str,
    updated: str,
    config: Config,
) -> str:
    """Produce a validated ``content.md`` string. Raises FormatError on failure."""
    title, summary, tags = extract_header_fields(raw_text, fallback_title, config)
    body = body_from_raw(raw_text, source_type)
    content = render_content_md(
        title=title,
        summary=summary,
        tags=tags,
        created=created,
        updated=updated,
        source=source,
        body=body,
    )
    validate_content(content, source_type, expected_source=source)
    return content


def parse_header(text: str) -> dict:
    """Extract header field values from an existing content.md (best effort)."""
    out: dict = {}
    m = re.match(r"#\s+(.+)", text)
    if m:
        out["title"] = m.group(1).strip()
    for key, label in [
        ("summary", "Summary"),
        ("tags", "Tags"),
        ("created", "Created"),
        ("updated", "Last Updated"),
        ("source", "Source File"),
    ]:
        mm = re.search(rf"\*\*{label}\*\*:\s*(.+)", text)
        if mm:
            out[key] = mm.group(1).strip()
    if "tags" in out:
        out["tags"] = [t.lstrip("#") for t in out["tags"].split() if t.strip()]
    return out
