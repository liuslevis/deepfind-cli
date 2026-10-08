import pytest

from deepfind.rag.formatter import (
    FormatError,
    body_from_raw,
    parse_header,
    render_content_md,
    validate_content,
)


def _sample(source_type: str) -> str:
    marker = (
        "<!-- page: 1 -->"
        if source_type == "pdf"
        else "<!-- time: 00:00:00-00:05:00 -->"
    )
    return render_content_md(
        title="Example",
        summary="A one line summary.",
        tags=["china", "macro"],
        created="2026-01-01T00:00:00Z",
        updated="2026-01-02T00:00:00Z",
        source="doc/pdf/example.pdf",
        body=f"{marker}\n\nsome body text",
    )


def test_render_and_validate_pdf():
    text = _sample("pdf")
    validate_content(text, "pdf", expected_source="doc/pdf/example.pdf")
    assert (
        text.index("**Summary**") < text.index("**Tags**") < text.index("**Created**")
    )


def test_validate_requires_page_marker_for_pdf():
    text = render_content_md(
        title="T",
        summary="s",
        tags=["a", "b"],
        created="c",
        updated="u",
        source="doc/pdf/x.pdf",
        body="no markers here",
    )
    with pytest.raises(FormatError):
        validate_content(text, "pdf")


def test_validate_requires_time_marker_for_media():
    text = render_content_md(
        title="T",
        summary="s",
        tags=["a", "b"],
        created="c",
        updated="u",
        source="doc/media/x.mp4",
        body="<!-- page: 1 -->\nno time marker",
    )
    with pytest.raises(FormatError):
        validate_content(text, "video")


def test_source_mismatch_fails():
    text = _sample("pdf")
    with pytest.raises(FormatError):
        validate_content(text, "pdf", expected_source="doc/pdf/other.pdf")


def test_body_from_raw_converts_media_time_headers():
    raw = "[00:00:00 - 00:05:00]\nhello\n\n[00:05:00 - 00:10:00]\nworld"
    out = body_from_raw(raw, "video")
    assert "<!-- time: 00:00:00-00:05:00 -->" in out
    assert "<!-- time: 00:05:00-00:10:00 -->" in out
    assert "[00:00:00" not in out


def test_parse_header_roundtrip():
    text = _sample("pdf")
    header = parse_header(text)
    assert header["title"] == "Example"
    assert header["tags"] == ["china", "macro"]
    assert header["source"] == "doc/pdf/example.pdf"
