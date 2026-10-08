from deepfind.rag.chunking import chunk_content
from deepfind.rag.formatter import render_content_md


def _pdf_content(body: str) -> str:
    return render_content_md(
        title="Report",
        summary="s",
        tags=["china"],
        created="c",
        updated="u",
        source="doc/pdf/report.pdf",
        body=body,
    )


def _media_content(body: str) -> str:
    return render_content_md(
        title="Lesson",
        summary="s",
        tags=["macro"],
        created="c",
        updated="u",
        source="doc/media/lesson.mp4",
        body=body,
    )


def test_pdf_chunks_carry_page_ranges():
    body = (
        "<!-- page: 3 -->\n\n"
        + ("段落内容。" * 200)
        + "\n\n<!-- page: 4 -->\n\n"
        + ("更多内容。" * 200)
    )
    chunks = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="abc",
    )
    assert chunks
    assert all(c.page_start is not None for c in chunks)
    assert all(c.start_seconds is None for c in chunks)
    assert min(c.page_start for c in chunks) == 3


def test_media_chunks_carry_time_ranges():
    body = "<!-- time: 00:00:00-00:05:00 -->\n\n" + ("讲话内容。" * 200)
    chunks = chunk_content(
        _media_content(body),
        source="doc/media/lesson.mp4",
        source_type="video",
        source_sha256="xyz",
    )
    assert chunks
    assert all(c.start_seconds == 0 for c in chunks)
    assert all(c.page_start is None for c in chunks)


def test_chunk_ids_stable_for_same_input():
    body = "<!-- page: 1 -->\n\n" + ("内容。" * 300)
    a = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="s1",
    )
    b = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="s1",
    )
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_ids_change_with_sha():
    body = "<!-- page: 1 -->\n\n" + ("内容。" * 300)
    a = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="s1",
    )
    b = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="s2",
    )
    assert a[0].chunk_id != b[0].chunk_id


def test_short_content_skipped():
    body = "<!-- page: 1 -->\n\ntiny"
    chunks = chunk_content(
        _pdf_content(body),
        source="doc/pdf/report.pdf",
        source_type="pdf",
        source_sha256="s",
    )
    assert chunks == []
