from deepfind.rag.ingest.pdf import SENTINEL, _annotate_pages


def test_annotate_pages_injects_markers():
    md = f"page one text{SENTINEL}page two text"
    out = _annotate_pages(md)
    assert "<!-- page: 1 -->" in out
    assert "<!-- page: 2 -->" in out
    assert out.index("<!-- page: 1 -->") < out.index("<!-- page: 2 -->")


def test_annotate_single_page():
    out = _annotate_pages("only page")
    assert out.count("<!-- page:") == 1
    assert "<!-- page: 1 -->" in out
