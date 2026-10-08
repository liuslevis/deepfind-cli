from deepfind.rag.indexer import _parsed_dir


def test_parsed_dir_strips_suffix():
    assert _parsed_dir("pdf/a/report.pdf") == "pdf/a/report"
    assert _parsed_dir("media/x/lesson.mp4") == "media/x/lesson"


def test_parsed_dir_multidot():
    assert _parsed_dir("pdf/company.report.v2.pdf") == "pdf/company.report.v2"
