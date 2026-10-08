from deepfind.rag.indexer import _parsed_dir


def test_parsed_dir_strips_suffix():
    assert _parsed_dir("doc/pdf/a/report.pdf") == "doc/pdf/a/report"
    assert _parsed_dir("doc/media/x/lesson.mp4") == "doc/media/x/lesson"


def test_parsed_dir_multidot():
    assert _parsed_dir("doc/pdf/company.report.v2.pdf") == "doc/pdf/company.report.v2"
