from pathlib import Path

from deepfind.rag.discovery import parsed_dir_for, to_posix_rel


def test_parsed_dir_removes_last_suffix(tmp_path: Path):
    assert parsed_dir_for(Path("a/b/report.pdf")).name == "report"
    assert parsed_dir_for(Path("a/company.report.v2.pdf")).name == "company.report.v2"
    assert parsed_dir_for(Path("x/lesson.mp4")).name == "lesson"


def test_to_posix_rel_uses_forward_slashes():
    from deepfind.rag.config import REPO_ROOT

    p = REPO_ROOT / "assets" / "pdf" / "x.pdf"
    assert to_posix_rel(p, REPO_ROOT / "assets") == "pdf/x.pdf"
