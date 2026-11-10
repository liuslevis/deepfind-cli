import tempfile
import unittest
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from deepfind.kindle_pdf import (
    KINDLE_PAGE_HEIGHT,
    KINDLE_PAGE_WIDTH,
    convert_to_pdf,
    is_convertible,
)


@contextmanager
def _scratch_dir():
    """Yield a writable working directory for a test.

    ``tempfile`` is preferred, but some sandboxes deny writes inside the
    directories it creates. Falling back to a plainly created directory keeps
    these tests runnable either way.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".probe").write_text("ok", encoding="utf-8")
            yield Path(tmpdir)
            return
    except (OSError, PermissionError):
        pass

    root = Path(__file__).resolve().parent.parent / "tmp" / "pytest-scratch"
    target = root / uuid.uuid4().hex
    target.mkdir(parents=True, exist_ok=False)
    yield target


class KindlePdfTests(unittest.TestCase):
    def _convert(self, name: str, text: str) -> PdfReader:
        with _scratch_dir() as tmpdir:
            source = tmpdir / name
            source.write_text(text, encoding="utf-8")
            output = convert_to_pdf(source, tmpdir / "out.pdf")
            return PdfReader(BytesIO(output.read_bytes()))

    def test_page_matches_kindle_gen8_screen(self) -> None:
        reader = self._convert("note.txt", "hello kindle")
        box = reader.pages[0].mediabox
        self.assertAlmostEqual(float(box.width), KINDLE_PAGE_WIDTH, places=2)
        self.assertAlmostEqual(float(box.height), KINDLE_PAGE_HEIGHT, places=2)

    def test_markdown_structure_and_cjk_render(self) -> None:
        text = (
            "# 标题\n\n"
            "正文 **粗体** 与 `code`\n\n"
            "- 条目一\n- 条目二\n\n"
            "| 列A | 列B |\n|---|---|\n| 值1 | 值2 |\n\n"
            "```\nprint('hi')\n```\n"
        )
        reader = self._convert("doc.md", text)
        rendered = "\n".join(page.extract_text() for page in reader.pages)
        for fragment in ("标题", "粗体", "条目一", "列A", "值2", "print('hi')"):
            self.assertIn(fragment, rendered)
        self.assertNotIn("**", rendered)

    def test_plain_text_keeps_line_breaks(self) -> None:
        reader = self._convert("note.txt", "first line\nsecond line\n")
        rendered = reader.pages[0].extract_text()
        self.assertIn("first line", rendered)
        self.assertIn("second line", rendered)

    def test_is_convertible(self) -> None:
        self.assertTrue(is_convertible(Path("a.md")))
        self.assertTrue(is_convertible(Path("a.TXT")))
        self.assertFalse(is_convertible(Path("a.pdf")))


class KindlePdfTitleTests(unittest.TestCase):
    """Regression tests for how the document title is injected."""

    def _render(self, name: str, text: str, *, title: str | None = None) -> str:
        with _scratch_dir() as tmpdir:
            source = tmpdir / name
            source.write_text(text, encoding="utf-8")
            output = convert_to_pdf(source, tmpdir / "out.pdf", title=title)
            reader = PdfReader(BytesIO(output.read_bytes()))
            return "\n".join(page.extract_text() for page in reader.pages)

    def test_markdown_h1_is_not_duplicated_by_injected_title(self) -> None:
        rendered = self._render("doc.md", "# My Doc Title\n\nBody text.\n", title="My Doc Title")
        self.assertEqual(rendered.count("My Doc Title"), 1)

    def test_missing_title_does_not_crash_and_falls_back_to_stem(self) -> None:
        rendered = self._render("My Doc.md", "# My Doc\n\nBody text.\n")
        self.assertIn("My Doc", rendered)
        self.assertIn("Body text.", rendered)

    def test_plain_text_document_still_gets_a_title(self) -> None:
        rendered = self._render("note.txt", "Hello body.", title="Plain Title")
        self.assertIn("Plain Title", rendered)
        self.assertIn("Hello body.", rendered)

    def test_markdown_without_leading_heading_gets_title(self) -> None:
        rendered = self._render(
            "doc.md", "Intro paragraph.\n\n## Later Section\n", title="Injected Title"
        )
        self.assertIn("Injected Title", rendered)

    def test_empty_document_produces_placeholder(self) -> None:
        rendered = self._render("empty.md", "", title="Empty Title")
        self.assertIn("Empty Title", rendered)
        self.assertIn("(empty document)", rendered)


if __name__ == "__main__":
    unittest.main()
