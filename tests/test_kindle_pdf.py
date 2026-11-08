import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from deepfind.kindle_pdf import (
    KINDLE_PAGE_HEIGHT,
    KINDLE_PAGE_WIDTH,
    convert_to_pdf,
    is_convertible,
)


class KindlePdfTests(unittest.TestCase):
    def _convert(self, name: str, text: str) -> PdfReader:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / name
            source.write_text(text, encoding="utf-8")
            output = convert_to_pdf(source, Path(tmpdir) / "out.pdf")
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


if __name__ == "__main__":
    unittest.main()
