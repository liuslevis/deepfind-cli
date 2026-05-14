from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepfind.gen_slides import (
    DEFAULT_TEMPLATE_NAME,
    SlideGenerationError,
    _clean_llm_output,
    _extract_title_from_html,
    _load_template,
    _split_template,
    _validate_template_name,
    generate_slides,
    resolve_slide_root,
)


def message_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ]
    )


class FakeChatCompletionsAPI:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return message_response(self.text)


class FakeClient:
    def __init__(self, text: str):
        self.chat = SimpleNamespace(completions=FakeChatCompletionsAPI(text))


SAMPLE_SLIDES_HTML = (
    '<section class="slide slide--cover light">\n'
    '  <h1 class="display">AI Briefing</h1>\n'
    '</section>\n'
    '<section class="slide slide--statement light">\n'
    '  <h2 class="h2">Overview</h2>\n'
    '  <p class="lead">Big trend one and two.</p>\n'
    '</section>'
)


class GenSlidesTests(unittest.TestCase):
    def test_resolve_slide_root_uses_repo_relative_default(self) -> None:
        root = resolve_slide_root(None)
        self.assertEqual(root.name, "slide")
        self.assertTrue(root.is_absolute())

    def test_generate_slides_requires_non_empty_prompt(self) -> None:
        with self.assertRaises(SlideGenerationError):
            generate_slides("   ", api_key="sk-test")

    def test_validate_template_name_default(self) -> None:
        self.assertEqual(_validate_template_name(None), DEFAULT_TEMPLATE_NAME)

    def test_validate_template_name_known(self) -> None:
        self.assertEqual(_validate_template_name("8-bit-orbit"), "8-bit-orbit")

    def test_validate_template_name_unknown_raises(self) -> None:
        with self.assertRaises(SlideGenerationError):
            _validate_template_name("nonexistent-template")

    def test_load_template_splits_correctly(self) -> None:
        tpl = _load_template("monochrome")
        self.assertIn("<head>", tpl["prefix"])
        self.assertIn("<section", tpl["slides_html"])
        self.assertIn("<script>", tpl["suffix"])
        self.assertTrue(len(tpl["meta"]) > 0)

    def test_split_template_raises_on_no_slides(self) -> None:
        with self.assertRaises(SlideGenerationError):
            _split_template("<html><body><p>No slides here</p></body></html>")

    def test_extract_title_from_html(self) -> None:
        self.assertEqual(_extract_title_from_html('<h1 class="display">Hello World</h1>'), "Hello World")
        self.assertEqual(_extract_title_from_html('<h2>Sub<br>Title</h2>'), "Sub Title")
        self.assertEqual(_extract_title_from_html("<p>No heading</p>"), "")

    def test_clean_llm_output_strips_fences(self) -> None:
        self.assertEqual(_clean_llm_output("```html\n<section>x</section>\n```"), "<section>x</section>")
        self.assertEqual(_clean_llm_output("  <section>x</section>  "), "<section>x</section>")

    def test_generate_slides_writes_template_based_html(self) -> None:
        fake_client = FakeClient(SAMPLE_SLIDES_HTML)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                result = generate_slides(
                    "Create slides about AI.",
                    api_key="sk-test",
                    slide_dir=tmpdir,
                    slide_count=2,
                )

            html_path = Path(result["html_path"])
            self.assertTrue(html_path.exists())
            self.assertEqual(html_path.name, "index.html")
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("AI Briefing", html_text)
            self.assertEqual(html_text.count('<section class="slide'), 2)
            self.assertEqual(result["title"], "AI Briefing")
            self.assertEqual(result["slide_count"], 2)
            self.assertEqual(result["template_name"], "monochrome")
            calls = fake_client.chat.completions.calls
            self.assertEqual(calls[0]["model"], "qwen3-max")

    def test_generate_slides_with_custom_template(self) -> None:
        fake_client = FakeClient(SAMPLE_SLIDES_HTML)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                result = generate_slides(
                    "Create slides about gaming.",
                    api_key="sk-test",
                    slide_dir=tmpdir,
                    slide_count=2,
                    template_name="8-bit-orbit",
                )
            self.assertEqual(result["template_name"], "8-bit-orbit")

    def test_generate_slides_rejects_invalid_llm_output(self) -> None:
        fake_client = FakeClient("This is not HTML at all")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                with self.assertRaises(SlideGenerationError):
                    generate_slides(
                        "Create slides.",
                        api_key="sk-test",
                        slide_dir=tmpdir,
                        slide_count=2,
                    )

    def test_generate_slides_edit_mode(self) -> None:
        fake_client = FakeClient(SAMPLE_SLIDES_HTML)
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_path = Path(tmpdir) / "test" / "index.html"
            initial_path.parent.mkdir(parents=True)
            tpl = _load_template("monochrome")
            initial_html = tpl["prefix"] + SAMPLE_SLIDES_HTML + tpl["suffix"]
            initial_path.write_text(initial_html, encoding="utf-8")

            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                result = generate_slides(
                    "Change the title to Machine Learning",
                    api_key="sk-test",
                    slide_count=2,
                    html_path=str(initial_path),
                )
            self.assertEqual(result["html_path"], str(initial_path))
            self.assertTrue(initial_path.exists())
