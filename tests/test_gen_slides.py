from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepfind.gen_slides import SlideGenerationError, generate_slides, resolve_slide_root


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


class GenSlidesTests(unittest.TestCase):
    def test_resolve_slide_root_uses_repo_relative_default(self) -> None:
        root = resolve_slide_root(None)
        self.assertEqual(root.name, "tmp")
        self.assertTrue(root.is_absolute())

    def test_generate_slides_requires_non_empty_prompt(self) -> None:
        with self.assertRaises(SlideGenerationError):
            generate_slides("   ", api_key="sk-test")

    def test_generate_slides_writes_standalone_html(self) -> None:
        outline = (
            '{"title":"AI Briefing","slides":['
            '{"title":"Overview","bullets":["Big trend one","Big trend two"]},'
            '{"title":"Launches","bullets":["Product A","Product B"]}'
            "]}"
        )
        fake_client = FakeClient(outline)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                result = generate_slides(
                    "Create slides from this summary.",
                    api_key="sk-test",
                    slide_dir=tmpdir,
                    slide_count=2,
                )

            html_path = Path(result["html_path"])
            self.assertTrue(html_path.exists())
            self.assertEqual(html_path.parent, Path(tmpdir))
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("<style>", html_text)
            self.assertIn("<script>", html_text)
            self.assertEqual(html_text.count('<section class="slide"'), 2)
            self.assertEqual(result["title"], "AI Briefing")
            self.assertEqual(result["slide_count"], 2)
            calls = fake_client.chat.completions.calls
            self.assertEqual(calls[0]["model"], "qwen3-max")

    def test_generate_slides_rejects_non_json_outline(self) -> None:
        fake_client = FakeClient("not json")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                with self.assertRaises(SlideGenerationError):
                    generate_slides(
                        "Create slides from this summary.",
                        api_key="sk-test",
                        slide_dir=tmpdir,
                        slide_count=2,
                    )

    def test_generate_slides_rejects_wrong_slide_count(self) -> None:
        outline = '{"title":"AI Briefing","slides":[{"title":"Only Slide","bullets":["One point"]}]}'
        fake_client = FakeClient(outline)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_slides.OpenAI", return_value=fake_client):
                with self.assertRaises(SlideGenerationError):
                    generate_slides(
                        "Create slides from this summary.",
                        api_key="sk-test",
                        slide_dir=tmpdir,
                        slide_count=2,
                    )
