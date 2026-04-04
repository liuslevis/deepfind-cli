from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from deepfind.chat_store import ChatStore, repo_root
from deepfind.models import WorkerReport
from deepfind.web_service import DeepFindWebService, mode_to_agent_count
from deepfind.web_progress import ToolObservation


class FakeApp:
    def __init__(self, progress, *, should_fail: bool = False) -> None:
        self.progress = progress
        self.should_fail = should_fail

    def _run_turn_detailed(self, *, query, transcript, num_agent, max_iter_per_agent):
        self.progress.run_started(query, num_agent, max_iter_per_agent)
        self.progress.plan_ready(["core facts"])
        self.progress.worker_started("sub-1", "core facts")
        self.progress.iteration("sub-1", 1)
        self.progress.tool_call("sub-1", 1, "web_search", {"query": query, "engine": "google"})
        self.progress.tool_result(
            "sub-1",
            "web_search",
            '{"ok": true, "tool": "web_search", "data": {"items": [{"url": "https://example.com/source"}]}}',
        )
        self.progress.synthesize_started(1)
        if self.should_fail:
            raise RuntimeError("boom")
        report = WorkerReport(
            task="core facts",
            text="See https://example.com/report",
            citations=[],
            parsed={
                "summary": "summary",
                "facts": [
                    {
                        "point": "fact",
                        "source": "https://example.com/report",
                    }
                ],
                "gaps": [],
            },
        )
        return "Answer with https://example.com/final", [report]


class WebServiceTests(unittest.TestCase):
    def test_mode_mapping(self) -> None:
        self.assertEqual(mode_to_agent_count("fast"), 1)
        self.assertEqual(mode_to_agent_count("expert"), 4)

    def test_build_turn_result_collects_sources_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            image_path = repo_root() / "tmp" / f"{uuid4().hex}.png"
            html_path = repo_root() / "tmp" / f"{uuid4().hex}.html"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_text("png", encoding="utf-8")
            html_path.write_text("<html></html>", encoding="utf-8")
            try:
                result = service._build_turn_result(
                    answer="Final answer https://example.com/final",
                    reports=[
                        WorkerReport(
                            task="core facts",
                            text="extra https://example.com/report",
                            citations=[],
                            parsed={
                                "summary": "summary",
                                "facts": [
                                    {
                                        "point": "fact",
                                        "source": "https://example.com/report",
                                    }
                                ],
                                "gaps": [],
                            },
                        )
                    ],
                    observations=[
                        ToolObservation(
                            tool_name="gen_img",
                            output=json.dumps(
                                {
                                    "ok": True,
                                    "tool": "gen_img",
                                    "data": {"image_path": str(image_path)},
                                }
                            ),
                        ),
                        ToolObservation(
                            tool_name="gen_slides",
                            output=json.dumps(
                                {
                                    "ok": True,
                                    "tool": "gen_slides",
                                    "data": {"html_path": str(html_path)},
                                }
                            ),
                        ),
                        ToolObservation(
                            tool_name="web_search",
                            output='{"ok": true, "tool": "web_search", "data": {"items": [{"url": "https://example.com/source"}]}}',
                        ),
                    ],
                    mode="expert",
                )
            finally:
                image_path.unlink(missing_ok=True)
                html_path.unlink(missing_ok=True)

        self.assertEqual(result.mode, "expert")
        self.assertEqual(
            result.sources,
            [
                "https://example.com/final",
                "https://example.com/report",
                "https://example.com/source",
            ],
        )
        self.assertEqual(len(result.artifacts), 2)
        self.assertTrue(result.artifacts[0].url.startswith("/api/files?path="))

    def test_build_turn_result_collects_web_fetch_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            result = service._build_turn_result(
                answer="Final answer",
                reports=[],
                observations=[
                    ToolObservation(
                        tool_name="web_fetch",
                        output=json.dumps(
                            {
                                "ok": True,
                                "tool": "web_fetch",
                                "data": {
                                    "url": "https://example.com/start",
                                    "final_url": "https://example.com/final",
                                    "title": "Example",
                                    "summary": "summary",
                                    "content_type": "text/html",
                                    "truncated": False,
                                    "markdown_chars": 123,
                                },
                            }
                        ),
                    )
                ],
                mode="fast",
            )

        self.assertEqual(
            result.sources,
            [
                "https://example.com/start",
                "https://example.com/final",
            ],
        )

    def test_stream_message_emits_ordered_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress: FakeApp(progress),
            )
            chat = service.create_chat()
            events = list(service.stream_message(chat.id, "hello", "fast"))
            saved_chat = service.get_chat(chat.id)

        self.assertEqual(events[0].type, "run_started")
        self.assertIn("plan_ready", [event.type for event in events])
        self.assertIn("console_line", [event.type for event in events])
        self.assertEqual(events[-2].type, "answer_final")
        self.assertEqual(events[-1].type, "done")
        self.assertEqual(saved_chat.messages[-1].role, "assistant")

    def test_stream_message_emits_error_then_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress: FakeApp(progress, should_fail=True),
            )
            chat = service.create_chat()
            events = list(service.stream_message(chat.id, "hello", "expert"))

        self.assertEqual(events[-2].type, "error")
        self.assertEqual(events[-1].type, "done")
