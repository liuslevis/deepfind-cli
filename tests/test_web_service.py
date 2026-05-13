from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4
from unittest.mock import patch

from deepfind.chat_store import ChatStore, repo_root
from deepfind.config import Settings
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


class CapturingApp(FakeApp):
    def __init__(self, progress, *, settings: Settings, seen: dict[str, Any]) -> None:
        super().__init__(progress)
        seen["settings"] = settings


class WebServiceTests(unittest.TestCase):
    def test_mode_mapping(self) -> None:
        self.assertEqual(mode_to_agent_count("fast"), 1)
        self.assertEqual(mode_to_agent_count("expert"), 4)

    def test_build_turn_result_collects_sources_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            image_name = f"{uuid4().hex}.png"
            html_name = f"{uuid4().hex}.html"
            image_path = repo_root() / "tmp" / image_name
            html_path = repo_root() / "tmp" / html_name
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
                                    "data": {"image_path": f"tmp/{image_name}"},
                                }
                            ),
                        ),
                        ToolObservation(
                            tool_name="gen_slides",
                            output=json.dumps(
                                {
                                    "ok": True,
                                    "tool": "gen_slides",
                                    "data": {"html_path": f"tmp/{html_name}"},
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

    def test_build_turn_result_includes_structured_key_points_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            result = service._build_turn_result(
                answer="Final answer",
                reports=[],
                observations=[],
                mode="fast",
                envelope={
                    "lead": {
                        "overview_md": "Final answer",
                        "key_points": [
                            {
                                "text": "Key fact",
                                "citation_ids": ["c1"],
                                "confidence": "high",
                            }
                        ],
                    },
                    "citations_dedup": [
                        {
                            "id": "c1",
                            "canonical_url": "https://example.com/report",
                            "url": "https://example.com/report?utm_source=news",
                            "title": "Example Report",
                            "publisher": "Example Publisher",
                        }
                    ],
                },
            )

        self.assertEqual(result.key_points[0].text, "Key fact")
        self.assertEqual(result.key_points[0].citation_ids, ["c1"])
        self.assertEqual(result.key_points[0].confidence, "high")
        self.assertEqual(result.citations[0].id, "c1")
        self.assertEqual(result.citations[0].canonical_url, "https://example.com/report")
        self.assertEqual(result.citations[0].title, "Example Report")

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

    def test_stream_message_uses_mimo_settings_for_mimo_target(self) -> None:
        seen: dict[str, Any] = {}
        base_settings = Settings(
            api_key="qwen-key",
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_api_key="qwen-key",
            qwen_model="qwen3-max",
            qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            mimo_api_key="mimo-key",
            mimo_model="mimo-v2.5-pro",
            mimo_base_url="https://api.xiaomimimo.com/v1",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress, settings: CapturingApp(progress, settings=settings, seen=seen),
            )
            chat = service.create_chat()
            with patch("deepfind.web_service.Settings.from_env", return_value=base_settings):
                events = list(service.stream_message(chat.id, "hello", "fast", "mimo"))

        self.assertEqual(events[-2].type, "answer_final")
        self.assertEqual(seen["settings"].api_key, "mimo-key")
        self.assertEqual(seen["settings"].model, "mimo-v2.5-pro")
        self.assertEqual(seen["settings"].base_url, "https://api.xiaomimimo.com/v1")

    def test_stream_message_uses_minimax_settings_for_minimax_target(self) -> None:
        seen: dict[str, Any] = {}
        base_settings = Settings(
            api_key="qwen-key",
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qwen_api_key="qwen-key",
            qwen_model="qwen3-max",
            qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            minimax_api_key="minimax-key",
            minimax_model="MiniMax-M2.7",
            minimax_base_url="https://api.minimax.io/v1",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress, settings: CapturingApp(progress, settings=settings, seen=seen),
            )
            chat = service.create_chat()
            with patch("deepfind.web_service.Settings.from_env", return_value=base_settings):
                events = list(service.stream_message(chat.id, "hello", "fast", "minimax"))

        self.assertEqual(events[-2].type, "answer_final")
        self.assertEqual(seen["settings"].api_key, "minimax-key")
        self.assertEqual(seen["settings"].model, "MiniMax-M2.7")
        self.assertEqual(seen["settings"].base_url, "https://api.minimax.io/v1")

    def test_stream_message_list_tool_returns_catalog_without_research_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            chat = service.create_chat()
            events = list(service.stream_message(chat.id, "/list-tool", "fast"))
            saved_chat = service.get_chat(chat.id)

        self.assertEqual([event.type for event in events], ["answer_final", "done"])
        answer = events[0].data["answer_markdown"]
        self.assertIn("Available tools:", answer)
        self.assertIn("- `web_search`:", answer)
        self.assertIn("- `gen_slides`:", answer)
        self.assertEqual(saved_chat.messages[0].content, "/list-tool")
        self.assertEqual(saved_chat.messages[1].content, answer)
