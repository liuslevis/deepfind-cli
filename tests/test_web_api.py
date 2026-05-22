from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from deepfind.chat_store import ChatStore
from deepfind.config import Settings
from deepfind.models import WorkerReport
from deepfind.web_api import build_app
from deepfind.web_service import DeepFindWebService


class FakeApp:
    def __init__(self, progress) -> None:
        self.progress = progress

    def _run_turn_detailed(self, *, query, transcript, num_agent, max_iter_per_agent):
        self.progress.run_started(query, num_agent, max_iter_per_agent)
        self.progress.plan_ready(["task"])
        report = WorkerReport(
            task="task",
            text="text",
            citations=[],
            parsed={"summary": "summary", "facts": [], "gaps": []},
        )
        return "final answer", [report]


class WebApiTests(unittest.TestCase):
    def test_chat_endpoints_and_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress: FakeApp(progress),
            )
            client = TestClient(build_app(service))

            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["ok"], True)

            created = client.post("/api/chats", json={})
            self.assertEqual(created.status_code, 200)
            chat_id = created.json()["chat"]["id"]

            listed = client.get("/api/chats")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()["chats"]), 1)

            detail = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["chat"]["id"], chat_id)

            streamed = client.post(
                f"/api/chats/{chat_id}/messages/stream",
                json={"content": "hello", "mode": "fast"},
            )
            self.assertEqual(streamed.status_code, 200)
            self.assertIn("event: run_started", streamed.text)
            self.assertIn("event: console_line", streamed.text)
            self.assertIn("event: answer_final", streamed.text)

            after = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(len(after.json()["chat"]["messages"]), 2)

            deleted = client.delete(f"/api/chats/{chat_id}")
            self.assertEqual(deleted.status_code, 204)

    def test_stream_endpoint_accepts_legacy_cloud_model_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                app_factory=lambda progress: FakeApp(progress),
            )
            client = TestClient(build_app(service))
            chat_id = client.post("/api/chats", json={}).json()["chat"]["id"]

            streamed = client.post(
                f"/api/chats/{chat_id}/messages/stream",
                json={"content": "hello", "mode": "fast", "model_target": "cloud"},
            )

            self.assertEqual(streamed.status_code, 200)
            self.assertIn("event: answer_final", streamed.text)
            after = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(after.json()["chat"]["messages"][0]["model_target"], "qwen")

    def test_stream_endpoint_returns_plain_error_when_mimo_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(store=ChatStore(Path(temp_dir)))
            client = TestClient(build_app(service))
            chat_id = client.post("/api/chats", json={}).json()["chat"]["id"]
            settings = Settings(
                api_key="qwen-key",
                model="qwen3-max",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                qwen_api_key="qwen-key",
                qwen_model="qwen3-max",
                qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            with patch("deepfind.web_service.Settings.from_env", return_value=settings):
                streamed = client.post(
                    f"/api/chats/{chat_id}/messages/stream",
                    json={"content": "hello", "mode": "fast", "model_target": "mimo"},
                )

            self.assertEqual(streamed.status_code, 400)
            self.assertEqual(streamed.text, "Set MIMO_API_KEY or XIAOMI_API_KEY, or switch to another model.")
