from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from deepfind.chat_store import ChatStore
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
            self.assertIn("event: answer_final", streamed.text)

            after = client.get(f"/api/chats/{chat_id}")
            self.assertEqual(len(after.json()["chat"]["messages"]), 2)

            deleted = client.delete(f"/api/chats/{chat_id}")
            self.assertEqual(deleted.status_code, 204)
