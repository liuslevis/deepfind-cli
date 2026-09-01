from __future__ import annotations

import io
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from deepfind.chat_store import ChatStore
from deepfind.web_api import build_app
from deepfind.web_service import DeepFindWebService
from deepfind.workspace import (
    ChatContainerManager,
    ChatContainerCodingRuntime,
    WorkspaceConfig,
    WorkspaceError,
    validate_relative_path,
)
from deepfind.config import Settings
from deepfind.tools import Toolset


class FakeWorkspaceManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(enabled=True)
        self.created: list[str] = []
        self.removed: list[str] = []

    def ensure(self, chat_id: str) -> None:
        self.created.append(chat_id)

    def remove(self, chat_id: str) -> None:
        self.removed.append(chat_id)

    def status(self, chat_id: str) -> dict:
        return {"available": True, "status": "running", "chat_id": chat_id, "workspace_root": "."}

    def list_files(self, chat_id: str, path: str) -> dict:
        return {
            "path": path,
            "entries": [
                {
                    "name": "report.txt",
                    "path": "report.txt",
                    "size": 5,
                    "modified_at": "2026-08-31T00:00:00Z",
                    "is_directory": False,
                    "mime_type": "text/plain",
                }
            ],
            "truncated": False,
        }

    def metadata(self, chat_id: str, path: str) -> dict:
        return {
            "name": Path(path).name,
            "path": path,
            "size": 5,
            "modified_at": "2026-08-31T00:00:00Z",
            "is_directory": False,
            "mime_type": "text/plain",
        }

    def read_file(
        self,
        chat_id: str,
        path: str,
        *,
        limit: int,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        content = b"hello"
        return content[offset : offset + length if length is not None else None]

    def workbook(self, chat_id: str, path: str) -> dict:
        return {"file": self.metadata(chat_id, path), "sheets": [], "warnings": []}

    def sheet(self, chat_id: str, path: str, sheet_id: str, cell_range: str) -> dict:
        return {"sheet_id": sheet_id, "range": cell_range, "cells": [], "truncated": False}

    def word(self, chat_id: str, path: str) -> dict:
        return {"title": "Document", "outline": [], "html": "<p>hello</p>", "warnings": []}

    def create_terminal(self, chat_id: str) -> dict:
        return {"terminal_id": "term_" + ("a" * 32), "chat_id": chat_id, "status": "ready"}

    def list_terminals(self, chat_id: str) -> list[dict]:
        return []

    def close_terminal(self, chat_id: str, terminal_id: str) -> None:
        return None


class FakeCodingManager:
    def info(self):
        return SimpleNamespace(runtime="docker", version="test", image="test", image_digest="sha256:test")

    def run_coding(self, chat_id: str, request: dict, timeout: int) -> dict:
        return {
            "ok": True,
            "task_id": request["task_id"],
            "answer": "done",
            "commands": [],
            "artifacts": ["result.txt"],
            "error": None,
        }

    def export_files(self, chat_id: str, paths: list[str]) -> bytes:
        import tarfile

        self.exported_paths = paths
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            info = tarfile.TarInfo("result.txt")
            payload = b"done"
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        return archive.getvalue()

    def run_coding_task(self, chat_id: str, request: dict, timeout: int):
        result = self.run_coding(chat_id, request, timeout)
        return result, self.export_files(chat_id, result["artifacts"])


class WorkspaceTests(unittest.TestCase):
    def test_relative_paths_fail_closed(self) -> None:
        self.assertEqual(validate_relative_path("."), ".")
        self.assertEqual(validate_relative_path("reports/q3.pdf"), "reports/q3.pdf")
        for value in ("../secret", "/etc/passwd", "a//b", ".env", ".git/config", "a\\b", "a\x00b"):
            with self.subTest(value=value), self.assertRaises(WorkspaceError):
                validate_relative_path(value)

    def test_chat_lifecycle_creates_and_removes_workspace(self) -> None:
        manager = FakeWorkspaceManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            chat = service.create_chat()
            service.delete_chat(chat.id)
        self.assertEqual(manager.created, [chat.id])
        self.assertEqual(manager.removed, [chat.id])

    def test_workspace_api_is_chat_scoped_and_read_only(self) -> None:
        manager = FakeWorkspaceManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            client = TestClient(build_app(service))
            chat_id = client.post("/api/chats", json={}).json()["chat"]["id"]

            status = client.get(f"/api/chats/{chat_id}/workspace")
            listing = client.get(f"/api/chats/{chat_id}/workspace/files", params={"path": "."})
            content = client.get(
                f"/api/chats/{chat_id}/workspace/content",
                params={"path": "report.txt"},
            )
            ranged = client.get(
                f"/api/chats/{chat_id}/workspace/content",
                params={"path": "report.txt"},
                headers={"Range": "bytes=1-3"},
            )

        self.assertEqual(status.status_code, 200)
        self.assertEqual(listing.json()["entries"][0]["path"], "report.txt")
        self.assertEqual(content.content, b"hello")
        self.assertEqual(content.headers["cache-control"], "no-store")
        self.assertEqual(ranged.status_code, 206)
        self.assertEqual(ranged.content, b"ell")
        self.assertEqual(ranged.headers["content-range"], "bytes 1-3/5")

    def test_container_coding_runtime_keeps_outputs_in_chat_workspace(self) -> None:
        runtime = ChatContainerCodingRuntime(FakeCodingManager(), "chat_example")
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            control = workspace / ".deepfind"
            control.mkdir()
            (control / "request.json").write_text(
                json.dumps({"task_id": "task_example"}),
                encoding="utf-8",
            )
            result = runtime.run(workspace, "task_example", timeout=30)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual((workspace / "result.txt").read_text(encoding="utf-8"), "done")

    def test_agent_terminal_tool_only_returns_an_approval_proposal(self) -> None:
        toolset = Toolset(Settings(api_key=""))
        result = toolset.propose_terminal_command(
            "pytest -q",
            "Inspect the failing tests",
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["proposal"]["requires_approval"])
        self.assertEqual(result["proposal"]["cwd"], ".")
        self.assertNotIn("terminal_id", result)

    def test_coding_tasks_are_serialized_per_chat(self) -> None:
        manager = ChatContainerManager(
            WorkspaceConfig(
                enabled=False,
                runtime="docker",
                image="test",
                seccomp_profile=Path("deepfind/coding_seccomp.json"),
            )
        )
        active = 0
        peak = 0
        guard = threading.Lock()

        def run_coding(chat_id: str, request: dict, timeout: int) -> dict:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return {"ok": True, "artifacts": [], "task_id": request["task_id"]}

        with patch.object(manager, "run_coding", side_effect=run_coding):
            threads = [
                threading.Thread(
                    target=manager.run_coding_task,
                    args=("chat_same", {"task_id": f"task_{index}"}, 30),
                )
                for index in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
