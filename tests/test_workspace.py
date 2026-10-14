from __future__ import annotations

import io
import json
import queue
import subprocess
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

    def test_chat_lifecycle_does_not_create_workspace_and_still_cleans_up(self) -> None:
        manager = FakeWorkspaceManager()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            chat = service.create_chat()
            service.delete_chat(chat.id)
        self.assertEqual(manager.created, [])
        self.assertEqual(manager.removed, [chat.id])

    def test_chat_creation_does_not_require_workspace_runtime(self) -> None:
        manager = FakeWorkspaceManager()
        error = WorkspaceError("workspace_unavailable", "Start Docker Desktop, then retry.", 503)
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            client = TestClient(build_app(service))
            with patch.object(manager, "ensure", side_effect=error) as ensure:
                response = client.post("/api/chats", json={})
                ensure.assert_not_called()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(service.list_chats()), 1)
            self.assertEqual(manager.created, [])

    def test_opening_workspace_creates_it_on_demand_and_preserves_chat_on_failure(self) -> None:
        manager = FakeWorkspaceManager()
        error = WorkspaceError("workspace_unavailable", "Start Docker Desktop, then retry.", 503)
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            client = TestClient(build_app(service))
            chat_id = client.post("/api/chats", json={}).json()["chat"]["id"]
            with patch.object(manager, "ensure", side_effect=error):
                response = client.get(f"/api/chats/{chat_id}/workspace")

            self.assertEqual(response.status_code, error.status_code)
            self.assertEqual(
                response.json(),
                {"detail": {"code": error.code, "message": error.message}},
            )
            self.assertEqual(service.get_chat(chat_id).id, chat_id)

            retried = client.get(f"/api/chats/{chat_id}/workspace")
            self.assertEqual(retried.status_code, 200)
            self.assertEqual(len(service.list_chats()), 1)
            self.assertEqual(manager.created, [chat_id])

    def test_unselected_coding_does_not_initialize_coding_service(self) -> None:
        settings = Settings(api_key="", coding_enabled=True)
        manager = FakeWorkspaceManager()
        with patch("deepfind.tools.CodingService") as coding_service:
            with patch("deepfind.tools.get_coding_service") as get_coding_service:
                for chat_id in (None, "chat_example"):
                    with self.subTest(chat_id=chat_id):
                        tools = Toolset(
                            settings,
                            enabled_tools=["web_search"],
                            chat_id=chat_id,
                            workspace_manager=manager,
                        )
                        self.assertIsNone(tools._coding_service)
                coding_service.assert_not_called()
                get_coding_service.assert_not_called()
        self.assertEqual(manager.created, [])

    def test_concurrent_workspace_requests_create_only_one_container(self) -> None:
        manager = ChatContainerManager(
            WorkspaceConfig(
                enabled=True,
                runtime="docker",
                image="example/coding@sha256:" + ("a" * 64),
                seccomp_profile=Path(__file__).with_name("coding_seccomp.json"),
            )
        )
        created = threading.Event()
        start = threading.Barrier(2)
        errors: queue.Queue[Exception] = queue.Queue()

        def inspect(command, **kwargs):
            state = b'[{"State":{"Running":true}}]' if created.is_set() else b"[]"
            return subprocess.CompletedProcess(command, 0 if created.is_set() else 1, state, b"")

        def run(command, **kwargs):
            if command[1] == "create":
                time.sleep(0.02)
                created.set()
            return subprocess.CompletedProcess(command, 0, b"", b"")

        def ensure():
            try:
                start.wait(timeout=2)
                manager.ensure("chat_example")
            except Exception as exc:
                errors.put(exc)

        with patch.object(manager, "info"):
            with patch.object(manager, "_run", side_effect=run) as runtime:
                with patch("deepfind.workspace.subprocess.run", side_effect=inspect):
                    threads = [threading.Thread(target=ensure, daemon=True) for _ in range(2)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=2)
                    self.assertTrue(all(not thread.is_alive() for thread in threads))
                    self.assertTrue(errors.empty(), list(errors.queue))
                    creates = [call for call in runtime.call_args_list if call.args[0][1] == "create"]
                    self.assertEqual(len(creates), 1)

    def test_chat_deletion_returns_workspace_error_and_preserves_chat(self) -> None:
        manager = FakeWorkspaceManager()
        error = WorkspaceError("workspace_cleanup_failed", "The chat workspace could not be removed", 500)
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DeepFindWebService(
                store=ChatStore(Path(temp_dir)),
                workspace_manager=manager,
            )
            chat = service.create_chat()
            client = TestClient(build_app(service))
            with patch.object(manager, "remove", side_effect=error):
                response = client.delete(f"/api/chats/{chat.id}")

            self.assertEqual(response.status_code, error.status_code)
            self.assertEqual(
                response.json(),
                {"detail": {"code": error.code, "message": error.message}},
            )
            self.assertEqual(service.get_chat(chat.id).id, chat.id)

    def test_runtime_failures_include_recovery_guidance(self) -> None:
        manager = ChatContainerManager(
            WorkspaceConfig(
                enabled=True,
                runtime="docker",
                image="test",
                seccomp_profile=Path("deepfind/coding_seccomp.json"),
            )
        )
        for failure, guidance in (
            (FileNotFoundError("docker"), "PATH"),
            (subprocess.TimeoutExpired("docker", 15), "timed out"),
        ):
            with self.subTest(failure=failure):
                with patch("deepfind.workspace.subprocess.run", side_effect=failure):
                    with self.assertRaises(WorkspaceError) as raised:
                        manager.info()
                self.assertEqual(raised.exception.code, "workspace_unavailable")
                self.assertEqual(raised.exception.status_code, 503)
                self.assertIn(guidance, raised.exception.message)

        result = subprocess.CompletedProcess(
            args=["docker", "version"],
            returncode=1,
            stdout=b"",
            stderr=b"Cannot connect to the Docker daemon",
        )
        with patch("deepfind.workspace.subprocess.run", return_value=result):
            with self.assertLogs("deepfind.workspace", level="WARNING") as logs:
                with self.assertRaises(WorkspaceError) as raised:
                    manager.info()
        self.assertEqual(raised.exception.code, "workspace_unavailable")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("Docker Desktop", raised.exception.message)
        self.assertIn("configured coding image", raised.exception.message)
        self.assertIn("Cannot connect to the Docker daemon", logs.output[0])

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
