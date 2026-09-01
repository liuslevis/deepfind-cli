from __future__ import annotations

import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepfind.coding import (
    CodingConfig,
    CodingService,
    _collect_artifacts,
    _validate_input,
    _validate_plan,
)
from deepfind.coding_runtime import (
    CodingRuntime,
    CodingRuntimeError,
    RuntimeExecution,
    RuntimeInfo,
    _extract_workspace_archive,
)
from deepfind.config import Settings


class FakeRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def probe(self) -> RuntimeInfo:
        return RuntimeInfo(
            runtime="docker",
            version="test",
            image="example/coding@sha256:" + ("a" * 64),
            image_digest="sha256:" + ("a" * 64),
        )

    def run(
        self,
        workspace: Path,
        task_id: str,
        *,
        timeout: int | None = None,
    ) -> RuntimeExecution:
        (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")
        control = workspace / ".deepfind"
        (control / "result.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "task_id": task_id,
                    "answer": "Created and validated solution.py.",
                    "commands": [
                        {
                            "command": ["python", "solution.py"],
                            "exit_code": 0,
                            "duration_ms": 12,
                            "stdout": "ok\n",
                            "stderr": "",
                        }
                    ],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        return RuntimeExecution(exit_code=0, stdout="", stderr="")


class CodingTests(unittest.TestCase):
    def test_runtime_security_options_are_fail_closed(self) -> None:
        runtime = CodingRuntime(
            runtime="docker",
            image="sha256:" + ("a" * 64),
            seccomp_profile=Path("deepfind/coding_seccomp.json"),
            timeout=120,
        )
        options = runtime._container_options("test-container")
        self.assertIn("none", options)
        self.assertIn("--read-only", options)
        self.assertIn("ALL", options)
        self.assertIn("no-new-privileges", options)
        self.assertTrue(any(value.startswith("seccomp=") for value in options))
        self.assertNotIn("--privileged", options)

    def test_workspace_archive_rejects_links(self) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            link = tarfile.TarInfo("escape")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside"
            bundle.addfile(link)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(CodingRuntimeError) as raised:
                _extract_workspace_archive(archive.getvalue(), Path(temp_dir))
        self.assertEqual(raised.exception.code, "invalid_result")

    def test_input_limits(self) -> None:
        self.assertEqual(_validate_input("  ", None).code, "invalid_input")
        self.assertEqual(_validate_input("x" * 20_001, None).code, "invalid_input")
        self.assertEqual(_validate_input("ok", "x" * 100_001).code, "invalid_input")
        self.assertIsNone(_validate_input("ok", "context"))

    def test_plan_rejects_path_traversal_and_non_python_commands(self) -> None:
        with self.assertRaises(ValueError):
            _validate_plan(
                {
                    "answer": "bad",
                    "files": [{"path": "../escape.py", "content": ""}],
                    "commands": [],
                }
            )
        with self.assertRaises(ValueError):
            _validate_plan(
                {
                    "answer": "bad",
                    "files": [],
                    "commands": [["sh", "-c", "echo bad"]],
                }
            )

    def test_plan_allows_readonly_shell_commands(self) -> None:
        normalized = _validate_plan(
            {
                "answer": "ok",
                "files": [],
                "commands": [
                    ["ls", "-la"],
                    ["cat", "solution.py"],
                    ["rg", "--", "pattern", "."],
                    ["grep", "-n", "TODO", "solution.py"],
                ],
            }
        )
        self.assertEqual(len(normalized["commands"]), 4)
        self.assertEqual(normalized["commands"][0][0], "ls")

    def test_plan_rejects_disallowed_binaries(self) -> None:
        for bad in (["rm", "-rf", "/"], ["curl", "http://x"], ["bash", "-lc", "x"], ["/usr/bin/ls"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                _validate_plan(
                    {
                        "answer": "bad",
                        "files": [],
                        "commands": [bad],
                    }
                )

    def test_collect_artifacts_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target.txt"
            target.write_text("safe", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(ValueError):
                _collect_artifacts(root)

    def test_service_returns_structured_success_and_cleans_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                api_key="test",
                coding_enabled=True,
                coding_image="example/coding@sha256:" + ("a" * 64),
                coding_root=temp_dir,
            )
            config = CodingConfig(
                runtime="docker",
                image=settings.coding_image,
                root=Path(temp_dir),
                timeout=120,
                max_concurrent=4,
                retention=0,
            )
            with (
                patch("deepfind.coding.CodingRuntime", FakeRuntime),
                patch(
                    "deepfind.coding.complete_text",
                    return_value=json.dumps(
                        {
                            "answer": "Created and validated solution.py.",
                            "files": [
                                {
                                    "path": "solution.py",
                                    "content": "print('ok')\n",
                                }
                            ],
                            "commands": [["python", "solution.py"]],
                        }
                    ),
                ),
            ):
                service = CodingService(settings, config)
                result = service.coding("Create a hello world program")

            self.assertTrue(result.ok)
            self.assertEqual(result.status, "completed")
            self.assertRegex(result.task_id, r"^task_[0-9a-f]{32}$")
            self.assertEqual(result.commands[0].stdout, "ok\n")
            self.assertEqual(result.artifacts[0].path, "solution.py")
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_invalid_input_does_not_create_task_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                api_key="test",
                coding_enabled=True,
                coding_image="example/coding@sha256:" + ("a" * 64),
                coding_root=temp_dir,
            )
            config = CodingConfig(
                runtime="docker",
                image=settings.coding_image,
                root=Path(temp_dir),
                timeout=120,
                max_concurrent=4,
                retention=0,
            )
            with patch("deepfind.coding.CodingRuntime", FakeRuntime):
                result = CodingService(settings, config).coding(" ")
            self.assertFalse(result.ok)
            self.assertEqual(result.error.code, "invalid_input")
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
