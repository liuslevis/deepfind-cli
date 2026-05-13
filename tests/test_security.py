"""Tests for security hardening features."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from deepfind.chat_store import ChatStore
from deepfind.web_fetch import SsrfBlockedError, validate_fetch_url
from deepfind.web_service import DeepFindWebService


class PathTraversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = DeepFindWebService(store=ChatStore(Path(self.tmp.name)))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rejects_absolute_path(self) -> None:
        with self.assertRaises(ValueError, msg="absolute paths not allowed"):
            self.service.resolve_file_path("/etc/passwd")

    def test_rejects_windows_absolute_path(self) -> None:
        with self.assertRaises(ValueError, msg="absolute paths not allowed"):
            self.service.resolve_file_path("C:\\Windows\\System32\\config\\SAM")

    def test_rejects_dotdot_traversal(self) -> None:
        with self.assertRaises(ValueError):
            self.service.resolve_file_path("../../etc/passwd")

    def test_rejects_dotenv(self) -> None:
        with self.assertRaises(ValueError, msg="access denied"):
            self.service.resolve_file_path(".env")

    def test_rejects_git_directory(self) -> None:
        with self.assertRaises(ValueError, msg="access denied"):
            self.service.resolve_file_path(".git/config")

    def test_accepts_valid_relative_path(self) -> None:
        result = self.service.resolve_file_path("tmp/test.png")
        self.assertTrue(str(result).endswith("test.png"))


class ChatIdValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rejects_path_traversal_chat_id(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store._path("../../../etc/passwd")

    def test_rejects_empty_chat_id(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store._path("")

    def test_rejects_json_injection(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store._path("chat_test.json/../../../etc")

    def test_accepts_valid_chat_id(self) -> None:
        path = self.store._path("chat_abcdef0123456789abcdef0123456789")
        self.assertTrue(str(path).endswith(".json"))


class SsrfProtectionTests(unittest.TestCase):
    def test_blocks_localhost(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("http://127.0.0.1/admin")

    def test_blocks_private_10_network(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("http://10.0.0.1/internal")

    def test_blocks_private_172_network(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("http://172.16.0.1/internal")

    def test_blocks_private_192_network(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("http://192.168.1.1/admin")

    def test_blocks_metadata_ip(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_ftp_scheme(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("ftp://evil.com/file")

    def test_blocks_file_scheme(self) -> None:
        with self.assertRaises(SsrfBlockedError):
            validate_fetch_url("file:///etc/passwd")

    def test_allows_public_url(self) -> None:
        # Should not raise
        validate_fetch_url("https://example.com/page")


class AuthMiddlewareTests(unittest.TestCase):
    def test_configured_token_returns_empty_without_env(self) -> None:
        import os
        old = os.environ.pop("DEEPFIND_WEB_TOKEN", None)
        try:
            from deepfind.web_api import _configured_token
            self.assertFalse(_configured_token())
        finally:
            if old is not None:
                os.environ["DEEPFIND_WEB_TOKEN"] = old

    def test_configured_token_returns_value_with_env(self) -> None:
        import os
        old = os.environ.get("DEEPFIND_WEB_TOKEN")
        os.environ["DEEPFIND_WEB_TOKEN"] = "test-secret-123"
        try:
            from deepfind.web_api import _configured_token
            self.assertEqual(_configured_token(), "test-secret-123")
        finally:
            if old is not None:
                os.environ["DEEPFIND_WEB_TOKEN"] = old
            else:
                os.environ.pop("DEEPFIND_WEB_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
