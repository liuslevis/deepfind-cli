from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepfind.bili_transcribe import (
    BiliDownloadError,
    InvalidBiliIdError,
    MissingDependencyError,
    TranscriptionError,
)
from deepfind.config import Settings
from deepfind.tools import Toolset


class ToolsetTests(unittest.TestCase):
    def test_specs_use_chat_completion_function_shape(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = toolset.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "twitter_search")

    def test_specs_include_bili_transcribe(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        names = [item["function"]["name"] for item in toolset.specs()]
        self.assertIn("bili_transcribe", names)

    def test_missing_binary_returns_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value=None):
            result = toolset.twitter_search("topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "twitter not found")

    def test_successful_run_parses_json(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch(
                "deepfind.tools.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"items":[1]}', stderr=""),
            ):
                result = toolset.xhs_search("topic", page=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["items"], [])

    def test_twitter_search_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/twitter"):
            with patch(
                "deepfind.tools.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"items":[]}', stderr=""),
            ) as run_mock:
                toolset.twitter_search("robotics", max_results=5, tab="Latest")
        command = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args[0][0]
        self.assertEqual(
            command,
            ["/usr/bin/twitter", "search", "--max", "5", "robotics", "--json"],
        )

    def test_twitter_search_keeps_top_when_requested(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/twitter"):
            with patch(
                "deepfind.tools.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"items":[]}', stderr=""),
            ) as run_mock:
                toolset.twitter_search("robotics", max_results=5, tab="top")
        command = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args[0][0]
        self.assertEqual(
            command,
            ["/usr/bin/twitter", "search", "--type", "top", "--max", "5", "robotics", "--json"],
        )

    def test_xhs_search_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch(
                "deepfind.tools.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"items":[]}', stderr=""),
            ) as run_mock:
                toolset.xhs_search("keyword", page=1, sort="most_popular", note_type="image")
        first_call = run_mock.call_args_list[0]
        command = first_call.kwargs["args"] if "args" in first_call.kwargs else first_call[0][0]
        self.assertEqual(
            command,
            ["/usr/bin/xhs", "search", "--sort", "popular", "--type", "image", "--page", "1", "keyword", "--json"],
        )

    def test_xhs_search_fetches_ten_pages_by_default(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        page_payload = SimpleNamespace(returncode=0, stdout='{"items":[],"has_more":true}', stderr="")
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch("deepfind.tools.subprocess.run", return_value=page_payload) as run_mock:
                toolset.xhs_search("keyword")
        self.assertEqual(run_mock.call_count, 10)
        first = run_mock.call_args_list[0][0][0]
        last = run_mock.call_args_list[-1][0][0]
        self.assertEqual(first, ["/usr/bin/xhs", "search", "--sort", "general", "--type", "all", "--page", "1", "keyword", "--json"])
        self.assertEqual(last, ["/usr/bin/xhs", "search", "--sort", "general", "--type", "all", "--page", "10", "keyword", "--json"])

    def test_xhs_search_uses_page_as_upper_bound(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        page_payload = SimpleNamespace(returncode=0, stdout='{"items":[],"has_more":false}', stderr="")
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch("deepfind.tools.subprocess.run", return_value=page_payload) as run_mock:
                result = toolset.xhs_search("keyword", page=3)
        self.assertEqual(run_mock.call_count, 3)
        first = run_mock.call_args_list[0][0][0]
        last = run_mock.call_args_list[-1][0][0]
        self.assertEqual(first, ["/usr/bin/xhs", "search", "--sort", "general", "--type", "all", "--page", "1", "keyword", "--json"])
        self.assertEqual(last, ["/usr/bin/xhs", "search", "--sort", "general", "--type", "all", "--page", "3", "keyword", "--json"])
        self.assertEqual(result["data"]["pages_requested"], 3)
        self.assertEqual(result["data"]["pages_fetched"], 3)

    def test_bili_transcribe_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_bili_audio",
            return_value={
                "bili_id": "BV1cgPSzeEj5",
                "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                "transcript": "line one",
            },
        ):
            result = toolset.bili_transcribe("https://www.bilibili.com/video/BV1cgPSzeEj5")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_transcribe")
        self.assertEqual(result["data"]["bili_id"], "BV1cgPSzeEj5")
        self.assertEqual(result["data"]["transcript"], "line one")

    def test_bili_transcribe_invalid_bili_id_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=InvalidBiliIdError("bad id")):
            result = toolset.bili_transcribe("not a video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_bili_id")

    def test_bili_transcribe_missing_dependency_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=MissingDependencyError("missing")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_dependency")

    def test_bili_transcribe_download_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=BiliDownloadError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "download_failed")

    def test_bili_transcribe_transcription_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=TranscriptionError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "transcription_failed")
