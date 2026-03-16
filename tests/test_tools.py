from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.tools import Toolset


class ToolsetTests(unittest.TestCase):
    def test_specs_use_chat_completion_function_shape(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = toolset.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "twitter_search")

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
                result = toolset.xhs_search("topic")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"], {"items": [1]})
