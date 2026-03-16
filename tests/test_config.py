from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepfind.config import DEFAULT_BASE_URL, DEFAULT_MODEL, Settings, _clean_env_value


class ConfigTests(unittest.TestCase):
    def test_clean_env_value_strips_inline_comment(self) -> None:
        self.assertEqual(
            _clean_env_value("qwen3-max # qwen-flash qwen-plus qwen3-max"),
            "qwen3-max",
        )

    def test_from_env_uses_sanitized_values(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": "/tmp/deepfind-missing.env",
            "QWEN_API_KEY": "sk-test",
            "QWEN_MODEL_NAME": "qwen3-max # comment",
            "QWEN_BASE_URL": f"{DEFAULT_BASE_URL} # comment",
            "TWITTER_CLI_BIN": "twitter # comment",
            "XHS_CLI_BIN": "xhs # comment",
            "DEEPFIND_TOOL_TIMEOUT": "45 # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-test")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_BASE_URL)
        self.assertEqual(settings.twitter_bin, "twitter")
        self.assertEqual(settings.xhs_bin, "xhs")
        self.assertEqual(settings.subprocess_timeout, 45)

    def test_from_env_loads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "QWEN_API_KEY=sk-from-dotenv",
                        "QWEN_MODEL_NAME=qwen3-max # comment",
                        "QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "TWITTER_CLI_BIN=twitter",
                        "XHS_CLI_BIN=xhs",
                        "DEEPFIND_TOOL_TIMEOUT=30",
                    ]
                )
            )
            with patch.dict(os.environ, {"DEEPFIND_ENV_FILE": str(env_path)}, clear=True):
                settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-from-dotenv")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_BASE_URL)
        self.assertEqual(settings.subprocess_timeout, 30)
