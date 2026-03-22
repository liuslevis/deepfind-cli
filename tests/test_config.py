from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepfind.config import DEFAULT_BASE_URL, DEFAULT_MODEL, Settings, _clean_env_value
from deepfind.gen_img import DEFAULT_IMAGE_DIR, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_SIZE


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
            "GOOGLE_NANO_BANANA_API_KEY": "nb-key # comment",
            "GOOGLE_NANO_BANANA_MODEL": f"{DEFAULT_IMAGE_MODEL} # comment",
            "DEEPFIND_IMAGE_DIR": f"{DEFAULT_IMAGE_DIR} # comment",
            "GOOGLE_NANO_BANANA_IMAGE_SIZE": f"{DEFAULT_IMAGE_SIZE} # comment",
            "OPENCLI_BIN": "opencli-custom # comment",
            "TWITTER_CLI_BIN": "twitter # comment",
            "XHS_CLI_BIN": "xhs # comment",
            "BILI_BIN": "bili # comment",
            "ASR_MODEL": "Qwen/Qwen3-ASR-1.7B # comment",
            "DEEPFIND_AUDIO_DIR": "audio # comment",
            "DEEPFIND_TOOL_TIMEOUT": "45 # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-test")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_BASE_URL)
        self.assertEqual(settings.nano_banana_api_key, "nb-key")
        self.assertEqual(settings.nano_banana_model, DEFAULT_IMAGE_MODEL)
        self.assertEqual(settings.image_dir, DEFAULT_IMAGE_DIR)
        self.assertEqual(settings.image_size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(settings.opencli_bin, "opencli-custom")
        self.assertEqual(settings.twitter_bin, "twitter")
        self.assertEqual(settings.xhs_bin, "xhs")
        self.assertEqual(settings.bili_bin, "bili")
        self.assertEqual(settings.asr_model, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(settings.audio_dir, "audio")
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
                        f"GOOGLE_NANO_BANANA_API_KEY=nb-key",
                        f"GOOGLE_NANO_BANANA_MODEL={DEFAULT_IMAGE_MODEL}",
                        f"DEEPFIND_IMAGE_DIR={DEFAULT_IMAGE_DIR}",
                        f"GOOGLE_NANO_BANANA_IMAGE_SIZE={DEFAULT_IMAGE_SIZE}",
                        "OPENCLI_BIN=opencli-dotenv",
                        "TWITTER_CLI_BIN=twitter",
                        "XHS_CLI_BIN=xhs",
                        "BILI_BIN=bili",
                        "ASR_MODEL=Qwen/Qwen3-ASR-1.7B",
                        "DEEPFIND_AUDIO_DIR=audio",
                        "DEEPFIND_TOOL_TIMEOUT=30",
                    ]
                )
            )
            with patch.dict(os.environ, {"DEEPFIND_ENV_FILE": str(env_path)}, clear=True):
                settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-from-dotenv")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_BASE_URL)
        self.assertEqual(settings.nano_banana_api_key, "nb-key")
        self.assertEqual(settings.nano_banana_model, DEFAULT_IMAGE_MODEL)
        self.assertEqual(settings.image_dir, DEFAULT_IMAGE_DIR)
        self.assertEqual(settings.image_size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(settings.opencli_bin, "opencli-dotenv")
        self.assertEqual(settings.bili_bin, "bili")
        self.assertEqual(settings.audio_dir, "audio")
        self.assertEqual(settings.subprocess_timeout, 30)
