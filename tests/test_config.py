from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepfind.config import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MIMO_BASE_URL,
    DEFAULT_MIMO_MODEL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    Settings,
    SettingsError,
    _clean_env_value,
)
from deepfind.gen_img import DEFAULT_IMAGE_DIR, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_SIZE


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._env_dir.cleanup)
        # An existing but empty .env keeps these tests hermetic: the loader stops
        # at the first readable file, so the developer's repository .env is not
        # picked up while the tests run.
        self.empty_env_file = Path(self._env_dir.name) / ".env"
        self.empty_env_file.write_text("")

    def test_storage_config_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {"DEEPFIND_ENV_FILE": str(self.empty_env_file)},
            clear=True,
        ):
            settings = Settings.from_env(require_api_key=False)
        self.assertEqual(settings.audio_dir, "assets/video")
        self.assertEqual(settings.rag_dir, "assets")

    def test_clean_env_value_strips_inline_comment(self) -> None:
        self.assertEqual(
            _clean_env_value("qwen3-max # qwen-flash qwen-plus qwen3-max"),
            "qwen3-max",
        )

    def test_from_env_uses_sanitized_values(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "QWEN_API_KEY": "sk-test",
            "QWEN_MODEL_NAME": "qwen3-max # comment",
            "QWEN_BASE_URL": f"{DEFAULT_QWEN_BASE_URL} # comment",
            "GOOGLE_NANO_BANANA_API_KEY": "nb-key # comment",
            "GOOGLE_NANO_BANANA_MODEL": f"{DEFAULT_IMAGE_MODEL} # comment",
            "DEEPFIND_IMAGE_DIR": f"{DEFAULT_IMAGE_DIR} # comment",
            "GOOGLE_NANO_BANANA_IMAGE_SIZE": f"{DEFAULT_IMAGE_SIZE} # comment",
            "OPENCLI_BIN": "opencli-custom # comment",
            "TWITTER_CLI_BIN": "twitter # comment",
            "XHS_CLI_BIN": "xhs # comment",
            "BILI_BIN": "bili # comment",
            "ASR_MODEL": "Qwen/Qwen3-ASR-1.7B # comment",
            "DEEPFIND_VIDEO_DIR": "video # comment",
            "DEEPFIND_TOOL_TIMEOUT": "45 # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-test")
        self.assertEqual(settings.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_QWEN_BASE_URL)
        self.assertEqual(settings.qwen_api_key, "sk-test")
        self.assertEqual(settings.qwen_model, DEFAULT_QWEN_MODEL)
        self.assertEqual(settings.qwen_base_url, DEFAULT_QWEN_BASE_URL)
        self.assertEqual(settings.mimo_api_key, "")
        self.assertEqual(settings.mimo_model, DEFAULT_MIMO_MODEL)
        self.assertEqual(settings.mimo_base_url, DEFAULT_MIMO_BASE_URL)
        self.assertEqual(settings.minimax_api_key, "")
        self.assertEqual(settings.minimax_model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.minimax_base_url, DEFAULT_MINIMAX_BASE_URL)
        self.assertEqual(settings.nano_banana_api_key, "nb-key")
        self.assertEqual(settings.nano_banana_model, DEFAULT_IMAGE_MODEL)
        self.assertEqual(settings.image_dir, DEFAULT_IMAGE_DIR)
        self.assertEqual(settings.image_size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(settings.opencli_bin, "opencli-custom")
        self.assertEqual(settings.twitter_bin, "twitter")
        self.assertEqual(settings.xhs_bin, "xhs")
        self.assertEqual(settings.bili_bin, "bili")
        self.assertEqual(settings.asr_model, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(settings.audio_dir, "video")
        self.assertEqual(settings.subprocess_timeout, 45)

    def test_from_env_qwen_sub_model_defaults_to_lead_model(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "QWEN_API_KEY": "sk-test",
            "QWEN_MODEL_NAME": "qwen3-max",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.qwen_model, "qwen3-max")
        self.assertEqual(settings.qwen_sub_model, "qwen3-max")
        self.assertEqual(settings.model, "qwen3-max")
        self.assertEqual(settings.sub_model, "qwen3-max")

    def test_from_env_qwen_sub_model_uses_dedicated_env_var(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "QWEN_API_KEY": "sk-test",
            "QWEN_MODEL_NAME": "qwen3-max",
            "QWEN_SUB_MODEL_NAME": "qwen3-flash # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.model, "qwen3-max")
        self.assertEqual(settings.qwen_sub_model, "qwen3-flash")
        self.assertEqual(settings.sub_model, "qwen3-flash")

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
                        "DEEPFIND_VIDEO_DIR=video",
                        "DEEPFIND_TOOL_TIMEOUT=30",
                    ]
                )
            )
            with patch.dict(os.environ, {"DEEPFIND_ENV_FILE": str(env_path)}, clear=True):
                settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-from-dotenv")
        self.assertEqual(settings.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_QWEN_BASE_URL)
        self.assertEqual(settings.qwen_api_key, "sk-from-dotenv")
        self.assertEqual(settings.qwen_model, DEFAULT_QWEN_MODEL)
        self.assertEqual(settings.qwen_base_url, DEFAULT_QWEN_BASE_URL)
        self.assertEqual(settings.mimo_api_key, "")
        self.assertEqual(settings.mimo_model, DEFAULT_MIMO_MODEL)
        self.assertEqual(settings.mimo_base_url, DEFAULT_MIMO_BASE_URL)
        self.assertEqual(settings.minimax_api_key, "")
        self.assertEqual(settings.minimax_model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.minimax_base_url, DEFAULT_MINIMAX_BASE_URL)
        self.assertEqual(settings.nano_banana_api_key, "nb-key")
        self.assertEqual(settings.nano_banana_model, DEFAULT_IMAGE_MODEL)
        self.assertEqual(settings.image_dir, DEFAULT_IMAGE_DIR)
        self.assertEqual(settings.image_size, DEFAULT_IMAGE_SIZE)
        self.assertEqual(settings.opencli_bin, "opencli-dotenv")
        self.assertEqual(settings.bili_bin, "bili")
        self.assertEqual(settings.audio_dir, "video")
        self.assertEqual(settings.subprocess_timeout, 30)

    def test_from_env_falls_back_to_mimo_when_qwen_is_missing(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "MIMO_API_KEY": "sk-mimo",
            "MIMO_MODEL_NAME": "mimo-v2.5-pro # comment",
            "MIMO_BASE_URL": f"{DEFAULT_MIMO_BASE_URL} # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-mimo")
        self.assertEqual(settings.model, DEFAULT_MIMO_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_MIMO_BASE_URL)
        self.assertEqual(settings.qwen_api_key, "")
        self.assertEqual(settings.mimo_api_key, "sk-mimo")
        self.assertEqual(settings.mimo_model, DEFAULT_MIMO_MODEL)
        self.assertEqual(settings.mimo_base_url, DEFAULT_MIMO_BASE_URL)

    def test_from_env_falls_back_to_minimax_when_qwen_and_mimo_are_missing(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "MINIMAX_API_KEY": "sk-minimax",
            "MINIMAX_MODEL_NAME": "MiniMax-M2.7 # comment",
            "MINIMAX_BASE_URL": f"{DEFAULT_MINIMAX_BASE_URL} # comment",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-minimax")
        self.assertEqual(settings.model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_MINIMAX_BASE_URL)
        self.assertEqual(settings.qwen_api_key, "")
        self.assertEqual(settings.mimo_api_key, "")
        self.assertEqual(settings.minimax_api_key, "sk-minimax")
        self.assertEqual(settings.minimax_model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.minimax_base_url, DEFAULT_MINIMAX_BASE_URL)

    def test_from_env_defaults_to_deepseek_and_uses_responses_api(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "DEEPSEEK_MODEL_NAME": "deepseek-v4-flash",
            "DEEPSEEK_SUB_MODEL_NAME": "deepseek-v4-flash",
            "DEEPSEEK_BASE_URL": DEFAULT_DEEPSEEK_BASE_URL,
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-deepseek")
        self.assertEqual(settings.model, DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(settings.sub_model, DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_DEEPSEEK_BASE_URL)
        self.assertEqual(settings.api_mode, "responses")

    def test_from_env_prefers_deepseek_when_qwen_is_also_configured(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "QWEN_API_KEY": "sk-qwen",
            "DEEPSEEK_API_KEY": "sk-deepseek",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-deepseek")
        self.assertEqual(settings.model, DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_DEEPSEEK_BASE_URL)
        self.assertEqual(settings.api_mode, "responses")
        self.assertEqual(settings.qwen_api_key, "sk-qwen")

    def test_from_env_falls_back_to_qwen_when_deepseek_is_missing(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "QWEN_API_KEY": "sk-qwen",
            "QWEN_MODEL_NAME": "qwen3-max",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, "sk-qwen")
        self.assertEqual(settings.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_QWEN_BASE_URL)
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_from_env_without_keys_keeps_deepseek_as_default(self) -> None:
        with patch.dict(
            os.environ,
            {"DEEPFIND_ENV_FILE": str(self.empty_env_file)},
            clear=True,
        ):
            settings = Settings.from_env(require_api_key=False)
        self.assertEqual(settings.api_key, "")
        self.assertEqual(settings.model, DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(settings.base_url, DEFAULT_DEEPSEEK_BASE_URL)
        self.assertEqual(settings.api_mode, "responses")

    def test_from_env_finds_dotenv_above_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / ".env").write_text(
                "DEEPSEEK_API_KEY=sk-from-parent\nDEEPSEEK_MODEL_NAME=deepseek-v4-flash\n"
            )
            nested = project / "web" / "electron"
            nested.mkdir(parents=True)

            previous_cwd = Path.cwd()
            os.chdir(nested)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    settings = Settings.from_env()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(settings.deepseek_api_key, "sk-from-parent")
        self.assertEqual(settings.api_key, "sk-from-parent")

    def test_missing_env_file_falls_back_to_the_project_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / ".env").write_text("DEEPSEEK_API_KEY=sk-fallback\n")

            missing = project / "user-data" / ".env"
            previous_cwd = Path.cwd()
            os.chdir(project)
            try:
                with patch.dict(
                    os.environ,
                    {"DEEPFIND_ENV_FILE": str(missing)},
                    clear=True,
                ):
                    settings = Settings.from_env()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(settings.deepseek_api_key, "sk-fallback")

    def test_dotenv_fills_blank_environment_variables(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": str(self.empty_env_file),
            "DEEPSEEK_API_KEY": "",
        }
        self.empty_env_file.write_text("DEEPSEEK_API_KEY=sk-from-dotenv\n")
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.deepseek_api_key, "sk-from-dotenv")
