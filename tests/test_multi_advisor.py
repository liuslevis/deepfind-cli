from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "multi-advisor.py"
SPEC = importlib.util.spec_from_file_location("multi_advisor", MODULE_PATH)
multi_advisor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = multi_advisor
SPEC.loader.exec_module(multi_advisor)


class MultiAdvisorTests(unittest.TestCase):
    def test_load_advisor_settings_defaults_to_local_ollama(self) -> None:
        with patch.dict(os.environ, {"DEEPFIND_ENV_FILE": "/tmp/deepfind-missing.env"}, clear=True):
            settings = multi_advisor.load_advisor_settings()
        self.assertEqual(settings.model, "qwen2.5")
        self.assertEqual(settings.base_url, multi_advisor.OLLAMA_BASE_URL)
        self.assertEqual(settings.api_key, "ollama")

    def test_load_advisor_settings_uses_qwen_env(self) -> None:
        env = {
            "DEEPFIND_ENV_FILE": "/tmp/deepfind-missing.env",
            "QWEN_API_KEY": "sk-qwen",
            "QWEN_MODEL_NAME": "qwen3-max",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = multi_advisor.load_advisor_settings()
        self.assertEqual(settings.model, "qwen3-max")
        self.assertEqual(settings.base_url, multi_advisor.DEFAULT_BASE_URL)
        self.assertEqual(settings.api_key, "sk-qwen")

    def test_parse_targeted_input_handles_direct_and_broadcast_modes(self) -> None:
        self.assertEqual(multi_advisor.parse_targeted_input("@jobs 打磨首页"), ("jobs", "打磨首页"))
        self.assertEqual(multi_advisor.parse_targeted_input("@jobs"), ("jobs", ""))
        self.assertEqual(multi_advisor.parse_targeted_input("做一个 AI 工具"), (None, "做一个 AI 工具"))

    def test_extract_text_content_handles_list_payloads(self) -> None:
        content = [
            {"type": "text", "text": "第一段"},
            SimpleNamespace(text="第二段"),
        ]
        self.assertEqual(multi_advisor.extract_text_content(content), "第一段\n第二段")

    def test_build_messages_keeps_recent_history_only(self) -> None:
        history = [{"role": "assistant", "content": str(index)} for index in range(10)]
        with patch.object(multi_advisor, "HISTORY", history):
            messages = multi_advisor.build_messages("jobs", "下一步怎么做")
        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[1]["content"], "2")
        self.assertEqual(messages[-1]["content"], "下一步怎么做")

    def test_list_commands_include_list_alias(self) -> None:
        self.assertIn("list", multi_advisor.LIST_COMMANDS)
        self.assertIn("人格", multi_advisor.LIST_COMMANDS)


if __name__ == "__main__":
    unittest.main()
