from __future__ import annotations

import unittest

from deepfind.web_models import (
    DEFAULT_MODEL_TARGET,
    SendMessageRequest,
    TurnResult,
    WebMessage,
    normalize_model_target,
)


class WebModelTargetTests(unittest.TestCase):
    def test_default_model_target_is_deepseek(self) -> None:
        self.assertEqual(DEFAULT_MODEL_TARGET, "deepseek")

    def test_normalize_model_target_falls_back_to_deepseek(self) -> None:
        self.assertEqual(normalize_model_target(None), "deepseek")
        self.assertEqual(normalize_model_target(""), "deepseek")
        self.assertEqual(normalize_model_target("cloud"), "deepseek")

    def test_normalize_model_target_keeps_explicit_targets(self) -> None:
        for target in ("qwen", "mimo", "minimax", "glm", "deepseek", "gpu"):
            self.assertEqual(normalize_model_target(target), target)

    def test_normalize_model_target_rejects_unknown_targets(self) -> None:
        with self.assertRaises(ValueError):
            normalize_model_target("openai")

    def test_message_models_default_to_deepseek(self) -> None:
        message = WebMessage(
            id="msg_1",
            role="user",
            content="hello",
            created_at="2026-03-22T00:00:00+00:00",
        )
        turn_result = TurnResult(answer_markdown="answer", mode="fast")
        request = SendMessageRequest(content="hello", mode="fast")

        self.assertEqual(message.model_target, "deepseek")
        self.assertEqual(turn_result.model_target, "deepseek")
        self.assertEqual(request.model_target, "deepseek")


if __name__ == "__main__":
    unittest.main()
