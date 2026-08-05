from __future__ import annotations

import unittest
from types import SimpleNamespace

from deepfind.llm_transport import complete_text


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="DeepSeek response")


class LlmTransportTests(unittest.TestCase):
    def test_complete_text_uses_responses_api(self) -> None:
        responses = FakeResponsesAPI()
        client = SimpleNamespace(responses=responses)

        text = complete_text(
            client,
            api_mode="responses",
            model="deepseek-v4-flash",
            instructions="Be concise.",
            user_input="Hello",
            max_output_tokens=500,
        )

        self.assertEqual(text, "DeepSeek response")
        self.assertEqual(
            responses.calls,
            [
                {
                    "model": "deepseek-v4-flash",
                    "instructions": "Be concise.",
                    "input": "Hello",
                    "max_output_tokens": 500,
                }
            ],
        )
