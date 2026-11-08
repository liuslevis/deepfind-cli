from __future__ import annotations

import unittest
from types import SimpleNamespace

from deepfind.llm_transport import (
    TruncatedCompletionError,
    complete_text,
    response_output_text,
)


class FakeResponsesAPI:
    def __init__(self, response=None) -> None:
        self.calls = []
        self._response = response or SimpleNamespace(output_text="DeepSeek response")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _reasoning_only_response():
    return SimpleNamespace(
        output_text="",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[SimpleNamespace(type="reasoning_text", text="We need to decide...")],
            )
        ],
    )


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

    def test_response_output_text_ignores_reasoning_items(self) -> None:
        self.assertEqual(response_output_text(_reasoning_only_response()), "")

    def test_response_output_text_reads_message_items_without_output_text(self) -> None:
        response = SimpleNamespace(
            output_text="",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    content=[SimpleNamespace(type="reasoning_text", text="hidden thinking")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="visible answer")],
                ),
            ],
        )
        self.assertEqual(response_output_text(response), "visible answer")

    def test_complete_text_raises_when_response_is_truncated(self) -> None:
        client = SimpleNamespace(responses=FakeResponsesAPI(_reasoning_only_response()))
        with self.assertRaises(TruncatedCompletionError):
            complete_text(
                client,
                api_mode="responses",
                model="deepseek-v4-flash",
                instructions="Be concise.",
                user_input="Hello",
                max_output_tokens=500,
            )
