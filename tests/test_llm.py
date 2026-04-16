from __future__ import annotations

import unittest
from types import SimpleNamespace

from deepfind.llm import ResponseAgent


def message_response(text: str, *, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=tool_calls,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class FakeChatCompletionsAPI:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.items.pop(0)


class FakeClient:
    def __init__(self, items):
        self.chat = SimpleNamespace(completions=FakeChatCompletionsAPI(items))


class FakeSettings:
    model = "qwen3-max"

    def __init__(self, items):
        self._client = FakeClient(items)

    def new_client(self):
        return self._client


class FakeTools:
    def __init__(self):
        self.invocations = []
        self.output = '{"ok":true}'

    def specs(self):
        return [{"type": "function", "function": {"name": "twitter_search"}}]

    def call(self, name, arguments):
        self.invocations.append((name, arguments))
        return self.output


class ResponseAgentTests(unittest.TestCase):
    def test_runs_function_tool_loop(self) -> None:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="twitter_search", arguments='{"query":"openai"}'),
        )
        settings = FakeSettings(
            [
                message_response("", tool_calls=[tool_call]),
                message_response('{"summary":"ok","facts":[],"gaps":[]}'),
            ]
        )
        tools = FakeTools()
        agent = ResponseAgent(settings=settings, tools=tools, max_iter=3)

        result = agent.run("worker", "short prompt", "q=test", use_tools=True)

        self.assertEqual(result.iterations, 2)
        self.assertEqual(tools.invocations, [("twitter_search", {"query": "openai"})])
        self.assertEqual(result.text, '{"summary":"ok","facts":[],"gaps":[]}')
        calls = settings.new_client().chat.completions.calls
        self.assertEqual(calls[0]["messages"][0]["content"], "short prompt")
        self.assertEqual(calls[0]["max_tokens"], 1400)
        self.assertEqual(calls[1]["messages"][2]["tool_calls"][0]["function"]["name"], "twitter_search")
        self.assertEqual(calls[1]["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"query":"openai"}')
        self.assertEqual(calls[1]["messages"][3]["role"], "tool")

    def test_normalizes_tool_arguments_before_replaying_tool_call(self) -> None:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="twitter_search", arguments='{"query":"openai"} trailing'),
        )
        settings = FakeSettings(
            [
                message_response("", tool_calls=[tool_call]),
                message_response('{"summary":"ok","facts":[],"gaps":[]}'),
            ]
        )
        tools = FakeTools()
        agent = ResponseAgent(settings=settings, tools=tools, max_iter=3)

        result = agent.run("worker", "short prompt", "q=test", use_tools=True)

        self.assertEqual(result.iterations, 2)
        self.assertEqual(tools.invocations, [("twitter_search", {"query": "openai"})])
        calls = settings.new_client().chat.completions.calls
        self.assertEqual(calls[1]["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"query":"openai"}')

    def test_returns_empty_output_marker(self) -> None:
        settings = FakeSettings([message_response("")])
        agent = ResponseAgent(settings=settings, tools=FakeTools(), max_iter=1)

        result = agent.run("lead", "prompt", "q=test", use_tools=False)

        self.assertIn("empty_output", result.text)

    def test_collects_citations_from_tool_outputs(self) -> None:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="twitter_search", arguments='{"query":"openai"}'),
        )
        settings = FakeSettings(
            [
                message_response("", tool_calls=[tool_call]),
                message_response('{"summary":"ok","facts":[],"gaps":[]}'),
            ]
        )
        tools = FakeTools()
        tools.output = (
            '{"ok":true,"tool":"web_search","data":[{"url":"https://example.com/article","title":"Example"}]}'
        )
        agent = ResponseAgent(settings=settings, tools=tools, max_iter=3)

        result = agent.run("worker", "short prompt", "q=test", use_tools=True)

        self.assertEqual(result.citations, ["https://example.com/article"])

    def test_filters_tools_when_tool_names_are_provided(self) -> None:
        settings = FakeSettings([message_response('{"summary":"ok","facts":[],"gaps":[]}')])
        agent = ResponseAgent(settings=settings, tools=FakeTools(), max_iter=1)

        agent.run(
            "lead",
            "prompt",
            "q=test",
            use_tools=True,
            tool_names=["missing_tool"],
        )

        calls = settings.new_client().chat.completions.calls
        self.assertNotIn("tools", calls[0])

    def test_includes_history_messages_before_latest_input(self) -> None:
        settings = FakeSettings([message_response('{"summary":"ok","facts":[],"gaps":[]}')])
        agent = ResponseAgent(settings=settings, tools=FakeTools(), max_iter=1)

        agent.run(
            "lead",
            "prompt",
            "q=current",
            use_tools=False,
            history=[
                {"role": "user", "content": "q=previous"},
                {"role": "assistant", "content": "a=previous"},
            ],
        )

        messages = settings.new_client().chat.completions.calls[0]["messages"]
        self.assertEqual(messages[1], {"role": "user", "content": "q=previous"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "a=previous"})
        self.assertEqual(messages[3], {"role": "user", "content": "q=current"})

    def test_respects_max_tokens_override(self) -> None:
        settings = FakeSettings([message_response('{"summary":"ok","facts":[],"gaps":[]}')])
        agent = ResponseAgent(settings=settings, tools=FakeTools(), max_iter=1)

        agent.run("lead", "prompt", "q=test", use_tools=False, max_tokens=3200)

        calls = settings.new_client().chat.completions.calls
        self.assertEqual(calls[0]["max_tokens"], 3200)
