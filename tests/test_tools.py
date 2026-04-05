from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from deepfind.bili_transcribe import (
    BiliDownloadError,
    InvalidBiliIdError,
    MissingDependencyError,
    TranscriptionError,
)
from deepfind.config import Settings
from deepfind.gen_slides import SlideGenerationError
from deepfind.gen_img import ImageGenerationError, MissingImageApiKeyError
from deepfind.transcript_summary import BILI_TRANSCRIPT_SUMMARY_MODEL
from deepfind.tools import Toolset
from deepfind.web_fetch import WEB_FETCH_MAX_MARKDOWN_CHARS, WEB_FETCH_MODEL


def message_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=[],
                )
            )
        ]
    )


class FakeChatCompletionsAPI:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.items.pop(0)


class FakeOpenAIClient:
    def __init__(self, items):
        self.chat = SimpleNamespace(completions=FakeChatCompletionsAPI(items))


class FakeHttpClient:
    def __init__(self, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        self.calls.append(url)
        if self.exc is not None:
            raise self.exc
        return self.response


class ToolsetTests(unittest.TestCase):
    def test_specs_use_chat_completion_function_shape(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = toolset.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "web_search")

    def test_specs_include_bilibili_and_media_tools(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        names = [item["function"]["name"] for item in toolset.specs()]
        self.assertIn("web_search", names)
        self.assertIn("web_fetch", names)
        self.assertIn("arxiv_search", names)
        self.assertIn("x_search", names)
        self.assertIn("zhihu_search", names)
        self.assertIn("boss_search", names)
        self.assertIn("boss_detail", names)
        self.assertIn("boss_chatlist", names)
        self.assertIn("boss_send", names)
        self.assertIn("bili_search", names)
        self.assertIn("bili_get_user_videos", names)
        self.assertIn("bili_transcribe", names)
        self.assertIn("bili_transcribe_full", names)
        self.assertIn("youtube_transcribe", names)
        self.assertIn("gen_img", names)
        self.assertIn("gen_slides", names)

    def test_bili_transcribe_spec_requires_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = next(item for item in toolset.specs() if item["function"]["name"] == "bili_transcribe")
        self.assertEqual(spec["function"]["parameters"]["required"], ["bili_id", "query"])
        self.assertIn("query", spec["function"]["parameters"]["properties"])

    def test_web_search_missing_binary_returns_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value=None):
            result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "web_search")
        self.assertEqual(result["error_code"], "missing_dependency")
        self.assertEqual(result["error"], "opencli not found")

    def test_web_fetch_html_returns_summary_and_metadata(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>Example Title</title></head>"
                "<body><nav>skip</nav><main><h1>Hello</h1><p>World</p></main></body></html>"
            ),
            request=httpx.Request("GET", "https://example.com/final"),
        )
        fake_http = FakeHttpClient(response=response)
        fake_client = FakeOpenAIClient([message_response("Focused summary")])

        with patch("deepfind.web_fetch.httpx.Client", return_value=fake_http):
            with patch.object(Settings, "new_client", return_value=fake_client):
                result = toolset.web_fetch("https://example.com/start", "Summarize the page")

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "web_fetch")
        self.assertEqual(result["data"]["url"], "https://example.com/start")
        self.assertEqual(result["data"]["final_url"], "https://example.com/final")
        self.assertEqual(result["data"]["title"], "Example Title")
        self.assertEqual(result["data"]["summary"], "Focused summary")
        self.assertEqual(result["data"]["content_type"], "text/html")
        self.assertFalse(result["data"]["truncated"])
        self.assertGreater(result["data"]["markdown_chars"], 0)
        calls = fake_client.chat.completions.calls
        self.assertEqual(calls[0]["model"], WEB_FETCH_MODEL)
        self.assertIn("Example Title", calls[0]["messages"][1]["content"])
        self.assertEqual(fake_http.calls, ["https://example.com/start"])

    def test_web_fetch_truncates_long_content_before_summary(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        long_text = ("line " * (WEB_FETCH_MAX_MARKDOWN_CHARS // 2)) + "TAIL_MARKER"
        response = httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=long_text,
            request=httpx.Request("GET", "https://example.com/long"),
        )
        fake_client = FakeOpenAIClient([message_response("Short summary")])

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(response=response)):
            with patch.object(Settings, "new_client", return_value=fake_client):
                result = toolset.web_fetch("https://example.com/long", "Extract the main idea")

        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["truncated"])
        self.assertGreater(result["data"]["markdown_chars"], WEB_FETCH_MAX_MARKDOWN_CHARS)
        user_prompt = fake_client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertNotIn("TAIL_MARKER", user_prompt)

    def test_web_fetch_supports_plain_text_content(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="alpha\nbeta\n",
            request=httpx.Request("GET", "https://example.com/plain"),
        )

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(response=response)):
            with patch.object(
                Settings,
                "new_client",
                return_value=FakeOpenAIClient([message_response("Plain summary")]),
            ):
                result = toolset.web_fetch("https://example.com/plain", "Summarize")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content_type"], "text/plain")
        self.assertEqual(result["data"]["title"], "")
        self.assertEqual(result["data"]["summary"], "Plain summary")

    def test_web_fetch_returns_http_error_for_bad_status(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            404,
            headers={"content-type": "text/html"},
            text="missing",
            request=httpx.Request("GET", "https://example.com/missing"),
        )

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(response=response)):
            result = toolset.web_fetch("https://example.com/missing", "Summarize")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "http_error")

    def test_web_fetch_returns_timeout_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        timeout = httpx.ReadTimeout("slow", request=httpx.Request("GET", "https://example.com/slow"))

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(exc=timeout)):
            result = toolset.web_fetch("https://example.com/slow", "Summarize")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "timeout")

    def test_web_search_success_uses_limit_when_supported(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"google/search","args":[{"name":"query"},{"name":"limit"}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"item"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.web_search("google", "topic", limit=5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "web_search")
        self.assertEqual(result["engine"], "google")
        self.assertEqual(result["query"], "topic")
        self.assertEqual(result["data"], [{"title": "item"}])
        self.assertEqual(
            run_mock.call_args_list[0][0][0],
            ["/usr/bin/opencli", "list", "-f", "json"],
        )
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "google", "search", "topic", "--limit", "5", "-f", "json"],
        )

    def test_web_search_omits_limit_when_registry_does_not_advertise_it(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"bing/search","args":[{"name":"query"}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"item"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.web_search("bing", "topic", limit=7)
        self.assertTrue(result["ok"])
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "bing", "search", "topic", "-f", "json"],
        )

    def test_web_search_returns_unsupported_engine_when_missing_from_registry(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout='[{"command":"bing/search","args":[{"name":"query"}]}]',
                        stderr="",
                    ),
                ):
                    result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "unsupported_engine")

    def test_web_search_returns_registry_error_when_opencli_list_fails(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    return_value=SimpleNamespace(returncode=1, stdout="", stderr="registry failed"),
                ):
                    result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "registry_failed")

    def test_web_search_returns_invalid_registry_when_opencli_list_is_not_json(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="not json", stderr=""),
                ):
                    result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_registry")

    def test_web_search_returns_command_failed_on_non_zero_exit(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"google/search","args":[{"name":"query"},{"name":"limit"}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=1, stdout="", stderr="search failed"),
                    ],
                ):
                    result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "command_failed")

    def test_web_search_returns_invalid_json_when_search_output_is_not_json(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"baidu/search","args":[{"name":"query"},{"name":"limit"}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout="not json", stderr=""),
                    ],
                ):
                    result = toolset.web_search("baidu", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_json")

    def test_web_search_supports_js_opencli_path(self) -> None:
        toolset = Toolset(Settings(api_key="x", opencli_bin="/tmp/opencli/dist/main.js"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.Path.exists", return_value=True):
                with patch("deepfind.tools.shutil.which", side_effect=["/usr/bin/node"]):
                    with patch(
                        "deepfind.tools.subprocess.run",
                        side_effect=[
                            SimpleNamespace(
                                returncode=0,
                                stdout='[{"command":"bing/search","args":[{"name":"query"},{"name":"limit"}]}]',
                                stderr="",
                            ),
                            SimpleNamespace(returncode=0, stdout='[{"title":"item"}]', stderr=""),
                        ],
                    ) as run_mock:
                        result = toolset.web_search("bing", "topic", limit=3)
        self.assertTrue(result["ok"])
        self.assertEqual(
            run_mock.call_args_list[0][0][0],
            ["/usr/bin/node", str(Path("/tmp/opencli/dist/main.js")), "list", "-f", "json"],
        )
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/node",
                str(Path("/tmp/opencli/dist/main.js")),
                "bing",
                "search",
                "topic",
                "--limit",
                "3",
                "-f",
                "json",
            ],
        )

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

    def test_x_search_uses_same_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/twitter"):
            with patch(
                "deepfind.tools.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout='{"items":[]}', stderr=""),
            ) as run_mock:
                result = toolset.x_search("robotics", max_results=5, tab="Latest")
        command = run_mock.call_args.kwargs["args"] if "args" in run_mock.call_args.kwargs else run_mock.call_args[0][0]
        self.assertEqual(
            command,
            ["/usr/bin/twitter", "search", "--max", "5", "robotics", "--json"],
        )
        self.assertEqual(result["tool"], "x_search")

    def test_zhihu_search_uses_opencli_query_flag(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"zhihu/search","args":[{"name":"query","positional":false},{"name":"limit","positional":false}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"item"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.zhihu_search("AI", limit=4)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "zhihu_search")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "zhihu", "search", "--query", "AI", "--limit", "4", "-f", "json"],
        )

    def test_arxiv_search_uses_positional_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"arxiv/search","args":[{"name":"query","positional":true},{"name":"limit","positional":false}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"Attention Is All You Need"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.arxiv_search("transformer", limit=4)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "arxiv_search")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "arxiv", "search", "transformer", "--limit", "4", "-f", "json"],
        )

    def test_boss_search_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                '[{"command":"boss/search","args":['
                                '{"name":"query","positional":true},'
                                '{"name":"city","positional":false},'
                                '{"name":"experience","positional":false},'
                                '{"name":"degree","positional":false},'
                                '{"name":"salary","positional":false},'
                                '{"name":"industry","positional":false},'
                                '{"name":"page","positional":false},'
                                '{"name":"limit","positional":false}'
                                ']}]'
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"name":"Agent Engineer"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.boss_search("Agent", city="Shanghai", salary="30-50K", limit=5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "boss_search")
        self.assertEqual(result["query"], "Agent")
        self.assertEqual(result["city"], "Shanghai")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/opencli",
                "boss",
                "search",
                "Agent",
                "--city",
                "Shanghai",
                "--salary",
                "30-50K",
                "--page",
                "1",
                "--limit",
                "5",
                "-f",
                "json",
            ],
        )

    def test_boss_detail_uses_security_id_flag(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='[{"command":"boss/detail","args":[{"name":"security-id","positional":false}]}]',
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='{"name":"Agent Engineer"}', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.boss_detail("sec-123")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "boss_detail")
        self.assertEqual(result["security_id"], "sec-123")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "boss", "detail", "--security-id", "sec-123", "-f", "json"],
        )

    def test_boss_chatlist_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                '[{"command":"boss/chatlist","args":['
                                '{"name":"page","positional":false},'
                                '{"name":"limit","positional":false},'
                                '{"name":"job-id","positional":false}'
                                ']}]'
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"uid":"user-1"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.boss_chatlist(page=2, limit=10, job_id="job-123")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "boss_chatlist")
        self.assertEqual(result["job_id"], "job-123")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/opencli",
                "boss",
                "chatlist",
                "--page",
                "2",
                "--limit",
                "10",
                "--job-id",
                "job-123",
                "-f",
                "json",
            ],
        )

    def test_boss_send_uses_uid_and_positional_text(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                '[{"command":"boss/send","args":['
                                '{"name":"uid","positional":false},'
                                '{"name":"text","positional":true}'
                                ']}]'
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='{"status":"sent"}', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.boss_send("user-1", "Which company is this job with?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "boss_send")
        self.assertEqual(result["uid"], "user-1")
        self.assertEqual(result["text"], "Which company is this job with?")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/opencli", "boss", "send", "--uid", "user-1", "Which company is this job with?", "-f", "json"],
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

    def test_bili_search_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                '[{"command":"bilibili/search","args":['
                                '{"name":"query","positional":true},'
                                '{"name":"type","positional":false},'
                                '{"name":"page","positional":false},'
                                '{"name":"limit","positional":false}'
                                ']}]'
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"video"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.bili_search("deep research", search_type="user", page=2, limit=5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_search")
        self.assertEqual(result["query"], "deep research")
        self.assertEqual(result["search_type"], "user")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/opencli",
                "bilibili",
                "search",
                "deep research",
                "--type",
                "user",
                "--page",
                "2",
                "--limit",
                "5",
                "-f",
                "json",
            ],
        )

    def test_bili_get_user_videos_uses_supported_flags(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
            with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                with patch(
                    "deepfind.tools.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout=(
                                '[{"command":"bilibili/user-videos","args":['
                                '{"name":"uid","positional":false},'
                                '{"name":"limit","positional":false},'
                                '{"name":"order","positional":false},'
                                '{"name":"page","positional":false}'
                                ']}]'
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(returncode=0, stdout='[{"title":"upload"}]', stderr=""),
                    ],
                ) as run_mock:
                    result = toolset.bili_get_user_videos("946974", order="click", page=3, limit=8)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_get_user_videos")
        self.assertEqual(result["uid"], "946974")
        self.assertEqual(result["order"], "click")
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/opencli",
                "bilibili",
                "user-videos",
                "--uid",
                "946974",
                "--limit",
                "8",
                "--order",
                "click",
                "--page",
                "3",
                "-f",
                "json",
            ],
        )

    def test_bili_transcribe_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        fake_client = FakeOpenAIClient([message_response("condensed summary")])
        with patch(
            "deepfind.tools.transcribe_bili_audio",
            return_value={
                "bili_id": "BV1cgPSzeEj5",
                "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                "transcript": "line one\nline two",
            },
        ):
            with patch.object(Settings, "new_client", return_value=fake_client):
                result = toolset.bili_transcribe(
                    "https://www.bilibili.com/video/BV1cgPSzeEj5",
                    "总结覆盖户数、毛利率和盈利判断",
                )
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_transcribe")
        self.assertEqual(result["data"]["bili_id"], "BV1cgPSzeEj5")
        self.assertEqual(result["data"]["summary"], "condensed summary")
        self.assertEqual(result["data"]["transcript"], "condensed summary")
        self.assertEqual(result["data"]["summary_model"], BILI_TRANSCRIPT_SUMMARY_MODEL)
        self.assertEqual(fake_client.chat.completions.calls[0]["model"], BILI_TRANSCRIPT_SUMMARY_MODEL)

    def test_bili_transcribe_full_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_bili_audio",
            return_value={
                "bili_id": "BV1cgPSzeEj5",
                "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                "transcript": "line one",
            },
        ):
            result = toolset.bili_transcribe_full("https://www.bilibili.com/video/BV1cgPSzeEj5")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_transcribe_full")
        self.assertEqual(result["data"]["transcript"], "line one")

    def test_bili_transcribe_invalid_bili_id_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=InvalidBiliIdError("bad id")):
            result = toolset.bili_transcribe("not a video", "总结这个视频")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_bili_id")

    def test_bili_transcribe_missing_dependency_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=MissingDependencyError("missing")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5", "总结这个视频")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_dependency")

    def test_bili_transcribe_download_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=BiliDownloadError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5", "总结这个视频")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "download_failed")

    def test_bili_transcribe_transcription_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=TranscriptionError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5", "总结这个视频")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "transcription_failed")

    def test_bili_transcribe_rejects_empty_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        result = toolset.bili_transcribe("BV1cgPSzeEj5", "   ")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_query")

    def test_bili_transcribe_summary_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        fake_client = FakeOpenAIClient([message_response("")])
        with patch(
            "deepfind.tools.transcribe_bili_audio",
            return_value={
                "bili_id": "BV1cgPSzeEj5",
                "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                "transcript": "line one",
            },
        ):
            with patch.object(Settings, "new_client", return_value=fake_client):
                result = toolset.bili_transcribe("BV1cgPSzeEj5", "总结这个视频")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "summary_failed")

    def test_youtube_transcribe_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
                with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                    with patch(
                        "deepfind.tools.subprocess.run",
                        side_effect=[
                            SimpleNamespace(
                                returncode=0,
                                stdout='[{"command":"youtube/transcript","args":[{"name":"url","positional":true},{"name":"lang","positional":false},{"name":"mode","positional":false}]}]',
                                stderr="",
                            ),
                            SimpleNamespace(
                                returncode=0,
                                stdout='[{"timestamp":"0:01","speaker":"","text":"line one"},{"timestamp":"0:35","speaker":"Host","text":"line two"}]',
                                stderr="",
                            ),
                        ],
                    ) as run_mock:
                        result = toolset.youtube_transcribe("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertTrue(result["ok"])
            self.assertEqual(result["tool"], "youtube_transcribe")
            self.assertEqual(result["data"]["youtube_id"], "dQw4w9WgXcQ")
            self.assertEqual(result["data"]["transcript"], "[0:01] line one\n\n[0:35] Host: line two")
            self.assertEqual(
                run_mock.call_args_list[1][0][0],
                [
                    "/usr/bin/opencli",
                    "youtube",
                    "transcript",
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "--mode",
                    "grouped",
                    "-f",
                    "json",
                ],
            )
            transcript_path = Path(result["data"]["transcript_path"])
            self.assertTrue(transcript_path.exists())
            self.assertEqual(transcript_path.read_text(encoding="utf-8"), "[0:01] line one\n\n[0:35] Host: line two\n")

    def test_youtube_transcribe_uses_cached_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_dir = Path(tmpdir) / "transcripts" / "youtube"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            cached_path = transcript_dir / "dQw4w9WgXcQ.txt"
            cached_path.write_text("cached line\n", encoding="utf-8")

            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            with patch("deepfind.tools.subprocess.run") as run_mock:
                result = toolset.youtube_transcribe("dQw4w9WgXcQ")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["transcript"], "cached line")
        self.assertEqual(result["data"]["transcript_path"], str(cached_path))
        run_mock.assert_not_called()

    def test_youtube_transcribe_invalid_id_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        result = toolset.youtube_transcribe("not a video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_youtube_id")

    def test_youtube_transcribe_missing_dependency_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value=None):
            result = toolset.youtube_transcribe("dQw4w9WgXcQ")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_dependency")

    def test_youtube_transcribe_passes_lang_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            with patch.dict("deepfind.tools._OPENCLI_REGISTRY_CACHE", {}, clear=True):
                with patch("deepfind.tools.shutil.which", return_value="/usr/bin/opencli"):
                    with patch(
                        "deepfind.tools.subprocess.run",
                        side_effect=[
                            SimpleNamespace(
                                returncode=0,
                                stdout='[{"command":"youtube/transcript","args":[{"name":"url","positional":true},{"name":"lang","positional":false},{"name":"mode","positional":false}]}]',
                                stderr="",
                            ),
                            SimpleNamespace(
                                returncode=0,
                                stdout='[{"timestamp":"0:01","speaker":"","text":"line one"}]',
                                stderr="",
                            ),
                        ],
                    ) as run_mock:
                        result = toolset.youtube_transcribe("dQw4w9WgXcQ", lang="zh-Hans")
        self.assertTrue(result["ok"])
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            [
                "/usr/bin/opencli",
                "youtube",
                "transcript",
                "dQw4w9WgXcQ",
                "--lang",
                "zh-Hans",
                "--mode",
                "grouped",
                "-f",
                "json",
            ],
        )

    def test_gen_img_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(
                Settings(
                    api_key="x",
                    nano_banana_api_key="nb-key",
                    image_dir=tmpdir,
                )
            )
            fake_path = Path(tmpdir) / "image.png"
            with patch(
                "deepfind.tools.generate_image",
                return_value={
                    "prompt": "make art",
                    "model": "gemini-3.1-flash-image-preview",
                    "aspect_ratio": "1:1",
                    "image_size": "2K",
                    "mime_type": "image/png",
                    "image_path": str(fake_path),
                    "response_text": "done",
                },
            ):
                result = toolset.gen_img("make art")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "gen_img")
        self.assertEqual(result["data"]["image_path"], str(fake_path))

    def test_gen_img_missing_api_key_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.generate_image", side_effect=MissingImageApiKeyError("missing")):
            result = toolset.gen_img("make art")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_api_key")

    def test_gen_img_generation_error(self) -> None:
        toolset = Toolset(Settings(api_key="x", nano_banana_api_key="nb-key"))
        with patch("deepfind.tools.generate_image", side_effect=ImageGenerationError("failed")):
            result = toolset.gen_img("make art")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "generation_failed")

    def test_gen_slides_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "deck.html"
            with patch(
                "deepfind.tools.generate_slides",
                return_value={
                    "title": "AI Briefing",
                    "slide_count": 3,
                    "html_path": str(fake_path),
                },
            ):
                result = toolset.gen_slides("make deck", slide_count=3)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "gen_slides")
        self.assertEqual(result["data"]["html_path"], str(fake_path))
        self.assertEqual(result["data"]["slide_count"], 3)

    def test_gen_slides_generation_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.generate_slides", side_effect=SlideGenerationError("failed")):
            result = toolset.gen_slides("make deck")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "gen_slides")
        self.assertEqual(result["error_code"], "generation_failed")
