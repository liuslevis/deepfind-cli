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
from deepfind.tools import Toolset
from deepfind.youtube_audio_transcribe import YouTubeDownloadError
from deepfind.web_fetch import WEB_FETCH_MAX_MARKDOWN_CHARS


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


class FakePdfPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


def build_minimal_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT\n/F1 24 Tf\n72 100 Td\n({escaped_text}) Tj\nET\n"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(body.encode("latin-1"))
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\n".encode("latin-1"))
    pdf.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return bytes(pdf)


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
        self.assertIn("youtube_transcribe_full", names)
        self.assertEqual(names.count("youtube_transcribe"), 1)
        self.assertEqual(names.count("youtube_transcribe_full"), 1)
        self.assertIn("gen_img", names)
        self.assertIn("gen_slides", names)

    def test_bili_transcribe_spec_requires_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = next(item for item in toolset.specs() if item["function"]["name"] == "bili_transcribe")
        self.assertEqual(spec["function"]["parameters"]["required"], ["bili_id", "query"])
        self.assertIn("query", spec["function"]["parameters"]["properties"])

    def test_youtube_transcribe_spec_requires_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = next(item for item in toolset.specs() if item["function"]["name"] == "youtube_transcribe")
        self.assertEqual(spec["function"]["parameters"]["required"], ["url", "query"])
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
        toolset = Toolset(Settings(api_key="x", model="configured-web-model"))
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
        self.assertEqual(calls[0]["model"], "configured-web-model")
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

    def test_web_fetch_supports_pdf_content(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 fake body",
            request=httpx.Request("GET", "https://example.com/report.pdf"),
        )
        fake_http = FakeHttpClient(response=response)
        fake_client = FakeOpenAIClient([message_response("PDF summary")])
        fake_pdf = SimpleNamespace(
            metadata=SimpleNamespace(title="Quarterly Report"),
            pages=[FakePdfPage("Revenue increased"), FakePdfPage("Margins improved")],
        )

        with patch("deepfind.web_fetch.httpx.Client", return_value=fake_http) as client_cls:
            with patch("deepfind.web_fetch.PdfReader", return_value=fake_pdf) as pdf_reader:
                with patch.object(Settings, "new_client", return_value=fake_client):
                    result = toolset.web_fetch("https://example.com/report.pdf", "Summarize")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content_type"], "application/pdf")
        self.assertEqual(result["data"]["title"], "Quarterly Report")
        self.assertEqual(result["data"]["summary"], "PDF summary")
        self.assertEqual(pdf_reader.call_count, 1)
        accept = client_cls.call_args.kwargs["headers"]["Accept"]
        self.assertIn("application/pdf", accept)
        user_prompt = fake_client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn("Quarterly Report", user_prompt)
        self.assertIn("Revenue increased", user_prompt)
        self.assertIn("Margins improved", user_prompt)

    def test_web_fetch_sniffs_pdf_from_octet_stream(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"%PDF-1.6 fake body",
            request=httpx.Request("GET", "https://example.com/download?id=1"),
        )
        fake_pdf = SimpleNamespace(
            metadata=SimpleNamespace(title="Binary PDF"),
            pages=[FakePdfPage("Detected via magic header")],
        )

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(response=response)):
            with patch("deepfind.web_fetch.PdfReader", return_value=fake_pdf):
                with patch.object(
                    Settings,
                    "new_client",
                    return_value=FakeOpenAIClient([message_response("Binary summary")]),
                ):
                    result = toolset.web_fetch("https://example.com/download?id=1", "Summarize")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content_type"], "application/pdf")
        self.assertEqual(result["data"]["summary"], "Binary summary")

    def test_web_fetch_extracts_text_from_real_pdf_bytes(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        response = httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=build_minimal_pdf("Hello PDF"),
            request=httpx.Request("GET", "https://example.com/real.pdf"),
        )

        with patch("deepfind.web_fetch.httpx.Client", return_value=FakeHttpClient(response=response)):
            with patch.object(
                Settings,
                "new_client",
                return_value=FakeOpenAIClient([message_response("Real PDF summary")]),
            ) as new_client:
                result = toolset.web_fetch("https://example.com/real.pdf", "Summarize")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content_type"], "application/pdf")
        self.assertEqual(result["data"]["summary"], "Real PDF summary")
        user_prompt = new_client.return_value.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn("Hello PDF", user_prompt)

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
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout='{"items":[{"id":"note-1","title":"hello"}],"has_more":false}',
                    stderr="",
                ),
            ):
                result = toolset.xhs_search("topic", page=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["items"], [{"id": "note-1", "title": "hello"}])
        self.assertFalse(result["data"]["has_more"])

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

    def test_xhs_search_accepts_enveloped_json_output(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        page_payload = SimpleNamespace(
            returncode=0,
            stdout=(
                '{"ok":true,"schema_version":"1","data":{"items":[{"id":"note-1"},{"id":"note-2"}],"has_more":true}}'
            ),
            stderr="",
        )
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch("deepfind.tools.subprocess.run", return_value=page_payload):
                result = toolset.xhs_search("keyword", page=1, sort="latest")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["items"], [{"id": "note-1"}, {"id": "note-2"}])
        self.assertTrue(result["data"]["has_more"])

    def test_xhs_read_normalizes_note_for_model_consumption(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        payload = SimpleNamespace(
            returncode=0,
            stdout=(
                '{"ok":true,"schema_version":"1","data":{"items":[{"id":"69db389a000000002103a532","note_card":'
                '{"note_id":"69db389a000000002103a532","title":"美伊谈判破裂，一场没有赢家的博弈",'
                '"desc":"#国际新闻[话题]# #中东局势[话题]# #伊朗[话题]# #美伊谈判[话题]#",'
                '"type":"video","time":1775974554000,"last_update_time":1775974554000,'
                '"tag_list":[{"name":"国际新闻"},{"name":"中东局势"},{"name":"伊朗"},{"name":"美伊谈判"}],'
                '"user":{"nickname":"眀洞四方","user_id":"667a7216000000000b0320c0"},'
                '"interact_info":{"liked_count":"14","collected_count":"3","comment_count":"1","share_count":""},'
                '"video":{"capa":{"duration":213}},"image_list":[{}]}}]}}'
            ),
            stderr="",
        )
        with patch("deepfind.tools.shutil.which", return_value="/usr/bin/xhs"):
            with patch("deepfind.tools.subprocess.run", return_value=payload):
                result = toolset.xhs_read("69db389a000000002103a532")
        self.assertTrue(result["ok"])
        note = result["data"]["note"]
        self.assertEqual(note["id"], "69db389a000000002103a532")
        self.assertEqual(note["title"], "美伊谈判破裂，一场没有赢家的博弈")
        self.assertEqual(note["author"], "眀洞四方")
        self.assertEqual(note["media_type"], "video")
        self.assertEqual(note["text_mode"], "tags_only")
        self.assertEqual(note["video_duration_sec"], 213)
        self.assertEqual(note["stats"], {"likes": 14, "collects": 3, "comments": 1, "shares": 0})
        self.assertEqual(note["url"], "https://www.xiaohongshu.com/explore/69db389a000000002103a532")
        self.assertIn("the main substance may be in the spoken audio", note["content_hint"])
        self.assertIn("Title: 美伊谈判破裂，一场没有赢家的博弈", note["content_text"])
        self.assertIn("Media type: video", note["content_text"])

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
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", model="configured-bili-model", audio_dir=tmpdir))
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
                        "summarize coverage, gross margin, and profitability",
                    )
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_transcribe")
        self.assertEqual(result["data"]["bili_id"], "BV1cgPSzeEj5")
        self.assertEqual(result["data"]["summary"], "condensed summary")
        self.assertEqual(result["data"]["transcript"], "condensed summary")
        self.assertEqual(result["data"]["summary_model"], "configured-bili-model")
        self.assertEqual(fake_client.chat.completions.calls[0]["model"], "configured-bili-model")

    def test_bili_transcribe_uses_cache_for_same_bili_id_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            first_client = FakeOpenAIClient([message_response("cached summary body")])
            with patch(
                "deepfind.tools.transcribe_bili_audio",
                return_value={
                    "bili_id": "BV1cgPSzeEj5",
                    "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                    "transcript": "line one\nline two",
                },
            ) as first_transcribe_mock:
                with patch.object(Settings, "new_client", return_value=first_client):
                    first = toolset.bili_transcribe("BV1cgPSzeEj5", "summarize this video")
            second_client = FakeOpenAIClient([message_response("should not be used")])
            with patch("deepfind.tools.transcribe_bili_audio") as second_transcribe_mock:
                with patch.object(Settings, "new_client", return_value=second_client):
                    second = toolset.bili_transcribe(
                        "https://www.bilibili.com/video/BV1cgPSzeEj5",
                        "  summarize this video  ",
                    )
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(first["data"]["summary"], "cached summary body")
        self.assertEqual(second["data"]["summary"], "cached summary body")
        self.assertEqual(first["data"]["query"], "summarize this video")
        self.assertEqual(second["data"]["query"], "summarize this video")
        first_transcribe_mock.assert_called_once()
        second_transcribe_mock.assert_not_called()
        self.assertEqual(len(first_client.chat.completions.calls), 1)
        self.assertEqual(second_client.chat.completions.calls, [])

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
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
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
                    result = toolset.bili_transcribe("BV1cgPSzeEj5", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "summary_failed")

    def test_youtube_transcribe_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", model="configured-yt-model", audio_dir=tmpdir))
            fake_client = FakeOpenAIClient([message_response("condensed summary")])
            with patch(
                "deepfind.tools.transcribe_youtube_audio",
                return_value={
                    "youtube_id": "dQw4w9WgXcQ",
                    "transcript_path": "/tmp/audio/transcripts/youtube_audio/dQw4w9WgXcQ.txt",
                    "transcript": "line one\nline two",
                },
            ):
                with patch.object(Settings, "new_client", return_value=fake_client):
                    result = toolset.youtube_transcribe(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "summarize coverage, gross margin, and profitability",
                    )
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "youtube_transcribe")
        self.assertEqual(result["data"]["youtube_id"], "dQw4w9WgXcQ")
        self.assertEqual(result["data"]["summary"], "condensed summary")
        self.assertEqual(result["data"]["transcript"], "condensed summary")
        self.assertEqual(result["data"]["summary_model"], "configured-yt-model")
        self.assertEqual(fake_client.chat.completions.calls[0]["model"], "configured-yt-model")

    def test_youtube_transcribe_uses_cache_for_same_youtube_id_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            first_client = FakeOpenAIClient([message_response("cached summary body")])
            with patch(
                "deepfind.tools.transcribe_youtube_audio",
                return_value={
                    "youtube_id": "dQw4w9WgXcQ",
                    "transcript_path": "/tmp/audio/transcripts/youtube_audio/dQw4w9WgXcQ.txt",
                    "transcript": "line one\nline two",
                },
            ) as first_transcribe_mock:
                with patch.object(Settings, "new_client", return_value=first_client):
                    first = toolset.youtube_transcribe("dQw4w9WgXcQ", "summarize this video")
            second_client = FakeOpenAIClient([message_response("should not be used")])
            with patch("deepfind.tools.transcribe_youtube_audio") as second_transcribe_mock:
                with patch.object(Settings, "new_client", return_value=second_client):
                    second = toolset.youtube_transcribe(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "  summarize this video  ",
                    )
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(first["data"]["summary"], "cached summary body")
        self.assertEqual(second["data"]["summary"], "cached summary body")
        self.assertEqual(first["data"]["query"], "summarize this video")
        self.assertEqual(second["data"]["query"], "summarize this video")
        first_transcribe_mock.assert_called_once()
        second_transcribe_mock.assert_not_called()
        self.assertEqual(len(first_client.chat.completions.calls), 1)
        self.assertEqual(second_client.chat.completions.calls, [])

    def test_youtube_transcribe_full_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_youtube_audio",
            return_value={
                "youtube_id": "dQw4w9WgXcQ",
                "transcript_path": "/tmp/audio/transcripts/youtube_audio/dQw4w9WgXcQ.txt",
                "transcript": "line one",
            },
        ):
            result = toolset.youtube_transcribe_full("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "youtube_transcribe_full")
        self.assertEqual(result["data"]["transcript"], "line one")

    def test_youtube_transcribe_invalid_youtube_id_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        result = toolset.youtube_transcribe("not a video", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_youtube_id")

    def test_youtube_transcribe_missing_dependency_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_youtube_audio",
            side_effect=MissingDependencyError("missing"),
        ):
            result = toolset.youtube_transcribe("dQw4w9WgXcQ", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_dependency")

    def test_youtube_transcribe_download_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_youtube_audio",
            side_effect=YouTubeDownloadError("failed"),
        ):
            result = toolset.youtube_transcribe("dQw4w9WgXcQ", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "download_failed")

    def test_youtube_transcribe_transcription_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_youtube_audio",
            side_effect=TranscriptionError("failed"),
        ):
            result = toolset.youtube_transcribe("dQw4w9WgXcQ", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "transcription_failed")

    def test_youtube_transcribe_rejects_empty_query(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        result = toolset.youtube_transcribe("dQw4w9WgXcQ", "   ")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_query")

    def test_youtube_transcribe_summary_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolset = Toolset(Settings(api_key="x", audio_dir=tmpdir))
            fake_client = FakeOpenAIClient([message_response("")])
            with patch(
                "deepfind.tools.transcribe_youtube_audio",
                return_value={
                    "youtube_id": "dQw4w9WgXcQ",
                    "transcript_path": "/tmp/audio/transcripts/youtube_audio/dQw4w9WgXcQ.txt",
                    "transcript": "line one",
                },
            ):
                with patch.object(Settings, "new_client", return_value=fake_client):
                    result = toolset.youtube_transcribe("dQw4w9WgXcQ", "summarize this video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "summary_failed")

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
