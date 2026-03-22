from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class ToolsetTests(unittest.TestCase):
    def test_specs_use_chat_completion_function_shape(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        spec = toolset.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "web_search")

    def test_specs_include_bili_transcribe(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        names = [item["function"]["name"] for item in toolset.specs()]
        self.assertIn("web_search", names)
        self.assertIn("x_search", names)
        self.assertIn("zhihu_search", names)
        self.assertIn("bili_transcribe", names)
        self.assertIn("gen_img", names)
        self.assertIn("gen_slides", names)

    def test_web_search_missing_binary_returns_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.shutil.which", return_value=None):
            result = toolset.web_search("google", "topic")
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "web_search")
        self.assertEqual(result["error_code"], "missing_dependency")
        self.assertEqual(result["error"], "opencli not found")

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
            ["/usr/bin/node", "\\tmp\\opencli\\dist\\main.js", "list", "-f", "json"],
        )
        self.assertEqual(
            run_mock.call_args_list[1][0][0],
            ["/usr/bin/node", "\\tmp\\opencli\\dist\\main.js", "bing", "search", "topic", "--limit", "3", "-f", "json"],
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

    def test_bili_transcribe_success_payload(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch(
            "deepfind.tools.transcribe_bili_audio",
            return_value={
                "bili_id": "BV1cgPSzeEj5",
                "transcript_path": "/tmp/audio/transcripts/BV1cgPSzeEj5.txt",
                "transcript": "line one",
            },
        ):
            result = toolset.bili_transcribe("https://www.bilibili.com/video/BV1cgPSzeEj5")
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "bili_transcribe")
        self.assertEqual(result["data"]["bili_id"], "BV1cgPSzeEj5")
        self.assertEqual(result["data"]["transcript"], "line one")

    def test_bili_transcribe_invalid_bili_id_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=InvalidBiliIdError("bad id")):
            result = toolset.bili_transcribe("not a video")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_bili_id")

    def test_bili_transcribe_missing_dependency_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=MissingDependencyError("missing")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "missing_dependency")

    def test_bili_transcribe_download_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=BiliDownloadError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "download_failed")

    def test_bili_transcribe_transcription_error(self) -> None:
        toolset = Toolset(Settings(api_key="x"))
        with patch("deepfind.tools.transcribe_bili_audio", side_effect=TranscriptionError("failed")):
            result = toolset.bili_transcribe("BV1cgPSzeEj5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "transcription_failed")

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
