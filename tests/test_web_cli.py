from __future__ import annotations

import io
import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from deepfind import web_cli
from deepfind.web_cli import (
    CreatedChat,
    WebClientError,
    _parse_event_block,
    build_parser,
    iter_sse_events,
    main,
    resolve_base_url,
    resolve_token,
    stream_message,
)


class SseParsingTests(unittest.TestCase):
    def test_parse_event_block_reads_type_and_data(self) -> None:
        event = _parse_event_block('event: done\ndata: {"data": {"chat_id": "chat_1"}}')
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "done")
        self.assertEqual(event["data"], {"chat_id": "chat_1"})

    def test_parse_event_block_without_data_returns_none(self) -> None:
        self.assertIsNone(_parse_event_block("event: done"))

    def test_parse_event_block_ignores_invalid_json(self) -> None:
        self.assertIsNone(_parse_event_block("data: not-json"))

    def test_iter_sse_events_handles_split_chunks(self) -> None:
        chunks = [
            b'event: answer_delta\ndata: {"data": {"delta": "hel',
            b'lo"}}\n\n',
            b'event: done\ndata: {"data": {"assistant_message_id": "msg_1"}}\n\n',
        ]
        events = list(iter_sse_events(chunks))
        self.assertEqual([event["type"] for event in events], ["answer_delta", "done"])
        self.assertEqual(events[0]["data"], {"delta": "hello"})
        self.assertEqual(events[1]["data"], {"assistant_message_id": "msg_1"})

    def test_iter_sse_events_flushes_trailing_block(self) -> None:
        events = list(iter_sse_events([b'event: done\ndata: {"data": {}}']))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "done")

    def test_iter_sse_events_normalizes_crlf(self) -> None:
        raw = b'event: done\r\ndata: {"data": {"ok": true}}\r\n\r\n'
        events = list(iter_sse_events([raw]))
        self.assertEqual(events[0]["data"], {"ok": True})

    def test_stream_message_rejects_truncated_stream(self) -> None:
        response = nullcontext(
            [b'event: answer_delta\ndata: {"data": {"delta": "partial"}}\n\n']
        )
        with patch("deepfind.web_cli._request", return_value=response):
            with self.assertRaisesRegex(WebClientError, "ended before completion"):
                stream_message("http://localhost", "chat_1", "query", token="")

    def test_stream_message_requires_assistant_message(self) -> None:
        response = nullcontext([b'event: done\ndata: {"data": {"chat_id": "chat_1"}}\n\n'])
        with patch("deepfind.web_cli._request", return_value=response):
            with self.assertRaisesRegex(WebClientError, "without an assistant message"):
                stream_message("http://localhost", "chat_1", "query", token="")


class ResolutionTests(unittest.TestCase):
    def test_resolve_base_url_defaults_and_strips_trailing_slash(self) -> None:
        self.assertEqual(resolve_base_url("http://127.0.0.1:9000/"), "http://127.0.0.1:9000")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_base_url(None), web_cli.DEFAULT_BASE_URL)

    def test_resolve_token_prefers_explicit_value(self) -> None:
        with patch.dict("os.environ", {"DEEPFIND_WEB_TOKEN": "from-env"}):
            self.assertEqual(resolve_token(" explicit "), "explicit")
            self.assertEqual(resolve_token(None), "from-env")

    def test_parser_defaults_are_agents_chat_mode_on_deepseek(self) -> None:
        args = build_parser().parse_args(["hello"])
        self.assertEqual(args.mode, "fast")
        self.assertEqual(args.model, "deepseek")
        self.assertEqual(args.research_mode, "chat")


class MainFlowTests(unittest.TestCase):
    def _run_main(self, argv, *, requires_token=False, done=None, side_effect=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict("os.environ", {"DEEPFIND_WEB_TOKEN": "tok"}, clear=False),
            patch("deepfind.web_cli._load_dotenv_if_needed"),
            patch(
                "deepfind.web_cli.check_health",
                return_value={"ok": True, "requires_token": requires_token},
            ),
            patch(
                "deepfind.web_cli.create_chat",
                return_value=CreatedChat(id="chat_abc", title="t"),
            ),
            patch("deepfind.web_cli.stream_message", **(
                {"side_effect": side_effect} if side_effect else {"return_value": done or {}}
            )) as stream_mock,
        ):
            code = main(argv, stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue(), stream_mock

    def test_main_creates_chat_and_streams_with_defaults(self) -> None:
        code, out, err, stream_mock = self._run_main(
            ["summarize this"],
            done={"assistant_message_id": "msg_1"},
        )
        self.assertEqual(code, 0)
        self.assertIn("chat_abc", out)
        self.assertIn("Chat created: chat_abc", err)
        kwargs = stream_mock.call_args.kwargs
        self.assertEqual(kwargs["mode"], "fast")
        self.assertEqual(kwargs["model_target"], "deepseek")
        self.assertEqual(kwargs["research_mode"], "chat")

    def test_main_json_prints_identifiers(self) -> None:
        code, out, _, _ = self._run_main(
            ["summarize this", "--json"],
            done={"assistant_message_id": "msg_1"},
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["chat_id"], "chat_abc")
        self.assertEqual(payload["assistant_message_id"], "msg_1")

    def test_main_requires_token_when_backend_demands_it(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("deepfind.web_cli._load_dotenv_if_needed"),
            patch(
                "deepfind.web_cli.check_health",
                return_value={"ok": True, "requires_token": True},
            ),
            patch("deepfind.web_cli.create_chat") as create_mock,
        ):
            code = main(["q"], stdout=stdout, stderr=stderr)
        self.assertEqual(code, 1)
        self.assertIn("requires a token", stderr.getvalue())
        create_mock.assert_not_called()

    def test_main_reports_backend_errors(self) -> None:
        code, _, err, _ = self._run_main(
            ["q"],
            side_effect=WebClientError("boom"),
        )
        self.assertEqual(code, 1)
        self.assertIn("error: boom", err)


if __name__ == "__main__":
    unittest.main()
