from __future__ import annotations

import io
import struct
import types
import unittest
from unittest.mock import Mock, patch

from deepfind import chrome_host


class ChromeHostTests(unittest.TestCase):
    def test_read_native_message_reads_length_prefixed_json(self) -> None:
        payload = b'{"type":"ping","payload":{"count":1}}'
        stdin = types.SimpleNamespace(buffer=io.BytesIO(struct.pack("@I", len(payload)) + payload))

        with patch("deepfind.chrome_host.sys.stdin", stdin):
            message = chrome_host.read_native_message()

        self.assertEqual(message, {"type": "ping", "payload": {"count": 1}})

    def test_write_native_message_writes_length_prefixed_json(self) -> None:
        stream = io.BytesIO()
        stdout = types.SimpleNamespace(buffer=stream)

        with patch("deepfind.chrome_host.sys.stdout", stdout):
            chrome_host.write_native_message({"type": "pong", "payload": {"ok": True}})

        raw = stream.getvalue()
        self.assertEqual(struct.unpack("@I", raw[:4])[0], len(raw[4:]))
        self.assertEqual(raw[4:].decode("utf-8"), '{"type": "pong", "payload": {"ok": true}}')

    def test_resolve_base_url_uses_runtime_state_when_no_flag_is_given(self) -> None:
        with patch("deepfind.chrome_host.read_runtime_state", return_value={"base_url": " http://127.0.0.1:9000 "}):
            self.assertEqual(chrome_host._resolve_base_url(None), "http://127.0.0.1:9000")

    def test_main_handles_ping_and_closes_session(self) -> None:
        bridge = Mock()
        bridge.open_session.return_value = {"id": "session_1"}
        poll_thread = Mock()
        ping_message = {"type": "ping", "payload": {"hello": "world"}}

        with (
            patch("deepfind.chrome_host.BrowserBridgeClient", return_value=bridge) as bridge_cls,
            patch("deepfind.chrome_host.Thread", return_value=poll_thread),
            patch("deepfind.chrome_host.read_native_message", side_effect=[ping_message, None]),
            patch("deepfind.chrome_host.write_native_message") as write_message,
        ):
            exit_code = chrome_host.main(
                [
                    "--base-url",
                    "http://127.0.0.1:8123",
                    "--label",
                    "Chrome bridge",
                    "--host-name",
                    "com.example.host",
                ]
            )

        self.assertEqual(exit_code, 0)
        bridge_cls.assert_called_once_with("http://127.0.0.1:8123")
        bridge.open_session.assert_called_once_with(
            label="Chrome bridge",
            capabilities=["native-messaging"],
            metadata={"host_name": "com.example.host"},
        )
        self.assertEqual(write_message.call_args_list[0].args[0]["type"], "ready")
        self.assertEqual(write_message.call_args_list[1].args[0], {"type": "pong", "session_id": "session_1"})
        bridge.post_event.assert_called_once()
        self.assertEqual(bridge.post_event.call_args.args[0], "session_1")
        self.assertEqual(bridge.post_event.call_args.kwargs["event_type"], "ping")
        self.assertEqual(bridge.post_event.call_args.kwargs["payload"]["hello"], "world")
        self.assertEqual(bridge.post_event.call_args.kwargs["payload"]["raw"], ping_message)
        poll_thread.start.assert_called_once_with()
        poll_thread.join.assert_called_once_with(timeout=1.0)
        bridge.close_session.assert_called_once_with("session_1")
        bridge.close.assert_called_once_with()

    def test_main_wraps_non_mapping_payloads_before_posting(self) -> None:
        bridge = Mock()
        bridge.open_session.return_value = {"id": "session_2"}
        poll_thread = Mock()
        snapshot_message = {"type": "page_snapshot", "payload": "raw text"}

        with (
            patch("deepfind.chrome_host.BrowserBridgeClient", return_value=bridge),
            patch("deepfind.chrome_host.Thread", return_value=poll_thread),
            patch("deepfind.chrome_host.read_native_message", side_effect=[snapshot_message, None]),
            patch("deepfind.chrome_host.write_native_message") as write_message,
        ):
            exit_code = chrome_host.main(["--base-url", "http://127.0.0.1:8123"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            write_message.call_args_list[1].args[0],
            {"type": "ack", "session_id": "session_2", "received_type": "page_snapshot"},
        )
        self.assertEqual(bridge.post_event.call_args.kwargs["payload"]["value"], "raw text")
        self.assertEqual(bridge.post_event.call_args.kwargs["payload"]["raw"], snapshot_message)
