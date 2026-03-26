from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from deepfind import desktop_app


class DesktopAppTests(unittest.TestCase):
    def test_launch_window_uses_cocoa_on_macos(self) -> None:
        fake_webview = types.SimpleNamespace(create_window=Mock(), start=Mock())

        with (
            patch.dict(sys.modules, {"webview": fake_webview}),
            patch("deepfind.desktop_app.sys.platform", "darwin"),
        ):
            desktop_app._launch_window("http://127.0.0.1:8123", title="DeepFind", debug=True)

        fake_webview.create_window.assert_called_once_with(
            "DeepFind",
            "http://127.0.0.1:8123",
            width=1440,
            height=940,
            min_size=(1120, 720),
            text_select=True,
        )
        fake_webview.start.assert_called_once_with(debug=True, gui="cocoa")

    def test_main_installs_host_and_launches_server_url(self) -> None:
        server = Mock()
        server.base_url = "http://127.0.0.1:8123"

        with (
            patch("deepfind.desktop_app.configure_desktop_environment") as configure,
            patch("deepfind.desktop_app.resolve_web_dist_dir", return_value=Path("/tmp/web/dist")),
            patch("deepfind.desktop_app.install_native_host_manifest") as install_host,
            patch("deepfind.desktop_app.DesktopBackendServer", return_value=server) as server_cls,
            patch("deepfind.desktop_app._launch_window") as launch_window,
        ):
            exit_code = desktop_app.main(
                [
                    "--install-chrome-host",
                    "--chrome-extension-id",
                    "abcdefghijklmnopabcdefghijklmnop",
                    "--title",
                    "DeepFind Desktop",
                ]
            )

        self.assertEqual(exit_code, 0)
        configure.assert_called_once_with()
        install_host.assert_called_once_with(extension_ids=["abcdefghijklmnopabcdefghijklmnop"])
        server_cls.assert_called_once_with(host="127.0.0.1", port=0)
        server.start.assert_called_once_with()
        launch_window.assert_called_once_with(server.base_url, title="DeepFind Desktop", debug=False)
        server.stop.assert_called_once_with()

    def test_main_stops_server_when_window_launch_fails(self) -> None:
        server = Mock()
        server.base_url = "http://127.0.0.1:8123"

        with (
            patch("deepfind.desktop_app.configure_desktop_environment"),
            patch("deepfind.desktop_app.DesktopBackendServer", return_value=server),
            patch("deepfind.desktop_app._launch_window", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                desktop_app.main(["--url", "http://example.com"])

        server.start.assert_called_once_with()
        server.stop.assert_called_once_with()
