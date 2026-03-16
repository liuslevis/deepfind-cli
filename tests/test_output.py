from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from deepfind.output import render_answer


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class OutputTests(unittest.TestCase):
    def test_render_answer_prints_plain_when_not_tty(self) -> None:
        stream = io.StringIO()
        with patch("deepfind.output._run_frogmouth") as frogmouth_mock:
            render_answer("final answer", viewer="auto", stream=stream)
        self.assertEqual(stream.getvalue(), "final answer\n")
        frogmouth_mock.assert_not_called()

    def test_render_answer_uses_frogmouth_when_tty(self) -> None:
        stream = TtyStringIO()
        with patch("deepfind.output.sys.stdin.isatty", return_value=True):
            with patch("deepfind.output._run_frogmouth", return_value=True) as frogmouth_mock:
                render_answer("final answer", viewer="auto", stream=stream)
        self.assertEqual(stream.getvalue(), "")
        frogmouth_mock.assert_called_once_with("final answer")

    def test_render_answer_falls_back_when_frogmouth_unavailable(self) -> None:
        stream = TtyStringIO()
        with patch("deepfind.output.sys.stdin.isatty", return_value=True):
            with patch("deepfind.output._run_frogmouth", return_value=False) as frogmouth_mock:
                render_answer("final answer", viewer="frogmouth", stream=stream)
        self.assertEqual(stream.getvalue(), "final answer\n")
        frogmouth_mock.assert_called_once_with("final answer")
