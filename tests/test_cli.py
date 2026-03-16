from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from deepfind.cli import main


class CliTests(unittest.TestCase):
    def test_main_prints_answer(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            app_cls.return_value.run.return_value = "final answer"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["test query", "--num-agent", "3", "--max-iter-per-agent", "7"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "final answer")
        app_cls.return_value.run.assert_called_once_with(
            query="test query",
            num_agent=3,
            max_iter_per_agent=7,
        )

    def test_main_rejects_invalid_num_agent(self) -> None:
        with self.assertRaises(SystemExit):
            main(["test query", "--num-agent", "0"])
