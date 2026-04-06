from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from itertools import count
from unittest.mock import patch

from deepfind.cli import main


class NonTtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return False


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliTests(unittest.TestCase):
    def test_list_tools_prints_tools_without_initializing_app(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            stdout = io.StringIO()
            code = main(["--list-tools"], stdout=stdout, stderr=io.StringIO())
        self.assertEqual(code, 0)
        app_cls.assert_not_called()
        lines = stdout.getvalue().strip().splitlines()
        self.assertTrue(any(line.startswith("web_search\t") for line in lines))
        self.assertTrue(any(line.startswith("web_fetch\t") for line in lines))
        self.assertTrue(any(line.startswith("boss_search\t") for line in lines))
        self.assertTrue(any(line.startswith("boss_detail\t") for line in lines))
        self.assertTrue(any(line.startswith("bili_search\t") for line in lines))
        self.assertTrue(any(line.startswith("bili_get_user_videos\t") for line in lines))

    def test_main_prints_answer(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.return_value = "final answer"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["test query", "--num-agent", "3", "--max-iter-per-agent", "7"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "final answer")
        app_cls.return_value.session.assert_called_once_with(
            num_agent=3,
            max_iter_per_agent=7,
        )
        session.ask.assert_called_once_with("test query")
        session.ask_detailed.assert_not_called()

    def test_main_prints_structured_json_when_requested(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask_detailed.return_value = {
                "version": "research.v1",
                "lead": {"overview_md": "structured"},
                "agents": [],
                "citations": [],
                "citations_dedup": [],
                "meta": {},
            }
            stdout = io.StringIO()

            code = main(
                ["test query", "--json"],
                stdin=NonTtyStringIO(),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "version": "research.v1",
                "lead": {"overview_md": "structured"},
                "agents": [],
                "citations": [],
                "citations_dedup": [],
                "meta": {},
            },
        )
        session.ask_detailed.assert_called_once_with("test query")
        session.ask.assert_not_called()

    def test_main_rejects_invalid_num_agent(self) -> None:
        with self.assertRaises(SystemExit):
            main(["test query", "--num-agent", "0"])

    def test_main_requires_query_without_list_tools(self) -> None:
        with self.assertRaises(SystemExit):
            main([])

    def test_main_stays_one_shot_for_non_tty(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.return_value = "final answer"
            stdout = NonTtyStringIO()
            stderr = io.StringIO()
            code = main(
                ["test query"],
                stdin=NonTtyStringIO(),
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "final answer")
        session.ask.assert_called_once_with("test query")
        self.assertEqual(stderr.getvalue(), "")

    def test_main_respects_once_flag_in_tty(self) -> None:
        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.return_value = "final answer"
            stdout = TtyStringIO()
            code = main(
                ["test query", "--once"],
                stdin=TtyStringIO(),
                stdout=stdout,
                stderr=io.StringIO(),
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "final answer")
        session.ask.assert_called_once_with("test query")

    def test_main_json_stays_one_shot_in_tty(self) -> None:
        def fake_input(_: str) -> str:
            raise AssertionError("chat mode should be disabled for --json")

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask_detailed.return_value = {
                "version": "research.v1",
                "lead": {"overview_md": "structured"},
                "agents": [],
                "citations": [],
                "citations_dedup": [],
                "meta": {},
            }
            code = main(
                ["test query", "--json"],
                stdin=TtyStringIO(),
                stdout=TtyStringIO(),
                stderr=io.StringIO(),
                input_fn=fake_input,
            )

        self.assertEqual(code, 0)
        session.ask_detailed.assert_called_once_with("test query")

    def test_main_enters_chat_mode_and_handles_follow_up(self) -> None:
        prompts = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return {0: "follow up", 1: "exit"}[len(prompts) - 1]

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.side_effect = ["first answer", "second answer"]
            stdout = TtyStringIO()
            code = main(
                ["test query"],
                stdin=TtyStringIO(),
                stdout=stdout,
                stderr=io.StringIO(),
                input_fn=fake_input,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip().splitlines(), ["first answer", "second answer"])
        self.assertEqual(session.ask.call_args_list[0].args, ("test query",))
        self.assertEqual(session.ask.call_args_list[1].args, ("follow up",))
        self.assertEqual(prompts, ["deepfind> ", "deepfind> "])

    def test_blank_follow_up_is_ignored(self) -> None:
        calls = count()

        def fake_input(_: str) -> str:
            items = ["   ", "follow up", "quit"]
            return items[next(calls)]

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.side_effect = ["first answer", "second answer"]
            code = main(
                ["test query"],
                stdin=TtyStringIO(),
                stdout=TtyStringIO(),
                stderr=io.StringIO(),
                input_fn=fake_input,
            )
        self.assertEqual(code, 0)
        self.assertEqual(session.ask.call_count, 2)
        self.assertEqual(session.ask.call_args_list[1].args, ("follow up",))

    def test_eof_ends_chat_mode_cleanly(self) -> None:
        def fake_input(_: str) -> str:
            raise EOFError

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.return_value = "first answer"
            code = main(
                ["test query"],
                stdin=TtyStringIO(),
                stdout=TtyStringIO(),
                stderr=io.StringIO(),
                input_fn=fake_input,
            )
        self.assertEqual(code, 0)
        session.ask.assert_called_once_with("test query")

    def test_keyboard_interrupt_ends_chat_mode_cleanly(self) -> None:
        def fake_input(_: str) -> str:
            raise KeyboardInterrupt

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.return_value = "first answer"
            code = main(
                ["test query"],
                stdin=TtyStringIO(),
                stdout=TtyStringIO(),
                stderr=io.StringIO(),
                input_fn=fake_input,
            )
        self.assertEqual(code, 0)
        session.ask.assert_called_once_with("test query")

    def test_follow_up_error_keeps_chat_alive(self) -> None:
        answers = iter(["bad follow up", "good follow up", "quit"])

        def fake_input(_: str) -> str:
            return next(answers)

        with patch("deepfind.cli.DeepFind") as app_cls:
            session = app_cls.return_value.session.return_value
            session.ask.side_effect = ["first answer", RuntimeError("boom"), "third answer"]
            stdout = TtyStringIO()
            stderr = io.StringIO()
            code = main(
                ["test query"],
                stdin=TtyStringIO(),
                stdout=stdout,
                stderr=stderr,
                input_fn=fake_input,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip().splitlines(), ["first answer", "third answer"])
        self.assertIn("error: boom", stderr.getvalue())
        self.assertEqual(session.ask.call_count, 3)
