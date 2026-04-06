from __future__ import annotations

import io
import json
import unittest

from deepfind.output import render_answer, render_json_answer


class OutputTests(unittest.TestCase):
    def test_render_answer_prints_plain_text(self) -> None:
        stream = io.StringIO()

        render_answer("final answer", stream=stream)

        self.assertEqual(stream.getvalue(), "final answer\n")

    def test_render_json_answer_prints_compact_json(self) -> None:
        stream = io.StringIO()

        render_json_answer({"lead": {"overview_md": "hi"}, "agents": []}, stream=stream)

        self.assertEqual(json.loads(stream.getvalue()), {"lead": {"overview_md": "hi"}, "agents": []})
