from __future__ import annotations

import io
import unittest

from deepfind.progress import ConsoleProgress
from deepfind.web_progress import WebProgress


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class ProgressTests(unittest.TestCase):
    def test_console_progress_mirrors_lines_to_sink(self) -> None:
        stream = TtyStringIO()
        mirrored: list[str] = []
        progress = ConsoleProgress(
            stream=stream,
            use_color=True,
            line_sink=mirrored.append,
        )

        progress.run_started("hello world", 1, 5)
        progress.agent_done("lead-final", 3, '{"summary":"done"}')

        printed = stream.getvalue().splitlines()
        self.assertEqual(mirrored, printed)
        self.assertGreaterEqual(len(printed), 8)
        self.assertTrue(any("\x1b[1;36m| DEEPFIND" in line for line in printed))
        self.assertTrue(any("LEAD-FINAL" in line and "done" in line for line in printed))

    def test_web_progress_emits_console_line_events(self) -> None:
        progress = WebProgress()

        progress.run_started("hello", 1, 5)
        progress.plan_ready(["core facts"])
        progress.tool_result(
            "sub-1",
            "web_search",
            '{"ok": false, "tool": "web_search", "error": "timeout"}',
        )
        progress.close()

        events = list(progress.iter_events())

        self.assertEqual(events[0].type, "run_started")
        self.assertIn("plan_ready", [event.type for event in events])
        console_lines = [event for event in events if event.type == "console_line"]
        self.assertGreaterEqual(len(console_lines), 9)
        self.assertTrue(any("\x1b[1;36m| DEEPFIND" in str(event.data.get("text", "")) for event in console_lines))
        self.assertTrue(any("web_search timeout" in str(event.data.get("text", "")) for event in console_lines))

    def test_console_progress_summarizes_enveloped_xhs_read_output(self) -> None:
        stream = io.StringIO()
        progress = ConsoleProgress(stream=stream, use_color=False)

        progress.tool_result(
            "sub-1",
            "xhs_read",
            (
                '{"ok": true, "tool": "xhs_read", "data": {"ok": true, "schema_version": "1", "data": '
                '{"items": [{"id": "69dc33da000000001a021e16", "note_card": {"title": '
                '"美中央司令部：13日起封锁伊朗港口海上交通"}}]}}}'
            ),
        )

        line = stream.getvalue().strip()
        self.assertIn("xhs_read 美中央司令部：13日起封锁伊朗港口海上交通", line)
        self.assertNotIn("xhs_read xhs_read", line)

    def test_web_progress_summarizes_enveloped_xhs_read_output(self) -> None:
        progress = WebProgress()

        progress.tool_result(
            "sub-1",
            "xhs_read",
            (
                '{"ok": true, "tool": "xhs_read", "data": {"ok": true, "schema_version": "1", "data": '
                '{"items": [{"id": "69dc33da000000001a021e16", "note_card": {"title": '
                '"美中央司令部：13日起封锁伊朗港口海上交通"}}]}}}'
            ),
        )
        progress.close()

        tool_result_events = [event for event in progress.iter_events() if event.type == "tool_result"]
        self.assertEqual(len(tool_result_events), 1)
        self.assertEqual(tool_result_events[0].data["summary"], "美中央司令部：13日起封锁伊朗港口海上交通")

    def test_web_progress_summarizes_normalized_xhs_read_output(self) -> None:
        progress = WebProgress()

        progress.tool_result(
            "sub-1",
            "xhs_read",
            (
                '{"ok":true,"tool":"xhs_read","data":{"ref":"69db389a000000002103a532","note":'
                '{"title":"美伊谈判破裂，一场没有赢家的博弈"}}}'
            ),
        )
        progress.close()

        tool_result_events = [event for event in progress.iter_events() if event.type == "tool_result"]
        self.assertEqual(len(tool_result_events), 1)
        self.assertEqual(tool_result_events[0].data["summary"], "美伊谈判破裂，一场没有赢家的博弈")
