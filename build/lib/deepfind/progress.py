from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, TextIO

from .json_utils import try_load_json


def _short(value: Any, width: int = 88) -> str:
    text = str(value).replace("\n", " ").strip()
    return textwrap.shorten(text, width=width, placeholder="...")


def _tool_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    data = parsed.get("data")
    if not isinstance(data, dict):
        return {}
    nested = data.get("data")
    if "ok" in data and isinstance(nested, dict):
        return nested
    return data


def _note_title_from_items(data: dict[str, Any]) -> str:
    items = data.get("items")
    if not isinstance(items, list) or len(items) != 1 or data.get("pages_fetched"):
        return ""
    first = items[0]
    if not isinstance(first, dict):
        return ""
    note_card = first.get("note_card")
    if not isinstance(note_card, dict):
        return ""
    title = note_card.get("title") or note_card.get("display_title") or ""
    return str(title).strip()


def _tool_summary(parsed: dict[str, Any], *, width: int = 64) -> str:
    if parsed.get("error"):
        details: list[str] = []
        if parsed.get("returncode") is not None:
            details.append(f"rc={parsed['returncode']}")
        stderr = str(parsed.get("stderr") or "").strip()
        stdout = str(parsed.get("stdout") or "").strip()
        if stderr:
            details.append(f"stderr={stderr}")
        elif stdout:
            details.append(f"stdout={stdout}")
        else:
            details.append(str(parsed["error"]))
        return _short(" ".join(details), width)

    data = _tool_payload(parsed)
    if data:
        note_title = _note_title_from_items(data)
        if note_title:
            return _short(note_title, width)
        if isinstance(data.get("items"), list):
            pages = data.get("pages_fetched")
            read_errors = data.get("enrichment_errors")
            suffix = ""
            if isinstance(read_errors, list) and read_errors:
                suffix = f", read_errors={len(read_errors)}"
            if pages:
                return f"items={len(data['items'])}, pages={pages}{suffix}"
            return f"items={len(data['items'])}{suffix}"
        if isinstance(data.get("notes"), list):
            return f"notes={len(data['notes'])}"
        if isinstance(data.get("user_info_dtos"), list):
            return f"users={len(data['user_info_dtos'])}"
        if isinstance(data.get("interactions"), list):
            metrics = []
            for item in data["interactions"]:
                name = item.get("name")
                count = item.get("count")
                if name and count:
                    metrics.append(f"{name}={count}")
            if metrics:
                return ", ".join(metrics[:3])
        note = data.get("note") if isinstance(data.get("note"), dict) else None
        if note and note.get("title"):
            return _short(note["title"], width)
        if data.get("transcript_path"):
            return _short(f"transcript={data['transcript_path']}", width)
        if data.get("image_path"):
            return _short(f"image={data['image_path']}", width)
        if data.get("html_path"):
            return _short(f"html={data['html_path']}", width)
        if data.get("title"):
            return _short(data["title"], width)
        if data.get("final_url"):
            return _short(data["final_url"], width)

    return ""


@dataclass
class ConsoleProgress:
    enabled: bool = True
    stream: TextIO = sys.stderr
    use_color: bool | None = None
    print_enabled: bool = True
    line_sink: Callable[[str], None] | None = None
    truncate_width: int = 88
    tool_summary_width: int = 64
    done_summary_width: int = 68
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        if self.use_color is None:
            self.use_color = bool(getattr(self.stream, "isatty", lambda: False)())

    def _color(self, text: str, code: str) -> str:
        if not self.enabled or not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _line(self, text: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.print_enabled:
                print(text, file=self.stream, flush=True)
            if self.line_sink:
                self.line_sink(text)

    def _event(self, scope: str, action: str, detail: str = "", color: str = "36") -> None:
        prefix = f"[{self._stamp()}] {scope:<10} {action:<10}"
        prefix = self._color(prefix, color)
        self._line(f"{prefix} {detail}".rstrip())

    def _box(self, title: str, rows: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        width = 78
        border = "+" + "-" * width + "+"
        self._line(self._color(border, "2"))
        self._line(self._color(f"| {title:<{width-1}}|", "1;36"))
        self._line(self._color(border, "2"))
        for key, value in rows:
            wrapped = textwrap.wrap(value, width=width - 13) or [""]
            for index, chunk in enumerate(wrapped):
                label = f"{key:<10}" if index == 0 else " " * 10
                self._line(f"| {label} {chunk:<{width-12}}|")
        self._line(self._color(border, "2"))

    def run_started(self, query: str, num_agent: int, max_iter: int) -> None:
        self._box(
            "DEEPFIND",
            [
                ("query", query),
                ("agents", str(num_agent)),
                ("max_iter", str(max_iter)),
            ],
        )

    def plan_ready(self, tasks: list[str]) -> None:
        rows = [(f"task {index}", task) for index, task in enumerate(tasks, 1)]
        self._box("PLAN", rows)

    def worker_started(self, name: str, task: str) -> None:
        self._event(name.upper(), "start", _short(task, self.truncate_width), "35")

    def iteration(self, name: str, iteration: int) -> None:
        self._event(name.upper(), "iter", f"round {iteration}", "34")

    def tool_call(self, name: str, iteration: int, tool_name: str, arguments: dict[str, Any]) -> None:
        detail = f"{tool_name} {_short(arguments, self.truncate_width)}"
        self._event(name.upper(), f"tool {iteration}", detail, "33")

    def tool_result(self, name: str, tool_name: str, output: str) -> None:
        parsed = try_load_json(output)
        if isinstance(parsed, dict):
            ok = parsed.get("ok")
            suffix = _tool_summary(parsed, width=self.tool_summary_width)
        else:
            ok = None
            suffix = ""
        status = "ok" if ok is True else "err" if ok is False else "done"
        color = "32" if ok is True else "31" if ok is False else "36"
        detail = tool_name
        if suffix:
            detail = f"{tool_name} {_short(suffix, self.truncate_width)}".rstrip()
        self._event(name.upper(), status, detail, color)

    def synthesize_started(self, report_count: int) -> None:
        self._event("LEAD", "merge", f"{report_count} worker reports", "36")

    def agent_done(self, name: str, iterations: int, text: str) -> None:
        self._event(
            name.upper(),
            "done",
            f"{iterations} rounds | {_short(text, self.done_summary_width)}",
            "32",
        )
