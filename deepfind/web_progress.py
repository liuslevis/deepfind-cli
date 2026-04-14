from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Lock
from typing import Any, Iterator

from .chat_store import utc_now
from .json_utils import try_load_json
from .progress import ConsoleProgress
from .web_models import ProgressEvent, TurnResult

_WEB_CONSOLE_TRUNCATE_WIDTH = 2000


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    output: str


def _summary_from_output(output: str) -> tuple[str, str]:
    parsed = try_load_json(output)
    if not isinstance(parsed, dict):
        return "done", ""
    if parsed.get("ok") is True:
        status = "ok"
    elif parsed.get("ok") is False:
        status = "error"
    else:
        status = "done"
    if parsed.get("error"):
        return status, str(parsed["error"])

    data = parsed.get("data")
    if isinstance(data, dict):
        if data.get("image_path"):
            return status, str(data["image_path"])
        if data.get("html_path"):
            return status, str(data["html_path"])
        if isinstance(data.get("items"), list):
            pages = data.get("pages_fetched")
            if pages:
                return status, f"items={len(data['items'])}, pages={pages}"
            return status, f"items={len(data['items'])}"
        if isinstance(data.get("notes"), list):
            return status, f"notes={len(data['notes'])}"
        if data.get("transcript_path"):
            return status, str(data["transcript_path"])
        if data.get("title"):
            return status, str(data["title"])
        if data.get("final_url"):
            return status, str(data["final_url"])
    return status, str(parsed.get("tool", ""))


class WebProgress:
    def __init__(self) -> None:
        self._queue: Queue[ProgressEvent | object] = Queue()
        self._sentinel = object()
        self._lock = Lock()
        self.tool_outputs: list[ToolObservation] = []
        self._console_progress = ConsoleProgress(
            use_color=True,
            print_enabled=False,
            line_sink=self._emit_console_line,
            truncate_width=_WEB_CONSOLE_TRUNCATE_WIDTH,
            tool_summary_width=_WEB_CONSOLE_TRUNCATE_WIDTH,
            done_summary_width=_WEB_CONSOLE_TRUNCATE_WIDTH,
        )

    def _event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        self._queue.put(
            ProgressEvent(
                type=event_type,
                timestamp=utc_now(),
                data=data or {},
            )
        )

    def _emit_console_line(self, text: str) -> None:
        self._event("console_line", {"text": text})

    def run_started(self, query: str, num_agent: int, max_iter: int) -> None:
        self._event(
            "run_started",
            {
                "query": query,
                "num_agent": num_agent,
                "max_iter_per_agent": max_iter,
            },
        )
        self._console_progress.run_started(query, num_agent, max_iter)

    def plan_ready(self, tasks: list[str]) -> None:
        self._event("plan_ready", {"tasks": tasks})
        self._console_progress.plan_ready(tasks)

    def worker_started(self, name: str, task: str) -> None:
        self._event("worker_started", {"name": name, "task": task})
        self._console_progress.worker_started(name, task)

    def iteration(self, name: str, iteration: int) -> None:
        self._event("iteration", {"name": name, "iteration": iteration})
        self._console_progress.iteration(name, iteration)

    def tool_call(self, name: str, iteration: int, tool_name: str, arguments: dict[str, Any]) -> None:
        self._event(
            "tool_call",
            {
                "name": name,
                "iteration": iteration,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )
        self._console_progress.tool_call(name, iteration, tool_name, arguments)

    def tool_result(self, name: str, tool_name: str, output: str) -> None:
        status, summary = _summary_from_output(output)
        with self._lock:
            self.tool_outputs.append(ToolObservation(tool_name=tool_name, output=output))
        self._event(
            "tool_result",
            {
                "name": name,
                "tool_name": tool_name,
                "status": status,
                "summary": summary,
            },
        )
        self._console_progress.tool_result(name, tool_name, output)

    def synthesize_started(self, report_count: int) -> None:
        self._event("synthesize_started", {"report_count": report_count})
        self._console_progress.synthesize_started(report_count)

    def agent_done(self, name: str, iterations: int, text: str) -> None:
        self._event(
            "iteration",
            {
                "name": name,
                "iteration": iterations,
                "status": "done",
            },
        )
        self._console_progress.agent_done(name, iterations, text)

    def emit_answer_delta(self, delta: str) -> None:
        self._event("answer_delta", {"delta": delta})

    def emit_answer_final(self, turn_result: TurnResult) -> None:
        self._event("answer_final", turn_result.model_dump(mode="json"))

    def emit_error(self, message: str) -> None:
        self._event("error", {"message": message})

    def emit_done(self, data: dict[str, Any] | None = None) -> None:
        self._event("done", data or {})

    def close(self) -> None:
        self._queue.put(self._sentinel)

    def iter_events(self) -> Iterator[ProgressEvent]:
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item
