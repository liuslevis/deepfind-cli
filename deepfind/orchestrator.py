from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from .config import Settings
from .json_utils import dump_json, try_load_json
from .llm import ResponseAgent
from .models import WorkerReport
from .progress import ConsoleProgress
from .tools import Toolset

PLAN_PROMPT = "Split into {n} distinct research tasks. JSON array only."
WORKER_PROMPT = 'Do the task. Use tools. JSON only: {"summary":"","facts":[{"point":"","source":""}],"gaps":[]}.'
LEAD_PROMPT = "Merge worker reports. Fill gaps with tools if needed. Answer briefly with sources."


def _fallback_tasks(query: str, count: int) -> list[str]:
    pool = [
        f"Core facts: {query}",
        f"Recent web coverage: {query}",
        f"Social signals: {query}",
        f"Primary sources: {query}",
    ]
    return pool[:count] if count <= len(pool) else pool + [query] * (count - len(pool))


def _worker_payload(query: str, task: str) -> str:
    return f"d={date.today().isoformat()}\nq={query}\nt={task}"


def _normalize_task(item: object) -> str:
    if isinstance(item, dict):
        task = item.get("task") or item.get("title") or item.get("summary")
        if task:
            return str(task).strip()
    return str(item).strip()


def _parse_report(task: str, text: str, citations: list[str]) -> WorkerReport:
    parsed = try_load_json(text)
    if not isinstance(parsed, dict):
        parsed = {"summary": text.strip(), "facts": [], "gaps": ["non_json_output"]}
    if citations:
        for url in citations:
            if url not in text:
                parsed.setdefault("facts", [])
                parsed["facts"].append({"point": "citation", "source": url})
    return WorkerReport(task=task, text=text, citations=citations, parsed=parsed)


class DeepFind:
    def __init__(
        self,
        settings: Settings | None = None,
        progress: ConsoleProgress | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.tools = Toolset(self.settings)
        self.progress = progress

    def run(self, query: str, num_agent: int, max_iter_per_agent: int) -> str:
        num_agent = max(1, min(4, num_agent))
        if self.progress:
            self.progress.run_started(query, num_agent, max_iter_per_agent)
        tasks = self._plan(query, num_agent, max_iter_per_agent)
        reports = self._run_workers(query, tasks, max_iter_per_agent)
        return self._lead(query, reports, max_iter_per_agent).strip()

    def _plan(self, query: str, num_agent: int, max_iter: int) -> list[str]:
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        result = agent.run(
            name="lead-plan",
            instructions=PLAN_PROMPT.format(n=num_agent),
            user_input=f"d={date.today().isoformat()}\nq={query}",
            use_tools=False,
        )
        parsed = try_load_json(result.text)
        if isinstance(parsed, list):
            tasks = [_normalize_task(item) for item in parsed]
            tasks = [task for task in tasks if task]
            if tasks:
                if len(tasks) < num_agent:
                    tasks.extend(_fallback_tasks(query, num_agent - len(tasks)))
                if self.progress:
                    self.progress.plan_ready(tasks[:num_agent])
                return tasks[:num_agent]
        tasks = _fallback_tasks(query, num_agent)
        if self.progress:
            self.progress.plan_ready(tasks)
        return tasks

    def _run_workers(self, query: str, tasks: list[str], max_iter: int) -> list[WorkerReport]:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(self._run_worker, index + 1, query, task, max_iter)
                for index, task in enumerate(tasks)
            ]
        return [future.result() for future in futures]

    def _run_worker(self, index: int, query: str, task: str, max_iter: int) -> WorkerReport:
        name = f"sub-{index}"
        if self.progress:
            self.progress.worker_started(name, task)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        result = agent.run(
            name=name,
            instructions=WORKER_PROMPT,
            user_input=_worker_payload(query, task),
            use_tools=True,
        )
        return _parse_report(task, result.text, result.citations)

    def _lead(self, query: str, reports: list[WorkerReport], max_iter: int) -> str:
        if self.progress:
            self.progress.synthesize_started(len(reports))
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        report_blob = dump_json(
            [
                {
                    "task": report.task,
                    "summary": report.parsed.get("summary", ""),
                    "facts": report.parsed.get("facts", []),
                    "gaps": report.parsed.get("gaps", []),
                }
                for report in reports
            ]
        )
        result = agent.run(
            name="lead-final",
            instructions=LEAD_PROMPT,
            user_input=f"d={date.today().isoformat()}\nq={query}\nreports={report_blob}",
            use_tools=True,
        )
        return result.text
