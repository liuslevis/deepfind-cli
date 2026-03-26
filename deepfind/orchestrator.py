from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .config import Settings
from .json_utils import dump_json, try_load_json
from .llm import ResponseAgent
from .models import ChatMessage, WorkerReport
from .progress import ConsoleProgress
from .tools import Toolset

PLAN_PROMPT = (
    "You are the lead planner for an ongoing research chat. Use the prior conversation for context when needed, "
    "but focus on the latest user request. If the user wants an image or slides, plan only supporting research/context "
    "tasks and leave the final asset creation for the lead response. Prefer broad web research tasks that can use "
    "web_search, and reserve Xiaohongshu, X/Twitter, Bilibili, and BOSS直聘 job-specific tasks for the matching "
    "platform tools. "
    "Split it into {n} distinct research tasks. JSON array only."
)
WORKER_PROMPT = (
    "You are a research worker in an ongoing chat. Use the conversation history for context when the latest request "
    "depends on earlier turns. Do the task. Use tools. Prefer web_search for broad web research, and keep using "
    "the Xiaohongshu, X/Twitter, Bilibili, and BOSS直聘-specific tools for those platforms. Use boss_search for "
    "job/职位/岗位 searches and boss_detail when you need one posting's full description. If the task mentions "
    'Bilibili video/audio, call bili_transcribe with the URL or BVID before summarizing. If the latest user request asks for an image, do not call gen_img unless the assigned task explicitly asks you to produce the final image asset. If the latest user request asks for slides, do not call gen_slides unless the assigned task explicitly asks you to produce the final slide asset. JSON only: {"summary":"","facts":[{"point":"","source":""}],"gaps":[]}.'
)
LEAD_PROMPT = (
    "You are the lead researcher in an ongoing chat. Use the conversation history for context, answer the latest "
    "user request, merge worker reports, fill gaps with tools if needed, and answer briefly with sources. Prefer "
    "web_search for broad web research gaps, and use the platform-specific tools for Xiaohongshu, X/Twitter, "
    "Bilibili, and BOSS直聘 job tasks. If the latest user request asks for an image, call gen_img exactly once with "
    "a concrete prompt, "
    "then mention the saved image path in the final answer. If the latest user request asks for slides, call "
    "gen_slides exactly once with a concrete prompt and slide_count, then mention the saved html path in the final "
    "answer. Do not claim an asset exists unless the corresponding tool succeeds."
)


def _fallback_tasks(query: str, count: int) -> list[str]:
    pool = [
        f"Core facts: {query}",
        f"Recent web coverage: {query}",
        f"Social signals: {query}",
        f"Primary sources: {query}",
    ]
    return pool[:count] if count <= len(pool) else pool + [query] * (count - len(pool))


def _history_messages(transcript: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in transcript]


def _planner_payload(query: str) -> str:
    return f"d={date.today().isoformat()}\nlatest_user_request={query}"


def _worker_payload(query: str, task: str) -> str:
    return f"d={date.today().isoformat()}\nlatest_user_request={query}\nassigned_task={task}"


def _lead_payload(query: str, reports: str) -> str:
    return f"d={date.today().isoformat()}\nlatest_user_request={query}\nreports={reports}"


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

    def session(self, num_agent: int, max_iter_per_agent: int) -> "ChatSession":
        num_agent, max_iter_per_agent = self._validated_run_args(num_agent, max_iter_per_agent)
        return ChatSession(
            app=self,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
        )

    def run(self, query: str, num_agent: int, max_iter_per_agent: int) -> str:
        return self.session(num_agent, max_iter_per_agent).ask(query)

    def _validated_run_args(self, num_agent: int, max_iter_per_agent: int) -> tuple[int, int]:
        if not 1 <= num_agent <= 4:
            raise ValueError("num_agent must be between 1 and 4")
        if max_iter_per_agent < 1:
            raise ValueError("max_iter_per_agent must be >= 1")
        return num_agent, max_iter_per_agent

    def _run_turn(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        num_agent: int,
        max_iter_per_agent: int,
    ) -> str:
        answer, _ = self._run_turn_detailed(
            query=query,
            transcript=transcript,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
        )
        return answer

    def _run_turn_detailed(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        num_agent: int,
        max_iter_per_agent: int,
    ) -> tuple[str, list[WorkerReport]]:
        if self.progress:
            self.progress.run_started(query, num_agent, max_iter_per_agent)
        tasks = self._plan(query, transcript, num_agent, max_iter_per_agent)
        reports = self._run_workers(query, transcript, tasks, max_iter_per_agent)
        answer = self._lead(query, transcript, reports, max_iter_per_agent).strip()
        return answer, reports

    def _plan(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        num_agent: int,
        max_iter: int,
    ) -> list[str]:
        history = _history_messages(transcript)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        result = agent.run(
            name="lead-plan",
            instructions=PLAN_PROMPT.format(n=num_agent),
            user_input=_planner_payload(query),
            use_tools=False,
            history=history,
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

    def _run_workers(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        tasks: list[str],
        max_iter: int,
    ) -> list[WorkerReport]:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(self._run_worker, index + 1, query, transcript, task, max_iter)
                for index, task in enumerate(tasks)
            ]
        return [future.result() for future in futures]

    def _run_worker(
        self,
        index: int,
        query: str,
        transcript: Sequence[ChatMessage],
        task: str,
        max_iter: int,
    ) -> WorkerReport:
        name = f"sub-{index}"
        if self.progress:
            self.progress.worker_started(name, task)
        history = _history_messages(transcript)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        result = agent.run(
            name=name,
            instructions=WORKER_PROMPT,
            user_input=_worker_payload(query, task),
            use_tools=True,
            history=history,
        )
        return _parse_report(task, result.text, result.citations)

    def _lead(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        reports: list[WorkerReport],
        max_iter: int,
    ) -> str:
        if self.progress:
            self.progress.synthesize_started(len(reports))
        history = _history_messages(transcript)
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
            user_input=_lead_payload(query, report_blob),
            use_tools=True,
            history=history,
        )
        return result.text


@dataclass
class ChatSession:
    app: DeepFind
    num_agent: int
    max_iter_per_agent: int
    transcript: list[ChatMessage] = field(default_factory=list)

    def ask(self, query: str) -> str:
        transcript = list(self.transcript)
        answer = self.app._run_turn(
            query=query,
            transcript=transcript,
            num_agent=self.num_agent,
            max_iter_per_agent=self.max_iter_per_agent,
        )
        self.transcript.extend(
            [
                ChatMessage(role="user", content=query),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        return answer
