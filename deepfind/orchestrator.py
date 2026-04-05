from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from .config import Settings
from .json_utils import dump_json, try_load_json
from .llm import ResponseAgent
from .models import ChatMessage, WorkerReport
from .progress import ConsoleProgress
from .tools import Toolset

PLAN_PROMPT = (
    "You are the lead planner for an ongoing research chat. Use the prior conversation for context when needed, "
    "but focus on the latest user request. Use tools sparingly during planning when they help you discover the most "
    "important evidence paths. Prefer a two-step flow: use web_search to find candidate URLs, then use web_fetch on "
    "the most promising pages before splitting work. If the user wants an image or slides, plan only supporting "
    "research/context tasks and leave the final asset creation for the lead response. Reserve Xiaohongshu, "
    "X/Twitter, Bilibili, YouTube, and BOSS Zhipin-specific tasks for their matching platform tools. Split it into "
    "{n} distinct research tasks. Make each task specific and evidence-seeking, and include discovered URLs when "
    "helpful. JSON array only."
)
WORKER_PROMPT = (
    "You are a research worker in an ongoing chat. Use the conversation history for context when the latest request "
    "depends on earlier turns. Do the task. Use tools. For broad web research, prefer a two-step flow: use "
    "web_search to find candidate URLs, then use web_fetch to inspect the highest-value pages with a targeted prompt "
    "instead of relying only on snippets. Keep using the Xiaohongshu, X/Twitter, Bilibili, YouTube, and BOSS "
    "Zhipin-specific tools for those platforms. Use boss_search for job searches, boss_detail when you need one "
    "posting's full description, boss_chatlist when you need existing BOSS chat threads or uid values, and boss_send "
    "when you need to send a follow-up message such as asking which company a hidden-company posting belongs to. "
    "Prefer boss_detail before messaging because it may already reveal the company. If the task needs Bilibili "
    "creator discovery or channel uploads, use bili_search or bili_get_user_videos. If the task mentions Bilibili "
    "video/audio, call bili_transcribe with the URL or BVID plus a short query that captures the user's research "
    "goal; use bili_transcribe_full only when you truly need the raw transcript. If the task mentions YouTube "
    "video/audio, call youtube_transcribe with the URL or video ID before summarizing. If the latest user request "
    "asks for an image, do not call gen_img unless the assigned task explicitly asks you to produce the final image "
    "asset. If the latest user request asks for slides, do not call gen_slides unless the assigned task explicitly "
    'asks you to produce the final slide asset. JSON only: {"summary":"","facts":[{"point":"","source":""}],"gaps":[]}.'
)
SYNTHESIS_PROMPT = (
    "You are the lead synthesis coordinator in an ongoing research chat. Use the conversation history when needed, "
    "merge the worker reports, identify the strongest evidence, and fill gaps with tools when the reports are "
    "incomplete or conflicting. For broad web research, prefer the two-step flow: web_search first, then web_fetch "
    "for deep reading. Keep platform-specific work on the matching tools. JSON only: "
    '{"summary":"","evidence":[{"point":"","source":""}],"gaps":[],"sources":[],"next_steps":[]}.'
)
LEAD_PROMPT = (
    "You are the lead researcher in an ongoing chat. Use the conversation history for context and answer the latest "
    "user request using the provided synthesis JSON. Do not do more research in this stage. If tools are available, "
    "they are only for final asset creation: call gen_img exactly once when the latest user request explicitly asks "
    "for the final image asset, and call gen_slides exactly once when it explicitly asks for the final slide deck. "
    "Do not claim an asset exists unless the corresponding tool succeeds. Answer briefly with sources and mention "
    "uncertainty when the synthesis still has gaps."
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


def _synthesis_payload(query: str, reports: str) -> str:
    return f"d={date.today().isoformat()}\nlatest_user_request={query}\nreports={reports}"


def _lead_payload(query: str, synthesis: str) -> str:
    return f"d={date.today().isoformat()}\nlatest_user_request={query}\nsynthesis={synthesis}"


def _normalize_task(item: object) -> str:
    if isinstance(item, dict):
        task = item.get("task") or item.get("title") or item.get("summary")
        if task:
            return str(task).strip()
    return str(item).strip()


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
        elif isinstance(item, dict):
            candidate = item.get("value") or item.get("text") or item.get("source")
            if isinstance(candidate, str) and candidate.strip():
                items.append(candidate.strip())
    return _dedupe_keep_order(items)


def _normalize_evidence(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            point = str(item.get("point") or item.get("summary") or "").strip()
            source = str(item.get("source") or "").strip()
        else:
            point = str(item).strip()
            source = ""
        if point or source:
            evidence.append({"point": point, "source": source})
    return evidence


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _fallback_synthesis(reports: Sequence[WorkerReport]) -> dict[str, Any]:
    summaries: list[str] = []
    evidence: list[dict[str, str]] = []
    gaps: list[str] = []
    sources: list[str] = []

    for report in reports:
        summary = str(report.parsed.get("summary", "")).strip()
        if summary:
            summaries.append(summary)
        report_evidence = _normalize_evidence(report.parsed.get("facts", []))
        evidence.extend(report_evidence)
        gaps.extend(_normalize_string_list(report.parsed.get("gaps", [])))
        sources.extend(
            source
            for source in [item.get("source", "").strip() for item in report_evidence]
            if source
        )
        sources.extend(url for url in report.citations if url)

    unique_gaps = _dedupe_keep_order(gaps)
    unique_sources = _dedupe_keep_order(sources)
    next_steps = [f"Investigate gap: {gap}" for gap in unique_gaps[:5]]
    return {
        "summary": "\n".join(summaries).strip(),
        "evidence": evidence[:12],
        "gaps": unique_gaps,
        "sources": unique_sources,
        "next_steps": next_steps,
    }


def _normalize_synthesis(parsed: dict[str, Any], reports: Sequence[WorkerReport]) -> dict[str, Any]:
    fallback = _fallback_synthesis(reports)
    normalized = {
        "summary": str(parsed.get("summary", "")).strip() or fallback["summary"],
        "evidence": _normalize_evidence(parsed.get("evidence") or parsed.get("facts")) or fallback["evidence"],
        "gaps": _normalize_string_list(parsed.get("gaps")) or fallback["gaps"],
        "sources": _normalize_string_list(parsed.get("sources")),
        "next_steps": _normalize_string_list(parsed.get("next_steps")),
    }
    if not normalized["sources"]:
        normalized["sources"] = _dedupe_keep_order(
            [
                source
                for source in [item.get("source", "").strip() for item in normalized["evidence"]]
                if source
            ]
            + fallback["sources"]
        )
    if not normalized["next_steps"] and normalized["gaps"]:
        normalized["next_steps"] = [f"Investigate gap: {gap}" for gap in normalized["gaps"][:5]]
    return normalized


def _query_requests_image(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in ("image", "cover image", "illustration", "\u56fe\u7247", "\u56fe\u50cf"))


def _query_requests_slides(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in ("slides", "slide deck", "\u5e7b\u706f", "\u5e7b\u706f\u7247"))


def _lead_tool_names(query: str) -> list[str]:
    tools: list[str] = []
    if _query_requests_image(query):
        tools.append("gen_img")
    if _query_requests_slides(query):
        tools.append("gen_slides")
    return tools


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
        synthesis = self._synthesize(query, transcript, reports, max_iter_per_agent)
        answer = self._lead(query, transcript, synthesis, max_iter_per_agent).strip()
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
            use_tools=True,
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
        synthesis: dict[str, Any],
        max_iter: int,
    ) -> str:
        history = _history_messages(transcript)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        tool_names = _lead_tool_names(query)
        result = agent.run(
            name="lead-final",
            instructions=LEAD_PROMPT,
            user_input=_lead_payload(query, dump_json(synthesis)),
            use_tools=bool(tool_names),
            history=history,
            tool_names=tool_names or None,
        )
        return result.text

    def _synthesize(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        reports: list[WorkerReport],
        max_iter: int,
    ) -> dict[str, Any]:
        if self.progress:
            self.progress.synthesize_started(len(reports))
        history = _history_messages(transcript)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        report_blob = dump_json(
            [
                {
                    "task": report.task,
                    "text": report.text,
                    "summary": report.parsed.get("summary", ""),
                    "facts": report.parsed.get("facts", []),
                    "gaps": report.parsed.get("gaps", []),
                    "citations": report.citations,
                }
                for report in reports
            ]
        )
        result = agent.run(
            name="lead-synthesis",
            instructions=SYNTHESIS_PROMPT,
            user_input=_synthesis_payload(query, report_blob),
            use_tools=True,
            history=history,
        )
        parsed = try_load_json(result.text)
        if isinstance(parsed, dict):
            return _normalize_synthesis(parsed, reports)
        return _fallback_synthesis(reports)


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
