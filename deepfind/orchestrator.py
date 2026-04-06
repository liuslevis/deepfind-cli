from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
import re
from typing import Any, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .chat_store import utc_now
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
    'asks you to produce the final slide asset. JSON only: {"summary":"","claims":[{"text":"","citations":[],"confidence":"medium"}],"gaps":[]}.'
)
SYNTHESIS_PROMPT = (
    "You are the lead synthesis coordinator in an ongoing research chat. Use the conversation history when needed, "
    "merge the worker reports, identify the strongest evidence, and fill gaps with tools when the reports are "
    "incomplete or conflicting. For broad web research, prefer the two-step flow: web_search first, then web_fetch "
    "for deep reading. Keep platform-specific work on the matching tools. JSON only: "
    '{"overview_md":"","key_points":[{"text":"","citations":[],"confidence":"medium"}],"disagreements":[],"gaps":[],"next_steps":[]}.'
)
LEAD_PROMPT = (
    "You are the lead researcher in an ongoing chat. Use the conversation history for context and answer the latest "
    "user request using the provided synthesis JSON. Do not do more research in this stage. If tools are available, "
    "they are only for final asset creation: call gen_img exactly once when the latest user request explicitly asks "
    "for the final image asset, and call gen_slides exactly once when it explicitly asks for the final slide deck. "
    "Do not claim an asset exists unless the corresponding tool succeeds. Write concise Markdown that serves as the "
    "lead overview, mention uncertainty when the synthesis still has gaps, and rely on the provided synthesis instead "
    "of adding new facts."
)
LONG_REPORT_LEAD_PROMPT = (
    "You are the lead researcher in an ongoing chat. Use the conversation history for context and answer the latest "
    "user request using the provided synthesis JSON. Do not do more research in this stage. If tools are available, "
    "they are only for final asset creation: call gen_img exactly once when the latest user request explicitly asks "
    "for the final image asset, and call gen_slides exactly once when it explicitly asks for the final slide deck. "
    "Do not claim an asset exists unless the corresponding tool succeeds. Write Thesis-like Markdown that serves as the "
    "lead overview, including ## Abstract and ## Reference. In Reference, list only URLs that already appear in the "
    "synthesis JSON citations and do not invent or fetch new links."
)
FORMAT_FOLLOWUP_PROMPT = (
    "You are the lead editor in an ongoing chat. The user is asking you to transform the prior assistant answer into "
    "a new presentation format. Work only from the provided prior assistant answer and the latest user request. Do "
    "not do new research, do not call tools, and do not add facts that are not already present in the provided "
    "answer. Preserve links, names, and numbers when possible. If the user asks for a table, output a Markdown "
    "table. If the user asks for translation, translate only the provided content. If the prior answer lacks enough "
    "detail for the requested transformation, say that briefly instead of inventing content."
)

_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "gbraid",
        "wbraid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "ref_src",
        "srsltid",
    }
)
_ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})
_DEFAULT_MAX_TOKENS = 1400
_LONG_REPORT_LEAD_MAX_TOKENS = 3200
_REFERENCE_SECTION_RE = re.compile(r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:Reference|references)\s*$")
_QUERY_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_FORMAT_FOLLOWUP_MARKERS = (
    "表格",
    "表",
    "table",
    "markdown table",
    "列表",
    "清单",
    "list",
    "bullet",
    "要点",
    "提纲",
    "outline",
    "翻译",
    "translate",
    "英文版",
    "中文版",
    "改写",
    "重写",
    "rewrite",
    "rephrase",
    "润色",
    "polish",
    "精简",
    "简化",
    "shorten",
    "simplify",
    "扩写",
    "expand",
    "整理",
    "归纳",
    "格式化",
    "format",
    "改成",
    "转成",
    "转为",
    "json",
    "yaml",
    "csv",
)
_FORMAT_FOLLOWUP_DISQUALIFIERS = (
    "搜索",
    "查一下",
    "查询",
    "research",
    "search",
    "find",
    "latest",
    "news",
    "最新",
    "今天",
    "today",
    "重新研究",
    "再搜",
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


def _format_followup_payload(query: str, prior_answer: str, prior_user_request: str = "") -> str:
    payload = {
        "d": date.today().isoformat(),
        "latest_user_request": query,
        "prior_user_request": prior_user_request,
        "prior_assistant_answer": prior_answer,
    }
    return dump_json(payload)


def _normalize_task(item: object) -> str:
    if isinstance(item, dict):
        task = item.get("task") or item.get("title") or item.get("summary")
        if task:
            return str(task).strip()
    return str(item).strip()


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


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


def _normalize_confidence(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOWED_CONFIDENCE:
            return normalized
    return "medium"


def _normalize_url_list(value: object) -> list[str]:
    urls: list[str] = []
    if not isinstance(value, list):
        return urls
    for item in value:
        if isinstance(item, str) and item.strip():
            urls.append(item.strip())
        elif isinstance(item, dict):
            candidate = item.get("url") or item.get("source") or item.get("value")
            if isinstance(candidate, str) and candidate.strip():
                urls.append(candidate.strip())
    return urls


def _normalize_claims(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    claims: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("point") or item.get("summary") or "").strip()
            raw_citations = item.get("citations") or item.get("sources")
            citations = _normalize_url_list(raw_citations)
            source = item.get("source")
            if not citations and isinstance(source, str) and source.strip():
                citations = [source.strip()]
            confidence = _normalize_confidence(item.get("confidence"))
        else:
            text = str(item).strip()
            citations = []
            confidence = "medium"
        if text:
            claims.append(
                {
                    "text": text,
                    "citations": citations,
                    "confidence": confidence,
                }
            )
    return claims


def _normalize_worker_payload(parsed: dict[str, Any], text: str) -> dict[str, Any]:
    summary = str(parsed.get("summary") or "").strip()
    claims = _normalize_claims(parsed.get("claims") or parsed.get("facts"))[:5]
    gaps = _normalize_string_list(parsed.get("gaps"))[:3]
    if not summary and not claims and text.strip():
        summary = text.strip()
    return {
        "summary": summary,
        "claims": claims,
        "gaps": gaps,
    }


def _fallback_synthesis(reports: Sequence[WorkerReport]) -> dict[str, Any]:
    summaries: list[str] = []
    key_points: list[dict[str, Any]] = []
    gaps: list[str] = []

    for report in reports:
        summary = str(report.parsed.get("summary", "")).strip()
        if summary:
            summaries.append(summary)
        key_points.extend(_normalize_claims(report.parsed.get("claims") or report.parsed.get("facts")))
        gaps.extend(_normalize_string_list(report.parsed.get("gaps", [])))

    unique_gaps = _dedupe_keep_order(gaps)
    next_steps = [f"Investigate gap: {gap}" for gap in unique_gaps[:5]]
    overview_md = "\n\n".join(summaries).strip()
    if not overview_md and key_points:
        overview_md = "\n".join(f"- {claim['text']}" for claim in key_points[:5])
    return {
        "overview_md": overview_md,
        "key_points": key_points[:8],
        "disagreements": [],
        "gaps": unique_gaps,
        "next_steps": next_steps,
    }


def _normalize_synthesis(parsed: dict[str, Any], reports: Sequence[WorkerReport]) -> dict[str, Any]:
    fallback = _fallback_synthesis(reports)
    next_steps = _normalize_string_list(parsed.get("next_steps"))
    return {
        "overview_md": str(parsed.get("overview_md") or parsed.get("summary") or "").strip()
        or fallback["overview_md"],
        "key_points": _normalize_claims(
            parsed.get("key_points") or parsed.get("claims") or parsed.get("evidence") or parsed.get("facts")
        )[:8]
        or fallback["key_points"],
        "disagreements": _normalize_string_list(parsed.get("disagreements"))[:5],
        "gaps": _normalize_string_list(parsed.get("gaps")) or fallback["gaps"],
        "next_steps": next_steps[:5] or fallback["next_steps"],
    }


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


def _latest_message_content(transcript: Sequence[ChatMessage], role: str) -> str:
    for message in reversed(transcript):
        if message.role == role and message.content.strip():
            return message.content.strip()
    return ""


def _should_shortcut_format_follow_up(query: str, transcript: Sequence[ChatMessage]) -> bool:
    normalized = " ".join(query.lower().split())
    if not normalized or len(normalized) > 160:
        return False
    if _query_requests_image(query) or _query_requests_slides(query):
        return False
    if _QUERY_URL_RE.search(query):
        return False
    if not _latest_message_content(transcript, "assistant"):
        return False
    if any(marker in normalized for marker in _FORMAT_FOLLOWUP_DISQUALIFIERS):
        return False
    return any(marker in normalized for marker in _FORMAT_FOLLOWUP_MARKERS)


def _parse_report(agent_id: str, task: str, text: str, citations: list[str]) -> WorkerReport:
    parsed = try_load_json(text)
    if not isinstance(parsed, dict):
        parsed = {
            "summary": text.strip(),
            "claims": [],
            "gaps": ["non_json_output"],
        }
    else:
        parsed = _normalize_worker_payload(parsed, text)
    return WorkerReport(task=task, text=text, citations=citations, parsed=parsed, agent_id=agent_id)


def _answer_from_envelope(envelope: dict[str, Any]) -> str:
    lead = envelope.get("lead")
    if isinstance(lead, dict):
        return str(lead.get("overview_md") or "").strip()
    return ""


def _lead_instructions(long_report_mode: bool) -> str:
    return LONG_REPORT_LEAD_PROMPT if long_report_mode else LEAD_PROMPT


def _lead_max_tokens(long_report_mode: bool) -> int:
    return _LONG_REPORT_LEAD_MAX_TOKENS if long_report_mode else _DEFAULT_MAX_TOKENS


def _has_reference_section(text: str) -> bool:
    return bool(_REFERENCE_SECTION_RE.search(text or ""))


def _reference_urls_from_envelope(envelope: dict[str, Any]) -> list[str]:
    citations = envelope.get("citations_dedup")
    if not isinstance(citations, list):
        return []
    urls: list[str] = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        url = str(item.get("canonical_url") or item.get("url") or "").strip()
        if url:
            urls.append(url)
    return _dedupe_keep_order(urls)


def _finalize_turn_envelope(envelope: dict[str, Any], *, long_report_mode: bool) -> dict[str, Any]:
    if not long_report_mode:
        return envelope
    lead = envelope.get("lead")
    if not isinstance(lead, dict):
        return envelope
    overview = str(lead.get("overview_md") or "").strip()
    if _has_reference_section(overview):
        return envelope
    reference_urls = _reference_urls_from_envelope(envelope)
    if not reference_urls:
        return envelope
    reference_block = "## Reference\n\n" + "\n".join(f"- {url}" for url in reference_urls)
    lead["overview_md"] = f"{overview}\n\n{reference_block}".strip() if overview else reference_block
    return envelope


def _is_tracking_query_key(key: str) -> bool:
    lowered = key.strip().lower()
    return lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS


def _canonicalize_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if not parsed.scheme or not parsed.netloc:
        return raw

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return raw

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        auth = f"{auth}@"

    netloc = f"{auth}{host}"
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    filtered_query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not _is_tracking_query_key(key)
        ],
        doseq=True,
    )
    return urlunsplit((scheme, netloc, path, filtered_query, ""))


@dataclass
class _CitationCollector:
    dedup_by_canonical: dict[str, dict[str, str]] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    next_raw_id: int = 1
    next_dedup_id: int = 1

    def record(
        self,
        raw_url: str,
        *,
        source_agent: str,
        source_section: str,
        source_index: int,
    ) -> str:
        cleaned = raw_url.strip()
        if not cleaned:
            return ""
        canonical = _canonicalize_url(cleaned)
        if not canonical:
            return ""
        citation = self.dedup_by_canonical.get(canonical)
        if citation is None:
            citation = {
                "id": f"c{self.next_dedup_id}",
                "canonical_url": canonical,
                "url": cleaned,
                "title": "",
                "publisher": "",
            }
            self.dedup_by_canonical[canonical] = citation
            self.next_dedup_id += 1
        self.citations.append(
            {
                "id": f"r{self.next_raw_id}",
                "dedup_id": citation["id"],
                "url": cleaned,
                "title": citation["title"],
                "publisher": citation["publisher"],
                "source_agent": source_agent,
                "source_section": source_section,
                "source_index": source_index,
            }
        )
        self.next_raw_id += 1
        return str(citation["id"])

    def citations_dedup(self) -> list[dict[str, str]]:
        return list(self.dedup_by_canonical.values())


def _attach_citation_ids(
    claims: list[dict[str, Any]],
    collector: _CitationCollector,
    *,
    source_agent: str,
    source_section: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        dedup_ids: list[str] = []
        for raw_url in claim.get("citations", []):
            dedup_id = collector.record(
                str(raw_url),
                source_agent=source_agent,
                source_section=source_section,
                source_index=index,
            )
            if dedup_id:
                dedup_ids.append(dedup_id)
        normalized.append(
            {
                "text": str(claim.get("text") or "").strip(),
                "citation_ids": _dedupe_keep_order(dedup_ids),
                "confidence": _normalize_confidence(claim.get("confidence")),
            }
        )
    return [claim for claim in normalized if claim["text"]]


def _build_turn_envelope(
    query: str,
    lead_overview: str,
    synthesis: dict[str, Any],
    reports: Sequence[WorkerReport],
    num_agent: int,
    max_iter_per_agent: int,
) -> dict[str, Any]:
    collector = _CitationCollector()
    agents: list[dict[str, Any]] = []

    for report in reports:
        source_agent = report.agent_id or "worker"
        claims = _attach_citation_ids(
            _normalize_claims(report.parsed.get("claims") or report.parsed.get("facts")),
            collector,
            source_agent=source_agent,
            source_section="claim",
        )
        agents.append(
            {
                "agent_id": source_agent,
                "task": report.task,
                "summary": str(report.parsed.get("summary") or "").strip(),
                "claims": claims[:5],
                "gaps": _normalize_string_list(report.parsed.get("gaps"))[:3],
            }
        )

    lead = {
        "overview_md": lead_overview.strip() or str(synthesis.get("overview_md") or "").strip(),
        "key_points": _attach_citation_ids(
            _normalize_claims(synthesis.get("key_points") or synthesis.get("evidence") or synthesis.get("facts"))[:8],
            collector,
            source_agent="lead",
            source_section="key_point",
        ),
        "disagreements": _normalize_string_list(synthesis.get("disagreements"))[:5],
        "next_steps": _normalize_string_list(synthesis.get("next_steps"))[:5],
    }

    return {
        "version": "research.v1",
        "query": query,
        "lead": lead,
        "agents": agents,
        "citations": collector.citations,
        "citations_dedup": collector.citations_dedup(),
        "meta": {
            "num_agents": num_agent,
            "max_iter_per_agent": max_iter_per_agent,
            "generated_at": utc_now(),
        },
    }
 

class DeepFind:
    def __init__(
        self,
        settings: Settings | None = None,
        progress: ConsoleProgress | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.tools = Toolset(self.settings)
        self.progress = progress

    def session(
        self,
        num_agent: int,
        max_iter_per_agent: int,
        long_report_mode: bool = False,
    ) -> "ChatSession":
        num_agent, max_iter_per_agent = self._validated_run_args(num_agent, max_iter_per_agent)
        return ChatSession(
            app=self,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
            long_report_mode=long_report_mode,
        )

    def run(
        self,
        query: str,
        num_agent: int,
        max_iter_per_agent: int,
        long_report_mode: bool = False,
    ) -> str:
        return self.session(num_agent, max_iter_per_agent, long_report_mode=long_report_mode).ask(query)

    def run_detailed(
        self,
        query: str,
        num_agent: int,
        max_iter_per_agent: int,
        long_report_mode: bool = False,
    ) -> dict[str, Any]:
        return self.session(num_agent, max_iter_per_agent, long_report_mode=long_report_mode).ask_detailed(query)

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
        long_report_mode: bool = False,
    ) -> str:
        envelope, _ = self._run_turn_structured(
            query=query,
            transcript=transcript,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
            long_report_mode=long_report_mode,
        )
        return _answer_from_envelope(envelope)

    def _run_turn_detailed(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        num_agent: int,
        max_iter_per_agent: int,
        long_report_mode: bool = False,
    ) -> tuple[str, list[WorkerReport]]:
        envelope, reports = self._run_turn_structured(
            query=query,
            transcript=transcript,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
            long_report_mode=long_report_mode,
        )
        return _answer_from_envelope(envelope), reports

    def _run_turn_structured(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        num_agent: int,
        max_iter_per_agent: int,
        long_report_mode: bool = False,
    ) -> tuple[dict[str, Any], list[WorkerReport]]:
        if self.progress:
            self.progress.run_started(query, num_agent, max_iter_per_agent)
        if _should_shortcut_format_follow_up(query, transcript):
            envelope = self._format_follow_up(
                query,
                transcript,
                num_agent=num_agent,
                max_iter=max_iter_per_agent,
            )
            return envelope, []
        tasks = self._plan(query, transcript, num_agent, max_iter_per_agent)
        reports = self._run_workers(query, transcript, tasks, max_iter_per_agent)
        synthesis = self._synthesize(query, transcript, reports, max_iter_per_agent)
        lead_overview = self._lead(
            query,
            transcript,
            synthesis,
            max_iter_per_agent,
            long_report_mode=long_report_mode,
        ).strip()
        envelope = _build_turn_envelope(
            query=query,
            lead_overview=lead_overview,
            synthesis=synthesis,
            reports=reports,
            num_agent=num_agent,
            max_iter_per_agent=max_iter_per_agent,
        )
        _finalize_turn_envelope(envelope, long_report_mode=long_report_mode)
        return envelope, reports

    def _format_follow_up(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        *,
        num_agent: int,
        max_iter: int,
    ) -> dict[str, Any]:
        prior_answer = _latest_message_content(transcript, "assistant")
        prior_user_request = _latest_message_content(transcript, "user")
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        result = agent.run(
            name="lead-format",
            instructions=FORMAT_FOLLOWUP_PROMPT,
            user_input=_format_followup_payload(query, prior_answer, prior_user_request),
            use_tools=False,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )
        synthesis = {
            "overview_md": prior_answer,
            "key_points": [],
            "disagreements": [],
            "gaps": [],
            "next_steps": [],
        }
        envelope = _build_turn_envelope(
            query=query,
            lead_overview=result.text.strip(),
            synthesis=synthesis,
            reports=[],
            num_agent=num_agent,
            max_iter_per_agent=max_iter,
        )
        envelope["meta"]["shortcut"] = "format_follow_up"
        return envelope

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
        return _parse_report(name, task, result.text, result.citations)

    def _lead(
        self,
        query: str,
        transcript: Sequence[ChatMessage],
        synthesis: dict[str, Any],
        max_iter: int,
        long_report_mode: bool = False,
    ) -> str:
        history = _history_messages(transcript)
        agent = ResponseAgent(self.settings, self.tools, max_iter=max_iter, progress=self.progress)
        tool_names = _lead_tool_names(query)
        result = agent.run(
            name="lead-final",
            instructions=_lead_instructions(long_report_mode),
            user_input=_lead_payload(query, dump_json(synthesis)),
            use_tools=bool(tool_names),
            history=history,
            tool_names=tool_names or None,
            max_tokens=_lead_max_tokens(long_report_mode),
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
                    "agent_id": report.agent_id,
                    "task": report.task,
                    "text": report.text,
                    "summary": report.parsed.get("summary", ""),
                    "claims": report.parsed.get("claims", []),
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
    long_report_mode: bool = False
    transcript: list[ChatMessage] = field(default_factory=list)

    def _run_and_store(self, query: str) -> dict[str, Any]:
        transcript = list(self.transcript)
        envelope, _ = self.app._run_turn_structured(
            query=query,
            transcript=transcript,
            num_agent=self.num_agent,
            max_iter_per_agent=self.max_iter_per_agent,
            long_report_mode=self.long_report_mode,
        )
        answer = _answer_from_envelope(envelope)
        self.transcript.extend(
            [
                ChatMessage(role="user", content=query),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        return envelope

    def ask(self, query: str) -> str:
        envelope = self._run_and_store(query)
        return _answer_from_envelope(envelope)

    def ask_detailed(self, query: str) -> dict[str, Any]:
        return self._run_and_store(query)

