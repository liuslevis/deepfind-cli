from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .config import Settings
from .json_utils import dump_json, try_load_json
from .models import AgentResult
from .progress import ConsoleProgress
from .tools import Toolset

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_DEFAULT_TOOL_OUTPUT_CHAR_LIMIT = 16_000
_AGGRESSIVE_TOOL_OUTPUT_CHAR_LIMIT = 8_000
_DEFAULT_TEXT_MESSAGE_CHAR_LIMIT = 6_000
_AGGRESSIVE_TEXT_MESSAGE_CHAR_LIMIT = 3_000
_TRUNCATION_NOTICE = "...[truncated for model input]..."
_PRIORITY_KEYS = (
    "ok",
    "tool",
    "query",
    "ref",
    "url",
    "title",
    "name",
    "summary",
    "text",
    "content_text",
    "content_hint",
    "description",
    "snippet",
    "excerpt",
    "author",
    "username",
    "publisher",
    "source",
    "media_type",
    "stats",
    "note",
    "items",
    "results",
    "claims",
    "gaps",
    "data",
    "error_code",
    "error",
    "pages_fetched",
    "pages_requested",
    "has_more",
    "count",
    "total",
)
_PRIORITY_KEY_ORDER = {key: index for index, key in enumerate(_PRIORITY_KEYS)}


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    parsed = try_load_json(raw)
    if isinstance(parsed, dict):
        return parsed

    cleaned = raw.strip()
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index != -1]
    if starts:
        cleaned = cleaned[min(starts):]

    for _ in range(10):
        parsed = try_load_json(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if not cleaned:
            break
        cleaned = cleaned[:-1].strip()
    return {}


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        clean = item.strip().rstrip(").,")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        unique.append(clean)
    return unique


def _extract_urls_from_text(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _extract_urls_from_value(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        return _extract_urls_from_text(value)
    if isinstance(value, dict):
        for item in value.values():
            urls.extend(_extract_urls_from_value(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_urls_from_value(item))
    return urls


def _truncate_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    notice = _TRUNCATION_NOTICE
    if limit <= len(notice) + 8:
        return cleaned[:limit]
    head = max(1, limit - len(notice) - 1)
    return f"{cleaned[:head].rstrip()} {notice}"


def _prioritized_keys(value: dict[str, Any]) -> list[str]:
    return sorted(
        value.keys(),
        key=lambda key: (_PRIORITY_KEY_ORDER.get(key, len(_PRIORITY_KEY_ORDER)), key),
    )


def _compact_json_value(
    value: Any,
    *,
    aggressive: bool,
    depth: int = 0,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        limit = 220 if aggressive else 420
        if depth >= 2:
            limit = 140 if aggressive else 260
        return _truncate_text(value, limit)
    if isinstance(value, list):
        max_items = 4 if aggressive else 8
        if depth >= 2:
            max_items = 2 if aggressive else 4
        compacted = [
            _compact_json_value(item, aggressive=aggressive, depth=depth + 1)
            for item in value[:max_items]
        ]
        remaining = len(value) - max_items
        if remaining > 0:
            compacted.append({"_truncated_items": remaining})
        return compacted
    if isinstance(value, dict):
        max_keys = 10 if aggressive else 18
        if depth >= 2:
            max_keys = 6 if aggressive else 10
        result: dict[str, Any] = {}
        keys = _prioritized_keys(value)
        for key in keys[:max_keys]:
            result[key] = _compact_json_value(value[key], aggressive=aggressive, depth=depth + 1)
        remaining = len(keys) - max_keys
        if remaining > 0:
            result["_truncated_keys"] = remaining
        return result
    return _truncate_text(str(value), 120 if aggressive else 240)


def _compact_tool_output(output: str, *, aggressive: bool = False) -> str:
    limit = _AGGRESSIVE_TOOL_OUTPUT_CHAR_LIMIT if aggressive else _DEFAULT_TOOL_OUTPUT_CHAR_LIMIT
    raw = output.strip()
    if len(raw) <= limit:
        return raw

    parsed = try_load_json(raw)
    if parsed is None:
        return _truncate_text(raw, limit)

    compacted = _compact_json_value(parsed, aggressive=aggressive)
    text = dump_json(compacted)
    if len(text) <= limit:
        return text
    if not aggressive:
        return _compact_tool_output(raw, aggressive=True)

    tool_name = parsed.get("tool") if isinstance(parsed, dict) else ""
    fallback = {
        "tool": str(tool_name).strip(),
        "truncated": True,
        "original_chars": len(raw),
        "preview": _truncate_text(text, max(200, limit - 120)),
    }
    return dump_json(fallback)


def _compact_text_message(content: str, *, aggressive: bool = False) -> str:
    limit = _AGGRESSIVE_TEXT_MESSAGE_CHAR_LIMIT if aggressive else _DEFAULT_TEXT_MESSAGE_CHAR_LIMIT
    return _truncate_text(content, limit)


def _compact_messages_for_retry(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        item = dict(message)
        content = item.get("content")
        if not isinstance(content, str) or not content:
            compacted.append(item)
            continue
        if role == "tool":
            item["content"] = _compact_tool_output(content, aggressive=True)
        elif role != "system":
            item["content"] = _compact_text_message(content, aggressive=True)
        compacted.append(item)
    return compacted


def _is_input_length_error(exc: Exception) -> bool:
    text = str(exc)
    return "Range of input length should be" in text or "invalid_parameter_error" in text


@dataclass
class ResponseAgent:
    settings: Settings
    tools: Toolset
    max_iter: int
    progress: ConsoleProgress | None = None

    def __post_init__(self) -> None:
        self.client = self.settings.new_client()

    def run(
        self,
        name: str,
        instructions: str,
        user_input: str,
        use_tools: bool,
        history: Sequence[Mapping[str, str]] | None = None,
        tool_names: Sequence[str] | None = None,
        max_tokens: int = 1400,
    ) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
        ]
        if history:
            messages.extend(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in history
                if item.get("role") in {"user", "assistant"} and item.get("content")
            )
        messages.append({"role": "user", "content": user_input})
        citations: list[str] = []

        for iteration in range(1, self.max_iter + 1):
            if self.progress:
                self.progress.iteration(name, iteration)
            request: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if use_tools:
                tool_specs = self.tools.specs()
                if tool_names is not None:
                    allowed = set(tool_names)
                    tool_specs = [
                        spec
                        for spec in tool_specs
                        if spec.get("function", {}).get("name") in allowed
                    ]
                if tool_specs:
                    request["tools"] = tool_specs
                    request["tool_choice"] = "auto"
                    request["parallel_tool_calls"] = True

            try:
                response = self.client.chat.completions.create(**request)
            except Exception as exc:
                if not _is_input_length_error(exc):
                    raise
                retry_request = dict(request)
                retry_request["messages"] = _compact_messages_for_retry(request["messages"])
                response = self.client.chat.completions.create(**retry_request)
            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                normalized_tool_calls = []
                parsed_arguments: dict[str, dict[str, Any]] = {}
                for call in tool_calls:
                    parsed = _parse_tool_arguments(call.function.arguments)
                    parsed_arguments[call.id] = parsed
                    normalized_tool_calls.append(
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": dump_json(parsed),
                            },
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": normalized_tool_calls,
                    }
                )
                for call in tool_calls:
                    arguments = parsed_arguments.get(call.id, {})
                    if self.progress:
                        self.progress.tool_call(
                            name,
                            iteration,
                            call.function.name,
                            arguments,
                        )
                    output = self.tools.call(
                        call.function.name,
                        arguments,
                    )
                    parsed_output = try_load_json(output)
                    if parsed_output is not None:
                        citations.extend(_extract_urls_from_value(parsed_output))
                    else:
                        citations.extend(_extract_urls_from_text(output))
                    if self.progress:
                        self.progress.tool_result(name, call.function.name, output)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": _compact_tool_output(output),
                        }
                    )
                continue

            text = (message.content or "").strip()
            if not text:
                text = dump_json({"summary": "", "facts": [], "gaps": ["empty_output"]})
            if self.progress:
                self.progress.agent_done(name, iteration, text)
            final_citations = _dedupe_keep_order([*citations, *_extract_urls_from_text(text)])
            return AgentResult(name=name, text=text, citations=final_citations, iterations=iteration)

        if self.progress:
            self.progress.agent_done(name, self.max_iter, "max_iter_reached")
        return AgentResult(
            name=name,
            text=dump_json({"summary": "", "facts": [], "gaps": ["max_iter_reached"]}),
            citations=_dedupe_keep_order(citations),
            iterations=self.max_iter,
        )
