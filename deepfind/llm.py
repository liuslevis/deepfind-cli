from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .config import Settings
from .json_utils import dump_json, try_load_json
from .llm_transport import RESPONSES_API, response_output_text
from .models import AgentResult
from .progress import ConsoleProgress
from .tools import Toolset

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_TAG_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_THOUGHT_TAG_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)


def _clean_model_output(text: str) -> str:
    """Remove special tags like <think>, <reasoning>, <thought> from model output.

    Some models (e.g., MiniMax-M2.7) include reasoning process in special tags
    even when instructed to output only JSON or Markdown. This function removes
    those tags to ensure clean output.
    """
    if not text:
        return text

    # Remove <think>...</think> tags
    text = _THINK_TAG_RE.sub("", text)
    # Remove <reasoning>...</reasoning> tags
    text = _REASONING_TAG_RE.sub("", text)
    # Remove <thought>...</thought> tags
    text = _THOUGHT_TAG_RE.sub("", text)

    # Clean up any extra whitespace left after tag removal
    return text.strip()


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


def _response_item_input(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else None

    item_type = str(getattr(item, "type", ""))
    if item_type == "function_call":
        return {
            "type": "function_call",
            "call_id": str(getattr(item, "call_id", "") or getattr(item, "id", "")),
            "name": str(getattr(item, "name", "")),
            "arguments": str(getattr(item, "arguments", "") or ""),
        }
    return None


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
        if getattr(self.settings, "api_mode", "") == RESPONSES_API:
            return self._run_responses(
                name,
                instructions,
                user_input,
                use_tools,
                history=history,
                tool_names=tool_names,
                max_tokens=max_tokens,
            )

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
            if self.settings.think:
                request["extra_body"] = {"think": True}
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

            response = self.client.chat.completions.create(**request)
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
                            "content": output,
                        }
                    )
                continue

            text = (message.content or "").strip()
            # Clean special tags from model output (e.g., <think>, <reasoning>)
            text = _clean_model_output(text)
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

    def _run_responses(
        self,
        name: str,
        instructions: str,
        user_input: str,
        use_tools: bool,
        history: Sequence[Mapping[str, str]] | None = None,
        tool_names: Sequence[str] | None = None,
        max_tokens: int = 1400,
    ) -> AgentResult:
        input_items: list[dict[str, Any]] = []
        if history:
            input_items.extend(
                {"role": item["role"], "content": item["content"]}
                for item in history
                if item.get("role") in {"user", "assistant"} and item.get("content")
            )
        input_items.append({"role": "user", "content": user_input})
        citations: list[str] = []

        for iteration in range(1, self.max_iter + 1):
            if self.progress:
                self.progress.iteration(name, iteration)
            request: dict[str, Any] = {
                "model": self.settings.model,
                "instructions": instructions,
                "input": list(input_items),
                "max_output_tokens": max_tokens,
            }
            if use_tools:
                tool_specs = self._responses_tool_specs(tool_names)
                if tool_specs:
                    request["tools"] = tool_specs
                    request["tool_choice"] = "auto"

            response = self.client.responses.create(**request)
            function_calls = [
                item
                for item in getattr(response, "output", None) or []
                if getattr(item, "type", "") == "function_call"
            ]
            if function_calls:
                for item in getattr(response, "output", None) or []:
                    replay_item = _response_item_input(item)
                    if replay_item is not None:
                        input_items.append(replay_item)
                for call in function_calls:
                    arguments = _parse_tool_arguments(str(getattr(call, "arguments", "") or ""))
                    call_id = str(getattr(call, "call_id", "") or getattr(call, "id", ""))
                    tool_name = str(getattr(call, "name", ""))
                    if self.progress:
                        self.progress.tool_call(name, iteration, tool_name, arguments)
                    output = self.tools.call(tool_name, arguments)
                    parsed_output = try_load_json(output)
                    if parsed_output is not None:
                        citations.extend(_extract_urls_from_value(parsed_output))
                    else:
                        citations.extend(_extract_urls_from_text(output))
                    if self.progress:
                        self.progress.tool_result(name, tool_name, output)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        }
                    )
                continue

            text = _clean_model_output(response_output_text(response))
            if not text:
                text = dump_json({"summary": "", "facts": [], "gaps": ["empty_output"]})
            if self.progress:
                self.progress.agent_done(name, iteration, text)
            final_citations = _dedupe_keep_order(
                [*citations, *_extract_urls_from_text(text)]
            )
            return AgentResult(
                name=name,
                text=text,
                citations=final_citations,
                iterations=iteration,
            )

        if self.progress:
            self.progress.agent_done(name, self.max_iter, "max_iter_reached")
        return AgentResult(
            name=name,
            text=dump_json({"summary": "", "facts": [], "gaps": ["max_iter_reached"]}),
            citations=_dedupe_keep_order(citations),
            iterations=self.max_iter,
        )

    def _responses_tool_specs(
        self,
        tool_names: Sequence[str] | None,
    ) -> list[dict[str, Any]]:
        allowed = set(tool_names) if tool_names is not None else None
        response_specs: list[dict[str, Any]] = []
        for spec in self.tools.specs():
            function = spec.get("function", {})
            name = str(function.get("name", ""))
            if not name or (allowed is not None and name not in allowed):
                continue
            response_specs.append(
                {
                    "type": "function",
                    "name": name,
                    "description": str(function.get("description", "")),
                    "parameters": function.get("parameters", {"type": "object"}),
                    "strict": False,
                }
            )
        return response_specs
