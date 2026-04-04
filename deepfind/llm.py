from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import Settings
from .json_utils import dump_json, try_load_json
from .models import AgentResult
from .progress import ConsoleProgress
from .tools import Toolset


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

        for iteration in range(1, self.max_iter + 1):
            if self.progress:
                self.progress.iteration(name, iteration)
            request: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": 1400,
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
            if not text:
                text = dump_json({"summary": "", "facts": [], "gaps": ["empty_output"]})
            if self.progress:
                self.progress.agent_done(name, iteration, text)
            return AgentResult(name=name, text=text, citations=[], iterations=iteration)

        if self.progress:
            self.progress.agent_done(name, self.max_iter, "max_iter_reached")
        return AgentResult(
            name=name,
            text=dump_json({"summary": "", "facts": [], "gaps": ["max_iter_reached"]}),
            citations=[],
            iterations=self.max_iter,
        )
