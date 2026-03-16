from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .json_utils import dump_json, try_load_json
from .models import AgentResult
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

    def __post_init__(self) -> None:
        self.client = self.settings.new_client()

    def run(self, name: str, instructions: str, user_input: str, use_tools: bool) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ]

        for iteration in range(1, self.max_iter + 1):
            request: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1400,
            }
            if use_tools:
                request["tools"] = self.tools.specs()
                request["tool_choice"] = "auto"
                request["parallel_tool_calls"] = True

            response = self.client.chat.completions.create(**request)
            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    output = self.tools.call(
                        call.function.name,
                        _parse_tool_arguments(call.function.arguments),
                    )
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
            return AgentResult(name=name, text=text, citations=[], iterations=iteration)

        return AgentResult(
            name=name,
            text=dump_json({"summary": "", "facts": [], "gaps": ["max_iter_reached"]}),
            citations=[],
            iterations=self.max_iter,
        )
