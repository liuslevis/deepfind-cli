from __future__ import annotations

from typing import Any


CHAT_COMPLETIONS_API = "chat_completions"
RESPONSES_API = "responses"


def response_output_text(response: Any) -> str:
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def complete_text(
    client: Any,
    *,
    api_mode: str,
    model: str,
    instructions: str,
    user_input: str,
    max_output_tokens: int,
    timeout: int | None = None,
) -> str:
    if api_mode == RESPONSES_API:
        request: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": max_output_tokens,
        }
        if timeout is not None:
            request["timeout"] = timeout
        return response_output_text(client.responses.create(**request))

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ],
        "max_tokens": max_output_tokens,
    }
    if timeout is not None:
        request["timeout"] = timeout
    response = client.chat.completions.create(**request)
    return (response.choices[0].message.content or "").strip()
