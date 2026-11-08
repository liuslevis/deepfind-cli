from __future__ import annotations

from typing import Any


CHAT_COMPLETIONS_API = "chat_completions"
RESPONSES_API = "responses"

_MESSAGE_TEXT_PART_TYPES = frozenset({"output_text", "text", "input_text"})


class TruncatedCompletionError(RuntimeError):
    """Raised when a model response was cut off before producing an answer."""


def _incomplete_reason(response: Any) -> str:
    if str(getattr(response, "status", "") or "") != "incomplete":
        return ""
    details = getattr(response, "incomplete_details", None)
    return str(getattr(details, "reason", "") or "incomplete")


def response_output_text(response: Any) -> str:
    """Return the assistant message text, never the model's reasoning.

    Reasoning items also expose ``text`` content parts, so they must be filtered
    out explicitly. Otherwise a response truncated during reasoning (which has no
    message item at all) would surface raw chain-of-thought as the answer.
    """

    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        item_type = str(getattr(item, "type", "") or "")
        if item_type and item_type != "message":
            continue
        for content in getattr(item, "content", None) or []:
            content_type = str(getattr(content, "type", "") or "")
            if content_type and content_type not in _MESSAGE_TEXT_PART_TYPES:
                continue
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
        response = client.responses.create(**request)
        text = response_output_text(response)
        if not text:
            reason = _incomplete_reason(response)
            if reason:
                raise TruncatedCompletionError(
                    f"model response was cut off before answering ({reason}); "
                    f"max_output_tokens={max_output_tokens}"
                )
        return text

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
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text and str(getattr(choice, "finish_reason", "") or "") == "length":
        raise TruncatedCompletionError(
            f"model response was cut off before answering (length); max_tokens={max_output_tokens}"
        )
    return text
