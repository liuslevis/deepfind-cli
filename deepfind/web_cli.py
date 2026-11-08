"""Run a one-shot query that lands as a new chat in the DeepFind web UI.

The web UI (web/ + Electron shell) reads its chat list from the ``/api/chats``
endpoints of the FastAPI backend. This module drives the same HTTP API a
browser session would use, so a terminal query creates a real chat with a user
message and a streamed assistant answer that the UI can render directly.

Authentication mirrors the UI: the backend requires ``DEEPFIND_WEB_TOKEN``
whenever it was started with that variable set, so the client resolves the
token from the environment, then from the project ``.env`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import _load_dotenv
from .web_models import ChatMode, DEFAULT_MODEL_TARGET, ModelTarget, ResearchMode

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CHAT_MODE: ChatMode = "fast"
DEFAULT_RESEARCH_MODE: ResearchMode = "chat"
_REQUEST_TIMEOUT = 30.0


class WebClientError(RuntimeError):
    """Raised when the web backend rejects a request or cannot be reached."""


@dataclass(frozen=True)
class CreatedChat:
    id: str
    title: str


def resolve_token(explicit: str | None = None) -> str:
    """Return the API token the backend expects, or an empty string."""
    if explicit is not None:
        return explicit.strip()
    return (os.environ.get("DEEPFIND_WEB_TOKEN") or "").strip()


def resolve_base_url(explicit: str | None = None) -> str:
    configured = explicit or os.environ.get("DEEPFIND_WEB_URL") or DEFAULT_BASE_URL
    return configured.strip().rstrip("/")


def _load_dotenv_if_needed() -> None:
    """Populate DEEPFIND_WEB_TOKEN from .env when the environment lacks it."""
    if (os.environ.get("DEEPFIND_WEB_TOKEN") or "").strip():
        return
    _load_dotenv()


def _decode_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        body = ""
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body.strip() or f"HTTP {exc.code}"
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)
        message = payload.get("message") if isinstance(payload, dict) else None
        if isinstance(message, str) and message.strip():
            return message.strip()
        return body.strip()
    return f"HTTP {exc.code} {exc.reason}"


def _request(
    method: str,
    url: str,
    *,
    token: str,
    payload: dict | None = None,
    accept: str = "application/json",
    timeout: float = _REQUEST_TIMEOUT,
):
    headers = {"Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 401:
            raise WebClientError(
                "unauthorized: the backend rejected DEEPFIND_WEB_TOKEN. "
                "Start the backend with the same token that is set in .env."
            ) from exc
        raise WebClientError(_decode_error(exc)) from exc
    except URLError as exc:
        raise WebClientError(
            f"cannot reach the DeepFind web backend at {url} ({exc.reason}). "
            "Start it with `deepfind-web` or `uv run deepfind-web`."
        ) from exc
    except TimeoutError as exc:  # pragma: no cover - defensive
        raise WebClientError(f"request to {url} timed out") from exc


def check_health(base_url: str) -> dict:
    with _request("GET", f"{base_url}/api/health", token="") as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise WebClientError("backend returned a non-JSON health response") from exc


def create_chat(base_url: str, *, token: str, title: str | None = None) -> CreatedChat:
    body = {"title": title} if title else {}
    with _request("POST", f"{base_url}/api/chats", token=token, payload=body) as response:
        payload = response.read().decode("utf-8")
    try:
        chat = json.loads(payload)["chat"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WebClientError("backend returned an unexpected create-chat response") from exc
    return CreatedChat(id=str(chat["id"]), title=str(chat.get("title") or "New chat"))


def _parse_event_block(block: str) -> dict | None:
    event_type = "message"
    data_lines: list[str] = []
    for raw_line in block.split("\n"):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if not data_lines:
        return None
    try:
        decoded = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    return {
        "type": event_type,
        "timestamp": decoded.get("timestamp", ""),
        "data": decoded.get("data") or {},
    }


def iter_sse_events(lines: Iterable[bytes]) -> Iterable[dict]:
    """Yield decoded SSE events from a chunked byte stream."""
    buffer = ""
    for raw in lines:
        buffer += raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, _, buffer = buffer.partition("\n\n")
            event = _parse_event_block(block)
            if event is not None:
                yield event
    leftover = buffer.strip()
    if leftover:
        event = _parse_event_block(leftover)
        if event is not None:
            yield event


def stream_message(
    base_url: str,
    chat_id: str,
    content: str,
    *,
    token: str,
    mode: ChatMode = DEFAULT_CHAT_MODE,
    model_target: ModelTarget = DEFAULT_MODEL_TARGET,
    research_mode: ResearchMode = DEFAULT_RESEARCH_MODE,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Send a message and consume the SSE stream; return the final ``done`` data."""
    payload = {
        "content": content,
        "mode": mode,
        "model_target": model_target,
        "research_mode": research_mode,
    }
    url = f"{base_url}/api/chats/{chat_id}/messages/stream"
    done_data: dict = {}
    # Research turns can run for many minutes; disable the read timeout.
    with _request(
        "POST",
        url,
        token=token,
        payload=payload,
        accept="text/event-stream",
        timeout=None,
    ) as response:
        for event in iter_sse_events(response):
            if on_event is not None:
                on_event(event)
            if event["type"] == "done":
                done_data = event.get("data") or {}
            elif event["type"] == "error":
                detail = event.get("data") or {}
                message = detail.get("message") or detail.get("error") or "the backend reported an error"
                raise WebClientError(str(message))
    return done_data


def _progress_printer(stderr: TextIO) -> Callable[[dict], None]:
    """Mirror the useful server-side console lines to stderr while waiting."""

    def emit(event: dict) -> None:
        kind = event.get("type", "")
        data = event.get("data") or {}
        if kind == "console_line":
            text = str(data.get("text") or "").rstrip()
            if text:
                print(f"  {text}", file=stderr, flush=True)
        elif kind == "answer_final":
            print("  answer ready", file=stderr, flush=True)

    return emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfind-chat",
        description=(
            "Run one query and store it as a new chat in the DeepFind web UI. "
            "Defaults: agents/chat mode, DeepSeek model."
        ),
    )
    parser.add_argument("query", help="the question or instruction to run")
    parser.add_argument(
        "--url",
        default=None,
        help=f"web backend base URL (default: $DEEPFIND_WEB_URL or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="API token (default: $DEEPFIND_WEB_TOKEN, then .env)",
    )
    parser.add_argument("--title", default=None, help="chat title (default: derived from the query)")
    parser.add_argument(
        "--mode",
        choices=["fast", "expert"],
        default=DEFAULT_CHAT_MODE,
        help=f"chat mode (default: {DEFAULT_CHAT_MODE})",
    )
    parser.add_argument(
        "--model",
        choices=["deepseek", "qwen", "mimo", "minimax", "glm", "gpu"],
        default=DEFAULT_MODEL_TARGET,
        help=f"model target (default: {DEFAULT_MODEL_TARGET})",
    )
    parser.add_argument(
        "--research-mode",
        choices=["chat", "research", "deep_research"],
        default=DEFAULT_RESEARCH_MODE,
        help=f"research depth (default: {DEFAULT_RESEARCH_MODE})",
    )
    parser.add_argument("--quiet", action="store_true", help="hide progress lines")
    parser.add_argument("--json", action="store_true", help="print a JSON result instead of the answer")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    _load_dotenv_if_needed()

    base_url = resolve_base_url(args.url)
    token = resolve_token(args.token)

    try:
        health = check_health(base_url)
        if health.get("requires_token") and not token:
            raise WebClientError(
                "the backend requires a token but DEEPFIND_WEB_TOKEN is not set. "
                "Set it in .env or pass --token."
            )
        chat = create_chat(base_url, token=token, title=args.title)
        print(f"Chat created: {chat.id}", file=stderr, flush=True)
        done = stream_message(
            base_url,
            chat.id,
            args.query,
            token=token,
            mode=args.mode,
            model_target=args.model,
            research_mode=args.research_mode,
            on_event=None if args.quiet else _progress_printer(stderr),
        )
    except WebClientError as exc:
        print(f"error: {exc}", file=stderr)
        return 1

    assistant_id = str(done.get("assistant_message_id") or "")
    if args.json:
        print(
            json.dumps(
                {
                    "chat_id": chat.id,
                    "base_url": base_url,
                    "assistant_message_id": assistant_id,
                },
                ensure_ascii=False,
            ),
            file=stdout,
        )
    else:
        print(f"Chat ready in the UI: {chat.id}", file=stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
