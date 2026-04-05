from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from threading import Thread
from urllib.parse import quote
from uuid import uuid4

from .chat_store import ChatStore, repo_root, summarize_text, utc_now
from .config import Settings
from .json_utils import try_load_json
from .models import ChatMessage, WorkerReport
from .orchestrator import DeepFind
from .tools import Toolset
from .web_models import ArtifactKind, ArtifactLink, ChatMode, TurnResult, WebChatDetail, WebMessage
from .web_progress import ToolObservation, WebProgress

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_DEFAULT_MAX_ITER_PER_AGENT = 50
_LIST_TOOL_COMMAND = "/list-tool"
_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    (_LIST_TOOL_COMMAND, "List all available tools and their descriptions."),
)


def mode_to_agent_count(mode: ChatMode) -> int:
    return 1 if mode == "fast" else 4


def chunk_text(text: str, width: int = 180) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + width] for index in range(0, len(text), width)]


def _dedupe_keep_order(items: list[str]) -> list[str]:
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


def _extract_urls_from_value(value: object) -> list[str]:
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


def _path_from_value(value: object, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            found.append(raw)
        for item in value.values():
            found.extend(_path_from_value(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_path_from_value(item, key))
    return found


@lru_cache(maxsize=1)
def _tool_catalog() -> tuple[tuple[str, str], ...]:
    catalog: list[tuple[str, str]] = []
    for item in Toolset(Settings(api_key="web")).specs():
        if not isinstance(item, dict):
            continue
        function_spec = item.get("function")
        if not isinstance(function_spec, dict):
            continue
        name = str(function_spec.get("name", "")).strip()
        description = str(function_spec.get("description", "")).strip()
        if name and description:
            catalog.append((name, description))
    return tuple(catalog)


def _tool_catalog_markdown() -> str:
    tools = _tool_catalog()
    if not tools:
        return "No tools are currently available."
    lines = [f"- `{name}`: {description}" for name, description in tools]
    return "Available tools:\n\n" + "\n".join(lines)


def _unknown_command_markdown(query: str) -> str:
    lines = [f"- `{name}`: {description}" for name, description in _SLASH_COMMANDS]
    return (
        f"Unknown slash command `{query}`.\n\n"
        "Available slash commands:\n\n"
        + "\n".join(lines)
    )


class DeepFindWebService:
    def __init__(
        self,
        store: ChatStore | None = None,
        *,
        app_factory=None,
        max_iter_per_agent: int = _DEFAULT_MAX_ITER_PER_AGENT,
    ) -> None:
        self.store = store or ChatStore()
        self.app_factory = app_factory or (lambda progress: DeepFind(progress=progress))
        self.max_iter_per_agent = max_iter_per_agent
        self._repo_root = repo_root()

    def list_chats(self):
        return self.store.list_chats()

    def create_chat(self, title: str | None = None) -> WebChatDetail:
        return self.store.create_chat(title=title)

    def get_chat(self, chat_id: str) -> WebChatDetail:
        return self.store.get_chat(chat_id)

    def delete_chat(self, chat_id: str) -> None:
        self.store.delete_chat(chat_id)

    def stream_message(self, chat_id: str, content: str, mode: ChatMode):
        query = content.strip()
        if not query:
            raise ValueError("message content must not be empty")

        chat = self.get_chat(chat_id)
        prior_transcript = self._messages_to_transcript(chat.messages)
        updated_chat = chat.model_copy(deep=True)
        user_message = WebMessage(
            id=f"msg_{uuid4().hex}",
            role="user",
            content=query,
            created_at=utc_now(),
            mode=mode,
        )
        updated_chat.messages.append(user_message)
        updated_chat.title = self._next_title(updated_chat, fallback=query)
        updated_chat.updated_at = user_message.created_at
        self.store.save_chat(updated_chat)

        command_result = self._build_slash_command_result(query, mode)
        if command_result is not None:
            assistant_message = self._save_assistant_message(chat_id, command_result)
            progress = WebProgress()
            progress.emit_answer_final(command_result)
            progress.emit_done(
                {
                    "chat_id": chat_id,
                    "assistant_message_id": assistant_message.id,
                }
            )
            progress.close()
            return progress.iter_events()

        progress = WebProgress()

        def run_turn() -> None:
            try:
                app = self.app_factory(progress)
                answer, reports = app._run_turn_detailed(
                    query=query,
                    transcript=prior_transcript,
                    num_agent=mode_to_agent_count(mode),
                    max_iter_per_agent=self.max_iter_per_agent,
                )
                turn_result = self._build_turn_result(
                    answer=answer,
                    reports=reports,
                    observations=list(progress.tool_outputs),
                    mode=mode,
                )
                assistant_message = self._save_assistant_message(chat_id, turn_result)

                for delta in chunk_text(turn_result.answer_markdown):
                    progress.emit_answer_delta(delta)
                progress.emit_answer_final(turn_result)
                progress.emit_done(
                    {
                        "chat_id": chat_id,
                        "assistant_message_id": assistant_message.id,
                    }
                )
            except Exception as exc:
                progress.emit_error(str(exc))
                progress.emit_done({"chat_id": chat_id})
            finally:
                progress.close()

        Thread(target=run_turn, daemon=True).start()
        return progress.iter_events()

    def _build_slash_command_result(self, query: str, mode: ChatMode) -> TurnResult | None:
        if not query.startswith("/"):
            return None
        if query.lower() == _LIST_TOOL_COMMAND:
            answer = _tool_catalog_markdown()
        else:
            answer = _unknown_command_markdown(query)
        return TurnResult(
            answer_markdown=answer,
            sources=[],
            artifacts=[],
            mode=mode,
        )

    def _save_assistant_message(self, chat_id: str, turn_result: TurnResult) -> WebMessage:
        assistant_message = WebMessage(
            id=f"msg_{uuid4().hex}",
            role="assistant",
            content=turn_result.answer_markdown,
            created_at=utc_now(),
            mode=turn_result.mode,
            sources=turn_result.sources,
            artifacts=turn_result.artifacts,
        )
        latest_chat = self.get_chat(chat_id).model_copy(deep=True)
        latest_chat.messages.append(assistant_message)
        latest_chat.updated_at = assistant_message.created_at
        self.store.save_chat(latest_chat)
        return assistant_message

    def _build_turn_result(
        self,
        *,
        answer: str,
        reports: list[WorkerReport],
        observations: list[ToolObservation],
        mode: ChatMode,
    ) -> TurnResult:
        urls: list[str] = []
        urls.extend(_extract_urls_from_text(answer))
        for report in reports:
            urls.extend(_extract_urls_from_value(report.parsed))
            urls.extend(_extract_urls_from_text(report.text))
        artifacts = self._build_artifacts(observations)
        for observation in observations:
            parsed = try_load_json(observation.output)
            urls.extend(_extract_urls_from_value(parsed))
        return TurnResult(
            answer_markdown=answer,
            sources=_dedupe_keep_order(urls),
            artifacts=artifacts,
            mode=mode,
        )

    def _build_artifacts(self, observations: list[ToolObservation]) -> list[ArtifactLink]:
        artifacts: list[ArtifactLink] = []
        seen: set[str] = set()
        for observation in observations:
            parsed = try_load_json(observation.output)
            if not isinstance(parsed, dict) or parsed.get("ok") is not True:
                continue
            paths: list[tuple[str, ArtifactKind]] = []
            paths.extend((path, "image") for path in _path_from_value(parsed, "image_path"))
            paths.extend((path, "slides") for path in _path_from_value(parsed, "html_path"))
            for raw_path, kind in paths:
                resolved = self.resolve_file_path(raw_path)
                key = str(resolved)
                if key in seen or not resolved.exists():
                    continue
                seen.add(key)
                artifacts.append(
                    ArtifactLink(
                        kind=kind,
                        label=resolved.name,
                        path=str(resolved),
                        url=self.file_url_for(resolved),
                    )
                )
        return artifacts

    def resolve_file_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self._repo_root / path).resolve()
        else:
            path = path.resolve()
        return path

    def file_url_for(self, path: Path | str) -> str:
        resolved = self.resolve_file_path(str(path))
        return f"/api/files?path={quote(str(resolved))}"

    def _messages_to_transcript(self, messages: list[WebMessage]) -> list[ChatMessage]:
        return [ChatMessage(role=message.role, content=message.content) for message in messages]

    def _next_title(self, chat: WebChatDetail, fallback: str) -> str:
        if chat.title.strip() and chat.title != "New chat":
            return chat.title
        first_user = next((message.content for message in chat.messages if message.role == "user"), fallback)
        return summarize_text(first_user, width=48) or "New chat"
