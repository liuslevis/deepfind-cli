from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .web_models import WebChatDetail, WebChatSummary


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_chat_root() -> Path:
    return repo_root() / "tmp" / "web" / "chats"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_text(text: str, width: int = 96) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= width:
        return clean
    return f"{clean[: width - 1].rstrip()}..."


class ChatStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_chat_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def list_chats(self) -> list[WebChatSummary]:
        chats = [self._read_chat(path) for path in sorted(self.root.glob("*.json"))]
        chats.sort(key=lambda item: item.updated_at, reverse=True)
        return [self._to_summary(chat) for chat in chats]

    def create_chat(self, title: str | None = None) -> WebChatDetail:
        now = utc_now()
        chat = WebChatDetail(
            id=f"chat_{uuid4().hex}",
            title=(title or "New chat").strip() or "New chat",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self.save_chat(chat)
        return chat

    def get_chat(self, chat_id: str) -> WebChatDetail:
        path = self._path(chat_id)
        if not path.exists():
            raise FileNotFoundError(chat_id)
        return self._read_chat(path)

    def save_chat(self, chat: WebChatDetail) -> WebChatDetail:
        payload = chat.model_dump(mode="json")
        path = self._path(chat.id)
        temp_path = path.with_suffix(".json.tmp")
        with self._lock:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
        return chat

    def delete_chat(self, chat_id: str) -> None:
        path = self._path(chat_id)
        if not path.exists():
            raise FileNotFoundError(chat_id)
        with self._lock:
            path.unlink()

    def _read_chat(self, path: Path) -> WebChatDetail:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WebChatDetail.model_validate(data)

    def _path(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.json"

    def _to_summary(self, chat: WebChatDetail) -> WebChatSummary:
        preview = ""
        if chat.messages:
            preview = summarize_text(chat.messages[-1].content)
        return WebChatSummary(
            id=chat.id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            preview=preview,
        )
