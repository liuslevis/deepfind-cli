from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ChatMode = Literal["fast", "expert"]
MessageRole = Literal["user", "assistant"]
ArtifactKind = Literal["image", "slides", "file"]


class ArtifactLink(BaseModel):
    kind: ArtifactKind
    label: str
    path: str
    url: str


class CitationLink(BaseModel):
    id: str
    canonical_url: str
    url: str
    title: str = ""
    publisher: str = ""


class KeyPoint(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class WebMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    created_at: str
    mode: ChatMode | None = None
    sources: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    citations: list[CitationLink] = Field(default_factory=list)


class WebChatSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str = ""


class WebChatDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[WebMessage] = Field(default_factory=list)


class TurnResult(BaseModel):
    answer_markdown: str
    sources: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    citations: list[CitationLink] = Field(default_factory=list)
    mode: ChatMode


class ProgressEvent(BaseModel):
    type: str
    timestamp: str
    data: dict[str, Any] = Field(default_factory=dict)


class CreateChatResponse(BaseModel):
    chat: WebChatDetail


class ChatListResponse(BaseModel):
    chats: list[WebChatSummary]


class ChatDetailResponse(BaseModel):
    chat: WebChatDetail


class CreateChatRequest(BaseModel):
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str
    mode: ChatMode


class HealthResponse(BaseModel):
    ok: bool
    service: str
