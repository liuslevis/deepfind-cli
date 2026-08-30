from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ChatMode = Literal["fast", "expert"]
ResearchMode = Literal["deep_research", "research", "chat"]
MessageRole = Literal["user", "assistant"]
ArtifactKind = Literal["image", "slides", "file"]
ModelTarget = Literal["qwen", "mimo", "minimax", "glm", "deepseek", "gpu"]


def normalize_model_target(value: Any) -> ModelTarget:
    if value is None:
        return "qwen"
    normalized = str(value).strip().lower()
    if not normalized or normalized == "cloud":
        return "qwen"
    if normalized in {"qwen", "mimo", "minimax", "glm", "deepseek", "gpu"}:
        return normalized
    raise ValueError(f"unsupported model target: {value}")


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


class GpuInfo(BaseModel):
    available: bool = False
    name: str = ""
    memory_total_mb: int | None = None


class LocalModelInfo(BaseModel):
    available: bool = False
    backend: str = "ollama"
    model: str = ""
    base_url: str = ""
    reason: str = ""
    gpu: GpuInfo = Field(default_factory=GpuInfo)


class ToolOption(BaseModel):
    name: str
    description: str
    parameters: list[str] = Field(default_factory=list)


class WebMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    created_at: str
    mode: ChatMode | None = None
    research_mode: ResearchMode = "research"
    sources: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    citations: list[CitationLink] = Field(default_factory=list)
    model_target: ModelTarget = "qwen"
    model_label: str = ""

    @field_validator("model_target", mode="before")
    @classmethod
    def _normalize_model_target(cls, value: Any) -> ModelTarget:
        return normalize_model_target(value)


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
    research_mode: ResearchMode = "research"
    model_target: ModelTarget = "qwen"
    model_label: str = ""

    @field_validator("model_target", mode="before")
    @classmethod
    def _normalize_model_target(cls, value: Any) -> ModelTarget:
        return normalize_model_target(value)


class ProgressEvent(BaseModel):
    type: str
    timestamp: str
    data: dict[str, Any] = Field(default_factory=dict)


class CreateChatResponse(BaseModel):
    chat: WebChatDetail


class ChatListResponse(BaseModel):
    chats: list[WebChatSummary]
    local_model: LocalModelInfo | None = None
    tools: list[ToolOption] = Field(default_factory=list)


class ChatDetailResponse(BaseModel):
    chat: WebChatDetail


class CreateChatRequest(BaseModel):
    title: str | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(..., max_length=20000)
    mode: ChatMode
    model_target: ModelTarget = "qwen"
    deep_mode: bool = False
    research_mode: ResearchMode | None = None
    selected_tools: list[str] | None = None

    @field_validator("model_target", mode="before")
    @classmethod
    def _normalize_model_target(cls, value: Any) -> ModelTarget:
        return normalize_model_target(value)


class HealthResponse(BaseModel):
    ok: bool
    service: str
    local_model: LocalModelInfo | None = None
    requires_token: bool = False
