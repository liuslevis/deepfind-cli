"""Shared pydantic data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "video", "audio"]
Status = Literal[
    "discovered",
    "processing",
    "raw_ready",
    "formatted",
    "chunked",
    "indexed",
    "failed",
]


class SourceFile(BaseModel):
    """A discovered source file and its derived parsed directory."""

    path: str  # POSIX path relative to repo root, e.g. doc/pdf/a/report.pdf
    parsed_dir: str  # POSIX path relative to repo root, e.g. doc/pdf/a/report
    source_type: SourceType


class Fingerprint(BaseModel):
    sha256: str
    size: int
    mtime_ns: int


class Metadata(BaseModel):
    schema_version: int = 1
    source: str
    source_type: SourceType
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    parser: str
    parser_version: str = ""
    formatter_model: str = ""
    formatter_config_version: str = ""
    embedding_model: str = ""
    chunk_config_version: str = ""
    embedding_config_version: str = ""
    created_at: str
    updated_at: str
    status: Status = "discovered"
    error: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    chunk_index: int = 0
    source: str
    source_type: SourceType
    title: str
    tags: list[str] = Field(default_factory=list)
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    start_seconds: int | None = None
    end_seconds: int | None = None
    text: str


class SearchResult(BaseModel):
    text: str
    source: str
    title: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    start_seconds: int | None = None
    end_seconds: int | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
