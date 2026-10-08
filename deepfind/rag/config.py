"""Runtime configuration loaded from environment variables / .env.

The formatter is an OpenAI-compatible LLM. Variable names support both the
generic ``FORMATTER_*`` form and the DeepSeek names present in the local
``.env`` (``DEEPSEEK_*``), preferring the former when both are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FormatterConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def config_version(self) -> str:
        return self.model


@dataclass(frozen=True)
class Config:
    root_dir: Path = REPO_ROOT
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "investment_kb"

    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    asr_segment_seconds: int = 300
    ffmpeg_bin: str = "ffmpeg"
    asr_cleanup_work: bool = True

    dense_model: str = "intfloat/multilingual-e5-large"
    sparse_model: str = "Qdrant/bm25"

    formatter: FormatterConfig = field(
        default_factory=lambda: FormatterConfig("https://api.deepseek.com", "", "")
    )

    @property
    def doc_root(self) -> Path:
        return self.root_dir / "doc"

    @property
    def pdf_root(self) -> Path:
        return self.doc_root / "pdf"

    @property
    def media_root(self) -> Path:
        return self.doc_root / "media"

    @property
    def embedding_config_version(self) -> str:
        return f"{self.dense_model}|{self.sparse_model}"


def _resolve_root(root_dir: Path | str | None) -> Path:
    raw = root_dir or _env("DEEPFIND_RAG_DIR", default=".")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(root_dir: Path | str | None = None) -> Config:
    settings = Settings.from_env(require_api_key=False)
    formatter = FormatterConfig(
        base_url=_env("FORMATTER_BASE_URL", default=settings.deepseek_base_url),
        api_key=_env("FORMATTER_API_KEY", default=settings.deepseek_api_key),
        model=_env(
            "FORMATTER_MODEL",
            "DEEPSEEK_LIGHT_MODEL_NAME",
            default=settings.deepseek_model,
        ),
    )
    return Config(
        root_dir=_resolve_root(root_dir or settings.rag_dir),
        qdrant_url=_env("QDRANT_URL", default="http://127.0.0.1:6333"),
        qdrant_collection=_env("QDRANT_COLLECTION", default="investment_kb"),
        asr_model=settings.asr_model,
        asr_segment_seconds=int(_env("ASR_SEGMENT_SECONDS", default="300")),
        ffmpeg_bin=settings.ffmpeg_bin,
        asr_cleanup_work=_bool(_env("ASR_CLEANUP_WORK"), default=True),
        dense_model=_env("DENSE_MODEL", default="intfloat/multilingual-e5-large"),
        sparse_model=_env("SPARSE_MODEL", default="Qdrant/bm25"),
        formatter=formatter,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
