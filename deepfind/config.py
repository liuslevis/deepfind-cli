from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .bili_transcribe import DEFAULT_ASR_MODEL


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value[0] in {'"', "'"} and value[-1] == value[0]:
        value = value[1:-1].strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value or None


def _env(name: str, default: str | None = None) -> str | None:
    value = _clean_env_value(os.getenv(name))
    if value is not None:
        return value
    return default


def _load_dotenv() -> None:
    env_file = _clean_env_value(os.getenv("DEEPFIND_ENV_FILE")) or ".env"
    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(raw_value)
        if key and value is not None and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    twitter_bin: str = "twitter"
    xhs_bin: str = "xhs"
    bili_bin: str = "bili"
    asr_model: str = DEFAULT_ASR_MODEL
    audio_dir: str = "audio"
    subprocess_timeout: int = 90

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        api_key = _env("QWEN_API_KEY") or _env("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Set QWEN_API_KEY or DASHSCOPE_API_KEY.")
        timeout = _env("DEEPFIND_TOOL_TIMEOUT", "90")
        return cls(
            api_key=api_key,
            model=_env("QWEN_MODEL") or _env("QWEN_MODEL_NAME", DEFAULT_MODEL) or DEFAULT_MODEL,
            base_url=_env("QWEN_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            twitter_bin=_env("TWITTER_CLI_BIN", "twitter") or "twitter",
            xhs_bin=_env("XHS_CLI_BIN", "xhs") or "xhs",
            bili_bin=_env("BILI_BIN", "bili") or "bili",
            asr_model=_env("ASR_MODEL", DEFAULT_ASR_MODEL) or DEFAULT_ASR_MODEL,
            audio_dir=_env("DEEPFIND_AUDIO_DIR", "audio") or "audio",
            subprocess_timeout=int(timeout or "90"),
        )

    def new_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)
