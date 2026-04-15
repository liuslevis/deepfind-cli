from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI

from .asr import DEFAULT_ASR_MODEL
from .gen_img import DEFAULT_IMAGE_DIR, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_SIZE


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3.5:9b"
DEFAULT_LOCAL_API_KEY = "ollama"


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
    local_model: str = DEFAULT_LOCAL_MODEL
    local_base_url: str = DEFAULT_LOCAL_BASE_URL
    local_api_key: str = DEFAULT_LOCAL_API_KEY
    nano_banana_api_key: str | None = None
    nano_banana_model: str = DEFAULT_IMAGE_MODEL
    image_dir: str = DEFAULT_IMAGE_DIR
    image_size: str = DEFAULT_IMAGE_SIZE
    opencli_bin: str = "opencli"
    twitter_bin: str = "twitter"
    xhs_bin: str = "xhs"
    bili_bin: str = "bili"
    ytdlp_bin: str = "yt-dlp"
    ffmpeg_bin: str = "ffmpeg"
    asr_model: str = DEFAULT_ASR_MODEL
    audio_dir: str = "audio"
    subprocess_timeout: int = 90

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> "Settings":
        _load_dotenv()
        api_key = _env("QWEN_API_KEY") or _env("DASHSCOPE_API_KEY") or ""
        if require_api_key and not api_key:
            raise RuntimeError("Set QWEN_API_KEY or DASHSCOPE_API_KEY, or use local GPU mode.")
        timeout = _env("DEEPFIND_TOOL_TIMEOUT", "90")
        return cls(
            api_key=api_key,
            model=_env("QWEN_MODEL") or _env("QWEN_MODEL_NAME", DEFAULT_MODEL) or DEFAULT_MODEL,
            base_url=_env("QWEN_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            local_model=_env("DEEPFIND_LOCAL_MODEL", DEFAULT_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL,
            local_base_url=_env("DEEPFIND_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL) or DEFAULT_LOCAL_BASE_URL,
            local_api_key=_env("DEEPFIND_LOCAL_API_KEY", DEFAULT_LOCAL_API_KEY) or DEFAULT_LOCAL_API_KEY,
            nano_banana_api_key=(
                _env("GOOGLE_NANO_BANANA_API_KEY")
                or _env("GEMINI_API_KEY")
                or _env("GOOGLE_API_KEY")
            ),
            nano_banana_model=(
                _env("GOOGLE_NANO_BANANA_MODEL")
                or _env("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
                or DEFAULT_IMAGE_MODEL
            ),
            image_dir=_env("DEEPFIND_IMAGE_DIR", DEFAULT_IMAGE_DIR) or DEFAULT_IMAGE_DIR,
            image_size=(
                _env("GOOGLE_NANO_BANANA_IMAGE_SIZE")
                or _env("DEEPFIND_IMAGE_SIZE", DEFAULT_IMAGE_SIZE)
                or DEFAULT_IMAGE_SIZE
            ),
            opencli_bin=_env("OPENCLI_BIN", "opencli") or "opencli",
            twitter_bin=_env("TWITTER_CLI_BIN", "twitter") or "twitter",
            xhs_bin=_env("XHS_CLI_BIN", "xhs") or "xhs",
            bili_bin=_env("BILI_BIN", "bili") or "bili",
            ytdlp_bin=_env("YTDLP_BIN", "yt-dlp") or "yt-dlp",
            ffmpeg_bin=_env("FFMPEG_BIN", "ffmpeg") or "ffmpeg",
            asr_model=_env("ASR_MODEL", DEFAULT_ASR_MODEL) or DEFAULT_ASR_MODEL,
            audio_dir=_env("DEEPFIND_AUDIO_DIR", "audio") or "audio",
            subprocess_timeout=int(timeout or "90"),
        )

    def new_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ensure_remote_ready(self) -> "Settings":
        if not self.api_key:
            raise RuntimeError("Set QWEN_API_KEY or DASHSCOPE_API_KEY, or switch to GPU mode.")
        return self

    def with_local_gpu(self) -> "Settings":
        return replace(
            self,
            api_key=self.local_api_key,
            model=self.local_model,
            base_url=self.local_base_url,
        )
