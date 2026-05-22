from __future__ import annotations

import os
import platform
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI

from .asr import DEFAULT_ASR_MODEL
from .gen_img import DEFAULT_IMAGE_DIR, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_SIZE


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5-pro"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3.5:latest"
DEFAULT_LOCAL_API_KEY = "ollama"


class SettingsError(RuntimeError):
    pass


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
    qwen_api_key: str = ""
    qwen_model: str = DEFAULT_MODEL
    qwen_base_url: str = DEFAULT_BASE_URL
    mimo_api_key: str = ""
    mimo_model: str = DEFAULT_MIMO_MODEL
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    minimax_api_key: str = ""
    minimax_model: str = DEFAULT_MINIMAX_MODEL
    minimax_base_url: str = DEFAULT_MINIMAX_BASE_URL
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
    def _resolve_asr_model(cls) -> str:
        """Resolve ASR model based on platform.

        Priority:
        1. ASR_MODEL_MAC on Darwin (macOS)
        2. ASR_MODEL_PC on non-Darwin platforms
        3. DEFAULT_ASR_MODEL (fallback)
        """
        # Platform-specific selection
        is_darwin = platform.system().lower() == "darwin"

        if is_darwin:
            # macOS: prefer ASR_MODEL_MAC, fallback to mlx-whisper:large-v3
            mac_model = _env("ASR_MODEL_MAC")
            if mac_model:
                return mac_model
            # Default for Mac: use MLX Whisper for speed
            return "mlx-whisper:large-v3"
        else:
            # PC (Linux/Windows): prefer ASR_MODEL_PC
            pc_model = _env("ASR_MODEL_PC")
            if pc_model:
                return pc_model
            # Default for PC: use Qwen3-ASR
            return DEFAULT_ASR_MODEL

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> "Settings":
        _load_dotenv()
        qwen_api_key = _env("QWEN_API_KEY") or _env("DASHSCOPE_API_KEY") or ""
        qwen_model = _env("QWEN_MODEL") or _env("QWEN_MODEL_NAME", DEFAULT_MODEL) or DEFAULT_MODEL
        qwen_base_url = _env("QWEN_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
        mimo_api_key = _env("MIMO_API_KEY") or _env("XIAOMI_API_KEY") or ""
        mimo_model = (
            _env("MIMO_MODEL")
            or _env("MIMO_MODEL_NAME")
            or _env("XIAOMI_MODEL")
            or _env("XIAOMI_MODEL_NAME", DEFAULT_MIMO_MODEL)
            or DEFAULT_MIMO_MODEL
        )
        mimo_base_url = _env("MIMO_BASE_URL") or _env("XIAOMI_BASE_URL", DEFAULT_MIMO_BASE_URL) or DEFAULT_MIMO_BASE_URL
        minimax_api_key = _env("MINIMAX_API_KEY") or ""
        minimax_model = _env("MINIMAX_MODEL") or _env("MINIMAX_MODEL_NAME", DEFAULT_MINIMAX_MODEL) or DEFAULT_MINIMAX_MODEL
        minimax_base_url = _env("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL) or DEFAULT_MINIMAX_BASE_URL
        remote_target = "qwen"
        if not qwen_api_key:
            if minimax_api_key:
                remote_target = "minimax"
            elif mimo_api_key:
                remote_target = "mimo"
        if remote_target == "qwen":
            api_key = qwen_api_key
            model = qwen_model
            base_url = qwen_base_url
        elif remote_target == "minimax":
            api_key = minimax_api_key
            model = minimax_model
            base_url = minimax_base_url
        else:
            api_key = mimo_api_key
            model = mimo_model
            base_url = mimo_base_url
        if require_api_key and not api_key:
            raise SettingsError(
                "Set QWEN_API_KEY, DASHSCOPE_API_KEY, MIMO_API_KEY, XIAOMI_API_KEY, or MINIMAX_API_KEY, or use local GPU mode."
            )
        timeout = _env("DEEPFIND_TOOL_TIMEOUT", "90")
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            qwen_api_key=qwen_api_key,
            qwen_model=qwen_model,
            qwen_base_url=qwen_base_url,
            mimo_api_key=mimo_api_key,
            mimo_model=mimo_model,
            mimo_base_url=mimo_base_url,
            minimax_api_key=minimax_api_key,
            minimax_model=minimax_model,
            minimax_base_url=minimax_base_url,
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
            asr_model=cls._resolve_asr_model(),
            audio_dir=_env("DEEPFIND_AUDIO_DIR", "audio") or "audio",
            subprocess_timeout=int(timeout or "90"),
        )

    def new_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ensure_remote_ready(self) -> "Settings":
        if not self.api_key:
            raise SettingsError(
                "Set QWEN_API_KEY, DASHSCOPE_API_KEY, MIMO_API_KEY, XIAOMI_API_KEY, or MINIMAX_API_KEY, or switch to GPU mode."
            )
        return self

    def with_qwen_remote(self) -> "Settings":
        if not self.qwen_api_key:
            raise SettingsError("Set QWEN_API_KEY or DASHSCOPE_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.qwen_api_key,
            model=self.qwen_model,
            base_url=self.qwen_base_url,
        )

    def with_mimo_remote(self) -> "Settings":
        if not self.mimo_api_key:
            raise SettingsError("Set MIMO_API_KEY or XIAOMI_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.mimo_api_key,
            model=self.mimo_model,
            base_url=self.mimo_base_url,
        )

    def with_minimax_remote(self) -> "Settings":
        if not self.minimax_api_key:
            raise SettingsError("Set MINIMAX_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.minimax_api_key,
            model=self.minimax_model,
            base_url=self.minimax_base_url,
        )

    def with_local_gpu(self) -> "Settings":
        return replace(
            self,
            api_key=self.local_api_key,
            model=self.local_model,
            base_url=self.local_base_url,
        )
