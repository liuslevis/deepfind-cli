from __future__ import annotations

import os
import platform
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI

from .asr import DEFAULT_ASR_MODEL
from .gen_img import DEFAULT_IMAGE_DIR, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_SIZE
from .llm_transport import CHAT_COMPLETIONS_API, RESPONSES_API


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5-pro"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_MODEL = "glm-5.2"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen3.5:9B"  # Change this to upgrade (e.g., "qwen3.6:27B")
DEFAULT_LOCAL_API_KEY = "ollama"
DEFAULT_CODING_TIMEOUT = 120
DEFAULT_CODING_MAX_CONCURRENT = 4


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


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise SettingsError(f"{name} must be true or false")


def _env_positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = _env(name, str(default)) or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = ">= 0" if allow_zero else "> 0"
        raise SettingsError(f"{name} must be {comparator}")
    return value


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
    sub_model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_mode: str = CHAT_COMPLETIONS_API
    qwen_api_key: str = ""
    qwen_model: str = DEFAULT_MODEL
    qwen_sub_model: str = DEFAULT_MODEL
    qwen_base_url: str = DEFAULT_BASE_URL
    mimo_api_key: str = ""
    mimo_model: str = DEFAULT_MIMO_MODEL
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    minimax_api_key: str = ""
    minimax_model: str = DEFAULT_MINIMAX_MODEL
    minimax_sub_model: str = DEFAULT_MINIMAX_MODEL
    minimax_base_url: str = DEFAULT_MINIMAX_BASE_URL
    glm_api_key: str = ""
    glm_model: str = DEFAULT_GLM_MODEL
    glm_base_url: str = DEFAULT_GLM_BASE_URL
    deepseek_api_key: str = ""
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_sub_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    local_model: str = DEFAULT_LOCAL_MODEL
    local_base_url: str = DEFAULT_LOCAL_BASE_URL
    local_api_key: str = DEFAULT_LOCAL_API_KEY
    think: bool = False
    nano_banana_api_key: str | None = None
    nano_banana_model: str = DEFAULT_IMAGE_MODEL
    image_dir: str = DEFAULT_IMAGE_DIR
    image_size: str = DEFAULT_IMAGE_SIZE
    x_app_bearer_token: str = ""
    x_consumer_key: str = ""
    x_consumer_key_secret: str = ""
    x_access_key: str = ""
    x_access_token: str = ""
    opencli_bin: str = "opencli"
    twitter_bin: str = "twitter"
    xhs_bin: str = "xhs"
    bili_bin: str = "bili"
    ytdlp_bin: str = "yt-dlp"
    ytdlp_cookies_from_browser: str | None = None
    ytdlp_cookies: str | None = None
    ytdlp_js_runtimes: str | None = "node"
    ytdlp_extractor_args: str | None = "youtube:player_client=web;fetch_pot=always"
    ffmpeg_bin: str = "ffmpeg"
    asr_model: str = DEFAULT_ASR_MODEL
    audio_dir: str = "video"
    subprocess_timeout: int = 90
    rag_mcp_command: str = "uv"
    rag_mcp_project_dir: str = "../deepfind-rag"
    coding_enabled: bool = False
    coding_runtime: str = "docker"
    coding_image: str = ""
    coding_root: str = "sandbox"
    coding_timeout: int = DEFAULT_CODING_TIMEOUT
    coding_max_concurrent: int = DEFAULT_CODING_MAX_CONCURRENT
    coding_network: bool = False
    coding_retention: int = 0

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
        qwen_model = (
            _env("QWEN_MODEL")
            or _env("QWEN_MODEL_NAME", DEFAULT_MODEL)
            or DEFAULT_MODEL
        )
        qwen_sub_model = (
            _env("QWEN_SUB_MODEL")
            or _env("QWEN_SUB_MODEL_NAME", qwen_model)
            or qwen_model
        )
        qwen_base_url = _env("QWEN_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
        mimo_api_key = _env("MIMO_API_KEY") or _env("XIAOMI_API_KEY") or ""
        mimo_model = (
            _env("MIMO_MODEL")
            or _env("MIMO_MODEL_NAME")
            or _env("XIAOMI_MODEL")
            or _env("XIAOMI_MODEL_NAME", DEFAULT_MIMO_MODEL)
            or DEFAULT_MIMO_MODEL
        )
        mimo_base_url = (
            _env("MIMO_BASE_URL")
            or _env("XIAOMI_BASE_URL", DEFAULT_MIMO_BASE_URL)
            or DEFAULT_MIMO_BASE_URL
        )
        minimax_api_key = _env("MINIMAX_API_KEY") or ""
        minimax_model = (
            _env("MINIMAX_MODEL")
            or _env("MINIMAX_MODEL_NAME", DEFAULT_MINIMAX_MODEL)
            or DEFAULT_MINIMAX_MODEL
        )
        minimax_sub_model = (
            _env("MINIMAX_SUB_MODEL")
            or _env("MINIMAX_SUB_MODEL_NAME", minimax_model)
            or minimax_model
        )
        minimax_base_url = (
            _env("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL)
            or DEFAULT_MINIMAX_BASE_URL
        )
        glm_api_key = _env("GLM_API_KEY") or ""
        glm_model = (
            _env("GLM_MODEL")
            or _env("GLM_MODEL_NAME", DEFAULT_GLM_MODEL)
            or DEFAULT_GLM_MODEL
        )
        glm_base_url = (
            _env("GLM_BASE_URL", DEFAULT_GLM_BASE_URL) or DEFAULT_GLM_BASE_URL
        )
        deepseek_api_key = _env("DEEPSEEK_API_KEY") or ""
        deepseek_model = (
            _env("DEEPSEEK_MODEL")
            or _env("DEEPSEEK_MODEL_NAME", DEFAULT_DEEPSEEK_MODEL)
            or DEFAULT_DEEPSEEK_MODEL
        )
        deepseek_sub_model = (
            _env("DEEPSEEK_SUB_MODEL")
            or _env("DEEPSEEK_SUB_MODEL_NAME", deepseek_model)
            or deepseek_model
        )
        deepseek_base_url = (
            _env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
            or DEFAULT_DEEPSEEK_BASE_URL
        )
        api_mode = CHAT_COMPLETIONS_API
        remote_target = "qwen"
        if not qwen_api_key:
            if glm_api_key:
                remote_target = "glm"
            elif minimax_api_key:
                remote_target = "minimax"
            elif mimo_api_key:
                remote_target = "mimo"
            elif deepseek_api_key:
                remote_target = "deepseek"
        if remote_target == "qwen":
            api_key = qwen_api_key
            model = qwen_model
            sub_model = qwen_sub_model
            base_url = qwen_base_url
        elif remote_target == "minimax":
            api_key = minimax_api_key
            model = minimax_model
            sub_model = minimax_sub_model
            base_url = minimax_base_url
        elif remote_target == "glm":
            api_key = glm_api_key
            model = glm_model
            sub_model = glm_model
            base_url = glm_base_url
        elif remote_target == "mimo":
            api_key = mimo_api_key
            model = mimo_model
            sub_model = mimo_model
            base_url = mimo_base_url
        else:
            api_key = deepseek_api_key
            model = deepseek_model
            sub_model = deepseek_sub_model
            base_url = deepseek_base_url
            api_mode = RESPONSES_API
        if require_api_key and not api_key:
            raise SettingsError(
                "Set QWEN_API_KEY, DASHSCOPE_API_KEY, MIMO_API_KEY, XIAOMI_API_KEY, MINIMAX_API_KEY, GLM_API_KEY, DEEPSEEK_API_KEY, or use local GPU mode."
            )
        timeout = _env("DEEPFIND_TOOL_TIMEOUT", "90")
        coding_enabled = _env_bool("DEEPFIND_CODING_ENABLED")
        coding_runtime = (_env("DEEPFIND_CODING_RUNTIME", "docker") or "docker").lower()
        if coding_runtime not in {"docker", "podman"}:
            raise SettingsError("DEEPFIND_CODING_RUNTIME must be docker or podman")
        coding_image = _env("DEEPFIND_CODING_IMAGE", "") or ""
        if (
            coding_enabled
            and "@sha256:" not in coding_image
            and not coding_image.startswith("sha256:")
        ):
            raise SettingsError(
                "DEEPFIND_CODING_IMAGE must use an immutable repo digest or image ID when coding is enabled"
            )
        coding_network = _env_bool("DEEPFIND_CODING_NETWORK")
        if coding_network:
            raise SettingsError(
                "DEEPFIND_CODING_NETWORK=true is not supported by the Phase 1 sandbox"
            )
        return cls(
            api_key=api_key,
            model=model,
            sub_model=sub_model,
            base_url=base_url,
            api_mode=api_mode,
            qwen_api_key=qwen_api_key,
            qwen_model=qwen_model,
            qwen_sub_model=qwen_sub_model,
            qwen_base_url=qwen_base_url,
            mimo_api_key=mimo_api_key,
            mimo_model=mimo_model,
            mimo_base_url=mimo_base_url,
            minimax_api_key=minimax_api_key,
            minimax_model=minimax_model,
            minimax_sub_model=minimax_sub_model,
            minimax_base_url=minimax_base_url,
            glm_api_key=glm_api_key,
            glm_model=glm_model,
            glm_base_url=glm_base_url,
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            deepseek_sub_model=deepseek_sub_model,
            deepseek_base_url=deepseek_base_url,
            local_model=_env("DEEPFIND_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
            or DEFAULT_LOCAL_MODEL,
            local_base_url=_env("DEEPFIND_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL)
            or DEFAULT_LOCAL_BASE_URL,
            local_api_key=_env("DEEPFIND_LOCAL_API_KEY", DEFAULT_LOCAL_API_KEY)
            or DEFAULT_LOCAL_API_KEY,
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
            image_dir=_env("DEEPFIND_IMAGE_DIR", DEFAULT_IMAGE_DIR)
            or DEFAULT_IMAGE_DIR,
            image_size=(
                _env("GOOGLE_NANO_BANANA_IMAGE_SIZE")
                or _env("DEEPFIND_IMAGE_SIZE", DEFAULT_IMAGE_SIZE)
                or DEFAULT_IMAGE_SIZE
            ),
            x_app_bearer_token=_env("X_APP_BEARER_TOKEN", "") or "",
            x_consumer_key=_env("X_CONSUMER_KEY", "") or "",
            x_consumer_key_secret=_env("X_CONSUMER_KEY_SECRET", "") or "",
            x_access_key=_env("X_ACCESS_KEY", "") or "",
            x_access_token=_env("X_ACCESS_TOKEN", "") or "",
            opencli_bin=_env("OPENCLI_BIN", "opencli") or "opencli",
            twitter_bin=_env("TWITTER_CLI_BIN", "twitter") or "twitter",
            xhs_bin=_env("XHS_CLI_BIN", "xhs") or "xhs",
            bili_bin=_env("BILI_BIN", "bili") or "bili",
            ytdlp_bin=_env("YTDLP_BIN", "yt-dlp") or "yt-dlp",
            ytdlp_cookies_from_browser=_env("YTDLP_COOKIES_FROM_BROWSER"),
            ytdlp_cookies=_env("YTDLP_COOKIES"),
            ytdlp_js_runtimes=_env("YTDLP_JS_RUNTIMES", "node"),
            ytdlp_extractor_args=_env(
                "YTDLP_EXTRACTOR_ARGS", "youtube:player_client=web;fetch_pot=always"
            ),
            ffmpeg_bin=_env("FFMPEG_BIN", "ffmpeg") or "ffmpeg",
            asr_model=cls._resolve_asr_model(),
            audio_dir=_env("DEEPFIND_VIDEO_DIR", "video") or "video",
            subprocess_timeout=int(timeout or "90"),
            rag_mcp_command=_env("DEEPFIND_RAG_MCP_COMMAND", "uv") or "uv",
            rag_mcp_project_dir=(
                _env("DEEPFIND_RAG_MCP_PROJECT_DIR", "../deepfind-rag")
                or "../deepfind-rag"
            ),
            coding_enabled=coding_enabled,
            coding_runtime=coding_runtime,
            coding_image=coding_image,
            coding_root=_env("DEEPFIND_CODING_ROOT", "sandbox") or "sandbox",
            coding_timeout=_env_positive_int(
                "DEEPFIND_CODING_TIMEOUT",
                DEFAULT_CODING_TIMEOUT,
            ),
            coding_max_concurrent=_env_positive_int(
                "DEEPFIND_CODING_MAX_CONCURRENT",
                DEFAULT_CODING_MAX_CONCURRENT,
            ),
            coding_network=coding_network,
            coding_retention=_env_positive_int(
                "DEEPFIND_CODING_RETENTION",
                0,
                allow_zero=True,
            ),
        )

    def new_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ensure_remote_ready(self) -> "Settings":
        if not self.api_key:
            raise SettingsError(
                "Set QWEN_API_KEY, DASHSCOPE_API_KEY, MIMO_API_KEY, XIAOMI_API_KEY, MINIMAX_API_KEY, GLM_API_KEY, DEEPSEEK_API_KEY, or switch to GPU mode."
            )
        return self

    def with_qwen_remote(self) -> "Settings":
        if not self.qwen_api_key:
            raise SettingsError(
                "Set QWEN_API_KEY or DASHSCOPE_API_KEY, or switch to another model."
            )
        return replace(
            self,
            api_key=self.qwen_api_key,
            model=self.qwen_model,
            sub_model=self.qwen_sub_model,
            base_url=self.qwen_base_url,
            api_mode=CHAT_COMPLETIONS_API,
        )

    def with_mimo_remote(self) -> "Settings":
        if not self.mimo_api_key:
            raise SettingsError(
                "Set MIMO_API_KEY or XIAOMI_API_KEY, or switch to another model."
            )
        return replace(
            self,
            api_key=self.mimo_api_key,
            model=self.mimo_model,
            sub_model=self.mimo_model,
            base_url=self.mimo_base_url,
            api_mode=CHAT_COMPLETIONS_API,
        )

    def with_minimax_remote(self) -> "Settings":
        if not self.minimax_api_key:
            raise SettingsError("Set MINIMAX_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.minimax_api_key,
            model=self.minimax_model,
            sub_model=self.minimax_model,
            base_url=self.minimax_base_url,
            api_mode=CHAT_COMPLETIONS_API,
        )

    def with_glm_remote(self) -> "Settings":
        if not self.glm_api_key:
            raise SettingsError("Set GLM_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.glm_api_key,
            model=self.glm_model,
            sub_model=self.glm_model,
            base_url=self.glm_base_url,
            api_mode=CHAT_COMPLETIONS_API,
        )

    def with_deepseek_remote(self) -> "Settings":
        if not self.deepseek_api_key:
            raise SettingsError("Set DEEPSEEK_API_KEY, or switch to another model.")
        return replace(
            self,
            api_key=self.deepseek_api_key,
            model=self.deepseek_model,
            sub_model=self.deepseek_sub_model,
            base_url=self.deepseek_base_url,
            api_mode=RESPONSES_API,
        )

    def with_local_gpu(self) -> "Settings":
        return replace(
            self,
            api_key=self.local_api_key,
            model=self.local_model,
            sub_model=self.local_model,
            base_url=self.local_base_url,
            api_mode=CHAT_COMPLETIONS_API,
            think=True,
        )
