from __future__ import annotations

import gc
import logging
import os
import platform
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Semaphore, Thread, local as thread_local
from typing import Any


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
SEGMENT_SECONDS = 300
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac"}
DEFAULT_ASR_IDLE_UNLOAD_SECONDS = 3600
REPO_ROOT = Path(__file__).resolve().parent.parent
_GPU_ASR_SEMAPHORE = Semaphore(1)

# How many serialized ASR slots may hold a cached model at once (idempotent slots reuse).
_ASR_ACTIVE_SLOTS = 2
# Coarse clock keeps the reaper testable and immune to wall-clock jumps.
_MONOTONIC = time.monotonic


@dataclass(frozen=True, slots=True)
class _LoadedASRModel:
    """One resident ASR model plus the metadata needed to release it safely."""

    model: Any
    backend: str
    asr_model: str
    device: str
    last_used: float

    def touched(self) -> "_LoadedASRModel":
        return _LoadedASRModel(
            model=self.model,
            backend=self.backend,
            asr_model=self.asr_model,
            device=self.device,
            last_used=_MONOTONIC(),
        )


# Cached model per serialized ASR slot (slot 0 is the CPU-only fallback). A worker
# reuses its slot for as long as it processes jobs, so back-to-back tasks avoid
# reloading; models idle past the TTL are released by a background reaper.
_ASR_SLOTS: list[_LoadedASRModel | None] = [None] * _ASR_ACTIVE_SLOTS
_ASR_REAPER_LOCK = Lock()
_ASR_REAPER_STARTED = False
_ASR_REAPER_WAKE: Any = None
_ASR_UNLOAD_EVENTS: list[tuple[float, str, bool]] = []
_SLOT_HOLDER = thread_local()

# Supported model prefixes for different backends
MLX_WHISPER_MODELS = {"mlx-whisper", "whisper"}  # e.g., "mlx-whisper:large-v3" or "whisper:base"


def _logger() -> logging.Logger:
    uvicorn_logger = logging.getLogger("uvicorn.error")
    if uvicorn_logger.hasHandlers():
        return uvicorn_logger
    return logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    """Raised when required binaries or Python dependencies are unavailable."""


class TranscriptionError(RuntimeError):
    """Raised when ASR model loading or transcription fails."""


def resolve_audio_root(audio_dir: str | None) -> Path:
    from .video_resources import resolve_video_root

    return resolve_video_root(audio_dir)


def load_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return text or None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def load_local_secrets() -> None:
    secrets_path = REPO_ROOT / ".secrets"
    if not secrets_path.exists():
        return

    for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@contextmanager
def gpu_asr_slot():
    # Every supported backend uses GPU acceleration, so serialize all ASR jobs.
    _GPU_ASR_SEMAPHORE.acquire()
    holder = _SlotHolder()
    _SLOT_HOLDER.local = holder
    try:
        yield slot_index(holder)
    finally:
        if _SLOT_HOLDER.local is holder:
            _SLOT_HOLDER.local = None
        _GPU_ASR_SEMAPHORE.release()


class _SlotHolder:
    """Tracks which cache slot a single serialized ASR job is allowed to use."""

    __slots__ = ("owner",)

    def __init__(self) -> None:
        self.owner: int | None = None


def _current_holder() -> _SlotHolder:
    holder = getattr(_SLOT_HOLDER, "local", None)
    if holder is None:
        # No gpu_asr_slot() in flight (e.g. a direct call in tests): use a private slot.
        holder = _SlotHolder()
    return holder


def slot_index(holder: _SlotHolder | None = None) -> int:
    """Return the cache slot for the current ASR job, reusing a resident model."""
    holder = holder or _current_holder()
    if holder.owner is not None:
        return holder.owner

    for index, cached in enumerate(_ASR_SLOTS):
        if cached is not None:
            holder.owner = index
            return index

    index = 0
    for candidate in range(_ASR_ACTIVE_SLOTS):
        if _ASR_SLOTS[candidate] is None:
            index = candidate
            break
    else:
        index = _ASR_ACTIVE_SLOTS - 1
    holder.owner = index
    return index


def asr_model_idle_unload_seconds() -> float | None:
    """Return the idle time after which a resident ASR model is released.

    Zero or negative values disable unloading entirely.
    """
    raw = os.environ.get("ASR_IDLE_UNLOAD_SECONDS", "").strip()
    seconds: float = DEFAULT_ASR_IDLE_UNLOAD_SECONDS
    if raw:
        try:
            seconds = float(raw)
        except ValueError:
            seconds = DEFAULT_ASR_IDLE_UNLOAD_SECONDS
    return seconds if seconds > 0 else None


def _asr_slot_model(slot: int, asr_model: str) -> tuple[str, Any, Any, str]:
    """Return the model cached for ``slot``, loading or swapping as needed."""
    cached = _ASR_SLOTS[slot]
    if cached is not None and cached.asr_model == asr_model:
        _ASR_SLOTS[slot] = cached.touched()
        return cached.backend, cached.model, None, cached.device

    if cached is not None:
        # A different transcription model was requested: free the old one first.
        _unload_slot(slot, reason="model switch")

    backend, model, processor, device = load_model(asr_model)
    _ASR_SLOTS[slot] = _LoadedASRModel(
        model=model,
        backend=backend,
        asr_model=asr_model,
        device=device,
        last_used=_MONOTONIC(),
    )
    _start_unload_reaper()
    return backend, model, processor, device


def _unload_slot(slot: int, *, reason: str) -> bool:
    """Drop the model cached in ``slot`` and return its GPU memory to the driver."""
    cached = _ASR_SLOTS[slot]
    if cached is None:
        return False
    _ASR_SLOTS[slot] = None
    _ASR_UNLOAD_EVENTS.append((_MONOTONIC(), cached.asr_model, is_cuda_device(cached.device)))
    if len(_ASR_UNLOAD_EVENTS) > 32:
        del _ASR_UNLOAD_EVENTS[:-_ASR_ACTIVE_SLOTS]
    del cached  # Drop the last strong reference to the weights before collecting.
    gc.collect()
    release_gpu_memory()
    _logger().info("Released ASR model (%s) and its GPU memory", reason)
    return True


def unload_idle_models(*, idle_seconds: float | None = None) -> list[str]:
    """Release cached ASR models that have been unused for ``idle_seconds``.

    Returns the model names that were released. The reaper calls this on a timer;
    tests and tooling can call it directly.
    """
    limit = asr_model_idle_unload_seconds() if idle_seconds is None else idle_seconds
    if limit is None or limit <= 0:
        return []
    now = _MONOTONIC()
    released: list[str] = []
    for slot, cached in enumerate(_ASR_SLOTS):
        if cached is None or now - cached.last_used < limit:
            continue
        name = cached.asr_model
        if _unload_slot(slot, reason=f"idle for {now - cached.last_used:.0f}s"):
            released.append(name)
    return released


def unload_all_models() -> list[str]:
    """Release every cached ASR model (used for explicit shutdown requests)."""
    released: list[str] = []
    for slot, cached in enumerate(_ASR_SLOTS):
        if cached is None:
            continue
        name = cached.asr_model
        if _unload_slot(slot, reason="explicit release"):
            released.append(name)
    return released


def release_gpu_memory() -> None:
    """Collect garbage and hand cached CUDA blocks back to the driver.

    Safe to call from the idle reaper: it waits for the serialized ASR slot so it
    never collects torch tensors while a transcription is still using them.
    """
    acquired = _GPU_ASR_SEMAPHORE.acquire(blocking=False)
    try:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if not torch.cuda.is_available():
                return
            for index in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(index):
                        torch.cuda.synchronize(index)
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                except Exception:  # One unreachable device must not block the rest.
                    continue
        except Exception as exc:  # Never let cleanup hide the original failure.
            _logger().warning("Failed to release CUDA memory: %s", exc)
    finally:
        if acquired:
            _GPU_ASR_SEMAPHORE.release()


def is_cuda_device(device: str | None) -> bool:
    return bool(device) and str(device).lower().startswith("cuda")


def _start_unload_reaper() -> None:
    global _ASR_REAPER_STARTED, _ASR_REAPER_WAKE
    if _ASR_REAPER_STARTED or asr_model_idle_unload_seconds() is None:
        return
    with _ASR_REAPER_LOCK:
        if _ASR_REAPER_STARTED:
            return
        _ASR_REAPER_WAKE = Event()
        Thread(
            target=_unload_reaper_loop,
            name="deepfind-asr-idle-reaper",
            daemon=True,
        ).start()
        _ASR_REAPER_STARTED = True


def _unload_reaper_loop() -> None:
    # Poll once a minute; unload_idle_models() compares against the per-model TTL.
    interval = 60.0
    while True:
        _ASR_REAPER_WAKE.wait(interval)
        _ASR_REAPER_WAKE.clear()
        try:
            unload_idle_models()
        except Exception:  # The reaper must survive anything a single poll raises.
            _logger().exception("Idle ASR model reaper failed")


def is_qwen3_asr_model(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name


def is_mlx_whisper_model(model_name: str) -> bool:
    """Check if model name indicates MLX Whisper backend."""
    prefix = model_name.split(":")[0] if ":" in model_name else model_name.split("/")[0]
    return prefix.lower() in MLX_WHISPER_MODELS


def _candidate_hf_cache_dirs() -> list[Path]:
    cache_dirs: list[Path] = []
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    env_home = os.environ.get("HF_HOME")
    if env_cache:
        cache_dirs.append(Path(env_cache).expanduser())
    if env_home:
        cache_dirs.append(Path(env_home).expanduser() / "hub")
    cache_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")

    unique_dirs: list[Path] = []
    for cache_dir in cache_dirs:
        if cache_dir not in unique_dirs:
            unique_dirs.append(cache_dir)
    return unique_dirs


def _resolve_cached_hf_snapshot(model_name: str) -> Path | None:
    if "/" not in model_name:
        return None

    repo_dirname = f"models--{model_name.replace('/', '--')}"
    for cache_dir in _candidate_hf_cache_dirs():
        repo_dir = cache_dir / repo_dirname
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        revision = load_text(repo_dir / "refs" / "main")
        if revision:
            snapshot_dir = snapshots_dir / revision
            if snapshot_dir.is_dir():
                return snapshot_dir

        snapshot_candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if len(snapshot_candidates) == 1:
            return snapshot_candidates[0]

    return None


def resolve_model_source(model_name: str) -> str:
    candidate_path = Path(model_name).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())

    cached_snapshot = _resolve_cached_hf_snapshot(model_name)
    if cached_snapshot is not None:
        return str(cached_snapshot)

    return model_name


def load_model(model_name: str) -> tuple[str, Any, Any, str]:
    load_local_secrets()
    hf_token = os.environ.get("HF_TOKEN")

    # MLX Whisper backend for Apple Silicon
    if is_mlx_whisper_model(model_name):
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise TranscriptionError(
                "MLX Whisper ASR requires an Apple Silicon Mac with MLX/Metal support."
            )
        try:
            import mlx_whisper
        except ImportError as exc:
            raise MissingDependencyError(
                "mlx-whisper is not installed. Install with: uv sync --extra media"
            ) from exc

        # Extract model size from name (e.g., "mlx-whisper:large-v3" -> "large-v3")
        model_size = model_name.split(":", 1)[1] if ":" in model_name else "base"
        _logger().info("Loading MLX Whisper model '%s' on Apple Metal", model_size)

        # MLX Whisper doesn't return a traditional model object, it uses a functional API
        # We'll store the model size and use it in transcribe_audio
        return "mlx_whisper", model_size, None, "mps"

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "ASR dependencies are missing. Install with: uv sync --extra media"
        ) from exc

    if not torch.cuda.is_available():
        raise TranscriptionError(
            "CUDA ASR requires an NVIDIA GPU and a CUDA-enabled PyTorch build; "
            f"installed torch reports CUDA {torch.version.cuda or 'unavailable'}."
        )

    device = "cuda"
    dtype = torch.float16

    source = resolve_model_source(model_name)
    if source != model_name:
        _logger().info("Using cached ASR model '%s' from %s", model_name, source)
    _logger().info("Loading ASR model '%s' on %s", model_name, device.upper())

    if is_qwen3_asr_model(model_name):
        try:
            from qwen_asr import QwenASR
        except ImportError:
            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise MissingDependencyError(
                    "qwen_asr is not installed. Install with: uv sync --extra media"
                ) from exc

            try:
                model = Qwen3ASRModel.from_pretrained(
                    source,
                    dtype=torch.bfloat16,
                    device_map="cuda:0",
                )
            except Exception as exc:
                raise TranscriptionError(f"Failed to load ASR model '{model_name}': {exc}") from exc
            return "qwen3_asr", model, None, device

        try:
            model = QwenASR.from_pretrained(
                source,
                torch_dtype="bfloat16",
                device=device,
            )
        except Exception as exc:
            raise TranscriptionError(f"Failed to load ASR model '{model_name}': {exc}") from exc
        return "qwen3_asr", model, None, device

    try:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers is not installed. Install with: uv sync --extra media"
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(source, token=hf_token)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            source,
            token=hf_token,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)
    except Exception as exc:
        raise TranscriptionError(f"Failed to load ASR model '{model_name}': {exc}") from exc

    return "qwen2_audio", model, processor, device


def extract_qwen3_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        return "\n\n".join(filter(None, (extract_qwen3_text(item) for item in result))).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()


def transcribe_audio(audio_path: Path, backend: str, model: Any, processor: Any, device: str) -> str:
    if backend == "qwen3_asr":
        return extract_qwen3_text(model.transcribe(str(audio_path)))

    if backend == "mlx_whisper":
        try:
            import mlx_whisper
        except ImportError as exc:
            raise MissingDependencyError(
                "mlx-whisper is not installed. Install with: uv sync --extra media"
            ) from exc

        # model here is actually the model_size string (e.g., "large-v3")
        # MLX Whisper expects full HF repo path like "mlx-community/whisper-large-v3"
        model_size = model
        hf_repo = f"mlx-community/whisper-{model_size}"

        # Try to use cached model first
        cached_path = resolve_model_source(hf_repo)
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=cached_path)
        return result.get("text", "").strip()

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "torch is not installed. Install with: uv sync --extra media"
        ) from exc

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": "Transcribe this audio accurately. Return only the spoken content."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, audios=[str(audio_path)], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def transcribe_segments(
    segments: Sequence[Path],
    *,
    asr_model: str = DEFAULT_ASR_MODEL,
) -> str:
    with gpu_asr_slot():
        # Reuse the slot's resident model so back-to-back jobs skip the ~GB load.
        backend, model, processor, device = _asr_slot_model(slot_index(), asr_model)

        transcripts: list[str] = []
        for segment in segments:
            try:
                segment_text = transcribe_audio(segment, backend, model, processor, device)
            except (MissingDependencyError, TranscriptionError):
                raise
            except Exception as exc:
                raise TranscriptionError(f"Failed transcribing {segment.name}: {exc}") from exc
            if segment_text:
                transcripts.append(segment_text)

    transcript = "\n".join(transcripts).strip()
    if not transcript:
        raise TranscriptionError("ASR produced an empty transcript.")
    return transcript


__all__ = [
    "AUDIO_SUFFIXES",
    "DEFAULT_ASR_IDLE_UNLOAD_SECONDS",
    "DEFAULT_ASR_MODEL",
    "MLX_WHISPER_MODELS",
    "MissingDependencyError",
    "SEGMENT_SECONDS",
    "TranscriptionError",
    "asr_model_idle_unload_seconds",
    "extract_qwen3_text",
    "gpu_asr_slot",
    "is_cuda_device",
    "is_qwen3_asr_model",
    "is_mlx_whisper_model",
    "load_text",
    "load_local_secrets",
    "load_model",
    "release_gpu_memory",
    "resolve_audio_root",
    "resolve_model_source",
    "slot_index",
    "transcribe_segments",
    "transcribe_audio",
    "unload_all_models",
    "unload_idle_models",
    "write_text",
]
