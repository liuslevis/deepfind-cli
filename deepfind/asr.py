from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore
from typing import Any


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
SEGMENT_SECONDS = 300
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac"}
REPO_ROOT = Path(__file__).resolve().parent.parent
_GPU_ASR_SEMAPHORE = Semaphore(1)


class MissingDependencyError(RuntimeError):
    """Raised when required binaries or Python dependencies are unavailable."""


class TranscriptionError(RuntimeError):
    """Raised when ASR model loading or transcription fails."""


def resolve_audio_root(audio_dir: str | None) -> Path:
    raw = (audio_dir or "audio").strip() or "audio"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@contextmanager
def gpu_asr_slot():
    if not _gpu_available():
        yield
        return

    # Serialize GPU ASR jobs so concurrent requests queue instead of competing
    # for VRAM and model-load resources.
    _GPU_ASR_SEMAPHORE.acquire()
    try:
        yield
    finally:
        _GPU_ASR_SEMAPHORE.release()


def is_qwen3_asr_model(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name


def load_model(model_name: str) -> tuple[str, Any, Any, str]:
    load_local_secrets()
    hf_token = os.environ.get("HF_TOKEN")

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "ASR dependencies are missing. Install with: pip install 'deepfind-cli[media]'"
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    if is_qwen3_asr_model(model_name):
        try:
            from qwen_asr import QwenASR
        except ImportError:
            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise MissingDependencyError(
                    "qwen_asr is not installed. Install with: pip install 'deepfind-cli[media]'"
                ) from exc

            try:
                model = Qwen3ASRModel.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map="cuda:0" if device == "cuda" else "cpu",
                )
            except Exception as exc:
                raise TranscriptionError(f"Failed to load ASR model '{model_name}': {exc}") from exc
            return "qwen3_asr", model, None, device

        try:
            model = QwenASR.from_pretrained(
                model_name,
                torch_dtype="bfloat16" if device == "cuda" else "float32",
                device=device,
            )
        except Exception as exc:
            raise TranscriptionError(f"Failed to load ASR model '{model_name}': {exc}") from exc
        return "qwen3_asr", model, None, device

    try:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers is not installed. Install with: pip install 'deepfind-cli[media]'"
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
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

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "torch is not installed. Install with: pip install 'deepfind-cli[media]'"
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
        backend, model, processor, device = load_model(asr_model)

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
    "DEFAULT_ASR_MODEL",
    "MissingDependencyError",
    "SEGMENT_SECONDS",
    "TranscriptionError",
    "extract_qwen3_text",
    "gpu_asr_slot",
    "is_qwen3_asr_model",
    "load_text",
    "load_local_secrets",
    "load_model",
    "resolve_audio_root",
    "transcribe_segments",
    "transcribe_audio",
    "write_text",
]
