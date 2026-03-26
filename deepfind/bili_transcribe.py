from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
SEGMENT_SECONDS = 300
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac"}
BVID_PATTERN = re.compile(r"(BV[0-9A-Za-z]{10})")
REPO_ROOT = Path(__file__).resolve().parent.parent


class BiliTranscribeError(RuntimeError):
    """Base error for Bilibili transcription failures."""


class InvalidBiliIdError(BiliTranscribeError):
    """Raised when input does not contain a valid Bilibili BVID."""


class MissingDependencyError(BiliTranscribeError):
    """Raised when required binaries or Python dependencies are unavailable."""


class BiliDownloadError(BiliTranscribeError):
    """Raised when audio download fails."""


class TranscriptionError(BiliTranscribeError):
    """Raised when ASR model loading or transcription fails."""


def parse_bili_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise InvalidBiliIdError("bili_id cannot be empty.")

    match = BVID_PATTERN.search(raw)
    if not match:
        raise InvalidBiliIdError(
            "Invalid bili_id. Provide a Bilibili URL or BVID like BV1cgPSzeEj5."
        )
    return match.group(1)


def resolve_audio_root(audio_dir: str | None) -> Path:
    raw = (audio_dir or "audio").strip() or "audio"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def resolve_bili_bin(configured_bin: str | None) -> str:
    candidates: list[Path] = []
    if configured_bin:
        configured = configured_bin.strip()
        if configured:
            candidates.append(Path(configured).expanduser())
            which_configured = shutil.which(configured)
            if which_configured:
                candidates.append(Path(which_configured))

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "uv" / "tools" / "bilibili-cli" / "Scripts" / "bili.exe")
        candidates.append(Path.home() / ".local" / "bin" / "bili.exe")
    else:
        candidates.append(Path.home() / ".local" / "bin" / "bili")

    which_default = shutil.which("bili")
    if which_default:
        candidates.append(Path(which_default))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise MissingDependencyError(
        "bili CLI not found. Install bilibili-cli or set BILI_BIN to the executable path."
    )


def find_segments(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and path.stem.startswith("seg_")
    )


def load_cached_transcript(audio_root: Path, bili_id: str) -> tuple[Path, str] | None:
    candidate = audio_root / "transcripts" / f"{bili_id}.txt"
    if not candidate.is_file():
        return None
    try:
        transcript = candidate.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if transcript:
        return candidate, transcript
    return None


def ensure_segments(
    bili_id: str,
    output_dir: Path,
    bili_bin: str | None,
    timeout: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = find_segments(output_dir)
    if segments:
        return segments

    resolved_bin = resolve_bili_bin(bili_bin)
    command = [resolved_bin, "audio", bili_id, "--segment", str(SEGMENT_SECONDS), "-o", str(output_dir)]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BiliDownloadError(f"bili audio download timed out: {exc}") from exc
    except OSError as exc:
        raise MissingDependencyError(str(exc)) from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or "bili audio download failed."
        raise BiliDownloadError(message[:4000])

    segments = find_segments(output_dir)
    if not segments:
        raise BiliDownloadError(f"No segmented audio files were created under {output_dir}")
    return segments


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


def transcribe_bili_audio(
    bili_id: str,
    *,
    bili_bin: str | None = None,
    asr_model: str = DEFAULT_ASR_MODEL,
    audio_dir: str | None = None,
    timeout: int = 90,
) -> dict[str, str]:
    resolved_id = parse_bili_id(bili_id)
    audio_root = resolve_audio_root(audio_dir)
    cached = load_cached_transcript(audio_root, resolved_id)
    if cached:
        transcript_path, transcript = cached
        return {
            "bili_id": resolved_id,
            "transcript_path": str(transcript_path),
            "transcript": transcript,
        }

    audio_dir_path = audio_root / resolved_id
    transcript_root = audio_root / "transcripts"
    transcript_root.mkdir(parents=True, exist_ok=True)

    segments = ensure_segments(
        resolved_id,
        output_dir=audio_dir_path,
        bili_bin=bili_bin,
        timeout=timeout,
    )
    backend, model, processor, device = load_model(asr_model)

    transcripts: list[str] = []
    for segment in segments:
        try:
            segment_text = transcribe_audio(segment, backend, model, processor, device)
        except BiliTranscribeError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed transcribing {segment.name}: {exc}") from exc
        if segment_text:
            transcripts.append(segment_text)

    transcript = "\n".join(transcripts).strip()
    if not transcript:
        raise TranscriptionError("ASR produced an empty transcript.")

    transcript_path = transcript_root / f"{resolved_id}.txt"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    return {
        "bili_id": resolved_id,
        "transcript_path": str(transcript_path),
        "transcript": transcript,
    }


__all__ = [
    "BiliDownloadError",
    "BiliTranscribeError",
    "DEFAULT_ASR_MODEL",
    "InvalidBiliIdError",
    "MissingDependencyError",
    "TranscriptionError",
    "parse_bili_id",
    "resolve_audio_root",
    "transcribe_bili_audio",
]
