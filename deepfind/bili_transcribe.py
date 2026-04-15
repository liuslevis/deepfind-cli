from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .asr import (
    AUDIO_SUFFIXES,
    DEFAULT_ASR_MODEL,
    SEGMENT_SECONDS,
    MissingDependencyError,
    TranscriptionError,
    gpu_asr_slot,
    load_model,
    resolve_audio_root,
    transcribe_audio,
)

BVID_PATTERN = re.compile(r"(BV[0-9A-Za-z]{10})")


class BiliTranscribeError(RuntimeError):
    """Base error for Bilibili transcription failures."""


class InvalidBiliIdError(BiliTranscribeError):
    """Raised when input does not contain a valid Bilibili BVID."""


class BiliDownloadError(BiliTranscribeError):
    """Raised when audio download fails."""


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
    with gpu_asr_slot():
        backend, model, processor, device = load_model(asr_model)

        transcripts: list[str] = []
        for segment in segments:
            try:
                segment_text = transcribe_audio(segment, backend, model, processor, device)
            except (BiliTranscribeError, MissingDependencyError, TranscriptionError):
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
