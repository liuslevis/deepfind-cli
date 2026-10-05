from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from threading import Lock

from .asr import (
    AUDIO_SUFFIXES,
    DEFAULT_ASR_MODEL,
    SEGMENT_SECONDS,
    MissingDependencyError,
    TranscriptionError,
    gpu_asr_slot,
    load_model,
    transcribe_audio,
    resolve_audio_root,
    transcribe_segments,
    load_text,
    write_text,
)
from .youtube_audio_transcribe import resolve_ffmpeg_bin

BVID_PATTERN = re.compile(r"(BV[0-9A-Za-z]{10})")
_TRANSCRIPTION_LOCKS: dict[str, Lock] = {}
_TRANSCRIPTION_LOCKS_GUARD = Lock()


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


def find_source_audio(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_SUFFIXES
        and not path.stem.startswith("seg_")
    ]
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda path: path.stat().st_size)
    except OSError:
        return candidates[0]


def load_cached_transcript(audio_root: Path, bili_id: str) -> tuple[Path, str] | None:
    candidate = audio_root / "transcripts" / f"{bili_id}.txt"
    transcript = load_text(candidate)
    if transcript is None:
        return None
    return candidate, transcript


def _transcription_lock(bili_id: str) -> Lock:
    with _TRANSCRIPTION_LOCKS_GUARD:
        return _TRANSCRIPTION_LOCKS.setdefault(bili_id, Lock())


def ensure_segments(
    bili_id: str,
    output_dir: Path,
    bili_bin: str | None,
    timeout: int,
    ffmpeg_bin: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = find_segments(output_dir)
    if segments:
        return segments

    source_audio = find_source_audio(output_dir)
    if source_audio is None:
        resolved_bin = resolve_bili_bin(bili_bin)
        command = [resolved_bin, "audio", bili_id, "--no-split", "-o", str(output_dir)]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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

        source_audio = find_source_audio(output_dir)
        if source_audio is None:
            raise BiliDownloadError(f"bili finished but no audio file was created under {output_dir}")

    resolved_ffmpeg = resolve_ffmpeg_bin(ffmpeg_bin)
    segment_template = output_dir / "seg_%03d.wav"
    command = [
        resolved_ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SECONDS),
        "-reset_timestamps",
        "1",
        str(segment_template),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BiliDownloadError(f"ffmpeg segment timed out: {exc}") from exc
    except OSError as exc:
        raise MissingDependencyError(str(exc)) from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or "ffmpeg segment failed."
        raise BiliDownloadError(message[:4000])

    segments = find_segments(output_dir)
    if not segments:
        raise BiliDownloadError(f"No segmented audio files were created under {output_dir}")
    return segments


def transcribe_bili_audio(
    bili_id: str,
    *,
    bili_bin: str | None = None,
    ffmpeg_bin: str | None = None,
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

    with _transcription_lock(resolved_id):
        cached = load_cached_transcript(audio_root, resolved_id)
        if cached:
            transcript_path, transcript = cached
            return {
                "bili_id": resolved_id,
                "transcript_path": str(transcript_path),
                "transcript": transcript,
            }

        audio_dir_path = audio_root / resolved_id
        segments = ensure_segments(
            resolved_id,
            output_dir=audio_dir_path,
            bili_bin=bili_bin,
            timeout=timeout,
            ffmpeg_bin=ffmpeg_bin,
        )
        transcript = transcribe_segments(segments, asr_model=asr_model)

        transcript_path = audio_root / "transcripts" / f"{resolved_id}.txt"
        write_text(transcript_path, transcript)
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
