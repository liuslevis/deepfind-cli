from __future__ import annotations

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
from .youtube_audio_transcribe import resolve_ffmpeg_bin


class XhsTranscribeError(RuntimeError):
    """Base error for Xiaohongshu transcription failures."""


class InvalidXhsVideoError(XhsTranscribeError):
    """Raised when note metadata is missing a usable video stream."""


class XhsDownloadError(XhsTranscribeError):
    """Raised when ffmpeg cannot fetch or segment a Xiaohongshu video."""


def resolve_xhs_transcript_path(audio_root: Path, note_id: str) -> Path:
    return audio_root / "transcripts" / "xhs" / f"{note_id}.txt"


def load_cached_xhs_transcript(audio_root: Path, note_id: str) -> tuple[Path, str] | None:
    candidate = resolve_xhs_transcript_path(audio_root, note_id)
    if not candidate.is_file():
        return None
    try:
        transcript = candidate.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if transcript:
        return candidate, transcript
    return None


def store_xhs_transcript(audio_root: Path, note_id: str, transcript: str) -> Path:
    path = resolve_xhs_transcript_path(audio_root, note_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.strip() + "\n", encoding="utf-8")
    return path


def _find_segments(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and path.stem.startswith("seg_")
    )


def ensure_xhs_audio_segments(
    note_id: str,
    video_url: str,
    output_dir: Path,
    *,
    ffmpeg_bin: str | None,
    timeout: int,
) -> list[Path]:
    resolved_note_id = note_id.strip()
    resolved_video_url = video_url.strip()
    if not resolved_note_id:
        raise InvalidXhsVideoError("xhs note_id cannot be empty.")
    if not resolved_video_url:
        raise InvalidXhsVideoError("xhs video_url cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = _find_segments(output_dir)
    if segments:
        return segments

    resolved_ffmpeg = resolve_ffmpeg_bin(ffmpeg_bin)
    segment_template = output_dir / "seg_%03d.wav"
    ffmpeg_command = [
        resolved_ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        resolved_video_url,
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
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise XhsDownloadError(f"xhs ffmpeg segment timed out: {exc}") from exc
    except OSError as exc:
        raise MissingDependencyError(str(exc)) from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or "ffmpeg segment failed."
        raise XhsDownloadError(message[:4000])

    segments = _find_segments(output_dir)
    if not segments:
        raise XhsDownloadError(f"No segmented audio files were created under {output_dir}")
    return segments


def transcribe_xhs_video(
    note_id: str,
    video_url: str,
    *,
    ffmpeg_bin: str | None = None,
    asr_model: str = DEFAULT_ASR_MODEL,
    audio_dir: str | None = None,
    timeout: int = 90,
) -> dict[str, str]:
    resolved_note_id = note_id.strip()
    if not resolved_note_id:
        raise InvalidXhsVideoError("xhs note_id cannot be empty.")

    audio_root = resolve_audio_root(audio_dir)
    cached = load_cached_xhs_transcript(audio_root, resolved_note_id)
    if cached:
        transcript_path, transcript = cached
        return {
            "note_id": resolved_note_id,
            "transcript_path": str(transcript_path),
            "transcript": transcript,
        }

    audio_dir_path = audio_root / "xhs" / resolved_note_id
    segments = ensure_xhs_audio_segments(
        resolved_note_id,
        video_url,
        output_dir=audio_dir_path,
        ffmpeg_bin=ffmpeg_bin,
        timeout=timeout,
    )
    with gpu_asr_slot():
        backend, model, processor, device = load_model(asr_model)
        transcripts: list[str] = []
        for segment in segments:
            try:
                segment_text = transcribe_audio(segment, backend, model, processor, device)
            except XhsTranscribeError:
                raise
            except Exception as exc:
                raise TranscriptionError(f"Failed transcribing {segment.name}: {exc}") from exc
            if segment_text:
                transcripts.append(segment_text)

    transcript = "\n".join(transcripts).strip()
    if not transcript:
        raise TranscriptionError("ASR produced an empty transcript.")

    transcript_path = store_xhs_transcript(audio_root, resolved_note_id, transcript)
    return {
        "note_id": resolved_note_id,
        "transcript_path": str(transcript_path),
        "transcript": transcript,
    }


__all__ = [
    "InvalidXhsVideoError",
    "XhsDownloadError",
    "XhsTranscribeError",
    "load_cached_xhs_transcript",
    "resolve_xhs_transcript_path",
    "store_xhs_transcript",
    "transcribe_xhs_video",
]
