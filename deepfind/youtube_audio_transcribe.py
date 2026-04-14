from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .bili_transcribe import (
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
from .youtube_transcribe import InvalidYouTubeIdError, parse_youtube_id


class YouTubeAudioTranscribeError(RuntimeError):
    """Base error for YouTube audio transcription failures."""


class YouTubeDownloadError(YouTubeAudioTranscribeError):
    """Raised when yt-dlp download or ffmpeg segmentation fails."""


def resolve_youtube_audio_transcript_path(audio_root: Path, youtube_id: str) -> Path:
    return audio_root / "transcripts" / "youtube_audio" / f"{youtube_id}.txt"


def load_cached_youtube_audio_transcript(audio_root: Path, youtube_id: str) -> tuple[Path, str] | None:
    candidate = resolve_youtube_audio_transcript_path(audio_root, youtube_id)
    if not candidate.is_file():
        return None
    try:
        transcript = candidate.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if transcript:
        return candidate, transcript
    return None


def store_youtube_audio_transcript(audio_root: Path, youtube_id: str, transcript: str) -> Path:
    path = resolve_youtube_audio_transcript_path(audio_root, youtube_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.strip() + "\n", encoding="utf-8")
    return path


def _find_segments(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and path.stem.startswith("seg_")
    )


def resolve_ytdlp_bin(configured_bin: str | None) -> str:
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
            candidates.append(Path(appdata) / "uv" / "tools" / "yt-dlp" / "Scripts" / "yt-dlp.exe")
        candidates.append(Path.home() / ".local" / "bin" / "yt-dlp.exe")
    else:
        candidates.append(Path.home() / ".local" / "bin" / "yt-dlp")

    which_default = shutil.which("yt-dlp")
    if which_default:
        candidates.append(Path(which_default))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise MissingDependencyError(
        "yt-dlp not found. Install yt-dlp (pipx/uv tool/pip) or set YTDLP_BIN to the executable path."
    )


def resolve_ffmpeg_bin(configured_bin: str | None) -> str:
    candidates: list[Path] = []
    if configured_bin:
        configured = configured_bin.strip()
        if configured:
            candidates.append(Path(configured).expanduser())
            which_configured = shutil.which(configured)
            if which_configured:
                candidates.append(Path(which_configured))

    which_default = shutil.which("ffmpeg")
    if which_default:
        candidates.append(Path(which_default))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise MissingDependencyError(
        "ffmpeg not found. Install ffmpeg and ensure it is on PATH, or set FFMPEG_BIN to the executable path."
    )


def _resolve_source_audio(output_dir: Path) -> Path | None:
    excluded_suffixes = {".part", ".tmp", ".ytdl", ".json"}
    candidates: list[Path] = []
    for path in output_dir.glob("source.*"):
        if not path.is_file():
            continue
        if path.name.endswith(".info.json"):
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        candidates.append(path)

    if not candidates:
        return None
    try:
        return max(candidates, key=lambda item: item.stat().st_size)
    except OSError:
        return candidates[0]


def ensure_youtube_audio_segments(
    youtube_id: str,
    output_dir: Path,
    *,
    ytdlp_bin: str | None,
    ffmpeg_bin: str | None,
    timeout: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = _find_segments(output_dir)
    if segments:
        return segments

    resolved_ytdlp = resolve_ytdlp_bin(ytdlp_bin)
    resolved_ffmpeg = resolve_ffmpeg_bin(ffmpeg_bin)
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    download_template = output_dir / "source.%(ext)s"
    download_command = [
        resolved_ytdlp,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-o",
        str(download_template),
        url,
    ]
    try:
        proc = subprocess.run(
            download_command,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise YouTubeDownloadError(f"yt-dlp download timed out: {exc}") from exc
    except OSError as exc:
        raise MissingDependencyError(str(exc)) from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or "yt-dlp download failed."
        raise YouTubeDownloadError(message[:4000])

    source_audio = _resolve_source_audio(output_dir)
    if not source_audio:
        raise YouTubeDownloadError(f"yt-dlp finished but no audio file was created under {output_dir}")

    segment_template = output_dir / "seg_%03d.wav"
    ffmpeg_command = [
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
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise YouTubeDownloadError(f"ffmpeg segment timed out: {exc}") from exc
    except OSError as exc:
        raise MissingDependencyError(str(exc)) from exc

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or "ffmpeg segment failed."
        raise YouTubeDownloadError(message[:4000])

    segments = _find_segments(output_dir)
    if not segments:
        raise YouTubeDownloadError(f"No segmented audio files were created under {output_dir}")

    try:
        source_audio.unlink()
    except OSError:
        pass

    return segments


def transcribe_youtube_audio(
    url: str,
    *,
    ytdlp_bin: str | None = None,
    ffmpeg_bin: str | None = None,
    asr_model: str = DEFAULT_ASR_MODEL,
    audio_dir: str | None = None,
    timeout: int = 90,
) -> dict[str, str]:
    youtube_id = parse_youtube_id(url)
    audio_root = resolve_audio_root(audio_dir)
    cached = load_cached_youtube_audio_transcript(audio_root, youtube_id)
    if cached:
        transcript_path, transcript = cached
        return {
            "youtube_id": youtube_id,
            "transcript_path": str(transcript_path),
            "transcript": transcript,
        }

    audio_dir_path = audio_root / "youtube" / youtube_id
    segments = ensure_youtube_audio_segments(
        youtube_id,
        output_dir=audio_dir_path,
        ytdlp_bin=ytdlp_bin,
        ffmpeg_bin=ffmpeg_bin,
        timeout=timeout,
    )

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

    transcript_path = store_youtube_audio_transcript(audio_root, youtube_id, transcript)
    return {
        "youtube_id": youtube_id,
        "transcript_path": str(transcript_path),
        "transcript": transcript,
    }


__all__ = [
    "InvalidYouTubeIdError",
    "MissingDependencyError",
    "TranscriptionError",
    "YouTubeAudioTranscribeError",
    "YouTubeDownloadError",
    "ensure_youtube_audio_segments",
    "load_cached_youtube_audio_transcript",
    "resolve_audio_root",
    "resolve_ffmpeg_bin",
    "resolve_youtube_audio_transcript_path",
    "resolve_ytdlp_bin",
    "store_youtube_audio_transcript",
    "transcribe_youtube_audio",
]

