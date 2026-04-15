from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .asr import load_text, resolve_audio_root, write_text


YOUTUBE_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{11}$")
YOUTUBE_URL_PATTERN = re.compile(
    r"(?:youtu\.be/|youtube(?:-nocookie)?\.com/(?:watch\?.*?[?&]v=|embed/|shorts/|live/))([0-9A-Za-z_-]{11})"
)


class YouTubeTranscribeError(RuntimeError):
    """Base error for YouTube transcript failures."""


class InvalidYouTubeIdError(YouTubeTranscribeError):
    """Raised when input does not contain a valid YouTube video ID."""


def parse_youtube_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise InvalidYouTubeIdError("url cannot be empty.")

    if YOUTUBE_ID_PATTERN.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    candidate = _extract_youtube_id(parsed)
    if candidate and YOUTUBE_ID_PATTERN.fullmatch(candidate):
        return candidate

    match = YOUTUBE_URL_PATTERN.search(raw)
    if match:
        return match.group(1)

    raise InvalidYouTubeIdError(
        "Invalid YouTube URL. Provide a YouTube URL or video ID like dQw4w9WgXcQ."
    )


def _extract_youtube_id(parsed: Any) -> str:
    host = (getattr(parsed, "netloc", "") or "").lower()
    path = (getattr(parsed, "path", "") or "").strip("/")

    if host in {"youtu.be", "www.youtu.be"}:
        return path.split("/", 1)[0]

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if path == "watch":
            query = parse_qs(parsed.query)
            return (query.get("v") or [""])[0]
        for prefix in ("embed/", "shorts/", "live/"):
            if path.startswith(prefix):
                return path[len(prefix) :].split("/", 1)[0]

    return ""


def resolve_youtube_transcript_path(audio_root: Path, youtube_id: str) -> Path:
    return audio_root / "transcripts" / "youtube" / f"{youtube_id}.txt"


def load_cached_youtube_transcript(audio_root: Path, youtube_id: str) -> tuple[Path, str] | None:
    candidate = resolve_youtube_transcript_path(audio_root, youtube_id)
    transcript = load_text(candidate)
    if transcript is None:
        return None
    return candidate, transcript


def store_youtube_transcript(audio_root: Path, youtube_id: str, transcript: str) -> Path:
    path = resolve_youtube_transcript_path(audio_root, youtube_id)
    write_text(path, transcript)
    return path


def normalize_youtube_transcript(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, list):
        items: list[str] = []
        for item in data:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                speaker = str(item.get("speaker", "")).strip()
                timestamp = str(item.get("timestamp", "")).strip()
                prefix = ""
                if timestamp and speaker:
                    prefix = f"[{timestamp}] {speaker}: "
                elif timestamp:
                    prefix = f"[{timestamp}] "
                elif speaker:
                    prefix = f"{speaker}: "
                items.append(prefix + text)
                continue

            raw = str(item).strip()
            if raw:
                items.append(raw)
        return "\n\n".join(items).strip()

    if isinstance(data, dict):
        for key in ("transcript", "items", "segments", "data"):
            text = normalize_youtube_transcript(data.get(key))
            if text:
                return text
        text = str(data.get("text", "")).strip()
        if text:
            return text

    return ""


__all__ = [
    "InvalidYouTubeIdError",
    "YouTubeTranscribeError",
    "load_cached_youtube_transcript",
    "normalize_youtube_transcript",
    "parse_youtube_id",
    "resolve_audio_root",
    "resolve_youtube_transcript_path",
    "store_youtube_transcript",
]
