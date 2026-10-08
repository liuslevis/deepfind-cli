"""Media ingestion adapter built on DeepFind's shared ASR runtime."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ...asr import (
    MissingDependencyError,
    TranscriptionError,
    gpu_asr_slot,
    load_model,
    transcribe_audio,
)
from ..config import Config


class MediaError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_shared_model(model_name: str):
    return load_model(model_name)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _seconds_to_hms(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _probe_duration(ffmpeg_bin: str, path: Path) -> float | None:
    ffprobe = (
        ffmpeg_bin.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg_bin else "ffprobe"
    )
    proc = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _extract_audio(config: Config, source: Path, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            config.ffmpeg_bin,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            str(output),
        ]
    )
    if proc.returncode != 0 or not output.is_file():
        raise MediaError(
            f"ffmpeg audio extraction failed for {source.name}: {proc.stderr[-500:]}"
        )


def _segment_audio(config: Config, wav: Path, segment_dir: Path) -> list[Path]:
    segment_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(segment_dir.glob("seg_*.wav"))
    if existing:
        return existing
    proc = _run(
        [
            config.ffmpeg_bin,
            "-y",
            "-i",
            str(wav),
            "-f",
            "segment",
            "-segment_time",
            str(config.asr_segment_seconds),
            "-c",
            "copy",
            str(segment_dir / "seg_%05d.wav"),
        ]
    )
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg segmentation failed: {proc.stderr[-500:]}")
    segments = sorted(segment_dir.glob("seg_*.wav"))
    if not segments:
        raise MediaError("ffmpeg produced no audio segments")
    return segments


def _transcribe_segments(
    config: Config, segments: list[Path], checkpoint_dir: Path
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        segment
        for segment in segments
        if not (checkpoint_dir / f"{segment.stem}.txt").is_file()
    ]
    if not pending:
        return

    try:
        with gpu_asr_slot():
            backend, model, processor, device = _load_shared_model(config.asr_model)
            for segment in pending:
                text = transcribe_audio(segment, backend, model, processor, device)
                (checkpoint_dir / f"{segment.stem}.txt").write_text(
                    text.strip() + "\n",
                    encoding="utf-8",
                )
    except (MissingDependencyError, TranscriptionError) as exc:
        raise MediaError(str(exc)) from exc
    except Exception as exc:
        raise MediaError(f"ASR failed: {exc}") from exc


def transcribe_media(config: Config, source: Path, parsed_dir: Path) -> str:
    work_dir = parsed_dir / ".work"
    wav = work_dir / "audio.wav"
    _extract_audio(config, source, wav)
    segments = _segment_audio(config, wav, work_dir / "segments")
    checkpoint_dir = work_dir / "checkpoints"
    _transcribe_segments(config, segments, checkpoint_dir)

    blocks: list[str] = []
    for index, segment in enumerate(segments):
        text = (
            (checkpoint_dir / f"{segment.stem}.txt").read_text(encoding="utf-8").strip()
        )
        start = index * config.asr_segment_seconds
        duration = _probe_duration(config.ffmpeg_bin, segment)
        end = start + (duration or float(config.asr_segment_seconds))
        header = f"[{_seconds_to_hms(start)} - {_seconds_to_hms(end)}]"
        blocks.append(f"{header}\n{text}" if text else header)
    return "\n\n".join(blocks).strip() + "\n"


def cleanup_work(parsed_dir: Path) -> None:
    work_dir = parsed_dir / ".work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
