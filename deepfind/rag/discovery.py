"""Source-file discovery, parsed-dir mapping, fingerprints and metadata I/O.

This module also absorbs the former ``manifest`` responsibilities: reading and
writing each parsed directory's ``metadata.json`` and comparing source
fingerprints. There is no central state DB; each parsed directory's
``metadata.json`` is the single source of truth for that source.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from .config import Config, get_config
from .models import Fingerprint, Metadata, SourceFile

PDF_SUFFIXES = {".pdf"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac"}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", "assets", ".work"}
TEMP_SUFFIXES = {".tmp", ".part", ".crdownload", ".download", ".partial"}


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_posix_rel(path: Path, root_dir: Path) -> str:
    return path.resolve().relative_to(root_dir).as_posix()


def parsed_dir_for(source: Path) -> Path:
    """Sibling directory named after the file with its last suffix removed."""
    return source.with_suffix("")


def _is_temp(path: Path) -> bool:
    return path.suffix.lower() in TEMP_SUFFIXES or path.name.startswith("~")


def _iter_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden and excluded directories in place
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d not in EXCLUDED_DIR_NAMES
        ]
        current = Path(dirpath)
        for name in filenames:
            yield current / name


def _has_sibling_source(file_path: Path, media: bool) -> bool:
    """A parsed dir is named exactly like a sibling source file (sans suffix).

    We skip any file that lives under such a directory so parsed artifacts are
    never rescanned as if they were sources.
    """
    return False  # handled by suffix filtering below


def discover(
    source_type: str | None = None,
    path_filter: Path | None = None,
    *,
    config: Config | None = None,
) -> list[SourceFile]:
    """Return all source files, excluding parsed-dir artifacts.

    A file is a source only if its suffix is a recognized PDF/media type. Since
    parsed artifacts (raw.md, content.md, chunks.jsonl, assets, .work, audio
    segments) never share those source suffixes in a place that collides, suffix
    filtering plus directory pruning is sufficient.
    """
    config = config or get_config()
    results: list[SourceFile] = []

    def collect(root: Path, kind: str) -> None:
        for file_path in _iter_files(root):
            if _is_temp(file_path):
                continue
            suffix = file_path.suffix.lower()
            if kind == "pdf" and suffix not in PDF_SUFFIXES:
                continue
            if kind == "media" and suffix not in MEDIA_SUFFIXES:
                continue
            if path_filter is not None:
                try:
                    file_path.resolve().relative_to(path_filter.resolve())
                except ValueError:
                    continue
            if suffix in PDF_SUFFIXES:
                st = "pdf"
            elif suffix in VIDEO_SUFFIXES:
                st = "video"
            else:
                st = "audio"
            results.append(
                SourceFile(
                    path=to_posix_rel(file_path, config.root_dir),
                    parsed_dir=to_posix_rel(parsed_dir_for(file_path), config.root_dir),
                    source_type=st,  # type: ignore[arg-type]
                )
            )

    if source_type in (None, "pdf"):
        collect(config.pdf_root, "pdf")
    if source_type in (None, "media"):
        collect(config.media_root, "media")

    results.sort(key=lambda s: s.path)
    return results


def compute_fingerprint(source: Path) -> Fingerprint:
    sha = hashlib.sha256()
    with source.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(block)
    stat = source.stat()
    return Fingerprint(
        sha256=sha.hexdigest(), size=stat.st_size, mtime_ns=stat.st_mtime_ns
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def metadata_path(parsed_dir: Path) -> Path:
    return parsed_dir / "metadata.json"


def load_metadata(parsed_dir: Path) -> Metadata | None:
    mp = metadata_path(parsed_dir)
    if not mp.is_file():
        return None
    try:
        return Metadata.model_validate_json(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_metadata(parsed_dir: Path, meta: Metadata) -> None:
    atomic_write_text(metadata_path(parsed_dir), meta.model_dump_json(indent=2) + "\n")


def fingerprint_matches(meta: Metadata | None, fp: Fingerprint) -> bool:
    if meta is None:
        return False
    return (
        meta.source_sha256 == fp.sha256
        and meta.source_size == fp.size
        and meta.source_mtime_ns == fp.mtime_ns
    )
