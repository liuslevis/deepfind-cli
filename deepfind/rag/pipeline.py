"""Per-source ingestion pipeline with incremental / idempotent processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunking import CHUNK_CONFIG_VERSION, chunk_content, read_chunks, write_chunks
from .config import Config
from .discovery import (
    atomic_write_text,
    compute_fingerprint,
    fingerprint_matches,
    load_metadata,
    now_iso,
    save_metadata,
)
from .formatter import format_document
from .indexer import Indexer
from .models import Metadata, SourceFile

LOGGER = logging.getLogger("deepfind.rag.pipeline")


@dataclass
class Result:
    source: str
    status: str  # indexed | skipped | failed
    detail: str = ""


def _raw_name(source_type: str) -> str:
    return "raw.md" if source_type == "pdf" else "raw.txt"


def process_source(
    sf: SourceFile,
    config: Config,
    indexer: Indexer | None,
) -> Result:
    source_path = config.root_dir / sf.path
    parsed_dir = config.root_dir / sf.parsed_dir
    raw_path = parsed_dir / _raw_name(sf.source_type)
    content_path = parsed_dir / "content.md"
    chunks_path = parsed_dir / "chunks.jsonl"

    fp = compute_fingerprint(source_path)
    meta = load_metadata(parsed_dir)

    fmt_ver = config.formatter.config_version
    emb_ver = config.embedding_config_version

    fp_ok = fingerprint_matches(meta, fp)
    fully_indexed = (
        meta is not None
        and fp_ok
        and meta.status == "indexed"
        and meta.formatter_config_version == fmt_ver
        and meta.embedding_config_version == emb_ver
        and meta.chunk_config_version == CHUNK_CONFIG_VERSION
    )
    if fully_indexed and indexer is not None:
        return Result(sf.path, "skipped", "unchanged")

    source_changed = not fp_ok
    formatter_changed = meta is None or meta.formatter_config_version != fmt_ver
    chunk_changed = meta is None or meta.chunk_config_version != CHUNK_CONFIG_VERSION
    embedding_changed = meta is None or meta.embedding_config_version != emb_ver

    need_raw = source_changed or not raw_path.exists()
    need_content = need_raw or formatter_changed or not content_path.exists()
    need_chunks = need_content or chunk_changed or not chunks_path.exists()
    need_index = (
        need_chunks
        or embedding_changed
        or source_changed
        or meta is None
        or meta.status != "indexed"
    )

    created_at = meta.created_at if (meta and not source_changed) else now_iso()
    fallback_title = source_path.stem

    def write_meta(status: str, error: str | None = None) -> None:
        m = Metadata(
            source=sf.path,
            source_type=sf.source_type,
            source_sha256=fp.sha256,
            source_size=fp.size,
            source_mtime_ns=fp.mtime_ns,
            parser="docling" if sf.source_type == "pdf" else "qwen3-asr",
            formatter_model=config.formatter.model,
            formatter_config_version=fmt_ver,
            embedding_model=config.dense_model,
            embedding_config_version=emb_ver,
            chunk_config_version=CHUNK_CONFIG_VERSION,
            created_at=created_at,
            updated_at=now_iso(),
            status=status,  # type: ignore[arg-type]
            error=error,
        )
        save_metadata(parsed_dir, m)

    try:
        parsed_dir.mkdir(parents=True, exist_ok=True)

        # 1. raw
        if need_raw:
            LOGGER.info("[%s] parsing -> %s", sf.path, raw_path.name)
            if sf.source_type == "pdf":
                from .ingest.pdf import parse_pdf

                raw_text = parse_pdf(source_path, parsed_dir)
            else:
                from .ingest.media import transcribe_media

                raw_text = transcribe_media(config, source_path, parsed_dir)
            atomic_write_text(raw_path, raw_text)
            write_meta("raw_ready")
        raw_text = raw_path.read_text(encoding="utf-8")

        # 2. content.md
        if need_content:
            LOGGER.info("[%s] formatting -> content.md", sf.path)
            content = format_document(
                raw_text=raw_text,
                source=sf.path,
                source_type=sf.source_type,
                fallback_title=fallback_title,
                created=created_at,
                updated=now_iso(),
                config=config,
            )
            atomic_write_text(content_path, content)
            write_meta("formatted")
        content_md = content_path.read_text(encoding="utf-8")

        # 3. chunks.jsonl
        if need_chunks:
            LOGGER.info("[%s] chunking -> chunks.jsonl", sf.path)
            chunks = chunk_content(
                content_md,
                source=sf.path,
                source_type=sf.source_type,
                source_sha256=fp.sha256,
            )
            if not chunks:
                raise ValueError("chunking produced no chunks")
            write_chunks(chunks_path, chunks)
            write_meta("chunked")
        else:
            chunks = read_chunks(chunks_path)

        # 4. index
        if need_index and indexer is not None:
            LOGGER.info("[%s] indexing %d chunks", sf.path, len(chunks))
            indexer.replace_source(sf.path, chunks)
            write_meta("indexed")
        elif indexer is None:
            write_meta("chunked")

        # media cleanup
        if sf.source_type != "pdf" and config.asr_cleanup_work:
            from .ingest.media import cleanup_work

            cleanup_work(parsed_dir)

        return Result(sf.path, "indexed" if indexer is not None else "chunked", "ok")

    except Exception as exc:  # noqa: BLE001 - source-level failure boundary
        LOGGER.error("[%s] failed: %s", sf.path, exc)
        try:
            write_meta("failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:  # pragma: no cover
            LOGGER.exception("[%s] failed to persist error metadata", sf.path)
        return Result(sf.path, "failed", str(exc))
