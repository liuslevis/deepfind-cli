"""Reusable document discovery, ingestion, and pruning service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .discovery import discover
from .indexer import Indexer
from .pipeline import Result, process_source


@dataclass
class SyncResult:
    results: list[Result]
    pruned: list[str]

    @property
    def processed(self) -> int:
        return sum(result.status == "indexed" for result in self.results)

    @property
    def skipped(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def sync_documents(
    config: Config,
    *,
    source_type: str | None = None,
    path_filter: Path | None = None,
    prune: bool = False,
) -> SyncResult:
    """Process new or changed documents and optionally prune deleted sources."""
    sources = discover(source_type=source_type, path_filter=path_filter, config=config)
    if not sources and not prune:
        return SyncResult(results=[], pruned=[])

    indexer = Indexer(config)
    indexer.ensure_collection()
    results = [process_source(source, config, indexer) for source in sources]

    pruned: list[str] = []
    if prune:
        keep = {source.path for source in discover(config=config)}
        pruned = indexer.prune(keep)

    return SyncResult(results=results, pruned=pruned)
