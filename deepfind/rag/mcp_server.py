"""Stdio MCP server exposing a single ``search`` tool to Code Agents."""

from __future__ import annotations

import logging
import sys
import threading
from typing import Literal

from .config import Config, get_config
from .models import SearchResponse
from .search import SearchError, SearchService
from .sync_service import sync_documents

LOGGER = logging.getLogger("deepfind.rag.mcp")

_SEARCH_DOC = """Search the local investment-research knowledge base and return matching source passages.

Guidance for the calling agent:
- Prefer mode="sparse" for company names, tickers, numbers and financial terms.
- Prefer the default mode="hybrid" for concepts, paraphrases and natural-language questions.
- Cite PDF results by page (page_start/page_end); cite media results by time range
  (start_seconds/end_seconds).

Args:
  query: natural-language or keyword query (must be non-empty).
  limit: number of results, 1..20 (default 8).
  mode: "hybrid" | "dense" | "sparse".
  source_type: optional filter "pdf" | "video" | "audio".
  path_prefix: optional relative path under pdf/ or media/ (no absolute paths, no '..').
  tags: optional list of tags to require.
"""


def _build_server():
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # mcp < 2.x
        from mcp.server.fastmcp import FastMCP as _Server

    server = _Server("deepfind-rag")
    service = SearchService(get_config())

    @server.tool(description=_SEARCH_DOC)
    def search(
        query: str,
        limit: int = 8,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        source_type: Literal["pdf", "video", "audio"] | None = None,
        path_prefix: str | None = None,
        tags: list[str] | None = None,
    ) -> SearchResponse:
        try:
            return service.search(
                query,
                limit=limit,
                mode=mode,
                source_type=source_type,
                path_prefix=path_prefix,
                tags=tags,
            )
        except SearchError as exc:
            raise ValueError(str(exc)) from exc

    return server


def _auto_ingest_loop(stop: threading.Event, interval: int, config: Config) -> None:
    while not stop.is_set():
        try:
            result = sync_documents(config)
            if result.processed or result.failed:
                LOGGER.info(
                    "Automatic ingest finished: %d processed, %d skipped, %d failed",
                    result.processed,
                    result.skipped,
                    result.failed,
                )
        except Exception:
            LOGGER.exception("Automatic ingest scan failed")
        stop.wait(interval)


def main(*, auto_ingest: bool = True, interval: int = 60) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = get_config()
    stop = threading.Event()
    worker = None
    if auto_ingest:
        worker = threading.Thread(
            target=_auto_ingest_loop,
            args=(stop, interval, config),
            name="deepfind-rag-auto-ingest",
            daemon=True,
        )
        worker.start()
        LOGGER.info("Automatic ingest enabled (scan interval: %ds)", interval)

    server = _build_server()
    try:
        server.run()
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=1)


if __name__ == "__main__":
    main()
