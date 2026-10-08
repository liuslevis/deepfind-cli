"""Command-line interface for the built-in RAG knowledge base."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TextIO

from .config import Config, load_config
from .sync_service import sync_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfind rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("ingest", "sync"):
        command = subparsers.add_parser(name)
        command.add_argument("--type", choices=("pdf", "media"))
        command.add_argument("--path")
        if name == "sync":
            command.add_argument("--prune", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument(
        "--mode", choices=("hybrid", "dense", "sparse"), default="hybrid"
    )
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--source-type", choices=("pdf", "video", "audio"))
    search.add_argument("--path-prefix")

    mcp = subparsers.add_parser("mcp")
    mcp.add_argument("--interval", type=int, default=60)
    mcp.add_argument("--no-auto-ingest", action="store_true")
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _path_filter(config: Config, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config.root_dir / path
    return path.resolve()


def _run_sync(
    args: argparse.Namespace,
    config: Config,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    result = sync_documents(
        config,
        source_type=args.type,
        path_filter=_path_filter(config, args.path),
        prune=bool(getattr(args, "prune", False)),
    )
    if not result.results:
        print("No source files found.", file=stderr)
    for item in result.results:
        detail = f"  ({item.detail})" if item.status == "failed" else ""
        print(f"{item.status:8s}  {item.source}{detail}", file=stdout)
    for source in result.pruned:
        print(f"pruned    {source}", file=stdout)
    print(
        f"Done: {result.processed} processed, "
        f"{result.skipped} skipped, {result.failed} failed.",
        file=stdout,
    )
    return 1 if result.failed else 0


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    config = load_config()
    _configure_logging()

    if args.command in {"ingest", "sync"}:
        return _run_sync(args, config, stdout=stdout, stderr=stderr)
    if args.command == "search":
        from .search import SearchError, SearchService

        try:
            response = SearchService(config).search(
                args.query,
                mode=args.mode,
                limit=args.limit,
                source_type=args.source_type,
                path_prefix=args.path_prefix,
            )
        except SearchError as exc:
            print(f"error: {exc}", file=stderr)
            return 2
        print(response.model_dump_json(indent=2), file=stdout)
        return 0
    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main(auto_ingest=not args.no_auto_ingest, interval=args.interval)
        return 0
    raise AssertionError(f"unsupported RAG command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
