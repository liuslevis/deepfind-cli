from __future__ import annotations

import argparse
import sys
from typing import Callable, TextIO

from .config import Settings
from .orchestrator import ChatSession, DeepFind
from .output import render_answer, render_json_answer
from .progress import ConsoleProgress
from .tools import Toolset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfind")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--num-agent", type=int, default=2, help="1-4 sub agents")
    parser.add_argument(
        "--max-iter-per-agent",
        type=int,
        default=50,
        help="max response/tool rounds per agent",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable progress output",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="always exit after the first answer",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="list available deepfind tools and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print structured research JSON instead of human-readable Markdown",
    )
    parser.add_argument(
        "--long-report-mode",
        action="store_true",
        help="use long-form final report output instead of the default concise overview",
    )
    return parser


def _isatty(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _should_enter_chat_mode(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> bool:
    return not args.once and not args.json and _isatty(stdin) and _isatty(stdout)


def _print_tools(stdout: TextIO) -> None:
    toolset = Toolset(Settings(api_key=""))
    for spec in toolset.specs():
        function = spec.get("function", {})
        name = str(function.get("name", "")).strip()
        description = str(function.get("description", "")).strip()
        if name:
            print(f"{name}\t{description}", file=stdout)


def _chat_loop(
    session: ChatSession,
    *,
    stdout: TextIO,
    stderr: TextIO,
    input_fn: Callable[[str], str],
) -> int:
    while True:
        try:
            raw_query = input_fn("deepfind> ")
        except (EOFError, KeyboardInterrupt):
            return 0

        query = raw_query.strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0

        try:
            answer = session.ask(query)
        except Exception as exc:
            print(f"error: {exc}", file=stderr)
            continue

        render_answer(answer, stream=stdout)


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_tools:
        _print_tools(stdout)
        return 0
    if not args.query:
        parser.error("query is required unless --list-tools is set")
    if not 1 <= args.num_agent <= 4:
        parser.error("--num-agent must be between 1 and 4")
    if args.max_iter_per_agent < 1:
        parser.error("--max-iter-per-agent must be >= 1")

    try:
        app = DeepFind(progress=ConsoleProgress(enabled=not args.quiet, stream=stderr))
        session = app.session(
            num_agent=args.num_agent,
            max_iter_per_agent=args.max_iter_per_agent,
            long_report_mode=args.long_report_mode,
        )
        answer = session.ask_detailed(args.query) if args.json else session.ask(args.query)
    except Exception as exc:
        print(f"error: {exc}", file=stderr)
        return 1

    if args.json:
        render_json_answer(answer, stream=stdout)
    else:
        render_answer(answer, stream=stdout)
    if not _should_enter_chat_mode(args, stdin, stdout):
        return 0
    return _chat_loop(session, stdout=stdout, stderr=stderr, input_fn=input_fn)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
