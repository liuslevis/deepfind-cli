from __future__ import annotations

import argparse
import sys

from .orchestrator import DeepFind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfind")
    parser.add_argument("query")
    parser.add_argument("--num-agent", type=int, default=2, help="1-4 sub agents")
    parser.add_argument(
        "--max-iter-per-agent",
        type=int,
        default=50,
        help="max response/tool rounds per agent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.num_agent <= 4:
        parser.error("--num-agent must be between 1 and 4")
    if args.max_iter_per_agent < 1:
        parser.error("--max-iter-per-agent must be >= 1")

    app = DeepFind()
    try:
        answer = app.run(
            query=args.query,
            num_agent=args.num_agent,
            max_iter_per_agent=args.max_iter_per_agent,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
