"""Record which Bilibili videos already have a summary and reached Kindle.

Read-only against the repo: it inspects ``assets/video/bili/<bvid>/text`` and
``assets/kd`` so the tracking table in ``skill/SUMMARY_BB.md`` can be rebuilt
from real state instead of memory. Pass ``--list`` to print only the table.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from deepfind.config import _load_dotenv
from deepfind.send_kindle import sanitize_filename_stem

# Load .env before importing the sender: this should run in the repo root, but a
# script launched from elsewhere would otherwise see an empty SMTP_USER.
_load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_ROOT = REPO_ROOT / "assets" / "video" / "bili"
KINDLE_DIR = REPO_ROOT / "assets" / "kd"

# Shared naming rule for Kindle titles; the sender script owns the convention.
sys.path.insert(0, str(REPO_ROOT / "tmp"))
try:
    from send_summaries_kindle import kindle_title  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - the helper can be cleaned out of tmp/
    def kindle_title(raw: str) -> tuple[str, bool]:
        return raw, False

# Titles come from ``bili user-videos`` / ``bili video``; the cache avoids
# re-hitting the API per run.
TITLES_CACHE = REPO_ROOT / "tmp" / "bili_titles.json"
# Uploads list newest first, so a candidate record can borrow the date from the
# nearest newer *Bloomberg* upload (other channels post at different times).
DEFAULT_DATE = ""
# The archive copy is written right after a successful SMTP send. A PDF that is
# older than the archive cannot be the one that was mailed.
ARCHIVE_GRACE_SECONDS = 300


def _cached_titles() -> dict[str, str]:
    if TITLES_CACHE.is_file():
        return json.loads(TITLES_CACHE.read_text(encoding="utf-8"))
    return {}


def records_from_listing(path: Path) -> list[tuple[str, str]]:
    """Merge a ``bili user-videos`` dump into the title cache.

    Returns every ``(bvid, title)`` in the dump, newest first, after filling the
    cache. Entries without a ``YYMMDD`` prefix inherit the date of the nearest
    newer Bloomberg upload so the Kindle title can still start with ``MM/DD``.
    """
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("ok"):
        return []
    cached = _cached_titles()
    inherited = DEFAULT_DATE
    pairs: list[tuple[str, str]] = []
    for item in data.get("data", []):
        bvid, title = item.get("bvid") or item.get("id"), (item.get("title") or "").strip()
        if not bvid or not title:
            continue
        date = re.match(r"^(\d{6})\b", title)
        bloomberg = "bloomberg" in title.lower()
        if date:
            inherited = date.group(1)
        elif inherited and (bloomberg or title[:2] in {"CV", "BB"}):
            title = f"{inherited} Bloomberg {title}"
        cached[bvid] = title
        pairs.append((bvid, title))
    TITLES_CACHE.write_text(json.dumps(cached, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return pairs


def _heading(summary: str) -> str:
    for line in summary.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _archive_stem(title: str) -> str:
    return sanitize_filename_stem(f"Bloomberg {title}")


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def inspect(bvids: list[str]) -> list[dict[str, object]]:
    """Report state for each BVID: summary, PDF, and matched Kindle archive."""
    _load_dotenv()
    titles = _cached_titles()
    rows: list[dict[str, object]] = []
    for bvid in bvids:
        text_dir = VIDEO_ROOT / bvid / "text"
        summary_path = text_dir / "summary.md"
        summary = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
        upload_title = titles.get(bvid) or _heading(summary) or bvid
        title, _ = kindle_title(upload_title)

        pdf = _newest(sorted(text_dir.glob("*.pdf"))) if text_dir.is_dir() else None
        # Prefer the archive named after the Kindle title; fall back to older
        # names (upload title, summary heading, "Bloomberg <upload title>").
        candidates = [_archive_stem(title), _archive_stem(upload_title)]
        if heading := _heading(summary):
            candidates.extend([_archive_stem(heading), _archive_stem(f"Bloomberg {heading}")])
        # The sender names the archive with the bare Kindle title (no prefix).
        candidates.append(sanitize_filename_stem(title))
        archive = next(
            (KINDLE_DIR / f"{stem}.md" for stem in candidates if (KINDLE_DIR / f"{stem}.md").is_file()),
            None,
        )
        fresh = bool(
            pdf
            and archive
            and pdf.stat().st_mtime >= archive.stat().st_mtime - ARCHIVE_GRACE_SECONDS
        )
        sent_at = (
            datetime.fromtimestamp(archive.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if fresh and archive
            else ""
        )

        rows.append(
            {
                "bvid": bvid,
                "title": upload_title,
                "kindle_title": title,
                "url": f"https://www.bilibili.com/video/{bvid}",
                "transcribed": bool(summary),
                "summary_chars": len(summary),
                "sent": fresh,
                "sent_at": sent_at,
                "pdf": pdf.name if pdf else "",
                "archived": archive.name if fresh and archive else "",
            }
        )
    return rows


def to_markdown(rows: list[dict[str, object]]) -> str:
    header = "| Kindle 标题 | 上传标题 | BID 链接 | 转译 | 发送 |\n|---|---|---|---|---|\n"
    lines = [
        f"| {row['kindle_title']} | {row['title']} | {row['url']} | "
        f"{'是' if row['transcribed'] else '否'} | "
        f"{'是 ' + str(row['sent_at']) if row['sent'] else '否'} |"
        for row in rows
    ]
    return header + "\n".join(lines) + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]
    # --from-uploads <json>: ingest a fresh listing and check every video in it.
    if "--from-uploads" in args:
        index = args.index("--from-uploads")
        listing = Path(args[index + 1]) if index + 1 < len(args) else REPO_ROOT / "tmp" / "bili_uploads.json"
        pairs = records_from_listing(listing)
        bvids = [bvid for bvid, _ in pairs] or [a for a in args[1:index] + args[index + 2 :] if not a.startswith("--")]
        if not bvids:
            print(f"no usable videos in {listing}", file=sys.stderr)
            raise SystemExit(1)
    else:
        bvids = [arg for arg in args if not arg.startswith("--")]
    if not bvids:
        print("usage: verify_videos.py [BVID ...] [--list] | --from-uploads <json>", file=sys.stderr)
        raise SystemExit(2)
    data = inspect(bvids)
    if "--list" in args or "--from-uploads" in args:
        print(to_markdown(data), end="")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=1))
