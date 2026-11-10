"""Fetch the UP 主's first uploads page through the opencli browser bridge.

``bili user-videos`` is rate limited (412) on this host, so the listing comes
from the real page in the user's logged-in Chrome. The page renders relative
times ("15小时前", "昨天 07:56", "2天前", "11月6日"); we combine those with the
``YYMMDD`` title prefix to get the calendar date. When the two disagree the page
wins, and the row is still kept — a mismatch is worth seeing, not dropping.

Output: ``tmp/bili_uploads.json`` in the same shape as ``bili user-videos --json``
plus ``published_date`` / ``relative`` / ``date_conflict`` per item, so
``verify_videos.py --from-uploads`` can consume it unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = REPO_ROOT / "tmp" / "bili_uploads.json"
JS_FILE = REPO_ROOT / "tmp" / "extract_uploads.js"
SPACE_URL = "https://space.bilibili.com/670833898/upload/video"
SESSION = "bili"
DEFAULT_DAYS = 3

# Wrapping in an IIFE is not cosmetic: the opencli CLI runs `eval` through
# PowerShell, which mangles the inner double quotes on the Multi-line JS
# statement by statement. One line with single quotes and no newlines survives.
CARD_JS = (
    "(function(){var out=[];"
    "document.querySelectorAll('.bili-video-card').forEach(function(c){"
    "var t=c.querySelector('.bili-video-card__title'),a=c.querySelector('a[href*=\"/video/BV\"]');"
    "var txt=c.innerText.replace(/\\s+/g,' ');"
    "var m=txt.match(/(\\d+小时前|\\d+分钟前|\\d+天前|昨天 \\d{2}:\\d{2}|\\d{2}-\\d{2}|\\d+月\\d+日)/);"
    "var d=txt.match(/(\\d{2}:\\d{2}:\\d{2}|\\d{2}:\\d{2})/);"
    "if(t&&a){out.push({title:t.getAttribute('title'),href:a.href.split('?')[0],"
    "rel:m?m[1]:'',duration:d?d[1]:''});}});"
    "return JSON.stringify(out);})()"
)


def _opencli_bin() -> str:
    """Locate the opencli launcher, which pwsh and Python resolve differently."""
    found = shutil.which("opencli")
    if found:
        return found
    for name in ("opencli.cmd", "opencli.ps1", "opencli"):
        candidate = Path(sys.executable).parents[1] / name
        # Windows launchers live next to node; fall back to common node paths.
        if candidate.is_file():
            return str(candidate)
    for base in (Path(r"C:\Users\david\AppData\Local\nodejs"), Path(r"C:\Program Files\nodejs")):
        for match in base.glob("node-*/opencli.cmd"):
            return str(match)
        if (base / "opencli.cmd").is_file():
            return str(base / "opencli.cmd")
    raise RuntimeError("opencli not found on PATH; add its bin directory or set OPENCLI_BIN")


def _opencli(*args: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        [_opencli_bin(), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencli {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()[:400]}")
    return proc.stdout


def _eval(js: str) -> object:
    out = _opencli("browser", SESSION, "eval", js)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(("[", "{")):
            return json.loads(line)
        # The IIFE returns a JSON *string*, which the CLI prints either bare or
        # json-encoded; accept both shapes.
        if "BV" in line and line.count(",") > 1:
            try:
                return json.loads(json.loads(line))
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"unexpected opencli eval output: {out.strip()[:300]}")


def parse_relative(rel: str, today: date) -> date | None:
    rel = (rel or "").strip()
    if not rel:
        return None
    if m := re.fullmatch(r"(\d+)\s*分钟前", rel):
        return today
    if m := re.fullmatch(r"(\d+)\s*小时前", rel):
        return today if int(m.group(1)) < 24 else today - timedelta(days=int(m.group(1)) // 24)
    if m := re.fullmatch(r"昨天\s*\d{2}:\d{2}", rel):
        return today - timedelta(days=1)
    if m := re.fullmatch(r"(\d+)\s*天前", rel):
        return today - timedelta(days=int(m.group(1)))
    if m := re.fullmatch(r"(\d{1,2})月(\d{1,2})日", rel):
        return date(today.year, int(m.group(1)), int(m.group(2)))
    if m := re.fullmatch(r"(\d{2})-(\d{2})", rel):
        return date(today.year, int(m.group(1)), int(m.group(2)))
    return None


def fetch(*, days: int = DEFAULT_DAYS, today: date | None = None, refresh_page: bool = True) -> list[dict]:
    today = today or date.today()
    if refresh_page:
        _opencli("browser", SESSION, "open", SPACE_URL)
        time.sleep(3)

    rows = _eval(CARD_JS.strip())
    if not isinstance(rows, list):
        raise RuntimeError("opencli eval returned no card list")

    items: list[dict] = []
    seen: set[str] = set()
    # First pass: resolve each card's dates so a batch can be judged as a whole.
    parsed: list[tuple[str, str, str, object, object]] = []
    for row in rows:
        bvid = (re.search(r"/(BV[0-9A-Za-z]+)", row.get("href") or "") or [None, ""])[1]
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        title = (row.get("title") or "").strip()
        rel = row.get("rel") or ""
        page_date = parse_relative(rel, today)
        if page_date and (today - page_date).days > days:
            page_date = None
        title_match = re.match(r"^(\d{2})(\d{2})(\d{2})\b", title)
        title_date = (
            date(2000 + int(title_match.group(1)), int(title_match.group(2)), int(title_match.group(3)))
            if title_match
            else None
        )
        parsed.append((bvid, title, rel, page_date, title_date))

    # Titles keep the broadcast's YYMMDD, not the upload day: 261107 shows can
    # post on 11/08. A mismatch is expected, so it is only a conflict when the
    # title date appears nowhere else in the listing.
    title_dates = {item[4] for item in parsed if item[4]}
    for bvid, title, rel, page_date, title_date in parsed:
        published = page_date or title_date
        conflict = bool(page_date and title_date and page_date != title_date and title_date not in title_dates)
        items.append(
            {
                "bvid": bvid,
                "id": bvid,
                "title": title,
                "url": f"https://www.bilibili.com/video/{bvid}",
                "duration": next(
                    (row.get("duration") or "" for row in rows if bvid in (row.get("href") or "")), ""
                ),
                "relative": rel,
                "published_date": published.isoformat() if published else "",
                "title_date": title_date.isoformat() if title_date else "",
                "date_conflict": conflict,
            }
        )
    return items


def main(argv: list[str]) -> int:
    days = DEFAULT_DAYS
    if "--days" in argv:
        days = int(argv[argv.index("--days") + 1])
    today = date.today()
    items = fetch(days=days, today=today)
    cutoff = today - timedelta(days=days - 1)
    recent = [item for item in items if item["published_date"] and date.fromisoformat(item["published_date"]) >= cutoff]

    payload = {
        "ok": True,
        "source": SPACE_URL,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "today": today.isoformat(),
        "window": [cutoff.isoformat(), today.isoformat()],
        "data": recent,
        "all_items": len(items),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for item in recent:
        flag = "  <-- 日期冲突" if item["date_conflict"] else ""
        print(f"{item['published_date']} | {item['bvid']} | {item['title']}{flag}")
    print(f"\n{len(recent)} 条在 {cutoff} ~ {today} 内（页面共 {len(items)} 条），已写入 {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
