from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timedelta
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "skill" / "scripts" / "fetch_uploads.py"
SPEC = importlib.util.spec_from_file_location("fetch_uploads", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bloomberg_title_filter_is_explicit() -> None:
    assert MODULE.is_bloomberg_title("261114 Bloomberg Tech")
    assert MODULE.is_bloomberg_title("261114 BB China Show P1")
    assert not MODULE.is_bloomberg_title("261114 ABC World News Tonight")
    assert not MODULE.is_bloomberg_title("261114 BBC World News")


def test_relative_hours_cross_midnight() -> None:
    today = date(2026, 11, 16)
    assert MODULE.parse_relative("18小时前", today, datetime(2026, 11, 16, 7, 0)) == date(2026, 11, 15)
    assert MODULE.parse_relative("2小时前", today, datetime(2026, 11, 16, 7, 0)) == today
    assert MODULE.parse_relative("10分钟前", today, datetime(2026, 11, 16, 0, 5)) == date(2026, 11, 15)


def test_main_defaults_to_previous_day(monkeypatch, tmp_path: Path) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    items = [
        {
            "published_date": today.isoformat(),
            "bvid": "BVTODAY",
            "title": "261115 Bloomberg Today",
            "date_conflict": False,
        },
        {
            "published_date": yesterday.isoformat(),
            "bvid": "BVYESTERDAY",
            "title": "261114 Bloomberg Tech",
            "date_conflict": False,
        },
        {
            "published_date": yesterday.isoformat(),
            "bvid": "BVABC",
            "title": "261114 ABC World News Tonight",
            "date_conflict": False,
        },
        {
            "published_date": (today - timedelta(days=2)).isoformat(),
            "bvid": "BVOLDER",
            "title": "261113 BB China Show",
            "date_conflict": False,
        },
    ]
    output = tmp_path / "bili_uploads.json"
    monkeypatch.setattr(MODULE, "OUTPUT", output)
    monkeypatch.setattr(MODULE, "fetch", lambda **_: items)

    assert MODULE.main([]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["window"] == [yesterday.isoformat(), yesterday.isoformat()]
    assert [item["bvid"] for item in payload["data"]] == ["BVYESTERDAY"]
