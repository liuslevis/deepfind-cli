from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bili_transcribe import (
    BiliDownloadError,
    BiliTranscribeError,
    InvalidBiliIdError,
    MissingDependencyError,
    TranscriptionError,
    parse_bili_id,
    resolve_bili_bin,
    transcribe_bili_audio,
)
from .browser_fetch import fetch_web_document_browser
from .config import Settings
from .gen_slides import SlideGenerationError, generate_slides
from .gen_img import ImageGenerationError, MissingImageApiKeyError, generate_image
from .json_utils import dump_json, try_load_json
from .transcript_summary import (
    BILI_TRANSCRIPT_SUMMARY_MODEL,
    TranscriptSummaryError,
    summarize_transcript_for_query,
)
from .web_fetch import (
    WebFetchError,
    fetch_web_document,
    summarize_web_document,
)
from .youtube_transcribe import (
    InvalidYouTubeIdError,
    parse_youtube_id,
    resolve_audio_root,
)
from .youtube_audio_transcribe import (
    YouTubeDownloadError,
    transcribe_youtube_audio,
)
from .xhs_transcribe import (
    InvalidXhsVideoError,
    XhsDownloadError,
    transcribe_xhs_video,
)

_WEB_SEARCH_ENGINES = frozenset({"google", "baidu", "bing"})
_OPENCLI_REGISTRY_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_OPENCLI_REGISTRY_LOCK = Lock()
_XHS_TOPIC_TAG_RE = re.compile(r"#[^#\n]+#")
_XHS_TIMEZONE = timezone(timedelta(hours=8))
_XHS_SEARCH_ENRICH_LIMIT = 20
_XHS_TRUNCATION_MARKERS = ("...", "…")


def _twitter_type(value: str) -> str | None:
    mapping = {
        "top": "top",
        "latest": None,
        "photos": "photos",
        "videos": "videos",
        "Top": "top",
        "Latest": None,
        "Photos": "photos",
        "Videos": "videos",
        "People": "top",
    }
    return mapping.get(value)


def _xhs_sort(value: str) -> str:
    mapping = {
        "general": "general",
        "popular": "popular",
        "latest": "latest",
        "latest_popular": "latest",
        "most_popular": "popular",
    }
    return mapping.get(value, "general")


def _xhs_type(value: str) -> str:
    mapping = {
        "all": "all",
        "video": "video",
        "image": "image",
    }
    return mapping.get(value, "all")


def _xhs_ref(value: str, xsec_token: str = "") -> tuple[str, str, str]:
    ref = value.strip()
    token = xsec_token.strip()
    if not ref:
        return "", token, "ref cannot be empty"
    if any(marker in ref for marker in _XHS_TRUNCATION_MARKERS):
        return "", token, "ref appears to be truncated; paste the full Xiaohongshu URL or note ID"
    if ref.startswith("search_result/"):
        ref = ref.removeprefix("search_result/").strip()
    if not ref:
        return "", token, "ref cannot be empty"
    if "xiaohongshu.com" in ref and not token:
        query = parse_qs(urlparse(ref).query)
        token = str(query.get("xsec_token", [""])[0]).strip()
    return ref, token, ""


def _xhs_invalid_ref_response(tool: str, ref: str, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "error_code": "invalid_ref",
        "error": error,
        "ref": ref,
    }


def _xhs_empty_note_response(tool: str, ref: str, xsec_token: str) -> dict[str, Any]:
    if not xsec_token:
        return {
            "ok": False,
            "tool": tool,
            "error_code": "xsec_token_required",
            "error": "xhs read returned no note data; paste the Xiaohongshu share URL that includes xsec_token, or pass xsec_token explicitly",
            "ref": ref,
        }
    return {
        "ok": False,
        "tool": tool,
        "error_code": "invalid_note",
        "error": "xhs read returned no note data; the note may be unavailable or the xsec_token may be expired",
        "ref": ref,
    }


def _xhs_full_url(ref: str, xsec_token: str) -> str:
    """Construct full Xiaohongshu URL with xsec_token for opencli commands.

    opencli xiaohongshu note/comments requires a full signed URL.
    """
    # If ref is already a full URL, ensure xsec_token is in it
    if "xiaohongshu.com" in ref:
        # Already a URL
        if xsec_token and "xsec_token=" not in ref:
            # Add xsec_token if provided and not already in URL
            separator = "&" if "?" in ref else "?"
            return f"{ref}{separator}xsec_token={xsec_token}"
        return ref

    # ref is a note ID, construct /explore/ URL
    base_url = f"https://www.xiaohongshu.com/explore/{ref}"
    if xsec_token:
        return f"{base_url}?xsec_token={xsec_token}"
    return base_url


def _transform_opencli_comments(data: Any) -> list[dict[str, Any]]:
    """Transform opencli xiaohongshu comments output to internal format.

    opencli returns: [{"rank": 1, "author": "...", "text": "...", ...}, ...]
    We need a similar list format
    """
    if not isinstance(data, list):
        return []

    comments = []
    for item in data:
        if not isinstance(item, dict):
            continue
        comments.append({
            "author": item.get("author", ""),
            "text": item.get("text", ""),
            "likes": item.get("likes", 0),
            "time": item.get("time", ""),
            "is_reply": item.get("is_reply", False),
            "reply_to": item.get("reply_to", ""),
        })
    return comments


def _transform_opencli_note(data: Any, ref: str) -> dict[str, Any]:
    """Transform opencli xiaohongshu note output to internal format.

    opencli returns: [{"field": "title", "value": "..."}, ...]
    We need: {"title": "...", "author": "...", etc.}
    """
    if not isinstance(data, list):
        return {}

    # Convert field/value pairs to dict
    note_dict = {}
    for item in data:
        if isinstance(item, dict) and "field" in item and "value" in item:
            note_dict[item["field"]] = item["value"]

    if not note_dict:
        return {}

    # Extract note ID from ref
    note_id = ref
    if "xiaohongshu.com" in ref:
        # Extract from URL like /user/profile/{user_id}/{note_id} or /explore/{note_id}
        parts = ref.split('/')
        for part in reversed(parts):
            if part and '?' not in part and len(part) > 20:
                note_id = part
                break
            elif '?' in part:
                note_id = part.split('?')[0]
                break

    # Map opencli fields to internal format
    return {
        "id": note_id,
        "url": _xhs_note_url(note_id),
        "title": note_dict.get("title", ""),
        "author": note_dict.get("author", ""),
        "author_id": "",  # Not provided by opencli
        "desc": note_dict.get("content", ""),
        "tags": [],  # Not provided in simple format
        "media_type": note_dict.get("type", "unknown"),
        "text_mode": "full",
        "content_hint": "",
        "ip_location": "",  # Not provided by opencli
        "published_at_ms": 0,
        "updated_at_ms": 0,
        "image_count": 0,
        "video_duration_sec": 0,
        "stats": {
            "likes": int(note_dict.get("likes", 0)),
            "collects": int(note_dict.get("collects", 0)),
            "comments": int(note_dict.get("comments", 0)),
            "shares": 0,
        },
        "published_at": "",
        "published_date": "",
        "updated_at": "",
        "updated_date": "",
        "content_text": f"{note_dict.get('title', '')} {note_dict.get('content', '')}".strip(),
    }


def _xhs_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    payload = data.get("data")
    if isinstance(payload, dict):
        return payload
    return data


def _xhs_first_note_item(data: Any) -> dict[str, Any]:
    payload = _xhs_payload(data)
    items = payload.get("items")
    if not isinstance(items, list):
        return {}
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def _xhs_note_card(data: Any) -> dict[str, Any]:
    item = _xhs_first_note_item(data)
    note_card = item.get("note_card")
    if isinstance(note_card, dict):
        return note_card
    return {}


def _xhs_items(data: Any) -> list[dict[str, Any]]:
    payload = _xhs_payload(data)
    items = payload.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def _xhs_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lines = [line.strip() for line in value.replace("\r", "").split("\n")]
    return "\n".join(line for line in lines if line)


def _xhs_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _xhs_tags(note_card: dict[str, Any]) -> list[str]:
    tags = note_card.get("tag_list")
    if not isinstance(tags, list):
        return []
    names: list[str] = []
    for item in tags:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _xhs_desc_mode(desc: str) -> str:
    if not desc:
        return "empty"
    stripped = _XHS_TOPIC_TAG_RE.sub("", desc)
    stripped = re.sub(r"[\s#\[\],，。！？!?:：;；、()（）【】…·\-]+", "", stripped)
    return "full" if stripped else "tags_only"


def _xhs_media_type(note_card: dict[str, Any]) -> str:
    if note_card.get("type") == "video" or isinstance(note_card.get("video"), dict):
        return "video"
    return "image"


def _xhs_note_url(note_id: str) -> str:
    note_id = note_id.strip()
    if not note_id:
        return ""
    return f"https://www.xiaohongshu.com/explore/{note_id}"


def _xhs_search_result_ref(note_id: str) -> str:
    note_id = note_id.strip()
    if not note_id:
        return ""
    return f"search_result/{note_id}"


def _xhs_format_timestamp_ms(timestamp_ms: int) -> str:
    if timestamp_ms <= 0:
        return ""
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).astimezone(_XHS_TIMEZONE).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (OverflowError, OSError, ValueError):
        return ""


def _xhs_search_media_type(item: dict[str, Any], note_card: dict[str, Any]) -> str:
    if note_card:
        return _xhs_media_type(note_card)
    item_type = str(item.get("type") or "").strip().lower()
    if item_type in {"video", "image"}:
        return item_type
    return ""


def _xhs_search_item(item: dict[str, Any]) -> dict[str, Any]:
    note_card = item.get("note_card") if isinstance(item.get("note_card"), dict) else {}
    user = note_card.get("user") if isinstance(note_card.get("user"), dict) else {}
    interact = note_card.get("interact_info") if isinstance(note_card.get("interact_info"), dict) else {}
    note_id = str(note_card.get("note_id") or item.get("id") or "").strip()
    published_at_ms = _xhs_int(note_card.get("time") or item.get("publish_time") or item.get("time"))
    published_at = _xhs_format_timestamp_ms(published_at_ms)
    return {
        "id": note_id,
        "ref": _xhs_search_result_ref(note_id),
        "url": _xhs_note_url(note_id),
        "title": _xhs_text(note_card.get("title") or note_card.get("display_title") or item.get("title") or ""),
        "author": _xhs_text(user.get("nickname") or item.get("author") or item.get("nickname") or ""),
        "likes": _xhs_int(interact.get("liked_count") or item.get("liked_count") or item.get("likes")),
        "media_type": _xhs_search_media_type(item, note_card),
        "published_at": published_at,
        "published_date": published_at.split(" ", 1)[0] if published_at else "",
        "published_at_ms": published_at_ms,
    }


def _xhs_merge_search_item_with_note(item: dict[str, Any], note: dict[str, Any]) -> dict[str, Any]:
    stats = note.get("stats") if isinstance(note.get("stats"), dict) else {}
    merged = dict(note)
    merged["id"] = str(merged.get("id") or item.get("id") or "").strip()
    merged["ref"] = str(item.get("ref") or _xhs_search_result_ref(merged["id"])).strip()
    merged["url"] = str(merged.get("url") or item.get("url") or _xhs_note_url(merged["id"])).strip()
    merged["title"] = _xhs_text(merged.get("title") or item.get("title") or "")
    merged["author"] = _xhs_text(merged.get("author") or item.get("author") or "")
    merged["media_type"] = str(merged.get("media_type") or item.get("media_type") or "").strip()
    merged["published_at"] = str(
        merged.get("published_at") or _xhs_format_timestamp_ms(_xhs_int(merged.get("published_at_ms"))) or item.get("published_at") or ""
    ).strip()
    merged["published_date"] = str(merged.get("published_date") or item.get("published_date") or "").strip()
    if not merged["published_date"] and merged["published_at"]:
        merged["published_date"] = merged["published_at"].split(" ", 1)[0]
    merged["published_at_ms"] = _xhs_int(merged.get("published_at_ms") or item.get("published_at_ms"))
    merged["likes"] = _xhs_int(stats.get("likes") or item.get("likes"))
    merged["collects"] = _xhs_int(stats.get("collects"))
    merged["comments"] = _xhs_int(stats.get("comments"))
    merged["shares"] = _xhs_int(stats.get("shares"))
    return merged


def _xhs_enrichment_error(note_id: str, index: int, result: dict[str, Any]) -> dict[str, Any]:
    error: dict[str, Any] = {
        "phase": "read",
        "note_id": note_id,
        "index": index,
        "error_code": str(result.get("error_code") or "read_failed"),
        "error": str(result.get("error") or "xhs_read failed").strip(),
    }
    if result.get("returncode") is not None:
        error["returncode"] = _xhs_int(result.get("returncode"))
    stderr = str(result.get("stderr") or "").strip()
    stdout = str(result.get("stdout") or "").strip()
    if stderr:
        error["stderr"] = stderr
    if stdout:
        error["stdout"] = stdout
    return error


def _xhs_content_text(note: dict[str, Any]) -> str:
    lines: list[str] = []
    title = str(note.get("title", "")).strip()
    author = str(note.get("author", "")).strip()
    desc = str(note.get("desc", "")).strip()
    tags = note.get("tags")
    media_type = str(note.get("media_type", "")).strip()
    duration = note.get("video_duration_sec")
    hint = str(note.get("content_hint", "")).strip()

    if title:
        lines.append(f"Title: {title}")
    if author:
        lines.append(f"Author: {author}")
    if desc:
        lines.append(f"Body: {desc}")
    if isinstance(tags, list) and tags:
        lines.append(f"Tags: {', '.join(str(tag) for tag in tags if str(tag).strip())}")
    if media_type:
        lines.append(f"Media type: {media_type}")
    if isinstance(duration, int) and duration > 0:
        lines.append(f"Video duration seconds: {duration}")
    if hint:
        lines.append(f"Note: {hint}")
    return "\n".join(lines)


def _xhs_note(data: Any) -> dict[str, Any]:
    item = _xhs_first_note_item(data)
    note_card = _xhs_note_card(data)
    if not note_card:
        return {}

    note_id = str(note_card.get("note_id") or item.get("id") or "").strip()
    title = _xhs_text(note_card.get("title") or note_card.get("display_title") or "")
    desc = _xhs_text(note_card.get("desc"))
    tags = _xhs_tags(note_card)
    media_type = _xhs_media_type(note_card)
    desc_mode = _xhs_desc_mode(desc)
    video = note_card.get("video") if isinstance(note_card.get("video"), dict) else {}
    video_capa = video.get("capa") if isinstance(video.get("capa"), dict) else {}
    user = note_card.get("user") if isinstance(note_card.get("user"), dict) else {}
    interact = note_card.get("interact_info") if isinstance(note_card.get("interact_info"), dict) else {}

    content_hint = ""
    if media_type == "video" and desc_mode != "full":
        content_hint = "This is a video note and the text body is mostly tags or empty; the main substance may be in the spoken audio."
    elif desc_mode == "tags_only":
        content_hint = "The text body is mostly tags rather than full prose."

    note = {
        "id": note_id,
        "url": _xhs_note_url(note_id),
        "title": title,
        "author": _xhs_text(user.get("nickname")),
        "author_id": str(user.get("user_id", "")).strip(),
        "desc": desc,
        "tags": tags,
        "media_type": media_type,
        "text_mode": desc_mode,
        "content_hint": content_hint,
        "ip_location": _xhs_text(note_card.get("ip_location")),
        "published_at_ms": _xhs_int(note_card.get("time")),
        "updated_at_ms": _xhs_int(note_card.get("last_update_time")),
        "image_count": len(note_card.get("image_list", [])) if isinstance(note_card.get("image_list"), list) else 0,
        "video_duration_sec": _xhs_int(video_capa.get("duration")),
        "stats": {
            "likes": _xhs_int(interact.get("liked_count")),
            "collects": _xhs_int(interact.get("collected_count")),
            "comments": _xhs_int(interact.get("comment_count")),
            "shares": _xhs_int(interact.get("share_count")),
        },
    }
    note["published_at"] = _xhs_format_timestamp_ms(_xhs_int(note["published_at_ms"]))
    note["published_date"] = note["published_at"].split(" ", 1)[0] if note["published_at"] else ""
    note["updated_at"] = _xhs_format_timestamp_ms(_xhs_int(note["updated_at_ms"]))
    note["updated_date"] = note["updated_at"].split(" ", 1)[0] if note["updated_at"] else ""
    note["content_text"] = _xhs_content_text(note)
    return note


def _xhs_comment(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    user = data.get("user_info") if isinstance(data.get("user_info"), dict) else {}
    create_time_ms = _xhs_int(data.get("create_time"))
    comment = {
        "id": str(data.get("id") or "").strip(),
        "note_id": str(data.get("note_id") or "").strip(),
        "content": _xhs_text(data.get("content")),
        "author": _xhs_text(user.get("nickname")),
        "author_id": str(user.get("user_id") or "").strip(),
        "ip_location": _xhs_text(data.get("ip_location")),
        "like_count": _xhs_int(data.get("like_count")),
        "created_at_ms": create_time_ms,
        "created_at": _xhs_format_timestamp_ms(create_time_ms),
        "sub_comment_count": _xhs_int(data.get("sub_comment_count")),
        "sub_comment_has_more": bool(data.get("sub_comment_has_more")),
        "sub_comment_cursor": str(data.get("sub_comment_cursor") or "").strip(),
        "tags": [str(tag).strip() for tag in data.get("show_tags", []) if str(tag).strip()]
        if isinstance(data.get("show_tags"), list)
        else [],
        "sub_comments": [],
    }
    sub_comments = data.get("sub_comments")
    if isinstance(sub_comments, list):
        comment["sub_comments"] = [
            sub_comment
            for sub_comment in (_xhs_comment(item) for item in sub_comments)
            if sub_comment
        ]
    return comment


def _xhs_comments(data: Any, ref: str) -> dict[str, Any]:
    payload = _xhs_payload(data)
    if "comments" not in payload:
        return {}
    raw_comments = payload.get("comments")
    comments = []
    if isinstance(raw_comments, list):
        comments = [comment for comment in (_xhs_comment(item) for item in raw_comments) if comment]
    note_id = ""
    for comment in comments:
        note_id = str(comment.get("note_id") or "").strip()
        if note_id:
            break
    if not note_id:
        note_id = _xhs_ref(ref)[0]
        if "xiaohongshu.com" in note_id:
            note_id = urlparse(note_id).path.rstrip("/").split("/")[-1]
    inline_sub_comments = sum(
        len(comment.get("sub_comments", []))
        for comment in comments
        if isinstance(comment.get("sub_comments"), list)
    )
    return {
        "ref": ref,
        "note_id": note_id,
        "cursor": str(payload.get("cursor") or "").strip(),
        "has_more": bool(payload.get("has_more")),
        "fetched_at_ms": _xhs_int(payload.get("time")),
        "top_comment_count": len(comments),
        "inline_sub_comment_count": inline_sub_comments,
        "xsec_token_present": bool(str(payload.get("xsec_token") or "").strip()),
        "comments": comments,
    }


def _xhs_video_url(data: Any) -> str:
    note_card = _xhs_note_card(data)
    video = note_card.get("video")
    if not isinstance(video, dict):
        return ""
    media = video.get("media")
    if not isinstance(media, dict):
        return ""
    stream = media.get("stream")
    if not isinstance(stream, dict):
        return ""

    candidates: list[tuple[int, str]] = []
    for codec_key in ("h264", "h265", "av1", "h266"):
        values = stream.get(codec_key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            master_url = str(item.get("master_url", "")).strip()
            if not master_url:
                continue
            rank = _xhs_int(item.get("size")) or _xhs_int(item.get("avg_bitrate")) or 10**12
            candidates.append((rank, master_url))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _bili_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lines = [line.strip() for line in value.replace("\r", "").split("\n")]
    return "\n".join(line for line in lines if line)


def _bili_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return parse_bili_id(raw)
    except InvalidBiliIdError:
        return raw if raw.upper().startswith("BV") else ""


def _bili_video_url(bvid: str) -> str:
    bvid = _bili_id(bvid)
    if not bvid:
        return ""
    return f"https://www.bilibili.com/video/{bvid}"


def _bili_number(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value or "").strip().replace(",", "")
    if not text:
        return 0
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([万亿]?)", text)
    if not match:
        return 0
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "万":
        number *= 10_000
    elif unit == "亿":
        number *= 100_000_000
    return int(number)


def _bili_duration_seconds(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    text = str(value or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    parts = [part.strip() for part in text.split(":")]
    if not parts or not all(part.isdigit() for part in parts):
        return 0
    values = [int(part) for part in parts]
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    if len(values) == 3:
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds
    return 0


def _format_duration(seconds: int) -> str:
    total = max(seconds, 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _bili_duration_text(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        seconds = _bili_duration_seconds(value)
        return _format_duration(seconds) if seconds > 0 else ""
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" in text:
        return text
    seconds = _bili_duration_seconds(text)
    return _format_duration(seconds) if seconds > 0 else ""


def _bili_payload(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    payload = data.get("data")
    if isinstance(payload, (dict, list)):
        return payload
    return data


def _bili_owner(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("owner", "uploader", "user", "up"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _bili_stat(payload: dict[str, Any]) -> dict[str, Any]:
    stat = payload.get("stat")
    return stat if isinstance(stat, dict) else {}


def _bili_search_items(data: Any) -> list[dict[str, Any]]:
    payload = _bili_payload(data)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "videos", "result", "results", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _bili_search_item(item: dict[str, Any]) -> dict[str, Any]:
    owner = _bili_owner(item)
    stat = _bili_stat(item)
    bvid = _bili_id(item.get("bvid") or item.get("bv") or item.get("id") or item.get("url"))
    duration_source = item.get("duration") or item.get("length") or item.get("duration_text")
    duration_seconds = _bili_duration_seconds(item.get("duration_seconds") or item.get("duration_sec") or duration_source)
    return {
        "id": bvid,
        "bvid": bvid,
        "url": str(item.get("url") or item.get("link") or _bili_video_url(bvid)).strip(),
        "title": _bili_text(item.get("title") or item.get("name") or ""),
        "author": _bili_text(item.get("author") or item.get("uploader") or owner.get("name") or ""),
        "uid": str(item.get("uid") or item.get("mid") or owner.get("uid") or owner.get("mid") or "").strip(),
        "duration": _bili_duration_text(duration_source),
        "duration_seconds": duration_seconds,
        "play_count": _bili_number(item.get("play") or item.get("view") or stat.get("view")),
        "desc": _bili_text(item.get("desc") or item.get("description") or ""),
    }


def _bili_video_item(data: Any) -> dict[str, Any]:
    payload = _bili_payload(data)
    if isinstance(payload, dict) and isinstance(payload.get("video"), dict):
        payload = payload["video"]
    if not isinstance(payload, dict):
        return {}
    owner = _bili_owner(payload)
    stat = _bili_stat(payload)
    bvid = _bili_id(payload.get("bvid") or payload.get("bv") or payload.get("id") or payload.get("url"))
    duration_source = payload.get("duration") or payload.get("length") or payload.get("duration_text")
    duration_seconds = _bili_duration_seconds(payload.get("duration_seconds") or payload.get("duration_sec") or duration_source)
    return {
        "id": bvid,
        "bvid": bvid,
        "url": str(payload.get("url") or payload.get("link") or _bili_video_url(bvid)).strip(),
        "title": _bili_text(payload.get("title") or payload.get("name") or ""),
        "author": _bili_text(payload.get("author") or payload.get("uploader") or owner.get("name") or ""),
        "uid": str(payload.get("uid") or payload.get("mid") or owner.get("uid") or owner.get("mid") or "").strip(),
        "duration": _bili_duration_text(duration_source),
        "duration_seconds": duration_seconds,
        "play_count": _bili_number(payload.get("play") or payload.get("view") or stat.get("view")),
        "danmaku_count": _bili_number(payload.get("danmaku") or stat.get("danmaku")),
        "like_count": _bili_number(payload.get("likes") or payload.get("like") or stat.get("like")),
        "coin_count": _bili_number(payload.get("coins") or payload.get("coin") or stat.get("coin")),
        "favorite_count": _bili_number(payload.get("favorites") or payload.get("favorite") or stat.get("favorite")),
        "share_count": _bili_number(payload.get("shares") or payload.get("share") or stat.get("share")),
        "desc": _bili_text(payload.get("desc") or payload.get("description") or payload.get("intro") or ""),
    }


def _merge_bili_search_item_with_video(item: dict[str, Any], video: dict[str, Any]) -> dict[str, Any]:
    merged = dict(item)
    for key, value in video.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _bili_enrichment_error(bvid: str, index: int, result: dict[str, Any]) -> dict[str, Any]:
    error: dict[str, Any] = {
        "phase": "video",
        "bvid": bvid,
        "index": index,
        "error_code": str(result.get("error_code") or "video_failed"),
        "error": str(result.get("error") or "bili video failed").strip(),
    }
    if result.get("returncode") is not None:
        error["returncode"] = _bili_number(result.get("returncode"))
    stderr = str(result.get("stderr") or "").strip()
    stdout = str(result.get("stdout") or "").strip()
    if stderr:
        error["stderr"] = stderr
    if stdout:
        error["stdout"] = stdout
    return error


def _tool_error_response(
    tool_name: str,
    exc: Exception,
    mapping: tuple[tuple[type[BaseException], str], ...],
    *,
    default_code: str = "unknown_error",
) -> dict[str, Any]:
    for exc_type, code in mapping:
        if isinstance(exc, exc_type):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": code,
                "error": str(exc),
            }
    return {
        "ok": False,
        "tool": tool_name,
        "error_code": default_code,
        "error": str(exc),
    }


def _subprocess_failure(
    tool: str,
    command: list[str],
    *,
    returncode: int,
    stderr: str,
    stdout: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stderr_text = stderr.strip()[:4000]
    stdout_text = stdout.strip()[:4000]
    error_text = stderr_text or stdout_text or f"process exited with code {returncode}"
    payload = {
        "ok": False,
        "tool": tool,
        **(context or {}),
        "command": command,
        "error_code": "command_failed",
        "error": error_text,
        "returncode": returncode,
    }
    if stderr_text:
        payload["stderr"] = stderr_text
    if stdout_text:
        payload["stdout"] = stdout_text
    return payload


class Toolset:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._functions = {
            "web_search": self.web_search,
            "web_fetch": self.web_fetch,
            "browser_fetch": self.browser_fetch,
            "arxiv_search": self.arxiv_search,
            "paper_search": self.paper_search,
            "read_paper": self.read_paper,
            "twitter_search": self.twitter_search,
            "x_search": self.x_search,
            "twitter_read": self.twitter_read,
            "zhihu_search": self.zhihu_search,
            "boss_search": self.boss_search,
            "boss_detail": self.boss_detail,
            "boss_chatlist": self.boss_chatlist,
            "boss_send": self.boss_send,
            "xhs_search": self.xhs_search,
            "xhs_read": self.xhs_read,
            "xhs_read_cmt": self.xhs_read_cmt,
            "xhs_transcribe_full": self.xhs_transcribe_full,
            "xhs_user_posts": self.xhs_user_posts,
            "bili_search": self.bili_search,
            "bili_get_user_videos": self.bili_get_user_videos,
            "bili_transcribe": self.bili_transcribe,
            "bili_transcribe_full": self.bili_transcribe_full,
            "youtube_transcribe": self.youtube_transcribe,
            "youtube_transcribe_full": self.youtube_transcribe_full,
            "gen_img": self.gen_img,
            "gen_slides": self.gen_slides,
        }

    def specs(self) -> list[dict[str, Any]]:
        return [
            self._function_spec(
                "web_search",
                "Search the web through opencli. Prefer this for broad web research, and use the platform-specific tools for Xiaohongshu, X/Twitter, Bilibili, YouTube, and BOSS Zhipin.",
                {
                    "type": "object",
                    "properties": {
                        "engine": {
                            "type": "string",
                            "enum": sorted(_WEB_SEARCH_ENGINES),
                        },
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["engine", "query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "web_fetch",
                "Fetch one web page URL, convert it to Markdown, and return a targeted summary for the provided prompt. Prefer this after web_search when a result looks important.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["url", "prompt"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "browser_fetch",
                "Fetch one web page by rendering it in a real browser (Chrome via Playwright), then convert it to Markdown and return a targeted summary. Use this when web_fetch is blocked or the page requires JavaScript/cookies. Set headless=false when a site needs manual login or captcha verification.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "prompt": {"type": "string"},
                        "headless": {"type": "boolean"},
                    },
                    "required": ["url", "prompt"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "arxiv_search",
                "Search arXiv papers via opencli.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "paper_search",
                "Search arXiv papers and return detailed information including abstract, PDF/HTML URLs, and metadata. More comprehensive than arxiv_search.",
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for arXiv papers",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of papers to return",
                            "minimum": 1,
                            "maximum": 25,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "read_paper",
                "Fetch and analyze an arXiv paper from its HTML URL to answer specific questions about the content.",
                {
                    "type": "object",
                    "properties": {
                        "paper_html_url": {
                            "type": "string",
                            "description": "HTML URL of the paper (e.g., https://ar5iv.labs.arxiv.org/html/2105.02723 or https://arxiv.org/html/...)",
                        },
                        "query": {
                            "type": "string",
                            "description": "Question or prompt about the paper content",
                        },
                    },
                    "required": ["paper_html_url", "query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "twitter_search",
                "Search X.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                        "tab": {
                            "type": "string",
                            "enum": ["top", "photos", "videos"],
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "x_search",
                "Search X. Alias of twitter_search.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                        "tab": {
                            "type": "string",
                            "enum": ["top", "photos", "videos"],
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "twitter_read",
                "Read one X post by URL or ID.",
                {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "zhihu_search",
                "Search Zhihu via opencli.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "boss_search",
                "Search BOSS Zhipin job postings. Returns security_id values that can be passed to boss_detail.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "city": {"type": "string"},
                        "experience": {"type": "string"},
                        "degree": {"type": "string"},
                        "salary": {"type": "string"},
                        "industry": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1, "maximum": 20},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "boss_detail",
                "Read one BOSS Zhipin job posting by security_id from boss_search.",
                {
                    "type": "object",
                    "properties": {
                        "security_id": {"type": "string"},
                    },
                    "required": ["security_id"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "boss_chatlist",
                "List BOSS Zhipin chat threads. Use this to find uid values for boss_send and to check whether a conversation already exists for a job.",
                {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 1, "maximum": 50},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        "job_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "boss_send",
                "Send a BOSS Zhipin chat message by uid from boss_chatlist. Prefer boss_detail first because it may already reveal the company behind a hidden-company post.",
                {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["uid", "text"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_search",
                "Search Xiaohongshu notes.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1, "maximum": 10},
                        "sort": {
                            "type": "string",
                            "enum": ["general", "popular", "latest"],
                        },
                        "note_type": {
                            "type": "string",
                            "enum": ["all", "video", "image"],
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_read",
                "Read one Xiaohongshu note by full URL or note ID. Do not pass ellipsized or truncated URLs.",
                {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "xsec_token": {"type": "string"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_transcribe_full",
                "Read one Xiaohongshu note from a full URL or note ID and return all extractable content. For video notes, transcribe the spoken audio and return the full transcript. Do not pass ellipsized or truncated URLs.",
                {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "xsec_token": {"type": "string"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_read_cmt",
                "Read Xiaohongshu comments for one note by full URL or note ID. Pass xsec_token when the note URL requires it. Use fetch_all=true only when all top-level comments are needed.",
                {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "xsec_token": {"type": "string"},
                        "cursor": {"type": "string"},
                        "fetch_all": {"type": "boolean"},
                    },
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_user_posts",
                "List a Xiaohongshu user's posts by user ID. Returns posts with 'url' field containing full signed URLs that can be passed to xhs_read and xhs_read_cmt.",
                {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "cursor": {"type": "string"},
                    },
                    "required": ["user_id"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "bili_search",
                "Search Bilibili videos via bilibili-cli, then enrich each hit with bili video details.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1, "maximum": 50},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "bili_get_user_videos",
                "List a Bilibili user's uploaded videos by UID or username via opencli bilibili user-videos.",
                {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "order": {
                            "type": "string",
                            "enum": ["pubdate", "click", "stow"],
                        },
                        "page": {"type": "integer", "minimum": 1, "maximum": 50},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["uid"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "bili_transcribe",
                "Transcribe Bilibili video audio by URL or BVID, then summarize it for the research query with qwen-plus.",
                {
                    "type": "object",
                    "properties": {
                        "bili_id": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["bili_id", "query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "bili_transcribe_full",
                "Transcribe Bilibili video audio by URL or BVID and return the full transcript.",
                {
                    "type": "object",
                    "properties": {
                        "bili_id": {"type": "string"},
                    },
                    "required": ["bili_id"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "youtube_transcribe",
                "Download YouTube audio with yt-dlp, transcribe it with local ASR, then summarize it for the research query.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["url", "query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "youtube_transcribe_full",
                "Download YouTube audio with yt-dlp and transcribe it with local ASR, returning the full transcript.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "gen_img",
                "Generate one image with Nano Banana and save it under the local tmp directory.",
                {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                        },
                        "image_size": {
                            "type": "string",
                            "enum": ["1K", "2K", "4K"],
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "gen_slides",
                "Generate a beautiful HTML slide deck using a template from beautiful-html-templates. "
                "Output is saved under the slide/ directory. Supports editing existing decks via html_path.",
                {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "slide_count": {"type": "integer", "minimum": 1, "maximum": 12},
                        "template_name": {
                            "type": "string",
                            "description": "Template name (e.g. '8-bit-orbit', 'monochrome'). Defaults to 'monochrome'.",
                        },
                        "html_path": {
                            "type": "string",
                            "description": "Path to existing deck HTML for editing. If provided, prompt is treated as an edit instruction.",
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            ),
        ]

    def _function_spec(self, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._functions:
            return dump_json({"ok": False, "tool": name, "error": "unknown tool"})
        try:
            return dump_json(self._functions[name](**arguments))
        except Exception as exc:  # pragma: no cover - defensive
            return dump_json({"ok": False, "tool": name, "error": str(exc)})

    def web_search(
        self,
        engine: str,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        engine = engine.strip().lower()
        safe_limit = max(1, min(20, limit))

        command_prefix, resolve_error = self._opencli_command_prefix()
        if not command_prefix:
            return {
                "ok": False,
                "tool": "web_search",
                "engine": engine,
                "query": query,
                "error_code": "missing_dependency",
                "error": resolve_error or f"{self.settings.opencli_bin} not found",
            }

        registry_result = self._opencli_registry(command_prefix)
        if not registry_result.get("ok"):
            return {
                **registry_result,
                "tool": "web_search",
                "engine": engine,
                "query": query,
            }

        registry = registry_result["data"]
        command_name = f"{engine}/search"
        command_spec = registry.get(command_name)
        if engine not in _WEB_SEARCH_ENGINES or not command_spec:
            return {
                "ok": False,
                "tool": "web_search",
                "engine": engine,
                "query": query,
                "error_code": "unsupported_engine",
                "error": f"{engine} search is not available in the current opencli registry",
            }

        command = [*command_prefix, engine, "search", query]
        if self._opencli_supports_arg(command_spec, "limit"):
            command.extend(["--limit", str(safe_limit)])
        command.extend(["-f", "json"])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.subprocess_timeout,
        )
        if proc.returncode != 0:
            return _subprocess_failure(
                "web_search",
                command,
                returncode=proc.returncode,
                stderr=proc.stderr or "",
                stdout=proc.stdout or "",
                context={
                    "engine": engine,
                    "query": query,
                },
            )

        parsed = try_load_json(proc.stdout or "")
        if parsed is None:
            return {
                "ok": False,
                "tool": "web_search",
                "engine": engine,
                "query": query,
                "command": command,
                "error_code": "invalid_json",
                "error": "opencli returned non-JSON output",
            }

        return {
            "ok": True,
            "tool": "web_search",
            "engine": engine,
            "query": query,
            "command": command,
            "data": parsed,
        }

    def web_fetch(self, url: str, prompt: str) -> dict[str, Any]:
        clean_url = url.strip()
        clean_prompt = prompt.strip()
        try:
            document = fetch_web_document(clean_url, timeout=self.settings.subprocess_timeout)
            summary = summarize_web_document(
                self.settings.new_client(),
                prompt=clean_prompt,
                document=document,
                model=self.settings.model,
            )
        except WebFetchError as exc:
            return {
                "ok": False,
                "tool": "web_fetch",
                "url": clean_url,
                "prompt": clean_prompt,
                "error_code": exc.error_code,
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "web_fetch",
            "url": clean_url,
            "prompt": clean_prompt,
            "data": {
                "url": document.url,
                "final_url": document.final_url,
                "title": document.title,
                "summary": summary,
                "content_type": document.content_type,
                "truncated": document.truncated,
                "markdown_chars": document.markdown_chars,
            },
        }

    def browser_fetch(self, url: str, prompt: str, headless: bool = True) -> dict[str, Any]:
        clean_url = url.strip()
        clean_prompt = prompt.strip()
        try:
            document = fetch_web_document_browser(
                clean_url,
                timeout=self.settings.subprocess_timeout,
                headless=True,  # forced for security
            )
            summary = summarize_web_document(
                self.settings.new_client(),
                prompt=clean_prompt,
                document=document,
                model=self.settings.model,
            )
        except WebFetchError as exc:
            return {
                "ok": False,
                "tool": "browser_fetch",
                "url": clean_url,
                "prompt": clean_prompt,
                "error_code": exc.error_code,
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "browser_fetch",
            "url": clean_url,
            "prompt": clean_prompt,
            "data": {
                "url": document.url,
                "final_url": document.final_url,
                "title": document.title,
                "summary": summary,
                "content_type": document.content_type,
                "truncated": document.truncated,
                "markdown_chars": document.markdown_chars,
                "rendered": True,
            },
        }

    def arxiv_search(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        return self._opencli_site_search(
            site="arxiv",
            query=query,
            limit=max(1, min(25, limit)),
            tool="arxiv_search",
        )

    def paper_search(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search arXiv papers and return detailed information including abstract, PDF/HTML URLs."""
        safe_limit = max(1, min(25, limit))

        # Get opencli command prefix
        command_prefix, resolve_error = self._opencli_command_prefix()
        if not command_prefix:
            return {
                "ok": False,
                "tool": "paper_search",
                "query": query,
                "error_code": "missing_dependency",
                "error": resolve_error or f"{self.settings.opencli_bin} not found",
            }

        # Execute search: opencli arxiv search <query> --limit <limit> -f json
        search_command = [*command_prefix, "arxiv", "search", query, "--limit", str(safe_limit), "-f", "json"]

        try:
            search_proc = subprocess.run(
                search_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.settings.subprocess_timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "tool": "paper_search",
                "query": query,
                "error_code": "timeout",
                "error": f"Search timed out after {self.settings.subprocess_timeout}s",
            }

        if search_proc.returncode != 0:
            return _subprocess_failure(
                "paper_search",
                search_command,
                returncode=search_proc.returncode,
                stderr=search_proc.stderr or "",
                stdout=search_proc.stdout or "",
                context={"query": query},
            )

        # Parse search results
        search_results = try_load_json(search_proc.stdout or "")
        if search_results is None:
            return {
                "ok": False,
                "tool": "paper_search",
                "query": query,
                "error_code": "invalid_json",
                "error": "opencli arxiv search returned non-JSON output",
            }

        if not isinstance(search_results, list):
            search_results = []

        # For each paper, fetch detailed information
        papers = []
        errors = []

        for item in search_results:
            if not isinstance(item, dict):
                continue

            paper_id = item.get("id")
            if not paper_id:
                continue

            # Execute: opencli arxiv paper <paper_id> -f json
            paper_command = [*command_prefix, "arxiv", "paper", str(paper_id), "-f", "json"]

            try:
                paper_proc = subprocess.run(
                    paper_command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.settings.subprocess_timeout,
                )
            except subprocess.TimeoutExpired:
                errors.append({
                    "paper_id": paper_id,
                    "error": "timeout fetching paper details",
                })
                continue

            if paper_proc.returncode != 0:
                errors.append({
                    "paper_id": paper_id,
                    "error": "failed to fetch paper details",
                    "stderr": (paper_proc.stderr or "")[:500],
                })
                continue

            paper_data = try_load_json(paper_proc.stdout or "")
            if not paper_data or not isinstance(paper_data, list) or len(paper_data) == 0:
                errors.append({
                    "paper_id": paper_id,
                    "error": "invalid paper details response",
                })
                continue

            detail = paper_data[0]
            if not isinstance(detail, dict):
                continue

            papers.append({
                "paper_id": detail.get("id", ""),
                "paper_title": detail.get("title", ""),
                "paper_authors": detail.get("authors", ""),
                "publish_date": detail.get("published", ""),
                "paper_source": "arxiv",
                "paper_pdf_url": detail.get("pdf", ""),
                "paper_html_url": f"https://ar5iv.labs.arxiv.org/html/{detail.get('id', '')}",
                "paper_abs_url": detail.get("url", ""),
                "paper_abstract": detail.get("abstract", ""),
                "categories": detail.get("categories", ""),
                "primary_category": detail.get("primary_category", ""),
            })

        return {
            "ok": True,
            "tool": "paper_search",
            "data": {
                "query": query,
                "limit": safe_limit,
                "papers": papers,
                "errors": errors if errors else None,
            },
        }

    def read_paper(
        self,
        paper_html_url: str,
        query: str,
    ) -> dict[str, Any]:
        """Fetch and analyze an arXiv paper from its HTML URL to answer specific questions."""
        clean_url = paper_html_url.strip()
        clean_query = query.strip()

        try:
            # Reuse existing web_fetch infrastructure
            document = fetch_web_document(clean_url, timeout=self.settings.subprocess_timeout)
            answer = summarize_web_document(
                self.settings.new_client(),
                prompt=clean_query,
                document=document,
                model=self.settings.model,
            )
        except WebFetchError as exc:
            return {
                "ok": False,
                "tool": "read_paper",
                "url": clean_url,
                "query": clean_query,
                "error_code": exc.error_code,
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "read_paper",
            "url": clean_url,
            "query": clean_query,
            "data": {
                "url": document.url,
                "final_url": document.final_url,
                "title": document.title,
                "query": clean_query,
                "answer": answer,
            },
        }

    def twitter_search(
        self,
        query: str,
        max_results: int = 10,
        tab: str = "",
    ) -> dict[str, Any]:
        return self._twitter_search_impl(
            tool="twitter_search",
            query=query,
            max_results=max_results,
            tab=tab,
        )

    def x_search(
        self,
        query: str,
        max_results: int = 10,
        tab: str = "",
    ) -> dict[str, Any]:
        return self._twitter_search_impl(
            tool="x_search",
            query=query,
            max_results=max_results,
            tab=tab,
        )

    def zhihu_search(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        return self._opencli_site_search(
            site="zhihu",
            query=query,
            limit=limit,
            tool="zhihu_search",
        )

    def boss_search(
        self,
        query: str,
        city: str = "",
        experience: str = "",
        degree: str = "",
        salary: str = "",
        industry: str = "",
        page: int = 1,
        limit: int = 15,
    ) -> dict[str, Any]:
        return self._opencli_command(
            site="boss",
            action="search",
            tool="boss_search",
            values={
                "query": query,
                "city": city,
                "experience": experience,
                "degree": degree,
                "salary": salary,
                "industry": industry,
                "page": max(1, min(20, page)),
                "limit": max(1, min(50, limit)),
            },
            context={
                "query": query,
                "city": city,
                "experience": experience,
                "degree": degree,
                "salary": salary,
                "industry": industry,
            },
        )

    def boss_detail(self, security_id: str) -> dict[str, Any]:
        return self._opencli_command(
            site="boss",
            action="detail",
            tool="boss_detail",
            values={"security-id": security_id},
            context={"security_id": security_id},
        )

    def boss_chatlist(
        self,
        page: int = 1,
        limit: int = 20,
        job_id: str = "0",
    ) -> dict[str, Any]:
        return self._opencli_command(
            site="boss",
            action="chatlist",
            tool="boss_chatlist",
            values={
                "page": max(1, min(50, page)),
                "limit": max(1, min(100, limit)),
                "job-id": job_id,
            },
            context={
                "page": max(1, min(50, page)),
                "limit": max(1, min(100, limit)),
                "job_id": job_id,
            },
        )

    def boss_send(self, uid: str, text: str) -> dict[str, Any]:
        return self._opencli_command(
            site="boss",
            action="send",
            tool="boss_send",
            values={"uid": uid, "text": text},
            context={"uid": uid, "text": text},
        )

    def _twitter_search_impl(
        self,
        *,
        tool: str,
        query: str,
        max_results: int = 10,
        tab: str = "",
    ) -> dict[str, Any]:
        search_type = _twitter_type(tab)
        args = ["search"]
        if search_type:
            args.extend(["--type", search_type])
        args.extend(["--max", str(max_results), query, "--json"])
        return self._run(
            self.settings.twitter_bin,
            args,
            tool,
        )

    def _opencli_site_search(
        self,
        *,
        site: str,
        query: str,
        limit: int,
        tool: str,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(20, limit))
        command_prefix, resolve_error = self._opencli_command_prefix()
        if not command_prefix:
            return {
                "ok": False,
                "tool": tool,
                "query": query,
                "error_code": "missing_dependency",
                "error": resolve_error or f"{self.settings.opencli_bin} not found",
            }

        registry_result = self._opencli_registry(command_prefix)
        if not registry_result.get("ok"):
            return {
                **registry_result,
                "tool": tool,
                "query": query,
            }

        registry = registry_result["data"]
        command_name = f"{site}/search"
        command_spec = registry.get(command_name)
        if not command_spec:
            return {
                "ok": False,
                "tool": tool,
                "query": query,
                "error_code": "unsupported_command",
                "error": f"{command_name} is not available in the current opencli registry",
            }

        command = [*command_prefix, site, "search"]
        # Some sites use "keyword" instead of "query" (e.g., zhihu)
        query_param = "keyword" if self._opencli_supports_arg(command_spec, "keyword") else "query"
        if self._opencli_arg_is_positional(command_spec, query_param):
            command.append(query)
        else:
            command.extend([f"--{query_param}", query])
        if self._opencli_supports_arg(command_spec, "limit"):
            command.extend(["--limit", str(safe_limit)])
        command.extend(["-f", "json"])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.subprocess_timeout,
        )
        if proc.returncode != 0:
            return _subprocess_failure(
                tool,
                command,
                returncode=proc.returncode,
                stderr=proc.stderr or "",
                stdout=proc.stdout or "",
                context={"query": query},
            )

        parsed = try_load_json(proc.stdout or "")
        if parsed is None:
            return {
                "ok": False,
                "tool": tool,
                "query": query,
                "command": command,
                "error_code": "invalid_json",
                "error": "opencli returned non-JSON output",
            }

        return {
            "ok": True,
            "tool": tool,
            "query": query,
            "command": command,
            "data": parsed,
        }

    def _opencli_command(
        self,
        *,
        site: str,
        action: str,
        tool: str,
        values: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        command_prefix, resolve_error = self._opencli_command_prefix()
        if not command_prefix:
            return {
                "ok": False,
                "tool": tool,
                **(context or {}),
                "error_code": "missing_dependency",
                "error": resolve_error or f"{self.settings.opencli_bin} not found",
            }

        registry_result = self._opencli_registry(command_prefix)
        if not registry_result.get("ok"):
            return {
                **registry_result,
                "tool": tool,
                **(context or {}),
            }

        registry = registry_result["data"]
        command_name = f"{site}/{action}"
        command_spec = registry.get(command_name)
        if not command_spec:
            return {
                "ok": False,
                "tool": tool,
                **(context or {}),
                "error_code": "unsupported_command",
                "error": f"{command_name} is not available in the current opencli registry",
            }

        command = [*command_prefix, site, action]
        args = command_spec.get("args")
        if isinstance(args, list):
            for item in args:
                if not isinstance(item, dict):
                    continue
                arg_name = item.get("name")
                if not isinstance(arg_name, str) or arg_name not in values:
                    continue

                value = values[arg_name]
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                if isinstance(value, bool):
                    if not value:
                        continue
                    if item.get("positional"):
                        command.append("true")
                    else:
                        command.append(f"--{arg_name}")
                    continue

                if item.get("positional"):
                    command.append(str(value))
                else:
                    command.extend([f"--{arg_name}", str(value)])

        command.extend(["-f", "json"])

        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.subprocess_timeout,
        )
        if proc.returncode != 0:
            return _subprocess_failure(
                tool,
                command,
                returncode=proc.returncode,
                stderr=proc.stderr or "",
                stdout=proc.stdout or "",
                context=context,
            )

        parsed = try_load_json(proc.stdout or "")
        if parsed is None:
            return {
                "ok": False,
                "tool": tool,
                **(context or {}),
                "command": command,
                "error_code": "invalid_json",
                "error": "opencli returned non-JSON output",
            }

        return {
            "ok": True,
            "tool": tool,
            **(context or {}),
            "command": command,
            "data": parsed,
        }

    def twitter_read(self, ref: str) -> dict[str, Any]:
        return self._run(
            self.settings.twitter_bin,
            ["tweet", ref, "--json"],
            "twitter_read",
        )

    def xhs_search(
        self,
        query: str,
        page: int = 10,
        sort: str = "general",
        note_type: str = "all",
        pages: int | None = None,
    ) -> dict[str, Any]:
        page_limit = pages if pages is not None else page
        page_limit = max(1, min(10, page_limit))
        seen_ids: set[str] = set()
        merged_items: list[dict[str, Any]] = []
        enrichment_errors: list[dict[str, Any]] = []
        commands: list[list[str]] = []
        pages_fetched = 0
        has_more = False

        for current_page in range(1, page_limit + 1):
            result = self._run(
                self.settings.xhs_bin,
                [
                    "search",
                    "--sort",
                    _xhs_sort(sort),
                    "--type",
                    _xhs_type(note_type),
                    "--page",
                    str(current_page),
                    query,
                    "--json",
                ],
                "xhs_search",
            )
            if not result.get("ok"):
                return {
                    **result,
                    "phase": str(result.get("phase") or "search"),
                    "page": current_page,
                }

            command = result.get("command")
            if isinstance(command, list):
                commands.append(command)

            data = result.get("data")
            payload = _xhs_payload(data)
            items = _xhs_items(data)
            for item in items:
                normalized_item = _xhs_search_item(item)
                item_id = str(normalized_item.get("id", ""))
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                merged_items.append(normalized_item)

            pages_fetched += 1
            has_more = bool(payload.get("has_more"))

        enrich_limit = min(_XHS_SEARCH_ENRICH_LIMIT, len(merged_items))
        for index in range(enrich_limit):
            item = merged_items[index]
            note_id = str(item.get("id") or "").strip()
            if not note_id:
                continue
            read_result = self.xhs_read(note_id)
            if not read_result.get("ok"):
                enrichment_errors.append(_xhs_enrichment_error(note_id, index + 1, read_result))
                continue
            data = read_result.get("data") if isinstance(read_result, dict) else None
            note = data.get("note") if isinstance(data, dict) else None
            if isinstance(note, dict) and note:
                merged_items[index] = _xhs_merge_search_item_with_note(item, note)

        return {
            "ok": True,
            "tool": "xhs_search",
                "command": commands[-1] if commands else [],
            "commands": commands,
            "data": {
                "query": query,
                "sort": _xhs_sort(sort),
                "type": _xhs_type(note_type),
                "page_start": 1,
                "pages_requested": page_limit,
                "pages_fetched": pages_fetched,
                "has_more": has_more,
                "enriched_items": enrich_limit - len(enrichment_errors),
                "enrichment_errors": enrichment_errors,
                "items": merged_items,
            },
        }

    def xhs_read(self, ref: str, xsec_token: str = "") -> dict[str, Any]:
        resolved_ref, resolved_token, ref_error = _xhs_ref(ref, xsec_token)
        if ref_error:
            return _xhs_invalid_ref_response("xhs_read", ref, ref_error)

        # Construct full URL for opencli xiaohongshu note
        full_url = _xhs_full_url(resolved_ref, resolved_token)

        result = self._run(
            "opencli",
            ["xiaohongshu", "note", full_url, "-f", "json"],
            "xhs_read",
        )
        if not result.get("ok"):
            return result

        # opencli returns different format than old xhs CLI
        note = _transform_opencli_note(result.get("data"), resolved_ref)
        if not note:
            return _xhs_empty_note_response("xhs_read", resolved_ref, resolved_token)

        return {
            "ok": True,
            "tool": "xhs_read",
            "command": result.get("command", []),
            "data": {
                "ref": resolved_ref,
                "note": note,
            },
        }

    def xhs_read_cmt(
        self,
        ref: str,
        xsec_token: str = "",
        cursor: str = "",
        fetch_all: bool = False,
    ) -> dict[str, Any]:
        resolved_ref, resolved_token, ref_error = _xhs_ref(ref, xsec_token)
        if ref_error:
            return _xhs_invalid_ref_response("xhs_read_cmt", ref, ref_error)

        # Construct full URL for opencli xiaohongshu comments
        full_url = _xhs_full_url(resolved_ref, resolved_token)

        # Build opencli command args
        args = ["xiaohongshu", "comments", full_url]
        # Note: cursor is not supported by opencli, ignored
        # fetch_all maps to --limit 50 (max allowed by opencli)
        if fetch_all:
            args.extend(["--limit", "50"])
        args.extend(["-f", "json"])

        result = self._run(
            "opencli",
            args,
            "xhs_read_cmt",
        )
        if not result.get("ok"):
            return result

        # opencli returns different format than old xhs CLI
        comments = _transform_opencli_comments(result.get("data"))
        if not comments:
            return _xhs_empty_note_response("xhs_read_cmt", resolved_ref, resolved_token)

        return {
            "ok": True,
            "tool": "xhs_read_cmt",
            "command": result.get("command", []),
            "data": comments,
        }

    def xhs_transcribe_full(self, ref: str, xsec_token: str = "") -> dict[str, Any]:
        resolved_ref, resolved_token, ref_error = _xhs_ref(ref, xsec_token)
        if ref_error:
            return _xhs_invalid_ref_response("xhs_transcribe_full", ref, ref_error)

        result = self._run(
            self.settings.xhs_bin,
            _xhs_read_args(resolved_ref, resolved_token),
            "xhs_transcribe_full",
        )
        if not result.get("ok"):
            return result

        note = _xhs_note(result.get("data"))
        if not note:
            return _xhs_empty_note_response("xhs_transcribe_full", resolved_ref, resolved_token)

        if note.get("media_type") != "video":
            transcript = str(note.get("content_text", "")).strip()
            return {
                "ok": True,
                "tool": "xhs_transcribe_full",
                "command": result.get("command", []),
                "data": {
                    "ref": resolved_ref,
                    "note": note,
                    "transcript_kind": "note",
                    "transcript_path": "",
                    "transcript_chars": len(transcript),
                    "transcript": transcript,
                },
            }

        video_url = _xhs_video_url(result.get("data"))
        if not video_url:
            return {
                "ok": False,
                "tool": "xhs_transcribe_full",
                "error_code": "video_unavailable",
                "error": "xhs note does not expose a downloadable video stream",
            }

        try:
            data = self._transcribe_xhs_video_data(note["id"], video_url)
        except (
            InvalidXhsVideoError,
            MissingDependencyError,
            XhsDownloadError,
            TranscriptionError,
        ) as exc:
            return self._xhs_transcribe_error("xhs_transcribe_full", exc)

        return {
            "ok": True,
            "tool": "xhs_transcribe_full",
            "command": result.get("command", []),
            "data": {
                "ref": resolved_ref,
                "note": note,
                "transcript_kind": "audio",
                "transcript_path": data["transcript_path"],
                "transcript_chars": len(data["transcript"]),
                "transcript": data["transcript"],
            },
        }

    def xhs_user_posts(self, user_id: str, cursor: str = "") -> dict[str, Any]:
        args = ["xiaohongshu", "user", user_id]
        # Note: opencli doesn't support cursor, using limit instead
        # cursor parameter kept for backward compatibility but ignored
        args.extend(["--limit", "20"])
        return self._run(
            "opencli",
            args,
            "xhs_user_posts",
        )

    def bili_search(
        self,
        query: str,
        search_type: str = "video",
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_type = "video"
        safe_page = max(1, min(50, page))
        safe_limit = max(1, min(50, limit))
        try:
            bili_bin = resolve_bili_bin(self.settings.bili_bin)
        except MissingDependencyError as exc:
            return {
                "ok": False,
                "tool": "bili_search",
                "query": query,
                "search_type": normalized_type,
                "page": safe_page,
                "limit": safe_limit,
                "phase": "search",
                "error_code": "missing_dependency",
                "error": str(exc),
            }

        search_args = ["search", "--page", str(safe_page), "--type", normalized_type, query, "--json"]
        search_result = self._run(
            bili_bin,
            search_args,
            "bili_search",
        )
        if not search_result.get("ok"):
            return {
                **search_result,
                "query": query,
                "search_type": normalized_type,
                "page": safe_page,
                "limit": safe_limit,
                "phase": str(search_result.get("phase") or "search"),
            }

        commands: list[list[str]] = []
        search_command = search_result.get("command")
        if isinstance(search_command, list):
            commands.append(search_command)

        items: list[dict[str, Any]] = []
        seen_bvids: set[str] = set()
        for raw_item in _bili_search_items(search_result.get("data")):
            normalized_item = _bili_search_item(raw_item)
            bvid = str(normalized_item.get("bvid") or "").strip()
            if not bvid or bvid in seen_bvids:
                continue
            seen_bvids.add(bvid)
            items.append(normalized_item)
            if len(items) >= safe_limit:
                break

        enrichment_errors: list[dict[str, Any]] = []
        enriched_items = 0
        for index, item in enumerate(items, start=1):
            bvid = str(item.get("bvid") or "").strip()
            if not bvid:
                continue
            video_result = self._run(
                bili_bin,
                ["video", bvid, "--json"],
                "bili_search",
            )
            video_command = video_result.get("command")
            if isinstance(video_command, list):
                commands.append(video_command)
            if not video_result.get("ok"):
                enrichment_errors.append(_bili_enrichment_error(bvid, index, video_result))
                continue
            video_item = _bili_video_item(video_result.get("data"))
            if not video_item:
                enrichment_errors.append(
                    _bili_enrichment_error(
                        bvid,
                        index,
                        {
                            "error_code": "invalid_video",
                            "error": "bili video returned no video data",
                        },
                    )
                )
                continue
            items[index - 1] = _merge_bili_search_item_with_video(item, video_item)
            enriched_items += 1

        return {
            "ok": True,
            "tool": "bili_search",
            "query": query,
            "search_type": normalized_type,
            "page": safe_page,
            "limit": safe_limit,
            "command": search_command if isinstance(search_command, list) else [],
            "commands": commands,
            "data": {
                "query": query,
                "type": normalized_type,
                "page": safe_page,
                "limit": safe_limit,
                "enriched_items": enriched_items,
                "enrichment_errors": enrichment_errors,
                "items": items,
            },
        }

    def bili_get_user_videos(
        self,
        uid: str,
        order: str = "pubdate",
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        normalized_order = order.strip().lower() or "pubdate"
        if normalized_order not in {"pubdate", "click", "stow"}:
            normalized_order = "pubdate"
        safe_page = max(1, min(50, page))
        safe_limit = max(1, min(50, limit))
        return self._opencli_command(
            site="bilibili",
            action="user-videos",
            tool="bili_get_user_videos",
            values={
                "uid": uid,
                "order": normalized_order,
                "page": safe_page,
                "limit": safe_limit,
            },
            context={
                "uid": uid,
                "order": normalized_order,
                "page": safe_page,
                "limit": safe_limit,
            },
        )

    def _transcribe_bili_audio_data(self, bili_id: str) -> dict[str, str]:
        return transcribe_bili_audio(
            bili_id,
            bili_bin=self.settings.bili_bin,
            asr_model=self.settings.asr_model,
            audio_dir=self.settings.audio_dir,
            timeout=self.settings.subprocess_timeout,
        )

    def _transcribe_youtube_audio_data(self, url: str) -> dict[str, str]:
        return transcribe_youtube_audio(
            url,
            ytdlp_bin=self.settings.ytdlp_bin,
            ffmpeg_bin=self.settings.ffmpeg_bin,
            asr_model=self.settings.asr_model,
            audio_dir=self.settings.audio_dir,
            timeout=self.settings.subprocess_timeout,
        )

    def _transcribe_xhs_video_data(self, note_id: str, video_url: str) -> dict[str, str]:
        return transcribe_xhs_video(
            note_id,
            video_url,
            ffmpeg_bin=self.settings.ffmpeg_bin,
            asr_model=self.settings.asr_model,
            audio_dir=self.settings.audio_dir,
            timeout=self.settings.subprocess_timeout,
        )

    def _summary_cache_path(self, audio_root: Path, namespace: str, item_id: str, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return audio_root / "transcripts" / namespace / item_id / f"{digest}.json"

    def _load_cached_summary(
        self,
        cache_path: Path,
        *,
        id_key: str,
        expected_id: str,
        query: str,
        summary_model_default: str,
    ) -> dict[str, Any] | None:
        if not cache_path.is_file():
            return None

        try:
            raw = cache_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

        payload = try_load_json(raw)
        if not isinstance(payload, dict):
            return None

        cached_query = payload.get("query")
        summary = payload.get("summary")
        transcript_path = payload.get("transcript_path")
        summary_model = payload.get("summary_model")
        if payload.get(id_key) != expected_id:
            return None
        if not isinstance(cached_query, str) or cached_query != query:
            return None
        if not isinstance(summary, str) or not summary.strip():
            return None
        if not isinstance(transcript_path, str) or not transcript_path.strip():
            return None
        if not isinstance(summary_model, str) or not summary_model.strip():
            summary_model = summary_model_default

        chunk_count = payload.get("chunk_count")
        transcript_chars = payload.get("transcript_chars")
        summary_chars = payload.get("summary_chars")
        if not isinstance(chunk_count, int) or chunk_count < 1:
            chunk_count = 1
        if not isinstance(transcript_chars, int) or transcript_chars < 0:
            transcript_chars = 0
        if not isinstance(summary_chars, int) or summary_chars < 0:
            summary_chars = len(summary)

        return {
            id_key: expected_id,
            "query": query,
            "summary_model": summary_model,
            "transcript_path": transcript_path,
            "transcript_kind": "summary",
            "transcript_chars": transcript_chars,
            "chunk_count": chunk_count,
            "summary": summary,
            "summary_chars": summary_chars,
            "transcript": summary,
        }

    def _store_cached_summary(self, cache_path: Path, payload: dict[str, Any]) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(dump_json(payload), encoding="utf-8")
        except OSError:
            return

    def _bili_summary_cache_path(self, audio_root: Path, bili_id: str, query: str) -> Path:
        return self._summary_cache_path(audio_root, "bili_summary", bili_id, query)

    def _load_cached_bili_summary(
        self,
        audio_root: Path,
        bili_id: str,
        query: str,
    ) -> dict[str, Any] | None:
        cache_path = self._bili_summary_cache_path(audio_root, bili_id, query)
        cached = self._load_cached_summary(
            cache_path,
            id_key="bili_id",
            expected_id=bili_id,
            query=query,
            summary_model_default=BILI_TRANSCRIPT_SUMMARY_MODEL,
        )
        if cached is None:
            return None

        cached["summary_model"] = BILI_TRANSCRIPT_SUMMARY_MODEL
        return cached

    def _store_cached_bili_summary(
        self,
        audio_root: Path,
        bili_id: str,
        query: str,
        data: dict[str, Any],
    ) -> None:
        cache_path = self._bili_summary_cache_path(audio_root, bili_id, query)
        payload = {
            "bili_id": bili_id,
            "query": query,
            "summary_model": BILI_TRANSCRIPT_SUMMARY_MODEL,
            "transcript_path": data.get("transcript_path", ""),
            "transcript_chars": data.get("transcript_chars", 0),
            "chunk_count": data.get("chunk_count", 1),
            "summary": data.get("summary", ""),
            "summary_chars": data.get("summary_chars", 0),
        }
        self._store_cached_summary(cache_path, payload)

    def _youtube_audio_summary_cache_path(self, audio_root: Path, youtube_id: str, query: str) -> Path:
        return self._summary_cache_path(audio_root, "youtube_audio_summary", youtube_id, query)

    def _load_cached_youtube_audio_summary(
        self,
        audio_root: Path,
        youtube_id: str,
        query: str,
    ) -> dict[str, Any] | None:
        cache_path = self._youtube_audio_summary_cache_path(audio_root, youtube_id, query)
        return self._load_cached_summary(
            cache_path,
            id_key="youtube_id",
            expected_id=youtube_id,
            query=query,
            summary_model_default=self.settings.model,
        )

    def _store_cached_youtube_audio_summary(
        self,
        audio_root: Path,
        youtube_id: str,
        query: str,
        data: dict[str, Any],
    ) -> None:
        cache_path = self._youtube_audio_summary_cache_path(audio_root, youtube_id, query)
        payload = {
            "youtube_id": youtube_id,
            "query": query,
            "summary_model": data.get("summary_model", ""),
            "transcript_path": data.get("transcript_path", ""),
            "transcript_chars": data.get("transcript_chars", 0),
            "chunk_count": data.get("chunk_count", 1),
            "summary": data.get("summary", ""),
            "summary_chars": data.get("summary_chars", 0),
        }
        self._store_cached_summary(cache_path, payload)

    def _youtube_transcribe_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        mapping = (
            (InvalidYouTubeIdError, "invalid_youtube_id"),
            (MissingDependencyError, "missing_dependency"),
            (YouTubeDownloadError, "download_failed"),
            (TranscriptionError, "transcription_failed"),
            (TranscriptSummaryError, "summary_failed"),
        )
        return _tool_error_response(tool_name, exc, mapping)

    def _xhs_transcribe_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        mapping = (
            (InvalidXhsVideoError, "video_unavailable"),
            (MissingDependencyError, "missing_dependency"),
            (XhsDownloadError, "download_failed"),
            (TranscriptionError, "transcription_failed"),
        )
        return _tool_error_response(tool_name, exc, mapping)

    def _bili_transcribe_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        mapping = (
            (InvalidBiliIdError, "invalid_bili_id"),
            (MissingDependencyError, "missing_dependency"),
            (BiliDownloadError, "download_failed"),
            (TranscriptionError, "transcription_failed"),
            (TranscriptSummaryError, "summary_failed"),
            (BiliTranscribeError, "transcription_error"),
        )
        return _tool_error_response(tool_name, exc, mapping)

    def bili_transcribe(self, bili_id: str, query: str) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "invalid_query",
                "error": "query cannot be empty",
            }
        try:
            resolved_id = parse_bili_id(bili_id)
            audio_root = resolve_audio_root(self.settings.audio_dir)
            cached = self._load_cached_bili_summary(audio_root, resolved_id, normalized_query)
            if cached is not None:
                return {
                    "ok": True,
                    "tool": "bili_transcribe",
                    "data": cached,
                }

            data = self._transcribe_bili_audio_data(resolved_id)
            summary, chunk_count = summarize_transcript_for_query(
                self.settings.new_client(),
                transcript=data["transcript"],
                query=normalized_query,
                transcript_path=data["transcript_path"],
                model=self.settings.model,
            )
        except (
            InvalidBiliIdError,
            MissingDependencyError,
            BiliDownloadError,
            TranscriptionError,
            TranscriptSummaryError,
            BiliTranscribeError,
        ) as exc:
            return self._bili_transcribe_error("bili_transcribe", exc)

        result_data = {
            "bili_id": data["bili_id"],
            "query": normalized_query,
            "summary_model": self.settings.model,
            "transcript_path": data["transcript_path"],
            "transcript_kind": "summary",
            "transcript_chars": len(data["transcript"]),
            "chunk_count": chunk_count,
            "summary": summary,
            "summary_chars": len(summary),
            "transcript": summary,
        }
        self._store_cached_bili_summary(audio_root, data["bili_id"], normalized_query, result_data)
        return {
            "ok": True,
            "tool": "bili_transcribe",
            "data": result_data,
        }

    def bili_transcribe_full(self, bili_id: str) -> dict[str, Any]:
        try:
            data = self._transcribe_bili_audio_data(bili_id)
        except (
            InvalidBiliIdError,
            MissingDependencyError,
            BiliDownloadError,
            TranscriptionError,
            BiliTranscribeError,
        ) as exc:
            return self._bili_transcribe_error("bili_transcribe_full", exc)

        return {
            "ok": True,
            "tool": "bili_transcribe_full",
            "data": {
                "bili_id": data["bili_id"],
                "transcript_path": data["transcript_path"],
                "transcript": data["transcript"],
            },
        }

    def youtube_transcribe(self, url: str, query: str) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            return {
                "ok": False,
                "tool": "youtube_transcribe",
                "error_code": "invalid_query",
                "error": "query cannot be empty",
            }

        try:
            youtube_id = parse_youtube_id(url)
            audio_root = resolve_audio_root(self.settings.audio_dir)
            cached = self._load_cached_youtube_audio_summary(audio_root, youtube_id, normalized_query)
            if cached is not None:
                return {
                    "ok": True,
                    "tool": "youtube_transcribe",
                    "data": cached,
                }

            data = self._transcribe_youtube_audio_data(url)
            summary, chunk_count = summarize_transcript_for_query(
                self.settings.new_client(),
                transcript=data["transcript"],
                query=normalized_query,
                transcript_path=data["transcript_path"],
                model=self.settings.model,
            )
        except (
            InvalidYouTubeIdError,
            MissingDependencyError,
            YouTubeDownloadError,
            TranscriptionError,
            TranscriptSummaryError,
        ) as exc:
            return self._youtube_transcribe_error("youtube_transcribe", exc)

        result_data = {
            "youtube_id": data["youtube_id"],
            "query": normalized_query,
            "summary_model": self.settings.model,
            "transcript_path": data["transcript_path"],
            "transcript_kind": "summary",
            "transcript_chars": len(data["transcript"]),
            "chunk_count": chunk_count,
            "summary": summary,
            "summary_chars": len(summary),
            "transcript": summary,
        }
        self._store_cached_youtube_audio_summary(audio_root, data["youtube_id"], normalized_query, result_data)
        return {
            "ok": True,
            "tool": "youtube_transcribe",
            "data": result_data,
        }

    def youtube_transcribe_full(self, url: str) -> dict[str, Any]:
        try:
            data = self._transcribe_youtube_audio_data(url)
        except (
            InvalidYouTubeIdError,
            MissingDependencyError,
            YouTubeDownloadError,
            TranscriptionError,
        ) as exc:
            return self._youtube_transcribe_error("youtube_transcribe_full", exc)

        return {
            "ok": True,
            "tool": "youtube_transcribe_full",
            "data": {
                "youtube_id": data["youtube_id"],
                "transcript_path": data["transcript_path"],
                "transcript": data["transcript"],
            },
        }

    def gen_img(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str | None = None,
    ) -> dict[str, Any]:
        try:
            data = generate_image(
                prompt,
                api_key=self.settings.nano_banana_api_key,
                model=self.settings.nano_banana_model,
                image_dir=self.settings.image_dir,
                aspect_ratio=aspect_ratio,
                image_size=image_size or self.settings.image_size,
                timeout=self.settings.subprocess_timeout,
            )
        except MissingImageApiKeyError as exc:
            return {
                "ok": False,
                "tool": "gen_img",
                "error_code": "missing_api_key",
                "error": str(exc),
            }
        except ImageGenerationError as exc:
            return {
                "ok": False,
                "tool": "gen_img",
                "error_code": "generation_failed",
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "gen_img",
            "data": data,
        }

    def gen_slides(
        self,
        prompt: str,
        slide_count: int = 1,
        template_name: str | None = None,
        html_path: str | None = None,
    ) -> dict[str, Any]:
        try:
            data = generate_slides(
                prompt,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                model=self.settings.model,
                slide_count=slide_count,
                template_name=template_name,
                timeout=self.settings.subprocess_timeout,
                html_path=html_path,
            )
        except SlideGenerationError as exc:
            return {
                "ok": False,
                "tool": "gen_slides",
                "error_code": "generation_failed",
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "gen_slides",
            "data": data,
        }

    def _opencli_command_prefix(self) -> tuple[list[str] | None, str | None]:
        binary = self.settings.opencli_bin.strip() or "opencli"
        path = Path(binary)
        fallback = self._opencli_fallback_prefix(binary)
        if path.suffix.lower() == ".js" and path.exists():
            node = shutil.which("node")
            if not node:
                if fallback:
                    return fallback, None
                return None, "node not found"
            return [node, str(path)], None
        if path.exists():
            return [str(path)], None
        resolved = shutil.which(binary)
        if resolved:
            return [resolved], None
        if fallback:
            return fallback, None
        return None, f"{binary} not found"

    def _opencli_fallback_prefix(self, configured_binary: str) -> list[str] | None:
        if configured_binary.lower() == "opencli":
            return None
        if "\\" not in configured_binary and "/" not in configured_binary:
            return None
        resolved = shutil.which("opencli")
        if resolved:
            return [resolved]
        return None

    def _opencli_registry(self, command_prefix: list[str]) -> dict[str, Any]:
        cache_key = "\0".join(command_prefix)
        with _OPENCLI_REGISTRY_LOCK:
            cached = _OPENCLI_REGISTRY_CACHE.get(cache_key)
        if cached is not None:
            return {
                "ok": True,
                "data": cached,
            }

        command = [*command_prefix, "list", "-f", "json"]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.subprocess_timeout,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "command": command,
                "error_code": "registry_failed",
                "error": (proc.stderr or proc.stdout or "").strip()[:4000],
            }

        parsed = try_load_json(proc.stdout or "")
        if not isinstance(parsed, list):
            return {
                "ok": False,
                "command": command,
                "error_code": "invalid_registry",
                "error": "opencli list returned unreadable JSON output",
            }

        commands: dict[str, dict[str, Any]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            command_name = item.get("command")
            if isinstance(command_name, str) and command_name:
                commands[command_name] = item

        with _OPENCLI_REGISTRY_LOCK:
            _OPENCLI_REGISTRY_CACHE[cache_key] = commands
        return {
            "ok": True,
            "data": commands,
        }

    def _opencli_supports_arg(self, command_spec: dict[str, Any], arg_name: str) -> bool:
        args = command_spec.get("args")
        if not isinstance(args, list):
            return False
        for item in args:
            if isinstance(item, dict) and item.get("name") == arg_name:
                return True
        return False

    def _opencli_arg_is_positional(self, command_spec: dict[str, Any], arg_name: str) -> bool:
        args = command_spec.get("args")
        if not isinstance(args, list):
            return False
        for item in args:
            if isinstance(item, dict) and item.get("name") == arg_name:
                return bool(item.get("positional"))
        return False

    def _run(self, binary: str, args: list[str], tool: str) -> dict[str, Any]:
        resolved = shutil.which(binary)
        if not resolved:
            return {
                "ok": False,
                "tool": tool,
                "error": f"{binary} not found",
            }
        command = [resolved, *args]
        env = dict(os.environ)
        # Many of the optional CLIs are Python-based (click/typer) and may emit
        # emoji in titles or help text. Force UTF-8 so subprocess output remains
        # readable on Windows GBK locales and doesn't crash the tool.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.settings.subprocess_timeout,
            env=env,
        )
        if proc.returncode != 0:
            error_text = (proc.stderr or proc.stdout).strip()[:4000]
            parsed_error = try_load_json(error_text)
            if (
                isinstance(parsed_error, dict)
                and parsed_error.get("ok") is False
                and isinstance(parsed_error.get("error"), dict)
            ):
                payload = parsed_error["error"]
                code = payload.get("code")
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    error_payload = _subprocess_failure(
                        tool,
                        command,
                        returncode=proc.returncode,
                        stderr=proc.stderr or "",
                        stdout=proc.stdout or "",
                    )
                    error_payload["error_code"] = str(code or "command_failed")
                    error_payload["error"] = message.strip()[:4000]
                    return error_payload
            return _subprocess_failure(
                tool,
                command,
                returncode=proc.returncode,
                stderr=proc.stderr or "",
                stdout=proc.stdout or "",
            )

        parsed = try_load_json(proc.stdout)
        return {
            "ok": True,
            "tool": tool,
            "command": command,
            "data": parsed if parsed is not None else proc.stdout.strip(),
        }
