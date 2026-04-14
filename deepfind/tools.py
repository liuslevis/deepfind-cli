from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Any

from .bili_transcribe import (
    BiliDownloadError,
    BiliTranscribeError,
    InvalidBiliIdError,
    MissingDependencyError,
    TranscriptionError,
    parse_bili_id,
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

_WEB_SEARCH_ENGINES = frozenset({"google", "bing", "baidu"})
_OPENCLI_REGISTRY_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_OPENCLI_REGISTRY_LOCK = Lock()
_XHS_TOPIC_TAG_RE = re.compile(r"#[^#\n]+#")


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
    note["content_text"] = _xhs_content_text(note)
    return note


class Toolset:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._functions = {
            "web_search": self.web_search,
            "web_fetch": self.web_fetch,
            "browser_fetch": self.browser_fetch,
            "arxiv_search": self.arxiv_search,
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
            "xhs_search_user": self.xhs_search_user,
            "xhs_user": self.xhs_user,
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
                "Read one Xiaohongshu note by URL or ID.",
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
                "xhs_search_user",
                "Search Xiaohongshu users by keyword.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_user",
                "Read one Xiaohongshu user profile by user ID.",
                {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                    },
                    "required": ["user_id"],
                    "additionalProperties": False,
                },
            ),
            self._function_spec(
                "xhs_user_posts",
                "List a Xiaohongshu user's posts by user ID.",
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
                "Search Bilibili videos or users via opencli.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "search_type": {
                            "type": "string",
                            "enum": ["video", "user"],
                        },
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
                "Generate a standalone HTML slide deck and save it under the local tmp directory.",
                {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "slide_count": {"type": "integer", "minimum": 1, "maximum": 12},
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
            return {
                "ok": False,
                "tool": "web_search",
                "engine": engine,
                "query": query,
                "command": command,
                "error_code": "command_failed",
                "error": (proc.stderr or proc.stdout or "").strip()[:4000],
            }

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
                headless=headless,
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
        if self._opencli_arg_is_positional(command_spec, "query"):
            command.append(query)
        else:
            command.extend(["--query", query])
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
            return {
                "ok": False,
                "tool": tool,
                "query": query,
                "command": command,
                "error_code": "command_failed",
                "error": (proc.stderr or proc.stdout or "").strip()[:4000],
            }

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
            return {
                "ok": False,
                "tool": tool,
                **(context or {}),
                "command": command,
                "error_code": "command_failed",
                "error": (proc.stderr or proc.stdout or "").strip()[:4000],
            }

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
                return result

            command = result.get("command")
            if isinstance(command, list):
                commands.append(command)

            data = result.get("data")
            payload = _xhs_payload(data)
            items = _xhs_items(data)
            for item in items:
                item_id = str(item.get("id", ""))
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                merged_items.append(item)

            pages_fetched += 1
            has_more = bool(payload.get("has_more"))

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
                "items": merged_items,
            },
        }

    def xhs_read(self, ref: str) -> dict[str, Any]:
        result = self._run(
            self.settings.xhs_bin,
            ["read", ref, "--json"],
            "xhs_read",
        )
        if not result.get("ok"):
            return result

        note = _xhs_note(result.get("data"))
        if not note:
            return result

        return {
            "ok": True,
            "tool": "xhs_read",
            "command": result.get("command", []),
            "data": {
                "ref": ref,
                "note": note,
            },
        }

    def xhs_search_user(self, query: str) -> dict[str, Any]:
        return self._run(
            self.settings.xhs_bin,
            ["search-user", query, "--json"],
            "xhs_search_user",
        )

    def xhs_user(self, user_id: str) -> dict[str, Any]:
        return self._run(
            self.settings.xhs_bin,
            ["user", user_id, "--json"],
            "xhs_user",
        )

    def xhs_user_posts(self, user_id: str, cursor: str = "") -> dict[str, Any]:
        args = ["user-posts", user_id]
        if cursor:
            args.extend(["--cursor", cursor])
        args.append("--json")
        return self._run(
            self.settings.xhs_bin,
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
        normalized_type = search_type.strip().lower() or "video"
        if normalized_type not in {"video", "user"}:
            normalized_type = "video"
        safe_page = max(1, min(50, page))
        safe_limit = max(1, min(50, limit))
        return self._opencli_command(
            site="bilibili",
            action="search",
            tool="bili_search",
            values={
                "query": query,
                "type": normalized_type,
                "page": safe_page,
                "limit": safe_limit,
            },
            context={
                "query": query,
                "search_type": normalized_type,
                "page": safe_page,
                "limit": safe_limit,
            },
        )

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

    def _bili_summary_cache_path(self, audio_root: Path, bili_id: str, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return audio_root / "transcripts" / "bili_summary" / bili_id / f"{digest}.json"

    def _load_cached_bili_summary(
        self,
        audio_root: Path,
        bili_id: str,
        query: str,
    ) -> dict[str, Any] | None:
        cache_path = self._bili_summary_cache_path(audio_root, bili_id, query)
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
        if payload.get("bili_id") != bili_id:
            return None
        if not isinstance(cached_query, str) or cached_query != query:
            return None
        if not isinstance(summary, str) or not summary.strip():
            return None
        if not isinstance(transcript_path, str) or not transcript_path.strip():
            return None

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
            "bili_id": bili_id,
            "query": query,
            "summary_model": BILI_TRANSCRIPT_SUMMARY_MODEL,
            "transcript_path": transcript_path,
            "transcript_kind": "summary",
            "transcript_chars": transcript_chars,
            "chunk_count": chunk_count,
            "summary": summary,
            "summary_chars": summary_chars,
            "transcript": summary,
        }

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
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(dump_json(payload), encoding="utf-8")
        except OSError:
            return

    def _youtube_audio_summary_cache_path(self, audio_root: Path, youtube_id: str, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return audio_root / "transcripts" / "youtube_audio_summary" / youtube_id / f"{digest}.json"

    def _load_cached_youtube_audio_summary(
        self,
        audio_root: Path,
        youtube_id: str,
        query: str,
    ) -> dict[str, Any] | None:
        cache_path = self._youtube_audio_summary_cache_path(audio_root, youtube_id, query)
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
        if payload.get("youtube_id") != youtube_id:
            return None
        if not isinstance(cached_query, str) or cached_query != query:
            return None
        if not isinstance(summary, str) or not summary.strip():
            return None
        if not isinstance(transcript_path, str) or not transcript_path.strip():
            return None
        if not isinstance(summary_model, str) or not summary_model.strip():
            summary_model = self.settings.model

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
            "youtube_id": youtube_id,
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
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(dump_json(payload), encoding="utf-8")
        except OSError:
            return

    def _youtube_transcribe_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, InvalidYouTubeIdError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "invalid_youtube_id",
                "error": str(exc),
            }
        if isinstance(exc, MissingDependencyError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "missing_dependency",
                "error": str(exc),
            }
        if isinstance(exc, YouTubeDownloadError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "download_failed",
                "error": str(exc),
            }
        if isinstance(exc, TranscriptionError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "transcription_failed",
                "error": str(exc),
            }
        if isinstance(exc, TranscriptSummaryError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "summary_failed",
                "error": str(exc),
            }
        return {
            "ok": False,
            "tool": tool_name,
            "error_code": "unknown_error",
            "error": str(exc),
        }

    def _bili_transcribe_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, InvalidBiliIdError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "invalid_bili_id",
                "error": str(exc),
            }
        if isinstance(exc, MissingDependencyError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "missing_dependency",
                "error": str(exc),
            }
        if isinstance(exc, BiliDownloadError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "download_failed",
                "error": str(exc),
            }
        if isinstance(exc, TranscriptionError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "transcription_failed",
                "error": str(exc),
            }
        if isinstance(exc, TranscriptSummaryError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "summary_failed",
                "error": str(exc),
            }
        if isinstance(exc, BiliTranscribeError):
            return {
                "ok": False,
                "tool": tool_name,
                "error_code": "transcription_error",
                "error": str(exc),
            }
        return {
            "ok": False,
            "tool": tool_name,
            "error_code": "unknown_error",
            "error": str(exc),
        }

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
    ) -> dict[str, Any]:
        try:
            data = generate_slides(
                prompt,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                model=self.settings.model,
                slide_count=slide_count,
                timeout=self.settings.subprocess_timeout,
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
        binary = self.settings.opencli_bin.strip()
        path = Path(binary)
        if path.suffix.lower() == ".js" and path.exists():
            node = shutil.which("node")
            if not node:
                return None, "node not found"
            return [node, str(path)], None
        if path.exists():
            return [str(path)], None
        resolved = shutil.which(binary)
        if resolved:
            return [resolved], None
        return None, f"{binary} not found"

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
                    return {
                        "ok": False,
                        "tool": tool,
                        "command": command,
                        "error_code": str(code or "command_failed"),
                        "error": message.strip()[:4000],
                    }
            return {
                "ok": False,
                "tool": tool,
                "command": command,
                "error": error_text,
            }

        parsed = try_load_json(proc.stdout)
        return {
            "ok": True,
            "tool": tool,
            "command": command,
            "data": parsed if parsed is not None else proc.stdout.strip(),
        }
