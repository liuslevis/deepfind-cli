from __future__ import annotations

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
    transcribe_bili_audio,
)
from .config import Settings
from .gen_slides import SlideGenerationError, generate_slides
from .gen_img import ImageGenerationError, MissingImageApiKeyError, generate_image
from .json_utils import dump_json, try_load_json

_WEB_SEARCH_ENGINES = frozenset({"google", "bing", "baidu"})
_OPENCLI_REGISTRY_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_OPENCLI_REGISTRY_LOCK = Lock()


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


def _xhs_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


class Toolset:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._functions = {
            "web_search": self.web_search,
            "arxiv_search": self.arxiv_search,
            "twitter_search": self.twitter_search,
            "x_search": self.x_search,
            "twitter_read": self.twitter_read,
            "zhihu_search": self.zhihu_search,
            "boss_search": self.boss_search,
            "boss_detail": self.boss_detail,
            "xhs_search": self.xhs_search,
            "xhs_read": self.xhs_read,
            "xhs_search_user": self.xhs_search_user,
            "xhs_user": self.xhs_user,
            "xhs_user_posts": self.xhs_user_posts,
            "bili_transcribe": self.bili_transcribe,
            "gen_img": self.gen_img,
            "gen_slides": self.gen_slides,
        }

    def specs(self) -> list[dict[str, Any]]:
        return [
            self._function_spec(
                "web_search",
                "Search the web through opencli. Prefer this for broad web research, and use the platform-specific tools for Xiaohongshu, X/Twitter, and Bilibili.",
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
                "Search BOSS直聘 job postings / 职位 / 岗位. Use this for 招聘、岗位、职位 queries. Returns security_id values that can be passed to boss_detail.",
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
                "Read one BOSS直聘 job posting / 职位详情 by security_id from boss_search.",
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
                "bili_transcribe",
                "Transcribe Bilibili video audio by URL or BVID.",
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
            items = _xhs_items(data)
            for item in items:
                item_id = str(item.get("id", ""))
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                merged_items.append(item)

            pages_fetched += 1
            has_more = bool(data.get("has_more")) if isinstance(data, dict) else False

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
        return self._run(
            self.settings.xhs_bin,
            ["read", ref, "--json"],
            "xhs_read",
        )

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

    def bili_transcribe(self, bili_id: str) -> dict[str, Any]:
        try:
            data = transcribe_bili_audio(
                bili_id,
                bili_bin=self.settings.bili_bin,
                asr_model=self.settings.asr_model,
                audio_dir=self.settings.audio_dir,
                timeout=self.settings.subprocess_timeout,
            )
        except InvalidBiliIdError as exc:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "invalid_bili_id",
                "error": str(exc),
            }
        except MissingDependencyError as exc:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "missing_dependency",
                "error": str(exc),
            }
        except BiliDownloadError as exc:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "download_failed",
                "error": str(exc),
            }
        except TranscriptionError as exc:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "transcription_failed",
                "error": str(exc),
            }
        except BiliTranscribeError as exc:
            return {
                "ok": False,
                "tool": "bili_transcribe",
                "error_code": "transcription_error",
                "error": str(exc),
            }

        return {
            "ok": True,
            "tool": "bili_transcribe",
            "data": {
                "bili_id": data["bili_id"],
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
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.settings.subprocess_timeout,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "tool": tool,
                "command": command,
                "error": (proc.stderr or proc.stdout).strip()[:4000],
            }

        parsed = try_load_json(proc.stdout)
        return {
            "ok": True,
            "tool": tool,
            "command": command,
            "data": parsed if parsed is not None else proc.stdout.strip(),
        }
