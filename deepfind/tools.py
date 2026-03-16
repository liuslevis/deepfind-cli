from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .config import Settings
from .json_utils import dump_json, try_load_json


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
            "twitter_search": self.twitter_search,
            "twitter_read": self.twitter_read,
            "xhs_search": self.xhs_search,
            "xhs_read": self.xhs_read,
            "xhs_search_user": self.xhs_search_user,
            "xhs_user": self.xhs_user,
            "xhs_user_posts": self.xhs_user_posts,
        }

    def specs(self) -> list[dict[str, Any]]:
        return [
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

    def twitter_search(
        self,
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
            "twitter_search",
        )

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
