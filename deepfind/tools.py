from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .config import Settings
from .json_utils import dump_json, try_load_json


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
                            "enum": ["Top", "Latest", "People", "Photos", "Videos"],
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
                        "page": {"type": "integer", "minimum": 1, "maximum": 5},
                        "sort": {
                            "type": "string",
                            "enum": ["general", "latest_popular", "most_popular"],
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
        tab: str = "Latest",
    ) -> dict[str, Any]:
        return self._run(
            self.settings.twitter_bin,
            ["search", query, "--max", str(max_results), "--tab", tab, "--json"],
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
        page: int = 1,
        sort: str = "general",
        note_type: str = "all",
    ) -> dict[str, Any]:
        return self._run(
            self.settings.xhs_bin,
            [
                "search",
                query,
                "--page",
                str(page),
                "--sort",
                sort,
                "--note-type",
                note_type,
                "--json",
            ],
            "xhs_search",
        )

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
