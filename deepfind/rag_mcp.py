from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Settings


def _project_dir(settings: Settings) -> Path:
    path = Path(settings.rag_mcp_project_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _content_text(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


async def search_rag_mcp(
    settings: Settings,
    query: str,
    *,
    limit: int = 8,
    mode: str = "hybrid",
) -> dict[str, Any]:
    project_dir = _project_dir(settings)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"RAG MCP project directory not found: {project_dir}")

    server = StdioServerParameters(
        command=settings.rag_mcp_command,
        args=["run", "deepfind-rag", "mcp"],
        cwd=project_dir,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "search",
                {"query": query, "limit": limit, "mode": mode},
                read_timeout_seconds=float(settings.subprocess_timeout),
            )

    if result.is_error:
        raise RuntimeError(_content_text(result.content) or "RAG MCP search failed")
    if result.structured_content is not None:
        return result.structured_content

    text = _content_text(result.content)
    return {"query": query, "mode": mode, "results": [], "text": text}
