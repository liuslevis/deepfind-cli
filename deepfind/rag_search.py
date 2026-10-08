from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings


def _project_dir(settings: Settings) -> Path:
    path = Path(settings.rag_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


async def search_rag(
    settings: Settings,
    query: str,
    *,
    limit: int = 8,
    mode: str = "hybrid",
) -> dict[str, Any]:
    project_dir = _project_dir(settings)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"RAG directory not found: {project_dir}")

    from .rag.config import load_config
    from .rag.search import SearchService

    response = SearchService(load_config(project_dir)).search(
        query,
        limit=limit,
        mode=mode,
    )
    return response.model_dump()
