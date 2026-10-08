from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from deepfind.cli import main
from deepfind.config import Settings
from deepfind.rag.config import load_config
from deepfind.rag_search import search_rag


def test_load_config_uses_explicit_knowledge_base_root() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config = load_config(root)

    assert config.root_dir == root.resolve()
    assert config.pdf_root == root.resolve() / "pdf"
    assert config.media_root == root.resolve() / "media"


def test_search_rag_calls_in_process_service() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        response = Mock()
        response.model_dump.return_value = {
            "query": "rates",
            "mode": "hybrid",
            "results": [],
        }
        with patch("deepfind.rag.search.SearchService") as service_class:
            service_class.return_value.search.return_value = response
            result = asyncio.run(
                search_rag(Settings(api_key="", rag_dir=temp_dir), "rates")
            )

    service_class.assert_called_once()
    service_class.return_value.search.assert_called_once_with(
        "rates",
        limit=8,
        mode="hybrid",
    )
    assert result == response.model_dump.return_value


def test_main_dispatches_rag_subcommand() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("deepfind.rag.cli.main", return_value=7) as rag_main:
        code = main(
            ["rag", "search", "rates"],
            stdout=stdout,
            stderr=stderr,
        )

    assert code == 7
    rag_main.assert_called_once_with(
        ["search", "rates"],
        stdout=stdout,
        stderr=stderr,
    )
