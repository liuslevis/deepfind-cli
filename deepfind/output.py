from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO


def render_answer(answer: str, viewer: str = "pretify", stream: TextIO = sys.stdout) -> None:
    if viewer == "plain" or not _is_interactive(stream):
        print(answer, file=stream)
        return

    if viewer in {"pretify"} and _run_frogmouth(answer):
        return

    print(answer, file=stream)


def _is_interactive(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty()) and sys.stdin.isatty()
    except Exception:
        return False


def _run_frogmouth(answer: str) -> bool:
    command = _frogmouth_command()
    if not command:
        return False

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".md",
            delete=False,
        ) as handle:
            handle.write(answer)
            temp_path = Path(handle.name)
        subprocess.run([*command, str(temp_path)], check=False)
        return True
    except OSError:
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def _frogmouth_command() -> list[str]:
    if binary := shutil.which("frogmouth"):
        return [binary]
    if importlib.util.find_spec("frogmouth") is not None:
        return [sys.executable, "-m", "frogmouth"]
    return []
