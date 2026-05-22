from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from .json_utils import dump_json


def render_answer(answer: str, stream: TextIO = sys.stdout) -> None:
    print(answer, file=stream)


def render_json_answer(answer: Any, stream: TextIO = sys.stdout) -> None:
    print(dump_json(answer), file=stream)
