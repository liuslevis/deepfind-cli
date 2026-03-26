from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO


def render_answer(answer: str, stream: TextIO = sys.stdout) -> None:
    print(answer, file=stream)

