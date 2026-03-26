from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from deepfind.output import render_answer


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True
