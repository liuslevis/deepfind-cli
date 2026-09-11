"""Integration test: verify the deepfind-coding image ships the preinstalled
Python libraries the sandbox is documented to provide.

Skipped when Docker is unavailable or the image is not built locally. Set
DEEPFIND_CODING_TEST_IMAGE to override the image reference (default:
``deepfind-coding:local``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest


IMAGE = os.environ.get("DEEPFIND_CODING_TEST_IMAGE", "deepfind-coding:local")

PREINSTALLED = [
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "pypdf",
    "docx",  # python-docx installs as the `docx` module
    "scipy",
    "seaborn",
]

PROBE = (
    "import json, importlib;"
    f"mods={PREINSTALLED!r};"
    "print(json.dumps({m: importlib.import_module(m).__version__ "
    "for m in mods if hasattr(importlib.import_module(m), '__version__')}))"
)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@unittest.skipUnless(_docker_available(), f"docker or image {IMAGE} unavailable")
class CodingImagePreinstalledLibrariesTest(unittest.TestCase):
    def test_preinstalled_libraries_import(self) -> None:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--entrypoint",
                "/opt/venv-template/bin/python",
                IMAGE,
                "-c",
                PROBE,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"probe failed: stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        versions = json.loads(result.stdout.strip().splitlines()[-1])
        for module in PREINSTALLED:
            if module == "docx":
                continue  # python-docx exposes version via docx.__version__ >=1.0
            self.assertIn(module, versions, f"{module} missing from image")
            self.assertTrue(versions[module], f"{module} has empty version")


if __name__ == "__main__":
    unittest.main()
