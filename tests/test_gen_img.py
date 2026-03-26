from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepfind.gen_img import MissingImageApiKeyError, generate_image, resolve_image_root


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class GenImgTests(unittest.TestCase):
    def test_resolve_image_root_uses_repo_relative_default(self) -> None:
        root = resolve_image_root(None)
        self.assertEqual(root.name, "tmp")
        self.assertTrue(root.is_absolute())

    def test_generate_image_requires_api_key(self) -> None:
        with self.assertRaises(MissingImageApiKeyError):
            generate_image("make art", api_key=None)

    def test_generate_image_writes_first_inline_image(self) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "image ready"},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"png-bytes").decode("ascii"),
                                }
                            },
                        ]
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deepfind.gen_img.urllib.request.urlopen", return_value=FakeResponse(payload)):
                result = generate_image(
                    "make art",
                    api_key="nb-key",
                    image_dir=tmpdir,
                    aspect_ratio="16:9",
                    image_size="2K",
                )

            image_path = Path(result["image_path"])
            self.assertTrue(image_path.exists())
            self.assertEqual(image_path.parent, Path(tmpdir))
            self.assertEqual(image_path.read_bytes(), b"png-bytes")
            self.assertEqual(result["mime_type"], "image/png")
            self.assertEqual(result["response_text"], "image ready")
            self.assertEqual(result["aspect_ratio"], "16:9")
            self.assertEqual(result["image_size"], "2K")
