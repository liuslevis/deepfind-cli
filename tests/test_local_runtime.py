from __future__ import annotations

import unittest
from unittest.mock import patch

from deepfind.config import Settings
from deepfind.local_runtime import detect_local_model, ollama_tags_url


class LocalRuntimeTests(unittest.TestCase):
    def test_ollama_tags_url_strips_v1_suffix(self) -> None:
        self.assertEqual(ollama_tags_url("http://127.0.0.1:11434/v1"), "http://127.0.0.1:11434/api/tags")

    def test_detect_local_model_requires_gpu(self) -> None:
        settings = Settings(api_key="")

        with patch("deepfind.local_runtime.detect_gpu") as detect_gpu:
            detect_gpu.return_value = type("Gpu", (), {"available": False, "name": "", "memory_total_mb": None})()
            status = detect_local_model(settings)

        self.assertFalse(status.available)
        self.assertIn("No NVIDIA GPU", status.reason)

    def test_detect_local_model_requires_running_ollama(self) -> None:
        settings = Settings(api_key="", local_model="qwen3.5:9b")

        with (
            patch("deepfind.local_runtime.detect_gpu") as detect_gpu,
            patch("deepfind.local_runtime.list_ollama_models", side_effect=Exception("boom")),
        ):
            detect_gpu.return_value = type("Gpu", (), {"available": True, "name": "RTX", "memory_total_mb": 16384})()
            with patch("deepfind.local_runtime.httpx.HTTPError", Exception):
                status = detect_local_model(settings)

        self.assertFalse(status.available)
        self.assertIn("Ollama is not reachable", status.reason)

    def test_detect_local_model_requires_loaded_model(self) -> None:
        settings = Settings(api_key="", local_model="qwen3.5:9b")

        with (
            patch("deepfind.local_runtime.detect_gpu") as detect_gpu,
            patch("deepfind.local_runtime.list_ollama_models", return_value=["llama3.1:8b"]),
        ):
            detect_gpu.return_value = type("Gpu", (), {"available": True, "name": "RTX", "memory_total_mb": 16384})()
            status = detect_local_model(settings)

        self.assertFalse(status.available)
        self.assertIn("is not loaded in Ollama", status.reason)

    def test_detect_local_model_accepts_ollama_runtime(self) -> None:
        settings = Settings(api_key="", local_model="qwen3.5:9b")

        with (
            patch("deepfind.local_runtime.detect_gpu") as detect_gpu,
            patch("deepfind.local_runtime.list_ollama_models", return_value=["qwen3.5:9b"]),
        ):
            detect_gpu.return_value = type("Gpu", (), {"available": True, "name": "RTX", "memory_total_mb": 16384})()
            status = detect_local_model(settings)

        self.assertTrue(status.available)
        self.assertEqual(status.backend, "ollama")
        self.assertEqual(status.model, settings.local_model)
        self.assertEqual(status.base_url, settings.local_base_url)