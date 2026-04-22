from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import deepfind.asr as asr


class ASRTests(unittest.TestCase):
    def test_resolve_model_source_uses_cached_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir) / "hub"
            snapshot_dir = hub_dir / "models--Qwen--Qwen3-ASR-1.7B" / "snapshots" / "rev123"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir = hub_dir / "models--Qwen--Qwen3-ASR-1.7B" / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("rev123\n", encoding="utf-8")

            with patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": str(hub_dir)}, clear=False):
                self.assertEqual(asr.resolve_model_source("Qwen/Qwen3-ASR-1.7B"), str(snapshot_dir))

    def test_load_model_uses_cached_snapshot_for_qwen3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir) / "hub"
            snapshot_dir = hub_dir / "models--Qwen--Qwen3-ASR-1.7B" / "snapshots" / "rev123"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            refs_dir = hub_dir / "models--Qwen--Qwen3-ASR-1.7B" / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("rev123\n", encoding="utf-8")

            from_pretrained = MagicMock(return_value=object())
            fake_qwen_asr = SimpleNamespace(from_pretrained=from_pretrained)
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                float16="float16",
                float32="float32",
                bfloat16="bfloat16",
            )

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "qwen_asr": SimpleNamespace(QwenASR=fake_qwen_asr),
                },
            ):
                with patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": str(hub_dir)}, clear=False):
                    backend, model, processor, device = asr.load_model("Qwen/Qwen3-ASR-1.7B")

        self.assertEqual(backend, "qwen3_asr")
        self.assertIsNone(processor)
        self.assertEqual(device, "cpu")
        self.assertIsNotNone(model)
        from_pretrained.assert_called_once()
        self.assertEqual(from_pretrained.call_args.args[0], str(snapshot_dir))


if __name__ == "__main__":
    unittest.main()
