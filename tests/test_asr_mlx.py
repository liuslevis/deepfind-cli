"""Unit tests for MLX Whisper ASR backend."""

import platform
import unittest
from unittest.mock import MagicMock, patch, call

from deepfind.asr import (
    is_mlx_whisper_model,
    is_qwen3_asr_model,
    load_model,
    transcribe_audio,
    MissingDependencyError,
    MLX_WHISPER_MODELS,
)


class TestMLXWhisperDetection(unittest.TestCase):
    """Test MLX Whisper model detection."""

    def test_mlx_whisper_prefix(self):
        """Test detection of mlx-whisper: prefix."""
        self.assertTrue(is_mlx_whisper_model("mlx-whisper:large-v3"))
        self.assertTrue(is_mlx_whisper_model("mlx-whisper:medium"))
        self.assertTrue(is_mlx_whisper_model("mlx-whisper:base"))
        self.assertTrue(is_mlx_whisper_model("mlx-whisper:tiny"))

    def test_whisper_prefix(self):
        """Test detection of whisper: prefix."""
        self.assertTrue(is_mlx_whisper_model("whisper:large-v3"))
        self.assertTrue(is_mlx_whisper_model("whisper:base"))

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        self.assertTrue(is_mlx_whisper_model("MLX-WHISPER:base"))
        self.assertTrue(is_mlx_whisper_model("Whisper:medium"))

    def test_qwen_not_mlx(self):
        """Test that Qwen models are not detected as MLX."""
        self.assertFalse(is_mlx_whisper_model("Qwen/Qwen3-ASR-1.7B"))
        self.assertFalse(is_mlx_whisper_model("Qwen/Qwen2-Audio"))

    def test_unknown_not_mlx(self):
        """Test that unknown models are not detected as MLX."""
        self.assertFalse(is_mlx_whisper_model("some-random-model"))
        self.assertFalse(is_mlx_whisper_model(""))


class TestQwen3ASRDetection(unittest.TestCase):
    """Test Qwen3-ASR model detection."""

    def test_qwen3_asr_detection(self):
        """Test detection of Qwen3-ASR models."""
        self.assertTrue(is_qwen3_asr_model("Qwen/Qwen3-ASR-1.7B"))
        self.assertTrue(is_qwen3_asr_model("Qwen3-ASR-Flash"))

    def test_qwen2_not_qwen3(self):
        """Test that Qwen2 is not detected as Qwen3-ASR."""
        self.assertFalse(is_qwen3_asr_model("Qwen/Qwen2-Audio"))

    def test_mlx_not_qwen(self):
        """Test that MLX models are not detected as Qwen."""
        self.assertFalse(is_qwen3_asr_model("mlx-whisper:base"))


class TestMLXWhisperLoadModel(unittest.TestCase):
    """Test MLX Whisper model loading."""

    @patch("deepfind.asr.is_mlx_whisper_model")
    @patch("deepfind.asr.load_local_secrets")
    def test_load_mlx_whisper_success(self, mock_secrets, mock_is_mlx):
        """Test successful MLX Whisper model loading."""
        mock_is_mlx.return_value = True

        # Mock mlx_whisper module in sys.modules
        import sys
        mock_mlx = MagicMock()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            backend, model, processor, device = load_model("mlx-whisper:large-v3")

            self.assertEqual(backend, "mlx_whisper")
            self.assertEqual(model, "large-v3-mlx")  # Model size extracted
            self.assertIsNone(processor)
            self.assertEqual(device, "mps")

    @patch("deepfind.asr.is_mlx_whisper_model")
    @patch("deepfind.asr.load_local_secrets")
    def test_load_mlx_whisper_default_size(self, mock_secrets, mock_is_mlx):
        """Test MLX Whisper loading with default model size."""
        mock_is_mlx.return_value = True

        import sys
        mock_mlx = MagicMock()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            backend, model, processor, device = load_model("mlx-whisper")

            self.assertEqual(backend, "mlx_whisper")
            self.assertEqual(model, "base")  # Default size
            self.assertIsNone(processor)
            self.assertEqual(device, "mps")

    @patch("deepfind.asr.is_mlx_whisper_model")
    @patch("deepfind.asr.load_local_secrets")
    def test_load_mlx_whisper_missing_dependency(self, mock_secrets, mock_is_mlx):
        """Test MLX Whisper loading raises error when mlx-whisper not installed."""
        mock_is_mlx.return_value = True

        import sys
        # Make sure mlx_whisper is not in sys.modules
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            with self.assertRaises(MissingDependencyError) as ctx:
                load_model("mlx-whisper:base")

            self.assertIn("mlx-whisper is not installed", str(ctx.exception))


class TestMLXWhisperTranscribe(unittest.TestCase):
    """Test MLX Whisper transcription."""

    def test_transcribe_mlx_whisper_success(self):
        """Test successful MLX Whisper transcription."""
        from pathlib import Path
        import sys

        mock_audio_path = Path("/fake/audio.wav")
        mock_result = {"text": "  Hello world  "}

        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = mock_result

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            result = transcribe_audio(
                mock_audio_path,
                backend="mlx_whisper",
                model="large-v3",  # Model size
                processor=None,
                device="mps"
            )

            self.assertEqual(result, "Hello world")  # Trimmed
            mock_mlx.transcribe.assert_called_once_with(
                str(mock_audio_path),
                path_or_hf_repo="large-v3"
            )

    def test_transcribe_mlx_whisper_empty_result(self):
        """Test MLX Whisper transcription with empty result."""
        from pathlib import Path
        import sys

        mock_audio_path = Path("/fake/audio.wav")
        mock_result = {"text": "   "}

        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = mock_result

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            result = transcribe_audio(
                mock_audio_path,
                backend="mlx_whisper",
                model="base",
                processor=None,
                device="mps"
            )

            self.assertEqual(result, "")

    def test_transcribe_mlx_whisper_no_text_key(self):
        """Test MLX Whisper transcription when result has no text key."""
        from pathlib import Path
        import sys

        mock_audio_path = Path("/fake/audio.wav")
        mock_result = {"segments": []}  # No 'text' key

        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = mock_result

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            result = transcribe_audio(
                mock_audio_path,
                backend="mlx_whisper",
                model="tiny",
                processor=None,
                device="mps"
            )

            self.assertEqual(result, "")

    def test_transcribe_mlx_whisper_missing_dependency(self):
        """Test MLX Whisper transcription raises error when not installed."""
        from pathlib import Path
        import sys

        mock_audio_path = Path("/fake/audio.wav")

        # Make mlx_whisper import fail
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            with self.assertRaises(MissingDependencyError) as ctx:
                transcribe_audio(
                    mock_audio_path,
                    backend="mlx_whisper",
                    model="base",
                    processor=None,
                    device="mps"
                )

            self.assertIn("mlx-whisper is not installed", str(ctx.exception))


class TestPlatformAwareConfig(unittest.TestCase):
    """Test platform-aware ASR model configuration."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("deepfind.config.platform.system")
    def test_darwin_default_model(self, mock_platform):
        """Test that Darwin (macOS) defaults to MLX Whisper."""
        mock_platform.return_value = "Darwin"

        from deepfind.config import Settings

        settings = Settings._resolve_asr_model()
        self.assertEqual(settings, "mlx-whisper:large-v3")

    @patch.dict("os.environ", {}, clear=True)
    @patch("deepfind.config.platform.system")
    def test_linux_default_model(self, mock_platform):
        """Test that Linux defaults to Qwen3-ASR."""
        mock_platform.return_value = "Linux"

        from deepfind.config import Settings

        settings = Settings._resolve_asr_model()
        self.assertEqual(settings, "Qwen/Qwen3-ASR-1.7B")

    @patch.dict("os.environ", {"ASR_MODEL_MAC": "mlx-whisper:tiny"}, clear=True)
    @patch("deepfind.config.platform.system")
    def test_darwin_with_mac_env(self, mock_platform):
        """Test Darwin uses ASR_MODEL_MAC when set."""
        mock_platform.return_value = "Darwin"

        from deepfind.config import Settings

        settings = Settings._resolve_asr_model()
        self.assertEqual(settings, "mlx-whisper:tiny")

    @patch.dict("os.environ", {"ASR_MODEL_PC": "Qwen/Custom"}, clear=True)
    @patch("deepfind.config.platform.system")
    def test_linux_with_pc_env(self, mock_platform):
        """Test Linux uses ASR_MODEL_PC when set."""
        mock_platform.return_value = "Linux"

        from deepfind.config import Settings

        settings = Settings._resolve_asr_model()
        self.assertEqual(settings, "Qwen/Custom")



if __name__ == "__main__":
    unittest.main()
