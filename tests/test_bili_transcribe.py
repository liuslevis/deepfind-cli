from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import deepfind.asr as asr
import deepfind.bili_transcribe as bili_transcribe
from deepfind.bili_transcribe import (
    InvalidBiliIdError,
    ensure_segments,
    gpu_asr_slot,
    parse_bili_id,
    resolve_audio_root,
    transcribe_bili_audio,
)


class BiliTranscribeTests(unittest.TestCase):
    def test_parse_bili_id_accepts_bvid(self) -> None:
        self.assertEqual(parse_bili_id("BV1cgPSzeEj5"), "BV1cgPSzeEj5")

    def test_parse_bili_id_extracts_from_url(self) -> None:
        url = "https://www.bilibili.com/video/BV1cgPSzeEj5?spm_id_from=333.1007.tianma.1-1-1.click"
        self.assertEqual(parse_bili_id(url), "BV1cgPSzeEj5")

    def test_parse_bili_id_rejects_invalid_value(self) -> None:
        with self.assertRaises(InvalidBiliIdError):
            parse_bili_id("https://example.com/not-bilibili")

    def test_resolve_audio_root_uses_repo_relative_default(self) -> None:
        path = resolve_audio_root("audio")
        self.assertEqual(path.name, "audio")
        self.assertTrue(path.is_absolute())

    def test_ensure_segments_reuses_existing_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            first = output_dir / "seg_001.mp3"
            second = output_dir / "seg_002.wav"
            first.write_bytes(b"audio")
            second.write_bytes(b"audio")

            with patch("deepfind.bili_transcribe.resolve_bili_bin") as resolve_mock:
                segments = ensure_segments(
                    "BV1cgPSzeEj5",
                    output_dir=output_dir,
                    bili_bin="bili",
                    timeout=5,
                )

        self.assertEqual(segments, [first, second])
        resolve_mock.assert_not_called()

    def test_transcribe_bili_audio_writes_transcript_and_no_summary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seg1 = tmp_path / "seg_001.mp3"
            seg2 = tmp_path / "seg_002.mp3"
            seg1.write_bytes(b"a")
            seg2.write_bytes(b"b")

            with patch("deepfind.bili_transcribe.ensure_segments", return_value=[seg1, seg2]):
                with patch(
                    "deepfind.bili_transcribe.transcribe_segments",
                    return_value="first line\nsecond line",
                ):
                    result = transcribe_bili_audio(
                        "https://www.bilibili.com/video/BV1cgPSzeEj5",
                        audio_dir=tmpdir,
                        asr_model="Qwen/Qwen3-ASR-1.7B",
                        bili_bin="bili",
                        timeout=30,
                    )

            transcript_path = Path(result["transcript_path"])
            self.assertTrue(transcript_path.exists())
            self.assertEqual(result["bili_id"], "BV1cgPSzeEj5")
            self.assertEqual(result["transcript"], "first line\nsecond line")
            self.assertEqual(
                transcript_path.read_text(encoding="utf-8"),
                "first line\nsecond line\n",
            )
            self.assertFalse((tmp_path / "transcripts" / "summary.txt").exists())

    def test_transcribe_bili_audio_uses_cached_transcript_and_skips_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            transcript_dir = tmp_path / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            cached_path = transcript_dir / "BV1cgPSzeEj5.txt"
            cached_path.write_text("cached line\n", encoding="utf-8")

            with patch("deepfind.bili_transcribe.ensure_segments") as ensure_mock:
                with patch("deepfind.bili_transcribe.transcribe_segments") as transcribe_mock:
                    result = transcribe_bili_audio(
                        "BV1cgPSzeEj5",
                        audio_dir=tmpdir,
                    )

        self.assertEqual(result["bili_id"], "BV1cgPSzeEj5")
        self.assertEqual(result["transcript"], "cached line")
        self.assertEqual(result["transcript_path"], str(cached_path))
        ensure_mock.assert_not_called()
        transcribe_mock.assert_not_called()

    def test_transcribe_bili_audio_uses_gpu_queue_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seg1 = tmp_path / "seg_001.mp3"
            seg1.write_bytes(b"a")

            slot = MagicMock()
            slot.__enter__.return_value = None
            slot.__exit__.return_value = False

            with patch("deepfind.bili_transcribe.ensure_segments", return_value=[seg1]):
                with patch("deepfind.asr.gpu_asr_slot", return_value=slot) as slot_mock:
                    with patch("deepfind.asr.load_model", return_value=("qwen3_asr", object(), None, "cpu")):
                        with patch("deepfind.asr.transcribe_audio", return_value="line one"):
                            transcribe_bili_audio(
                                "BV1cgPSzeEj5",
                                audio_dir=tmpdir,
                                asr_model="Qwen/Qwen3-ASR-1.7B",
                                bili_bin="bili",
                                timeout=30,
                            )

        slot_mock.assert_called_once()
        slot.__enter__.assert_called_once()
        slot.__exit__.assert_called_once()

    def test_gpu_asr_slot_skips_semaphore_without_gpu(self) -> None:
        semaphore = MagicMock()
        with patch("deepfind.asr._gpu_available", return_value=False):
            with patch.object(asr, "_GPU_ASR_SEMAPHORE", semaphore):
                with gpu_asr_slot():
                    pass
        semaphore.acquire.assert_not_called()
        semaphore.release.assert_not_called()

    def test_gpu_asr_slot_acquires_and_releases_with_gpu(self) -> None:
        semaphore = MagicMock()
        with patch("deepfind.asr._gpu_available", return_value=True):
            with patch.object(asr, "_GPU_ASR_SEMAPHORE", semaphore):
                with gpu_asr_slot():
                    pass
        semaphore.acquire.assert_called_once()
        semaphore.release.assert_called_once()
