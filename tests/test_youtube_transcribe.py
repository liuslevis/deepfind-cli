from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepfind.youtube_transcribe import (
    InvalidYouTubeIdError,
    load_cached_youtube_transcript,
    normalize_youtube_transcript,
    parse_youtube_id,
    store_youtube_transcript,
)


class YouTubeTranscribeTests(unittest.TestCase):
    def test_parse_youtube_id_accepts_video_id(self) -> None:
        self.assertEqual(parse_youtube_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_parse_youtube_id_extracts_from_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley"
        self.assertEqual(parse_youtube_id(url), "dQw4w9WgXcQ")

    def test_parse_youtube_id_extracts_from_short_url(self) -> None:
        self.assertEqual(
            parse_youtube_id("https://youtu.be/dQw4w9WgXcQ?si=test"),
            "dQw4w9WgXcQ",
        )

    def test_parse_youtube_id_rejects_invalid_value(self) -> None:
        with self.assertRaises(InvalidYouTubeIdError):
            parse_youtube_id("https://example.com/not-youtube")

    def test_normalize_youtube_transcript_flattens_segments(self) -> None:
        transcript = normalize_youtube_transcript(
            [
                {"timestamp": "0:01", "speaker": "", "text": "first line"},
                {"timestamp": "0:35", "speaker": "Host", "text": "second line"},
            ]
        )
        self.assertEqual(transcript, "[0:01] first line\n\n[0:35] Host: second line")

    def test_store_and_load_cached_youtube_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_root = Path(tmpdir)
            stored = store_youtube_transcript(audio_root, "dQw4w9WgXcQ", "cached line")
            cached = load_cached_youtube_transcript(audio_root, "dQw4w9WgXcQ")
            assert cached is not None
            cached_path, transcript = cached
            self.assertEqual(cached_path, stored)
            self.assertEqual(transcript, "cached line")
            self.assertEqual(stored.read_text(encoding="utf-8"), "cached line\n")
