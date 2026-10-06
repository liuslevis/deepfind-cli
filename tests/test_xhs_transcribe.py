from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepfind.xhs_transcribe import load_cached_xhs_transcript, store_xhs_transcript


class XhsTranscribeTests(unittest.TestCase):
    def test_store_and_load_transcript_uses_grouped_resource_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_root = Path(temp_dir)
            stored = store_xhs_transcript(video_root, "note-1", "cached line")
            cached = load_cached_xhs_transcript(video_root, "note-1")

            self.assertEqual(
                stored,
                video_root / "xhs" / "note-1" / "text" / "transcript.txt",
            )
            self.assertEqual(cached, (stored, "cached line"))
            self.assertTrue((video_root / "xhs" / "note-1" / "audio").is_dir())
            self.assertTrue((video_root / "xhs" / "note-1" / "video").is_dir())


if __name__ == "__main__":
    unittest.main()
