from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepfind.video_resources import resolve_video_root, video_resource_paths


class VideoResourceTests(unittest.TestCase):
    def test_resolve_video_root_uses_repo_relative_default(self) -> None:
        path = resolve_video_root(None)
        self.assertEqual(path.name, "video")
        self.assertTrue(path.is_absolute())

    def test_video_resource_paths_groups_all_asset_types_by_video_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = video_resource_paths(
                Path(temp_dir),
                "youtube",
                "dQw4w9WgXcQ",
                create=True,
            )

            self.assertEqual(paths.transcript, paths.root / "text" / "transcript.txt")
            self.assertTrue(paths.audio.is_dir())
            self.assertTrue(paths.video.is_dir())
            self.assertTrue(paths.text.is_dir())

    def test_video_resource_paths_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                video_resource_paths(Path(temp_dir), "bili", "../escape")


if __name__ == "__main__":
    unittest.main()
