from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PLATFORMS = {"bili", "youtube", "xhs"}


@dataclass(frozen=True)
class VideoResourcePaths:
    root: Path
    audio: Path
    video: Path
    text: Path

    @property
    def transcript(self) -> Path:
        return self.text / "transcript.txt"


def resolve_video_root(video_dir: str | None) -> Path:
    raw = (video_dir or "assets/video").strip() or "assets/video"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def video_resource_paths(
    video_root: Path,
    platform: str,
    video_id: str,
    *,
    create: bool = False,
) -> VideoResourcePaths:
    normalized_platform = platform.strip().lower()
    normalized_id = video_id.strip()
    if normalized_platform not in VIDEO_PLATFORMS:
        raise ValueError(f"unsupported video platform: {platform}")
    if (
        not normalized_id
        or normalized_id in {".", ".."}
        or "/" in normalized_id
        or "\\" in normalized_id
    ):
        raise ValueError("video_id must be a single non-empty path component")

    root = video_root / normalized_platform / normalized_id
    paths = VideoResourcePaths(
        root=root,
        audio=root / "audio",
        video=root / "video",
        text=root / "text",
    )
    if create:
        for directory in (paths.audio, paths.video, paths.text):
            directory.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = [
    "VIDEO_PLATFORMS",
    "VideoResourcePaths",
    "resolve_video_root",
    "video_resource_paths",
]
