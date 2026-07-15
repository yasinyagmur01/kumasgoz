"""Frame capture stubs: folder replay, video file, webcam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    """One line-camera frame for the live loop."""

    image_bgr: np.ndarray
    path: str
    label: int | None = None
    index: int = 0


class FrameSource(Protocol):
    def frames(self, limit: int = 0) -> Iterator[Frame]: ...


def discover_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def infer_label_from_path(path: Path) -> int | None:
    """MVTec-style: parent folder `good` → 0, else anomaly → 1."""
    parent = path.parent.name.lower()
    if parent == "good":
        return 0
    # test set defect folders (color, cut, hole, ...)
    if path.parts and "test" in {p.lower() for p in path.parts}:
        return 1 if parent != "good" else 0
    return None


@dataclass
class FolderFrameSource:
    """Replay images from a directory tree (simulated line camera)."""

    folder: Path

    def frames(self, limit: int = 0) -> Iterator[Frame]:
        images = discover_images(self.folder)
        if limit > 0:
            images = images[:limit]
        for i, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                continue
            yield Frame(
                image_bgr=image,
                path=str(path.resolve()),
                label=infer_label_from_path(path),
                index=i,
            )


@dataclass
class VideoFrameSource:
    """Read frames from a video file or webcam index."""

    source: str | int
    path_prefix: str = "capture"
    max_frames: int = 0

    def frames(self, limit: int = 0) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open capture source: {self.source}")
        n_limit = limit if limit > 0 else self.max_frames
        i = 0
        try:
            while True:
                if n_limit > 0 and i >= n_limit:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield Frame(
                    image_bgr=frame,
                    path=f"{self.path_prefix}/{i:06d}.jpg",
                    label=None,
                    index=i,
                )
                i += 1
        finally:
            cap.release()


def open_frame_source(
    kind: str,
    *,
    path: str | Path | None = None,
    device_index: int = 0,
    path_prefix: str = "capture",
    max_frames: int = 0,
) -> FrameSource:
    """Factory: kind = folder | video | webcam."""
    kind = kind.lower().strip()
    if kind == "folder":
        if path is None:
            raise ValueError("folder capture requires path=")
        return FolderFrameSource(folder=Path(path))
    if kind == "video":
        if path is None:
            raise ValueError("video capture requires path=")
        return VideoFrameSource(source=str(path), path_prefix=path_prefix, max_frames=max_frames)
    if kind == "webcam":
        return VideoFrameSource(
            source=int(device_index),
            path_prefix=path_prefix or "webcam",
            max_frames=max_frames,
        )
    raise ValueError(f"Unknown capture kind: {kind}")
