from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import FrameEvidence, FrameQuality


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    image_rgb: np.ndarray
    quality: FrameQuality

    def evidence(self) -> FrameEvidence:
        return FrameEvidence(
            frame_index=self.frame_index,
            quality=self.quality,
        )


def uniform_sample_indices(frame_count: int, target_count: int) -> list[int]:
    if frame_count <= 0:
        return []

    if target_count <= 0:
        raise ValueError("target_count must be positive")

    if frame_count <= target_count:
        return list(range(frame_count))

    raw = np.linspace(
        0,
        frame_count - 1,
        num=target_count,
    )
    indices = sorted(
        {
            int(round(value))
            for value in raw
        }
    )

    candidate = 0
    while len(indices) < target_count and candidate < frame_count:
        if candidate not in indices:
            indices.append(candidate)
        candidate += 1

    return sorted(indices)[:target_count]


def frame_quality(image_rgb: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    return FrameQuality(
        brightness=float(gray.mean()),
        blur_laplacian_var=float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
    )


def video_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return 0
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def read_sampled_frames(
    video_path: Path,
    *,
    target_count: int,
) -> list[SampledFrame]:
    frame_count = video_frame_count(video_path)
    indices = uniform_sample_indices(
        frame_count,
        target_count,
    )

    if not indices:
        return []

    capture = cv2.VideoCapture(str(video_path))
    frames: list[SampledFrame] = []
    used_indices: set[int] = set()

    try:
        if not capture.isOpened():
            return []

        for index in indices:
            frame_bgr = None
            actual_index = index

            for candidate in (index, index - 1, index + 1):
                if candidate < 0 or candidate >= frame_count:
                    continue
                if candidate in used_indices:
                    continue

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    candidate,
                )
                ok, candidate_frame = capture.read()
                if ok and candidate_frame is not None:
                    frame_bgr = candidate_frame
                    actual_index = candidate
                    used_indices.add(candidate)
                    break

            if frame_bgr is None:
                continue

            image_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(
                SampledFrame(
                    frame_index=actual_index,
                    image_rgb=image_rgb,
                    quality=frame_quality(image_rgb),
                )
            )
    finally:
        capture.release()

    return frames
