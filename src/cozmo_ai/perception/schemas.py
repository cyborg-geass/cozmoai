from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SEMANTIC_STATUSES = {
    "measured",
    "supported",
    "low_confidence",
    "insufficient_evidence",
    "not_observed",
    "not_evaluated",
    "model_unavailable",
    "skipped",
    "error",
}


@dataclass(frozen=True)
class ClassScore:
    label: str
    score: float


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    blur_laplacian_var: float


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    quality: FrameQuality
    status: str = "sampled"


@dataclass(frozen=True)
class RoomSemantics:
    label: str
    confidence: float
    stability: float
    method: str
    model: str
    sampled_frames: list[int]
    top_classes: list[ClassScore]
    status: str
    frame_evidence: list[FrameEvidence] = field(default_factory=list)
    contact_sheet: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.status not in SEMANTIC_STATUSES:
            raise ValueError(f"Unsupported semantic status: {self.status}")

        payload = asdict(self)
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }


def skipped_room_semantics(
    *,
    sampled_frames: list[int],
    frame_evidence: list[FrameEvidence],
    contact_sheet: str | None,
    reason: str,
    model: str,
    status: str = "skipped",
) -> RoomSemantics:
    return RoomSemantics(
        label="unknown",
        confidence=0.0,
        stability=0.0,
        method="zero_shot_vlm",
        model=model,
        sampled_frames=sampled_frames,
        top_classes=[],
        status=status,
        frame_evidence=frame_evidence,
        contact_sheet=contact_sheet,
        message=reason,
    )
