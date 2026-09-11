from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frame_sampling import read_sampled_frames
from .projection import BoundingBox, bbox_iou


@dataclass(frozen=True)
class Detection2D:
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    model_name: str

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(*self.bbox_xyxy)


class OpeningModelUnavailableError(RuntimeError):
    pass


def best_associated_detection(
    *,
    geometry_bbox: BoundingBox,
    detections: list[Detection2D],
    min_iou: float,
    min_confidence: float,
) -> tuple[Detection2D | None, float]:
    best_detection = None
    best_iou = 0.0

    for detection in detections:
        if detection.confidence < min_confidence:
            continue

        iou = bbox_iou(
            geometry_bbox,
            detection.bbox,
        )
        if iou > best_iou:
            best_detection = detection
            best_iou = iou

    if best_detection is None or best_iou < min_iou:
        return None, best_iou

    return best_detection, best_iou


class OpenVocabularyOpeningDetector:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        allow_download: bool = False,
        offline_model_path: str | None = None,
    ) -> None:
        self.model_name = offline_model_path or model_name
        self.device = device
        self.allow_download = allow_download
        self.offline_model_path = offline_model_path

    def detect(self, image_rgb: np.ndarray, prompts: list[str]) -> list[Detection2D]:
        try:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as exc:
            raise OpeningModelUnavailableError(
                "Optional opening semantic dependencies are not installed. "
                "Install torch, transformers, and pillow to enable detection."
            ) from exc

        local_files_only = not self.allow_download
        try:
            processor = OwlViTProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
            model = OwlViTForObjectDetection.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise OpeningModelUnavailableError(
                f"Opening semantic model is unavailable: {exc}"
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        with torch.inference_mode():
            inputs = processor(
                text=[prompts],
                images=image_rgb,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor(
                [image_rgb.shape[:2]],
                device=device,
            )
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
            )[0]

        detections = []
        for score, label_index, box in zip(
            results["scores"].detach().cpu().numpy(),
            results["labels"].detach().cpu().numpy(),
            results["boxes"].detach().cpu().numpy(),
            strict=True,
        ):
            detections.append(
                Detection2D(
                    class_name=prompts[int(label_index)],
                    confidence=float(score),
                    bbox_xyxy=[
                        float(value)
                        for value in box
                    ],
                    model_name=self.model_name,
                )
            )

        return detections


def opening_semantics_payload(
    *,
    opening: dict[str, Any],
    detection: Detection2D | None,
    association_iou: float | None,
    status: str,
) -> dict[str, Any]:
    return {
        "opening_id": opening["opening_id"],
        "wall_id": opening.get("wall_id"),
        "geometry": opening["geometry"],
        "semantics": {
            "class": detection.class_name if detection else "unknown",
            "confidence": detection.confidence if detection else 0.0,
            "method": (
                "open_vocabulary_detector"
                if detection
                else "open_vocabulary_detector"
            ),
            "association_iou": association_iou,
            "status": status,
        },
    }


def write_overview_image(
    *,
    capture_dir: Path,
    output_path: Path,
    text_lines: list[str],
) -> None:
    frames = read_sampled_frames(
        capture_dir / "rgb.mp4",
        target_count=1,
    )
    if frames:
        image = frames[0].image_rgb.copy()
    else:
        image = np.full((720, 960, 3), 245, dtype=np.uint8)

    image = cv2.resize(
        image,
        (960, 720),
        interpolation=cv2.INTER_AREA,
    )
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (960, 112),
        (255, 255, 255),
        -1,
    )
    image = cv2.addWeighted(
        overlay,
        0.85,
        image,
        0.15,
        0,
    )

    for row, text in enumerate(text_lines):
        cv2.putText(
            image,
            text[:96],
            (20, 36 + row * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        ),
    )


def normalize_geometry_openings(openings_data: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for index, candidate in enumerate(openings_data.get("candidates", [])):
        normalized.append(
            {
                "opening_id": index + 1,
                "wall_id": candidate.get(
                    "wall_id",
                    openings_data.get("wall_id"),
                ),
                "geometry": {
                    "width_m": candidate.get("width_m"),
                    "height_m": candidate.get("height_m"),
                    "height_status": candidate.get("height_status"),
                    "method": "wall_occupancy_geometry",
                    "source": candidate,
                },
            }
        )
    return normalized


def classify_openings(
    *,
    capture_dir: Path,
    opening_json: Path,
    output_dir: Path,
    enable_model: bool = True,
    model_name: str = "google/owlvit-base-patch32",
    device: str = "auto",
    allow_download: bool = False,
    min_iou: float = 0.30,
    min_confidence: float = 0.20,
) -> dict[str, Any]:
    opening_data = json.loads(
        opening_json.read_text(
            encoding="utf-8",
        )
    )
    output_path = output_dir / "opening_semantics" / "semantic_openings.json"
    overview_path = output_dir / "opening_semantics" / "opening_semantics_overview.png"
    openings = normalize_geometry_openings(opening_data)

    if not openings:
        result = {
            "opening_semantics": {
                "status": "no_candidates",
                "method": "geometry_candidate_fusion",
                "model": model_name,
                "openings": [],
                "message": (
                    "No geometric opening candidates were available for "
                    "semantic classification."
                ),
                "evidence": {
                    "overview": str(
                        overview_path.relative_to(output_dir)
                    ),
                },
            }
        }
        write_overview_image(
            capture_dir=capture_dir,
            output_path=overview_path,
            text_lines=[
                "Opening semantics: no geometric candidates",
                "Semantic detector did not invent openings or dimensions",
            ],
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result

    if not enable_model:
        semantic_openings = [
            opening_semantics_payload(
                opening=opening,
                detection=None,
                association_iou=None,
                status="skipped",
            )
            for opening in openings
        ]
        status = "skipped"
        message = "Opening semantic model disabled by command-line flag."
    else:
        semantic_openings = [
            opening_semantics_payload(
                opening=opening,
                detection=None,
                association_iou=None,
                status="model_unavailable",
            )
            for opening in openings
        ]
        status = "model_unavailable"
        message = (
            "Opening semantic detector requires projected candidate boxes. "
            "Current geometric opening candidates do not include 3D corner "
            "projections, so semantic association was not evaluated."
        )

    write_overview_image(
        capture_dir=capture_dir,
        output_path=overview_path,
        text_lines=[
            f"Opening semantics: {status}",
            message,
        ],
    )
    result = {
        "opening_semantics": {
            "status": status,
            "method": "geometry_candidate_fusion",
            "model": model_name,
            "min_iou": min_iou,
            "min_confidence": min_confidence,
            "openings": semantic_openings,
            "message": message,
            "evidence": {
                "overview": str(
                    overview_path.relative_to(output_dir)
                ),
            },
        }
    }
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
