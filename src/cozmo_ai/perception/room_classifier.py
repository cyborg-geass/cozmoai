from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frame_sampling import SampledFrame, read_sampled_frames
from .schemas import ClassScore, FrameEvidence, RoomSemantics, skipped_room_semantics


DEFAULT_LABELS = [
    "bedroom",
    "living room",
    "kitchen",
    "bathroom",
    "corridor",
    "office",
    "dining room",
    "storage room",
    "utility room",
    "other room",
]

PROMPT_TEMPLATES = {
    "bedroom": [
        "a photo of a bedroom",
        "an indoor bedroom",
        "a room containing a bed",
        "a residential bedroom interior",
    ],
    "corridor": [
        "a photo of an indoor corridor",
        "a hallway inside a building",
        "an indoor passageway",
    ],
}

GENERIC_PROMPTS = [
    "a photo of a {label}",
    "an indoor {label}",
    "a residential {label} interior",
]


class ModelUnavailableError(RuntimeError):
    pass


def prompts_for_label(label: str) -> list[str]:
    return PROMPT_TEMPLATES.get(
        label,
        [
            template.format(label=label)
            for template in GENERIC_PROMPTS
        ],
    )


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def aggregate_frame_probabilities(
    frame_probabilities: list[dict[str, float]],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    if not frame_probabilities:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "stability": 0.0,
            "top_classes": [],
        }

    labels = list(frame_probabilities[0].keys())
    mean_scores = {
        label: float(
            np.mean(
                [
                    probabilities.get(label, 0.0)
                    for probabilities in frame_probabilities
                ]
            )
        )
        for label in labels
    }
    final_label = max(
        mean_scores,
        key=mean_scores.get,
    )
    top_classes = [
        ClassScore(
            label=label,
            score=score,
        )
        for label, score in sorted(
            mean_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]
    frame_top_labels = [
        max(
            probabilities,
            key=probabilities.get,
        )
        for probabilities in frame_probabilities
    ]
    stability = frame_top_labels.count(final_label) / len(frame_top_labels)

    return {
        "label": final_label,
        "confidence": mean_scores[final_label],
        "stability": float(stability),
        "top_classes": top_classes,
    }


class ZeroShotRoomClassifier:
    def __init__(
        self,
        *,
        model_name: str,
        labels: list[str],
        device: str = "auto",
        allow_download: bool = False,
        offline_model_path: str | None = None,
    ) -> None:
        self.model_name = offline_model_path or model_name
        self.labels = labels
        self.device_name = device
        self.allow_download = allow_download
        self._model = None
        self._processor = None
        self._torch = None

    def _resolve_device(self):
        torch = self._torch
        if self.device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_name

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ModelUnavailableError(
                "Optional AI dependencies are not installed. "
                "Install torch, transformers, and pillow to enable room semantics."
            ) from exc

        try:
            local_files_only = not self.allow_download
            self._processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
            self._model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
            )
            self._torch = torch
            self._model.to(
                self._resolve_device()
            )
            self._model.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"Room-classification model is unavailable: {exc}"
            ) from exc

    def predict(self, frames: list[SampledFrame]) -> list[dict[str, float]]:
        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        prompts_by_label = [
            prompts_for_label(label)
            for label in self.labels
        ]
        prompts = [
            prompt
            for group in prompts_by_label
            for prompt in group
        ]

        device = self._resolve_device()
        torch = self._torch

        with torch.inference_mode():
            text_inputs = self._processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)
            text_features = self._model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            label_features = []
            offset = 0
            for prompt_group in prompts_by_label:
                width = len(prompt_group)
                features = text_features[offset: offset + width].mean(dim=0)
                features = features / features.norm()
                label_features.append(features)
                offset += width
            label_features = torch.stack(label_features)

            frame_probabilities = []
            for frame in frames:
                image_inputs = self._processor(
                    images=frame.image_rgb,
                    return_tensors="pt",
                ).to(device)
                image_features = self._model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                logits = (image_features @ label_features.T).squeeze(0)
                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                ).detach().cpu().numpy()
                frame_probabilities.append(
                    {
                        label: float(probability)
                        for label, probability in zip(
                            self.labels,
                            probabilities,
                            strict=True,
                        )
                    }
                )

        return frame_probabilities


def write_contact_sheet(
    *,
    frames: list[SampledFrame],
    output_path: Path,
    title: str,
    frame_labels: list[str] | None = None,
    columns: int = 4,
) -> None:
    if not frames:
        return

    cell_width = 320
    cell_height = 240
    header_height = 54
    rows = int(np.ceil(len(frames) / columns))
    sheet = np.full(
        (
            rows * (cell_height + header_height),
            columns * cell_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for i, frame in enumerate(frames):
        row = i // columns
        column = i % columns
        x0 = column * cell_width
        y0 = row * (cell_height + header_height)
        resized = cv2.resize(
            frame.image_rgb,
            (cell_width, cell_height),
            interpolation=cv2.INTER_AREA,
        )
        sheet[y0 + header_height: y0 + header_height + cell_height, x0: x0 + cell_width] = resized

        label = f"frame {frame.frame_index}"
        if frame_labels:
            label = f"{label} | {frame_labels[i]}"
        cv2.putText(
            sheet,
            label,
            (x0 + 8, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            title[:52],
            (x0 + 8, y0 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            sheet,
            cv2.COLOR_RGB2BGR,
        ),
    )


def classify_room(
    capture_dir: Path,
    output_dir: Path,
    *,
    labels: list[str] | None = None,
    sample_count: int = 16,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    allow_download: bool = False,
    offline_model_path: str | None = None,
    skip_model: bool = False,
) -> RoomSemantics:
    labels = labels or DEFAULT_LABELS
    room_output_dir = output_dir / "room_semantics"
    room_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = read_sampled_frames(
        capture_dir / "rgb.mp4",
        target_count=sample_count,
    )
    frame_evidence = [
        frame.evidence()
        for frame in frames
    ]
    sampled_frames = [
        frame.frame_index
        for frame in frames
    ]
    contact_sheet = room_output_dir / "contact_sheet.png"

    if not frames:
        semantics = skipped_room_semantics(
            sampled_frames=[],
            frame_evidence=[],
            contact_sheet=None,
            reason="No RGB frames could be sampled from the capture.",
            model=model_name,
            status="error",
        )
    elif skip_model:
        write_contact_sheet(
            frames=frames,
            output_path=contact_sheet,
            title="room semantics skipped",
        )
        semantics = skipped_room_semantics(
            sampled_frames=sampled_frames,
            frame_evidence=frame_evidence,
            contact_sheet=str(contact_sheet.relative_to(output_dir)),
            reason="AI room classification disabled by command-line flag.",
            model=model_name,
            status="skipped",
        )
    else:
        classifier = ZeroShotRoomClassifier(
            model_name=model_name,
            labels=labels,
            device=device,
            allow_download=allow_download,
            offline_model_path=offline_model_path,
        )
        try:
            frame_probabilities = classifier.predict(frames)
            aggregate = aggregate_frame_probabilities(frame_probabilities)
            frame_labels = [
                max(
                    probabilities,
                    key=probabilities.get,
                )
                for probabilities in frame_probabilities
            ]
            title = (
                f"{aggregate['label']} "
                f"conf={aggregate['confidence']:.2f} "
                f"stability={aggregate['stability']:.2f}"
            )
            write_contact_sheet(
                frames=frames,
                output_path=contact_sheet,
                title=title,
                frame_labels=frame_labels,
            )
            semantics = RoomSemantics(
                label=aggregate["label"],
                confidence=aggregate["confidence"],
                stability=aggregate["stability"],
                method="zero_shot_vlm",
                model=classifier.model_name,
                sampled_frames=sampled_frames,
                top_classes=aggregate["top_classes"],
                status="measured",
                frame_evidence=frame_evidence,
                contact_sheet=str(contact_sheet.relative_to(output_dir)),
            )
        except ModelUnavailableError as exc:
            write_contact_sheet(
                frames=frames,
                output_path=contact_sheet,
                title="room model unavailable",
            )
            semantics = skipped_room_semantics(
                sampled_frames=sampled_frames,
                frame_evidence=frame_evidence,
                contact_sheet=str(contact_sheet.relative_to(output_dir)),
                reason=str(exc),
                model=model_name,
                status="model_unavailable",
            )

    output_path = room_output_dir / "room_semantics.json"
    output_path.write_text(
        json.dumps(
            {
                "room_semantics": semantics.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return semantics
