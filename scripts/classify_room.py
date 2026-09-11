from __future__ import annotations

import argparse
import json
from pathlib import Path

from cozmo_ai.perception import classify_room
from cozmo_ai.perception.room_classifier import DEFAULT_LABELS


def parse_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None

    path = Path(value)
    if path.exists():
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
        if isinstance(payload, dict):
            payload = payload.get("labels", [])
        return [
            str(label)
            for label in payload
        ]

    labels = [
        label.strip()
        for label in value.split(",")
    ]
    return [
        label
        for label in labels
        if label
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify room type from sampled RGB video frames.",
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--room-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--room-labels", default=None)
    parser.add_argument("--room-samples", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offline-model-path", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--skip-model", action="store_true")

    args = parser.parse_args()
    labels = parse_labels(args.room_labels) or DEFAULT_LABELS

    semantics = classify_room(
        args.capture_dir,
        args.output_dir,
        labels=labels,
        sample_count=args.room_samples,
        model_name=args.room_model,
        device=args.device,
        allow_download=args.allow_model_download,
        offline_model_path=args.offline_model_path,
        skip_model=args.skip_model,
    )

    print(
        json.dumps(
            {
                "room_semantics": semantics.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
