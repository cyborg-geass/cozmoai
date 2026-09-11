from __future__ import annotations

import argparse
import json
from pathlib import Path

from cozmo_ai.perception.opening_semantics import classify_openings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse geometric opening candidates with optional door/window "
            "semantics."
        ),
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("opening_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--opening-model", default="google/owlvit-base-patch32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--skip-model", action="store_true")

    args = parser.parse_args()

    result = classify_openings(
        capture_dir=args.capture_dir,
        opening_json=args.opening_json,
        output_dir=args.output_dir,
        enable_model=not args.skip_model,
        model_name=args.opening_model,
        device=args.device,
        allow_download=args.allow_model_download,
        min_iou=args.min_iou,
        min_confidence=args.min_confidence,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
