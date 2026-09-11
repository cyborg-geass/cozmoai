"""Final orchestration for the validated single-room pipeline.

Run from repository root:
    uv run python scripts/finalize_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO.parent / "cozmo-dataset" / "raw_dataset"
SINGLE_ROOM = DATASET / "single_room" / "c00a170fe1"
OUT = REPO / "outputs" / "single_room"


def run(label: str, args: list[str]) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(">", " ".join(args))
    subprocess.run(args, cwd=REPO, check=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    steps = [
        ("1/7 Build production point cloud", [
            "uv", "run", "python", "scripts/build_pointcloud.py",
            str(SINGLE_ROOM), str(OUT / "pointcloud_production.ply"),
        ]),
        ("2/7 Detect floor", [
            "uv", "run", "python", "scripts/detect_floor.py",
            str(OUT / "pointcloud_production.ply"),
        ]),
        ("3/7 Detect walls", [
            "uv", "run", "python", "scripts/detect_walls.py",
            str(OUT / "pointcloud_production.ply"),
        ]),
        ("4/7 Build room geometry", [
            "uv", "run", "python", "scripts/build_room_geometry.py",
            str(OUT / "walls.json"),
        ]),
        ("5/7 Select room envelope", [
            "uv", "run", "python", "scripts/select_room_envelope.py",
            str(OUT / "walls.json"), str(SINGLE_ROOM),
        ]),
        ("6/7 Evaluate measurements", [
            "uv", "run", "python", "scripts/evaluate_measurements.py",
            str(OUT),
        ]),
        ("7/7 Detect openings", [
            "uv", "run", "python", "scripts/detect_openings.py",
            str(OUT / "pointcloud_production.ply"),
            str(OUT / "room_geometry.json"),
            str(OUT / "walls.json"),
        ]),
    ]

    for label, args in steps:
        run(label, args)

    artifacts = [
        "pointcloud_production.ply",
        "floor_plane.ply",
        "walls.json",
        "room_geometry.json",
        "room_envelope.json",
        "room_envelope.png",
        "measurement_evaluation.json",
        "openings_wall5.json",
    ]

    manifest = {
        "pipeline": "cozmo_ai_single_room_final",
        "primary_capture": str(SINGLE_ROOM),
        "validated_outputs": [],
        "not_evaluated": [
            "photo-tier registration error <= 1.0 px",
            "photo-tier drift <= 0.5% of diagonal",
            "3+ room multi-room benchmark",
            "two-class damage benchmark",
            "laser/tape ground truth",
            "repeat-room reproducibility",
        ],
        "ceiling": {
            "status": "not_demonstrated",
            "provisional_height_m": 3.06,
            "reason": (
                "Available ceiling evidence does not support the required "
                "1.5 cm gate."
            ),
        },
    }

    for name in artifacts:
        p = OUT / name
        if p.exists():
            manifest["validated_outputs"].append(str(p.relative_to(REPO)))

    (OUT / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nFinalization complete.")
    print(f"Outputs: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
