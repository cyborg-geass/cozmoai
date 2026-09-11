"""Final orchestration for the validated single-room pipeline.

Run from repository root:
    uv run python scripts/finalize_pipeline.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO.parent / "cozmo-dataset" / "raw_dataset"
SINGLE_ROOM = DATASET / "single_room" / "c00a170fe1"
OUT = REPO / "outputs" / "single_room"
SUBMISSION_EVIDENCE = REPO / "submission" / "evidence" / "single_room"

MIN_POINTCLOUD_BYTES = 1024
MIN_PNG_BYTES = 1024
MIN_FLOOR_POINTS = 20_000
MIN_WALLS = 4
MIN_TRAJECTORY_INSIDE_RATIO = 0.50

REQUIRED_CAPTURE_FILES = [
    "camera_matrix.csv",
    "odometry.csv",
    "imu.csv",
    "rgb.mp4",
]

REQUIRED_CAPTURE_DIRS = [
    "depth",
    "confidence",
]

GENERATED_OUTPUTS = [
    "pointcloud_production.ply",
    "floor_plane.ply",
    "walls.json",
    "room_geometry.json",
    "room_geometry.png",
    "room_envelope.json",
    "room_envelope.png",
    "measurement_evaluation.json",
    "openings_wall5.json",
    "wall5_opening_profile.csv",
    "submission_manifest.json",
]

SUBMISSION_EVIDENCE_FILES = [
    "submission_manifest.json",
    "measurement_evaluation.json",
    "room_envelope.json",
    "room_envelope.png",
    "room_geometry.json",
    "room_geometry.png",
    "walls.json",
    "openings_wall5.json",
    "wall5_opening_profile.csv",
]

OUTPUT_PATTERNS = [
    "wall_plane_*.ply",
]


class PipelineValidationError(RuntimeError):
    pass


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO))


def load_json(path: Path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise PipelineValidationError(
            f"Invalid JSON artifact: {path}"
        ) from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineValidationError(message)


def validate_file(path: Path, min_bytes: int = 1) -> dict:
    require(
        path.exists(),
        f"Required artifact is missing: {path}",
    )

    require(
        path.is_file(),
        f"Required artifact is not a file: {path}",
    )

    size = path.stat().st_size

    require(
        size >= min_bytes,
        f"Artifact is unexpectedly small: {path} ({size} bytes)",
    )

    return {
        "path": relative_to_repo(path),
        "status": "generated",
        "bytes": size,
    }


def validate_capture(capture_dir: Path) -> dict:
    require(
        capture_dir.exists(),
        f"Capture directory does not exist: {capture_dir}",
    )

    require(
        capture_dir.is_dir(),
        f"Capture path is not a directory: {capture_dir}",
    )

    files = {}

    for name in REQUIRED_CAPTURE_FILES:
        path = capture_dir / name
        require(
            path.exists(),
            f"Required capture file is missing: {path}",
        )
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
        }

    directories = {}

    for name in REQUIRED_CAPTURE_DIRS:
        path = capture_dir / name
        require(
            path.is_dir(),
            f"Required capture directory is missing: {path}",
        )

        png_count = len(
            list(path.glob("*.png"))
        )

        require(
            png_count > 0,
            f"Capture directory contains no PNG frames: {path}",
        )

        directories[name] = {
            "path": str(path),
            "png_count": png_count,
        }

    depth_count = directories["depth"]["png_count"]
    confidence_count = directories["confidence"]["png_count"]

    require(
        depth_count == confidence_count,
        "Depth and confidence frame counts differ: "
        f"{depth_count} vs {confidence_count}",
    )

    return {
        "path": str(capture_dir),
        "status": "validated",
        "files": files,
        "directories": directories,
    }


def validate_pointcloud(path: Path) -> dict:
    return validate_file(
        path,
        MIN_POINTCLOUD_BYTES,
    )


def validate_png(path: Path) -> dict:
    return validate_file(
        path,
        MIN_PNG_BYTES,
    )


def validate_walls(path: Path) -> dict:
    artifact = validate_file(path)
    data = load_json(path)

    walls = data.get("walls", [])

    require(
        isinstance(walls, list),
        f"walls.json has no wall list: {path}",
    )

    require(
        len(walls) >= MIN_WALLS,
        f"Expected at least {MIN_WALLS} walls, found {len(walls)}",
    )

    for wall in walls:
        require(
            "wall_id" in wall,
            "Wall record is missing wall_id.",
        )

        require(
            len(wall.get("plane", [])) == 4,
            f"Wall {wall.get('wall_id')} has invalid plane.",
        )

        require(
            wall.get("saved_points", 0) > 0,
            f"Wall {wall.get('wall_id')} has no saved support points.",
        )

    artifact.update(
        {
            "status": "validated",
            "wall_count": len(walls),
        }
    )

    return artifact


def validate_room_geometry(path: Path) -> dict:
    artifact = validate_file(path)
    data = load_json(path)

    walls = data.get("walls", [])
    pairs = data.get("parallel_wall_pairs", [])

    require(
        len(walls) >= MIN_WALLS,
        f"room_geometry.json has fewer than {MIN_WALLS} walls.",
    )

    require(
        len(pairs) >= 2,
        "room_geometry.json does not contain enough parallel wall pairs.",
    )

    artifact.update(
        {
            "status": "validated",
            "wall_count": len(walls),
            "parallel_pair_count": len(pairs),
        }
    )

    return artifact


def validate_room_envelope(path: Path) -> dict:
    artifact = validate_file(path)
    data = load_json(path)

    dimensions = data.get(
        "dimensions_m",
        {},
    )

    area = data.get("area_m2")
    trajectory_ratio = data.get(
        "trajectory_inside_ratio"
    )
    wall_pairs = data.get(
        "selected_wall_pairs",
        [],
    )

    require(
        dimensions.get("dimension_a", 0) > 0,
        "Room envelope dimension_a is missing or invalid.",
    )

    require(
        dimensions.get("dimension_b", 0) > 0,
        "Room envelope dimension_b is missing or invalid.",
    )

    require(
        area is not None and area > 0,
        "Room envelope area is missing or invalid.",
    )

    require(
        trajectory_ratio is not None
        and trajectory_ratio >= MIN_TRAJECTORY_INSIDE_RATIO,
        "Room envelope trajectory containment is too low.",
    )

    require(
        len(wall_pairs) == 2,
        "Room envelope must select exactly two wall pairs.",
    )

    artifact.update(
        {
            "status": "validated",
            "dimensions_m": dimensions,
            "area_m2": area,
            "trajectory_inside_ratio": trajectory_ratio,
            "selected_wall_pairs": wall_pairs,
        }
    )

    return artifact


def validate_measurements(path: Path) -> dict:
    artifact = validate_file(path)
    data = load_json(path)

    floor = data.get(
        "floor",
        {},
    )
    room = data.get(
        "room",
        {},
    )
    dimensions = room.get(
        "dimensions",
        {},
    )
    area = room.get(
        "area",
        {},
    )

    require(
        floor.get("point_count", 0) >= MIN_FLOOR_POINTS,
        "Measurement artifact reports insufficient floor support.",
    )

    residuals = floor.get(
        "residuals",
        {},
    )
    require(
        residuals.get("p95_m") is not None,
        "Measurement artifact is missing floor residual metrics.",
    )

    for name in ("dimension_a", "dimension_b"):
        measurement = dimensions.get(
            name,
            {},
        )
        uncertainty = measurement.get(
            "uncertainty",
            {},
        )

        require(
            measurement.get("value_m", 0) > 0,
            f"{name} is missing or invalid.",
        )

        require(
            uncertainty.get("status") == "model_based_uncertainty",
            f"{name} does not report model-based uncertainty.",
        )

        require(
            len(uncertainty.get("ci95_m", [])) == 2,
            f"{name} does not report a 95% confidence interval.",
        )

    footprint = area.get(
        "reconstructed_footprint",
        {},
    )

    require(
        footprint.get("value_m2", 0) > 0,
        "Reconstructed footprint area is missing or invalid.",
    )

    artifact.update(
        {
            "status": "validated",
            "dimension_a_m": dimensions["dimension_a"]["value_m"],
            "dimension_b_m": dimensions["dimension_b"]["value_m"],
            "area_m2": footprint["value_m2"],
            "floor_p95_residual_m": residuals["p95_m"],
        }
    )

    return artifact


def validate_openings(path: Path) -> dict:
    artifact = validate_file(path)
    data = load_json(path)

    candidates = data.get(
        "candidates",
        [],
    )

    require(
        isinstance(candidates, list),
        "Opening artifact candidates must be a list.",
    )

    for candidate in candidates:
        if candidate.get("height_m") is None:
            require(
                candidate.get("height_status") == "not_observed",
                "Opening with null height must be marked not_observed.",
            )
        else:
            require(
                candidate.get("height_m") > 0,
                "Opening height must be positive when observed.",
            )

    artifact.update(
        {
            "status": "validated",
            "candidate_count": len(candidates),
        }
    )

    return artifact


def validate_artifact(path: Path) -> dict:
    validators = {
        "pointcloud_production.ply": validate_pointcloud,
        "floor_plane.ply": validate_pointcloud,
        "walls.json": validate_walls,
        "room_geometry.json": validate_room_geometry,
        "room_geometry.png": validate_png,
        "room_envelope.json": validate_room_envelope,
        "room_envelope.png": validate_png,
        "measurement_evaluation.json": validate_measurements,
        "openings_wall5.json": validate_openings,
    }

    validator = validators.get(path.name)

    if validator is None:
        return validate_file(path)

    return validator(path)


def clear_previous_outputs() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in GENERATED_OUTPUTS:
        path = OUT / name

        if path.exists():
            path.unlink()

    for pattern in OUTPUT_PATTERNS:
        for path in OUT.glob(pattern):
            if path.is_file():
                path.unlink()


def sync_submission_evidence() -> list[str]:
    SUBMISSION_EVIDENCE.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    for name in SUBMISSION_EVIDENCE_FILES:
        source = OUT / name
        destination = SUBMISSION_EVIDENCE / name

        validate_artifact(source)
        shutil.copy2(
            source,
            destination,
        )
        validate_artifact(destination)
        copied.append(
            relative_to_repo(destination)
        )

    return copied


def run(label: str, args: list[str]) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(">", " ".join(args))

    env = os.environ.copy()
    env.setdefault(
        "UV_CACHE_DIR",
        str(REPO / ".uv-cache"),
    )

    subprocess.run(
        args,
        cwd=REPO,
        check=True,
        env=env,
    )


def validate_step_outputs(paths: list[Path]) -> dict:
    return {
        relative_to_repo(path): validate_artifact(path)
        for path in paths
    }


def main() -> int:
    capture_summary = validate_capture(SINGLE_ROOM)

    clear_previous_outputs()

    steps = [
        {
            "label": "1/7 Build production point cloud",
            "args": [
                "uv", "run", "python", "scripts/build_pointcloud.py",
                str(SINGLE_ROOM), str(OUT / "pointcloud_production.ply"),
            ],
            "outputs": [
                OUT / "pointcloud_production.ply",
            ],
        },
        {
            "label": "2/7 Detect floor",
            "args": [
                "uv", "run", "python", "scripts/detect_floor.py",
                str(OUT / "pointcloud_production.ply"),
            ],
            "outputs": [
                OUT / "floor_plane.ply",
            ],
        },
        {
            "label": "3/7 Detect walls",
            "args": [
                "uv", "run", "python", "scripts/detect_walls.py",
                str(OUT / "pointcloud_production.ply"),
            ],
            "outputs": [
                OUT / "walls.json",
            ],
        },
        {
            "label": "4/7 Build room geometry",
            "args": [
                "uv", "run", "python", "scripts/build_room_geometry.py",
                str(OUT / "walls.json"),
            ],
            "outputs": [
                OUT / "room_geometry.json",
                OUT / "room_geometry.png",
            ],
        },
        {
            "label": "5/7 Select room envelope",
            "args": [
                "uv", "run", "python", "scripts/select_room_envelope.py",
                str(OUT / "walls.json"), str(SINGLE_ROOM),
            ],
            "outputs": [
                OUT / "room_envelope.json",
                OUT / "room_envelope.png",
            ],
        },
        {
            "label": "6/7 Evaluate measurements",
            "args": [
                "uv", "run", "python", "scripts/evaluate_measurements.py",
                str(OUT),
            ],
            "outputs": [
                OUT / "measurement_evaluation.json",
            ],
        },
        {
            "label": "7/7 Detect openings",
            "args": [
                "uv", "run", "python", "scripts/detect_openings.py",
                str(OUT / "pointcloud_production.ply"),
                str(OUT / "room_geometry.json"),
                str(OUT / "walls.json"),
            ],
            "outputs": [
                OUT / "openings_wall5.json",
                OUT / "wall5_opening_profile.csv",
            ],
        },
    ]

    stage_outputs = {}

    for step in steps:
        run(
            step["label"],
            step["args"],
        )
        stage_outputs.update(
            validate_step_outputs(
                step["outputs"]
            )
        )

    artifacts = [
        "pointcloud_production.ply",
        "floor_plane.ply",
        "walls.json",
        "room_geometry.json",
        "room_geometry.png",
        "room_envelope.json",
        "room_envelope.png",
        "measurement_evaluation.json",
        "openings_wall5.json",
        "wall5_opening_profile.csv",
    ]

    manifest = {
        "pipeline": "cozmo_ai_single_room_final",
        "primary_capture": str(SINGLE_ROOM),
        "capture": capture_summary,
        "validated_outputs": [],
        "artifacts": stage_outputs,
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
        manifest["validated_outputs"].append(
            relative_to_repo(p)
        )

    (OUT / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    validate_file(
        OUT / "submission_manifest.json"
    )

    submission_evidence = sync_submission_evidence()

    print("\nFinalization complete.")
    print(f"Outputs: {OUT}")
    print(f"GitHub evidence: {SUBMISSION_EVIDENCE}")
    print(f"Evidence files copied: {len(submission_evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
