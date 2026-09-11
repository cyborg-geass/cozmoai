from __future__ import annotations

import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "single_room"
SUBMISSION = REPO / "submission"
BUNDLE = REPO / "outputs" / "cozmo_ai_submission_bundle.zip"

TRACKED_FILES = [
    REPO / ".gitignore",
    REPO / ".python-version",
    REPO / "README.md",
    REPO / "pyproject.toml",
    REPO / "uv.lock",
    REPO / "configs" / "compliance_matrix.json",
    REPO / "configs" / "capture_matrix.json",
    REPO / "docs" / "FINAL_PIPELINE_STATUS.md",
    REPO / "docs" / "SUBMISSION_CHECKLIST.md",
    REPO / "docs" / "TECHNICAL_REPORT.md",
]

CODE_DIRS = [
    REPO / "src",
    REPO / "scripts",
    REPO / "tests",
    SUBMISSION,
]

GENERATED_ARTIFACTS = [
    OUT / "final_result.json",
    OUT / "submission_manifest.json",
    OUT / "measurement_evaluation.json",
    OUT / "room_envelope.json",
    OUT / "room_envelope.png",
    OUT / "room_geometry.json",
    OUT / "room_geometry.png",
    OUT / "walls.json",
    OUT / "openings_wall5.json",
    OUT / "wall5_opening_profile.csv",
    OUT / "opening_semantics" / "semantic_openings.json",
    OUT / "opening_semantics" / "opening_semantics_overview.png",
    OUT / "pointcloud_production.ply",
    OUT / "floor_plane.ply",
    OUT / "room_semantics" / "room_semantics.json",
    OUT / "room_semantics" / "contact_sheet.png",
]

SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
}


class SubmissionPackagingError(RuntimeError):
    pass


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SubmissionPackagingError(f"Required file is missing: {path}")


def archive_name(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def iter_code_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.is_file():
            yield path


def load_manifest() -> dict:
    manifest_path = OUT / "submission_manifest.json"
    require_file(manifest_path)
    return json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )


def validate_manifest(manifest: dict) -> None:
    outputs = manifest.get(
        "validated_outputs",
        [],
    )
    artifacts = manifest.get(
        "artifacts",
        {},
    )

    if len(outputs) != 15:
        raise SubmissionPackagingError(
            f"Expected 15 manifest outputs, found {len(outputs)}"
        )

    if len(artifacts) != 15:
        raise SubmissionPackagingError(
            f"Expected 15 manifest artifacts, found {len(artifacts)}"
        )


def main() -> int:
    manifest = load_manifest()
    validate_manifest(manifest)

    files = []
    for path in TRACKED_FILES + GENERATED_ARTIFACTS:
        require_file(path)
        files.append(path)

    for directory in CODE_DIRS:
        if not directory.is_dir():
            raise SubmissionPackagingError(f"Code directory is missing: {directory}")
        files.extend(iter_code_files(directory))

    unique_files = sorted(
        set(files),
        key=lambda path: archive_name(path),
    )

    BUNDLE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        BUNDLE,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in unique_files:
            archive.write(
                path,
                archive_name(path),
            )

    print(f"Wrote {BUNDLE}")
    print(f"Files: {len(unique_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
