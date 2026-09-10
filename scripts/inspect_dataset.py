from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def format_bytes(size: int) -> str:
    """Convert bytes to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def count_files(path: Path) -> int:
    """Count all files recursively under a directory."""
    return sum(1 for p in path.rglob("*") if p.is_file())


def directory_size(path: Path) -> int:
    """Calculate total size of files under a directory."""
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def print_separator(char="=", width=80):
    print(char * width)


# ============================================================
# CSV inspection
# ============================================================

def inspect_csv(path: Path, max_rows: int = 5):
    print_separator("-")
    print(f"CSV: {path.name}")
    print(f"Path: {path}")

    try:
        df = pd.read_csv(path)

        print(f"Shape       : {df.shape}")
        print(f"Columns     : {list(df.columns)}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nFirst rows:")
        print(df.head(max_rows).to_string(index=False))

        print("\nMissing values:")
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("None")
        else:
            print(missing.to_string())

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:
            print("\nNumeric summary:")
            print(
                df[numeric_cols]
                .describe()
                .transpose()
                .to_string()
            )

    except Exception as e:
        print(f"ERROR reading CSV: {e}")


# ============================================================
# Video inspection
# ============================================================

def inspect_video(path: Path):
    print_separator("-")
    print(f"VIDEO: {path.name}")
    print(f"Path: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else None

    print(f"Frame count : {frame_count}")
    print(f"FPS         : {fps}")
    print(f"Resolution  : {width} x {height}")

    if duration is not None:
        print(f"Duration    : {duration:.2f} seconds")

    # Try reading one frame
    ret, frame = cap.read()

    if ret:
        print(f"Decoded frame shape: {frame.shape}")
        print(f"Decoded frame dtype: {frame.dtype}")
    else:
        print("WARNING: Could not decode first frame.")

    cap.release()


# ============================================================
# Image / depth inspection
# ============================================================

def inspect_image(path: Path):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Could not decode image: {path}")
            return

        print(f"  {path.name}")
        print(f"    shape : {img.shape}")
        print(f"    dtype : {img.dtype}")

        if np.issubdtype(img.dtype, np.number):
            print(f"    min   : {np.min(img)}")
            print(f"    max   : {np.max(img)}")

            if np.issubdtype(img.dtype, np.floating):
                print(f"    mean  : {np.mean(img):.6f}")
            else:
                print(f"    mean  : {np.mean(img):.3f}")

    except Exception as e:
        print(f"ERROR inspecting {path}: {e}")


def inspect_sensor_directory(path: Path, max_examples: int = 5):
    print_separator("-")
    print(f"DIRECTORY: {path.name}")
    print(f"Path: {path}")

    files = [
        p for p in path.rglob("*")
        if p.is_file()
    ]

    print(f"File count : {len(files)}")
    print(f"Total size : {format_bytes(directory_size(path))}")

    extensions = Counter(
        p.suffix.lower() if p.suffix else "<no extension>"
        for p in files
    )

    print("\nExtensions:")
    for ext, count in extensions.most_common():
        print(f"  {ext:10s} : {count}")

    print("\nExample files:")

    for file in files[:max_examples]:
        print(f"  {file.relative_to(path)}")

    # Inspect image files if possible
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    image_files = [
        p for p in files
        if p.suffix.lower() in image_extensions
    ]

    if image_files:
        print("\nExample image metadata:")

        for file in image_files[:max_examples]:
            inspect_image(file)


# ============================================================
# Capture inspection
# ============================================================

def inspect_capture(capture_path: Path):
    print("\n")
    print_separator("=")
    print(f"CAPTURE: {capture_path.name}")
    print(f"Path: {capture_path}")
    print_separator("=")

    # --------------------------------------------------------
    # Overall capture statistics
    # --------------------------------------------------------

    files = [
        p for p in capture_path.rglob("*")
        if p.is_file()
    ]

    print(f"Total files : {len(files)}")
    print(f"Total size  : {format_bytes(directory_size(capture_path))}")

    # --------------------------------------------------------
    # File extension summary
    # --------------------------------------------------------

    extensions = Counter(
        p.suffix.lower() if p.suffix else "<no extension>"
        for p in files
    )

    print("\nFile extensions:")

    for ext, count in extensions.most_common():
        print(f"  {ext:10s} : {count}")

    # --------------------------------------------------------
    # CSV files
    # --------------------------------------------------------

    csv_files = [
        p for p in files
        if p.suffix.lower() == ".csv"
    ]

    if csv_files:
        print("\nCSV FILES")

        for csv_file in csv_files:
            inspect_csv(csv_file)

    # --------------------------------------------------------
    # Video files
    # --------------------------------------------------------

    video_files = [
        p for p in files
        if p.suffix.lower() in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
        }
    ]

    if video_files:
        print("\nVIDEO FILES")

        for video_file in video_files:
            inspect_video(video_file)

    # --------------------------------------------------------
    # Important directories
    # --------------------------------------------------------

    for dirname in [
        "depth",
        "confidence",
    ]:
        directory = capture_path / dirname

        if directory.exists() and directory.is_dir():
            inspect_sensor_directory(directory)


# ============================================================
# Dataset inspection
# ============================================================

def inspect_dataset(root: Path):
    print_separator("=")
    print("COZMO AI DATASET INSPECTION")
    print_separator("=")

    print(f"Dataset root: {root.resolve()}")

    if not root.exists():
        print("\nERROR: Dataset path does not exist.")
        return

    if not root.is_dir():
        print("\nERROR: Dataset path is not a directory.")
        return

    # --------------------------------------------------------
    # Dataset-level statistics
    # --------------------------------------------------------

    all_files = [
        p for p in root.rglob("*")
        if p.is_file()
    ]

    print(f"\nTotal files : {len(all_files)}")
    print(f"Total size  : {format_bytes(directory_size(root))}")

    extensions = Counter(
        p.suffix.lower() if p.suffix else "<no extension>"
        for p in all_files
    )

    print("\nDataset extensions:")

    for ext, count in extensions.most_common():
        print(f"  {ext:10s} : {count}")

    # --------------------------------------------------------
    # Top-level directories
    # --------------------------------------------------------

    top_level_dirs = [
        p for p in root.iterdir()
        if p.is_dir()
    ]

    print("\nTop-level directories:")

    for directory in sorted(top_level_dirs):
        print(
            f"  {directory.name:35s}"
            f" files={count_files(directory):6d}"
            f" size={format_bytes(directory_size(directory))}"
        )

    # --------------------------------------------------------
    # Identify captures
    #
    # A capture is currently assumed to be a directory
    # containing one or more of:
    #
    # rgb.mp4
    # camera_matrix.csv
    # imu.csv
    # odometry.csv
    # --------------------------------------------------------

    captures = []

    for directory in root.rglob("*"):

        if not directory.is_dir():
            continue

        filenames = {
            p.name.lower()
            for p in directory.iterdir()
            if p.is_file()
        }

        marker_files = {
            "rgb.mp4",
            "camera_matrix.csv",
            "imu.csv",
            "odometry.csv",
        }

        if filenames.intersection(marker_files):
            captures.append(directory)

    # Remove nested duplicates if necessary
    captures = sorted(set(captures))

    print("\n")
    print_separator("=")
    print(f"CAPTURES FOUND: {len(captures)}")
    print_separator("=")

    for capture in captures:
        inspect_capture(capture)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print_separator("=")
    print("SUMMARY")
    print_separator("=")

    print(f"Dataset       : {root.name}")
    print(f"Total files   : {len(all_files)}")
    print(f"Total size    : {format_bytes(directory_size(root))}")
    print(f"Captures      : {len(captures)}")

    if captures:
        print("\nCapture paths:")

        for capture in captures:
            print(f"  {capture.relative_to(root)}")

    print_separator("=")
    print("INSPECTION COMPLETE")
    print_separator("=")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspect the Cozmo AI case-study dataset."
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the extracted dataset root."
    )

    args = parser.parse_args()

    inspect_dataset(args.dataset)


if __name__ == "__main__":
    main()
