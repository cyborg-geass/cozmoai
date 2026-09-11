import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

DISTANCE_THRESHOLD = 0.03
RANSAC_N = 3
NUM_ITERATIONS = 2000

MIN_PLANE_POINTS = 1000

MAX_PLANES = 10


# ============================================================
# Helpers
# ============================================================

def plane_normal_angle_to_axis(normal, axis):
    """
    Return the smallest angle between a plane normal
    and an axis, in degrees.
    """

    normal = normal / np.linalg.norm(normal)

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    cosine = abs(np.dot(normal, axis))

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(
        np.arccos(cosine)
    )


def describe_plane(
    plane_model,
    points,
):

    a, b, c, d = plane_model

    normal = np.array(
        [a, b, c],
        dtype=np.float64,
    )

    normal /= np.linalg.norm(normal)

    centroid = points.mean(axis=0)

    # Distance of centroid from origin along plane normal.
    signed_offset = np.dot(
        normal,
        centroid,
    )

    angles = {
        "X": plane_normal_angle_to_axis(
            normal,
            [1, 0, 0],
        ),
        "Y": plane_normal_angle_to_axis(
            normal,
            [0, 1, 0],
        ),
        "Z": plane_normal_angle_to_axis(
            normal,
            [0, 0, 1],
        ),
    }

    return (
        normal,
        signed_offset,
        angles,
        centroid,
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\detect_planes.py "
            "outputs\\single_room\\pointcloud_production.ply"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    print("=" * 70)
    print("DOMINANT PLANE DETECTION")
    print("=" * 70)

    print(
        f"\nInput: {input_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    cloud = o3d.io.read_point_cloud(
        str(input_path)
    )

    if cloud.is_empty():

        raise RuntimeError(
            "Point cloud is empty."
        )

    print(
        f"Input points: "
        f"{len(cloud.points):,}"
    )

    # --------------------------------------------------------
    # Work on a copy
    # --------------------------------------------------------

    remaining = cloud

    detected_planes = []

    # --------------------------------------------------------
    # Iterative RANSAC
    # --------------------------------------------------------

    for plane_index in range(
        MAX_PLANES
    ):

        if len(remaining.points) < MIN_PLANE_POINTS:

            break

        plane_model, inliers = (
            remaining.segment_plane(
                distance_threshold=DISTANCE_THRESHOLD,
                ransac_n=RANSAC_N,
                num_iterations=NUM_ITERATIONS,
            )
        )

        if len(inliers) < MIN_PLANE_POINTS:

            break

        plane_points = (
            np.asarray(
                remaining.points
            )[inliers]
        )

        (
            normal,
            offset,
            angles,
            centroid,
        ) = describe_plane(
            plane_model,
            plane_points,
        )

        detected_planes.append(
            (
                plane_index + 1,
                plane_model,
                len(inliers),
                normal,
                offset,
                angles,
                centroid,
            )
        )

        # Remove plane.
        remaining = (
            remaining.select_by_index(
                inliers,
                invert=True,
            )
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DETECTED PLANES")
    print("=" * 70)

    for (
        index,
        plane,
        count,
        normal,
        offset,
        angles,
        centroid,
    ) in detected_planes:

        a, b, c, d = plane

        print(
            f"\nPlane {index}"
        )

        print(
            f"  Equation:"
            f" {a:.5f}x "
            f"+ {b:.5f}y "
            f"+ {c:.5f}z "
            f"+ {d:.5f} = 0"
        )

        print(
            f"  Points: {count:,}"
        )

        print(
            f"  Normal: "
            f"[{normal[0]:.5f}, "
            f"{normal[1]:.5f}, "
            f"{normal[2]:.5f}]"
        )

        print(
            f"  Centroid: "
            f"[{centroid[0]:.3f}, "
            f"{centroid[1]:.3f}, "
            f"{centroid[2]:.3f}]"
        )

        print(
            f"  Offset from origin: "
            f"{offset:.3f} m"
        )

        print(
            f"  Normal angle to X: "
            f"{angles['X']:.2f}°"
        )

        print(
            f"  Normal angle to Y: "
            f"{angles['Y']:.2f}°"
        )

        print(
            f"  Normal angle to Z: "
            f"{angles['Z']:.2f}°"
        )

    # --------------------------------------------------------
    # Remaining
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nPlanes detected: "
        f"{len(detected_planes)}"
    )

    print(
        f"Remaining points: "
        f"{len(remaining.points):,}"
    )

    # --------------------------------------------------------
    # Save remaining cloud
    # --------------------------------------------------------

    output_path = (
        input_path.parent
        / "pointcloud_without_dominant_planes.ply"
    )

    o3d.io.write_point_cloud(
        str(output_path),
        remaining,
    )

    print(
        f"\nRemaining cloud saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()
