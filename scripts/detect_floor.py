import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

NORMAL_Y_THRESHOLD = 0.95

MIN_FLOOR_HEIGHT_BELOW_CAMERA = 0.5
MAX_FLOOR_HEIGHT_BELOW_CAMERA = 2.5

MIN_FLOOR_POINTS = 20_000

RANSAC_DISTANCE_THRESHOLD = 0.03
RANSAC_ITERATIONS = 3000


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python scripts\\detect_floor.py "
            "outputs\\single_room\\pointcloud_production.ply"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------

    cloud = o3d.io.read_point_cloud(
        str(input_path)
    )

    if cloud.is_empty():

        raise RuntimeError(
            "Point cloud is empty."
        )

    points = np.asarray(
        cloud.points
    )

    print("=" * 70)
    print("FLOOR PLANE DETECTION")
    print("=" * 70)

    print(
        f"\nInput points: {len(points):,}"
    )

    # --------------------------------------------------------
    # Camera trajectory / reference height
    #
    # For this capture, the camera Y coordinate is close
    # to zero. We use the origin as the first diagnostic
    # reference.
    #
    # The detected floor should be substantially below it.
    # --------------------------------------------------------

    reference_camera_y = 0.0

    print(
        f"\nReference camera Y: "
        f"{reference_camera_y:.3f} m"
    )

    # --------------------------------------------------------
    # RANSAC planes
    # --------------------------------------------------------

    remaining = cloud

    candidates = []

    for iteration in range(15):

        if len(remaining.points) < MIN_FLOOR_POINTS:
            break

        # ----------------------------------------------------
        # Detect dominant plane
        # ----------------------------------------------------

        plane_model, inliers = (
            remaining.segment_plane(
                distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                ransac_n=3,
                num_iterations=RANSAC_ITERATIONS,
            )
        )

        if len(inliers) < MIN_FLOOR_POINTS:
            break

        a, b, c, d = plane_model

        normal = np.array(
            [a, b, c],
            dtype=np.float64,
        )

        norm = np.linalg.norm(
            normal
        )

        if norm == 0:
            break

        # ----------------------------------------------------
        # Normalize plane normal
        # ----------------------------------------------------

        normal /= norm

        # The Open3D plane equation is:
        #
        #     ax + by + cz + d = 0
        #
        # Make the normal point approximately +Y.
        #
        # If we flip the normal, we MUST also flip d.
        # ----------------------------------------------------

        if normal[1] < 0:

            normal = -normal
            d = -d

        # ----------------------------------------------------
        # Calculate Y coordinate of the plane
        #
        # For a nearly horizontal plane:
        #
        #     nx*x + ny*y + nz*z + d = 0
        #
        # Therefore, at the plane's reference location:
        #
        #     y = -d / ny
        # ----------------------------------------------------

        if abs(normal[1]) > 1e-8:

            plane_y = (
                -d / normal[1]
            )

        else:

            plane_y = np.nan

        # ----------------------------------------------------
        # Get plane points
        # ----------------------------------------------------

        candidate_points = np.asarray(
            remaining.points
        )[inliers]

        centroid = candidate_points.mean(
            axis=0
        )

        # ----------------------------------------------------
        # Angle between plane normal and Y axis
        # ----------------------------------------------------

        angle_to_y = np.degrees(
            np.arccos(
                np.clip(
                    abs(normal[1]),
                    0.0,
                    1.0,
                )
            )
        )

        # ----------------------------------------------------
        # Distance below camera
        # ----------------------------------------------------

        below_camera = (
            reference_camera_y
            - plane_y
        )

        # ----------------------------------------------------
        # Candidate tests
        # ----------------------------------------------------

        is_horizontal = (
            abs(normal[1])
            >= NORMAL_Y_THRESHOLD
        )

        is_below_camera = (
            MIN_FLOOR_HEIGHT_BELOW_CAMERA
            <= below_camera
            <= MAX_FLOOR_HEIGHT_BELOW_CAMERA
        )

        # ----------------------------------------------------
        # Save valid floor candidate
        # ----------------------------------------------------

        if (
            is_horizontal
            and is_below_camera
        ):

            candidates.append(
                {
                    "plane": np.array(
                        [
                            normal[0],
                            normal[1],
                            normal[2],
                            d,
                        ],
                        dtype=np.float64,
                    ),

                    "points": len(inliers),

                    "normal": normal,

                    "plane_y": plane_y,

                    # IMPORTANT:
                    # This was missing before and caused
                    # the KeyError during sorting.
                    "below_camera": below_camera,

                    "angle_to_y": angle_to_y,

                    "centroid": centroid,

                    "inliers": inliers,
                }
            )

        # ----------------------------------------------------
        # Remove detected plane and continue searching
        # ----------------------------------------------------

        remaining = remaining.select_by_index(
            inliers,
            invert=True,
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FLOOR CANDIDATES"
    )

    print(
        "=" * 70
    )

    if not candidates:

        print(
            "\nNo floor candidates found."
        )

        return

    # --------------------------------------------------------
    # Sort candidates
    #
    # 1. Prefer planes with more points.
    # 2. If tied, prefer the plane closer to the camera.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            -x["points"],
            x["below_camera"],
        )
    )

    # --------------------------------------------------------
    # Print candidates
    # --------------------------------------------------------

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate {i}"
        )

        print(
            f"  Points: "
            f"{candidate['points']:,}"
        )

        print(
            f"  Y: "
            f"{candidate['plane_y']:.4f} m"
        )

        print(
            f"  Height below camera: "
            f"{candidate['below_camera']:.4f} m"
        )

        print(
            f"  Angle to Y: "
            f"{candidate['angle_to_y']:.3f}°"
        )

        print(
            f"  Normal: "
            f"{candidate['normal']}"
        )

        print(
            f"  Centroid: "
            f"{candidate['centroid']}"
        )

    # --------------------------------------------------------
    # Select best floor
    # --------------------------------------------------------

    floor = candidates[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED FLOOR"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPoints: "
        f"{floor['points']:,}"
    )

    print(
        f"Floor Y: "
        f"{floor['plane_y']:.4f} m"
    )

    print(
        f"Height below camera: "
        f"{floor['below_camera']:.4f} m"
    )

    print(
        f"Normal: "
        f"{floor['normal']}"
    )

    print(
        f"Angle to Y: "
        f"{floor['angle_to_y']:.3f}°"
    )

    print(
        f"Centroid: "
        f"{floor['centroid']}"
    )

    # --------------------------------------------------------
    # Save floor points
    # --------------------------------------------------------

    floor_cloud = cloud.select_by_index(
        floor["inliers"]
    )

    output_path = (
        input_path.parent
        / "floor_plane.ply"
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        floor_cloud,
    )

    if not success:

        raise RuntimeError(
            f"Failed to save floor plane to "
            f"{output_path}"
        )

    print(
        f"\nSaved floor plane:"
        f"\n{output_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
