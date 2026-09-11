import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

ROBUST_SIGMA_FACTOR = 1.4826
CONFIDENCE_Z = 1.96

MIN_POINTS_FOR_STATS = 100


# ============================================================
# Basic geometry
# ============================================================

def normalize_plane(plane):
    """
    Normalize:

        ax + by + cz + d = 0

    so that ||[a,b,c]|| = 1.
    """

    plane = np.asarray(
        plane,
        dtype=np.float64,
    )

    if plane.shape != (4,):
        raise ValueError(
            f"Expected plane with 4 values, "
            f"got shape {plane.shape}"
        )

    norm = np.linalg.norm(
        plane[:3]
    )

    if norm < 1e-12:
        raise ValueError(
            "Plane normal has near-zero magnitude."
        )

    return plane / norm


def point_plane_distances(
    points,
    plane,
):
    """
    Perpendicular distances from points
    to a plane.
    """

    plane = normalize_plane(
        plane
    )

    distances = np.abs(
        points @ plane[:3]
        + plane[3]
    )

    return distances


# ============================================================
# Statistics
# ============================================================

def robust_statistics(values):
    """
    Compute robust statistics of plane residuals.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "count": 0,
            "median_m": None,
            "p95_m": None,
            "rmse_m": None,
            "mad_m": None,
            "robust_sigma_m": None,
        }

    median = float(
        np.median(values)
    )

    p95 = float(
        np.percentile(
            values,
            95,
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    robust_sigma = (
        ROBUST_SIGMA_FACTOR * mad
    )

    return {
        "count": int(
            len(values)
        ),
        "median_m": median,
        "p95_m": p95,
        "rmse_m": rmse,
        "mad_m": mad,
        "robust_sigma_m": float(
            robust_sigma
        ),
    }


def plane_uncertainty(
    statistics
):
    """
    Estimate the positional uncertainty
    of a plane from its residual distribution.

    IMPORTANT:

    This is a model-based uncertainty estimate.

    It is NOT a calibrated absolute accuracy
    measurement.
    """

    sigma = statistics[
        "robust_sigma_m"
    ]

    if sigma is None:
        return {
            "sigma_m": None,
            "ci95_half_width_m": None,
        }

    sigma = max(
        float(sigma),
        1e-6,
    )

    return {
        "sigma_m": sigma,
        "ci95_half_width_m": (
            CONFIDENCE_Z * sigma
        ),
    }


# ============================================================
# File loading
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_points(path):

    cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Point cloud is empty: {path}"
        )

    return np.asarray(
        cloud.points,
        dtype=np.float64,
    )


# ============================================================
# Plane evaluation
# ============================================================

def evaluate_plane(
    plane,
    cloud_path,
    label,
):

    print(
        f"\nEvaluating {label}"
    )

    points = load_points(
        cloud_path
    )

    print(
        f"  Points: {len(points):,}"
    )

    distances = point_plane_distances(
        points,
        plane,
    )

    statistics = robust_statistics(
        distances
    )

    uncertainty = plane_uncertainty(
        statistics
    )

    print(
        f"  Median residual: "
        f"{statistics['median_m']:.6f} m"
    )

    print(
        f"  P95 residual: "
        f"{statistics['p95_m']:.6f} m"
    )

    print(
        f"  RMSE: "
        f"{statistics['rmse_m']:.6f} m"
    )

    print(
        f"  Robust sigma: "
        f"{statistics['robust_sigma_m']:.6f} m"
    )

    return {
        "plane": [
            float(x)
            for x in normalize_plane(
                plane
            )
        ],
        "point_count": int(
            len(points)
        ),
        "residuals": statistics,
        "position_uncertainty": uncertainty,
    }


# ============================================================
# Wall geometry
# ============================================================

def align_plane_signs(
    plane_a,
    plane_b,
):
    """
    Make the two normals point in approximately
    the same direction.
    """

    a = normalize_plane(
        plane_a
    )

    b = normalize_plane(
        plane_b
    )

    if np.dot(
        a[:3],
        b[:3],
    ) < 0:

        b = -b

    return a, b


def wall_separation(
    wall_a,
    wall_b,
):
    """
    Distance between approximately parallel
    normalized wall planes.
    """

    a, b = align_plane_signs(
        wall_a["plane"],
        wall_b["plane"],
    )

    angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(
                    a[:3],
                    b[:3],
                ),
                -1.0,
                1.0,
            )
        )
    )

    separation = abs(
        a[3] - b[3]
    )

    return (
        float(separation),
        float(angle),
    )


def combined_sigma(
    sigma_a,
    sigma_b,
):

    if (
        sigma_a is None
        or sigma_b is None
    ):
        return None

    return float(
        np.sqrt(
            sigma_a ** 2
            + sigma_b ** 2
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "uv run python "
            "scripts\\evaluate_measurements.py "
            "outputs\\single_room"
        )

        sys.exit(1)

    output_dir = Path(
        sys.argv[1]
    )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{output_dir}"
        )

    walls_path = (
        output_dir
        / "walls.json"
    )

    room_path = (
        output_dir
        / "room_envelope.json"
    )

    floor_cloud_path = (
        output_dir
        / "floor_plane.ply"
    )

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    required = [
        walls_path,
        room_path,
        floor_cloud_path,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    walls_data = load_json(
        walls_path
    )

    room_data = load_json(
        room_path
    )

    floor_plane = np.asarray(
        walls_data["floor_plane"],
        dtype=np.float64,
    )

    walls = walls_data[
        "walls"
    ]

    # --------------------------------------------------------
    # Floor
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FLOOR"
    )

    print(
        "=" * 70
    )

    floor_result = evaluate_plane(
        floor_plane,
        floor_cloud_path,
        "floor plane",
    )

    # --------------------------------------------------------
    # Walls
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "WALLS"
    )

    print(
        "=" * 70
    )

    wall_results = {}

    for wall in walls:

        wall_id = int(
            wall["wall_id"]
        )

        wall_cloud_path = (
            output_dir
            / f"wall_plane_{wall_id}.ply"
        )

        if not wall_cloud_path.exists():

            print(
                f"\nWARNING:"
                f" missing {wall_cloud_path}"
            )

            continue

        result = evaluate_plane(
            wall["plane"],
            wall_cloud_path,
            f"wall {wall_id}",
        )

        wall_results[
            wall_id
        ] = result

    # --------------------------------------------------------
    # Room dimensions
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ROOM DIMENSIONS"
    )

    print(
        "=" * 70
    )

    selected_pairs = (
        room_data.get(
            "selected_wall_pairs",
            []
        )
    )

    dimensions = {}

    for index, pair in enumerate(
        selected_pairs,
        start=1,
    ):

        pair_walls = pair[
            "walls"
        ]

        if len(pair_walls) != 2:
            continue

        wall_a_id = int(
            pair_walls[0]
        )

        wall_b_id = int(
            pair_walls[1]
        )

        if (
            wall_a_id not in wall_results
            or wall_b_id not in wall_results
        ):
            print(
                f"\nSkipping pair "
                f"{wall_a_id}, {wall_b_id}: "
                f"missing wall evaluation."
            )
            continue

        wall_a = next(
            w for w in walls
            if int(w["wall_id"])
            == wall_a_id
        )

        wall_b = next(
            w for w in walls
            if int(w["wall_id"])
            == wall_b_id
        )

        value, angle = wall_separation(
            wall_a,
            wall_b,
        )

        sigma_a = wall_results[
            wall_a_id
        ][
            "position_uncertainty"
        ][
            "sigma_m"
        ]

        sigma_b = wall_results[
            wall_b_id
        ][
            "position_uncertainty"
        ][
            "sigma_m"
        ]

        sigma_dimension = (
            combined_sigma(
                sigma_a,
                sigma_b,
            )
        )

        if sigma_dimension is not None:

            half_width = (
                CONFIDENCE_Z
                * sigma_dimension
            )

            ci_low = (
                value
                - half_width
            )

            ci_high = (
                value
                + half_width
            )

        else:

            half_width = None
            ci_low = None
            ci_high = None

        key = (
            f"dimension_{chr(96 + index)}"
        )

        dimensions[key] = {
            "wall_pair": [
                wall_a_id,
                wall_b_id,
            ],

            "value_m": value,

            "wall_normal_angle_deg":
                angle,

            "uncertainty": {
                "sigma_m":
                    sigma_dimension,

                "ci95_m": (
                    [
                        float(ci_low),
                        float(ci_high),
                    ]
                    if ci_low is not None
                    else None
                ),

                "ci95_half_width_m": (
                    float(half_width)
                    if half_width is not None
                    else None
                ),

                "method":
                    "wall_plane_residual_propagation",

                "status":
                    "model_based_uncertainty",
            },
        }

        print(
            f"\n{key}:"
        )

        print(
            f"  Walls: "
            f"{wall_a_id} <-> {wall_b_id}"
        )

        print(
            f"  Separation: "
            f"{value:.4f} m"
        )

        print(
            f"  Normal angle: "
            f"{angle:.3f} deg"
        )

        if (
            ci_low is not None
        ):

            print(
                f"  95% CI: "
                f"[{ci_low:.4f}, "
                f"{ci_high:.4f}] m"
            )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "AREA"
    )

    print(
        "=" * 70
    )

    rectangular_area_value = None
    rectangular_area_sigma = None
    rectangular_area_ci = None

    polygon_area_value = None

    # --------------------------------------------------------
    # Rectangular reference area
    #
    # This is simply:
    #
    #     dimension_a * dimension_b
    #
    # It is NOT the actual footprint area because the
    # reconstructed room envelope is not perfectly rectangular.
    # --------------------------------------------------------

    if (
        "dimension_a" in dimensions
        and "dimension_b" in dimensions
    ):

        a = dimensions[
            "dimension_a"
        ]["value_m"]

        b = dimensions[
            "dimension_b"
        ]["value_m"]

        sigma_a = dimensions[
            "dimension_a"
        ]["uncertainty"]["sigma_m"]

        sigma_b = dimensions[
            "dimension_b"
        ]["uncertainty"]["sigma_m"]

        rectangular_area_value = a * b

        if (
            sigma_a is not None
            and sigma_b is not None
        ):

            # A = a*b
            #
            # sigma_A² =
            #     (b*sigma_a)²
            #   + (a*sigma_b)²

            rectangular_area_sigma = np.sqrt(
                (b * sigma_a) ** 2
                + (a * sigma_b) ** 2
            )

            rectangular_area_half_width = (
                CONFIDENCE_Z
                * rectangular_area_sigma
            )

            rectangular_area_ci = [
                float(
                    max(
                        0.0,
                        rectangular_area_value
                        - rectangular_area_half_width,
                    )
                ),
                float(
                    rectangular_area_value
                    + rectangular_area_half_width
                ),
            ]

        else:

            rectangular_area_half_width = None

        print(
            f"\nRectangular reference area: "
            f"{rectangular_area_value:.4f} m²"
        )

        if rectangular_area_ci is not None:

            print(
                f"95% CI: "
                f"[{rectangular_area_ci[0]:.4f}, "
                f"{rectangular_area_ci[1]:.4f}] m²"
            )

    else:

        rectangular_area_half_width = None

        print(
            "\nRectangular reference area "
            "could not be computed."
        )

    # --------------------------------------------------------
    # Actual reconstructed footprint
    #
    # The room envelope is an arbitrary quadrilateral,
    # so calculate its area using the shoelace formula.
    #
    # IMPORTANT:
    #
    # We do NOT assign a confidence interval here because
    # the current uncertainty model is based on wall-to-wall
    # dimensions, not corner covariance.
    # --------------------------------------------------------

    corners = room_data.get(
        "corners",
        []
    )

    if len(corners) >= 3:

        polygon = np.asarray(
            [
                [
                    float(corner["u"]),
                    float(corner["v"]),
                ]
                for corner in corners
            ],
            dtype=np.float64,
        )

        x = polygon[:, 0]
        y = polygon[:, 1]

        polygon_area_value = 0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                - y * np.roll(x, -1)
            )
        )

        print(
            f"\nReconstructed footprint area: "
            f"{polygon_area_value:.4f} m²"
        )

    else:

        print(
            "\nReconstructed footprint area "
            "could not be computed: "
            "fewer than 3 corners."
        )

    # --------------------------------------------------------
    # Edge lengths
    # --------------------------------------------------------

    edge_lengths = []

    if len(corners) >= 2:

        for i in range(
            len(corners)
        ):

            current = polygon[i]

            next_point = polygon[
                (i + 1) % len(polygon)
            ]

            length = np.linalg.norm(
                next_point - current
            )

            edge_lengths.append(
                float(length)
            )

        print(
            "\nEnvelope edge lengths:"
        )

        for i, length in enumerate(
            edge_lengths,
            start=1,
        ):

            print(
                f"  Edge {i}: "
                f"{length:.4f} m"
            )

    # --------------------------------------------------------
    # Final area result
    # --------------------------------------------------------

    area_result = {

        "rectangular_reference": {

            "value_m2": (
                float(
                    rectangular_area_value
                )
                if rectangular_area_value
                is not None
                else None
            ),

            "uncertainty": {

                "sigma_m2": (
                    float(
                        rectangular_area_sigma
                    )
                    if rectangular_area_sigma
                    is not None
                    else None
                ),

                "ci95_m2":
                    rectangular_area_ci,

                "ci95_half_width_m2": (
                    float(
                        rectangular_area_half_width
                    )
                    if rectangular_area_half_width
                    is not None
                    else None
                ),

                "method":
                    "first_order_area_propagation",

                "status":
                    "model_based_uncertainty",

            },

            "interpretation":
                (
                    "Reference area obtained by "
                    "multiplying the two selected "
                    "wall-to-wall dimensions. It is "
                    "not the reconstructed polygon "
                    "footprint."
                ),
        },

        "reconstructed_footprint": {

            "value_m2": (
                float(
                    polygon_area_value
                )
                if polygon_area_value
                is not None
                else None
            ),

            "geometry":
                "quadrilateral",

            "corners": [
                {
                    "u_m": float(
                        corner["u"]
                    ),
                    "v_m": float(
                        corner["v"]
                    ),
                }
                for corner in corners
            ],

            "edge_lengths_m":
                edge_lengths,

            "uncertainty": {

                "ci95_m2": None,

                "status":
                    "not_estimated",

                "reason":
                    (
                        "Corner covariance is not "
                        "currently available. The "
                        "dimension-based uncertainty "
                        "model cannot be directly "
                        "applied to the irregular "
                        "quadrilateral footprint."
                    ),
            },

            "method":
                "shoelace_formula",
        },
    }

    result = {

        "schema_version":
            "1.0",

        "source": {
            "walls":
                str(walls_path),

            "room_envelope":
                str(room_path),

            "floor_cloud":
                str(floor_cloud_path),
        },

        "floor":
            floor_result,

        "walls":
            {
                str(k): v
                for k, v
                in wall_results.items()
            },

        "room": {

            "dimensions":
                dimensions,

            "area":
                area_result,
        },
        "calibration": {

            "ground_truth_available":
                False,

            "ground_truth_type":
                None,

            "uncertainty_status":
                "not_ground_truth_calibrated",

            "note":
                (
                    "Intervals are derived from "
                    "plane residuals and first-order "
                    "uncertainty propagation. They "
                    "should not be interpreted as "
                    "demonstrated absolute accuracy."
                ),
        },
    }

    output_path = (
        output_dir
        / "measurement_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MEASUREMENT EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()
