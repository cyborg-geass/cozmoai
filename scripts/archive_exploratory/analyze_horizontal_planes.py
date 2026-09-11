from pathlib import Path
import json
import numpy as np
import open3d as o3d


INPUT = Path("outputs/single_scan_with_ceiling/pointcloud_production.ply")

# These are the planes from the detector output.
PLANES = [
    (-0.00000, 1.00000, -0.00092, 1.49729),
    (0.01037, 0.99987, -0.01201, -0.81423),
    (-0.00238, 0.99999, 0.00251, -1.58137),
    (0.00132, 0.99999, -0.00421, 0.69349),
    (-0.01617, 0.99914, -0.03805, 0.49185),
    (-0.00853, 0.99971, -0.02263, 0.25624),
]

pcd = o3d.io.read_point_cloud(str(INPUT))
points = np.asarray(pcd.points)

print("=" * 70)
print("HORIZONTAL PLANE SPATIAL ANALYSIS")
print("=" * 70)
print(f"Input points: {len(points):,}")


for idx, plane in enumerate(PLANES, start=1):

    normal = np.array(plane[:3], dtype=np.float64)
    d = float(plane[3])

    normal /= np.linalg.norm(normal)

    # Signed distance to plane.
    distances = points @ normal + d

    # Tight plane membership.
    mask = np.abs(distances) < 0.02
    pts = points[mask]

    if len(pts) == 0:
        print(f"\nPlane {idx}: NO POINTS")
        continue

    centroid = pts.mean(axis=0)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    extent = maxs - mins

    print()
    print(f"Plane {idx}")
    print("-" * 50)

    print(f"Points: {len(pts):,}")

    print(
        f"Centroid: "
        f"[{centroid[0]:.3f}, "
        f"{centroid[1]:.3f}, "
        f"{centroid[2]:.3f}]"
    )

    print(
        f"X range: {mins[0]:.3f} -> {maxs[0]:.3f} "
        f"(extent {extent[0]:.3f} m)"
    )

    print(
        f"Y range: {mins[1]:.3f} -> {maxs[1]:.3f} "
        f"(extent {extent[1]:.3f} m)"
    )

    print(
        f"Z range: {mins[2]:.3f} -> {maxs[2]:.3f} "
        f"(extent {extent[2]:.3f} m)"
    )

    print(
        f"Horizontal footprint XZ: "
        f"{extent[0] * extent[2]:.3f} m²"
    )

    print(
        f"Distance residual: "
        f"median={np.median(np.abs(distances[mask])):.4f} m, "
        f"p95={np.percentile(np.abs(distances[mask]), 95):.4f} m"
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)
