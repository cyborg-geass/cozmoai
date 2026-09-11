import argparse
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path"
    )

    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(
        args.path
    )

    print(cloud)

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="Cozmo Point Cloud",
    )


if __name__ == "__main__":
    main()
