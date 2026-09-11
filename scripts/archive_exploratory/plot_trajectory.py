from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CAPTURE = Path(
    r"..\cozmo-dataset\raw_dataset\single_room\c00a170fe1"
)

OUTPUT = Path(
    "outputs/single_room/trajectory.png"
)


def main():
    odometry = pd.read_csv(
        CAPTURE / "odometry.csv"
    )

    odometry.columns = odometry.columns.str.strip()

    x = odometry["x"].to_numpy()
    z = odometry["z"].to_numpy()

    plt.figure(figsize=(10, 8))

    plt.plot(x, z)

    plt.scatter(
        x[0],
        z[0],
        marker="o",
        label="Start",
    )

    plt.scatter(
        x[-1],
        z[-1],
        marker="x",
        label="End",
    )

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Camera Trajectory")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
