import numpy as np
import matplotlib.pyplot as plt


def generate_cones():
    # X positions along the track
    x = np.arange(0, 50, 5)

    # Blue cones on the left
    blue_cones = np.column_stack((x, np.full(len(x), 3)))

    # Yellow cones on the right
    yellow_cones = np.column_stack((x, np.full(len(x), -3)))

    return blue_cones, yellow_cones


def plot_cones(blue_cones, yellow_cones):
    plt.figure(figsize=(10, 5))

    plt.scatter(
        blue_cones[:, 0],
        blue_cones[:, 1],
        color="blue",
        label="Blue Cones"
    )

    plt.scatter(
        yellow_cones[:, 0],
        yellow_cones[:, 1],
        color="gold",
        label="Yellow Cones"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("FSAE Driverless Cone Map")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    blue_cones, yellow_cones = generate_cones()
    plot_cones(blue_cones, yellow_cones)