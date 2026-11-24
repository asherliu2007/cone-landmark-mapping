import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.fsoco_loader import (
    load_fsoco_image,
    extract_cone_detections,
    draw_bounding_boxes
)

from src.camera_projection import (
    CameraIntrinsics,
    project_detections
)


def main():
    image, annotation = load_fsoco_image()

    width, height = image.size

    # Assumed camera horizontal field of view
    horizontal_fov = np.deg2rad(90)

    focal_length = (
        width / (2 * np.tan(horizontal_fov / 2))
    )

    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width / 2,
        cy=height / 2
    )

    raw_detections = extract_cone_detections(
        annotation
    )

    detections = project_detections(
        raw_detections,
        intrinsics
    )

    print(f"Image size: {width} x {height}")
    print(f"Assumed focal length: {focal_length:.2f}")
    print(f"Raw detections: {len(raw_detections)}")
    print(f"Projected detections: {len(detections)}")

    print("\nEstimated cone positions:")

    for i, detection in enumerate(detections):
        print(
            f"Cone {i + 1}: "
            f"{detection.cone_class}, "
            f"Position: "
            f"{np.round(detection.position, 3)}"
        )

    # Display original image with bounding boxes
    annotated_image = draw_bounding_boxes(
        image, annotation
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(annotated_image)
    plt.title("FSOCO Cone Detections")
    plt.axis("off")

    # Display estimated cone positions
    colors = {
        "blue_cone": "blue",
        "yellow_cone": "gold",
        "orange_cone": "orange",
        "large_orange_cone": "red"
    }

    plt.figure(figsize=(10, 8))

    for cone_class, color in colors.items():

        positions = np.array([
            detection.position
            for detection in detections
            if detection.cone_class == cone_class
        ])

        if len(positions) == 0:
            continue

        plt.scatter(
            -positions[:, 1],
            positions[:, 0],
            color=color,
            label=cone_class
        )

    ax = plt.gca()

    for detection in detections:

        # Convert covariance to plotting coordinates:
        # horizontal = right, vertical = forward
        T = np.array([
            [0, -1],
            [1, 0]
        ])

        plot_covariance = (
            T @ detection.covariance @ T.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            plot_covariance
        )

        order = eigenvalues.argsort()[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0]
        ))

        # 95% confidence ellipse for 2 dimensions
        scale = np.sqrt(5.991)

        width, height = (
            2 * scale * np.sqrt(eigenvalues)
        )

        ellipse = Ellipse(
            xy=(
                -detection.position[1],
                detection.position[0]
            ),
            width=width,
            height=height,
            angle=angle,
            edgecolor=colors.get(
                detection.cone_class, "gray"
            ),
            facecolor="none",
            alpha=0.5,
            linewidth=1
        )

        ax.add_patch(ellipse)

    plt.scatter(
        0, 0,
        color="green",
        marker="^",
        s=150,
        label="Vehicle"
    )

    plt.xlabel("Lateral Position (m, right positive)")
    plt.ylabel("Forward Position (m)")
    plt.title("Estimated FSOCO Cone Positions")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()