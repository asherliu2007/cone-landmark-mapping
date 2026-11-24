import numpy as np
import matplotlib.pyplot as plt

from src.fsoco_loader import (
    load_fsoco_image,
    extract_cone_detections
)

from src.camera_projection import (
    CameraIntrinsics,
    project_detections
)

from src.landmark_mapping import LandmarkMapper


def main():

    # Load FSOCO image and annotations
    image, annotation = load_fsoco_image()

    width, height = image.size

    # Temporary camera calibration
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

    # Extract and project detections
    raw_detections = extract_cone_detections(
        annotation
    )

    detections = project_detections(
        raw_detections,
        intrinsics
    )

    # Initialize landmark mapper
    mapper = LandmarkMapper(
        min_observations=2
    )

    # Assume vehicle starts at the origin
    vehicle_pose = np.array([
        0.0, 0.0, 0.0
    ])

    # Process the camera observations
    mapper.update(detections, vehicle_pose)

    landmarks = mapper.get_map(
        confirmed_only=False
    )

    confirmed = mapper.get_map(
        confirmed_only=True
    )

    print("FSOCO Mapping Results")
    print("-------------------------")
    print(f"Raw detections: {len(raw_detections)}")
    print(f"Projected detections: {len(detections)}")
    print(f"Total landmarks: {len(landmarks)}")
    print(f"Confirmed landmarks: {len(confirmed)}")
    print(
        f"Tentative landmarks: "
        f"{len(landmarks) - len(confirmed)}"
    )

    colors = {
        "blue_cone": "blue",
        "yellow_cone": "gold",
        "orange_cone": "orange",
        "large_orange_cone": "red"
    }

    plt.figure(figsize=(10, 8))

    # Plot projected camera detections
    for detection in detections:

        position = detection.position

        plt.scatter(
            -position[1],
            position[0],
            color=colors.get(
                detection.cone_class, "gray"
            ),
            alpha=0.5
        )

    # Plot mapped landmarks
    for landmark in landmarks.values():

        position = landmark["position"]

        plt.scatter(
            -position[1],
            position[0],
            color="black",
            marker="x",
            s=70
        )

    # Plot vehicle
    plt.scatter(
        0, 0,
        color="green",
        marker="^",
        s=150,
        label="Vehicle"
    )

    plt.scatter(
        [], [],
        color="black",
        marker="x",
        label="Mapped landmarks"
    )

    plt.xlabel("Lateral Position (m)")
    plt.ylabel("Forward Position (m)")
    plt.title("FSOCO Camera Landmark Mapping")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()