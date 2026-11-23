import numpy as np
import matplotlib.pyplot as plt

from src.detection import ConeDetection
from src.landmark_mapping import LandmarkMapper
from src.simulation import (
    generate_cones,
    simulate_vehicle_movement,
    simulate_cone_detections
)


def main():
    np.random.seed(42)

    blue_cones, yellow_cones = generate_cones()
    actual_cones = np.vstack((blue_cones, yellow_cones))

    vehicle_poses = simulate_vehicle_movement()
    mapper = LandmarkMapper(association_threshold=1.5)

    for step, pose in enumerate(vehicle_poses):
        noisy_positions = simulate_cone_detections(
            actual_cones,
            pose,
            max_range=20.0,
            fov_degrees=120,
            noise_std=0.3
        )

        detections = [
            ConeDetection(
                position=position,
                covariance=np.eye(2) * 0.09,
                source="simulated",
                timestamp=float(step),
                cone_class=None
            )
            for position in noisy_positions
        ]

        mapper.update(detections, pose)

    landmark_map = mapper.get_map()

    print(f"Actual cones: {len(actual_cones)}")
    print(f"Mapped landmarks: {len(landmark_map)}")

    plt.figure(figsize=(12, 6))

    plt.scatter(
        blue_cones[:, 0],
        blue_cones[:, 1],
        color="blue",
        label="Actual Blue Cones"
    )

    plt.scatter(
        yellow_cones[:, 0],
        yellow_cones[:, 1],
        color="gold",
        label="Actual Yellow Cones"
    )

    if landmark_map:
        estimated_positions = np.array([
            landmark["position"]
            for landmark in landmark_map.values()
        ])

        plt.scatter(
            estimated_positions[:, 0],
            estimated_positions[:, 1],
            color="red",
            marker="x",
            s=70,
            label="Mapped Landmarks"
        )

    plt.plot(
        vehicle_poses[:, 0],
        vehicle_poses[:, 1],
        color="green",
        label="Vehicle Path"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Kalman Filter Cone Landmark Mapping")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()