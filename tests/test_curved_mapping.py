import numpy as np
import matplotlib.pyplot as plt

from src.detection import ConeDetection
from src.landmark_mapping import LandmarkMapper
from src.evaluation import evaluate_map

from src.simulation import (
    generate_curved_track,
    simulate_curved_movement,
    simulate_cone_detections
)


def main():
    np.random.seed(42)

    blue_cones, yellow_cones = generate_curved_track()

    actual_cones = np.vstack((
        blue_cones,
        yellow_cones
    ))

    vehicle_poses = simulate_curved_movement()

    mapper = LandmarkMapper()

    noise_std = 0.3

    for step, pose in enumerate(vehicle_poses):

        noisy_positions = simulate_cone_detections(
            actual_cones,
            pose,
            max_range=20.0,
            fov_degrees=120,
            noise_std=noise_std
        )

        detections = [
            ConeDetection(
                position=position,
                covariance=np.eye(2) * noise_std ** 2,
                source="simulated",
                timestamp=float(step),
                cone_class=None
            )
            for position in noisy_positions
        ]

        mapper.update(detections, pose)

    landmark_map = mapper.get_map()

    estimated_positions = np.array([
        landmark["position"]
        for landmark in landmark_map.values()
    ]).reshape(-1, 2)

    # All cones should be observable during a full lap
    metrics = evaluate_map(
        actual_cones,
        estimated_positions
    )

    print("\nCurved Track Mapping Evaluation")
    print("--------------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    plt.figure(figsize=(10, 10))

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

    plt.scatter(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        color="red",
        marker="x",
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
    plt.title("Curved Track Landmark Mapping")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()