import numpy as np

from src.detection import ConeDetection
from src.landmark_mapping import LandmarkMapper


def main():
    np.random.seed(42)

    mapper = LandmarkMapper()

    # Three stationary cones in global coordinates
    true_cones = [
        (np.array([10.0, 3.0]), "blue_cone"),
        (np.array([15.0, -3.0]), "yellow_cone"),
        (np.array([20.0, 3.0]), "blue_cone")
    ]

    # Vehicle moves along the track
    for step in range(10):
        vehicle_pose = np.array([step * 1.0, 0.0, 0.0])

        detections = []

        for position, cone_class in true_cones:
            local_position = position - vehicle_pose[:2]

            noisy_position = local_position + np.random.normal(
                0, 0.3, size=2
            )

            detections.append(
                ConeDetection(
                    position=noisy_position,
                    covariance=np.eye(2) * 0.09,
                    source="simulated",
                    timestamp=float(step),
                    cone_class=cone_class
                )
            )

        mapper.update(detections, vehicle_pose)

    print(f"Mapped {len(mapper.landmarks)} landmarks:\n")

    for landmark_id, landmark in mapper.get_map().items():
        print(
            f"Landmark {landmark_id}: "
            f"{landmark['class']}, "
            f"Position: {np.round(landmark['position'], 3)}"
        )


if __name__ == "__main__":
    main()