import numpy as np
import matplotlib.pyplot as plt

from src.detection import ConeDetection
from src.landmark_mapping import LandmarkMapper
from src.evaluation import evaluate_map

from src.simulation import (
    generate_cones,
    simulate_vehicle_movement,
    simulate_cone_detections
)


def run_experiment(noise_std, seed):
    np.random.seed(seed)

    blue_cones, yellow_cones = generate_cones()
    actual_cones = np.vstack((blue_cones, yellow_cones))

    observable_cones = actual_cones[
        actual_cones[:, 0] > 0
    ]

    vehicle_poses = simulate_vehicle_movement()
    mapper = LandmarkMapper()

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

    estimated_positions = np.array([
        landmark["position"]
        for landmark in mapper.get_map().values()
    ])

    return evaluate_map(
        observable_cones,
        estimated_positions
    )


def main():
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    num_trials = 30

    mean_errors = []
    rmse_values = []
    precision_values = []
    recall_values = []

    for noise in noise_levels:

        results = [
            run_experiment(noise, seed)
            for seed in range(num_trials)
        ]

        mean_error = np.nanmean([
            r["mean_error"] for r in results
        ])

        rmse = np.nanmean([
            r["rmse"] for r in results
        ])

        precision = np.mean([
            r["precision"] for r in results
        ])

        recall = np.mean([
            r["recall"] for r in results
        ])

        mean_errors.append(mean_error)
        rmse_values.append(rmse)
        precision_values.append(precision)
        recall_values.append(recall)

        print(f"\nNoise standard deviation: {noise} m")
        print(f"Mean error: {mean_error:.4f} m")
        print(f"RMSE: {rmse:.4f} m")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")

    plt.figure(figsize=(10, 5))

    plt.plot(
        noise_levels, mean_errors,
        marker="o", label="Mean Error"
    )

    plt.plot(
        noise_levels, rmse_values,
        marker="s", label="RMSE"
    )

    plt.xlabel("Measurement Noise Std. Dev. (m)")
    plt.ylabel("Mapping Error (m)")
    plt.title("Mapping Accuracy vs. Sensor Noise")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))

    plt.plot(
        noise_levels, precision_values,
        marker="o", label="Precision"
    )

    plt.plot(
        noise_levels, recall_values,
        marker="s", label="Recall"
    )

    plt.xlabel("Measurement Noise Std. Dev. (m)")
    plt.ylabel("Score")
    plt.title("Mapping Reliability vs. Sensor Noise")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()