import numpy as np
import matplotlib.pyplot as plt

from src.landmark_mapping import LandmarkMapper
from src.evaluation import evaluate_map

from src.simulation import (
    generate_track,
    simulate_sensor_frame
)


def run_simulation(
    track_type="straight",
    seed=42,
    noise_std=0.3,
    association_threshold=5.991,
    min_observations=2,
    verbose=True,
    visualize=True
):
    rng = np.random.default_rng(seed)

    # Generate track and vehicle trajectory
    blue_cones, yellow_cones, vehicle_poses = (
        generate_track(track_type)
    )

    actual_cones = np.vstack((
        blue_cones,
        yellow_cones
    ))

    actual_classes = np.array(
        ["blue_cone"] * len(blue_cones)
        + ["yellow_cone"] * len(yellow_cones)
    )

    visibility_counts = np.zeros(
        len(actual_cones),
        dtype=int
    )

    mapper = LandmarkMapper(
        association_threshold=association_threshold,
        min_observations=min_observations
    )

    # Run the mapping simulation
    for step, pose in enumerate(vehicle_poses):

        detections, visible_indices = (
            simulate_sensor_frame(
                cones=actual_cones,
                cone_classes=actual_classes,
                vehicle_pose=pose,
                rng=rng,
                timestamp=float(step),
                max_range=20.0,
                fov_degrees=120,
                noise_std=noise_std
            )
        )

        visibility_counts[visible_indices] += 1

        mapper.update(detections, pose)

    # Extract confirmed landmarks
    landmark_map = mapper.get_map()

    estimated_positions = np.array([
        landmark["position"]
        for landmark in landmark_map.values()
    ]).reshape(-1, 2)

    estimated_classes = [
        landmark["class"]
        for landmark in landmark_map.values()
    ]

    # Fixed evaluation eligibility requirement
    eligible_mask = visibility_counts >= 2

    eligible_cones = actual_cones[eligible_mask]

    eligible_classes = actual_classes[eligible_mask]

    # Evaluate mapping performance
    metrics = evaluate_map(
        eligible_cones,
        estimated_positions,
        actual_classes=eligible_classes,
        estimated_classes=estimated_classes
    )

    if verbose:

        print(f"\n{track_type.upper()} Track Evaluation")
        print("--------------------------------")

        print(f"Seed: {seed}")
        print(f"Noise std: {noise_std} m")
        print(f"Vehicle frames: {len(vehicle_poses)}")
        print(f"Actual cones: {len(actual_cones)}")

        print(
            f"Observable cones: "
            f"{np.count_nonzero(visibility_counts)}"
        )

        print(f"Eligible cones: {len(eligible_cones)}")

        print(
            f"Confirmed landmarks: "
            f"{len(landmark_map)}"
        )

        print(
            f"Total landmarks: "
            f"{len(mapper.get_map(confirmed_only=False))}"
        )

        print("\nMapping Metrics")
        print("----------------")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    if visualize:

        plt.figure(figsize=(12, 8))

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
            s=60,
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

        plt.title(
            f"{track_type.capitalize()} Track Mapping"
        )

        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics


def main():

    track_types = [
        "straight",
        "curved",
        "slalom"
    ]

    for track_type in track_types:

        run_simulation(
            track_type=track_type,
            seed=42,
            noise_std=0.3,
            verbose=True,
            visualize=True
        )


if __name__ == "__main__":
    main()