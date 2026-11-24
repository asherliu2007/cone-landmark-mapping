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

from src.detection import ConeDetection
from src.landmark_mapping import LandmarkMapper
from src.evaluation import evaluate_map

from scipy.spatial.distance import cdist
from src.data_association import associate_detections

def main(seed=42, verbose=True, visualize=True):

    rng = np.random.default_rng(seed)

    # Load FSOCO image
    image, annotation = load_fsoco_image()

    width, height = image.size

    focal_length = width / (
        2 * np.tan(np.deg2rad(90) / 2)
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

    reference_cones = project_detections(
        raw_detections,
        intrinsics
    )

    # Initialize mapper
    mapper = LandmarkMapper(
        association_threshold=9.210,
        min_observations=2,
    )

    vehicle_path = []

    visibility_counts = np.zeros(
        len(reference_cones),
        dtype=int
    )

    max_range = 20.0
    half_fov = np.deg2rad(45)

    # Simulate vehicle movement diagonally
    # toward the FSOCO-derived track
    heading = np.deg2rad(-35)

    for frame in range(60):

        distance = frame * 0.5

        vehicle_pose = np.array([
            distance * np.cos(heading),
            distance * np.sin(heading),
            heading
        ])

        vehicle_path.append(
            vehicle_pose[:2].copy()
        )

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        detections = []

        for cone_idx, cone in enumerate(reference_cones):

            # Transform global cone position
            # into vehicle coordinates
            local_position = rotation.T @ (
                cone.position - vehicle_pose[:2]
            )

            forward, lateral = local_position

            cone_range = np.linalg.norm(
                local_position
            )

            bearing = np.arctan2(
                lateral, forward
            )

            # Simulate camera visibility
            if forward <= 0:
                continue

            if cone_range > max_range:
                continue

            if abs(bearing) > half_fov:
                continue
            
            visibility_counts[cone_idx] += 1

            # Simulate distance-dependent
            # camera measurement uncertainty
            depth_std = max(
                0.1,
                2 * forward ** 2 /
                (focal_length * 0.325)
            )

            lateral_std = max(
                0.05,
                0.02 * cone_range
            )

            covariance = np.diag([
                depth_std ** 2,
                lateral_std ** 2
            ])

            # Generate noisy measurement
            noise = rng.multivariate_normal(
                np.zeros(2),
                covariance
            )

            measured_position = (
                local_position + noise
            )

            detections.append(
                ConeDetection(
                    position=measured_position,
                    covariance=covariance,
                    source="simulated_camera",
                    timestamp=frame * 0.1,
                    cone_class=cone.cone_class
                )
            )

        # Update landmark map
        mapper.update(
            detections,
            vehicle_pose
        )

        confirmed = mapper.get_map(
            confirmed_only=True
        )

        if verbose:  
            print(
                f"Frame {frame + 1:02d}: "
                f"Detections = {len(detections)}, "
                f"Confirmed = {len(confirmed)}"
            )

    landmarks = mapper.get_map(
        confirmed_only=True
    )

    all_landmarks = mapper.get_map(
        confirmed_only=False
    )
    
    observable_cones = [
        cone
        for i, cone in enumerate(reference_cones)
        if visibility_counts[i] > 0
    ]
    
    eligible_cones = [
        cone
        for i, cone in enumerate(reference_cones)
        if visibility_counts[i] >= mapper.min_observations
    ]
    
    # Extract eligible reference positions and classes
    actual_positions = [
        cone.position
        for cone in eligible_cones
    ]
    
    actual_classes = [
        cone.cone_class
        for cone in eligible_cones
    ]
    
    # Extracte confirmed landmark positions and classes
    estimated_positions = [
        landmark["position"]
        for landmark in landmarks.values()
    ]
    
    estimated_classes = [
        landmark["class"]
        for landmark in landmarks.values()
    ]
    
    metrics = evaluate_map(
        actual_positions,
        estimated_positions,
        matching_threshold=1.0,
        actual_classes=actual_classes,
        estimated_classes=estimated_classes
    )
    
    if verbose:
        print("\nFSOCO Mapping Evaluation")
        print("------------------------")
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nVisibility Evaluation")
        print("---------------------")
        print(f"Reference cones: {len(reference_cones)}")
        print(f"Observable cones: {len(observable_cones)}")
        print(f"Eligible cones: {len(eligible_cones)}")
        print(f"Confirmed landmarks: {len(landmarks)}")

        print("\nCone visibility details:")

        for i, cone in enumerate(reference_cones):
            print(
                f"Cone {i}: "
                f"{cone.cone_class}, "
                f"Visible frames: {visibility_counts[i]}"
            )

        print("\nMulti-frame Mapping Results")
        print("----------------------------")
        print(f"Reference cones: {len(reference_cones)}")
        print(f"Confirmed landmarks: {len(landmarks)}")
        print(f"Total landmarks: {len(all_landmarks)}")

        print("\nLandmark details:")

        for landmark_id, landmark in landmarks.items():

            print(
                f"ID: {landmark_id}, "
                f"Position: "
                f"{np.round(landmark['position'], 3)}, "
                f"Observations: "
                f"{landmark['observations']}"
            )

        # Diagnose unmatched landmarks
        landmark_ids = list(landmarks.keys())

        estimated_positions = np.array([
            landmarks[landmark_id]["position"]
            for landmark_id in landmark_ids
        ])

        actual_positions = np.array([
            cone.position
            for cone in eligible_cones
        ])

        distances = cdist(
            estimated_positions,
            actual_positions
        )
        
        matching_costs = distances.copy()
        
        for i, landmark_id in enumerate(landmark_ids):
            for j, cone in enumerate(eligible_cones):
                if landmarks[landmark_id]["class"] != cone.cone_class:
                    matching_costs[i, j] = np.inf

        matches, unmatched_estimates, unmatched_actual = (
            associate_detections(
                matching_costs,
                threshold=1.0
            )
        )

        print("\nUnmatched Landmark Diagnostics")
        print("------------------------------")

        for idx in unmatched_estimates:

            landmark_id = landmark_ids[idx]
            landmark = landmarks[landmark_id]

            same_class_indices = [
                j for j, cone in enumerate(eligible_cones)
                if cone.cone_class == landmark["class"]
            ]

            if not same_class_indices:
                print(f"Landmark ID: {landmark_id}")
                print("No reference cone of the same class")
                continue

            nearest_idx = min(
                same_class_indices,
                key=lambda j: distances[idx, j]
            )

            nearest_cone = eligible_cones[nearest_idx]

            print(f"\nLandmark ID: {landmark_id}")
            print(f"Class: {landmark['class']}")
            print(f"Position: {landmark['position']}")
            print(f"Observations: {landmark['observations']}")

            print(
                f"Nearest reference cone: "
                f"{nearest_cone.position}"
            )

            print(
                f"Nearest cone class: "
                f"{nearest_cone.cone_class}"
            )

            print(
                f"Distance to nearest cone: "
                f"{distances[idx, nearest_idx]:.4f} m"
            )

            print("Landmark covariance:")
            print(landmark["covariance"])

    # Visualization
    if visualize:
        colors = {
            "blue_cone": "blue",
            "yellow_cone": "gold",
            "orange_cone": "orange",
            "large_orange_cone": "red"
        }

        plt.figure(figsize=(10, 8))

        for cone in reference_cones:

            plt.scatter(
                -cone.position[1],
                cone.position[0],
                color=colors.get(
                    cone.cone_class, "gray"
                )
            )

        for landmark in landmarks.values():

            position = landmark["position"]

            plt.scatter(
                -position[1],
                position[0],
                color="black",
                marker="x",
                s=70
            )

        vehicle_path = np.array(vehicle_path)

        plt.plot(
            -vehicle_path[:, 1],
            vehicle_path[:, 0],
            color="green",
            label="Vehicle path"
        )

        plt.scatter(
            [], [],
            color="black",
            marker="x",
            label="Confirmed landmarks"
        )

        plt.xlabel("Lateral Position (m)")
        plt.ylabel("Forward Position (m)")
        plt.title("Multi-frame FSOCO Landmark Mapping")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        plt.show()
    
    return metrics

if __name__ == "__main__":
    main()