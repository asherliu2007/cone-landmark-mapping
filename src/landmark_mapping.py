import numpy as np

from src.kalman_filter import KalmanFilter2D
from src.data_association import associate_detections

class LandmarkMapper:
    def __init__(self, association_threshold=1.5, min_observations=2, max_tentative_age=5):
        self.landmarks = {}
        self.next_id = 0
        self.association_threshold = association_threshold
        
        self.min_observations = min_observations
        self.max_tentative_age = max_tentative_age
        self.frame_count = 0

    def transform_to_global(self, local_position, vehicle_pose):
        x, y, heading = vehicle_pose

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        return rotation @ np.asarray(local_position) + np.array([x, y])

    def is_duplicate(self, position, cone_class=None, threshold=1.0):
        for landmark in self.landmarks.values():
            if (
                cone_class is not None and
                landmark["class"] is not None and
                cone_class != landmark["class"]
            ):
                continue
            
            distance = np.linalg.norm(position - landmark["filter"].x)
            
            if distance < threshold:
                return True
            
        return False

    def update(self, detections, vehicle_pose):
        self.frame_count += 1
        x, y, heading = vehicle_pose

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        # Convert all detections into global coordinates
        global_detections = []

        for detection in detections:
            global_position = self.transform_to_global(
                detection.position, vehicle_pose
            )

            global_covariance = (
                rotation @ detection.covariance @ rotation.T
            )

            global_detections.append({
                "position": global_position,
                "covariance": global_covariance,
                "class": detection.cone_class
            })

        # Existing landmark IDs
        landmark_ids = list(self.landmarks.keys())

        # Build Mahalanobis distance cost matrix
        cost_matrix = np.full(
            (len(global_detections), len(landmark_ids)),
            np.inf
        )

        for i, detection in enumerate(global_detections):
            for j, landmark_id in enumerate(landmark_ids):
                landmark = self.landmarks[landmark_id]

                if (
                    detection["class"] is not None
                    and landmark["class"] is not None
                    and detection["class"] != landmark["class"]
                ):
                    continue

                difference = (
                    detection["position"] - landmark["filter"].x
                )

                S = (
                    landmark["filter"].P
                    + detection["covariance"]
                )

                cost_matrix[i, j] = (
                    difference.T @ np.linalg.solve(S, difference)
                )

        # Find globally optimal matches
        matches, unmatched_detections, _ = (
            associate_detections(cost_matrix)
        )

        # Update matched landmarks
        for detection_idx, landmark_idx in matches:
            detection = global_detections[detection_idx]
            landmark_id = landmark_ids[landmark_idx]

            landmark = self.landmarks[landmark_id]

            landmark["observations"] += 1
            landmark["last_seen"] = self.frame_count
            
            if landmark["observations"] >= self.min_observations:
                landmark["confirmed"] = True
            
            landmark["filter"].predict()
            landmark["filter"].update(
                detection["position"],
                detection["covariance"]
            )

            if landmark["class"] is None:
                landmark["class"] = detection["class"]

        # Create new landmarks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = global_detections[detection_idx]

            if self.is_duplicate(
                detection["position"],
                detection["class"]
            ):
                continue

            self.landmarks[self.next_id] = {
                "filter": KalmanFilter2D(
                    detection["position"],
                    detection["covariance"].copy()
                ),
                "class": detection["class"],
                "observations": 1,
                "last_seen": self.frame_count,
                "confirmed": self.min_observations <= 1
            }

            self.next_id += 1
            
        self.remove_stale_landmarks()

    def remove_stale_landmarks(self):
        stale_ids = []
        
        for landmark_id, landmark in self.landmarks.items():
            
            age = self.frame_count - landmark["last_seen"]
            
            if (
                not landmark["confirmed"]
                and age >= self.max_tentative_age
            ):
                stale_ids.append(landmark_id)
        
        for landmark_id in stale_ids:
            del self.landmarks[landmark_id]

    def get_map(self, confirmed_only=True):
        return {
            landmark_id: {
                "position": landmark["filter"].x.copy(),
                "covariance": landmark["filter"].P.copy(),
                "class": landmark["class"],
                "observations": landmark["observations"],
                "confirmed": landmark["confirmed"]
            }
            for landmark_id, landmark in self.landmarks.items()
            if not confirmed_only or landmark["confirmed"]
        }