import numpy as np

from src.kalman_filter import KalmanFilter2D
from src.data_association import associate_detections

class LandmarkMapper:
    def __init__(self, association_threshold=5.991, min_observations=2, max_tentative_age=5):
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

    def is_duplicate(
        self,
        position,
        covariance,
        cone_class=None,
        threshold=13.816,
        max_distance=2.0,
        existing_ids=None,
        matched_ids=None
    ):
        if matched_ids is None:
            matched_ids = set()

        for landmark_id, landmark in self.landmarks.items():

            # Only consider previously existing landmarks
            if (
                existing_ids is not None
                and landmark_id not in existing_ids
            ):
                continue

            # Do not suppress detections based on landmarks
            # already assigned another detection this frame
            if landmark_id in matched_ids:
                continue

            # Require compatible cone classes
            if (
                cone_class is not None
                and landmark["class"] is not None
                and cone_class != landmark["class"]
            ):
                continue

            landmark_filter = landmark["filter"]

            difference = position - landmark_filter.x

            # Euclidean safety gate
            distance = np.linalg.norm(difference)

            if distance > max_distance:
                continue

            # Combined measurement uncertainty
            S = landmark_filter.P + covariance

            mahalanobis = (
                difference.T
                @ np.linalg.solve(S, difference)
            )

            if mahalanobis <= threshold:
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
        existing_ids = set(landmark_ids)

        # Predict all existing landmarks before association
        for landmark in self.landmarks.values():
            landmark["filter"].predict()

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
            associate_detections(
                cost_matrix,
                threshold=self.association_threshold
            )
        )

        # Track landmarks already assigned a detection
        matched_ids = {
            landmark_ids[landmark_idx]
            for _, landmark_idx in matches
        }

        # Update matched landmarks
        for detection_idx, landmark_idx in matches:
            detection = global_detections[detection_idx]
            landmark_id = landmark_ids[landmark_idx]

            landmark = self.landmarks[landmark_id]

            landmark["observations"] += 1
            landmark["last_seen"] = self.frame_count
            
            if landmark["observations"] >= self.min_observations:
                landmark["confirmed"] = True
            
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
                position=detection["position"],
                covariance=detection["covariance"],
                cone_class=detection["class"],
                existing_ids=existing_ids,
                matched_ids=matched_ids
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
        self.merge_duplicate_landmarks()

    def merge_duplicate_landmarks(
        self,
        distance_threshold=0.35,
        mahalanobis_threshold=5.991
    ):
        merged = True

        while merged:

            merged = False
            landmark_ids = list(self.landmarks.keys())

            for i in range(len(landmark_ids)):

                if merged:
                    break

                id_a = landmark_ids[i]
                landmark_a = self.landmarks[id_a]

                for j in range(i + 1, len(landmark_ids)):

                    id_b = landmark_ids[j]
                    landmark_b = self.landmarks[id_b]

                    # Only merge confirmed landmarks
                    if not (
                        landmark_a["confirmed"]
                        and landmark_b["confirmed"]
                    ):
                        continue

                    # Require matching cone classes
                    if landmark_a["class"] != landmark_b["class"]:
                        continue

                    filter_a = landmark_a["filter"]
                    filter_b = landmark_b["filter"]

                    difference = filter_a.x - filter_b.x

                    # Euclidean distance gate
                    distance = np.linalg.norm(difference)

                    if distance > distance_threshold:
                        continue

                    # Mahalanobis compatibility gate
                    S = filter_a.P + filter_b.P

                    mahalanobis = (
                        difference.T
                        @ np.linalg.solve(S, difference)
                    )

                    if mahalanobis > mahalanobis_threshold:
                        continue

                    # Covariance intersection
                    weight = 0.5

                    P_a_inv = np.linalg.inv(filter_a.P)
                    P_b_inv = np.linalg.inv(filter_b.P)

                    merged_covariance = np.linalg.inv(
                        weight * P_a_inv
                        + (1 - weight) * P_b_inv
                    )

                    merged_position = merged_covariance @ (
                        weight * P_a_inv @ filter_a.x
                        + (1 - weight) * P_b_inv @ filter_b.x
                    )

                    # Update the surviving landmark
                    filter_a.x = merged_position
                    filter_a.P = merged_covariance

                    landmark_a["observations"] = max(
                        landmark_a["observations"],
                        landmark_b["observations"]
                    )

                    landmark_a["last_seen"] = max(
                        landmark_a["last_seen"],
                        landmark_b["last_seen"]
                    )

                    # Remove the duplicate landmark
                    del self.landmarks[id_b]

                    merged = True
                    break

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