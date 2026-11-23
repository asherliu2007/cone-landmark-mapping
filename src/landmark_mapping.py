from turtle import distance

import numpy as np

from src.kalman_filter import KalmanFilter2D


class LandmarkMapper:
    def __init__(self, association_threshold=1.5):
        self.landmarks = {}
        self.next_id = 0
        self.association_threshold = association_threshold

    def transform_to_global(self, local_position, vehicle_pose):
        x, y, heading = vehicle_pose

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        return rotation @ np.asarray(local_position) + np.array([x, y])

    def find_nearest_landmark(self, position, covariance, cone_class=None, excluded_ids=None):
        if (excluded_ids is None):
            excluded_ids = set()
        
        nearest_id = None
        
        #Chi-square threshold for 2 dimensions at 95% confidence level
        threshold = 5.991
        
        min_distance = threshold
        
        for landmark_id, landmark in self.landmarks.items():
            if landmark_id in excluded_ids:
                continue
            
            if (
                cone_class is not None and
                landmark["class"] is not None and
                cone_class != landmark["class"]
            ):
                continue
            
            # Difference between measurement and landmark estimate
            difference = position - landmark["filter"].x
            
            # Combined uncertainty
            S = landmark["filter"].P + covariance
            
            # Squared Mahalanobis distance
            distance = difference.T @ np.linalg.inv(S) @ difference
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = landmark_id
                
        return nearest_id

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
        x, y, heading = vehicle_pose

        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        matched_ids = set()
        
        for detection in detections:
            global_position = self.transform_to_global(
                detection.position, vehicle_pose
            )

            # Rotate covariance into the global coordinate frame
            global_covariance = (
                rotation @ detection.covariance @ rotation.T
            )

            landmark_id = self.find_nearest_landmark(
                global_position,
                global_covariance,
                detection.cone_class,
                excluded_ids=matched_ids
            )

            if landmark_id is None:
                
                # Avoid creating a duplicate landmark
                if self.is_duplicate(
                    global_position,
                    detection.cone_class
                ):
                    continue
                
                self.landmarks[self.next_id] = {
                    "filter": KalmanFilter2D(
                        global_position,
                        global_covariance.copy()
                    ),
                    "class": detection.cone_class
                }

                self.next_id += 1

            else:
                landmark = self.landmarks[landmark_id]
                matched_ids.add(landmark_id)
                
                landmark["filter"].predict()
                landmark["filter"].update(
                    global_position,
                    global_covariance
                )

                if landmark["class"] is None:
                    landmark["class"] = detection.cone_class

    def get_map(self):
        return {
            landmark_id: {
                "position": landmark["filter"].x.copy(),
                "covariance": landmark["filter"].P.copy(),
                "class": landmark["class"]
            }
            for landmark_id, landmark in self.landmarks.items()
        }