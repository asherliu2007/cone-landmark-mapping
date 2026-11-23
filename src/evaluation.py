import numpy as np
from scipy.spatial.distance import cdist

from src.data_association import associate_detections


def evaluate_map(
    actual_positions,
    estimated_positions,
    matching_threshold=1.0
):
    actual_positions = np.asarray(
        actual_positions, dtype=float
    ).reshape(-1, 2)

    estimated_positions = np.asarray(
        estimated_positions, dtype=float
    ).reshape(-1, 2)

    # Calculate distances between estimated and actual cones
    distances = cdist(
        estimated_positions,
        actual_positions
    )

    # Match estimated landmarks to actual cones
    matches, unmatched_estimates, unmatched_actual = (
        associate_detections(
            distances,
            threshold=matching_threshold
        )
    )

    errors = np.array([
        distances[i, j]
        for i, j in matches
    ])

    num_matched = len(matches)

    return {
        "actual_cones": len(actual_positions),
        "mapped_landmarks": len(estimated_positions),
        "matched_landmarks": num_matched,
        "unmatched_landmarks": len(unmatched_estimates),
        "missed_cones": len(unmatched_actual),

        "mean_error": (
            np.mean(errors) if num_matched else np.nan
        ),

        "rmse": (
            np.sqrt(np.mean(errors ** 2))
            if num_matched else np.nan
        ),

        "max_error": (
            np.max(errors) if num_matched else np.nan
        ),

        "precision": (
            num_matched / len(estimated_positions)
            if len(estimated_positions) else 0.0
        ),

        "recall": (
            num_matched / len(actual_positions)
            if len(actual_positions) else 0.0
        )
    }