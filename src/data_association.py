import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_detections(cost_matrix, threshold=5.991):
    """
    Match detections to landmarks using global assignment.

    Rows: detections
    Columns: existing landmarks

    Returns:
        matches: list of (detection_index, landmark_index)
        unmatched_detections: list of detection indices
        unmatched_landmarks: list of landmark indices
    """

    costs = np.asarray(cost_matrix, dtype=float)

    if costs.ndim != 2:
        raise ValueError("Cost matrix must be 2D")

    num_detections, num_landmarks = costs.shape

    if num_detections == 0 or num_landmarks == 0:
        return (
            [],
            list(range(num_detections)),
            list(range(num_landmarks))
        )

    # Exclude invalid or statistically unlikely matches
    valid = np.isfinite(costs) & (costs <= threshold)

    # Dummy columns allow detections to remain unmatched.
    # This prevents an invalid match from consuming a landmark.
    dummy_cost = threshold + 1.0

    assignment_costs = np.full(
        (num_detections, num_landmarks + num_detections),
        dummy_cost
    )

    assignment_costs[:, :num_landmarks] = np.where(
        valid, costs, dummy_cost + 1.0
    )

    row_indices, col_indices = linear_sum_assignment(
        assignment_costs
    )

    matches = []
    matched_detections = set()
    matched_landmarks = set()

    for row, col in zip(row_indices, col_indices):
        if col < num_landmarks and valid[row, col]:
            matches.append((int(row), int(col)))
            matched_detections.add(int(row))
            matched_landmarks.add(int(col))

    unmatched_detections = [
        i for i in range(num_detections)
        if i not in matched_detections
    ]

    unmatched_landmarks = [
        j for j in range(num_landmarks)
        if j not in matched_landmarks
    ]

    return matches, unmatched_detections, unmatched_landmarks