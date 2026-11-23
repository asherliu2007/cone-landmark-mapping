import numpy as np

from src.data_association import associate_detections


def main():
    # Rows = detections
    # Columns = landmarks
    costs = np.array([
        [1.0, 2.0],
        [1.1, 5.0]
    ])

    matches, unmatched_detections, unmatched_landmarks = (
        associate_detections(costs)
    )

    print("Matches:", matches)
    print("Unmatched detections:", unmatched_detections)
    print("Unmatched landmarks:", unmatched_landmarks)


if __name__ == "__main__":
    main()